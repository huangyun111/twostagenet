"""Run Stage 2 residual diffusion inference and evaluation."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.utils.data import DataLoader, Subset
from transformers import CLIPTextModel, CLIPTokenizer, PretrainedConfig

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.stage2_residual_dataset import Stage2ResidualDataset  # noqa: E402
from models.PolarControlnet import PolarControl  # noqa: E402
from models.utils import load_params  # noqa: E402
from train_stage2_residual import (  # noqa: E402
    Stage2ResidualDiffusionModel,
    polar_to_vae_input,
    resolve_device,
    resolve_pretrained_model_path,
    strip_module_prefix,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRIC_NAMES = (
    "dolp_mae",
    "dolp_rmse",
    "cos_mae",
    "sin_mae",
    "cos_sin_vector_error",
    "weighted_aolp_error_deg",
    "high_dolp_aolp_error_deg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer and evaluate Stage 2 residual diffusion."
    )
    parser.add_argument("--root_dir", type=str, default=str(Path.home() / "Documents"))
    parser.add_argument("--stage1_dir", type=str, default="./stage1_exports")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument("--output_dir", type=str, default="./stage2_test_outputs")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default="denoised polarized images")
    parser.add_argument("--vae_scaling_factor", type=float, default=0.18215)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--vis_every", type=int, default=50)
    parser.add_argument("--vae_encode_mode", choices=("mode", "sample"), default="mode")
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument(
        "--inference_mode",
        choices=("stage2", "prior_vae", "gt_vae"),
        default="stage2",
    )
    return parser.parse_args()


def build_dataloader(args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, int]:
    dataset = Stage2ResidualDataset(
        root_dir=args.root_dir,
        stage1_dir=args.stage1_dir,
        image_size=args.image_size,
    )
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("max_samples must be positive or None.")
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    return dataloader, len(dataset)


def build_models(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[AutoencoderKL, DDPMScheduler, Stage2ResidualDiffusionModel]:
    checkpoint = resolve_pretrained_model_path(args.pretrained_model_name_or_path)
    tokenizer = CLIPTokenizer.from_pretrained(checkpoint, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(checkpoint, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(checkpoint, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(checkpoint, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(checkpoint, subfolder="scheduler")

    controlnet = PolarControl(PretrainedConfig())
    load_params(controlnet, unet)

    model = Stage2ResidualDiffusionModel(
        unet=unet,
        controlnet=controlnet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=args.prompt,
    )
    load_stage2_checkpoint(model, args.checkpoint)

    vae.requires_grad_(False)
    model.requires_grad_(False)
    vae.eval()
    model.eval()

    vae.to(device)
    model.to(device)
    return vae, scheduler, model


def load_stage2_checkpoint(
    model: Stage2ResidualDiffusionModel,
    checkpoint_path: str,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    required_keys = (
        "unet_state_dict",
        "controlnet_state_dict",
        "condition_adapter_state_dict",
    )
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise KeyError(
            f"Checkpoint is missing required Stage 2 keys: {', '.join(missing_keys)}"
        )

    model.unet.load_state_dict(strip_module_prefix(checkpoint["unet_state_dict"]))
    model.controlnet.load_state_dict(strip_module_prefix(checkpoint["controlnet_state_dict"]))
    model.condition_adapter.load_state_dict(
        strip_module_prefix(checkpoint["condition_adapter_state_dict"])
    )
    print(f"Loaded Stage 2 checkpoint: {checkpoint_path}", flush=True)


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    if device.type == "cuda":
        generator = torch.Generator(device="cuda")
    else:
        generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def infer_batch(
    batch: dict[str, torch.Tensor | list[str]],
    vae: AutoencoderKL,
    scheduler: DDPMScheduler,
    model: Stage2ResidualDiffusionModel,
    device: torch.device,
    generator: torch.Generator,
    num_inference_steps: int,
    vae_scaling_factor: float,
    vae_encode_mode: str,
    residual_scale: float,
    inference_mode: str,
) -> torch.Tensor:
    rgb = batch["rgb"].to(device, non_blocking=True)
    prior = batch["prior"].to(device, non_blocking=True)
    confidence = batch["confidence"].to(device, non_blocking=True)

    prior_vae = polar_to_vae_input(prior)
    z_c = encode_latents(vae, prior_vae, generator, vae_encode_mode, vae_scaling_factor)

    if inference_mode == "prior_vae":
        decoded = vae.decode(z_c / vae_scaling_factor).sample
        return postprocess_decoded_polar(decoded)

    if inference_mode == "gt_vae":
        polar_gt = batch["polar_gt"].to(device, non_blocking=True)
        gt_vae = polar_to_vae_input(polar_gt)
        z_gt = encode_latents(vae, gt_vae, generator, vae_encode_mode, vae_scaling_factor)
        decoded = vae.decode(z_gt / vae_scaling_factor).sample
        return postprocess_decoded_polar(decoded)

    delta_z = torch.randn(
        z_c.shape,
        generator=generator,
        device=device,
        dtype=z_c.dtype,
    )

    scheduler.set_timesteps(num_inference_steps, device=device)
    condition = torch.cat([rgb, prior, confidence], dim=1)
    for timestep in scheduler.timesteps:
        model_timestep = timestep.expand(delta_z.shape[0])
        noise_pred = model(delta_z, model_timestep, condition)
        delta_z = scheduler.step(noise_pred, timestep, delta_z).prev_sample

    z_final = z_c + residual_scale * delta_z
    decoded = vae.decode(z_final / vae_scaling_factor).sample
    return postprocess_decoded_polar(decoded)


def encode_latents(
    vae: AutoencoderKL,
    vae_input: torch.Tensor,
    generator: torch.Generator,
    vae_encode_mode: str,
    vae_scaling_factor: float,
) -> torch.Tensor:
    latent_dist = vae.encode(vae_input).latent_dist
    if vae_encode_mode == "mode":
        latents = latent_dist.mode()
    elif vae_encode_mode == "sample":
        latents = latent_dist.sample(generator=generator)
    else:
        raise ValueError(f"Unknown vae_encode_mode: {vae_encode_mode}")
    return latents * vae_scaling_factor


def postprocess_decoded_polar(decoded: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dolp = ((decoded[:, 0:1] + 1.0) * 0.5).clamp(0.0, 1.0)
    cos_sin = decoded[:, 1:3]
    norm = torch.sqrt((cos_sin * cos_sin).sum(dim=1, keepdim=True))
    normalized = cos_sin / norm.clamp_min(eps)
    default_direction = torch.zeros_like(cos_sin)
    default_direction[:, 0:1] = 1.0
    cos_sin = torch.where(norm > eps, normalized, default_direction)
    return torch.cat((dolp, cos_sin), dim=1).contiguous()


def compute_metric_dict(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    pred = pred.detach().float().cpu()
    gt = gt.detach().float().cpu()

    dolp_error = (pred[0] - gt[0]).abs()
    cos_error = (pred[1] - gt[1]).abs()
    sin_error = (pred[2] - gt[2]).abs()
    vector_error = torch.sqrt((pred[1] - gt[1]) ** 2 + (pred[2] - gt[2]) ** 2)

    aolp_error_deg = compute_aolp_error_deg(pred, gt)
    weights = torch.clamp((gt[0] - 0.03) / (0.15 - 0.03), 0.0, 1.0)
    high_mask = gt[0] > 0.15

    return {
        "dolp_mae": float(dolp_error.mean()),
        "dolp_rmse": float(torch.sqrt(torch.mean((pred[0] - gt[0]) ** 2))),
        "cos_mae": float(cos_error.mean()),
        "sin_mae": float(sin_error.mean()),
        "cos_sin_vector_error": float(vector_error.mean()),
        "weighted_aolp_error_deg": weighted_mean(aolp_error_deg, weights),
        "high_dolp_aolp_error_deg": masked_mean(aolp_error_deg, high_mask),
    }


def compute_aolp_error_deg(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    cross = pred[2] * gt[1] - pred[1] * gt[2]
    dot = pred[1] * gt[1] + pred[2] * gt[2]
    two_theta_error = torch.atan2(cross, dot).abs()
    return two_theta_error * 0.5 * (180.0 / math.pi)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float:
    weight_sum = weights.sum()
    if float(weight_sum) <= 0.0:
        return float("nan")
    return float((values * weights).sum() / weight_sum)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    return float(values[mask].mean())


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def tensor_to_rgb_image(rgb: torch.Tensor) -> np.ndarray:
    image = rgb.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip((image + 1.0) * 0.5, 0.0, 1.0)


def to_display_map(tensor: torch.Tensor, value_range: tuple[float, float]) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    min_value, max_value = value_range
    array = (array - min_value) / (max_value - min_value)
    return np.clip(array, 0.0, 1.0)


def save_visualization(
    rgb: torch.Tensor,
    polar_gt: torch.Tensor,
    prior: torch.Tensor,
    refined: torch.Tensor,
    confidence: torch.Tensor,
    path: Path,
) -> None:
    rgb_image = tensor_to_rgb_image(rgb)
    stage1_dolp_error = (prior[0] - polar_gt[0]).abs()
    stage2_dolp_error = (refined[0] - polar_gt[0]).abs()
    stage1_vector_error = torch.sqrt((prior[1] - polar_gt[1]) ** 2 + (prior[2] - polar_gt[2]) ** 2)
    stage2_vector_error = torch.sqrt(
        (refined[1] - polar_gt[1]) ** 2 + (refined[2] - polar_gt[2]) ** 2
    )
    confidence_aolp = confidence[1:3].mean(dim=0)

    panels = [
        ("RGB", rgb_image, None),
        ("GT DoLP", polar_gt[0].detach().cpu().numpy(), (0.0, 1.0)),
        ("Stage1 prior DoLP", prior[0].detach().cpu().numpy(), (0.0, 1.0)),
        ("Stage2 refined DoLP", refined[0].detach().cpu().numpy(), (0.0, 1.0)),
        ("Stage1 DoLP error", stage1_dolp_error.detach().cpu().numpy(), (0.0, 1.0)),
        ("Stage2 DoLP error", stage2_dolp_error.detach().cpu().numpy(), (0.0, 1.0)),
        ("Confidence DoLP", confidence[0].detach().cpu().numpy(), (0.0, 1.0)),
        ("Confidence AoLP", confidence_aolp.detach().cpu().numpy(), (0.0, 1.0)),
        ("GT cos2", to_display_map(polar_gt[1], (-1.0, 1.0)), (0.0, 1.0)),
        ("Prior cos2", to_display_map(prior[1], (-1.0, 1.0)), (0.0, 1.0)),
        ("Refined cos2", to_display_map(refined[1], (-1.0, 1.0)), (0.0, 1.0)),
        ("GT sin2", to_display_map(polar_gt[2], (-1.0, 1.0)), (0.0, 1.0)),
        ("Prior sin2", to_display_map(prior[2], (-1.0, 1.0)), (0.0, 1.0)),
        ("Refined sin2", to_display_map(refined[2], (-1.0, 1.0)), (0.0, 1.0)),
        ("Stage1 vector error", stage1_vector_error.detach().cpu().numpy(), (0.0, 2.0)),
        ("Stage2 vector error", stage2_vector_error.detach().cpu().numpy(), (0.0, 2.0)),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for axis, (title, image, value_range) in zip(axes.flat, panels):
        if image.ndim == 3:
            axis.imshow(image)
        else:
            vmin, vmax = value_range if value_range is not None else (None, None)
            axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_metrics_csv(metrics_path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = ["name"]
    for prefix in ("stage1", "stage2"):
        fieldnames.extend(f"{prefix}_{name}" for name in METRIC_NAMES)

    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(rows: list[dict[str, float | str]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for prefix in ("stage1", "stage2"):
        for metric_name in METRIC_NAMES:
            key = f"{prefix}_{metric_name}"
            values = np.array([float(row[key]) for row in rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            summary[key] = float(finite.mean()) if finite.size else float("nan")
    return summary


def write_summary(
    summary_path: Path,
    args: argparse.Namespace,
    sample_count: int,
    summary: dict[str, float],
) -> None:
    lines = [
        "Stage 2 residual diffusion test summary",
        f"samples: {sample_count}",
        f"root_dir: {args.root_dir}",
        f"stage1_dir: {args.stage1_dir}",
        f"checkpoint: {args.checkpoint}",
        f"pretrained_model_name_or_path: {args.pretrained_model_name_or_path}",
        f"image_size: {args.image_size}",
        f"num_inference_steps: {args.num_inference_steps}",
        f"vae_encode_mode: {args.vae_encode_mode}",
        f"residual_scale: {args.residual_scale}",
        f"inference_mode: {args.inference_mode}",
        f"seed: {args.seed}",
        f"prompt: {args.prompt}",
        "",
        "Mean metrics:",
    ]
    for key in sorted(summary):
        lines.append(f"{key}: {format_float(summary[key])}")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def run(args: argparse.Namespace) -> None:
    if args.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive.")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    generator = make_generator(device, args.seed)

    output_dir = Path(args.output_dir)
    refined_dir = output_dir / "refined_npy"
    vis_dir = output_dir / "vis"
    refined_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    dataloader, dataset_size = build_dataloader(args, device)
    vae, scheduler, model = build_models(args, device)

    rows: list[dict[str, float | str]] = []
    processed = 0
    with torch.no_grad():
        for batch in dataloader:
            refined = infer_batch(
                batch=batch,
                vae=vae,
                scheduler=scheduler,
                model=model,
                device=device,
                generator=generator,
                num_inference_steps=args.num_inference_steps,
                vae_scaling_factor=args.vae_scaling_factor,
                vae_encode_mode=args.vae_encode_mode,
                residual_scale=args.residual_scale,
                inference_mode=args.inference_mode,
            ).cpu()

            names = batch["name"]
            polar_gt = batch["polar_gt"].cpu()
            prior = batch["prior"].cpu()
            confidence = batch["confidence"].cpu()
            rgb = batch["rgb"].cpu()

            for offset, name in enumerate(names):
                sample_index = processed + offset
                sample_refined = refined[offset]
                np.save(
                    refined_dir / f"{name}.npy",
                    sample_refined.numpy().astype(np.float32),
                )

                row: dict[str, float | str] = {"name": name}
                row.update(prefix_metrics("stage1", compute_metric_dict(prior[offset], polar_gt[offset])))
                row.update(
                    prefix_metrics(
                        "stage2",
                        compute_metric_dict(sample_refined, polar_gt[offset]),
                    )
                )
                rows.append(row)

                if args.vis_every > 0 and sample_index % args.vis_every == 0:
                    save_visualization(
                        rgb=rgb[offset],
                        polar_gt=polar_gt[offset],
                        prior=prior[offset],
                        refined=sample_refined,
                        confidence=confidence[offset],
                        path=vis_dir / f"{name}.png",
                    )

            processed += len(names)
            print(f"processed {processed}/{dataset_size}", flush=True)

    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.txt"
    write_metrics_csv(metrics_path, rows)
    summary = summarize_rows(rows)
    write_summary(summary_path, args, len(rows), summary)
    print(f"Done. Wrote outputs to {output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
