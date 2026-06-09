"""Run Stage 2 angular residual refiner inference and evaluation."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.stage2_residual_dataset import Stage2ResidualDataset  # noqa: E402
from models.stage2_angular_refiner_net import ConfidenceGuidedAngularResidualRefiner  # noqa: E402

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
    "dop_mae",
    "dop_rmse",
    "aop_mae_deg",
    "weighted_aop_mae_deg",
    "high_dop_aop_mae_deg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer and evaluate Stage 2 angular refiner.")
    parser.add_argument("--root_dir", type=str, default=str(Path.home() / "Documents"))
    parser.add_argument("--stage1_dir", type=str, default="./stage1_exports")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./stage2_angular_refiner_outputs")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument(
        "--preprocess_mode",
        choices=("resize256", "official_infer"),
        default="resize256",
    )
    parser.add_argument("--divisible_by", type=int, default=32)
    parser.add_argument(
        "--normalize_mode",
        choices=("fixed255", "image_max"),
        default="fixed255",
    )
    parser.add_argument("--resize_output_to_gt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--vis_every", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--residual_scale", type=float, default=0.3)
    parser.add_argument("--angle_residual_scale", type=float, default=math.pi)
    parser.add_argument("--min_gate", type=float, default=0.2)
    parser.add_argument("--base_channels", type=int, default=64)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_dataloader(args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, int]:
    dataset = Stage2ResidualDataset(
        root_dir=args.root_dir,
        stage1_dir=args.stage1_dir,
        image_size=args.image_size,
        preprocess_mode=args.preprocess_mode,
        normalize_mode=args.normalize_mode,
        divisible_by=args.divisible_by,
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


def build_model(args: argparse.Namespace, device: torch.device) -> ConfidenceGuidedAngularResidualRefiner:
    model = ConfidenceGuidedAngularResidualRefiner(
        base_channels=args.base_channels,
        residual_scale=args.residual_scale,
        angle_residual_scale=args.angle_residual_scale,
        min_gate=args.min_gate,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(strip_module_prefix(state_dict))
    model.to(device)
    model.eval()
    return model


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


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
    dolp_mae = float(dolp_error.mean())
    dolp_rmse = float(torch.sqrt(torch.mean((pred[0] - gt[0]) ** 2)))
    aop_mae_deg = float(aolp_error_deg.mean())
    weighted_aop_mae_deg = weighted_mean(aolp_error_deg, weights)
    high_dop_aop_mae_deg = masked_mean(aolp_error_deg, high_mask)

    return {
        "dolp_mae": dolp_mae,
        "dolp_rmse": dolp_rmse,
        "cos_mae": float(cos_error.mean()),
        "sin_mae": float(sin_error.mean()),
        "cos_sin_vector_error": float(vector_error.mean()),
        "weighted_aolp_error_deg": weighted_aop_mae_deg,
        "high_dolp_aolp_error_deg": high_dop_aop_mae_deg,
        "dop_mae": dolp_mae,
        "dop_rmse": dolp_rmse,
        "aop_mae_deg": aop_mae_deg,
        "weighted_aop_mae_deg": weighted_aop_mae_deg,
        "high_dop_aop_mae_deg": high_dop_aop_mae_deg,
    }


def resize_polar_to_gt_if_requested(
    tensor: torch.Tensor,
    gt: torch.Tensor,
    resize_output_to_gt: bool,
) -> tuple[torch.Tensor, bool]:
    if tensor.shape[-2:] == gt.shape[-2:]:
        return tensor, False
    if not resize_output_to_gt:
        raise ValueError(
            "Prediction/prior and GT sizes differ. Use --resize_output_to_gt "
            f"for metrics. pred={tuple(tensor.shape[-2:])}, gt={tuple(gt.shape[-2:])}"
        )
    resized = torch.nn.functional.interpolate(
        tensor.unsqueeze(0),
        size=gt.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    dolp = resized[0:1].clamp(0.0, 1.0)
    cos_sin = resized[1:3].clamp(-1.0, 1.0)
    norm = torch.sqrt((cos_sin * cos_sin).sum(dim=0, keepdim=True) + 1e-6)
    return torch.cat((dolp, cos_sin / norm), dim=0).contiguous(), True


def compute_aolp_error_deg(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    cross = pred[2] * gt[1] - pred[1] * gt[2]
    dot = pred[1] * gt[1] + pred[2] * gt[2]
    two_theta_error = torch.atan2(cross.abs(), dot)
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
    return np.clip((array - min_value) / (max_value - min_value), 0.0, 1.0)


def polar_to_aolp(polar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.atan2(polar[2], polar[1])


def polar_to_aolp_deg(polar: torch.Tensor | np.ndarray) -> np.ndarray:
    array = polar.detach().float().cpu().numpy() if isinstance(polar, torch.Tensor) else np.asarray(polar)
    if array.ndim != 3:
        raise ValueError(f"Expected polar shape [3,H,W] or [H,W,3], got {array.shape}.")
    if array.shape[0] == 3:
        polar_chw = array
    elif array.shape[-1] == 3:
        polar_chw = np.moveaxis(array, -1, 0)
    else:
        raise ValueError(f"Expected polar shape [3,H,W] or [H,W,3], got {array.shape}.")
    aolp_rad = 0.5 * np.arctan2(polar_chw[2], polar_chw[1])
    return np.clip(aolp_rad * (180.0 / math.pi), -90.0, 90.0)


def aolp_to_display(aolp: torch.Tensor) -> np.ndarray:
    return np.clip(((aolp.detach().cpu().numpy() + math.pi / 2.0) / math.pi), 0.0, 1.0)


def polar_encoding_display(polar: torch.Tensor) -> np.ndarray:
    polar_cpu = polar.detach().cpu()
    r = polar_cpu[0].clamp(0.0, 1.0).numpy()
    g = (polar_cpu[1].clamp(-1.0, 1.0).numpy() * 0.5 + 0.5)
    b = (polar_cpu[2].clamp(-1.0, 1.0).numpy() * 0.5 + 0.5)
    return np.stack([r, g, b], axis=-1)


def save_polar_encoding_png(polar: torch.Tensor, path: Path) -> None:
    """Save PA-style uint16 RGB semantics [DoLP, cos2, sin2] as PNG."""
    polar_cpu = polar.detach().float().cpu()
    dolp = polar_cpu[0].clamp(0.0, 1.0).numpy()
    cos2 = polar_cpu[1].clamp(-1.0, 1.0).numpy() * 0.5 + 0.5
    sin2 = polar_cpu[2].clamp(-1.0, 1.0).numpy() * 0.5 + 0.5
    rgb = np.stack([dolp, cos2, sin2], axis=-1)
    rgb_u16 = np.rint(rgb * 65535.0).clip(0, 65535).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    # cv2.imwrite expects BGR input; this writes file RGB semantics as [DoLP, cos2, sin2].
    bgr_u16 = rgb_u16[..., ::-1]
    if not cv2.imwrite(str(path), bgr_u16):
        raise IOError(f"Failed to write encoding PNG: {path}")


def save_paper_aop_dop_vis(
    rgb_or_none: torch.Tensor | np.ndarray | None,
    gt_polar: torch.Tensor | np.ndarray,
    pred_polar: torch.Tensor | np.ndarray,
    save_path: Path,
) -> None:
    del rgb_or_none
    gt_array = gt_polar.detach().float().cpu().numpy() if isinstance(gt_polar, torch.Tensor) else np.asarray(gt_polar)
    pred_array = pred_polar.detach().float().cpu().numpy() if isinstance(pred_polar, torch.Tensor) else np.asarray(pred_polar)
    if gt_array.ndim == 3 and gt_array.shape[-1] == 3:
        gt_array = np.moveaxis(gt_array, -1, 0)
    if pred_array.ndim == 3 and pred_array.shape[-1] == 3:
        pred_array = np.moveaxis(pred_array, -1, 0)

    panels = [
        ("Captured AoLP", polar_to_aolp_deg(gt_array), "hsv", -90.0, 90.0, [-90.0, 0.0, 90.0]),
        ("Captured DoLP", np.clip(gt_array[0], 0.0, 1.0), "GnBu", 0.0, 1.0, [0.0, 1.0]),
        ("Gen. AoLP", polar_to_aolp_deg(pred_array), "hsv", -90.0, 90.0, [-90.0, 0.0, 90.0]),
        ("Gen. DoLP", np.clip(pred_array[0], 0.0, 1.0), "GnBu", 0.0, 1.0, [0.0, 1.0]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.patches.extend(
        [
            plt.Rectangle((0.0, 0.5), 1.0, 0.5, transform=fig.transFigure, color="#fff0f4", zorder=-1),
            plt.Rectangle((0.0, 0.0), 1.0, 0.5, transform=fig.transFigure, color="#eefcff", zorder=-1),
        ]
    )
    for axis, (title, image, cmap, vmin, vmax, ticks) in zip(axes.flat, panels):
        axis.set_facecolor("#fff0f4" if "Captured" in title else "#eefcff")
        im = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title, fontsize=13)
        axis.axis("off")
        colorbar = fig.colorbar(im, ax=axis, fraction=0.046, pad=0.03)
        colorbar.set_ticks(ticks)
        if "AoLP" in title:
            colorbar.set_ticklabels([r"$-90^\circ$", r"$0^\circ$", r"$90^\circ$"])
        else:
            colorbar.set_ticklabels(["0", "1"])
        colorbar.ax.tick_params(labelsize=10)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_visualization(
    rgb: torch.Tensor,
    polar_gt: torch.Tensor,
    prior: torch.Tensor,
    confidence: torch.Tensor,
    pred: dict[str, torch.Tensor],
    path: Path,
) -> None:
    refined = pred["refined"]
    stage1_dolp_error = (prior[0] - polar_gt[0]).abs()
    stage2_dolp_error = (refined[0] - polar_gt[0]).abs()
    stage1_aolp_error = compute_aolp_error_deg(prior, polar_gt)
    stage2_aolp_error = compute_aolp_error_deg(refined, polar_gt)
    confidence_aolp = confidence[1:3].mean(dim=0)
    delta_angle_deg = pred["delta_angle"][0] * (180.0 / math.pi)

    panels = [
        ("RGB", tensor_to_rgb_image(rgb), None),
        ("GT DoLP", polar_gt[0].detach().cpu().numpy(), (0.0, 1.0)),
        ("Stage1 prior DoLP", prior[0].detach().cpu().numpy(), (0.0, 1.0)),
        ("Stage2 refined DoLP", refined[0].detach().cpu().numpy(), (0.0, 1.0)),
        ("Stage1 DoLP error", stage1_dolp_error.detach().cpu().numpy(), (0.0, 1.0)),
        ("Stage2 DoLP error", stage2_dolp_error.detach().cpu().numpy(), (0.0, 1.0)),
        ("GT AoLP", aolp_to_display(polar_to_aolp(polar_gt)), (0.0, 1.0)),
        ("Stage1 AoLP", aolp_to_display(polar_to_aolp(prior)), (0.0, 1.0)),
        ("Stage2 AoLP", aolp_to_display(polar_to_aolp(refined)), (0.0, 1.0)),
        ("Stage1 AoLP error", stage1_aolp_error.detach().cpu().numpy(), (0.0, 90.0)),
        ("Stage2 AoLP error", stage2_aolp_error.detach().cpu().numpy(), (0.0, 90.0)),
        ("Delta Angle (deg)", delta_angle_deg.detach().cpu().numpy(), (-180.0, 180.0)),
        ("GT Encoding", polar_encoding_display(polar_gt), None),
        ("Stage1 Encoding", polar_encoding_display(prior), None),
        ("Stage2 Encoding", polar_encoding_display(refined), None),
        ("Confidence DoLP", confidence[0].detach().cpu().numpy(), (0.0, 1.0)),
        ("Confidence AoLP", confidence_aolp.detach().cpu().numpy(), (0.0, 1.0)),
        ("Gate Angle", pred["gate_angle"][0].detach().cpu().numpy(), (0.0, 1.0)),
    ]

    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
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
    input_native_sizes: set[str],
    input_sizes: set[str],
    output_resized_count: int,
) -> None:
    lines = [
        "Stage 2 angular residual refiner test summary",
        "model_type: angular_residual_refiner",
        f"residual_scale: {args.residual_scale}",
        f"angle_residual_scale: {args.angle_residual_scale}",
        f"min_gate: {args.min_gate}",
        f"base_channels: {args.base_channels}",
        f"preprocess_mode: {args.preprocess_mode}",
        f"normalize_mode: {args.normalize_mode}",
        f"input_native_size: {format_size_set(input_native_sizes)}",
        f"input_size: {format_size_set(input_sizes)}",
        f"divisible_by: {args.divisible_by}",
        f"resize_output_to_gt: {args.resize_output_to_gt}",
        f"output_resized_to_gt_count: {output_resized_count}",
        f"samples: {sample_count}",
        f"root_dir: {args.root_dir}",
        f"stage1_dir: {args.stage1_dir}",
        f"checkpoint: {args.checkpoint}",
        f"image_size: {args.image_size}",
        "saved_pred_encoding_png: True",
        "saved_stage1_encoding_png: True",
        "saved_gt_encoding_png: True",
        "encoding_png_dtype: uint16",
        "encoding_png_channel_order: [DoLP, cos2, sin2]",
        "aolp_visualization_range: [-90deg, 90deg]",
        "aolp_error_range_deg: [0, 90]",
        "",
        "Mean metrics:",
    ]
    for key in sorted(summary):
        lines.append(f"{key}: {format_float(summary[key])}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_size_set(values: set[str]) -> str:
    if not values:
        return "unknown"
    ordered = sorted(values)
    if len(ordered) <= 8:
        return ", ".join(ordered)
    return ", ".join(ordered[:8]) + f", ... ({len(ordered)} unique)"


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def run(args: argparse.Namespace) -> None:
    if args.preprocess_mode == "official_infer" and args.batch_size != 1:
        raise ValueError("official_infer keeps native sizes; use --batch_size 1.")
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    refined_dir = output_dir / "refined_npy"
    pred_encoding_dir = output_dir / "pred_encoding_png"
    stage1_encoding_dir = output_dir / "stage1_encoding_png"
    gt_encoding_dir = output_dir / "gt_encoding_png"
    vis_dir = output_dir / "vis"
    paper_vis_dir = output_dir / "paper_vis"
    refined_dir.mkdir(parents=True, exist_ok=True)
    pred_encoding_dir.mkdir(parents=True, exist_ok=True)
    stage1_encoding_dir.mkdir(parents=True, exist_ok=True)
    gt_encoding_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    paper_vis_dir.mkdir(parents=True, exist_ok=True)

    dataloader, dataset_size = build_dataloader(args, device)
    model = build_model(args, device)

    rows: list[dict[str, float | str]] = []
    processed = 0
    output_resized_count = 0
    input_native_sizes: set[str] = set()
    input_sizes: set[str] = set()
    with torch.no_grad():
        for batch in dataloader:
            input_native_sizes.update(str(value) for value in batch["input_native_size"])
            input_sizes.update(str(value) for value in batch["input_size"])
            rgb = batch["rgb"].to(device, non_blocking=True)
            prior = batch["prior"].to(device, non_blocking=True)
            confidence = batch["confidence"].to(device, non_blocking=True)
            pred = model(rgb, prior, confidence)

            names = batch["name"]
            polar_gt_cpu = batch["polar_gt"].cpu()
            prior_cpu = batch["prior"].cpu()
            confidence_cpu = batch["confidence"].cpu()
            rgb_cpu = batch["rgb"].cpu()
            pred_cpu = {key: value.detach().cpu() for key, value in pred.items()}

            for offset, name in enumerate(names):
                sample_index = processed + offset
                sample_refined = pred_cpu["refined"][offset]
                sample_prior = prior_cpu[offset]
                sample_gt = polar_gt_cpu[offset]
                np.save(
                    refined_dir / f"{name}.npy",
                    sample_refined.numpy().astype(np.float32),
                )
                prior_eval, prior_resized = resize_polar_to_gt_if_requested(
                    sample_prior,
                    sample_gt,
                    args.resize_output_to_gt,
                )
                refined_eval, refined_resized = resize_polar_to_gt_if_requested(
                    sample_refined,
                    sample_gt,
                    args.resize_output_to_gt,
                )
                if prior_resized or refined_resized:
                    output_resized_count += 1

                save_polar_encoding_png(refined_eval, pred_encoding_dir / f"{name}.png")
                save_polar_encoding_png(prior_eval, stage1_encoding_dir / f"{name}.png")
                save_polar_encoding_png(sample_gt, gt_encoding_dir / f"{name}.png")

                row: dict[str, float | str] = {"name": name}
                row.update(prefix_metrics("stage1", compute_metric_dict(prior_eval, sample_gt)))
                row.update(prefix_metrics("stage2", compute_metric_dict(refined_eval, sample_gt)))
                rows.append(row)

                save_paper_aop_dop_vis(
                    rgb_or_none=None,
                    gt_polar=sample_gt,
                    pred_polar=refined_eval,
                    save_path=paper_vis_dir / f"{name}_aop_dop_compare.png",
                )

                if args.vis_every > 0 and sample_index % args.vis_every == 0:
                    save_visualization(
                        rgb=rgb_cpu[offset],
                        polar_gt=sample_gt,
                        prior=prior_eval,
                        confidence=confidence_cpu[offset],
                        pred={
                            **{key: value[offset] for key, value in pred_cpu.items()},
                            "refined": refined_eval,
                        },
                        path=vis_dir / f"{name}.png",
                    )

            processed += len(names)
            print(f"processed {processed}/{dataset_size}", flush=True)

    write_metrics_csv(output_dir / "metrics.csv", rows)
    summary = summarize_rows(rows)
    write_summary(
        output_dir / "summary.txt",
        args,
        len(rows),
        summary,
        input_native_sizes,
        input_sizes,
        output_resized_count,
    )
    print(f"Done. Wrote outputs to {output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
