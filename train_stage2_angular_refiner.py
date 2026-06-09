"""Train the Stage 2 angular residual refiner."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.stage2_residual_dataset import Stage2ResidualDataset  # noqa: E402
from losses.stage2_angular_refiner_loss import Stage2AngularRefinerLoss  # noqa: E402
from models.stage2_angular_refiner_net import ConfidenceGuidedAngularResidualRefiner  # noqa: E402
from utils.ddp_utils import (  # noqa: E402
    barrier,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    setup_distributed,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRIC_KEYS = (
    "loss",
    "loss_dolp_final",
    "loss_vector_final",
    "loss_aolp_final",
    "loss_delta_dolp",
    "loss_delta_angle",
    "loss_lowfreq",
    "loss_edge",
    "loss_residual_reg",
    "loss_gate_reg",
    "mean_reliability",
    "mean_gate_dolp",
    "mean_gate_angle",
    "mean_abs_delta_dolp",
    "mean_abs_delta_angle_deg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 2 angular residual refiner.")
    parser.add_argument("--root_dir", type=str, default=str(Path.home() / "Documents"))
    parser.add_argument("--stage1_dir", type=str, default="./stage1_exports")
    parser.add_argument("--val_root_dir", type=str, default=None)
    parser.add_argument("--val_stage1_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_stage2_angular_refiner")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument(
        "--preprocess_mode",
        choices=("resize256", "official_train"),
        default="resize256",
    )
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument(
        "--normalize_mode",
        choices=("fixed255", "image_max"),
        default="fixed255",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--residual_scale", type=float, default=0.3)
    parser.add_argument("--angle_residual_scale", type=float, default=math.pi)
    parser.add_argument("--min_gate", type=float, default=0.2)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--save_optimizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--distributed", action="store_true")
    return parser.parse_args()


def normalize_optional_path(value: str | None) -> str | None:
    if value is None:
        return None
    if value.lower() in {"none", "null", ""}:
        return None
    return value


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(
    root_dir: str,
    stage1_dir: str,
    image_size: int,
    preprocess_mode: str,
    crop_size: int,
    normalize_mode: str,
    random_crop: bool,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    max_samples: int | None,
    pin_memory: bool,
    distributed: bool = False,
) -> tuple[DataLoader, int, DistributedSampler | None]:
    dataset = Stage2ResidualDataset(
        root_dir=root_dir,
        stage1_dir=stage1_dir,
        image_size=image_size,
        preprocess_mode=preprocess_mode,
        crop_size=crop_size,
        normalize_mode=normalize_mode,
        random_crop=random_crop,
    )
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive or None.")
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    sampler = (
        DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
        if distributed
        else None
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataloader, len(dataset), sampler


def build_val_dataloader(args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, int] | tuple[None, int]:
    val_root_dir = normalize_optional_path(args.val_root_dir)
    val_stage1_dir = normalize_optional_path(args.val_stage1_dir)
    if val_root_dir is None and val_stage1_dir is None:
        return None, 0
    if val_root_dir is None or val_stage1_dir is None:
        raise ValueError("val_root_dir and val_stage1_dir must be provided together.")
    dataloader, size, _ = build_dataloader(
        root_dir=val_root_dir,
        stage1_dir=val_stage1_dir,
        image_size=args.image_size,
        preprocess_mode=args.preprocess_mode,
        crop_size=args.crop_size,
        normalize_mode=args.normalize_mode,
        random_crop=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        max_samples=args.max_val_samples,
        pin_memory=device.type == "cuda",
        distributed=False,
    )
    return dataloader, size


def build_model(args: argparse.Namespace, device: torch.device) -> ConfidenceGuidedAngularResidualRefiner:
    model = ConfidenceGuidedAngularResidualRefiner(
        base_channels=args.base_channels,
        residual_scale=args.residual_scale,
        angle_residual_scale=args.angle_residual_scale,
        min_gate=args.min_gate,
    )
    return model.to(device)


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model_to_save.state_dict(),
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }
    if args.save_optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(strip_module_prefix(state_dict))
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    return start_epoch, global_step, best_val_loss


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def prune_epoch_checkpoints(save_dir: Path, save_total_limit: int | None) -> None:
    if save_total_limit is None:
        return
    epoch_checkpoints = sorted(save_dir.glob("epoch_*.pth"))
    while len(epoch_checkpoints) > save_total_limit:
        old_checkpoint = epoch_checkpoints.pop(0)
        old_checkpoint.unlink()
        print(f"Deleted old checkpoint: {old_checkpoint}", flush=True)


def append_log(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def write_config(path: Path, args: argparse.Namespace) -> None:
    payload = {
        **vars(args),
        "model_type": "angular_residual_refiner",
        "checkpoint_policy": (
            "Writes best_val.pth, last.pth, and epoch_xxx.pth at save_freq epochs. "
            "Optimizer state is saved only when --save_optimizer is set."
        ),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def move_batch_to_device(
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = batch["rgb"].to(device, non_blocking=True)
    polar_gt = batch["polar_gt"].to(device, non_blocking=True)
    prior = batch["prior"].to(device, non_blocking=True)
    confidence = batch["confidence"].to(device, non_blocking=True)
    return rgb, polar_gt, prior, confidence


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: Stage2AngularRefinerLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    log_freq: int,
    global_step: int,
) -> tuple[dict[str, float], int, dict[str, torch.Tensor] | None]:
    model.train()
    sums = {key: 0.0 for key in METRIC_KEYS}
    num_samples = 0
    vis_data = None

    for batch_index, batch in enumerate(dataloader):
        rgb, polar_gt, prior, confidence = move_batch_to_device(batch, device)
        batch_size = rgb.shape[0]

        optimizer.zero_grad(set_to_none=True)
        pred = model(rgb, prior, confidence)
        loss_dict = loss_fn(pred, polar_gt, prior, confidence)
        loss_dict["loss"].backward()
        optimizer.step()

        global_step += 1
        num_samples += batch_size
        for key in sums:
            sums[key] += float(loss_dict[key].detach()) * batch_size

        if vis_data is None:
            vis_data = {
                "rgb": rgb.detach().cpu(),
                "polar_gt": polar_gt.detach().cpu(),
                "prior": prior.detach().cpu(),
                "confidence": confidence.detach().cpu(),
                "pred": {key: value.detach().cpu() for key, value in pred.items()},
            }

        if log_freq > 0 and global_step % log_freq == 0 and is_main_process():
            print(
                f"epoch {epoch}/{num_epochs} "
                f"step {global_step} "
                f"batch {batch_index + 1}/{len(dataloader)} "
                f"loss {float(loss_dict['loss'].detach()):.6f}",
                flush=True,
            )

    if num_samples == 0:
        raise RuntimeError("Training dataloader produced no samples.")
    return {key: value / num_samples for key, value in sums.items()}, global_step, vis_data


def evaluate(
    model: torch.nn.Module,
    loss_fn: Stage2AngularRefinerLoss,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sums = {key: 0.0 for key in METRIC_KEYS}
    num_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            rgb, polar_gt, prior, confidence = move_batch_to_device(batch, device)
            batch_size = rgb.shape[0]
            pred = model(rgb, prior, confidence)
            loss_dict = loss_fn(pred, polar_gt, prior, confidence)
            num_samples += batch_size
            for key in sums:
                sums[key] += float(loss_dict[key].detach()) * batch_size
    if num_samples == 0:
        raise RuntimeError("Validation dataloader produced no samples.")
    return {key: value / num_samples for key, value in sums.items()}


def tensor_to_rgb_image(rgb: torch.Tensor) -> np.ndarray:
    image = rgb.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip((image + 1.0) * 0.5, 0.0, 1.0)


def to_display_map(tensor: torch.Tensor, value_range: tuple[float, float]) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    min_value, max_value = value_range
    return np.clip((array - min_value) / (max_value - min_value), 0.0, 1.0)


def save_visualization(
    vis_data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    path: Path,
) -> None:
    rgb = vis_data["rgb"][0]
    polar_gt = vis_data["polar_gt"][0]
    prior = vis_data["prior"][0]
    pred = vis_data["pred"]
    refined = pred["refined"][0]
    delta_dolp = pred["delta_dolp"][0, 0]
    delta_angle_deg = pred["delta_angle"][0, 0] * (180.0 / np.pi)
    gate_dolp = pred["gate_dolp"][0, 0]
    gate_angle = pred["gate_angle"][0, 0]
    vector_error_prior = torch.sqrt((prior[1] - polar_gt[1]) ** 2 + (prior[2] - polar_gt[2]) ** 2)
    vector_error_refined = torch.sqrt(
        (refined[1] - polar_gt[1]) ** 2 + (refined[2] - polar_gt[2]) ** 2
    )

    panels = [
        ("RGB", tensor_to_rgb_image(rgb), None),
        ("GT DoLP", polar_gt[0].numpy(), (0.0, 1.0)),
        ("Prior DoLP", prior[0].numpy(), (0.0, 1.0)),
        ("Refined DoLP", refined[0].numpy(), (0.0, 1.0)),
        ("GT cos2", to_display_map(polar_gt[1], (-1.0, 1.0)), (0.0, 1.0)),
        ("Prior cos2", to_display_map(prior[1], (-1.0, 1.0)), (0.0, 1.0)),
        ("Refined cos2", to_display_map(refined[1], (-1.0, 1.0)), (0.0, 1.0)),
        ("GT sin2", to_display_map(polar_gt[2], (-1.0, 1.0)), (0.0, 1.0)),
        ("Prior sin2", to_display_map(prior[2], (-1.0, 1.0)), (0.0, 1.0)),
        ("Refined sin2", to_display_map(refined[2], (-1.0, 1.0)), (0.0, 1.0)),
        ("Delta DoLP", delta_dolp.numpy(), (-0.3, 0.3)),
        ("Delta Angle (deg)", delta_angle_deg.numpy(), (-180.0, 180.0)),
        ("Gate DoLP", gate_dolp.numpy(), (0.0, 1.0)),
        ("Gate Angle", gate_angle.numpy(), (0.0, 1.0)),
        ("Vector Error Prior", vector_error_prior.numpy(), (0.0, 2.0)),
        ("Vector Error Refined", vector_error_refined.numpy(), (0.0, 2.0)),
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


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return " | ".join(f"{prefix}_{key}={metrics[key]:.6f}" for key in METRIC_KEYS)


def should_save_vis(epoch: int, num_epochs: int) -> bool:
    interval = max(num_epochs // 10, 1)
    return epoch == 1 or epoch == num_epochs or epoch % interval == 0


def main() -> None:
    args = parse_args()
    if args.save_total_limit is not None and args.save_total_limit < 0:
        raise ValueError("save_total_limit must be non-negative or None.")

    set_seed(args.seed)
    device, distributed = setup_distributed(args)
    rank = get_rank()
    world_size = get_world_size()
    save_dir = Path(args.save_dir)
    vis_dir = save_dir / "vis"
    if is_main_process():
        save_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)
        write_config(save_dir / "config.json", args)
    barrier()
    log_path = save_dir / "train_log.txt"

    train_loader, train_size, train_sampler = build_dataloader(
        root_dir=args.root_dir,
        stage1_dir=args.stage1_dir,
        image_size=args.image_size,
        preprocess_mode=args.preprocess_mode,
        crop_size=args.crop_size,
        normalize_mode=args.normalize_mode,
        random_crop=args.preprocess_mode == "official_train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        max_samples=args.max_train_samples,
        pin_memory=device.type == "cuda",
        distributed=distributed,
    )
    if is_main_process():
        val_loader, val_size = build_val_dataloader(args, device)
    else:
        val_loader, val_size = None, 0

    model = build_model(args, device)
    loss_fn = Stage2AngularRefinerLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    global_step = 0
    best_score = float("inf")
    if normalize_optional_path(args.resume) is not None:
        start_epoch, global_step, best_score = load_checkpoint(
            args.resume,
            model,
            optimizer,
            device,
        )

    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )

    try:
        if is_main_process():
            append_log(
                log_path,
                f"Start Stage2 angular refiner training on {device}, "
                f"train_samples={train_size}, val_samples={val_size}, "
                f"batch_size_per_gpu={args.batch_size}, world_size={world_size}, "
                f"rank={rank}, local_rank={args.local_rank}, "
                f"effective_batch_size={args.batch_size * world_size}, "
                f"distributed={distributed}, base_channels={args.base_channels}, "
                f"residual_scale={args.residual_scale}, "
                f"angle_residual_scale={args.angle_residual_scale}, min_gate={args.min_gate}, "
                f"preprocess_mode={args.preprocess_mode}, crop_size={args.crop_size}, "
                f"normalize_mode={args.normalize_mode}, save_optimizer={args.save_optimizer}, "
                f"root_dir={args.root_dir}, stage1_dir={args.stage1_dir}, "
                f"val_root_dir={args.val_root_dir}, val_stage1_dir={args.val_stage1_dir}, "
                f"num_epochs={args.num_epochs}, lr={args.lr}, weight_decay={args.weight_decay}",
            )

        for epoch in range(start_epoch, args.num_epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_metrics, global_step, vis_data = train_one_epoch(
                model=model,
                loss_fn=loss_fn,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                num_epochs=args.num_epochs,
                log_freq=args.log_freq,
                global_step=global_step,
            )

            if is_main_process():
                eval_model = model.module if hasattr(model, "module") else model
                val_metrics = None
                if val_loader is not None:
                    val_metrics = evaluate(eval_model, loss_fn, val_loader, device)
                    current_score = val_metrics["loss"]
                    if current_score < best_score:
                        best_score = current_score
                        save_checkpoint(
                            save_dir / "best_val.pth",
                            epoch,
                            global_step,
                            model,
                            optimizer,
                            best_score,
                            args,
                        )
                else:
                    current_score = train_metrics["loss"]
                    if current_score < best_score:
                        best_score = current_score
                        save_checkpoint(
                            save_dir / "best_train.pth",
                            epoch,
                            global_step,
                            model,
                            optimizer,
                            best_score,
                            args,
                        )

                save_checkpoint(
                    save_dir / "last.pth",
                    epoch,
                    global_step,
                    model,
                    optimizer,
                    best_score,
                    args,
                )
                if args.save_freq > 0 and epoch % args.save_freq == 0:
                    save_checkpoint(
                        save_dir / f"epoch_{epoch:04d}.pth",
                        epoch,
                        global_step,
                        model,
                        optimizer,
                        best_score,
                        args,
                    )
                    prune_epoch_checkpoints(save_dir, args.save_total_limit)

                message = f"epoch {epoch}/{args.num_epochs} | {format_metrics('train', train_metrics)}"
                if val_metrics is not None:
                    message += " | " + format_metrics("val", val_metrics)
                append_log(log_path, message)

                if vis_data is not None and should_save_vis(epoch, args.num_epochs):
                    save_visualization(vis_data, vis_dir / f"epoch_{epoch:03d}.png")
            barrier()

        if is_main_process():
            append_log(log_path, f"Finished training. best_loss={best_score:.6f}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
