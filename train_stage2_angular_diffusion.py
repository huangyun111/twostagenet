"""Train Stage 2 angular physical residual diffusion."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.stage2_residual_dataset import Stage2ResidualDataset  # noqa: E402
from losses.stage2_angular_diffusion_loss import Stage2AngularDiffusionLoss  # noqa: E402
from models.stage2_angular_diffusion_net import (  # noqa: E402
    DiffusionSchedule,
    Stage2AngularResidualDiffusion,
    target_residual_from_prior,
)
from utils.ddp_utils import (  # noqa: E402
    barrier,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    setup_distributed,
)


METRIC_KEYS = (
    "loss",
    "loss_delta_dolp",
    "loss_delta_angle",
    "loss_dolp_final",
    "loss_vector_final",
    "loss_aolp_final",
    "loss_edge",
    "loss_lowfreq",
    "loss_residual_reg",
    "loss_conf_residual",
    "mean_abs_delta_dolp",
    "mean_abs_delta_angle",
    "mean_reliability",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 2 angular residual diffusion.")
    parser.add_argument("--root_dir", type=str, default=str(Path.home() / "Documents"))
    parser.add_argument("--stage1_dir", type=str, default="./stage1_exports")
    parser.add_argument("--val_root_dir", type=str, default=None)
    parser.add_argument("--val_stage1_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_stage2_angular_diffusion")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument(
        "--preprocess_mode",
        choices=("resize256", "official_train"),
        default="official_train",
    )
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument(
        "--normalize_mode",
        choices=("fixed255", "image_max"),
        default="image_max",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--residual_scale", type=float, default=0.3)
    parser.add_argument("--angle_residual_scale", type=float, default=math.pi)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--val_inference_steps", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--save_optimizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--val_max_samples", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def normalize_optional_path(value: str | None) -> str | None:
    if value is None:
        return None
    if value.lower() in {"", "none", "null"}:
        return None
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(
    root_dir: str,
    stage1_dir: str,
    args: argparse.Namespace,
    random_crop: bool,
    shuffle: bool,
    max_samples: int | None,
    device: torch.device,
    distributed: bool = False,
) -> tuple[DataLoader, int, DistributedSampler | None]:
    dataset = Stage2ResidualDataset(
        root_dir=root_dir,
        stage1_dir=stage1_dir,
        image_size=args.image_size,
        preprocess_mode=args.preprocess_mode,
        crop_size=args.crop_size,
        normalize_mode=args.normalize_mode,
        random_crop=random_crop,
    )
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive or None.")
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False) if distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    return dataloader, len(dataset), sampler


def build_val_dataloader(args: argparse.Namespace, device: torch.device) -> tuple[DataLoader | None, int]:
    val_root_dir = normalize_optional_path(args.val_root_dir)
    val_stage1_dir = normalize_optional_path(args.val_stage1_dir)
    if val_root_dir is None and val_stage1_dir is None:
        return None, 0
    if val_root_dir is None or val_stage1_dir is None:
        raise ValueError("val_root_dir and val_stage1_dir must be provided together.")
    dataloader, size, _ = build_dataloader(
        root_dir=val_root_dir,
        stage1_dir=val_stage1_dir,
        args=args,
        random_crop=False,
        shuffle=False,
        max_samples=args.val_max_samples,
        device=device,
        distributed=False,
    )
    return dataloader, size


def build_model(args: argparse.Namespace, device: torch.device) -> Stage2AngularResidualDiffusion:
    model = Stage2AngularResidualDiffusion(
        base_channels=args.base_channels,
        residual_scale=args.residual_scale,
        angle_residual_scale=args.angle_residual_scale,
    )
    return model.to(device)


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model_to_save.state_dict(),
        "best_val_loss": best_loss,
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
    best_loss = float(checkpoint.get("best_val_loss", float("inf")))
    return start_epoch, global_step, best_loss


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
        "model_type": "stage2_angular_residual_diffusion",
        "residual_type": "[delta_dolp, delta_angle]",
        "uses_vae": False,
        "prediction_type": "x0_residual",
        "checkpoint_policy": (
            "Writes best_val.pth, last.pth, and sparse epoch checkpoints. "
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
    loss_fn: Stage2AngularDiffusionLoss,
    schedule: DiffusionSchedule,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    log_freq: int,
    global_step: int,
) -> tuple[dict[str, float], int]:
    model.train()
    sums = {key: 0.0 for key in METRIC_KEYS}
    num_samples = 0

    for batch_index, batch in enumerate(dataloader):
        rgb, polar_gt, prior, confidence = move_batch_to_device(batch, device)
        batch_size = rgb.shape[0]
        residual_gt = target_residual_from_prior(prior, polar_gt)
        timestep = torch.randint(
            0,
            schedule.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn_like(residual_gt)
        residual_noisy = schedule.q_sample(residual_gt, timestep, noise)

        optimizer.zero_grad(set_to_none=True)
        pred = model(rgb, prior, confidence, residual_noisy, timestep)
        loss_dict = loss_fn(pred, polar_gt, prior, confidence, residual_gt)
        loss_dict["loss"].backward()
        optimizer.step()

        global_step += 1
        num_samples += batch_size
        for key in sums:
            sums[key] += float(loss_dict[key].detach()) * batch_size

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
    return {key: value / num_samples for key, value in sums.items()}, global_step


def evaluate_x0(
    model: torch.nn.Module,
    loss_fn: Stage2AngularDiffusionLoss,
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
            residual_gt = target_residual_from_prior(prior, polar_gt)
            timestep = torch.zeros(batch_size, device=device, dtype=torch.long)
            pred = model(rgb, prior, confidence, residual_gt, timestep)
            loss_dict = loss_fn(pred, polar_gt, prior, confidence, residual_gt)
            num_samples += batch_size
            for key in sums:
                sums[key] += float(loss_dict[key].detach()) * batch_size
    if num_samples == 0:
        raise RuntimeError("Validation dataloader produced no samples.")
    return {key: value / num_samples for key, value in sums.items()}


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return " | ".join(f"{prefix}_{key}={metrics[key]:.6f}" for key in METRIC_KEYS)


def main() -> None:
    args = parse_args()
    if args.save_total_limit is not None and args.save_total_limit < 0:
        raise ValueError("save_total_limit must be non-negative or None.")
    if args.num_inference_steps <= 0 or args.val_inference_steps <= 0:
        raise ValueError("num_inference_steps and val_inference_steps must be positive.")

    set_seed(args.seed)
    device, distributed = setup_distributed(args)
    rank = get_rank()
    world_size = get_world_size()
    save_dir = Path(args.save_dir)
    if is_main_process():
        save_dir.mkdir(parents=True, exist_ok=True)
        write_config(save_dir / "config.json", args)
    barrier()
    log_path = save_dir / "train_log.txt"

    train_loader, train_size, train_sampler = build_dataloader(
        root_dir=args.root_dir,
        stage1_dir=args.stage1_dir,
        args=args,
        random_crop=args.preprocess_mode == "official_train",
        shuffle=True,
        max_samples=args.max_samples,
        device=device,
        distributed=distributed,
    )
    if is_main_process():
        val_loader, val_size = build_val_dataloader(args, device)
    else:
        val_loader, val_size = None, 0

    model = build_model(args, device)
    loss_fn = Stage2AngularDiffusionLoss().to(device)
    schedule = DiffusionSchedule(args.num_train_timesteps, beta_schedule="cosine", device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
                f"Start Stage2 angular residual diffusion training on {device}, "
                f"train_samples={train_size}, val_samples={val_size}, "
                f"batch_size_per_gpu={args.batch_size}, world_size={world_size}, "
                f"rank={rank}, local_rank={args.local_rank}, "
                f"effective_batch_size={args.batch_size * world_size}, "
                f"base_channels={args.base_channels}, "
                f"residual_scale={args.residual_scale}, "
                f"angle_residual_scale={args.angle_residual_scale}, "
                f"num_train_timesteps={args.num_train_timesteps}, "
                f"num_inference_steps={args.num_inference_steps}, "
                f"val_inference_steps={args.val_inference_steps}, "
                f"preprocess_mode={args.preprocess_mode}, crop_size={args.crop_size}, "
                f"normalize_mode={args.normalize_mode}, save_optimizer={args.save_optimizer}",
            )

        for epoch in range(start_epoch, args.num_epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_metrics, global_step = train_one_epoch(
                model=model,
                loss_fn=loss_fn,
                schedule=schedule,
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
                    val_metrics = evaluate_x0(eval_model, loss_fn, val_loader, device)
                    current_score = val_metrics["loss"]
                else:
                    current_score = train_metrics["loss"]

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
            barrier()

        if is_main_process():
            append_log(log_path, f"Finished training. best_loss={best_score:.6f}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
