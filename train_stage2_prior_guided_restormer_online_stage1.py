"""Train Version C with an online frozen weak Stage1 prior."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.stage1_prior_dataset import Stage1PriorDataset
from losses.stage2_prior_guided_restormer_loss import Stage2PriorGuidedRestormerLoss
from models.polar_prior_net import PolarPriorNet
from models.stage2_prior_guided_restormer_refiner import Stage2PriorGuidedRestormerRefiner
from train_direct_unetpp_baseline import append_log, resolve_device, strip_module_prefix
from train_stage2_prior_guided_restormer import METRIC_KEYS, average_metrics, parse_int_tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Version C with online weak Stage1.")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--val_root_dir", type=str, required=True)
    parser.add_argument("--stage1_checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_stage2_prior_guided_restormer_online_stage1")
    parser.add_argument("--preprocess_mode", choices=("resize256", "official_train"), default="official_train")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--normalize_mode", choices=("fixed255", "image_max"), default="image_max")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--num_blocks", type=lambda value: parse_int_tuple(value, 3), default=(2, 2, 4))
    parser.add_argument("--num_heads", type=lambda value: parse_int_tuple(value, 3), default=(1, 2, 4))
    parser.add_argument("--ffn_expansion", type=float, default=2.66)
    parser.add_argument("--residual_scale", type=float, default=0.5)
    parser.add_argument("--angle_residual_scale", type=float, default=math.pi)
    parser.add_argument("--min_gate", type=float, default=0.05)
    parser.add_argument("--weak_factor", type=int, default=4)
    parser.add_argument("--confidence_scale", type=float, default=0.5)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--save_freq", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(args: argparse.Namespace, root_dir: str, augment: bool, max_samples: int | None):
    dataset = Stage1PriorDataset(
        root_dir=root_dir,
        image_size=args.image_size if args.preprocess_mode == "resize256" else None,
        preprocess_mode=args.preprocess_mode,
        crop_size=args.crop_size,
        normalize_mode=args.normalize_mode,
        random_crop=augment,
        augment=False,
    )
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return dataset


def build_stage2(args: argparse.Namespace, device: torch.device) -> Stage2PriorGuidedRestormerRefiner:
    return Stage2PriorGuidedRestormerRefiner(
        dim=args.dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        expansion=args.ffn_expansion,
        residual_scale=args.residual_scale,
        angle_residual_scale=args.angle_residual_scale,
        min_gate=args.min_gate,
    ).to(device)


def build_stage1(checkpoint_path: str, device: torch.device) -> PolarPriorNet:
    model = PolarPriorNet(encoder_weights=None)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(strip_module_prefix(state_dict))
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def lowpass(tensor: torch.Tensor, factor: int) -> torch.Tensor:
    if factor <= 1:
        return tensor
    height, width = tensor.shape[-2:]
    small = F.interpolate(tensor, size=(max(height // factor, 1), max(width // factor, 1)), mode="area")
    return F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)


def clamp_prior(prior: torch.Tensor) -> torch.Tensor:
    dolp = prior[:, 0:1].clamp(0.0, 1.0)
    cos_sin = prior[:, 1:3].clamp(-1.0, 1.0)
    norm = torch.sqrt((cos_sin * cos_sin).sum(dim=1, keepdim=True).clamp_min(1e-6))
    return torch.cat([dolp, cos_sin / norm], dim=1)


def make_online_prior(
    stage1: nn.Module,
    rgb: torch.Tensor,
    weak_factor: int,
    confidence_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        use_amp = rgb.is_cuda
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            output = stage1(rgb)
            prior = clamp_prior(lowpass(output["polar_prior"], weak_factor))
            confidence = (lowpass(output["confidence"], weak_factor) * confidence_scale).clamp(0.0, 1.0)
    return prior.float(), confidence.float()


def run_epoch(
    stage2: nn.Module,
    stage1: nn.Module,
    loss_fn: Stage2PriorGuidedRestormerLoss,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
) -> dict[str, float]:
    is_train = optimizer is not None
    stage2.train(is_train)
    stage1.eval()
    sums = {key: 0.0 for key in METRIC_KEYS}
    total = 0
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        target = batch["polar"].to(device, non_blocking=True)
        prior, confidence = make_online_prior(stage1, rgb, args.weak_factor, args.confidence_scale)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            pred = stage2(rgb, prior, confidence)
            losses = loss_fn(pred, target, prior, confidence)
            if is_train:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(stage2.parameters(), 1.0)
                optimizer.step()
        batch_size = int(rgb.shape[0])
        total += batch_size
        for key in METRIC_KEYS:
            sums[key] += float(losses[key].detach()) * batch_size
    return average_metrics(sums, total)


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return " | ".join(f"{prefix}_{key}={metrics[key]:.6f}" for key in METRIC_KEYS)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
            "model_type": "stage2_prior_guided_restormer_refiner_online_stage1",
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(strip_module_prefix(state_dict))
    optimizer_state = checkpoint.get("optimizer")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
    return int(checkpoint.get("epoch", 0)) + 1, float(checkpoint.get("best_val_loss", float("inf")))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_path = save_dir / "train.log"

    train_dataset = build_dataset(args, args.root_dir, augment=True, max_samples=args.max_train_samples)
    val_dataset = build_dataset(args, args.val_root_dir, augment=False, max_samples=args.max_val_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    stage1 = build_stage1(args.stage1_checkpoint, device)
    stage2 = build_stage2(args, device)
    optimizer = torch.optim.AdamW(stage2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = Stage2PriorGuidedRestormerLoss().to(device)

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(args.resume, stage2, optimizer, device)

    append_log(
        log_path,
        f"[config] train={len(train_dataset)} val={len(val_dataset)} batch_size={args.batch_size} "
        f"stage1_checkpoint={args.stage1_checkpoint} weak_factor={args.weak_factor} "
        f"confidence_scale={args.confidence_scale} dim={args.dim} num_blocks={args.num_blocks}",
    )
    for epoch in range(start_epoch, args.num_epochs + 1):
        train_metrics = run_epoch(stage2, stage1, loss_fn, train_loader, device, optimizer, args)
        with torch.no_grad():
            val_metrics = run_epoch(stage2, stage1, loss_fn, val_loader, device, optimizer=None, args=args)
        score = val_metrics["loss"]
        append_log(
            log_path,
            f"epoch {epoch}/{args.num_epochs} | "
            f"{format_metrics('train', train_metrics)} | {format_metrics('val', val_metrics)}",
        )
        if score < best_val_loss:
            best_val_loss = score
            save_checkpoint(save_dir / "best_val.pth", epoch, stage2, optimizer, best_val_loss, args)
            append_log(log_path, f"[best_val] epoch={epoch} loss={best_val_loss:.6f}")
        save_checkpoint(save_dir / "last.pth", epoch, stage2, optimizer, best_val_loss, args)
        if args.save_freq > 0 and epoch % args.save_freq == 0:
            save_checkpoint(save_dir / f"epoch_{epoch:03d}.pth", epoch, stage2, optimizer, best_val_loss, args)
    append_log(log_path, f"Finished training. best_val_loss={best_val_loss:.6f}")


if __name__ == "__main__":
    main()
