"""Train Version E on a manifest split with a frozen online Stage1 prior."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from datasets.manifest_dataset import Stage1ManifestDataset
from losses.stage2_asymmetric_restormer_loss import Stage2AsymmetricRestormerLoss
from models.stage2_asymmetric_restormer_refiner import Stage2AsymmetricRestormerRefiner
from train_direct_unetpp_baseline import append_log, resolve_device, strip_module_prefix
from train_stage2_prior_guided_restormer import average_metrics, parse_int_tuple
from train_stage2_prior_guided_restormer_online_stage1 import (
    build_stage1,
    make_online_prior,
    set_seed,
    unwrap_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Version E on final_new512 with online frozen Stage1 guidance."
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--val_split", choices=("train", "val"), default="val")
    parser.add_argument("--stage1_checkpoint", type=str, required=True)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints_stage2_asymmetric_restormer_ve_manifest",
    )
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument(
        "--num_blocks", type=lambda value: parse_int_tuple(value, 3), default=(2, 2, 4)
    )
    parser.add_argument(
        "--num_heads", type=lambda value: parse_int_tuple(value, 3), default=(1, 2, 4)
    )
    parser.add_argument("--ffn_expansion", type=float, default=2.66)
    parser.add_argument("--dolp_residual_scale", type=float, default=0.25)
    parser.add_argument("--angle_residual_scale", type=float, default=math.pi / 2.0)
    parser.add_argument("--min_angle_gate", type=float, default=0.05)
    parser.add_argument("--weak_factor", type=int, default=4)
    parser.add_argument("--confidence_scale", type=float, default=0.5)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--save_freq", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data_parallel", action="store_true")
    return parser.parse_args()


def limit_dataset(dataset: Stage1ManifestDataset, maximum: int | None) -> Stage1ManifestDataset | Subset:
    if maximum is None:
        return dataset
    if maximum <= 0:
        raise ValueError("sample limits must be positive")
    return Subset(dataset, range(min(maximum, len(dataset))))


def build_stage2(args: argparse.Namespace, device: torch.device) -> Stage2AsymmetricRestormerRefiner:
    return Stage2AsymmetricRestormerRefiner(
        dim=args.dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        expansion=args.ffn_expansion,
        dolp_residual_scale=args.dolp_residual_scale,
        angle_residual_scale=args.angle_residual_scale,
        min_angle_gate=args.min_angle_gate,
    ).to(device)


def run_epoch(
    stage2: nn.Module,
    stage1: nn.Module,
    loss_fn: Stage2AsymmetricRestormerLoss,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
) -> dict[str, float]:
    is_train = optimizer is not None
    stage2.train(is_train)
    stage1.eval()
    sums = {key: 0.0 for key in loss_fn.METRIC_KEYS}
    total = 0
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        target = batch["polar"].to(device, non_blocking=True)
        prior, confidence = make_online_prior(
            stage1, rgb, args.weak_factor, args.confidence_scale
        )
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
        for key in loss_fn.METRIC_KEYS:
            sums[key] += float(losses[key].detach()) * batch_size
    return average_metrics(sums, total)


def format_metrics(prefix: str, metrics: dict[str, float], keys: tuple[str, ...]) -> str:
    return " | ".join(f"{prefix}_{key}={metrics[key]:.6f}" for key in keys)


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
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
            "model_type": "stage2_asymmetric_restormer_refiner_ve_online_stage1_manifest",
            "architecture_version": "E",
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
    unwrap_model(model).load_state_dict(strip_module_prefix(state_dict))
    optimizer_state = checkpoint.get("optimizer")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
    return int(checkpoint.get("epoch", 0)) + 1, float(
        checkpoint.get("best_val_loss", float("inf"))
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log_path = save_dir / "train.log"

    train_dataset = limit_dataset(
        Stage1ManifestDataset(
            args.manifest,
            dataset_root=args.dataset_root,
            split=args.split,
            crop_size=480,
            random_crop=True,
            augment=True,
        ),
        args.max_train_samples,
    )
    val_dataset = limit_dataset(
        Stage1ManifestDataset(
            args.manifest,
            dataset_root=args.dataset_root,
            split=args.val_split,
            crop_size=480,
            random_crop=False,
            augment=False,
        ),
        args.max_val_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    stage1: nn.Module = build_stage1(args.stage1_checkpoint, device)
    stage2: nn.Module = build_stage2(args, device)
    use_data_parallel = (
        args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1
    )
    if use_data_parallel:
        stage1 = nn.DataParallel(stage1)
        stage2 = nn.DataParallel(stage2)
    optimizer = torch.optim.AdamW(
        stage2.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = Stage2AsymmetricRestormerLoss(
        dolp_residual_scale=args.dolp_residual_scale,
        angle_residual_scale=args.angle_residual_scale,
    ).to(device)

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, stage2, optimizer, device
        )

    append_log(
        log_path,
        f"[config] architecture=E dataset=manifest train={len(train_dataset)} "
        f"val={len(val_dataset)} batch_size={args.batch_size} "
        f"data_parallel={use_data_parallel} visible_gpus={torch.cuda.device_count()} "
        f"stage1_checkpoint={args.stage1_checkpoint} weak_factor={args.weak_factor} "
        f"confidence_scale={args.confidence_scale} dolp_residual_scale="
        f"{args.dolp_residual_scale} dim={args.dim} num_blocks={args.num_blocks}",
    )
    for epoch in range(start_epoch, args.num_epochs + 1):
        train_metrics = run_epoch(
            stage2, stage1, loss_fn, train_loader, device, optimizer, args
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                stage2, stage1, loss_fn, val_loader, device, optimizer=None, args=args
            )
        score = val_metrics["loss"]
        append_log(
            log_path,
            f"epoch {epoch}/{args.num_epochs} | "
            f"{format_metrics('train', train_metrics, loss_fn.METRIC_KEYS)} | "
            f"{format_metrics('val', val_metrics, loss_fn.METRIC_KEYS)}",
        )
        if score < best_val_loss:
            best_val_loss = score
            save_checkpoint(
                save_dir / "best_val.pth", epoch, stage2, optimizer, best_val_loss, args
            )
            append_log(log_path, f"[best_val] epoch={epoch} loss={best_val_loss:.6f}")
        save_checkpoint(save_dir / "last.pth", epoch, stage2, optimizer, best_val_loss, args)
        if args.save_freq > 0 and epoch % args.save_freq == 0:
            save_checkpoint(
                save_dir / f"epoch_{epoch:03d}.pth",
                epoch,
                stage2,
                optimizer,
                best_val_loss,
                args,
            )
    append_log(log_path, f"Finished training. best_val_loss={best_val_loss:.6f}")


if __name__ == "__main__":
    main()
