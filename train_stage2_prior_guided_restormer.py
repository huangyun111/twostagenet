"""Train Version C: prior-guided angular Restormer Stage 2 refiner."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from datasets.manifest_dataset import Stage2ManifestDataset
from datasets.stage2_residual_dataset import Stage2ResidualDataset
from losses.stage2_prior_guided_restormer_loss import Stage2PriorGuidedRestormerLoss
from models.stage2_prior_guided_restormer_refiner import Stage2PriorGuidedRestormerRefiner
from train_direct_unetpp_baseline import append_log, resolve_device, strip_module_prefix


METRIC_KEYS = (
    "loss",
    "loss_dolp",
    "loss_vector",
    "loss_aolp",
    "loss_high_dolp_aolp",
    "loss_edge",
    "loss_residual_reg",
    "loss_gate_reg",
    "mean_refinement_gate",
    "mean_abs_delta_dolp",
    "mean_abs_delta_angle_deg",
)


def parse_int_tuple(value: str, expected_len: int) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(items) != expected_len:
        raise argparse.ArgumentTypeError(f"expected {expected_len} comma-separated integers")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Stage2 Prior-Guided Angular Restormer Refiner."
    )
    parser.add_argument("--root_dir", type=str, default=str(Path.home() / "Documents"))
    parser.add_argument("--stage1_dir", type=str, required=True)
    parser.add_argument("--val_root_dir", type=str, default=None)
    parser.add_argument("--val_stage1_dir", type=str, default=None)
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Path to assembly_manifest.json. When set, manifest mode is used; "
        "--root_dir/--val_* are ignored and priors are read from --stage1_dir.",
    )
    parser.add_argument("--dataset_root", type=str, default="")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--val_split", choices=("val", "none"), default="val")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_stage2_prior_guided_restormer")
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
    parser.add_argument("--batch_size", type=int, default=7)
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
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data_parallel", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_optional_path(value: str | None) -> str | None:
    if value is None:
        return None
    if value.lower() in {"none", "null", ""}:
        return None
    return value


def build_dataset(
    args: argparse.Namespace,
    split: str,
    max_samples: int | None,
) -> Stage2ManifestDataset | Stage2ResidualDataset | Subset:
    if args.manifest:
        dataset: Stage2ManifestDataset | Stage2ResidualDataset = Stage2ManifestDataset(
            manifest_path=args.manifest,
            stage1_dir=args.stage1_dir,
            dataset_root=args.dataset_root or None,
            split=split,
            augment=split == args.split,
        )
    else:
        is_train_split = split == args.split
        if is_train_split:
            root_dir = args.root_dir
            stage1_dir = args.stage1_dir
        else:
            root_dir = normalize_optional_path(args.val_root_dir)
            stage1_dir = normalize_optional_path(args.val_stage1_dir)
            if root_dir is None or stage1_dir is None:
                raise ValueError(
                    "PNG-root validation requires --val_root_dir and --val_stage1_dir, "
                    "or set --val_split none."
                )
        dataset = Stage2ResidualDataset(
            root_dir=root_dir,
            stage1_dir=stage1_dir,
            image_size=args.image_size if args.preprocess_mode == "resize256" else None,
            preprocess_mode=args.preprocess_mode,
            crop_size=args.crop_size,
            normalize_mode=args.normalize_mode,
            random_crop=is_train_split,
        )
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive or None.")
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return dataset


def build_model(args: argparse.Namespace, device: torch.device) -> Stage2PriorGuidedRestormerRefiner:
    model = Stage2PriorGuidedRestormerRefiner(
        dim=args.dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        expansion=args.ffn_expansion,
        residual_scale=args.residual_scale,
        angle_residual_scale=args.angle_residual_scale,
        min_gate=args.min_gate,
    )
    return model.to(device)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


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
            "model_type": "stage2_prior_guided_restormer_refiner",
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
    return int(checkpoint.get("epoch", 0)) + 1, float(checkpoint.get("best_val_loss", float("inf")))


def average_metrics(sums: dict[str, float], total: int) -> dict[str, float]:
    return {key: value / max(total, 1) for key, value in sums.items()}


def run_epoch(
    model: nn.Module,
    loss_fn: Stage2PriorGuidedRestormerLoss,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    sums = {key: 0.0 for key in METRIC_KEYS}
    total = 0
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        target = batch["polar_gt"].to(device, non_blocking=True)
        prior = batch["prior"].to(device, non_blocking=True)
        confidence = batch["confidence"].to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            pred = model(rgb, prior, confidence)
            losses = loss_fn(pred, target, prior, confidence)
            if is_train:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch_size = int(rgb.shape[0])
        total += batch_size
        for key in METRIC_KEYS:
            sums[key] += float(losses[key].detach()) * batch_size
    return average_metrics(sums, total)


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return " | ".join(f"{prefix}_{key}={metrics[key]:.6f}" for key in METRIC_KEYS)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log_path = save_dir / "train.log"

    train_dataset = build_dataset(args, args.split, args.max_train_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = None
    if args.val_split != "none":
        val_dataset = build_dataset(args, args.val_split, args.max_val_samples)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    model: nn.Module = build_model(args, device)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = Stage2PriorGuidedRestormerLoss().to(device)

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer, device)

    append_log(
        log_path,
        f"[config] train={len(train_dataset)} val={len(val_loader.dataset) if val_loader else 0} "
        f"batch_size={args.batch_size} data_parallel={isinstance(model, nn.DataParallel)} "
        f"dim={args.dim} num_blocks={args.num_blocks} stage1_dir={args.stage1_dir}",
    )
    for epoch in range(start_epoch, args.num_epochs + 1):
        train_metrics = run_epoch(model, loss_fn, train_loader, device, optimizer)
        message = f"epoch {epoch}/{args.num_epochs} | {format_metrics('train', train_metrics)}"
        score = train_metrics["loss"]
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(model, loss_fn, val_loader, device, optimizer=None)
            score = val_metrics["loss"]
            message += " | " + format_metrics("val", val_metrics)
        append_log(log_path, message)

        if score < best_val_loss:
            best_val_loss = score
            save_checkpoint(save_dir / "best_val.pth", epoch, model, optimizer, best_val_loss, args)
            append_log(log_path, f"[best_val] epoch={epoch} loss={best_val_loss:.6f}")
        save_checkpoint(save_dir / "last.pth", epoch, model, optimizer, best_val_loss, args)
        if args.save_freq > 0 and epoch % args.save_freq == 0:
            save_checkpoint(save_dir / f"epoch_{epoch:03d}.pth", epoch, model, optimizer, best_val_loss, args)
    append_log(log_path, f"Finished training. best_val_loss={best_val_loss:.6f}")


if __name__ == "__main__":
    main()
