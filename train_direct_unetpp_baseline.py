"""Train the Direct U-Net++ baseline on final_new512 manifest splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.manifest_dataset import Stage1ManifestDataset
from losses.direct_polar_loss import DirectPolarLoss
from models.direct_unetpp_baseline import DirectUnetPlusPlusBaseline


METRIC_KEYS = (
    "loss",
    "loss_dolp",
    "loss_vector",
    "loss_aolp",
    "loss_lowfreq",
    "loss_edge",
    "mean_aolp_reliability",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Direct U-Net++ baseline.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--val_split", choices=("val", "none"), default="val")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_direct_unetpp_13168")
    parser.add_argument("--preprocess_mode", choices=("official_train",), default="official_train")
    parser.add_argument("--crop_size", type=int, default=480)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--encoder_name", type=str, default="resnet34")
    parser.add_argument("--encoder_weights", type=str, default=None)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def normalize_encoder_weights(value: str | None) -> str | None:
    if value is None or value.lower() in {"", "none", "null"}:
        return None
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def build_dataset(args: argparse.Namespace, split: str, max_samples: int | None) -> Stage1ManifestDataset | Subset:
    dataset = Stage1ManifestDataset(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        split=split,
        image_size=None,
        crop_size=args.crop_size,
        random_crop=split == args.split,
        augment=split == args.split,
    )
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return dataset


def build_optimizer(
    model: DirectUnetPlusPlusBaseline,
    lr: float,
    encoder_lr: float,
) -> torch.optim.Optimizer:
    encoder_ids = {id(param) for param in model.net.encoder.parameters()}
    encoder_params = []
    other_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in encoder_ids:
            encoder_params.append(param)
        else:
            other_params.append(param)
    groups: list[dict[str, Any]] = [{"params": encoder_params, "lr": encoder_lr}]
    if other_params:
        groups.append({"params": other_params, "lr": lr})
    return torch.optim.AdamW(groups)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: DirectUnetPlusPlusBaseline,
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
        },
        path,
    )


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def load_checkpoint(
    path: str,
    model: DirectUnetPlusPlusBaseline,
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


def average_metrics(sums: dict[str, float], total: int) -> dict[str, float]:
    return {key: value / max(total, 1) for key, value in sums.items()}


def run_epoch(
    model: DirectUnetPlusPlusBaseline,
    loss_fn: DirectPolarLoss,
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
        target = batch["polar"].to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            pred = model(rgb)
            losses = loss_fn(pred, target)
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


def append_log(path: Path, message: str) -> None:
    print(message, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


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

    model = DirectUnetPlusPlusBaseline(
        encoder_name=args.encoder_name,
        encoder_weights=normalize_encoder_weights(args.encoder_weights),
    ).to(device)
    optimizer = build_optimizer(model, lr=args.lr, encoder_lr=args.encoder_lr)
    loss_fn = DirectPolarLoss().to(device)

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer, device)

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
