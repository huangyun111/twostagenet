"""Train the Direct Restormer baseline on final_new512 manifest splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from losses.direct_polar_loss import DirectPolarLoss
from models.direct_restormer_baseline import DirectRestormerBaseline
from train_direct_unetpp_baseline import (
    append_log,
    build_dataset,
    format_metrics,
    resolve_device,
    run_epoch,
    set_seed,
    strip_module_prefix,
)


def parse_int_tuple(value: str, expected_len: int) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(items) != expected_len:
        raise argparse.ArgumentTypeError(f"expected {expected_len} comma-separated integers")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Direct Restormer baseline.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--val_split", choices=("val", "none"), default="val")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_direct_restormer_13168")
    parser.add_argument("--crop_size", type=int, default=480)
    parser.add_argument("--batch_size", type=int, default=14)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--num_blocks", type=lambda value: parse_int_tuple(value, 4), default=(2, 2, 2, 4))
    parser.add_argument("--num_heads", type=lambda value: parse_int_tuple(value, 4), default=(1, 2, 4, 8))
    parser.add_argument("--ffn_expansion", type=float, default=2.66)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data_parallel", action="store_true")
    return parser.parse_args()


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
    train_loader = torch.utils.data.DataLoader(
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
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    model: nn.Module = DirectRestormerBaseline(
        dim=args.dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        expansion=args.ffn_expansion,
    ).to(device)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = DirectPolarLoss().to(device)

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer, device)

    append_log(
        log_path,
        f"[config] samples train={len(train_dataset)} val={len(val_loader.dataset) if val_loader else 0} "
        f"batch_size={args.batch_size} data_parallel={isinstance(model, nn.DataParallel)} "
        f"cuda_devices={torch.cuda.device_count() if device.type == 'cuda' else 0}",
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
