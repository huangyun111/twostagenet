"""Pretrain the Direct NAFNet baseline on final_new512 manifest splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from datasets.manifest_dataset import Stage1ManifestDataset
from losses.direct_polar_loss import DirectPolarLoss
from models.direct_nafnet_baseline import DirectNAFNetBaseline
from train_direct_unetpp_baseline import (
    append_log,
    format_metrics,
    resolve_device,
    run_epoch,
    set_seed,
    strip_module_prefix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain Direct NAFNet on manifest splits.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--val_split", choices=("val", "none"), default="val")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_direct_nafnet_13168")
    parser.add_argument("--crop_size", type=int, default=480)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--enc_blk_nums", type=int, nargs="+", default=(2, 2, 4, 8))
    parser.add_argument("--middle_blk_num", type=int, default=12)
    parser.add_argument("--dec_blk_nums", type=int, nargs="+", default=(2, 2, 2, 2))
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data_parallel", action="store_true")
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> DirectNAFNetBaseline:
    return DirectNAFNetBaseline(
        width=args.width,
        enc_blk_nums=args.enc_blk_nums,
        middle_blk_num=args.middle_blk_num,
        dec_blk_nums=args.dec_blk_nums,
    )


def build_dataset(
    args: argparse.Namespace,
    split: str,
    max_samples: int | None,
) -> Stage1ManifestDataset | Subset:
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
    model: DirectNAFNetBaseline,
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    return model.state_dict()


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: DirectNAFNetBaseline,
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
    return int(checkpoint.get("epoch", 0)) + 1, float(
        checkpoint.get("best_val_loss", float("inf"))
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    if not args.resume and (save_dir / "last.pth").exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing run: {save_dir}. Use --resume to continue it."
        )
    save_dir.mkdir(parents=True, exist_ok=True)
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
    val_dataset = None
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

    model = build_model(args).to(device)
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer, device)

    use_data_parallel = (
        args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1
    )
    if use_data_parallel:
        model = torch.nn.DataParallel(model)
    loss_fn = DirectPolarLoss().to(device)

    config = vars(args).copy()
    config.update(
        {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset) if val_dataset is not None else 0,
            "visible_gpus": torch.cuda.device_count() if device.type == "cuda" else 0,
            "effective_data_parallel": use_data_parallel,
        }
    )
    (save_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    append_log(
        log_path,
        f"[config] train={len(train_dataset)} val={config['val_samples']} "
        f"batch={args.batch_size} visible_gpus={config['visible_gpus']} "
        f"data_parallel={use_data_parallel} resume={args.resume or ''}",
    )

    for epoch in range(start_epoch, args.num_epochs + 1):
        train_metrics = run_epoch(model, loss_fn, train_loader, device, optimizer)
        score = train_metrics["loss"]
        message = f"epoch {epoch}/{args.num_epochs} | {format_metrics('train', train_metrics)}"
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
            save_checkpoint(
                save_dir / f"epoch_{epoch:03d}.pth",
                epoch,
                model,
                optimizer,
                best_val_loss,
                args,
            )

    append_log(log_path, f"Finished training. best_val_loss={best_val_loss:.6f}")


if __name__ == "__main__":
    main()
