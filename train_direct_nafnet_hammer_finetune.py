"""Train Direct NAFNet on HAMMER train/val splits.

Fresh end-to-end HAMMER training is the default. ``--init_checkpoint`` remains
available for explicitly named transfer experiments, while ``--resume`` is
only for continuing an interrupted run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from datasets.stage1_prior_dataset import Stage1PriorDataset
from losses.direct_polar_loss import DirectPolarLoss
from models.direct_nafnet_baseline import DirectNAFNetBaseline
from train_direct_nafnet_baseline import (
    build_optimizer,
    load_checkpoint,
    save_checkpoint,
)
from train_direct_unetpp_baseline import (
    append_log,
    format_metrics,
    resolve_device,
    run_epoch,
    set_seed,
    strip_module_prefix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Direct NAFNet end-to-end on HAMMER train/val splits."
    )
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--val_root_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--init_checkpoint",
        type=str,
        help="Load weights only and reset HAMMER epoch, optimizer, and best-val state.",
    )
    source.add_argument("--resume", type=str, help="Resume an interrupted HAMMER run.")
    parser.add_argument("--preprocess_mode", choices=("official_train",), default="official_train")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--normalize_mode", choices=("fixed255", "image_max"), default="image_max")
    parser.add_argument("--divisible_by", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--enc_blk_nums", type=int, nargs="+", default=(2, 2, 4, 8))
    parser.add_argument("--middle_blk_num", type=int, default=12)
    parser.add_argument("--dec_blk_nums", type=int, nargs="+", default=(2, 2, 2, 2))
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--save_freq", type=int, default=20)
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
    root_dir: str,
    args: argparse.Namespace,
    *,
    train: bool,
    max_samples: int | None,
) -> Stage1PriorDataset | Subset:
    dataset = Stage1PriorDataset(
        root_dir=root_dir,
        preprocess_mode=args.preprocess_mode,
        crop_size=args.crop_size,
        normalize_mode=args.normalize_mode,
        divisible_by=args.divisible_by,
        augment=train,
    )
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return dataset


def load_weights_only(
    path: str,
    model: DirectNAFNetBaseline,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(strip_module_prefix(state_dict))
    if isinstance(checkpoint, dict):
        return {
            "source_epoch": checkpoint.get("epoch"),
            "source_best_val_loss": checkpoint.get("best_val_loss"),
            "source_args": checkpoint.get("args"),
        }
    return {"source_epoch": None, "source_best_val_loss": None, "source_args": None}


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

    train_dataset = build_dataset(
        args.root_dir,
        args,
        train=True,
        max_samples=args.max_train_samples,
    )
    val_dataset = build_dataset(
        args.val_root_dir,
        args,
        train=False,
        max_samples=args.max_val_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args).to(device)
    source_metadata: dict[str, object] | None = None
    if args.init_checkpoint:
        source_metadata = load_weights_only(args.init_checkpoint, model, device)
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
            "val_samples": len(val_dataset),
            "visible_gpus": torch.cuda.device_count() if device.type == "cuda" else 0,
            "effective_data_parallel": use_data_parallel,
            "source_metadata": source_metadata,
        }
    )
    (save_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    append_log(
        log_path,
        f"[config] train={len(train_dataset)} val={len(val_dataset)} batch={args.batch_size} "
        f"visible_gpus={config['visible_gpus']} data_parallel={use_data_parallel} "
        f"init_checkpoint={args.init_checkpoint or ''} resume={args.resume or ''}",
    )

    for epoch in range(start_epoch, args.num_epochs + 1):
        train_metrics = run_epoch(model, loss_fn, train_loader, device, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, loss_fn, val_loader, device, optimizer=None)
        score = val_metrics["loss"]
        append_log(
            log_path,
            f"epoch {epoch}/{args.num_epochs} | {format_metrics('train', train_metrics)} | "
            f"{format_metrics('val', val_metrics)}",
        )
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
