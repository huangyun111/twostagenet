"""Infer and evaluate Direct NAFNet on a HAMMER folder split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.stage1_prior_dataset import Stage1PriorDataset
from infer_stage2_angular_refiner import (
    compute_metric_dict,
    resize_polar_to_gt_if_requested,
    save_polar_encoding_png,
)
from models.direct_nafnet_baseline import DirectNAFNetBaseline
from train_direct_unetpp_baseline import resolve_device, strip_module_prefix


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
    parser = argparse.ArgumentParser(description="Infer Direct NAFNet on HAMMER.")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--preprocess_mode",
        choices=("resize256", "official_infer"),
        default="official_infer",
    )
    parser.add_argument(
        "--normalize_mode", choices=("fixed255", "image_max"), default="image_max"
    )
    parser.add_argument("--divisible_by", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--enc_blk_nums", type=int, nargs="+", default=(2, 2, 4, 8))
    parser.add_argument("--middle_blk_num", type=int, default=12)
    parser.add_argument("--dec_blk_nums", type=int, nargs="+", default=(2, 2, 2, 2))
    parser.add_argument("--resize_output_to_gt", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> DirectNAFNetBaseline:
    model = DirectNAFNetBaseline(
        width=args.width,
        enc_blk_nums=args.enc_blk_nums,
        middle_blk_num=args.middle_blk_num,
        dec_blk_nums=args.dec_blk_nums,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(strip_module_prefix(state_dict))
    model.to(device)
    model.eval()
    return model


def summarize(rows: list[dict[str, float | str]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in METRIC_NAMES:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[key] = float(finite.mean()) if finite.size else float("nan")
    return summary


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", *METRIC_NAMES])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "pred_encoding_png"
    gt_dir = output_dir / "gt_encoding_png"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    dataset = Stage1PriorDataset(
        root_dir=args.root_dir,
        preprocess_mode=args.preprocess_mode,
        crop_size=512,
        normalize_mode=args.normalize_mode,
        divisible_by=args.divisible_by,
        augment=False,
        return_path=False,
    )
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    if args.num_shards > 1:
        dataset = Subset(
            dataset,
            list(range(args.shard_index, len(dataset), args.num_shards)),
        )
    print(
        f"running shard {args.shard_index + 1}/{args.num_shards}: {len(dataset)} samples",
        flush=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = load_model(args, device)
    rows: list[dict[str, float | str]] = []
    resized_count = 0
    with torch.no_grad():
        for batch in loader:
            predictions = model(batch["rgb"].to(device, non_blocking=True)).cpu()
            targets = batch["polar"].cpu()
            for offset, name in enumerate(batch["name"]):
                prediction, was_resized = resize_polar_to_gt_if_requested(
                    predictions[offset], targets[offset], args.resize_output_to_gt
                )
                resized_count += int(was_resized)
                save_polar_encoding_png(prediction, pred_dir / f"{name}.png")
                save_polar_encoding_png(targets[offset], gt_dir / f"{name}.png")
                row: dict[str, float | str] = {"name": name}
                row.update(compute_metric_dict(prediction, targets[offset]))
                rows.append(row)
            print(f"processed {len(rows)}/{len(dataset)}", flush=True)

    payload = {
        "model_type": "direct_nafnet_hammer_end_to_end",
        "samples": len(rows),
        "checkpoint": args.checkpoint,
        "root_dir": args.root_dir,
        "resize_output_to_gt": args.resize_output_to_gt,
        "output_resized_to_gt_count": resized_count,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        **summarize(rows),
    }
    write_csv(output_dir / "metrics.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
