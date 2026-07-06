"""Infer and evaluate the Direct Restormer baseline on a manifest split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.manifest_dataset import Stage1ManifestDataset
from infer_stage2_angular_refiner import compute_metric_dict, save_polar_encoding_png
from models.direct_restormer_baseline import DirectRestormerBaseline
from train_direct_restormer_baseline import parse_int_tuple
from train_direct_unetpp_baseline import strip_module_prefix


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
    parser = argparse.ArgumentParser(description="Infer Direct Restormer baseline.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./direct_restormer_13168_test_outputs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--num_blocks", type=lambda value: parse_int_tuple(value, 4), default=(2, 2, 2, 4))
    parser.add_argument("--num_heads", type=lambda value: parse_int_tuple(value, 4), default=(1, 2, 4, 8))
    parser.add_argument("--ffn_expansion", type=float, default=2.66)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_model(args: argparse.Namespace, device: torch.device) -> DirectRestormerBaseline:
    model = DirectRestormerBaseline(
        dim=args.dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        expansion=args.ffn_expansion,
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
        values = np.array([float(row[key]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[key] = float(finite.mean()) if finite.size else float("nan")
    return summary


def write_csv(path: Path, rows: list[dict[str, float | str]], summary: dict[str, float]) -> None:
    fieldnames = ["name", *METRIC_NAMES]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"name": "Overall", **summary})


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "pred_encoding_png"
    gt_dir = output_dir / "gt_encoding_png"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    dataset = Stage1ManifestDataset(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        split=args.split,
        image_size=None,
        crop_size=0,
        random_crop=False,
        augment=False,
    )
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")
    if args.num_shards > 1:
        shard_indices = list(range(args.shard_index, len(dataset), args.num_shards))
        dataset = Subset(dataset, shard_indices)
        print(f"running shard {args.shard_index + 1}/{args.num_shards}: {len(dataset)} samples", flush=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = load_model(args, device)
    rows: list[dict[str, float | str]] = []
    processed = 0
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            pred = model(rgb).cpu()
            target = batch["polar"].cpu()
            names = batch["name"]
            for offset, name in enumerate(names):
                sample_pred = pred[offset]
                sample_gt = target[offset]
                save_polar_encoding_png(sample_pred, pred_dir / f"{name}.png")
                save_polar_encoding_png(sample_gt, gt_dir / f"{name}.png")
                row: dict[str, float | str] = {"name": name}
                row.update(compute_metric_dict(sample_pred, sample_gt))
                rows.append(row)
            processed += len(names)
            print(f"processed {processed}/{len(dataset)}", flush=True)

    summary = summarize(rows)
    write_csv(output_dir / "metrics.csv", rows, summary)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
