"""Infer final_new512-trained Direct Restormer on a HAMMER folder split."""

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

from datasets.stage1_prior_dataset import Stage1PriorDataset
from infer_stage2_angular_refiner import (
    compute_metric_dict,
    resize_polar_to_gt_if_requested,
    save_polar_encoding_png,
)
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
    parser = argparse.ArgumentParser(description="Infer Direct Restormer on HAMMER.")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./direct_restormer_13168_to_hammer_zeroshot_outputs",
    )
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
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument(
        "--num_blocks", type=lambda value: parse_int_tuple(value, 4), default=(2, 2, 2, 4)
    )
    parser.add_argument(
        "--num_heads", type=lambda value: parse_int_tuple(value, 4), default=(1, 2, 4, 8)
    )
    parser.add_argument("--ffn_expansion", type=float, default=2.66)
    parser.add_argument("--resize_output_to_gt", action="store_true")
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
    resized_count = 0
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            pred = model(rgb).cpu()
            target = batch["polar"].cpu()
            names = batch["name"]
            for offset, name in enumerate(names):
                sample_pred = pred[offset]
                sample_gt = target[offset]
                pred_eval, was_resized = resize_polar_to_gt_if_requested(
                    sample_pred, sample_gt, args.resize_output_to_gt
                )
                if was_resized:
                    resized_count += 1
                save_polar_encoding_png(pred_eval, pred_dir / f"{name}.png")
                save_polar_encoding_png(sample_gt, gt_dir / f"{name}.png")
                row: dict[str, float | str] = {"name": name}
                row.update(compute_metric_dict(pred_eval, sample_gt))
                rows.append(row)
            processed += len(names)
            print(f"processed {processed}/{len(dataset)}", flush=True)

    summary = summarize(rows)
    payload = {
        "model_type": "direct_restormer_baseline",
        "samples": len(rows),
        "checkpoint": args.checkpoint,
        "root_dir": args.root_dir,
        "resize_output_to_gt": args.resize_output_to_gt,
        "output_resized_to_gt_count": resized_count,
        **summary,
    }
    write_csv(output_dir / "metrics.csv", rows, summary)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
