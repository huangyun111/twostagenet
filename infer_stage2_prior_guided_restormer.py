"""Infer Version C prior-guided Restormer Stage 2 refiner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.manifest_dataset import Stage2ManifestDataset  # noqa: E402
from datasets.stage2_residual_dataset import Stage2ResidualDataset  # noqa: E402
from infer_stage2_angular_refiner import (  # noqa: E402
    METRIC_NAMES,
    compute_metric_dict,
    prefix_metrics,
    resize_polar_to_gt_if_requested,
    save_paper_aop_dop_vis,
    save_polar_encoding_png,
)
from models.stage2_prior_guided_restormer_refiner import (  # noqa: E402
    Stage2PriorGuidedRestormerRefiner,
)
from train_direct_unetpp_baseline import resolve_device, strip_module_prefix  # noqa: E402
from train_stage2_prior_guided_restormer import parse_int_tuple  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer Stage2 Prior-Guided Angular Restormer Refiner."
    )
    parser.add_argument("--root_dir", type=str, default=str(Path.home() / "Documents"))
    parser.add_argument("--stage1_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--dataset_root", type=str, default="")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./stage2_prior_guided_restormer_outputs")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument(
        "--preprocess_mode",
        choices=("resize256", "official_infer"),
        default="official_infer",
    )
    parser.add_argument("--divisible_by", type=int, default=32)
    parser.add_argument(
        "--normalize_mode",
        choices=("fixed255", "image_max"),
        default="image_max",
    )
    parser.add_argument("--resize_output_to_gt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--paper_vis_every", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--num_blocks", type=lambda value: parse_int_tuple(value, 3), default=(2, 2, 4))
    parser.add_argument("--num_heads", type=lambda value: parse_int_tuple(value, 3), default=(1, 2, 4))
    parser.add_argument("--ffn_expansion", type=float, default=2.66)
    parser.add_argument("--residual_scale", type=float, default=0.5)
    parser.add_argument("--angle_residual_scale", type=float, default=math.pi)
    parser.add_argument("--min_gate", type=float, default=0.05)
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> Stage2ManifestDataset | Stage2ResidualDataset | Subset:
    if args.manifest:
        dataset: Stage2ManifestDataset | Stage2ResidualDataset | Subset = Stage2ManifestDataset(
            manifest_path=args.manifest,
            stage1_dir=args.stage1_dir,
            dataset_root=args.dataset_root or None,
            split=args.split,
            augment=False,
        )
    else:
        dataset = Stage2ResidualDataset(
            root_dir=args.root_dir,
            stage1_dir=args.stage1_dir,
            image_size=args.image_size,
            preprocess_mode=args.preprocess_mode,
            normalize_mode=args.normalize_mode,
            divisible_by=args.divisible_by,
        )
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("max_samples must be positive or None.")
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    return dataset


def load_model(args: argparse.Namespace, device: torch.device) -> Stage2PriorGuidedRestormerRefiner:
    model = Stage2PriorGuidedRestormerRefiner(
        dim=args.dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        expansion=args.ffn_expansion,
        residual_scale=args.residual_scale,
        angle_residual_scale=args.angle_residual_scale,
        min_gate=args.min_gate,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(strip_module_prefix(state_dict))
    model.to(device)
    model.eval()
    return model


def summarize_rows(rows: list[dict[str, float | str]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    if not rows:
        return summary
    metric_keys = [key for key in rows[0] if key != "name"]
    for key in metric_keys:
        values = np.array([float(row[key]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[key] = float(finite.mean()) if finite.size else float("nan")
    return summary


def write_metrics_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = ["name"]
    for prefix in ("stage1", "stage2"):
        fieldnames.extend(f"{prefix}_{name}" for name in METRIC_NAMES)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    refined_dir = output_dir / "refined_npy"
    pred_encoding_dir = output_dir / "pred_encoding_png"
    stage1_encoding_dir = output_dir / "stage1_encoding_png"
    gt_encoding_dir = output_dir / "gt_encoding_png"
    paper_vis_dir = output_dir / "paper_vis"
    for path in (refined_dir, pred_encoding_dir, stage1_encoding_dir, gt_encoding_dir, paper_vis_dir):
        path.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
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
            prior = batch["prior"].to(device, non_blocking=True)
            confidence = batch["confidence"].to(device, non_blocking=True)
            pred = model(rgb, prior, confidence)
            pred_cpu = {key: value.detach().cpu() for key, value in pred.items()}
            prior_cpu = batch["prior"].cpu()
            target_cpu = batch["polar_gt"].cpu()
            names = batch["name"]

            for offset, name in enumerate(names):
                sample_index = processed + offset
                sample_prior = prior_cpu[offset]
                sample_gt = target_cpu[offset]
                sample_refined = pred_cpu["refined"][offset]
                prior_eval, prior_resized = resize_polar_to_gt_if_requested(
                    sample_prior,
                    sample_gt,
                    args.resize_output_to_gt,
                )
                refined_eval, refined_resized = resize_polar_to_gt_if_requested(
                    sample_refined,
                    sample_gt,
                    args.resize_output_to_gt,
                )
                if prior_resized or refined_resized:
                    resized_count += 1

                np.save(refined_dir / f"{name}.npy", refined_eval.numpy().astype(np.float32))
                save_polar_encoding_png(refined_eval, pred_encoding_dir / f"{name}.png")
                save_polar_encoding_png(prior_eval, stage1_encoding_dir / f"{name}.png")
                save_polar_encoding_png(sample_gt, gt_encoding_dir / f"{name}.png")

                row: dict[str, float | str] = {"name": name}
                row.update(prefix_metrics("stage1", compute_metric_dict(prior_eval, sample_gt)))
                row.update(prefix_metrics("stage2", compute_metric_dict(refined_eval, sample_gt)))
                rows.append(row)

                if args.paper_vis_every > 0 and sample_index % args.paper_vis_every == 0:
                    save_paper_aop_dop_vis(
                        rgb_or_none=None,
                        gt_polar=sample_gt,
                        pred_polar=refined_eval,
                        save_path=paper_vis_dir / f"{name}_aop_dop_compare.png",
                    )
            processed += len(names)
            print(f"processed {processed}/{len(dataset)}", flush=True)

    write_metrics_csv(output_dir / "metrics.csv", rows)
    summary = summarize_rows(rows)
    payload = {
        "model_type": "stage2_prior_guided_restormer_refiner",
        "samples": len(rows),
        "checkpoint": args.checkpoint,
        "stage1_dir": args.stage1_dir,
        "root_dir": args.root_dir,
        "manifest": args.manifest,
        "resize_output_to_gt": args.resize_output_to_gt,
        "output_resized_to_gt_count": resized_count,
        **summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
