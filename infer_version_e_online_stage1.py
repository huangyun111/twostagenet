"""Evaluate Version E with its frozen Stage1 checkpoint, without prior exports."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.manifest_dataset import Stage1ManifestDataset
from datasets.stage1_prior_dataset import Stage1PriorDataset
from infer_stage2_angular_refiner import (
    METRIC_NAMES,
    compute_metric_dict,
    prefix_metrics,
    save_paper_aop_dop_vis,
    save_polar_encoding_png,
)
from models.stage2_asymmetric_restormer_refiner import Stage2AsymmetricRestormerRefiner
from train_direct_unetpp_baseline import resolve_device, strip_module_prefix
from train_stage2_prior_guided_restormer import parse_int_tuple
from train_stage2_prior_guided_restormer_online_stage1 import build_stage1, make_online_prior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Version E with online Stage1.")
    parser.add_argument("--root_dir", type=str, default="")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--dataset_root", type=str, default="")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--stage1_checkpoint", type=str, required=True)
    parser.add_argument("--stage2_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--preprocess_mode", choices=("resize256", "official_infer"), default="official_infer"
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--divisible_by", type=int, default=32)
    parser.add_argument(
        "--normalize_mode", choices=("fixed255", "image_max"), default="image_max"
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--paper_vis_every", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--weak_factor", type=int, default=4)
    parser.add_argument("--confidence_scale", type=float, default=0.5)
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
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    return parser.parse_args()


def build_dataset(args: argparse.Namespace):
    if args.manifest:
        dataset = Stage1ManifestDataset(
            args.manifest,
            dataset_root=args.dataset_root or None,
            split=args.split,
            crop_size=0,
            augment=False,
        )
    else:
        if not args.root_dir:
            raise ValueError("--root_dir is required when --manifest is not set")
        dataset = Stage1PriorDataset(
            root_dir=args.root_dir,
            image_size=args.image_size,
            preprocess_mode=args.preprocess_mode,
            normalize_mode=args.normalize_mode,
            divisible_by=args.divisible_by,
            augment=False,
        )
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max_samples must be positive")
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    if args.num_shards > 1:
        dataset = Subset(dataset, range(args.shard_index, len(dataset), args.num_shards))
    return dataset


def load_stage2(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(args.stage2_checkpoint, map_location="cpu")
    architecture = str(checkpoint.get("architecture_version", "")).upper()
    if architecture and architecture != "E":
        raise ValueError(f"Expected Version E checkpoint, found architecture={architecture}")
    model = Stage2AsymmetricRestormerRefiner(
        dim=args.dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        expansion=args.ffn_expansion,
        dolp_residual_scale=args.dolp_residual_scale,
        angle_residual_scale=args.angle_residual_scale,
        min_angle_gate=args.min_angle_gate,
    )
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(strip_module_prefix(state_dict))
    return model.to(device).eval()


def summarize(rows: list[dict[str, float | str]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in rows[0]:
        if key == "name":
            continue
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[key] = float(finite.mean()) if finite.size else float("nan")
    return result


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "pred_encoding_png"
    stage1_dir = output_dir / "stage1_encoding_png"
    gt_dir = output_dir / "gt_encoding_png"
    paper_dir = output_dir / "paper_vis"
    for directory in (pred_dir, stage1_dir, gt_dir, paper_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    stage1 = build_stage1(args.stage1_checkpoint, device)
    stage2 = load_stage2(args, device)
    rows: list[dict[str, float | str]] = []
    processed = 0
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            target = batch["polar"].cpu()
            prior, confidence = make_online_prior(
                stage1, rgb, args.weak_factor, args.confidence_scale
            )
            pred = stage2(rgb, prior, confidence)["refined"].cpu()
            prior = prior.cpu()
            for offset, name in enumerate(batch["name"]):
                sample_gt = target[offset]
                sample_prior = prior[offset]
                sample_pred = pred[offset]
                if sample_pred.shape[-2:] != sample_gt.shape[-2:]:
                    raise ValueError(
                        f"prediction/GT shape mismatch for {name}: "
                        f"{tuple(sample_pred.shape)} vs {tuple(sample_gt.shape)}"
                    )
                save_polar_encoding_png(sample_pred, pred_dir / f"{name}.png")
                save_polar_encoding_png(sample_prior, stage1_dir / f"{name}.png")
                save_polar_encoding_png(sample_gt, gt_dir / f"{name}.png")
                row: dict[str, float | str] = {"name": name}
                row.update(prefix_metrics("stage1", compute_metric_dict(sample_prior, sample_gt)))
                row.update(prefix_metrics("stage2", compute_metric_dict(sample_pred, sample_gt)))
                rows.append(row)
                global_index = args.shard_index + (processed + offset) * args.num_shards
                if args.paper_vis_every > 0 and global_index % args.paper_vis_every == 0:
                    save_paper_aop_dop_vis(
                        rgb_or_none=None,
                        gt_polar=sample_gt,
                        pred_polar=sample_pred,
                        save_path=paper_dir / f"{name}_aop_dop_compare.png",
                    )
            processed += len(batch["name"])
            print(f"processed {processed}/{len(dataset)}", flush=True)

    if not rows:
        raise RuntimeError("no samples were evaluated")
    fieldnames = ["name"] + [
        f"{prefix}_{metric}" for prefix in ("stage1", "stage2") for metric in METRIC_NAMES
    ]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "model_type": "stage2_asymmetric_restormer_refiner_ve_online_stage1",
        "architecture_version": "E",
        "samples": len(rows),
        "stage1_checkpoint": args.stage1_checkpoint,
        "stage2_checkpoint": args.stage2_checkpoint,
        "manifest": args.manifest,
        "root_dir": args.root_dir,
        "split": args.split,
        "weak_factor": args.weak_factor,
        "confidence_scale": args.confidence_scale,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        **summarize(rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
