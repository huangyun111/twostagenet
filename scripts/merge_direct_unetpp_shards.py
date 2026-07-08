"""Merge sharded Direct U-Net++ inference outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np


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
    parser = argparse.ArgumentParser(description="Merge Direct U-Net++ shard outputs.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_count", type=int, default=1474)
    return parser.parse_args()


def copy_pngs(output_dir: Path, subdir: str) -> int:
    merged_dir = output_dir / subdir
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for shard_dir in sorted(output_dir.glob("shard_*")):
        source_dir = shard_dir / subdir
        if not source_dir.exists():
            raise FileNotFoundError(source_dir)
        for src in sorted(source_dir.glob("*.png")):
            dst = merged_dir / src.name
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            count += 1
    return count


def read_metric_rows(output_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for shard_dir in sorted(output_dir.glob("shard_*")):
        metrics_path = shard_dir / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["name"] != "Overall":
                    rows.append(row)
    rows.sort(key=lambda row: row["name"])
    return rows


def summarize(rows: list[dict[str, str]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in METRIC_NAMES:
        values = np.array([float(row[key]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[key] = float(finite.mean()) if finite.size else float("nan")
    return summary


def write_metrics(output_dir: Path, rows: list[dict[str, str]], summary: dict[str, float]) -> None:
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", *METRIC_NAMES])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"name": "Overall", **summary})
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    rows = read_metric_rows(output_dir)
    pred_count = copy_pngs(output_dir, "pred_encoding_png")
    gt_count = copy_pngs(output_dir, "gt_encoding_png")
    if len(rows) != args.expected_count or pred_count != args.expected_count or gt_count != args.expected_count:
        raise RuntimeError(
            "count mismatch: "
            f"rows={len(rows)} pred={pred_count} gt={gt_count} expected={args.expected_count}"
        )
    summary = summarize(rows)
    write_metrics(output_dir, rows, summary)
    print(json.dumps({"rows": len(rows), "pred_png": pred_count, "gt_png": gt_count, **summary}, indent=2))


if __name__ == "__main__":
    main()
