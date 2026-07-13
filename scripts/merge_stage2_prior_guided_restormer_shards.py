"""Merge Version D sharded inference metrics without duplicating large outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Version D inference shards.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_count", type=int, required=True)
    return parser.parse_args()


def read_rows(output_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    rows: list[dict[str, str]] = []
    shard_summaries: list[dict[str, object]] = []
    for shard_dir in sorted(output_dir.glob("shard_*")):
        metrics_path = shard_dir / "metrics.csv"
        summary_path = shard_dir / "summary.json"
        if not metrics_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Incomplete shard: {shard_dir}")
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
        shard_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    rows.sort(key=lambda row: row["name"])
    return rows, shard_summaries


def summarize(rows: list[dict[str, str]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in rows[0]:
        if key == "name":
            continue
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[key] = float(finite.mean()) if finite.size else float("nan")
    return summary


def main() -> None:
    args = parse_args()
    rows, shard_summaries = read_rows(args.output_dir)
    unique_names = {row["name"] for row in rows}
    if len(rows) != args.expected_count or len(unique_names) != args.expected_count:
        raise RuntimeError(
            f"count mismatch: rows={len(rows)} unique={len(unique_names)} "
            f"expected={args.expected_count}"
        )
    summary = summarize(rows)
    fieldnames = list(rows[0])
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    first = shard_summaries[0]
    payload = {
        "model_type": first.get("model_type"),
        "architecture_version": first.get("architecture_version"),
        "samples": len(rows),
        "checkpoint": first.get("checkpoint"),
        "stage1_dir": first.get("stage1_dir"),
        "root_dir": first.get("root_dir"),
        "resize_output_to_gt": first.get("resize_output_to_gt"),
        "num_shards": len(shard_summaries),
        **summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
