"""Merge sharded online Stage1 + Version E evaluation metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_count", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    summaries: list[dict[str, object]] = []
    for shard_dir in sorted(args.output_dir.glob("shard_*")):
        metrics_path = shard_dir / "metrics.csv"
        summary_path = shard_dir / "summary.json"
        if not metrics_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"incomplete shard: {shard_dir}")
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    rows.sort(key=lambda row: row["name"])
    if len(rows) != args.expected_count or len({row["name"] for row in rows}) != args.expected_count:
        raise RuntimeError(
            f"count mismatch: rows={len(rows)} unique={len({r['name'] for r in rows})} "
            f"expected={args.expected_count}"
        )
    summary: dict[str, float] = {}
    for key in rows[0]:
        if key == "name":
            continue
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[key] = float(finite.mean()) if finite.size else float("nan")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    first = summaries[0]
    payload = {
        "model_type": first["model_type"],
        "architecture_version": first["architecture_version"],
        "samples": len(rows),
        "stage1_checkpoint": first["stage1_checkpoint"],
        "stage2_checkpoint": first["stage2_checkpoint"],
        "manifest": first["manifest"],
        "root_dir": first["root_dir"],
        "split": first["split"],
        "weak_factor": first["weak_factor"],
        "confidence_scale": first["confidence_scale"],
        "num_shards": len(summaries),
        **summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
