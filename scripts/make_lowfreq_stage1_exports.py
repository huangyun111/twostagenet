"""Create low-frequency Stage1 exports for Weak/Coarse Stage1 experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample-then-upsample Stage1 prior/confidence exports."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--confidence_scale", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def lowpass_chw(array: np.ndarray, factor: int) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected CHW array, got {array.shape}")
    if factor <= 1:
        return array.astype(np.float32, copy=True)
    tensor = torch.from_numpy(array.astype(np.float32)).unsqueeze(0)
    height, width = tensor.shape[-2:]
    small_h = max(height // factor, 1)
    small_w = max(width // factor, 1)
    small = F.interpolate(tensor, size=(small_h, small_w), mode="area")
    coarse = F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)
    return coarse.squeeze(0).numpy().astype(np.float32)


def clamp_and_normalize_prior(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if array.shape[0] != 3:
        raise ValueError(f"Expected prior shape [3,H,W], got {array.shape}")
    dolp = np.clip(array[0:1], 0.0, 1.0)
    cos_sin = np.clip(array[1:3], -1.0, 1.0)
    norm = np.sqrt(np.sum(cos_sin * cos_sin, axis=0, keepdims=True) + eps)
    return np.concatenate([dolp, cos_sin / norm], axis=0).astype(np.float32)


def process_export_pair(
    prior_path: Path,
    confidence_path: Path,
    output_prior_dir: Path,
    output_confidence_dir: Path,
    factor: int,
    confidence_scale: float,
) -> None:
    prior = np.load(prior_path).astype(np.float32)
    confidence = np.load(confidence_path).astype(np.float32)
    coarse_prior = clamp_and_normalize_prior(lowpass_chw(prior, factor))
    coarse_confidence = np.clip(lowpass_chw(confidence, factor) * confidence_scale, 0.0, 1.0)

    output_prior_dir.mkdir(parents=True, exist_ok=True)
    output_confidence_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_prior_dir / prior_path.name, coarse_prior.astype(np.float32))
    np.save(output_confidence_dir / confidence_path.name, coarse_confidence.astype(np.float32))


def process_task(task: tuple[Path, Path, Path, Path, int, float]) -> str:
    torch.set_num_threads(1)
    prior_path, confidence_path, output_prior_dir, output_confidence_dir, factor, confidence_scale = task
    process_export_pair(
        prior_path=prior_path,
        confidence_path=confidence_path,
        output_prior_dir=output_prior_dir,
        output_confidence_dir=output_confidence_dir,
        factor=factor,
        confidence_scale=confidence_scale,
    )
    return prior_path.name


def main() -> None:
    args = parse_args()
    input_prior_dir = args.input_dir / "prior_npy"
    input_confidence_dir = args.input_dir / "confidence_npy"
    output_prior_dir = args.output_dir / "prior_npy"
    output_confidence_dir = args.output_dir / "confidence_npy"
    if not input_prior_dir.is_dir():
        raise FileNotFoundError(input_prior_dir)
    if not input_confidence_dir.is_dir():
        raise FileNotFoundError(input_confidence_dir)
    tasks: list[tuple[Path, Path, Path, Path, int, float]] = []
    skipped = 0
    for prior_path in sorted(input_prior_dir.glob("*.npy")):
        confidence_path = input_confidence_dir / prior_path.name
        if not confidence_path.is_file():
            raise FileNotFoundError(confidence_path)
        output_prior_path = output_prior_dir / prior_path.name
        output_confidence_path = output_confidence_dir / confidence_path.name
        if args.skip_existing and output_prior_path.is_file() and output_confidence_path.is_file():
            skipped += 1
            continue
        tasks.append(
            (
                prior_path,
                confidence_path,
                output_prior_dir,
                output_confidence_dir,
                args.factor,
                args.confidence_scale,
            )
        )
    if args.num_workers <= 0:
        raise ValueError("num_workers must be positive.")
    if args.num_workers == 1:
        for task in tasks:
            process_task(task)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for _ in executor.map(process_task, tasks, chunksize=1):
                pass
    count = skipped + len(tasks)
    print(
        f"wrote {count} coarse Stage1 exports to {args.output_dir} "
        f"(factor={args.factor}, confidence_scale={args.confidence_scale}, "
        f"workers={args.num_workers}, skipped={skipped})",
        flush=True,
    )


if __name__ == "__main__":
    main()
