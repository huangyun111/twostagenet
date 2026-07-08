"""Export low-frequency Stage1 priors directly, without storing full priors first."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.official_preprocess import crop_to_divisible, read_rgb_image  # noqa: E402
from models.polar_prior_net import PolarPriorNet  # noqa: E402


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class S0OnlyDataset(Dataset):
    def __init__(self, root_dir: str | Path, normalize_mode: str, divisible_by: int) -> None:
        root = Path(root_dir)
        s0_dir = root / "S0"
        if not s0_dir.is_dir():
            s0_dir = root / "s0"
        if not s0_dir.is_dir():
            raise FileNotFoundError(f"S0 directory not found under {root}")
        self.paths = sorted(
            path for path in s0_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No S0 images found under {s0_dir}")
        self.normalize_mode = normalize_mode
        self.divisible_by = divisible_by

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[index]
        rgb = read_rgb_image(path, normalize_mode=self.normalize_mode)
        rgb, _ = crop_to_divisible(rgb, self.divisible_by)
        return {"rgb": rgb.contiguous(), "name": path.stem}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export direct low-frequency Stage1 prior/confidence.")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--confidence_scale", type=float, default=0.5)
    parser.add_argument("--normalize_mode", choices=("fixed255", "image_max"), default="image_max")
    parser.add_argument("--divisible_by", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def normalize_encoder_weights(value: str | None) -> str | None:
    if value is None:
        return None
    if value.lower() in {"none", "null", ""}:
        return None
    return value


def load_model(checkpoint_path: str, device: torch.device) -> PolarPriorNet:
    model = PolarPriorNet(encoder_weights=None)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def lowpass(tensor: torch.Tensor, factor: int) -> torch.Tensor:
    if factor <= 1:
        return tensor
    height, width = tensor.shape[-2:]
    small = F.interpolate(tensor, size=(max(height // factor, 1), max(width // factor, 1)), mode="area")
    return F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)


def clamp_prior(prior: torch.Tensor) -> torch.Tensor:
    dolp = prior[:, 0:1].clamp(0.0, 1.0)
    cos_sin = prior[:, 1:3].clamp(-1.0, 1.0)
    norm = torch.sqrt((cos_sin * cos_sin).sum(dim=1, keepdim=True).clamp_min(1e-6))
    return torch.cat([dolp, cos_sin / norm], dim=1)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    prior_dir = out_dir / "prior_npy"
    confidence_dir = out_dir / "confidence_npy"
    prior_dir.mkdir(parents=True, exist_ok=True)
    confidence_dir.mkdir(parents=True, exist_ok=True)

    dataset = S0OnlyDataset(args.root_dir, args.normalize_mode, args.divisible_by)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = load_model(args.checkpoint, device)
    out_dtype = np.float16 if args.dtype == "float16" else np.float32

    count = 0
    with torch.inference_mode():
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            names = batch["name"]
            use_amp = device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(rgb)
                prior = clamp_prior(lowpass(output["polar_prior"], args.factor))
                confidence = (lowpass(output["confidence"], args.factor) * args.confidence_scale).clamp(0.0, 1.0)
            prior_np = prior.detach().cpu().numpy()
            confidence_np = confidence.detach().cpu().numpy()
            for offset, name in enumerate(names):
                np.save(prior_dir / f"{name}.npy", prior_np[offset].astype(out_dtype, copy=False))
                np.save(confidence_dir / f"{name}.npy", confidence_np[offset].astype(out_dtype, copy=False))
                count += 1
            print(f"exported {count}/{len(dataset)}", flush=True)

    summary = {
        "root_dir": args.root_dir,
        "checkpoint": args.checkpoint,
        "samples": count,
        "factor": args.factor,
        "confidence_scale": args.confidence_scale,
        "dtype": args.dtype,
        "prior_format": "[DoLP, cos(2AoLP), sin(2AoLP)]",
        "confidence_format": "[3,H,W] in [0,1]",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
