"""Export Stage 1 coarse polarization priors for Stage 2 training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.stage1_prior_dataset import Stage1PriorDataset  # noqa: E402
from models.polar_prior_net import PolarPriorNet  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Stage 1 prior and confidence maps."
    )
    parser.add_argument("--root_dir", type=str, default=str(Path.home() / "Documents"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./stage1_exports")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--encoder_weights", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--vis_every", type=int, default=50)
    return parser.parse_args()


def normalize_encoder_weights(value: str | None) -> str | None:
    if value is None:
        return None
    if value.lower() in {"none", "null", ""}:
        return None
    return value


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model(args: argparse.Namespace, device: torch.device) -> PolarPriorNet:
    model = PolarPriorNet(encoder_weights=normalize_encoder_weights(args.encoder_weights))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def tensor_to_rgb_image(rgb: torch.Tensor) -> np.ndarray:
    image = rgb.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip((image + 1.0) * 0.5, 0.0, 1.0)


def to_display_map(tensor: torch.Tensor, value_range: tuple[float, float]) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    min_value, max_value = value_range
    array = (array - min_value) / (max_value - min_value)
    return np.clip(array, 0.0, 1.0)


def save_visualization(
    rgb: torch.Tensor,
    prior: torch.Tensor,
    confidence: torch.Tensor,
    path: Path,
) -> None:
    rgb_image = tensor_to_rgb_image(rgb)
    conf_aolp = confidence[1:3].mean(dim=0)

    panels = [
        ("RGB", rgb_image, None),
        ("Pred DoLP", prior[0].detach().cpu().numpy(), (0.0, 1.0)),
        ("Pred cos2", to_display_map(prior[1], (-1.0, 1.0)), (0.0, 1.0)),
        ("Pred sin2", to_display_map(prior[2], (-1.0, 1.0)), (0.0, 1.0)),
        ("Confidence DoLP", confidence[0].detach().cpu().numpy(), (0.0, 1.0)),
        ("Confidence AoLP", conf_aolp.detach().cpu().numpy(), (0.0, 1.0)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for axis, (title, image, value_range) in zip(axes.flat, panels):
        if image.ndim == 3:
            axis.imshow(image)
        else:
            vmin, vmax = value_range if value_range is not None else (None, None)
            axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def export_batch(
    batch: dict[str, torch.Tensor | list[str]],
    model: PolarPriorNet,
    device: torch.device,
    prior_dir: Path,
    confidence_dir: Path,
    vis_dir: Path,
    start_index: int,
    vis_every: int,
) -> int:
    rgb = batch["rgb"].to(device, non_blocking=True)
    names = batch["name"]

    with torch.no_grad():
        output = model(rgb)
        prior = output["polar_prior"].detach().cpu().clamp(
            torch.tensor([0.0, -1.0, -1.0]).view(1, 3, 1, 1),
            torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1),
        )
        confidence = output["confidence"].detach().cpu().clamp(0.0, 1.0)

    for offset, name in enumerate(names):
        sample_index = start_index + offset
        prior_array = prior[offset].numpy().astype(np.float32)
        confidence_array = confidence[offset].numpy().astype(np.float32)
        np.save(prior_dir / f"{name}.npy", prior_array)
        np.save(confidence_dir / f"{name}.npy", confidence_array)

        if vis_every > 0 and sample_index % vis_every == 0:
            save_visualization(
                batch["rgb"][offset],
                prior[offset],
                confidence[offset],
                vis_dir / f"{name}.png",
            )

    return len(names)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir)
    prior_dir = output_dir / "prior_npy"
    confidence_dir = output_dir / "confidence_npy"
    vis_dir = output_dir / "vis"
    prior_dir.mkdir(parents=True, exist_ok=True)
    confidence_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    dataset = Stage1PriorDataset(
        root_dir=args.root_dir,
        image_size=args.image_size,
        augment=False,
        return_path=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = load_model(args, device)

    exported = 0
    for batch in dataloader:
        exported += export_batch(
            batch=batch,
            model=model,
            device=device,
            prior_dir=prior_dir,
            confidence_dir=confidence_dir,
            vis_dir=vis_dir,
            start_index=exported,
            vis_every=args.vis_every,
        )
        print(f"exported {exported}/{len(dataset)}", flush=True)

    print(f"Done. Exported {exported} samples to {output_dir}")


if __name__ == "__main__":
    main()
