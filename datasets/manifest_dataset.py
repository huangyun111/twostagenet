"""Manifest-driven dataset for the 13168 final_new512 clean split.

Unlike the flat-directory :class:`Stage1PriorDataset`, this dataset reads
``assembly_manifest.json`` produced by the data-assembly stage. Each frame in
the manifest binds an S0 input (``.npy``) and a Polarization_Encoding target
(``.png``) plus a frozen ``split`` field (train/val/test, scene-exclusive).

Key differences from the official512 path (deliberate, see CLAUDE.md):
  * Input is a single-channel **S0 intensity** stored as float32 ``.npy`` already
    normalised to [0, 1]. We replicate it to 3 channels and map [0, 1] -> [-1, 1]
    (NO division by 255 — the npy is already unit range).
  * Target PNGs are uint8 ``[DoLP, cos2AoLP, sin2AoLP]`` (R/G/B). Decoding here is
    **cv2-free** (imageio) and was verified bit-identical to the cv2-based
    ``official_preprocess.read_polar_encoding``. This keeps the manifest training
    path importable even on servers whose env lacks OpenCV.

All frames are 480x480 (480 % 32 == 0), so the default path needs no resize/crop.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# 16 source-corrupted polar_target PNGs (truncated / 0-byte), all in the train
# split. Discovered during the downstream-normal stage (see CLAUDE.md); the bad
# bytes exist on both the local D: copy and the dl114 copy, so they cannot be
# re-fetched. They must be skipped or PIL/imageio raises "image file is
# truncated" mid-training. Matches downstream_normal_deepsfp exclusion exactly.
_EXCLUDED_POLAR_TARGETS = {
    "dataset/5/Polarization_Encoding/polar_%05d.png" % i for i in range(26, 40)
} | {
    "dataset/10/Polarization_Encoding/polar_00026.png",
    "dataset/11/Polarization_Encoding/polar_00021.png",
}


def _clamp_polar(polar: torch.Tensor) -> torch.Tensor:
    """[DoLP, cos2, sin2] -> DoLP in [0,1], cos/sin in [-1,1]."""
    dolp = polar[0:1].clamp(0.0, 1.0)
    cos_sin = polar[1:3].clamp(-1.0, 1.0)
    return torch.cat((dolp, cos_sin), dim=0).contiguous()


def _resize_chw(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    channels = []
    for channel in tensor:
        image = Image.fromarray(channel.numpy().astype(np.float32), mode="F")
        resized = image.resize((width, height), Image.BILINEAR)
        channels.append(np.asarray(resized, dtype=np.float32))
    return torch.from_numpy(np.stack(channels, axis=0)).contiguous()


def read_s0_npy(path: str | Path) -> torch.Tensor:
    """Load an S0 ``.npy`` (HxW float32 in [0,1]) as a 3-channel [-1,1] tensor."""
    array = np.load(path)
    if array.ndim == 3:
        array = array[..., 0] if array.shape[-1] <= 4 else array[0]
    array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    array = np.clip(array, 0.0, 1.0) * 2.0 - 1.0
    chw = np.repeat(array[None, ...], 3, axis=0)
    return torch.from_numpy(np.ascontiguousarray(chw))


def read_polar_target_png(path: str | Path) -> torch.Tensor:
    """Decode a uint8/uint16 ``[DoLP, cos2, sin2]`` PNG -> [3,H,W] tensor (cv2-free)."""
    img = iio.imread(path)
    if img.ndim != 3 or img.shape[-1] < 3:
        raise ValueError(f"Expected 3-channel Polarization_Encoding image: {path}")
    maxval = 65535.0 if img.dtype == np.uint16 else 255.0
    unit = img[..., :3].astype(np.float32) / maxval
    dolp = np.clip(unit[..., 0], 0.0, 1.0)
    cos2 = np.clip(unit[..., 1] * 2.0 - 1.0, -1.0, 1.0)
    sin2 = np.clip(unit[..., 2] * 2.0 - 1.0, -1.0, 1.0)
    return _clamp_polar(torch.from_numpy(np.stack((dolp, cos2, sin2), axis=0)))


class Stage1ManifestDataset(Dataset):
    """RGB(S0)->polar pairs for Stage 1, sourced from ``assembly_manifest.json``.

    Args:
        manifest_path: Path to ``assembly_manifest.json``.
        dataset_root: Directory the manifest's frame paths are relative to.
            Defaults to ``manifest_path.parent.parent`` (the project root that
            contains the ``dataset/`` tree). MUST be overridable per machine.
        split: One of ``train`` / ``val`` / ``test`` (or ``all``).
        image_size: If set, both S0 and target are resized to this square size.
        crop_size: When >0 and < native size, a random/center crop is applied.
        random_crop: Random crop (train) vs center crop. Defaults to ``augment``.
        augment: Random horizontal flip.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: str | Path | None = None,
        split: str = "train",
        image_size: int | None = None,
        crop_size: int = 0,
        random_crop: bool | None = None,
        augment: bool = False,
        return_path: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        self.dataset_root = (
            Path(dataset_root)
            if dataset_root is not None
            else self.manifest_path.parent.parent
        )
        self.split = split
        self.image_size = image_size
        self.crop_size = crop_size
        self.random_crop = augment if random_crop is None else random_crop
        self.augment = augment
        self.return_path = return_path

        if image_size is not None and image_size <= 0:
            raise ValueError("image_size must be positive or None.")
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"Unsupported split: {split}")

        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        frames = manifest.get("frames") or manifest.get("items") or []
        if split != "all":
            frames = [f for f in frames if f.get("split") == split]
        frames = [
            f for f in frames if f.get("polar_target") not in _EXCLUDED_POLAR_TARGETS
        ]
        self.samples = [
            (
                self.dataset_root / frame["s0"],
                self.dataset_root / frame["polar_target"],
                f"{frame['scene']}_{frame['frame']}",
            )
            for frame in frames
        ]
        if not self.samples:
            raise RuntimeError(
                f"No frames for split={split} in manifest {self.manifest_path}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        s0_path, polar_path, stem = self.samples[index]
        rgb = read_s0_npy(s0_path)
        polar = read_polar_target_png(polar_path)
        native_h, native_w = rgb.shape[-2:]

        if self.image_size is not None:
            rgb = _resize_chw(rgb, self.image_size, self.image_size).clamp(-1.0, 1.0)
            polar = _clamp_polar(_resize_chw(polar, self.image_size, self.image_size))

        if self.crop_size and self.crop_size > 0:
            rgb, polar = self._crop([rgb, polar], self.crop_size, index)

        if self.augment and torch.rand(()) < 0.5:
            rgb = torch.flip(rgb, dims=(-1,))
            polar = torch.flip(polar, dims=(-1,))

        polar = _clamp_polar(polar)
        item: dict[str, torch.Tensor | str] = {
            "rgb": rgb.contiguous(),
            "polar": polar.contiguous(),
            "name": stem,
            "input_native_size": f"{native_h}x{native_w}",
            "input_size": f"{rgb.shape[-2]}x{rgb.shape[-1]}",
        }
        if self.return_path:
            item["rgb_path"] = str(s0_path)
            item["polar_path"] = str(polar_path)
        return item

    def _crop(
        self, tensors: list[torch.Tensor], crop: int, index: int
    ) -> list[torch.Tensor]:
        height, width = tensors[0].shape[-2:]
        if crop > height or crop > width:
            return tensors
        if self.random_crop:
            rng = random.Random(index)
            top = rng.randint(0, height - crop)
            left = rng.randint(0, width - crop)
        else:
            top = (height - crop) // 2
            left = (width - crop) // 2
        return [t[:, top : top + crop, left : left + crop] for t in tensors]


def _load_chw_npy(path: Path, expect_channels: int = 3) -> torch.Tensor:
    """Load a [C,H,W] float32 npy (Stage1-exported prior / confidence)."""
    array = np.load(path).astype(np.float32)
    if array.ndim != 3 or array.shape[0] != expect_channels:
        raise ValueError(
            f"Expected npy with shape [{expect_channels},H,W], got {array.shape}: {path}"
        )
    return torch.from_numpy(np.ascontiguousarray(array))


class Stage2ManifestDataset(Dataset):
    """Stage 2 training pairs from ``assembly_manifest.json`` + Stage1 exports.

    Yields ``{rgb, polar_gt, prior, confidence, name}`` where:
      * ``rgb``        — S0 .npy replicated to 3ch, mapped to [-1,1] (same as Stage1).
      * ``polar_gt``   — Polarization_Encoding PNG decoded to [DoLP,cos2,sin2].
      * ``prior``      — Stage1-exported ``prior_npy/<scene>_<frame>.npy`` ([3,H,W]).
      * ``confidence`` — Stage1-exported ``confidence_npy/<scene>_<frame>.npy`` ([3,H,W]).

    All frames are native 480x480, so no resize/crop is applied by default; an
    optional horizontal flip (``augment``) is synchronised across all four maps.
    Frames whose Stage1 export is missing are skipped (with a one-line warning),
    so the dataset stays consistent with a partial export. The same 16 corrupted
    polar_target PNGs as :class:`Stage1ManifestDataset` are excluded.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        stage1_dir: str | Path,
        dataset_root: str | Path | None = None,
        split: str = "train",
        augment: bool = False,
        return_path: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        self.stage1_dir = Path(stage1_dir)
        self.prior_dir = self.stage1_dir / "prior_npy"
        self.confidence_dir = self.stage1_dir / "confidence_npy"
        self.dataset_root = (
            Path(dataset_root)
            if dataset_root is not None
            else self.manifest_path.parent.parent
        )
        self.split = split
        self.augment = augment
        self.return_path = return_path

        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"Unsupported split: {split}")

        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        frames = manifest.get("frames") or manifest.get("items") or []
        if split != "all":
            frames = [f for f in frames if f.get("split") == split]
        frames = [
            f for f in frames if f.get("polar_target") not in _EXCLUDED_POLAR_TARGETS
        ]

        self.samples: list[tuple[Path, Path, Path, Path, str]] = []
        missing = 0
        for frame in frames:
            name = f"{frame['scene']}_{frame['frame']}"
            prior_path = self.prior_dir / f"{name}.npy"
            confidence_path = self.confidence_dir / f"{name}.npy"
            if not prior_path.is_file() or not confidence_path.is_file():
                missing += 1
                continue
            self.samples.append(
                (
                    self.dataset_root / frame["s0"],
                    self.dataset_root / frame["polar_target"],
                    prior_path,
                    confidence_path,
                    name,
                )
            )
        if missing:
            print(
                f"[Stage2ManifestDataset] split={split}: skipped {missing} frames "
                f"with no Stage1 export under {self.stage1_dir}.",
                flush=True,
            )
        if not self.samples:
            raise RuntimeError(
                f"No Stage2 samples for split={split}: manifest={self.manifest_path}, "
                f"stage1_dir={self.stage1_dir}. Did Stage1 export run for this split?"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        s0_path, polar_path, prior_path, confidence_path, name = self.samples[index]
        rgb = read_s0_npy(s0_path)
        polar_gt = read_polar_target_png(polar_path)
        prior = _clamp_polar(_load_chw_npy(prior_path))
        confidence = _load_chw_npy(confidence_path).clamp(0.0, 1.0)
        native_h, native_w = rgb.shape[-2:]

        if self.augment and torch.rand(()) < 0.5:
            rgb = torch.flip(rgb, dims=(-1,))
            polar_gt = torch.flip(polar_gt, dims=(-1,))
            prior = torch.flip(prior, dims=(-1,))
            confidence = torch.flip(confidence, dims=(-1,))

        item: dict[str, torch.Tensor | str] = {
            "rgb": rgb.contiguous(),
            "polar_gt": polar_gt.contiguous(),
            "prior": prior.contiguous(),
            "confidence": confidence.contiguous(),
            "name": name,
            "input_native_size": f"{native_h}x{native_w}",
            "input_size": f"{rgb.shape[-2]}x{rgb.shape[-1]}",
        }
        if self.return_path:
            item["rgb_path"] = str(s0_path)
            item["polar_path"] = str(polar_path)
            item["prior_path"] = str(prior_path)
            item["confidence_path"] = str(confidence_path)
        return item
