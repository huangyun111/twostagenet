"""Dataset for Stage 1 coarse polarization prior training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.official_preprocess import (
    clamp_polar,
    crop_to_divisible,
    crop_tensors,
    format_hw,
    read_polar_encoding,
    read_rgb_image,
    resize_polar_hw,
    resize_rgb_hw,
    resize_short_side_if_needed,
)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class Stage1PriorDataset(Dataset):
    """Load RGB images and Polarization_Encoding targets for Stage 1.

    Targets are always physical [DoLP, cos2AoLP, sin2AoLP]; DoLP stays [0, 1].
    official_train resizes the short side to crop_size only when needed, then
    applies train random crop or deterministic center crop.
    """

    def __init__(
        self,
        root_dir: str | Path,
        rgb_subdir: str = "s0",
        polar_subdir: str = "Polarization_Encoding",
        image_size: int | None = None,
        preprocess_mode: str = "resize256",
        crop_size: int = 512,
        normalize_mode: str = "fixed255",
        divisible_by: int = 32,
        random_crop: bool | None = None,
        augment: bool = False,
        return_path: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.rgb_dir = self._resolve_subdir(self.root_dir, rgb_subdir)
        self.polar_dir = self._resolve_subdir(self.root_dir, polar_subdir)
        self.image_size = image_size
        self.preprocess_mode = preprocess_mode
        self.crop_size = crop_size
        self.normalize_mode = normalize_mode
        self.divisible_by = divisible_by
        self.random_crop = augment if random_crop is None else random_crop
        self.augment = augment
        self.return_path = return_path

        if image_size is not None and image_size <= 0:
            raise ValueError("image_size must be positive or None.")
        if preprocess_mode not in {"resize256", "official_train", "official_infer"}:
            raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")
        if crop_size <= 0:
            raise ValueError("crop_size must be positive.")
        if normalize_mode not in {"fixed255", "image_max"}:
            raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")
        if divisible_by <= 0:
            raise ValueError("divisible_by must be positive.")

        self.samples = self._collect_pairs()
        if not self.samples:
            raise RuntimeError(
                "No matched RGB/Polarization_Encoding samples found under "
                f"{self.rgb_dir} and {self.polar_dir}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        rgb_path, polar_path, stem = self.samples[index]

        rgb = read_rgb_image(rgb_path, normalize_mode=self.normalize_mode)
        polar = self._read_polar(polar_path)
        input_native_size = format_hw(rgb)

        if self.preprocess_mode == "resize256" and self.image_size is not None:
            rgb = self._resize_rgb(rgb, self.image_size)
            polar = self._resize_polar(polar, self.image_size)
        elif self.preprocess_mode == "official_train":
            rgb, polar = resize_short_side_if_needed(
                [rgb, polar],
                self.crop_size,
                ["rgb", "polar"],
            )
            import random

            rgb, polar = crop_tensors(
                [rgb, polar],
                self.crop_size,
                self.crop_size,
                random_crop=self.random_crop,
                rng=random.Random(index),
            )
        elif self.preprocess_mode == "official_infer":
            rgb, _ = crop_to_divisible(rgb, self.divisible_by)
            polar, _ = crop_to_divisible(polar, self.divisible_by)

        if self.augment and torch.rand(()) < 0.5:
            rgb = torch.flip(rgb, dims=(-1,))
            polar = torch.flip(polar, dims=(-1,))

        polar = self._clamp_polar(polar)
        input_size = format_hw(rgb)

        item: dict[str, torch.Tensor | str] = {
            "rgb": rgb,
            "polar": polar,
            "name": stem,
            "input_native_size": input_native_size,
            "input_size": input_size,
        }
        if self.return_path:
            item["rgb_path"] = str(rgb_path)
            item["polar_path"] = str(polar_path)
        return item

    @staticmethod
    def _resolve_subdir(root_dir: Path, subdir: str) -> Path:
        """Resolve a subdirectory, tolerating case differences such as s0/S0."""
        direct = root_dir / subdir
        if direct.is_dir():
            return direct

        target = subdir.lower()
        if root_dir.is_dir():
            for child in root_dir.iterdir():
                if child.is_dir() and child.name.lower() == target:
                    return child

        raise FileNotFoundError(f"Required directory not found: {direct}")

    def _collect_pairs(self) -> list[tuple[Path, Path, str]]:
        rgb_files = self._index_images_by_stem(self.rgb_dir)
        polar_files = self._index_images_by_stem(self.polar_dir)
        matched_stems = sorted(set(rgb_files) & set(polar_files))
        return [(rgb_files[stem], polar_files[stem], stem) for stem in matched_stems]

    @staticmethod
    def _index_images_by_stem(directory: Path) -> dict[str, Path]:
        files: dict[str, Path] = {}
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                files.setdefault(path.stem, path)
        return files

    @staticmethod
    def _read_rgb(path: Path) -> torch.Tensor:
        return read_rgb_image(path, normalize_mode="fixed255")

    @classmethod
    def _read_polar(cls, path: Path) -> torch.Tensor:
        return read_polar_encoding(path)

    @staticmethod
    def _to_unit_range(array: np.ndarray) -> np.ndarray:
        """Convert integer or float image data to [0, 1] without double scaling."""
        if array.dtype == np.uint16:
            return array.astype(np.float32) / 65535.0
        if array.dtype == np.uint8:
            return array.astype(np.float32) / 255.0

        array_float = array.astype(np.float32)
        finite = array_float[np.isfinite(array_float)]
        if finite.size == 0:
            return np.zeros_like(array_float, dtype=np.float32)

        max_value = float(finite.max())
        min_value = float(finite.min())
        if min_value >= 0.0 and max_value <= 1.0:
            return array_float
        if min_value >= 0.0 and max_value <= 255.0:
            return array_float / 255.0
        if min_value >= 0.0 and max_value <= 65535.0:
            return array_float / 65535.0
        return np.clip(array_float, 0.0, 1.0)

    @staticmethod
    def _resize_rgb(rgb: torch.Tensor, image_size: int) -> torch.Tensor:
        return resize_rgb_hw(rgb, image_size, image_size)

    @classmethod
    def _resize_polar(cls, polar: torch.Tensor, image_size: int) -> torch.Tensor:
        return resize_polar_hw(polar, image_size, image_size)

    @staticmethod
    def _clamp_polar(polar: torch.Tensor) -> torch.Tensor:
        return clamp_polar(polar)
