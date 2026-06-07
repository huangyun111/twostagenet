"""Dataset for Stage 2 residual polarization diffusion training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.official_preprocess import (
    clamp_polar,
    crop_tensors,
    crop_to_common_size,
    format_hw,
    read_polar_encoding,
    read_rgb_image,
    resize_confidence_hw,
    resize_polar_hw,
    resize_rgb_hw,
    resize_short_side_if_needed,
)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class Stage2ResidualDataset(Dataset):
    """Load RGB, GT polarization, Stage 1 prior, and confidence tensors.

    GT and prior use physical [DoLP, cos2AoLP, sin2AoLP]. official_train
    synchronizes RGB/GT/prior/confidence crops; official_infer keeps GT native
    and deterministically crops RGB/prior/confidence to a common legal size.
    """

    def __init__(
        self,
        root_dir: str | Path,
        stage1_dir: str | Path,
        rgb_subdir: str = "s0",
        polar_subdir: str = "Polarization_Encoding",
        prior_subdir: str = "prior_npy",
        confidence_subdir: str = "confidence_npy",
        image_size: int | None = None,
        preprocess_mode: str = "resize256",
        crop_size: int = 512,
        normalize_mode: str = "fixed255",
        divisible_by: int = 32,
        random_crop: bool = False,
        return_path: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.stage1_dir = Path(stage1_dir)
        self.rgb_dir = self._resolve_subdir(self.root_dir, rgb_subdir)
        self.polar_dir = self._resolve_subdir(self.root_dir, polar_subdir)
        self.prior_dir = self._resolve_subdir(self.stage1_dir, prior_subdir)
        self.confidence_dir = self._resolve_subdir(self.stage1_dir, confidence_subdir)
        self.image_size = image_size
        self.preprocess_mode = preprocess_mode
        self.crop_size = crop_size
        self.normalize_mode = normalize_mode
        self.divisible_by = divisible_by
        self.random_crop = random_crop
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

        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(
                "No matched Stage 2 samples found under "
                f"{self.rgb_dir}, {self.polar_dir}, {self.prior_dir}, "
                f"and {self.confidence_dir}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        rgb_path, polar_path, prior_path, confidence_path, stem = self.samples[index]

        rgb = read_rgb_image(rgb_path, normalize_mode=self.normalize_mode)
        polar_gt = self._read_polar(polar_path)
        prior = self._read_prior(prior_path)
        confidence = self._read_confidence(confidence_path)
        input_native_size = format_hw(rgb)

        if self.preprocess_mode == "resize256" and self.image_size is not None:
            rgb = self._resize_rgb(rgb, self.image_size)
            polar_gt = self._resize_polar(polar_gt, self.image_size)
            prior = self._resize_polar(prior, self.image_size)
            confidence = self._resize_map(confidence, self.image_size).clamp(0.0, 1.0)
        elif self.preprocess_mode == "official_train":
            rgb, polar_gt, prior, confidence = crop_to_common_size(
                [rgb, polar_gt, prior, confidence]
            )
            rgb, polar_gt, prior, confidence = resize_short_side_if_needed(
                [rgb, polar_gt, prior, confidence],
                self.crop_size,
                ["rgb", "polar", "polar", "confidence"],
            )
            import random

            rgb, polar_gt, prior, confidence = crop_tensors(
                [rgb, polar_gt, prior, confidence],
                self.crop_size,
                self.crop_size,
                random_crop=self.random_crop,
                rng=random.Random(index),
            )
        elif self.preprocess_mode == "official_infer":
            rgb, prior, confidence = crop_to_common_size([rgb, prior, confidence])
            height, width = rgb.shape[-2:]
            target_height = (height // self.divisible_by) * self.divisible_by
            target_width = (width // self.divisible_by) * self.divisible_by
            if target_height <= 0 or target_width <= 0:
                raise ValueError(
                    f"Input is too small for divisible_by={self.divisible_by}: {height}x{width}"
                )
            rgb, prior, confidence = crop_tensors(
                [rgb, prior, confidence],
                target_height,
                target_width,
                random_crop=False,
                rng=__import__("random").Random(0),
            )

        polar_gt = self._clamp_polar(polar_gt)
        prior = self._clamp_polar(prior)
        confidence = confidence.clamp(0.0, 1.0)
        input_size = format_hw(rgb)

        item: dict[str, torch.Tensor | str] = {
            "rgb": rgb,
            "polar_gt": polar_gt,
            "prior": prior,
            "confidence": confidence,
            "name": stem,
            "input_native_size": input_native_size,
            "input_size": input_size,
        }
        if self.return_path:
            item["rgb_path"] = str(rgb_path)
            item["polar_path"] = str(polar_path)
            item["prior_path"] = str(prior_path)
            item["confidence_path"] = str(confidence_path)
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

    def _collect_samples(self) -> list[tuple[Path, Path, Path, Path, str]]:
        rgb_files = self._index_images_by_stem(self.rgb_dir)
        polar_files = self._index_images_by_stem(self.polar_dir)
        prior_files = self._index_npy_by_stem(self.prior_dir)
        confidence_files = self._index_npy_by_stem(self.confidence_dir)

        matched_stems = sorted(
            set(rgb_files) & set(polar_files) & set(prior_files) & set(confidence_files)
        )
        return [
            (
                rgb_files[stem],
                polar_files[stem],
                prior_files[stem],
                confidence_files[stem],
                stem,
            )
            for stem in matched_stems
        ]

    @staticmethod
    def _index_images_by_stem(directory: Path) -> dict[str, Path]:
        files: dict[str, Path] = {}
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                files.setdefault(path.stem, path)
        return files

    @staticmethod
    def _index_npy_by_stem(directory: Path) -> dict[str, Path]:
        files: dict[str, Path] = {}
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() == ".npy":
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

    @classmethod
    def _read_prior(cls, path: Path) -> torch.Tensor:
        array = np.load(path).astype(np.float32)
        cls._validate_chw_array(array, path, "prior")
        return cls._clamp_polar(torch.from_numpy(array))

    @classmethod
    def _read_confidence(cls, path: Path) -> torch.Tensor:
        array = np.load(path).astype(np.float32)
        cls._validate_chw_array(array, path, "confidence")
        return torch.from_numpy(array).clamp(0.0, 1.0).contiguous()

    @staticmethod
    def _validate_chw_array(array: np.ndarray, path: Path, name: str) -> None:
        if array.ndim != 3 or array.shape[0] != 3:
            raise ValueError(
                f"Expected {name} npy with shape [3,H,W], got {array.shape}: {path}"
            )

    @staticmethod
    def _resize_rgb(rgb: torch.Tensor, image_size: int) -> torch.Tensor:
        return resize_rgb_hw(rgb, image_size, image_size)

    @classmethod
    def _resize_polar(cls, polar: torch.Tensor, image_size: int) -> torch.Tensor:
        return resize_polar_hw(polar, image_size, image_size)

    @staticmethod
    def _resize_map(tensor: torch.Tensor, image_size: int) -> torch.Tensor:
        return resize_confidence_hw(tensor, image_size, image_size)

    @staticmethod
    def _clamp_polar(polar: torch.Tensor) -> torch.Tensor:
        return clamp_polar(polar)
