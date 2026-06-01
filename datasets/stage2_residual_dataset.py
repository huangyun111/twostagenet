"""Dataset for Stage 2 residual polarization diffusion training."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class Stage2ResidualDataset(Dataset):
    """Load RGB, GT polarization, Stage 1 prior, and confidence tensors.

    Polarization_Encoding is read with imageio/PIL-style RGB ordering, not cv2.
    In RGB order the channels are [DoLP, cos2AoLP, sin2AoLP].
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
        return_path: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.stage1_dir = Path(stage1_dir)
        self.rgb_dir = self._resolve_subdir(self.root_dir, rgb_subdir)
        self.polar_dir = self._resolve_subdir(self.root_dir, polar_subdir)
        self.prior_dir = self._resolve_subdir(self.stage1_dir, prior_subdir)
        self.confidence_dir = self._resolve_subdir(self.stage1_dir, confidence_subdir)
        self.image_size = image_size
        self.return_path = return_path

        if image_size is not None and image_size <= 0:
            raise ValueError("image_size must be positive or None.")

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

        rgb = self._read_rgb(rgb_path)
        polar_gt = self._read_polar(polar_path)
        prior = self._read_prior(prior_path)
        confidence = self._read_confidence(confidence_path)

        if self.image_size is not None:
            rgb = self._resize_rgb(rgb, self.image_size)
            polar_gt = self._resize_polar(polar_gt, self.image_size)
            prior = self._resize_polar(prior, self.image_size)
            confidence = self._resize_map(confidence, self.image_size).clamp(0.0, 1.0)

        item: dict[str, torch.Tensor | str] = {
            "rgb": rgb,
            "polar_gt": polar_gt,
            "prior": prior,
            "confidence": confidence,
            "name": stem,
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
        image = Image.open(path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = array * 2.0 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    @classmethod
    def _read_polar(cls, path: Path) -> torch.Tensor:
        encoded = iio.imread(path)
        if encoded.ndim == 2:
            raise ValueError(f"Expected 3-channel Polarization_Encoding image: {path}")
        if encoded.shape[-1] < 3:
            raise ValueError(f"Expected at least 3 channels in Polarization_Encoding: {path}")

        encoded = encoded[..., :3]
        encoded_float = cls._to_unit_range(encoded)

        dolp = encoded_float[..., 0]
        cos2 = encoded_float[..., 1] * 2.0 - 1.0
        sin2 = encoded_float[..., 2] * 2.0 - 1.0

        polar = np.stack((dolp, cos2, sin2), axis=0).astype(np.float32)
        return cls._clamp_polar(torch.from_numpy(polar))

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
        array = rgb.permute(1, 2, 0).numpy()
        array = ((array + 1.0) * 0.5 * 255.0).clip(0.0, 255.0).astype(np.uint8)
        image = Image.fromarray(array, mode="RGB")
        image = image.resize((image_size, image_size), Image.BILINEAR)
        resized = np.asarray(image, dtype=np.float32) / 255.0
        resized = resized * 2.0 - 1.0
        return torch.from_numpy(resized).permute(2, 0, 1).contiguous()

    @classmethod
    def _resize_polar(cls, polar: torch.Tensor, image_size: int) -> torch.Tensor:
        resized = cls._resize_map(polar, image_size)
        return cls._clamp_polar(resized)

    @staticmethod
    def _resize_map(tensor: torch.Tensor, image_size: int) -> torch.Tensor:
        channels = []
        for channel in tensor:
            image = Image.fromarray(channel.numpy().astype(np.float32), mode="F")
            image = image.resize((image_size, image_size), Image.BILINEAR)
            channels.append(np.asarray(image, dtype=np.float32))
        return torch.from_numpy(np.stack(channels, axis=0)).contiguous()

    @staticmethod
    def _clamp_polar(polar: torch.Tensor) -> torch.Tensor:
        dolp = polar[0:1].clamp(0.0, 1.0)
        cos_sin = polar[1:3].clamp(-1.0, 1.0)
        return torch.cat((dolp, cos_sin), dim=0).contiguous()
