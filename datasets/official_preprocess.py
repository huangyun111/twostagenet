"""Shared preprocessing helpers for official-style twostagenet experiments."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image


def read_rgb_image(path: str | Path, normalize_mode: str = "fixed255") -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read RGB/S0 image: {path}")
    image = bgr_like_to_rgb(image)
    array = image.astype(np.float32)
    if normalize_mode == "image_max":
        max_value = float(np.nanmax(array))
        if max_value <= 0.0:
            raise ValueError(f"RGB/S0 image has non-positive max value: {path}")
        denom = max_value
    else:
        denom = 255.0
    array = np.clip(array / denom, 0.0, 1.0) * 2.0 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def read_polar_encoding(path: str | Path) -> torch.Tensor:
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None:
        encoded = iio.imread(path)
        if encoded.ndim == 2 or encoded.shape[-1] < 3:
            raise ValueError(f"Expected 3-channel Polarization_Encoding image: {path}")
        encoded_rgb = encoded[..., :3]
    else:
        if encoded.ndim != 3 or encoded.shape[-1] < 3:
            raise ValueError(f"Expected 3-channel Polarization_Encoding image: {path}")
        encoded_rgb = bgr_like_to_rgb(encoded)

    unit = encoded_rgb[..., :3].astype(np.float32) / dtype_max(encoded_rgb.dtype)
    dolp = unit[..., 0].clip(0.0, 1.0)
    cos2 = (unit[..., 1] * 2.0 - 1.0).clip(-1.0, 1.0)
    sin2 = (unit[..., 2] * 2.0 - 1.0).clip(-1.0, 1.0)
    return clamp_polar(torch.from_numpy(np.stack((dolp, cos2, sin2), axis=0)))


def bgr_like_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Expected image with at least 3 channels, got {image.shape}.")
    if image.shape[-1] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)


def dtype_max(dtype: np.dtype) -> float:
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 1.0


def clamp_polar(polar: torch.Tensor) -> torch.Tensor:
    dolp = polar[0:1].clamp(0.0, 1.0)
    cos_sin = polar[1:3].clamp(-1.0, 1.0)
    return torch.cat((dolp, cos_sin), dim=0).contiguous()


def resize_tensor_hw(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    channels = []
    for channel in tensor:
        image = Image.fromarray(channel.numpy().astype(np.float32), mode="F")
        resized = image.resize((width, height), Image.BILINEAR)
        channels.append(np.asarray(resized, dtype=np.float32))
    return torch.from_numpy(np.stack(channels, axis=0)).contiguous()


def resize_rgb_hw(rgb: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return resize_tensor_hw(rgb, height, width).clamp(-1.0, 1.0)


def resize_polar_hw(polar: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return clamp_polar(resize_tensor_hw(polar, height, width))


def resize_confidence_hw(confidence: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return resize_tensor_hw(confidence, height, width).clamp(0.0, 1.0)


def resize_short_side_if_needed(
    tensors: list[torch.Tensor],
    crop_size: int,
    kinds: list[str],
) -> list[torch.Tensor]:
    height, width = tensors[0].shape[-2:]
    if height >= crop_size and width >= crop_size:
        return tensors
    scale = crop_size / min(height, width)
    target_height = int(round(height * scale))
    target_width = int(round(width * scale))
    return [
        resize_by_kind(tensor, target_height, target_width, kind)
        for tensor, kind in zip(tensors, kinds)
    ]


def resize_by_kind(tensor: torch.Tensor, height: int, width: int, kind: str) -> torch.Tensor:
    if kind == "rgb":
        return resize_rgb_hw(tensor, height, width)
    if kind == "polar":
        return resize_polar_hw(tensor, height, width)
    if kind == "confidence":
        return resize_confidence_hw(tensor, height, width)
    raise ValueError(f"Unsupported resize kind: {kind}")


def crop_tensors(
    tensors: list[torch.Tensor],
    crop_height: int,
    crop_width: int,
    random_crop: bool,
    rng: random.Random,
) -> list[torch.Tensor]:
    height, width = tensors[0].shape[-2:]
    if crop_height > height or crop_width > width:
        raise ValueError(f"Cannot crop {crop_height}x{crop_width} from {height}x{width}.")
    if random_crop:
        top = rng.randint(0, height - crop_height)
        left = rng.randint(0, width - crop_width)
    else:
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
    return [tensor[:, top : top + crop_height, left : left + crop_width] for tensor in tensors]


def crop_to_common_size(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    height = min(tensor.shape[-2] for tensor in tensors)
    width = min(tensor.shape[-1] for tensor in tensors)
    cropped = []
    for tensor in tensors:
        source_height, source_width = tensor.shape[-2:]
        top = (source_height - height) // 2
        left = (source_width - width) // 2
        cropped.append(tensor[:, top : top + height, left : left + width])
    return cropped


def crop_to_divisible(tensor: torch.Tensor, divisible_by: int) -> tuple[torch.Tensor, str]:
    if divisible_by <= 0:
        raise ValueError("divisible_by must be positive.")
    height, width = tensor.shape[-2:]
    target_height = (height // divisible_by) * divisible_by
    target_width = (width // divisible_by) * divisible_by
    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"Input is too small for divisible_by={divisible_by}: {height}x{width}")
    cropped = crop_tensors([tensor], target_height, target_width, random_crop=False, rng=random.Random(0))[0]
    return cropped, f"{target_height}x{target_width}"


def format_hw(tensor: torch.Tensor) -> str:
    height, width = tensor.shape[-2:]
    return f"{height}x{width}"
