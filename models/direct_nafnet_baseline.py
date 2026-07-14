"""Direct NAFNet baseline for RGB/S0 -> polarization encoding.

The backbone follows the official NAFNet width-32 architecture from
https://github.com/megvii-research/NAFNet.  It is adapted here to predict
``[DoLP, cos(2*AoLP), sin(2*AoLP)]``: the image-restoration input residual is
removed because RGB/S0 and polarization encoding are different quantities,
and a physical output projection is applied instead.

NAFNet source license (MIT):

Copyright (c) 2022 megvii-model

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise layer normalization for NCHW feature maps."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        return normalized * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    """Nonlinear-activation-free restoration block."""

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0 or dw_expand <= 0 or ffn_expand <= 0:
            raise ValueError("channels and expansion factors must be positive.")
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand
        if dw_channels % 2 or ffn_channels % 2:
            raise ValueError("expanded channel counts must be even for SimpleGate.")

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            padding=1,
            groups=dw_channels,
        )
        self.gate1 = SimpleGate()
        gated_channels = dw_channels // 2
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gated_channels, gated_channels, kernel_size=1),
        )
        self.conv3 = nn.Conv2d(gated_channels, channels, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, kernel_size=1)
        self.gate2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.gate1(self.conv2(self.conv1(self.norm1(inputs))))
        x = x * self.sca(x)
        x = self.conv3(x)
        first_residual = inputs + self.dropout1(x) * self.beta

        x = self.conv5(self.gate2(self.conv4(self.norm2(first_residual))))
        return first_residual + self.dropout2(x) * self.gamma


def _make_stage(channels: int, num_blocks: int) -> nn.Sequential:
    if num_blocks < 0:
        raise ValueError("block counts cannot be negative.")
    return nn.Sequential(*(NAFBlock(channels) for _ in range(num_blocks)))


class DirectNAFNetBaseline(nn.Module):
    """Official-width NAFNet backbone with a polarization output projection."""

    def __init__(
        self,
        width: int = 32,
        enc_blk_nums: Sequence[int] = (2, 2, 4, 8),
        middle_blk_num: int = 12,
        dec_blk_nums: Sequence[int] = (2, 2, 2, 2),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        enc_blk_nums = tuple(int(value) for value in enc_blk_nums)
        dec_blk_nums = tuple(int(value) for value in dec_blk_nums)
        if width <= 0:
            raise ValueError("width must be positive.")
        if len(enc_blk_nums) != len(dec_blk_nums) or not enc_blk_nums:
            raise ValueError("encoder and decoder block lists must have equal non-zero length.")
        if middle_blk_num < 0:
            raise ValueError("middle_blk_num cannot be negative.")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")

        self.eps = eps
        self.intro = nn.Conv2d(3, width, kernel_size=3, padding=1)
        self.ending = nn.Conv2d(width, 3, kernel_size=3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = width
        for num_blocks in enc_blk_nums:
            self.encoders.append(_make_stage(channels, num_blocks))
            self.downs.append(nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2))
            channels *= 2

        self.middle_blocks = _make_stage(channels, middle_blk_num)

        for num_blocks in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(_make_stage(channels, num_blocks))

        self.padder_size = 2 ** len(enc_blk_nums)

    def _pad(self, inputs: torch.Tensor) -> torch.Tensor:
        height, width = inputs.shape[-2:]
        pad_h = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - width % self.padder_size) % self.padder_size
        if not pad_h and not pad_w:
            return inputs
        return F.pad(inputs, (0, pad_w, 0, pad_h))

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        height, width = rgb.shape[-2:]
        x = self.intro(self._pad(rgb))
        skip_features = []
        for encoder, downsample in zip(self.encoders, self.downs):
            x = encoder(x)
            skip_features.append(x)
            x = downsample(x)

        x = self.middle_blocks(x)
        for decoder, upsample, skip in zip(
            self.decoders,
            self.ups,
            reversed(skip_features),
        ):
            x = decoder(upsample(x) + skip)

        raw = self.ending(x)[..., :height, :width]
        dolp = torch.sigmoid(raw[:, 0:1])
        raw_cos_sin = raw[:, 1:3]
        norm = raw_cos_sin.square().sum(dim=1, keepdim=True).sqrt().clamp_min(self.eps)
        cos_sin = raw_cos_sin / norm
        return torch.cat((dolp, cos_sin), dim=1).contiguous()
