"""Direct Restormer baseline for RGB/S0 -> polarization encoding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: float = 2.66) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class MDTA(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        head_dim = c // self.num_heads
        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, expansion: float = 2.66) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = FeedForward(dim, expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def make_blocks(dim: int, num_blocks: int, heads: int, expansion: float) -> nn.Sequential:
    return nn.Sequential(*(TransformerBlock(dim, heads, expansion) for _ in range(num_blocks)))


class DirectRestormerBaseline(nn.Module):
    """Restormer-style deterministic baseline with physical output projection."""

    def __init__(
        self,
        dim: int = 32,
        num_blocks: tuple[int, int, int, int] = (2, 2, 2, 4),
        num_heads: tuple[int, int, int, int] = (1, 2, 4, 8),
        expansion: float = 2.66,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if len(num_blocks) != 4 or len(num_heads) != 4:
            raise ValueError("num_blocks and num_heads must each have 4 entries.")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        self.eps = eps
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=3, padding=1)

        self.encoder1 = make_blocks(dim, num_blocks[0], num_heads[0], expansion)
        self.down1 = Downsample(dim)
        self.encoder2 = make_blocks(dim * 2, num_blocks[1], num_heads[1], expansion)
        self.down2 = Downsample(dim * 2)
        self.encoder3 = make_blocks(dim * 4, num_blocks[2], num_heads[2], expansion)
        self.down3 = Downsample(dim * 4)

        self.latent = make_blocks(dim * 8, num_blocks[3], num_heads[3], expansion)

        self.up3 = Upsample(dim * 8)
        self.reduce3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1)
        self.decoder3 = make_blocks(dim * 4, num_blocks[2], num_heads[2], expansion)
        self.up2 = Upsample(dim * 4)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1)
        self.decoder2 = make_blocks(dim * 2, num_blocks[1], num_heads[1], expansion)
        self.up1 = Upsample(dim * 2)
        self.reduce1 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.decoder1 = make_blocks(dim, num_blocks[0], num_heads[0], expansion)

        self.output = nn.Conv2d(dim, 3, kernel_size=3, padding=1)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        height, width = rgb.shape[-2:]
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        if pad_h or pad_w:
            rgb = F.pad(rgb, (0, pad_w, 0, pad_h), mode="reflect")

        x1 = self.encoder1(self.patch_embed(rgb))
        x2 = self.encoder2(self.down1(x1))
        x3 = self.encoder3(self.down2(x2))
        latent = self.latent(self.down3(x3))

        y = self.up3(latent)
        y = self.decoder3(self.reduce3(torch.cat([y, x3], dim=1)))
        y = self.up2(y)
        y = self.decoder2(self.reduce2(torch.cat([y, x2], dim=1)))
        y = self.up1(y)
        y = self.decoder1(self.reduce1(torch.cat([y, x1], dim=1)))
        raw = self.output(y)[..., :height, :width]

        dolp = torch.sigmoid(raw[:, 0:1])
        raw_cos_sin = raw[:, 1:3]
        norm = torch.sqrt((raw_cos_sin * raw_cos_sin).sum(dim=1, keepdim=True)).clamp_min(self.eps)
        cos_sin = raw_cos_sin / norm
        return torch.cat((dolp, cos_sin), dim=1).contiguous()
