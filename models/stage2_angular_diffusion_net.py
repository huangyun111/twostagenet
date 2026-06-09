"""Physical residual diffusion model for Stage 2 polarization refinement."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _num_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    exponent = -math.log(10000.0) * torch.arange(
        half_dim,
        device=timesteps.device,
        dtype=torch.float32,
    )
    exponent = exponent / max(half_dim - 1, 1)
    args = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def normalize_cos_sin(cos_sin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norm = torch.sqrt((cos_sin * cos_sin).sum(dim=1, keepdim=True) + eps)
    return cos_sin / norm


def target_residual_from_prior(prior: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    delta_dolp = target[:, 0:1] - prior[:, 0:1]
    cos_p = prior[:, 1:2]
    sin_p = prior[:, 2:3]
    cos_t = target[:, 1:2]
    sin_t = target[:, 2:3]
    delta_angle = torch.atan2(
        sin_t * cos_p - cos_t * sin_p,
        cos_t * cos_p + sin_t * sin_p,
    )
    return torch.cat([delta_dolp, delta_angle], dim=1).contiguous()


def apply_residual_to_prior(
    prior: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    delta_dolp = residual[:, 0:1]
    delta_angle = residual[:, 1:2]
    dolp_refined = torch.clamp(prior[:, 0:1] + delta_dolp, 0.0, 1.0)

    cos_p = prior[:, 1:2]
    sin_p = prior[:, 2:3]
    cos_d = torch.cos(delta_angle)
    sin_d = torch.sin(delta_angle)
    cos_refined = cos_d * cos_p - sin_d * sin_p
    sin_refined = sin_d * cos_p + cos_d * sin_p
    cos_sin = normalize_cos_sin(torch.cat([cos_refined, sin_refined], dim=1), eps)
    return torch.cat([dolp_refined, cos_sin], dim=1).contiguous()


class TimestepResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_channels, out_channels)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(x)


class Stage2AngularResidualDiffusion(nn.Module):
    """Predict x0 physical residual [delta_dolp, delta_angle], not noise."""

    def __init__(
        self,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 4, 4),
        time_channels: int | None = None,
        residual_scale: float = 0.3,
        angle_residual_scale: float = math.pi,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        if not channel_mult:
            raise ValueError("channel_mult must not be empty.")

        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.time_channels = time_channels or base_channels * 4
        self.residual_scale = residual_scale
        self.angle_residual_scale = angle_residual_scale
        self.eps = eps

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, self.time_channels),
            nn.SiLU(inplace=True),
            nn.Linear(self.time_channels, self.time_channels),
        )
        self.input_conv = nn.Conv2d(11, base_channels, kernel_size=3, padding=1)

        channels = [base_channels * mult for mult in channel_mult]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_channels = base_channels
        for index, out_channels in enumerate(channels):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        TimestepResidualBlock(in_channels, out_channels, self.time_channels),
                        TimestepResidualBlock(out_channels, out_channels, self.time_channels),
                    ]
                )
            )
            if index < len(channels) - 1:
                self.downsamples.append(Downsample(out_channels))
            in_channels = out_channels

        bottleneck_channels = channels[-1]
        self.mid_block1 = TimestepResidualBlock(
            bottleneck_channels,
            bottleneck_channels,
            self.time_channels,
        )
        self.mid_block2 = TimestepResidualBlock(
            bottleneck_channels,
            bottleneck_channels,
            self.time_channels,
        )

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        current_channels = bottleneck_channels
        for skip_channels in reversed(channels):
            self.upsamples.append(Upsample(current_channels))
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        TimestepResidualBlock(
                            current_channels + skip_channels,
                            skip_channels,
                            self.time_channels,
                        ),
                        TimestepResidualBlock(skip_channels, skip_channels, self.time_channels),
                    ]
                )
            )
            current_channels = skip_channels

        self.output_norm = nn.GroupNorm(_num_groups(current_channels), current_channels)
        self.output_conv = nn.Conv2d(current_channels, 2, kernel_size=3, padding=1)
        self.output_act = nn.SiLU(inplace=True)

    def forward(
        self,
        rgb: torch.Tensor,
        prior: torch.Tensor,
        confidence: torch.Tensor,
        residual_noisy: torch.Tensor,
        timestep: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if timestep.ndim == 0:
            timestep = timestep.expand(rgb.shape[0])
        time_emb = sinusoidal_timestep_embedding(timestep, self.base_channels)
        time_emb = self.time_mlp(time_emb)

        x = torch.cat([residual_noisy, rgb, prior, confidence], dim=1)
        x = self.input_conv(x)

        skips: list[torch.Tensor] = []
        for index, blocks in enumerate(self.down_blocks):
            for block in blocks:
                x = block(x, time_emb)
            skips.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)

        x = self.mid_block1(x, time_emb)
        x = self.mid_block2(x, time_emb)

        for upsample, blocks in zip(self.upsamples, self.up_blocks):
            skip = skips.pop()
            x = upsample(x, skip)
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                x = block(x, time_emb)

        raw_residual = self.output_conv(self.output_act(self.output_norm(x)))
        pred_delta_dolp = self.residual_scale * torch.tanh(raw_residual[:, 0:1])
        pred_delta_angle = self.angle_residual_scale * torch.tanh(raw_residual[:, 1:2])
        pred_residual = torch.cat([pred_delta_dolp, pred_delta_angle], dim=1).contiguous()
        refined = apply_residual_to_prior(prior, pred_residual, self.eps)
        return {
            "pred_residual": pred_residual,
            "pred_delta_dolp": pred_delta_dolp,
            "pred_delta_angle": pred_delta_angle,
            "refined": refined,
            "raw_residual": raw_residual,
        }


class DiffusionSchedule:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        device: torch.device | str = "cpu",
    ) -> None:
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive.")
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        betas = self._make_betas(num_train_timesteps, beta_schedule).to(device)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    @staticmethod
    def _make_betas(num_train_timesteps: int, beta_schedule: str) -> torch.Tensor:
        if beta_schedule == "linear":
            return torch.linspace(1e-4, 0.02, num_train_timesteps, dtype=torch.float32)
        if beta_schedule != "cosine":
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")

        steps = num_train_timesteps + 1
        t = torch.linspace(0, num_train_timesteps, steps, dtype=torch.float64)
        s = 0.008
        alpha_bar = torch.cos(((t / num_train_timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        return betas.clamp(1e-5, 0.999).float()

    def to(self, device: torch.device | str) -> "DiffusionSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    def q_sample(
        self,
        residual_start: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alphas_cumprod[timestep].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar) * residual_start + torch.sqrt(1.0 - alpha_bar) * noise

    def inference_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive.")
        steps = torch.linspace(
            self.num_train_timesteps - 1,
            0,
            num_inference_steps,
            device=device,
        )
        return steps.round().long().unique_consecutive()

    def ddim_step(
        self,
        current: torch.Tensor,
        pred_start: torch.Tensor,
        timestep: torch.Tensor,
        prev_timestep: torch.Tensor | None,
    ) -> torch.Tensor:
        alpha_bar_t = self.alphas_cumprod[timestep].view(-1, 1, 1, 1)
        pred_noise = (current - torch.sqrt(alpha_bar_t) * pred_start) / torch.sqrt(
            (1.0 - alpha_bar_t).clamp_min(1e-8)
        )
        if prev_timestep is None:
            return pred_start
        alpha_bar_prev = self.alphas_cumprod[prev_timestep].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar_prev) * pred_start + torch.sqrt(1.0 - alpha_bar_prev) * pred_noise


@torch.no_grad()
def ddim_sample_residual(
    model: Stage2AngularResidualDiffusion,
    schedule: DiffusionSchedule,
    rgb: torch.Tensor,
    prior: torch.Tensor,
    confidence: torch.Tensor,
    num_inference_steps: int = 10,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    device = rgb.device
    residual = torch.randn(
        rgb.shape[0],
        2,
        rgb.shape[-2],
        rgb.shape[-1],
        device=device,
        generator=generator,
    )
    timesteps = schedule.inference_timesteps(num_inference_steps, device=device)
    last_pred: dict[str, torch.Tensor] | None = None
    for index, timestep_value in enumerate(timesteps):
        timestep = torch.full((rgb.shape[0],), int(timestep_value), device=device, dtype=torch.long)
        pred = model(rgb, prior, confidence, residual, timestep)
        last_pred = pred
        if index + 1 < len(timesteps):
            prev_value = timesteps[index + 1]
            prev_timestep = torch.full((rgb.shape[0],), int(prev_value), device=device, dtype=torch.long)
        else:
            prev_timestep = None
        residual = schedule.ddim_step(residual, pred["pred_residual"], timestep, prev_timestep)

    if last_pred is None:
        raise RuntimeError("DDIM sampler produced no prediction.")
    final_residual = last_pred["pred_residual"]
    refined = apply_residual_to_prior(prior, final_residual, model.eps)
    return {
        "pred_residual": final_residual,
        "pred_delta_dolp": final_residual[:, 0:1],
        "pred_delta_angle": final_residual[:, 1:2],
        "refined": refined,
    }
