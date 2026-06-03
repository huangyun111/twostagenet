"""Train Stage 2 residual latent diffusion for polarization refinement."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers import CLIPTextModel, CLIPTokenizer, PretrainedConfig

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.stage2_residual_dataset import Stage2ResidualDataset  # noqa: E402
from models.PolarControlnet import PolarControl  # noqa: E402
from models.utils import load_params  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Stage 2 residual latent diffusion."
    )
    parser.add_argument("--root_dir", type=str, default=str(Path.home() / "Documents"))
    parser.add_argument("--stage1_dir", type=str, default="./stage1_exports")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument("--author_pretrained_ckpt", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_stage2_residual")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=500)
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help=(
            "Maximum number of numbered step_*.pth checkpoints to keep. "
            "Set 0 to keep only last.pth."
        ),
    )
    parser.add_argument("--prompt", type=str, default="denoised polarized images")
    parser.add_argument("--vae_scaling_factor", type=float, default=0.18215)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae_encode_mode", choices=("mode", "sample"), default="mode")
    parser.add_argument(
        "--train_target",
        choices=("all", "controlnet_adapter_only"),
        default="all",
    )
    parser.add_argument("--val_root_dir", type=str, default=None)
    parser.add_argument("--val_stage1_dir", type=str, default=None)
    return parser.parse_args()


def normalize_optional_path(value: str | None) -> str | None:
    if value is None:
        return None
    if value.lower() in {"none", "null", ""}:
        return None
    return value


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


class Stage2ResidualDiffusionModel(nn.Module):
    """Predict residual latent diffusion noise from [RGB, prior, confidence]."""

    def __init__(
        self,
        unet: UNet2DConditionModel,
        controlnet: PolarControl,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        prompt: str,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.condition_adapter = nn.Sequential(
            nn.Conv2d(9, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )

    def encode_prompt(self, batch_size: int, device: torch.device) -> torch.Tensor:
        text_inputs = self.tokenizer(
            [self.prompt] * batch_size,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        return self.text_encoder(input_ids)[0]

    def forward(
        self,
        noisy_delta_z: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        encoder_hidden_states = self.encode_prompt(noisy_delta_z.shape[0], noisy_delta_z.device)
        condition_3ch = self.condition_adapter(condition)
        control_down, control_mid = self.controlnet(
            noisy_delta_z,
            timesteps,
            encoder_hidden_states,
            condition=condition_3ch,
        )
        return self.unet(
            noisy_delta_z,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=control_down,
            mid_block_additional_residual=control_mid,
        ).sample


def build_dataloader(args: argparse.Namespace) -> DataLoader:
    dataset = Stage2ResidualDataset(
        root_dir=args.root_dir,
        stage1_dir=args.stage1_dir,
        image_size=args.image_size,
    )
    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("max_train_samples must be positive or None.")
        dataset = Subset(dataset, range(min(args.max_train_samples, len(dataset))))
    if args.save_total_limit is not None and args.save_total_limit < 0:
        raise ValueError("save_total_limit must be non-negative or None.")

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_val_dataloader(args: argparse.Namespace) -> DataLoader | None:
    val_root_dir = normalize_optional_path(args.val_root_dir)
    val_stage1_dir = normalize_optional_path(args.val_stage1_dir)
    if val_root_dir is None and val_stage1_dir is None:
        return None
    if val_root_dir is None or val_stage1_dir is None:
        raise ValueError("val_root_dir and val_stage1_dir must be provided together.")

    dataset = Stage2ResidualDataset(
        root_dir=val_root_dir,
        stage1_dir=val_stage1_dir,
        image_size=args.image_size,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_models(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[AutoencoderKL, DDPMScheduler, Stage2ResidualDiffusionModel]:
    checkpoint = resolve_pretrained_model_path(args.pretrained_model_name_or_path)
    tokenizer = CLIPTokenizer.from_pretrained(checkpoint, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(checkpoint, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(checkpoint, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(checkpoint, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(checkpoint, subfolder="scheduler")

    controlnet = PolarControl(PretrainedConfig())
    load_params(controlnet, unet)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    model = Stage2ResidualDiffusionModel(
        unet=unet,
        controlnet=controlnet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=args.prompt,
    )
    load_author_checkpoint(model, normalize_optional_path(args.author_pretrained_ckpt))

    vae.to(device)
    model.to(device)
    return vae, noise_scheduler, model


def configure_train_target(
    model: Stage2ResidualDiffusionModel,
    train_target: str,
) -> None:
    model.text_encoder.requires_grad_(False)
    if train_target == "all":
        model.unet.requires_grad_(True)
        model.controlnet.requires_grad_(True)
        model.condition_adapter.requires_grad_(True)
    elif train_target == "controlnet_adapter_only":
        model.unet.requires_grad_(False)
        model.controlnet.requires_grad_(True)
        model.condition_adapter.requires_grad_(True)
    else:
        raise ValueError(f"Unknown train_target: {train_target}")


def resolve_pretrained_model_path(model_name_or_path: str) -> str:
    model_path = Path(model_name_or_path).expanduser()
    if model_path.exists():
        return str(model_path)

    local_snapshot = find_local_hf_snapshot(model_name_or_path)
    if local_snapshot is not None:
        print(f"Using local Hugging Face snapshot: {local_snapshot}", flush=True)
        return str(local_snapshot)

    return model_name_or_path


def find_local_hf_snapshot(repo_id: str) -> Path | None:
    if "/" not in repo_id:
        return None

    repo_cache_name = "models--" + repo_id.replace("/", "--")
    candidates: list[Path] = []

    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        candidates.append(Path(hf_hub_cache).expanduser() / repo_cache_name)

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home).expanduser() / "hub" / repo_cache_name)

    candidates.append(Path.home() / "Documents" / "hf_cache" / "hub" / repo_cache_name)
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub" / repo_cache_name)

    for repo_cache_dir in candidates:
        snapshot_dir = choose_complete_snapshot(repo_cache_dir / "snapshots")
        if snapshot_dir is not None:
            return snapshot_dir
    return None


def choose_complete_snapshot(snapshots_dir: Path) -> Path | None:
    if not snapshots_dir.is_dir():
        return None

    snapshots = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for snapshot in snapshots:
        if is_complete_sd_snapshot(snapshot):
            return snapshot
    return None


def is_complete_sd_snapshot(snapshot: Path) -> bool:
    required_files = (
        "vae/config.json",
        "unet/config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer_config.json",
        "scheduler/scheduler_config.json",
    )
    return all((snapshot / path).is_file() for path in required_files)


def load_author_checkpoint(
    model: Stage2ResidualDiffusionModel,
    checkpoint_path: str | None,
) -> None:
    if checkpoint_path is None:
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "unet_state_dict" in checkpoint:
        model.unet.load_state_dict(
            strip_module_prefix(checkpoint["unet_state_dict"]),
            strict=False,
        )
    if "controlnet_state_dict" in checkpoint:
        model.controlnet.load_state_dict(
            strip_module_prefix(checkpoint["controlnet_state_dict"]),
            strict=False,
        )
    if "condition_adapter_state_dict" in checkpoint:
        model.condition_adapter.load_state_dict(
            strip_module_prefix(checkpoint["condition_adapter_state_dict"]),
            strict=False,
        )
    print(f"Loaded author checkpoint: {checkpoint_path}")


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def polar_to_vae_input(polar: torch.Tensor) -> torch.Tensor:
    polar_vae = polar.clone()
    polar_vae[:, 0:1] = polar_vae[:, 0:1] * 2.0 - 1.0
    return polar_vae


def compute_residual_latents(
    vae: AutoencoderKL,
    polar_gt: torch.Tensor,
    prior: torch.Tensor,
    vae_scaling_factor: float,
    vae_encode_mode: str = "mode",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    polar_gt_vae = polar_to_vae_input(polar_gt)
    prior_vae = polar_to_vae_input(prior)

    with torch.no_grad():
        z_gt = encode_vae_latents(vae, polar_gt_vae, vae_encode_mode, vae_scaling_factor)
        z_c = encode_vae_latents(vae, prior_vae, vae_encode_mode, vae_scaling_factor)

    delta_z = z_gt - z_c
    return z_gt, z_c, delta_z


def encode_vae_latents(
    vae: AutoencoderKL,
    vae_input: torch.Tensor,
    vae_encode_mode: str,
    vae_scaling_factor: float,
) -> torch.Tensor:
    latent_dist = vae.encode(vae_input).latent_dist
    if vae_encode_mode == "mode":
        latents = latent_dist.mode()
    elif vae_encode_mode == "sample":
        latents = latent_dist.sample()
    else:
        raise ValueError(f"Unknown vae_encode_mode: {vae_encode_mode}")
    return latents * vae_scaling_factor


def print_shape_summary(
    batch: dict[str, torch.Tensor | list[str]],
    delta_z: torch.Tensor,
    noisy_delta_z: torch.Tensor,
) -> None:
    print("First batch shapes:", flush=True)
    for key in ("rgb", "polar_gt", "prior", "confidence"):
        value = batch[key]
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {tuple(value.shape)}", flush=True)
    print(f"  delta_z: {tuple(delta_z.shape)}", flush=True)
    print(f"  noisy_delta_z: {tuple(noisy_delta_z.shape)}", flush=True)


def save_checkpoint(
    save_dir: Path,
    epoch: int,
    global_step: int,
    model: Stage2ResidualDiffusionModel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    filename: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "unet_state_dict": model.unet.state_dict(),
        "controlnet_state_dict": model.controlnet.state_dict(),
        "condition_adapter_state_dict": model.condition_adapter.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(checkpoint, save_dir / filename)


def prune_step_checkpoints(save_dir: Path, save_total_limit: int | None) -> None:
    if save_total_limit is None:
        return

    step_checkpoints = sorted(save_dir.glob("step_*.pth"))
    while len(step_checkpoints) > save_total_limit:
        old_checkpoint = step_checkpoints.pop(0)
        old_checkpoint.unlink()
        print(f"Deleted old checkpoint: {old_checkpoint}", flush=True)


def load_resume_checkpoint(
    model: Stage2ResidualDiffusionModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    checkpoint_path: str | None,
) -> tuple[int, int]:
    checkpoint_path = normalize_optional_path(checkpoint_path)
    if checkpoint_path is None:
        return 0, 0

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.unet.load_state_dict(strip_module_prefix(checkpoint["unet_state_dict"]))
    model.controlnet.load_state_dict(strip_module_prefix(checkpoint["controlnet_state_dict"]))
    model.condition_adapter.load_state_dict(
        strip_module_prefix(checkpoint["condition_adapter_state_dict"])
    )
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)

    global_step = int(checkpoint.get("global_step", 0))
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    print(
        f"Resumed training from {checkpoint_path}: "
        f"start_epoch={start_epoch}, global_step={global_step}",
        flush=True,
    )
    return start_epoch, global_step


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def compute_noise_prediction_loss(
    batch: dict[str, torch.Tensor | list[str]],
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    model: Stage2ResidualDiffusionModel,
    device: torch.device,
    vae_scaling_factor: float,
    vae_encode_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = batch["rgb"].to(device, non_blocking=True)
    polar_gt = batch["polar_gt"].to(device, non_blocking=True)
    prior = batch["prior"].to(device, non_blocking=True)
    confidence = batch["confidence"].to(device, non_blocking=True)

    _, _, delta_z = compute_residual_latents(
        vae=vae,
        polar_gt=polar_gt,
        prior=prior,
        vae_scaling_factor=vae_scaling_factor,
        vae_encode_mode=vae_encode_mode,
    )

    noise = torch.randn_like(delta_z)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (delta_z.shape[0],),
        device=device,
    ).long()
    noisy_delta_z = noise_scheduler.add_noise(delta_z, noise, timesteps)

    condition = torch.cat([rgb, prior, confidence], dim=1)
    noise_pred = model(noisy_delta_z, timesteps, condition)
    loss = F.mse_loss(noise_pred.float(), noise.float())
    return loss, delta_z, noisy_delta_z


def evaluate(
    dataloader: DataLoader,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    model: Stage2ResidualDiffusionModel,
    device: torch.device,
    vae_scaling_factor: float,
    vae_encode_mode: str,
) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            loss, _, _ = compute_noise_prediction_loss(
                batch=batch,
                vae=vae,
                noise_scheduler=noise_scheduler,
                model=model,
                device=device,
                vae_scaling_factor=vae_scaling_factor,
                vae_encode_mode=vae_encode_mode,
            )
            total_loss += float(loss.detach())
    if was_training:
        model.train()
        model.text_encoder.eval()
    return total_loss / max(len(dataloader), 1)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)

    dataloader = build_dataloader(args)
    val_dataloader = build_val_dataloader(args)
    vae, noise_scheduler, model = build_models(args, device)
    configure_train_target(model, args.train_target)
    model.train()
    model.text_encoder.eval()

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-3,
        eps=1e-8,
    )

    start_epoch, global_step = load_resume_checkpoint(model, optimizer, device, args.resume)
    printed_shapes = False
    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0.0
        for batch_index, batch in enumerate(dataloader):
            loss, delta_z, noisy_delta_z = compute_noise_prediction_loss(
                batch=batch,
                vae=vae,
                noise_scheduler=noise_scheduler,
                model=model,
                device=device,
                vae_scaling_factor=args.vae_scaling_factor,
                vae_encode_mode=args.vae_encode_mode,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if not printed_shapes:
                rgb = batch["rgb"].to(device, non_blocking=True)
                polar_gt = batch["polar_gt"].to(device, non_blocking=True)
                prior = batch["prior"].to(device, non_blocking=True)
                confidence = batch["confidence"].to(device, non_blocking=True)
                shape_batch = {
                    "rgb": rgb,
                    "polar_gt": polar_gt,
                    "prior": prior,
                    "confidence": confidence,
                }
                print_shape_summary(shape_batch, delta_z, noisy_delta_z)
                printed_shapes = True

            global_step += 1
            epoch_loss += float(loss.detach())

            if args.log_freq > 0 and global_step % args.log_freq == 0:
                print(
                    f"epoch {epoch + 1}/{args.num_epochs} "
                    f"step {global_step} "
                    f"batch {batch_index + 1}/{len(dataloader)} "
                    f"loss {float(loss.detach()):.6f}",
                    flush=True,
                )

            if args.save_freq > 0 and global_step % args.save_freq == 0:
                save_checkpoint(
                    save_dir=save_dir,
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    filename="last.pth",
                )
                if args.save_total_limit != 0:
                    save_checkpoint(
                        save_dir=save_dir,
                        epoch=epoch,
                        global_step=global_step,
                        model=model,
                        optimizer=optimizer,
                        args=args,
                        filename=f"step_{global_step:06d}.pth",
                    )
                    prune_step_checkpoints(save_dir, args.save_total_limit)

        avg_loss = epoch_loss / max(len(dataloader), 1)
        val_avg_loss = None
        if val_dataloader is not None:
            val_avg_loss = evaluate(
                dataloader=val_dataloader,
                vae=vae,
                noise_scheduler=noise_scheduler,
                model=model,
                device=device,
                vae_scaling_factor=args.vae_scaling_factor,
                vae_encode_mode=args.vae_encode_mode,
            )
            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                save_checkpoint(
                    save_dir=save_dir,
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    filename="best_val.pth",
                )
        if val_avg_loss is not None:
            print(
                f"epoch {epoch + 1}/{args.num_epochs} "
                f"train_avg_loss {avg_loss:.6f} "
                f"val_avg_loss {val_avg_loss:.6f}",
                flush=True,
            )
        else:
            print(
                f"epoch {epoch + 1}/{args.num_epochs} "
                f"train_avg_loss {avg_loss:.6f}",
                flush=True,
            )

    save_checkpoint(
        save_dir=save_dir,
        epoch=args.num_epochs - 1,
        global_step=global_step,
        model=model,
        optimizer=optimizer,
        args=args,
        filename="last.pth",
    )
    print(f"Saved final checkpoint to {save_dir / 'last.pth'}", flush=True)


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
