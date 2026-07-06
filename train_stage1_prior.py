"""Train Stage 1 coarse polarization prior generator."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from datasets.manifest_dataset import Stage1ManifestDataset
from datasets.stage1_prior_dataset import Stage1PriorDataset
from losses.stage1_prior_loss import Stage1PriorLoss
from models.polar_prior_net import PolarPriorNet
from utils.ddp_utils import (
    barrier,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    setup_distributed,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Stage 1 RGB -> coarse polarization prior model."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=r"D:\PolarAnything\data\PolarAnything_subset",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Path to assembly_manifest.json. When set, the 13168 clean-split "
        "manifest dataset is used instead of --root_dir.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="",
        help="Directory the manifest frame paths are relative to. Defaults to "
        "the manifest's grandparent dir.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "val", "test", "all"),
        help="Which manifest split to train on (manifest mode only).",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="val",
        choices=("none", "train", "val", "test"),
        help="Manifest split used for val checkpoint selection (manifest mode "
        "only). 'none' disables validation and falls back to train-loss best.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Cap the number of training samples (for smoke tests).",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="Cap the number of validation samples (for smoke tests).",
    )
    parser.add_argument("--save_dir", type=str, default="./checkpoints_stage1_prior")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument(
        "--preprocess_mode",
        choices=("resize256", "official_train"),
        default="resize256",
    )
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument(
        "--normalize_mode",
        choices=("fixed255", "image_max"),
        default="fixed255",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--encoder_weights", type=str, default=None)
    parser.add_argument("--vis_freq", type=int, default=5)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distributed", action="store_true")
    return parser.parse_args()


def normalize_encoder_weights(value: str | None) -> str | None:
    if value is None:
        return None
    if value.lower() in {"none", "null", ""}:
        return None
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(
    model: PolarPriorNet,
    lr: float,
    encoder_lr: float,
) -> torch.optim.Optimizer:
    """Use a lower LR for the SMP encoder and the main LR for other params."""
    encoder_param_ids: set[int] = set()
    if hasattr(model, "net") and hasattr(model.net, "encoder"):
        encoder_param_ids = {id(param) for param in model.net.encoder.parameters()}

    encoder_params = []
    other_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in encoder_param_ids:
            encoder_params.append(param)
        else:
            other_params.append(param)

    if not encoder_params:
        # Fallback for unexpected model structures where encoder cannot be identified.
        return torch.optim.AdamW(model.parameters(), lr=lr)

    param_groups: list[dict[str, Any]] = [{"params": encoder_params, "lr": encoder_lr}]
    if other_params:
        param_groups.append({"params": other_params, "lr": lr})
    return torch.optim.AdamW(param_groups)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_loss: float,
    args: argparse.Namespace,
) -> None:
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint = {
        "epoch": epoch,
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    else:
        state_dict = checkpoint
    model.load_state_dict(strip_module_prefix(state_dict))
    if isinstance(checkpoint, dict):
        optimizer_state = checkpoint.get("optimizer", checkpoint.get("optimizer_state_dict"))
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            move_optimizer_state_to_device(optimizer, device)
    start_epoch = int(checkpoint.get("epoch", 0)) + 1 if isinstance(checkpoint, dict) else 1
    best_loss = float(checkpoint.get("best_loss", float("inf"))) if isinstance(checkpoint, dict) else float("inf")
    return start_epoch, best_loss


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def append_log(log_path: Path, message: str) -> None:
    print(message)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def write_config(path: Path, args: argparse.Namespace) -> None:
    payload = {
        **vars(args),
        "crop_strategy": (
            "official_train resizes the short side to crop_size only when needed, "
            "then random-crops train samples to crop_size."
            if args.preprocess_mode == "official_train"
            else "resize256 uses image_size fixed resize."
        ),
        "target_format": "[DoLP, cos(2AoLP), sin(2AoLP)] with DoLP in [0,1]",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tensor_to_numpy_image(rgb: torch.Tensor) -> np.ndarray:
    image = rgb.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip((image + 1.0) * 0.5, 0.0, 1.0)


def to_display_map(tensor: torch.Tensor, value_range: tuple[float, float]) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    min_value, max_value = value_range
    array = (array - min_value) / (max_value - min_value)
    return np.clip(array, 0.0, 1.0)


def save_visualization(
    rgb: torch.Tensor,
    target: torch.Tensor,
    pred_dict: dict[str, torch.Tensor],
    path: Path,
) -> None:
    rgb_image = tensor_to_numpy_image(rgb[0])
    target_polar = target[0].detach().cpu()
    pred_polar = pred_dict["polar_prior"][0].detach().cpu()
    confidence = pred_dict["confidence"][0].detach().cpu()
    aolp_reliability = torch.clamp((target_polar[0] - 0.03) / (0.15 - 0.03), 0.0, 1.0)
    confidence_aolp = confidence[1:3].mean(dim=0)

    panels = [
        ("RGB", rgb_image, None),
        ("Target DoLP", target_polar[0].numpy(), (0.0, 1.0)),
        ("Pred DoLP", pred_polar[0].numpy(), (0.0, 1.0)),
        ("Abs DoLP Error", torch.abs(pred_polar[0] - target_polar[0]).numpy(), (0.0, 1.0)),
        ("AoLP Reliability", aolp_reliability.numpy(), (0.0, 1.0)),
        ("Confidence DoLP", confidence[0].numpy(), (0.0, 1.0)),
        ("Confidence AoLP", confidence_aolp.numpy(), (0.0, 1.0)),
        ("Target cos2", to_display_map(target_polar[1], (-1.0, 1.0)), (0.0, 1.0)),
        ("Pred cos2", to_display_map(pred_polar[1], (-1.0, 1.0)), (0.0, 1.0)),
        ("Target sin2", to_display_map(target_polar[2], (-1.0, 1.0)), (0.0, 1.0)),
        ("Pred sin2", to_display_map(pred_polar[2], (-1.0, 1.0)), (0.0, 1.0)),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for axis, (title, image, value_range) in zip(axes.flat, panels):
        if image.ndim == 3:
            axis.imshow(image)
        else:
            vmin, vmax = value_range if value_range is not None else (None, None)
            axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    for axis in list(axes.flat)[len(panels):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def format_lrs(optimizer: torch.optim.Optimizer) -> str:
    return ",".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: Stage1PriorLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, torch.Tensor] | None]:
    model.train()
    sums = {
        "loss": 0.0,
        "loss_dolp": 0.0,
        "loss_aolp": 0.0,
        "loss_conf": 0.0,
        "loss_unc": 0.0,
        "loss_lowfreq": 0.0,
        "loss_edge": 0.0,
        "mean_aolp_reliability": 0.0,
        "mean_conf_dolp": 0.0,
        "mean_conf_aolp": 0.0,
    }
    num_samples = 0
    vis_data = None

    for batch in dataloader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        target = batch["polar"].to(device, non_blocking=True)
        batch_size = rgb.shape[0]

        optimizer.zero_grad(set_to_none=True)
        pred_dict = model(rgb)
        loss_dict = loss_fn(pred_dict, target)
        loss_dict["loss"].backward()
        optimizer.step()

        num_samples += batch_size
        for key in sums:
            sums[key] += float(loss_dict[key].detach()) * batch_size

        if vis_data is None:
            vis_data = {
                "rgb": rgb.detach().cpu(),
                "target": target.detach().cpu(),
                "pred_dict": {
                    "polar_prior": pred_dict["polar_prior"].detach().cpu(),
                    "confidence": pred_dict["confidence"].detach().cpu(),
                },
            }

    if num_samples == 0:
        raise RuntimeError("Training dataloader produced no samples.")
    averages = {key: value / num_samples for key, value in sums.items()}
    return averages, vis_data


def evaluate(
    model: torch.nn.Module,
    loss_fn: Stage1PriorLoss,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Run a no-grad pass over the val split and average the loss terms."""
    model.eval()
    sums = {
        "loss": 0.0,
        "loss_dolp": 0.0,
        "loss_aolp": 0.0,
        "loss_conf": 0.0,
        "loss_unc": 0.0,
        "loss_lowfreq": 0.0,
        "loss_edge": 0.0,
        "mean_aolp_reliability": 0.0,
        "mean_conf_dolp": 0.0,
        "mean_conf_aolp": 0.0,
    }
    num_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            target = batch["polar"].to(device, non_blocking=True)
            batch_size = rgb.shape[0]
            pred_dict = model(rgb)
            loss_dict = loss_fn(pred_dict, target)
            num_samples += batch_size
            for key in sums:
                sums[key] += float(loss_dict[key].detach()) * batch_size
    if num_samples == 0:
        raise RuntimeError("Validation dataloader produced no samples.")
    return {key: value / num_samples for key, value in sums.items()}


def main() -> None:
    args = parse_args()
    args.encoder_weights = normalize_encoder_weights(args.encoder_weights)
    set_seed(args.seed)
    device, distributed = setup_distributed(args)
    rank = get_rank()
    world_size = get_world_size()

    save_dir = Path(args.save_dir)
    vis_dir = save_dir / "vis"
    if is_main_process():
        save_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)
        write_config(save_dir / "config.json", args)
    barrier()
    log_path = save_dir / "train_log.txt"

    if args.manifest:
        dataset = Stage1ManifestDataset(
            manifest_path=args.manifest,
            dataset_root=args.dataset_root or None,
            split=args.split,
            image_size=args.image_size if args.preprocess_mode == "resize256" else None,
            crop_size=args.crop_size if args.preprocess_mode == "official_train" else 0,
            random_crop=args.preprocess_mode == "official_train",
            augment=args.preprocess_mode == "official_train",
        )
    else:
        dataset = Stage1PriorDataset(
            root_dir=args.root_dir,
            image_size=args.image_size,
            preprocess_mode=args.preprocess_mode,
            crop_size=args.crop_size,
            normalize_mode=args.normalize_mode,
            random_crop=args.preprocess_mode == "official_train",
        )
    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("max_train_samples must be positive or None.")
        dataset = Subset(dataset, range(min(args.max_train_samples, len(dataset))))
    train_sampler = (
        DistributedSampler(dataset, shuffle=True, drop_last=False)
        if distributed
        else None
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Validation loader for clean val-based checkpoint selection (manifest mode
    # only). Evaluated on the main process only; no random crop / augmentation so
    # full 480x480 frames are scored deterministically.
    val_loader = None
    val_size = 0
    if args.manifest and args.val_split != "none" and is_main_process():
        val_dataset = Stage1ManifestDataset(
            manifest_path=args.manifest,
            dataset_root=args.dataset_root or None,
            split=args.val_split,
            image_size=args.image_size if args.preprocess_mode == "resize256" else None,
            crop_size=args.crop_size if args.preprocess_mode == "official_train" else 0,
            random_crop=False,
            augment=False,
        )
        if args.max_val_samples is not None:
            if args.max_val_samples <= 0:
                raise ValueError("max_val_samples must be positive or None.")
            val_dataset = Subset(
                val_dataset, range(min(args.max_val_samples, len(val_dataset)))
            )
        val_size = len(val_dataset)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    model = PolarPriorNet(encoder_weights=args.encoder_weights).to(device)
    loss_fn = Stage1PriorLoss().to(device)
    optimizer = build_optimizer(model, args.lr, args.encoder_lr)

    start_epoch = 1
    best_loss = float("inf")
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_loss = load_checkpoint(args.resume, model, optimizer, device)

    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )

    try:
        if is_main_process():
            append_log(
                log_path,
                f"Start training on {device}, samples={len(dataset)}, "
                f"batch_size_per_gpu={args.batch_size}, world_size={world_size}, "
                f"rank={rank}, local_rank={args.local_rank}, "
                f"effective_batch_size={args.batch_size * world_size}, "
                f"distributed={distributed}, lrs={format_lrs(optimizer)}, "
                f"preprocess_mode={args.preprocess_mode}, crop_size={args.crop_size}, "
                f"normalize_mode={args.normalize_mode}, root_dir={args.root_dir}, "
                f"num_epochs={args.num_epochs}, lr={args.lr}, encoder_lr={args.encoder_lr}, "
                f"encoder_weights={args.encoder_weights}",
            )

        for epoch in range(start_epoch, args.num_epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            metrics, vis_data = train_one_epoch(model, loss_fn, dataloader, optimizer, device)

            if is_main_process():
                val_metrics = None
                if val_loader is not None:
                    eval_model = model.module if hasattr(model, "module") else model
                    val_metrics = evaluate(eval_model, loss_fn, val_loader, device)

                message = (
                    f"epoch {epoch}/{args.num_epochs} | "
                    f"loss={metrics['loss']:.6f} | "
                    f"loss_dolp={metrics['loss_dolp']:.6f} | "
                    f"loss_aolp={metrics['loss_aolp']:.6f} | "
                    f"loss_conf={metrics['loss_conf']:.6f} | "
                    f"loss_unc={metrics['loss_unc']:.6f} | "
                    f"loss_lowfreq={metrics['loss_lowfreq']:.6f} | "
                    f"loss_edge={metrics['loss_edge']:.6f} | "
                    f"aolp_rel={metrics['mean_aolp_reliability']:.6f} | "
                    f"conf_dolp={metrics['mean_conf_dolp']:.6f} | "
                    f"conf_aolp={metrics['mean_conf_aolp']:.6f} | "
                    f"lr={format_lrs(optimizer)}"
                )
                if val_metrics is not None:
                    message += (
                        f" || val_loss={val_metrics['loss']:.6f} | "
                        f"val_loss_dolp={val_metrics['loss_dolp']:.6f} | "
                        f"val_loss_aolp={val_metrics['loss_aolp']:.6f}"
                    )
                append_log(log_path, message)

                # Clean discipline: best_val.pth is selected by the val split.
                # best.pth (train loss) is kept for backward compatibility only.
                if val_metrics is not None and val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    save_checkpoint(
                        save_dir / "best_val.pth", epoch, model, optimizer, best_val_loss, args
                    )

                if metrics["loss"] < best_loss:
                    best_loss = metrics["loss"]
                    save_checkpoint(save_dir / "best.pth", epoch, model, optimizer, best_loss, args)

                save_checkpoint(save_dir / "last.pth", epoch, model, optimizer, best_loss, args)

                if args.save_freq > 0 and epoch % args.save_freq == 0:
                    save_checkpoint(
                        save_dir / f"epoch_{epoch:03d}.pth",
                        epoch,
                        model,
                        optimizer,
                        best_loss,
                        args,
                    )

                if args.vis_freq > 0 and epoch % args.vis_freq == 0 and vis_data is not None:
                    save_visualization(
                        vis_data["rgb"],
                        vis_data["target"],
                        vis_data["pred_dict"],
                        vis_dir / f"epoch_{epoch:03d}.png",
                    )
            barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
