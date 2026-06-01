"""Train Stage 1 coarse polarization prior generator."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.stage1_prior_dataset import Stage1PriorDataset
from losses.stage1_prior_loss import Stage1PriorLoss
from models.polar_prior_net import PolarPriorNet

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
    parser.add_argument("--save_dir", type=str, default="./checkpoints_stage1_prior")
    parser.add_argument("--image_size", type=int, default=256)
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
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
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
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_loss = float(checkpoint.get("best_loss", float("inf")))
    return start_epoch, best_loss


def append_log(log_path: Path, message: str) -> None:
    print(message)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


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


def main() -> None:
    args = parse_args()
    args.encoder_weights = normalize_encoder_weights(args.encoder_weights)
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    vis_dir = save_dir / "vis"
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train_log.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Stage1PriorDataset(
        root_dir=args.root_dir,
        image_size=args.image_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = PolarPriorNet(encoder_weights=args.encoder_weights).to(device)
    loss_fn = Stage1PriorLoss().to(device)
    optimizer = build_optimizer(model, args.lr, args.encoder_lr)

    start_epoch = 1
    best_loss = float("inf")
    if args.resume:
        start_epoch, best_loss = load_checkpoint(args.resume, model, optimizer, device)

    append_log(
        log_path,
        f"Start training on {device}, samples={len(dataset)}, "
        f"batch_size={args.batch_size}, lrs={format_lrs(optimizer)}",
    )

    for epoch in range(start_epoch, args.num_epochs + 1):
        metrics, vis_data = train_one_epoch(model, loss_fn, dataloader, optimizer, device)

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
        append_log(log_path, message)

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


if __name__ == "__main__":
    main()
