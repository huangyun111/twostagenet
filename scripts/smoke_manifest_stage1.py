"""Smoke test: Stage1 manifest dataloader (13168 clean split) end-to-end.

Verifies that the 13168 ``assembly_manifest.json`` path loads, has correct
channels/normalisation, and runs a few real train steps + one eval frame
through PolarPriorNet + Stage1PriorLoss. CPU-friendly (default --device cpu).

Run (dl114 PA env):
  /home/hy/miniconda3/envs/PA/bin/python scripts/smoke_manifest_stage1.py \
    --manifest ~/Documents/hy_FNdataset/dataset/assembly_manifest.json \
    --device cpu --steps 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from datasets.manifest_dataset import Stage1ManifestDataset
from losses.stage1_prior_loss import Stage1PriorLoss
from models.polar_prior_net import PolarPriorNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--dataset_root", type=str, default="")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--image_size", type=int, default=256, help="resize for smoke speed")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def describe(name: str, t: torch.Tensor) -> str:
    return f"{name}: shape={tuple(t.shape)} min={t.min():.4f} max={t.max():.4f} mean={t.float().mean():.4f}"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    root = args.dataset_root or None

    train_ds = Stage1ManifestDataset(
        manifest_path=args.manifest, dataset_root=root, split="train",
        image_size=args.image_size, augment=True,
    )
    val_ds = Stage1ManifestDataset(
        manifest_path=args.manifest, dataset_root=root, split="val",
        image_size=args.image_size, augment=False,
    )
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )

    # --- channel / normalisation sanity on one batch ---
    batch = next(iter(loader))
    rgb, polar = batch["rgb"], batch["polar"]
    print("[sanity]", describe("rgb(S0->3ch)", rgb))
    print("[sanity]", describe("polar.DoLP", polar[:, 0]))
    print("[sanity]", describe("polar.cos2", polar[:, 1]))
    print("[sanity]", describe("polar.sin2", polar[:, 2]))
    assert rgb.shape[1] == 3 and polar.shape[1] == 3, "expected 3ch rgb + 3ch polar"
    assert rgb.min() >= -1.001 and rgb.max() <= 1.001, "rgb must be in [-1,1]"
    assert polar[:, 0].min() >= -1e-4 and polar[:, 0].max() <= 1.0001, "DoLP in [0,1]"
    h, w = rgb.shape[-2:]
    assert h % 32 == 0 and w % 32 == 0, f"H,W must be /32, got {h}x{w}"
    print(f"[sanity] H,W={h}x{w} divisible-by-32 OK")

    # --- a few real train steps ---
    model = PolarPriorNet(encoder_weights=None).to(device)
    loss_fn = Stage1PriorLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    it = iter(loader)
    for step in range(args.steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        rgb = b["rgb"].to(device)
        target = b["polar"].to(device)
        opt.zero_grad(set_to_none=True)
        pred = model(rgb)
        loss_dict = loss_fn(pred, target)
        loss_dict["loss"].backward()
        opt.step()
        print(f"[train] step {step+1}/{args.steps} loss={float(loss_dict['loss']):.5f} "
              f"dolp={float(loss_dict['loss_dolp']):.5f} aolp={float(loss_dict['loss_aolp']):.5f}")

    # --- eval one val frame ---
    model.eval()
    v = val_ds[0]
    with torch.no_grad():
        pred = model(v["rgb"].unsqueeze(0).to(device))
    pp = pred["polar_prior"][0].cpu()
    gt = v["polar"]
    dolp_mae = (pp[0] - gt[0]).abs().mean()
    print(f"[eval] frame={v['name']} pred_polar shape={tuple(pp.shape)} "
          f"DoLP[min={pp[0].min():.4f},max={pp[0].max():.4f}] DoLP_MAE_vs_gt={float(dolp_mae):.4f}")
    print("[OK] Stage1 manifest smoke passed.")


if __name__ == "__main__":
    main()
