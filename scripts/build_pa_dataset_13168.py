"""Build a PolarAnything-format dataset from the 13168 final_new512 clean split.

PA's PolarDataset reads two FLAT png directories (matched by filename) and:
  * conditioning/rgb dir  -> normalized by ``/65535`` (so it MUST be uint16)
  * polarization/GT  dir  -> normalized by ``image_max`` (uint8 fine)
and distinguishes train/test only by pointing at different directories (no
split field). So we materialize, per split, two dirs:

  <out>/<split>/RGB/<scene>_<frame>.png                 # condition: S0 -> uint16 3ch
  <out>/<split>/Polarization_Encoding/<scene>_<frame>.png  # GT: copy of the 13168 uint8 PNG

- condition: S0 .npy is float32 in [0,1] (single channel). We map to uint16 via
  round(clip(s0,0,1)*65535) and replicate to 3 channels, so PA's ``/65535``
  recovers [0,1] -> [-1,1]. (Storing uint8 here would collapse to ~-1.)
- GT: the 13168 Polarization_Encoding PNG is uint8 [DoLP,cos2,sin2]; we copy it
  byte-for-byte (zero re-encode, no channel/colorspace risk).
- The 16 source-corrupted polar_target PNGs (all train) are excluded, matching
  the twostagenet manifest datasets.

Native 480x480 is preserved; any resolution adaptation for PA's UNet happens in
the training/inference step, not here.
"""

from __future__ import annotations

import argparse
import json
import shutil
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

# Same 16 corrupted train-split polar_target PNGs excluded in datasets/manifest_dataset.py
_EXCLUDED_POLAR_TARGETS = {
    "dataset/5/Polarization_Encoding/polar_%05d.png" % i for i in range(26, 40)
} | {
    "dataset/10/Polarization_Encoding/polar_00026.png",
    "dataset/11/Polarization_Encoding/polar_00021.png",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build PA-format dataset from 13168 clean split.")
    p.add_argument("--manifest", required=True, help="assembly_manifest.json")
    p.add_argument("--dataset_root", default="", help="root the manifest paths are relative to "
                   "(default: manifest parent's parent)")
    p.add_argument("--out_dir", required=True, help="output root; creates <split>/{RGB,Polarization_Encoding}")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="per-split cap for a smoke run (0 = all)")
    return p.parse_args()


def s0_npy_to_uint16_png(s0_path: Path, out_path: Path) -> None:
    arr = np.load(s0_path)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    u16 = np.rint(arr * 65535.0).astype(np.uint16)
    rgb = np.stack([u16, u16, u16], axis=-1)  # gray -> 3ch (BGR==RGB for gray)
    if not cv2.imwrite(str(out_path), rgb):
        raise IOError(f"failed to write {out_path}")


def _process_one(task: tuple[str, str, str, str]) -> tuple[bool, str]:
    s0_path, polar_src, rgb_out, gt_out = task
    try:
        s0_npy_to_uint16_png(Path(s0_path), Path(rgb_out))
        shutil.copyfile(polar_src, gt_out)  # GT: byte-for-byte copy of uint8 PNG
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{s0_path}: {exc}"


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    dataset_root = Path(args.dataset_root) if args.dataset_root else manifest_path.parent.parent
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    frames = manifest.get("frames") or manifest.get("items") or []

    out_root = Path(args.out_dir)
    summary: dict[str, int] = {}
    for split in args.splits:
        sel = [f for f in frames if f.get("split") == split
               and f.get("polar_target") not in _EXCLUDED_POLAR_TARGETS]
        if args.limit:
            sel = sel[: args.limit]
        rgb_dir = out_root / split / "RGB"
        gt_dir = out_root / split / "Polarization_Encoding"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        for f in sel:
            name = f"{f['scene']}_{f['frame']}.png"
            tasks.append((
                str(dataset_root / f["s0"]),
                str(dataset_root / f["polar_target"]),
                str(rgb_dir / name),
                str(gt_dir / name),
            ))

        ok = 0
        errs: list[str] = []
        with Pool(args.workers) as pool:
            for good, msg in pool.imap_unordered(_process_one, tasks, chunksize=16):
                if good:
                    ok += 1
                else:
                    errs.append(msg)
        summary[split] = ok
        print(f"[{split}] wrote {ok}/{len(tasks)} pairs -> {out_root / split}", flush=True)
        for e in errs[:10]:
            print(f"  ERR {e}", flush=True)
        if errs:
            print(f"  ... {len(errs)} errors total", flush=True)

    (out_root / "build_summary.txt").write_text(
        "PA-format 13168 dataset\n"
        + f"manifest: {manifest_path}\n"
        + f"dataset_root: {dataset_root}\n"
        + "condition(RGB): S0 npy -> uint16 3ch PNG (/65535 -> [0,1])\n"
        + "GT(Polarization_Encoding): byte-copy of uint8 [DoLP,cos2,sin2] PNG\n"
        + "excluded 16 corrupted train polar PNGs\n"
        + "".join(f"{k}: {v}\n" for k, v in summary.items()),
        encoding="utf-8",
    )
    print("Done.", {k: v for k, v in summary.items()}, flush=True)


if __name__ == "__main__":
    main()
