import argparse
import csv
import io
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import tifffile
from tqdm import tqdm


def is_target_tiff(zip_name: str) -> bool:
    """
    Select final_new512/<scene>/flir/**/*.tif[f],
    excluding any path containing grey.
    """
    name = zip_name.replace("\\", "/")
    low = name.lower()

    if not (low.endswith(".tiff") or low.endswith(".tif")):
        return False

    parts = low.split("/")
    if "grey" in parts:
        return False

    if "flir" not in parts:
        return False

    return True


def get_scene_and_output_name(zip_name: str):
    """
    Example:
    final_new512/161/flir/xxx.tiff
    -> scene=161, out_name=xxx.png

    If there are nested folders under flir, join them with underscores.
    """
    p = PurePosixPath(zip_name.replace("\\", "/"))
    parts = list(p.parts)

    low_parts = [x.lower() for x in parts]
    flir_idx = low_parts.index("flir")

    if flir_idx == 0:
        raise ValueError(f"Cannot infer scene from path: {zip_name}")

    scene = parts[flir_idx - 1]

    rel_after_flir = parts[flir_idx + 1:]
    stem_parts = list(PurePosixPath(*rel_after_flir).with_suffix("").parts)
    out_name = "_".join(stem_parts) + ".png"

    return scene, out_name


def read_7ch_tiff_from_zip(zf: zipfile.ZipFile, member_name: str) -> np.ndarray:
    data = zf.read(member_name)
    arr = tifffile.imread(io.BytesIO(data))
    arr = np.asarray(arr)

    # Expected common shapes:
    # H, W, 7
    # 7, H, W
    if arr.ndim != 3:
        raise ValueError(f"Unsupported TIFF shape {arr.shape} for {member_name}")

    if arr.shape[-1] >= 7:
        img = arr
    elif arr.shape[0] >= 7:
        img = np.moveaxis(arr, 0, -1)
    else:
        raise ValueError(f"Cannot find 7 channels in shape {arr.shape} for {member_name}")

    return img.astype(np.float32)


def channel_stats(img: np.ndarray):
    result = {}
    for c in range(min(img.shape[-1], 7)):
        ch = img[..., c]
        finite = np.isfinite(ch)
        if not finite.any():
            result[f"ch{c}_min"] = np.nan
            result[f"ch{c}_max"] = np.nan
            result[f"ch{c}_mean"] = np.nan
        else:
            valid = ch[finite]
            result[f"ch{c}_min"] = float(valid.min())
            result[f"ch{c}_max"] = float(valid.max())
            result[f"ch{c}_mean"] = float(valid.mean())
    return result


def encode_signed_to_01(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) + 1.0) * 0.5


def encode_01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def decide_sin_mode(sin_ch: np.ndarray) -> str:
    """
    auto rule:
    - if ch2 is already mostly within [0, 1] and has no obvious negative values,
      treat it as encoded.
    - otherwise treat it as raw sin2AoP in [-1, 1].
    """
    finite = np.isfinite(sin_ch)
    if not finite.any():
        return "encoded"

    mn = float(sin_ch[finite].min())
    mx = float(sin_ch[finite].max())

    if mn >= -0.02 and mx <= 1.02:
        return "encoded"
    return "raw"


def convert_one(img: np.ndarray, sin_mode: str):
    """
    Input 7ch:
    ch1 = DoP / DoLP
    ch2 = sin2AoP or encoded sin2AoP
    ch3 = cos2AoP raw

    Output RGB:
    R = DoLP
    G = cos2AoP encoded to [0,1]
    B = sin2AoP encoded to [0,1]
    """
    dop = img[..., 1]
    sin_ch = img[..., 2]
    cos_ch = img[..., 3]

    dop_01 = encode_01(dop)

    # ch3 你给的范围是 -0.9983 ~ 0.9954，所以按 raw cos2AoP 处理
    cos_01 = encode_signed_to_01(cos_ch)

    if sin_mode == "auto":
        actual_sin_mode = decide_sin_mode(sin_ch)
    else:
        actual_sin_mode = sin_mode

    if actual_sin_mode == "encoded":
        sin_01 = encode_01(sin_ch)
    elif actual_sin_mode == "raw":
        sin_01 = encode_signed_to_01(sin_ch)
    else:
        raise ValueError(f"Unknown sin_mode: {sin_mode}")

    out = np.stack([dop_01, cos_01, sin_01], axis=-1)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    out = np.clip(out, 0.0, 1.0)

    out_u16 = np.round(out * 65535.0).astype(np.uint16)

    return out_u16, actual_sin_mode


def inspect(zip_path: Path, limit: int):
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [x for x in zf.namelist() if is_target_tiff(x)]

        print(f"[INFO] Found target TIFF files: {len(members)}")
        print(f"[INFO] Inspect first {min(limit, len(members))} files\n")

        for name in members[:limit]:
            print("=" * 100)
            print(name)

            img = read_7ch_tiff_from_zip(zf, name)
            print("shape:", img.shape, "dtype:", img.dtype)

            stats = channel_stats(img)
            for c in range(7):
                print(
                    f"ch{c}: "
                    f"min={stats[f'ch{c}_min']:.6f}, "
                    f"max={stats[f'ch{c}_max']:.6f}, "
                    f"mean={stats[f'ch{c}_mean']:.6f}"
                )

            sin_mode = decide_sin_mode(img[..., 2])
            print(f"\n[AUTO DECISION] ch2 sin_mode = {sin_mode}")
            print("If ch2 range is around 0~1, use --sin_mode encoded")
            print("If ch2 range is around -1~1, use --sin_mode raw")


def convert(zip_path: Path, output_root: Path, sin_mode: str, limit: int, overwrite: bool):
    output_root.mkdir(parents=True, exist_ok=True)
    index_csv = output_root / "convert_index.csv"

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [x for x in zf.namelist() if is_target_tiff(x)]

        if limit is not None and limit > 0:
            members = members[:limit]

        print(f"[INFO] Total files to convert: {len(members)}")
        print(f"[INFO] Output root: {output_root}")
        print(f"[INFO] sin_mode: {sin_mode}")
        print(f"[INFO] overwrite: {overwrite}")

        with open(index_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "scene",
                "zip_member",
                "output_png",
                "height",
                "width",
                "sin_mode_used",
                "ch1_dop_min",
                "ch1_dop_max",
                "ch2_sin_min",
                "ch2_sin_max",
                "ch3_cos_min",
                "ch3_cos_max",
            ])

            for name in tqdm(members, desc="Converting"):
                try:
                    scene, out_name = get_scene_and_output_name(name)
                    out_dir = output_root / scene / "Polarization_Encoding"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / out_name

                    if out_path.exists() and not overwrite:
                        continue

                    img = read_7ch_tiff_from_zip(zf, name)
                    out_u16_rgb, sin_mode_used = convert_one(img, sin_mode)

                    # OpenCV 写 PNG 时按 BGR 处理，所以这里先 RGB -> BGR
                    out_u16_bgr = cv2.cvtColor(out_u16_rgb, cv2.COLOR_RGB2BGR)

                    ok = cv2.imwrite(str(out_path), out_u16_bgr)
                    if not ok:
                        raise RuntimeError(f"cv2.imwrite failed: {out_path}")

                    h, w = out_u16_rgb.shape[:2]
                    dop = img[..., 1]
                    sin_ch = img[..., 2]
                    cos_ch = img[..., 3]

                    writer.writerow([
                        scene,
                        name,
                        str(out_path),
                        h,
                        w,
                        sin_mode_used,
                        float(np.nanmin(dop)),
                        float(np.nanmax(dop)),
                        float(np.nanmin(sin_ch)),
                        float(np.nanmax(sin_ch)),
                        float(np.nanmin(cos_ch)),
                        float(np.nanmax(cos_ch)),
                    ])

                except Exception as e:
                    print(f"\n[ERROR] Failed: {name}")
                    print(e)

    print(f"\n[DONE] Index saved to: {index_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", type=str, required=True)
    parser.add_argument("--output_root", type=str, default=r"D:\final_new512_polar_encoding")
    parser.add_argument("--mode", type=str, choices=["inspect", "convert"], required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sin_mode", type=str, choices=["auto", "encoded", "raw"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    output_root = Path(args.output_root)

    if args.mode == "inspect":
        inspect(zip_path, limit=args.limit if args.limit > 0 else 2)
    else:
        convert(
            zip_path=zip_path,
            output_root=output_root,
            sin_mode=args.sin_mode,
            limit=args.limit,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()