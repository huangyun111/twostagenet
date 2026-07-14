"""Render the formal HAMMER Version E comparison as a paper-style PNG table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render HAMMER Version E result table.")
    parser.add_argument("--version-e-summary", type=Path, required=True)
    parser.add_argument("--version-d-summary", type=Path, required=True)
    parser.add_argument("--direct-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=Path(r"C:\Windows\Fonts\msyh.ttc"))
    return parser.parse_args()


def load_font(path: Path, size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [path.with_name("msyhbd.ttc")] if bold else []
    candidates.extend([path, Path(r"C:\Windows\Fonts\simhei.ttf")])
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def stage_metrics(summary: dict[str, object], prefix: str) -> list[float]:
    return [
        float(summary[f"{prefix}dolp_mae"]),
        float(summary[f"{prefix}dolp_rmse"]),
        float(summary[f"{prefix}cos_sin_vector_error"]),
        float(summary[f"{prefix}weighted_aolp_error_deg"]),
        float(summary[f"{prefix}high_dolp_aolp_error_deg"]),
    ]


def direct_metrics(summary: dict[str, object]) -> list[float]:
    return [
        float(summary["dolp_mae"]),
        float(summary["dolp_rmse"]),
        float(summary["cos_sin_vector_error"]),
        float(summary["weighted_aolp_error_deg"]),
        float(summary["high_dolp_aolp_error_deg"]),
    ]


def format_metrics(values: list[float]) -> list[str]:
    return [f"{values[0]:.4f}", f"{values[1]:.4f}", f"{values[2]:.4f}", f"{values[3]:.2f}°", f"{values[4]:.2f}°"]


def rounded_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    *,
    fill: str = "#ECEDEF",
) -> None:
    x, y = xy
    box = draw.textbbox((x, y), text, font=font)
    draw.rounded_rectangle((x - 10, y - 5, box[2] + 10, box[3] + 5), radius=10, fill=fill)
    draw.text((x, y), text, font=font, fill="#202B3D")


def main() -> None:
    args = parse_args()
    version_e = json.loads(args.version_e_summary.read_text(encoding="utf-8"))
    version_d = json.loads(args.version_d_summary.read_text(encoding="utf-8"))
    direct = json.loads(args.direct_summary.read_text(encoding="utf-8"))

    width, height = 1536, 790
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(args.font, 35, bold=True)
    header_font = load_font(args.font, 23, bold=True)
    body_font = load_font(args.font, 22)
    best_font = load_font(args.font, 22, bold=True)
    small_font = load_font(args.font, 18)
    mono_font = load_font(args.font, 19)
    dark, muted, line = "#202B3D", "#687386", "#D9DEE6"

    draw.text((65, 42), "HAMMER：Version E 正式监督训练结果", font=title_font, fill=dark)
    draw.text(
        (65, 96),
        "验证集选择 best_val（epoch 72）；冻结测试集 1414 帧；所有指标均为逐帧算术平均",
        font=small_font,
        fill=muted,
    )

    columns = [650, 815, 990, 1185, 1395]
    draw.text((65, 150), "训练权重 → 测试集", font=header_font, fill=dark)
    headers = ["DoLP MAE\n↓", "RMSE\n↓", "vector\n↓", "weighted AoLP\n↓", "high-DoLP AoLP\n↓"]
    for x, header in zip(columns, headers):
        draw.multiline_text((x, 166), header, font=header_font, fill=dark, anchor="mm", align="center", spacing=2)
    draw.line((65, 215, 1470, 215), fill=line, width=2)

    rows = [
        ("Frozen Stage1 → HAMMER", "HAMMER 流程中的冻结低频先验，1414 帧", stage_metrics(version_e, "stage1_"), False),
        ("Version D epoch 40 → HAMMER", "验证集选择 best_val，同一 HAMMER 协议，1414 帧", stage_metrics(version_d, "stage2_"), False),
        ("Version E epoch 72 → HAMMER", "验证集选择 best_val，同一 HAMMER 协议，1414 帧", stage_metrics(version_e, "stage2_"), True),
        ("Direct U-Net++ → HAMMER", "同一 HAMMER train/val/test 协议，1414 帧", direct_metrics(direct), False),
    ]

    y = 244
    for label, detail, metrics, is_best in rows:
        if is_best:
            draw.rounded_rectangle((55, y - 13, 1480, y + 86), radius=12, fill="#F0F8F3")
        rounded_label(draw, (75, y), label, mono_font, fill="#DCEFE3" if is_best else "#ECEDEF")
        draw.text((65, y + 43), detail, font=small_font, fill=dark)
        value_font = best_font if is_best else body_font
        for x, value in zip(columns, format_metrics(metrics)):
            draw.text((x, y + 23), value, font=value_font, fill=dark, anchor="mm")
        draw.line((65, y + 89, 1470, y + 89), fill=line, width=1)
        y += 105

    note_y = y + 10
    draw.rounded_rectangle((65, note_y, 1470, note_y + 67), radius=12, fill="#F5F6F8")
    draw.text(
        (85, note_y + 19),
        "结论：在本次同协议正式测试中，Version E 的五项指标均低于 Version D 与 Direct U-Net++。",
        font=small_font,
        fill=muted,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, quality=95)
    print(args.output)


if __name__ == "__main__":
    main()
