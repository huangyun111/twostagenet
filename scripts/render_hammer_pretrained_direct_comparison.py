"""Render the HAMMER comparison with a 13168-pretrained Direct U-Net++ baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the fair-pretraining HAMMER table.")
    parser.add_argument("--version-e-summary", type=Path, required=True)
    parser.add_argument("--version-d-summary", type=Path, required=True)
    parser.add_argument("--direct-scratch-summary", type=Path, required=True)
    parser.add_argument("--direct-pretrained-summary", type=Path, required=True)
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


def format_value(value: float, column: int) -> str:
    if column < 3:
        return f"{value:.4f}"
    return f"{value:.2f}°"


def rounded_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    *,
    fill: str,
) -> None:
    x, y = xy
    box = draw.textbbox((x, y), text, font=font)
    draw.rounded_rectangle((x - 10, y - 5, box[2] + 10, box[3] + 5), radius=10, fill=fill)
    draw.text((x, y), text, font=font, fill="#202B3D")


def main() -> None:
    args = parse_args()
    version_e = json.loads(args.version_e_summary.read_text(encoding="utf-8"))
    version_d = json.loads(args.version_d_summary.read_text(encoding="utf-8"))
    direct_scratch = json.loads(args.direct_scratch_summary.read_text(encoding="utf-8"))
    direct_pretrained = json.loads(args.direct_pretrained_summary.read_text(encoding="utf-8"))

    rows = [
        (
            "13168 Stage1（冻结）→ HAMMER",
            "13168 训练的 Stage1 在 HAMMER 上 zero-shot，1414 帧",
            stage_metrics(version_e, "stage1_"),
            "plain",
        ),
        (
            "Version D epoch 40 → HAMMER",
            "13168 Stage1 冻结；HAMMER 监督训练 Stage2，1414 帧",
            stage_metrics(version_d, "stage2_"),
            "plain",
        ),
        (
            "Version E epoch 72 → HAMMER",
            "13168 Stage1 冻结；HAMMER 监督训练 Stage2，1414 帧",
            stage_metrics(version_e, "stage2_"),
            "version_e",
        ),
        (
            "Direct U-Net++ scratch epoch 76 → HAMMER",
            "仅使用 HAMMER train/val 从头训练，1414 帧",
            direct_metrics(direct_scratch),
            "plain",
        ),
        (
            "Direct U-Net++ 13168→HAMMER epoch 70",
            "13168 预训练后用 HAMMER train/val 微调，1414 帧",
            direct_metrics(direct_pretrained),
            "pretrained",
        ),
    ]
    best_values = [min(row[2][column] for row in rows) for column in range(5)]

    width, height = 1536, 895
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(args.font, 35, bold=True)
    header_font = load_font(args.font, 23, bold=True)
    body_font = load_font(args.font, 22)
    best_font = load_font(args.font, 22, bold=True)
    small_font = load_font(args.font, 18)
    mono_font = load_font(args.font, 18)
    dark, muted, line = "#202B3D", "#687386", "#D9DEE6"

    draw.text((65, 42), "HAMMER：13168 预训练公平对照正式结果", font=title_font, fill=dark)
    draw.text(
        (65, 96),
        "验证集选择 best_val；冻结测试集 1414 帧；所有指标均为逐帧算术平均",
        font=small_font,
        fill=muted,
    )
    columns = [650, 815, 990, 1185, 1395]
    draw.text((65, 150), "训练权重 → 测试集", font=header_font, fill=dark)
    headers = ["DoLP MAE\n↓", "RMSE\n↓", "vector\n↓", "weighted AoLP\n↓", "high-DoLP AoLP\n↓"]
    for x, header in zip(columns, headers):
        draw.multiline_text((x, 166), header, font=header_font, fill=dark, anchor="mm", align="center", spacing=2)
    draw.line((65, 215, 1470, 215), fill=line, width=2)

    y = 244
    for label, detail, metrics, style in rows:
        background = None
        label_fill = "#ECEDEF"
        if style == "version_e":
            background, label_fill = "#F0F8F3", "#DCEFE3"
        elif style == "pretrained":
            background, label_fill = "#F1F6FC", "#DDEAF8"
        if background:
            draw.rounded_rectangle((55, y - 13, 1480, y + 86), radius=12, fill=background)
        rounded_label(draw, (75, y), label, mono_font, fill=label_fill)
        draw.text((65, y + 43), detail, font=small_font, fill=dark)
        for column, (x, value) in enumerate(zip(columns, metrics)):
            font = best_font if abs(value - best_values[column]) < 1e-12 else body_font
            draw.text((x, y + 23), format_value(value, column), font=font, fill=dark, anchor="mm")
        draw.line((65, y + 89, 1470, y + 89), fill=line, width=1)
        y += 105

    note_y = y + 10
    draw.rounded_rectangle((65, note_y, 1470, note_y + 67), radius=12, fill="#F5F6F8")
    draw.text(
        (85, note_y + 19),
        "结论：Version E 的 DoLP MAE、RMSE、weighted AoLP 更低；预训练 Direct U-Net++ 的 vector、high-DoLP AoLP 更低。",
        font=small_font,
        fill=muted,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, quality=95)
    print(args.output)


if __name__ == "__main__":
    main()
