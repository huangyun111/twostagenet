"""Render the formal HAMMER Version D result summary as a paper-style PNG table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render HAMMER Version D result table.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--direct_summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=Path(r"C:\Windows\Fonts\msyh.ttc"))
    return parser.parse_args()


def load_font(path: Path, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = []
    if bold:
        candidates.append(path.with_name("msyhbd.ttc"))
    candidates.extend([path, Path(r"C:\Windows\Fonts\simhei.ttf")])
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def metric_row(summary: dict[str, object], prefix: str) -> list[str]:
    return [
        f"{float(summary[f'{prefix}dolp_mae']):.4f}",
        f"{float(summary[f'{prefix}dolp_rmse']):.4f}",
        f"{float(summary[f'{prefix}cos_sin_vector_error']):.4f}",
        f"{float(summary[f'{prefix}weighted_aolp_error_deg']):.2f}°",
        f"{float(summary[f'{prefix}high_dolp_aolp_error_deg']):.2f}°",
    ]


def rounded_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    box = draw.textbbox((x, y), text, font=font)
    draw.rounded_rectangle((x - 10, y - 5, box[2] + 10, box[3] + 5), radius=10, fill="#ECEDEF")
    draw.text((x, y), text, font=font, fill="#202B3D")


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    direct = None
    if args.direct_summary and args.direct_summary.is_file():
        direct = json.loads(args.direct_summary.read_text(encoding="utf-8"))

    width = 1536
    rows = 3 if direct else 2
    height = 360 + rows * 105
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(args.font, 35, bold=True)
    header_font = load_font(args.font, 23, bold=True)
    body_font = load_font(args.font, 22)
    small_font = load_font(args.font, 18)
    mono_font = load_font(args.font, 19)
    dark = "#202B3D"
    muted = "#687386"
    line = "#D9DEE6"

    draw.text((65, 42), "HAMMER：实验室服务器正式监督训练结果", font=title_font, fill=dark)
    draw.text(
        (65, 96),
        "Version D 使用验证集选择的 best_val（epoch 40）；正式测试集 1414 帧",
        font=small_font,
        fill=muted,
    )

    columns = [650, 815, 990, 1185, 1395]
    draw.text((65, 150), "训练权重 → 测试集", font=header_font, fill=dark)
    headers = ["DoLP MAE\n↓", "RMSE\n↓", "vector\n↓", "weighted AoLP\n↓", "high-DoLP AoLP\n↓"]
    for x, header in zip(columns, headers):
        anchor = "mm"
        draw.multiline_text((x, 166), header, font=header_font, fill=dark, anchor=anchor, align="center", spacing=2)
    draw.line((65, 215, 1470, 215), fill=line, width=2)

    table_rows: list[tuple[str, str, list[str]]] = [
        (
            "Frozen Stage1 → HAMMER",
            "HAMMER 训练流程中的冻结低频先验，1414 帧",
            metric_row(summary, "stage1_"),
        ),
        (
            "Version D epoch 40 → HAMMER",
            "HAMMER 监督训练，验证集选择 best_val，1414 帧",
            metric_row(summary, "stage2_"),
        ),
    ]
    if direct:
        table_rows.append(
            (
                "Direct U-Net++ → HAMMER",
                "同一 HAMMER train/val/test 协议，1414 帧",
                [
                    f"{float(direct['dolp_mae']):.4f}",
                    f"{float(direct['dolp_rmse']):.4f}",
                    f"{float(direct['cos_sin_vector_error']):.4f}",
                    f"{float(direct['weighted_aolp_error_deg']):.2f}°",
                    f"{float(direct['high_dolp_aolp_error_deg']):.2f}°",
                ],
            )
        )

    y = 244
    for label, detail, values in table_rows:
        rounded_label(draw, (75, y), label, mono_font)
        draw.text((65, y + 43), detail, font=small_font, fill=dark)
        for x, value in zip(columns, values):
            draw.text((x, y + 23), value, font=body_font, fill=dark, anchor="mm")
        draw.line((65, y + 89, 1470, y + 89), fill=line, width=1)
        y += 105

    note_y = y + 10
    draw.rounded_rectangle((65, note_y, 1470, note_y + 60), radius=12, fill="#F5F6F8")
    draw.text(
        (85, note_y + 18),
        "说明：所有指标均为逐帧指标的算术平均，箭头向下表示越低越好。",
        font=small_font,
        fill=muted,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, quality=95)
    print(args.output)


if __name__ == "__main__":
    main()
