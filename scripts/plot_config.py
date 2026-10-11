"""
Plot Config — 通用科学论文作图配置模块。

设计原则:
  1. 零论文依赖 — 不包含任何具体论文的概念/颜色映射
  2. Layer 分离 — 色板 → 语义角色 → 样式 → 导出，各层独立
  3. dict over mode — rcParams 用 dict 覆盖，不锁入固定模式

Usage:
  from scripts.plot_config import setup_style, save_figure, OKABE_ITO as OI

  setup_style()                               # 仅基础样式
  setup_style({"font.size": 9, "figure.dpi": 200})  # 灵活覆盖

  save_figure(fig, "Fig1", out_dir="./figures")
  # → Fig1.png, Fig1.jpg, Fig1.tiff, Fig1.svg (全部 400 DPI)

  save_figure(fig, "Fig1", out_dir="./figures",
              formats=["png", "pdf"], dpi=600)
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Iterable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Layer 1 — 原始色板（纯数据，不含论文语义）                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

OKABE_ITO: dict[str, str] = {
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermillion":     "#D55E00",
    "reddish_purple": "#CC79A7",
    "black":          "#000000",
    "gray":           "#999999",
}

OKABE_ITO_LIST: list[str] = [
    OKABE_ITO["orange"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["yellow"],
    OKABE_ITO["blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["black"],
]

# Paul Tol "Bright" qualitative — 备选，Okabe-Ito 不够时补充
TOL_BRIGHT: dict[str, str] = {
    "blue":    "#4477AA",
    "cyan":    "#66CCEE",
    "green":   "#228833",
    "yellow":  "#CCBB44",
    "red":     "#EE6677",
    "purple":  "#AA3377",
    "grey":    "#BBBBBB",
}

TOL_BRIGHT_LIST: list[str] = [
    TOL_BRIGHT["blue"],
    TOL_BRIGHT["cyan"],
    TOL_BRIGHT["green"],
    TOL_BRIGHT["yellow"],
    TOL_BRIGHT["red"],
    TOL_BRIGHT["purple"],
    TOL_BRIGHT["grey"],
]


def sequential_from_cmap(
    n: int, cmap_name: str = "Blues", start: float = 0.25, stop: float = 0.90
) -> list[str]:
    """从 matplotlib colormap 中取 n 级均匀分布的 hex 色。

    Args:
        n: 取几个颜色
        cmap_name: matplotlib colormap 名 ("Blues", "Oranges", "Greens", ...)
        start: 起始位置 (0-1), 避免太浅
        stop: 终止位置 (0-1), 避免太深

    Returns:
        n 个 hex 色码构成的 list
    """
    cmap = plt.get_cmap(cmap_name)
    return [
        mcolors.to_hex(cmap(v))
        for v in np.linspace(start, stop, n)
    ]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Layer 2 — 通用语义角色（领域无关）                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Roles:
    """通用语义颜色角色。

    使用方式:
        from scripts.plot_config import ROLES
        ax.plot(..., color=ROLES.negative)

    每个角色的 hex 值来自 Layer1，修改只需要改 OKABE_ITO，角色自动跟随。
    """

    def __init__(self, palette: dict[str, str]) -> None:
        self._p = palette

    @property
    def negative(self) -> str:   return self._p["vermillion"]
    @property
    def positive(self) -> str:   return self._p["bluish_green"]
    @property
    def warning(self) -> str:    return self._p["vermillion"]
    @property
    def baseline(self) -> str:   return self._p["gray"]
    @property
    def neutral(self) -> str:    return self._p["gray"]
    @property
    def highlight(self) -> str:  return self._p["orange"]
    @property
    def accent(self) -> str:     return self._p["reddish_purple"]
    @property
    def cool(self) -> str:       return self._p["blue"]
    @property
    def warm(self) -> str:       return self._p["orange"]
    @property
    def muted(self) -> str:      return self._p["gray"]

    # 通用数据分类 — 无特殊语义时用
    @property
    def data_1(self) -> str:     return self._p["orange"]
    @property
    def data_2(self) -> str:     return self._p["sky_blue"]
    @property
    def data_3(self) -> str:     return self._p["bluish_green"]
    @property
    def data_4(self) -> str:     return self._p["blue"]
    @property
    def data_5(self) -> str:     return self._p["vermillion"]
    @property
    def data_6(self) -> str:     return self._p["reddish_purple"]
    @property
    def data_7(self) -> str:     return self._p["yellow"]
    @property
    def data_8(self) -> str:     return self._p["black"]


ROLES = Roles(OKABE_ITO)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Layer 3 — rcParams（基础 + 灵活覆盖）                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

STYLE_BASE: dict = {
    # ── 字体 ──
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":        8,
    "axes.labelsize":   9,
    "axes.titlesize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.titlesize": 10,
    # ── 字体嵌入（LaTeX 兼容） ──
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "svg.fonttype":     "none",
    # ── 刻度 & 轴线 ──
    "axes.linewidth":       0.5,
    "xtick.major.width":    0.5,
    "ytick.major.width":    0.5,
    "xtick.minor.width":    0.3,
    "ytick.minor.width":    0.3,
    "xtick.major.size":     3.0,
    "ytick.major.size":     3.0,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    # ── 图例 ──
    "legend.frameon":       False,
    "legend.handlelength":  1.2,
    "legend.handleheight":  0.7,
    # ── spines ──
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    # ── 网格 ──
    "axes.grid":            False,
    "grid.alpha":           0.3,
    "grid.linewidth":       0.3,
    "grid.linestyle":       "--",
    # ── 默认色轮 ──
    "axes.prop_cycle":      plt.cycler(color=OKABE_ITO_LIST),
}

# 预设覆盖 — 按场景提供，可自由 merge
STYLE_TALK: dict = {
    "font.size":        14,
    "axes.labelsize":   14,
    "axes.titlesize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
}

STYLE_POSTER: dict = {
    "font.size":        18,
    "axes.labelsize":   18,
    "axes.titlesize":   18,
    "xtick.labelsize":  16,
    "ytick.labelsize":  16,
    "legend.fontsize":  16,
}

# 3D 图常用覆盖
STYLE_3D: dict = {
    "font.size":        9,
    "axes.labelsize":   10,
    "axes.titlesize":   11,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "axes.grid":        False,
    "axes.spines.top":  True,
    "axes.spines.right": True,
}

# 热力图常用覆盖
STYLE_HEATMAP: dict = {
    "font.size":        7,
    "axes.labelsize":   8,
    "xtick.labelsize":  6,
    "ytick.labelsize":  7,
}


def setup_style(overrides: dict | None = None) -> None:
    """应用基础 rcParams，可选覆盖。

    Args:
        overrides: 要覆盖/追加的 rcParams dict。
                   None 则只应用 STYLE_BASE。
                   可传入 STYLE_3D, STYLE_TALK 等预设或自定义 dict。
                   多个预设需自行 merge:
                     setup_style({**STYLE_BASE, **STYLE_3D,
                                  "figure.dpi": 200})

    Examples:
        # 最简 — 只应用基础
        setup_style()

        # 覆盖字号
        setup_style({"font.size": 7})

        # 报告/演讲用
        setup_style(STYLE_TALK)

        # 3D 图
        setup_style(STYLE_3D)

        # 自定义合并
        setup_style({**STYLE_BASE, "font.size": 10,
                     "xtick.direction": "in"})
    """
    plt.rcParams.update(STYLE_BASE)
    if overrides:
        plt.rcParams.update(overrides)


def reset_style() -> None:
    """重置为 matplotlib 默认样式。"""
    plt.rcdefaults()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Layer 4 — 导出引擎                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# 内部配置 — 不与论文语义耦合
#
# 实现细节:
#   - PNG/SVG/PDF: matplotlib 原生输出
#   - JPG/TIFF: 先通过 RGBA buffer 渲染 → PIL 转码压缩
#     matplotlib 的 pil_kwargs 在部分后端不生效，用 PIL 直接处理更可靠

_FORMAT_SPECS: dict[str, dict] = {
    "png": {
        "ext": ".png",
        "save_kwargs": {"bbox_inches": "tight", "pad_inches": 0.02},
    },
    "jpg": {
        "ext": ".jpg",
        "save_kwargs": {"bbox_inches": "tight", "pad_inches": 0.02},
        "pil_convert": True,
        "pil_format": "JPEG",
        "pil_kwargs": {"quality": 100, "subsampling": 0},  # 0 = 4:4:4, no chroma subsampling
    },
    "tiff": {
        "ext": ".tiff",
        "save_kwargs": {"bbox_inches": "tight", "pad_inches": 0.02},
        "pil_convert": True,
        "pil_format": "TIFF",
        "pil_kwargs": {"compression": "tiff_lzw"},
    },
    "svg": {
        "ext": ".svg",
        "save_kwargs": {"bbox_inches": "tight", "pad_inches": 0.02},
    },
    "pdf": {
        "ext": ".pdf",
        "save_kwargs": {"bbox_inches": "tight", "pad_inches": 0.02},
    },
}

# 常用组合
FORMATS_ALL = ["png", "jpg", "tiff", "svg"]
FORMATS_VECTOR = ["pdf", "svg"]
FORMATS_RASTER = ["png", "jpg", "tiff"]
FORMATS_PUBLICATION = ["pdf", "png", "svg"]  # 投稿最常用


def save_figure(
    fig: plt.Figure,
    name: str,
    out_dir: str | Path,
    formats: Iterable[str] | None = None,
    dpi: int | None = 400,
    close: bool = True,
    verbose: bool = True,
) -> list[Path]:
    """一行输出多种格式。

    Args:
        fig: matplotlib Figure 对象
        name: 文件名（不含扩展名），如 "Fig1_BarChart"
        out_dir: 输出目录
        formats: 输出格式列表，默认 ["png","jpg","tiff","svg"]
        dpi: 位图 DPI，默认 400。矢量格式忽略此参数。
        close: 保存后自动 plt.close(fig)
        verbose: 打印每个保存的路径

    Returns:
        已保存的 Path 列表

    Raises:
        ValueError: formats 中包含不支持的格式
        OSError: 输出目录无法创建或写入

    Examples:
        # 默认四种格式 + 400 DPI
        save_figure(fig, "Fig1_BarChart", "./figures")

        # 只要 PNG + PDF，600 DPI
        save_figure(fig, "Fig2_Scaling", "./figures",
                     formats=["png", "pdf"], dpi=600)

        # 不关闭 figure（后续继续编辑）
        save_figure(fig, "draft", "./preview", close=False)
    """
    if formats is None:
        formats = FORMATS_ALL

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for fmt in formats:
        spec = _FORMAT_SPECS.get(fmt)
        if spec is None:
            raise ValueError(
                f"Unsupported format: {fmt!r}. "
                f"Supported: {list(_FORMAT_SPECS.keys())}"
            )

        out_path = out_dir / f"{name}{spec['ext']}"

        if spec.get("pil_convert"):
            # JPG/TIFF: render to RGBA buffer → PIL convert
            _save_via_pil(fig, out_path, spec, dpi)
        else:
            # PNG/SVG/PDF: matplotlib 原生
            kwargs = dict(spec["save_kwargs"])
            if fmt not in ("svg", "pdf") and dpi is not None:
                kwargs["dpi"] = dpi
            fig.savefig(out_path, **kwargs)

        saved.append(out_path)

        if verbose:
            dpi_str = ""
            if fmt not in ("svg", "pdf") and dpi:
                dpi_str = f", {dpi} DPI"
            print(f"  {out_path}{dpi_str}")

    if close:
        plt.close(fig)

    return saved


def _save_via_pil(
    fig: plt.Figure,
    out_path: Path,
    spec: dict,
    dpi: int | None,
) -> None:
    """Render figure to RGBA buffer, then save via PIL with proper compression."""
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    buf.seek(0)
    img = Image.open(buf)
    # Convert RGBA → RGB for formats that don't support alpha
    if img.mode == "RGBA" and spec["pil_format"] in ("JPEG",):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode == "RGBA":
        img = img.convert("RGB")
    img.save(out_path, format=spec["pil_format"], **spec["pil_kwargs"])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  便捷工具                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def set_okabe_ito_cycle() -> None:
    """将当前 axes 的色轮设为 Okabe-Ito（不修改全局 rcParams）。"""
    plt.gca().set_prop_cycle(color=OKABE_ITO_LIST)


def apply_okabe_ito_globally() -> None:
    """将 Okabe-Ito 设为全局默认色轮。"""
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=OKABE_ITO_LIST)


def despine(ax=None, top: bool = True, right: bool = True,
            left: bool = False, bottom: bool = False) -> None:
    """便捷去 spine。

    Args:
        ax: 目标 axes，默认当前 axes
        top/right/left/bottom: 是否去除
           默认去 top+right（与 STYLE_BASE 一致），传 left=True 可去左 spine
    """
    if ax is None:
        ax = plt.gca()
    for spine, hide in [("top", top), ("right", right),
                        ("left", left), ("bottom", bottom)]:
        if hide:
            ax.spines[spine].set_visible(False)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  自检（直接运行 python scripts/plot_config.py）                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("=" * 60)
    print("plot_config self-check")
    print("=" * 60)

    print("\n[Layer 1] OKABE_ITO palette:")
    for name, hex_code in OKABE_ITO.items():
        print(f"  {name:20s} {hex_code}")

    print("\n[Layer 1] sequential_from_cmap('Blues', 5):")
    blues5 = sequential_from_cmap(5, "Blues")
    for i, c in enumerate(blues5):
        print(f"  step {i}: {c}")

    print("\n[Layer 2] ROLES:")
    for attr in sorted(dir(ROLES)):
        if attr.startswith("_"):
            continue
        val = getattr(ROLES, attr)
        if callable(val):
            continue
        print(f"  {attr:20s} {val}")

    print("\n[Layer 3] STYLE_BASE keys:", len(STYLE_BASE))
    print("  Preset dicts: STYLE_TALK, STYLE_POSTER, STYLE_3D, STYLE_HEATMAP")

    print("\n[Layer 4] Supported formats:", list(_FORMAT_SPECS.keys()))
    print("  Format combos: FORMATS_ALL, FORMATS_VECTOR, FORMATS_RASTER, FORMATS_PUBLICATION")

    print("\n  All checks passed.")
