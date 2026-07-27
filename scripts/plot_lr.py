# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Learning-rate schedule.

Reads an evaltune JSONL log and draws LR over training: a glow curve over a
gradient fade, warmup ramps shaded, warm-restart lines marked, peak and final
values called out.

    uv run scripts/plot_lr.py <evaltune.jsonl> [options]

Options:
    -o, --output PATH   image path (default: <log>_lr.png)
    --log-scale         logarithmic y axis
    --dpi INT           output DPI (default: 200)
    --width / --height  figure size in inches (default: 12 × 5.5)
    --show              open interactively after saving
"""

from __future__ import annotations

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import soulplot as sp

from pathlib import Path


LR = "#b48ade"  # the schedule curve
WARMUP = "#5b9fd4"  # warmup shading


def parse_log(path: str) -> dict:
    """Pull the LR series and restart epochs out of an evaltune log."""
    epochs, lrs = [], []
    restarts: list[int] = []

    for e in sp.iter_log(path):
        if e.get("event") == "epoch":
            ep, lr = e.get("epoch"), e.get("lr")

            if ep is None or lr is None:
                continue

            epochs.append(int(ep))
            lrs.append(float(lr))
        elif e.get("event") == "restart" and (ep := e.get("epoch")) is not None:
            restarts.append(int(ep))

    if not epochs:
        return {}

    ep_arr = np.array(epochs, dtype=np.int64)
    lr_arr = np.array(lrs, dtype=np.float64)

    return {
        "epochs": ep_arr,
        "lrs": lr_arr,
        "restarts": restarts,
        "peak_lr": float(lr_arr.max()),
        "min_lr": float(lr_arr.min()),
        "final_lr": float(lr_arr[-1]),
    }


def detect_warmups(epochs: np.ndarray, lrs: np.ndarray, min_span: int = 2) -> list[tuple[int, int]]:
    """Runs of strictly rising LR spanning ≥ `min_span` steps → (start, peak) epochs."""
    regions: list[tuple[int, int]] = []
    start: int | None = None

    for i in range(1, len(lrs)):
        if lrs[i] > lrs[i - 1]:
            if start is None:
                start = i - 1
        else:
            if start is not None and (i - 1) - start >= min_span:
                regions.append((int(epochs[start]), int(epochs[i - 1])))
            start = None

    if start is not None and (len(lrs) - 1) - start >= min_span:
        regions.append((int(epochs[start]), int(epochs[-1])))
    return regions


def plot_lr(
    log_path: str,
    output: str | None = None,
    log_scale: bool = False,
    dpi: int = 200,
    fig_width: float = 12.0,
    fig_height: float = 5.5,
    show: bool = False,
) -> None:
    data = parse_log(log_path)

    if not data:
        print(f"error: no epoch entries in '{log_path}'.", file=sys.stderr)
        sys.exit(1)

    epochs, lrs, restarts = data["epochs"], data["lrs"], data["restarts"]
    peak_lr, min_lr, final_lr = data["peak_lr"], data["min_lr"], data["final_lr"]
    total_ep = int(epochs[-1])
    peak_idx = int(np.argmax(lrs))

    warmups = detect_warmups(epochs, lrs)
    # The tuner's restart gate fires on any ≥1.5× LR jump, which includes the
    # initial warmup. Keep only restarts after the first warmup region ends.
    if restarts and warmups:
        restarts = [r for r in restarts if r > warmups[0][1]]

    sp.use_theme()
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for i, (ws, we) in enumerate(warmups):
        ax.axvspan(ws, we, color=WARMUP, alpha=0.05, zorder=0)
        ax.axvline(we, color=WARMUP, ls=":", lw=0.7, alpha=0.30, zorder=1)

        if i == 0:
            ax.text((ws + we) / 2, peak_lr * 0.93, "warmup", ha="center", va="top",
                    fontsize=7.5, color=WARMUP, alpha=0.6)

    floor = min_lr * 0.5 if log_scale else 0.0
    sp.gradient_fill(ax, epochs, lrs, LR, base=floor, top_alpha=0.22, zorder=1)

    sp.glow_line(ax, epochs, lrs, LR, lw=2.2, zorder=4)

    for i, r in enumerate(restarts):
        ax.axvline(r, color=sp.GOLD, ls="--", lw=0.7, alpha=0.35, zorder=2)

        if i == 0:
            ax.text(r + total_ep * 0.005, peak_lr * 0.97, "restart", ha="left", va="top",
                    fontsize=7, color=sp.GOLD, alpha=0.6)

    left = epochs[peak_idx] < (epochs[0] + epochs[-1]) / 2
    ax.scatter([epochs[peak_idx]], [peak_lr], color=LR, s=26, zorder=5, edgecolors="none")
    ax.annotate(f"peak {peak_lr:.2e}", xy=(epochs[peak_idx], peak_lr),
                xytext=(12 if left else -12, 10), textcoords="offset points",
                ha="left" if left else "right", fontsize=7.5, color=LR, alpha=0.9,
                arrowprops=dict(arrowstyle="-", color=LR, lw=0.5, alpha=0.3))
    ax.scatter([epochs[-1]], [final_lr], color=sp.MUTE, s=18, zorder=5, edgecolors="none")

    ax.set_xlabel("epoch", labelpad=8)
    ax.set_ylabel("learning rate", labelpad=8)
    ax.set_xlim(epochs[0], total_ep)

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(min_lr * 0.8, peak_lr * 1.08)
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    else:
        ax.set_ylim(0.0, peak_lr * 1.10)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))
    sp.style_axes(ax)

    sp.title(
        fig, "learning-rate schedule",
        f"{Path(log_path).stem}   ·   {sp.format_count(total_ep)} epochs   ·   "
        f"peak {peak_lr:.2e}   ·   final {final_lr:.2e}",
    )
    plt.subplots_adjust(top=0.86, bottom=0.12, left=0.10, right=0.96)

    out = output or Path(log_path).with_suffix("").as_posix() + "_lr.png"
    sp.save(fig, out, show=show, dpi=dpi)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot the learning-rate schedule.",
        formatter_class=sp.HelpFormatter,
    )

    ap.add_argument("log", help="path to evaltune .jsonl log")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--log-scale", action="store_true", help="logarithmic y axis")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--width", type=float, default=12.0)
    ap.add_argument("--height", type=float, default=5.5)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    plot_lr(
        log_path=args.log, output=args.output, log_scale=args.log_scale,
        dpi=args.dpi, fig_width=args.width, fig_height=args.height, show=args.show,
    )


if __name__ == "__main__":
    main()
