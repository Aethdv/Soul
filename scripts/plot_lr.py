# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Learning-rate schedule from an evaltune JSONL log.

    uv run scripts/plot_lr.py <evaltune.jsonl> [options]
"""

from __future__ import annotations

import argparse
import math
import sys

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import soulplot as sp


LR     = "#b48ade"
WARMUP = "#5b9fd4"


@dataclass(slots=True)
class Schedule:
    epochs:   np.ndarray
    lrs:      np.ndarray
    restarts: list[int]


def parse_log(path: str) -> Schedule | None:
    epochs, lrs = [], []
    restarts: list[int] = []

    for e in sp.iter_log(path):
        event = e.get("event")
        if event == "epoch":
            ep, lr = e.get("epoch"), e.get("lr")
            if ep is None or lr is None:
                continue
            val = float(lr)
            if math.isfinite(val):
                epochs.append(int(ep))
                lrs.append(val)
        elif event == "restart" and (ep := e.get("epoch")) is not None:
            restarts.append(int(ep))

    if not epochs:
        return None
    return Schedule(np.array(epochs, dtype=np.int64), np.array(lrs, dtype=np.float64), restarts)


def detect_warmups(epochs: np.ndarray, lrs: np.ndarray, min_span: int = 2) -> list[tuple[int, int]]:
    """Runs of strictly rising LR spanning ≥ `min_span` steps → (start, peak) epochs."""
    regions: list[tuple[int, int]] = []
    start: int | None = None

    for i in range(1, len(lrs) + 1):
        if i < len(lrs) and lrs[i] > lrs[i - 1]:
            if start is None:
                start = i - 1
        elif start is not None:
            if (i - 1) - start >= min_span:
                regions.append((int(epochs[start]), int(epochs[i - 1])))
            start = None
    return regions


def plot_lr(
    sched: Schedule,
    stem: str,
    out: str,
    *,
    log_scale: bool,
    dpi: int,
    figsize: tuple[float, float],
    show: bool,
) -> None:
    epochs, lrs = sched.epochs, sched.lrs
    first_ep, total_ep = int(epochs[0]), int(epochs[-1])
    peak_lr, min_lr, final_lr = float(lrs.max()), float(lrs.min()), float(lrs[-1])
    peak_ep = int(epochs[int(np.argmax(lrs))])

    warmups = detect_warmups(epochs, lrs)
    # The tuner's restart gate fires on any ≥1.5× LR jump, including initial warmup.
    restarts = [r for r in sched.restarts if r > warmups[0][1]] if warmups else sched.restarts

    sp.use_theme()
    fig, ax = plt.subplots(figsize=figsize)

    for ws, we in warmups:
        ax.axvspan(ws, we, color=WARMUP, alpha=0.05, zorder=0)
        ax.axvline(we, color=WARMUP, ls=":", lw=0.7, alpha=0.30, zorder=1)
    if warmups:
        ws, we = warmups[0]
        ax.text((ws + we) / 2, peak_lr * 0.93, "warmup", ha="center", va="top",
                fontsize=7.5, color=WARMUP, alpha=0.6)

    floor = min_lr * 0.5 if log_scale else 0.0
    sp.gradient_fill(ax, epochs, lrs, LR, base=floor, top_alpha=0.22, zorder=1)
    sp.glow_line(ax, epochs, lrs, LR, lw=2.2, zorder=4)

    for r in restarts:
        ax.axvline(r, color=sp.GOLD, ls="--", lw=0.7, alpha=0.35, zorder=2)
    if restarts:
        span = max(total_ep - first_ep, 1)
        ax.text(restarts[0] + span * 0.005, peak_lr * 0.97, "restart", ha="left", va="top",
                fontsize=7, color=sp.GOLD, alpha=0.6)

    left = peak_ep < (first_ep + total_ep) / 2
    ax.scatter([peak_ep], [peak_lr], color=LR, s=26, zorder=5, edgecolors="none")
    ax.annotate(f"peak {peak_lr:.2e}", xy=(peak_ep, peak_lr),
                xytext=(12 if left else -12, 10), textcoords="offset points",
                ha="left" if left else "right", fontsize=7.5, color=LR, alpha=0.9,
                arrowprops={"arrowstyle": "-", "color": LR, "lw": 0.5, "alpha": 0.3})
    ax.scatter([total_ep], [final_lr], color=sp.MUTE, s=18, zorder=5, edgecolors="none")

    ax.set_xlabel("epoch", labelpad=8)
    ax.set_ylabel("learning rate", labelpad=8)
    if first_ep < total_ep:
        ax.set_xlim(first_ep, total_ep)

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
        f"{stem}   ·   {sp.format_count(total_ep)} epochs   ·   "
        f"peak {peak_lr:.2e}   ·   final {final_lr:.2e}",
    )
    plt.subplots_adjust(top=0.86, bottom=0.12, left=0.10, right=0.96)
    sp.save(fig, out, show=show, dpi=dpi)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot the learning-rate schedule.",
        formatter_class=sp.HelpFormatter,
    )

    ap.add_argument("log", help="path to evaltune .jsonl log")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--log-scale", action="store_true", help="logarithmic y axis")
    ap.add_argument("--dpi", type=int, default=sp.DPI)
    ap.add_argument("--width", type=float, default=12.0)
    ap.add_argument("--height", type=float, default=5.5)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    sched = parse_log(args.log)
    if sched is None:
        sys.exit(f"error: no epoch entries in '{args.log}'.")

    out = args.output or Path(args.log).with_suffix("").as_posix() + "_lr.png"
    plot_lr(
        sched, Path(args.log).stem, out,
        log_scale=args.log_scale, dpi=args.dpi,
        figsize=(args.width, args.height), show=args.show,
    )


if __name__ == "__main__":
    main()
