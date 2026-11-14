# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Search-score distribution and result breakdown.

Reads `.scores.csv` (`result,score`) and draws the score histogram with win/draw/loss
split and per-bin result fraction (plus logistic calibration K when WDL labels exist).

    uv run scripts/plot_deltas.py <file.scores.csv> [more.csv ...] [options]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

import soulplot as sp

WIN  = sp.advantage(0.7)
DRAW = sp.MUTE
LOSS = sp.advantage(-0.7)

FILE_COLORS = ["#86b06a", "#e0935a", "#e6b450", "#7d97b8", "#b79be0", "#cf7d7d"]


@dataclass(slots=True)
class Dataset:
    stem:   str
    result: np.ndarray
    score:  np.ndarray


def _load(path: str) -> Dataset:
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.size == 0:
        sys.exit(f"error: no rows in '{path}'")
    return Dataset(Path(path).stem.removesuffix(".scores"), data[:, 0], data[:, 1].astype(float))


def _wdl_masks(result: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.abs(result - 1.0) < 0.01,
        np.abs(result - 0.5) < 0.01,
        np.abs(result - 0.0) < 0.01,
    )


def _frac(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros(num.shape, dtype=float), where=den > 0)


def _single(files: list[str], args: argparse.Namespace) -> None:
    ds = _load(files[0])
    masks = _wdl_masks(ds.result)

    sp.use_theme()
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True,
    )

    clip = args.max_cp
    bins = np.linspace(-clip, clip, args.bins + 1)

    ax_top.hist(
        [ds.score[m] for m in masks],
        bins=bins, stacked=True, zorder=3, alpha=0.9,
        color=[WIN, DRAW, LOSS], label=["win", "draw", "loss"],
    )

    n_total = len(ds.score)
    pct_w, pct_d, pct_b = (100 * m.sum() / n_total for m in masks)

    counts_w, counts_d, counts_b = (np.histogram(ds.score[m], bins=bins)[0] for m in masks)
    counts_total = counts_w + counts_d + counts_b

    frac_w = _frac(counts_w, counts_total)
    frac_d = _frac(counts_d, counts_total)
    frac_b = _frac(counts_b, counts_total)

    centers = (bins[:-1] + bins[1:]) / 2
    width = 2 * clip / args.bins * 0.95

    # Dimmed so the calibration curves read on top.
    bottom = np.zeros_like(centers)
    for frac, color in ((frac_w, WIN), (frac_d, DRAW), (frac_b, LOSS)):
        ax_bot.bar(centers, frac, bottom=bottom, width=width, color=color, alpha=0.5, zorder=3)
        bottom += frac

    # Empirical expected score E = win + ½·draw against the logistic fit for K.
    if masks[0].any() or masks[2].any():
        sig = counts_total > 0
        exp_score = frac_w + 0.5 * frac_d
        k = _fit_k(centers[sig], exp_score[sig], counts_total[sig].astype(float))
        grid = np.linspace(-clip, clip, 400)

        ax_bot.plot(grid, 1.0 / (1.0 + 10.0 ** (-grid / k)),
                    color=sp.GOLD, ls=(0, (5, 2)), lw=1.4, alpha=0.9, zorder=4)
        ax_bot.plot(centers[sig], exp_score[sig], color=sp.TEXT, lw=1.7, alpha=0.95, zorder=5)
        ax_bot.text(0.012, 0.94, f"K = {k:.0f}", transform=ax_bot.transAxes, ha="left",
                    va="top", fontsize=8.5, fontweight="bold", color=sp.GOLD)

    ax_bot.set_ylabel("fraction", labelpad=8)

    n_sent = int((ds.score == 32767).sum())
    sent_str = f"   ·   sentinel {n_sent:,}" if n_sent else ""
    sub = (f"{ds.stem}   ·   {n_total:,} samples   ·   "
           f"W {pct_w:.1f}%  D {pct_d:.1f}%  B {pct_b:.1f}%   ·   "
           f"{args.bins} bins   ·   ±{clip} cp{sent_str}")
    _finish(fig, ax_top, ax_bot, clip,
            legend=["win", "draw", "loss"], legend_colors=[WIN, DRAW, LOSS],
            sub=sub, output=args.output or f"{ds.stem}_scores.png", dpi=args.dpi)


def _multi(files: list[str], args: argparse.Namespace) -> None:
    sp.use_theme()
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True,
    )

    clip = args.max_cp
    bins = np.linspace(-clip, clip, args.bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    width = 2 * clip / args.bins * 0.90

    datasets = [_load(path) for path in files]
    labels = [ds.stem for ds in datasets]
    colors = [FILE_COLORS[i % len(FILE_COLORS)] for i in range(len(datasets))]

    all_counts = []
    for ds, c in zip(datasets, colors):
        ax_top.hist(ds.score, bins=bins, zorder=3, alpha=0.32, color=c, label=ds.stem,
                    histtype="stepfilled", edgecolor=c, linewidth=0.8)
        all_counts.append(np.histogram(ds.score, bins=bins)[0].astype(float))

    # WDL win-fraction when any file carries wins/losses, else relative dataset share.
    wdl = [_wdl_masks(ds.result) for ds in datasets]
    if any(w.any() or b.any() for w, _, b in wdl):
        for i, (ds, masks, c) in enumerate(zip(datasets, wdl, colors)):
            w, d, b = (np.histogram(ds.score[m], bins=bins)[0] for m in masks)
            ax_bot.plot(centers, _frac(w, w + d + b), color=c, lw=1.6, alpha=0.9, zorder=3 + i)
        ax_bot.set_ylabel("win fraction", labelpad=8)
    else:
        total_counts = np.sum(all_counts, axis=0)
        bottom = np.zeros_like(centers)
        for counts, c in zip(all_counts, colors):
            frac = _frac(counts, total_counts)
            ax_bot.bar(centers, frac, bottom=bottom, width=width, color=c, alpha=0.85, zorder=3)
            bottom += frac
        ax_bot.set_ylabel("fraction", labelpad=8)

    total_str = "   ".join(f"{len(ds.score):,}" for ds in datasets)
    sub = f"{', '.join(labels)}   ·   {total_str} samples   ·   {args.bins} bins   ·   ±{clip} cp"
    _finish(fig, ax_top, ax_bot, clip,
            legend=labels, legend_colors=colors,
            sub=sub, output=args.output or "deltas_compare.png", dpi=args.dpi)


def _fit_k(cp: np.ndarray, exp_score: np.ndarray, weight: np.ndarray) -> float:
    """Fit logistic scale `K` (cp per win-prob) by weighted least squares."""
    def err(k: float) -> float:
        pred = 1.0 / (1.0 + 10.0 ** (-cp / k))
        return float(np.sum(weight * (exp_score - pred) ** 2))

    a, b = 20.0, 1000.0
    g = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - g * (b - a), a + g * (b - a)

    for _ in range(60):
        if err(c) < err(d):
            b, d = d, c
            c = b - g * (b - a)
        else:
            a, c = c, d
            d = a + g * (b - a)
    return (a + b) / 2.0


def _finish(fig, ax_top, ax_bot, clip: int, *, legend: list[str], legend_colors: list,
            sub: str, output: str, dpi: int) -> None:
    ax_top.axvline(0, color=sp.MUTE, lw=0.8, alpha=0.6, zorder=2)
    ax_top.set_ylabel("positions", labelpad=8)
    ax_top.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax_bot.axhline(0.5, color=sp.GOLD, ls="--", lw=0.8, alpha=0.7, zorder=2)
    ax_bot.text(0.995, 0.53, "50%", transform=ax_bot.transAxes, ha="right", va="bottom",
                fontsize=7, color=sp.GOLD, alpha=0.8)
    ax_bot.set_xlabel("search score (cp)", labelpad=8)
    ax_bot.set_xlim(-clip, clip)
    ax_bot.set_ylim(0, 1)
    ax_bot.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax_bot.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

    plt.setp(ax_top.get_xticklabels(), visible=False)
    sp.style_axes(ax_top)
    sp.style_axes(ax_bot)

    handles = [Line2D([0], [0], color=c, lw=6, alpha=0.9) for c in legend_colors]
    leg = ax_top.legend(handles, legend, loc="upper right", frameon=True, fontsize=8,
                        facecolor=sp.PANEL, edgecolor=sp.LINE, labelcolor=sp.TEXT)
    leg.get_frame().set_alpha(0.9)

    sp.title(fig, "score distribution", sub)
    plt.subplots_adjust(top=0.90, bottom=0.09, left=0.08, right=0.97, hspace=0.07)
    sp.save(fig, output, dpi=dpi)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot search-score distribution from a .scores.csv dataset.",
        formatter_class=sp.HelpFormatter,
    )

    ap.add_argument("csv", nargs="+", help="one or more .scores.csv files (comma-separated ok)")
    ap.add_argument("--bins", type=int, default=60, help="histogram bins (default: 60)")
    ap.add_argument("--max-cp", type=int, default=500, help="score cap for the bin range")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--dpi", type=int, default=sp.DPI)
    args = ap.parse_args()

    if args.bins < 1 or args.max_cp <= 0:
        ap.error("--bins and --max-cp must be >= 1")

    files = [s.strip() for a in args.csv for s in a.split(",") if s.strip()]
    if not files:
        ap.error("supply at least one .scores.csv file")

    (_single if len(files) == 1 else _multi)(files, args)


if __name__ == "__main__":
    main()
