# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Search-score distribution and result breakdown.

Reads a `.scores.csv` (result, score) and draws the score histogram with
the win/draw/loss split, plus a per-bin result fraction below, the eval
calibration curve when the data carries real WDL labels.

    uv run scripts/plot_deltas.py <file.scores.csv> [more.csv …] [options]

One file:   stacked histogram + stacked result fraction.
Many files: overlaid histograms + per-file win-fraction (or density) curves.

Options:
    --bins INT       histogram bins (default: 60)
    --max-cp INT     score cap for the bin range (default: 500)
    -o, --output     image path (default: <stem>_scores.png)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

import soulplot as sp

WIN  = sp.advantage(0.7)
DRAW = sp.MUTE
LOSS = sp.advantage(-0.7)

# Distinct hues for multi-file overlays, legible on the warm ground.
FILE_COLORS = ["#86b06a", "#e0935a", "#e6b450", "#7d97b8", "#b79be0", "#cf7d7d"]


def _load(path: str) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Return (stem, result, score, sentinel_mask) for a `.scores.csv`."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    result = data[:, 0]
    score  = data[:, 1].astype(float)
    sentinel = score == 32767

    return (Path(path).stem.removesuffix(".scores"), result, score, sentinel)


def _single(files: list[str], args: argparse.Namespace) -> None:
    stem, result, score, sentinel = _load(files[0])
    w_mask = np.abs(result - 1.0) < 0.01
    d_mask = np.abs(result - 0.5) < 0.01
    b_mask = np.abs(result - 0.0) < 0.01

    sp.use_theme()
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True,
    )

    clip = args.max_cp
    bins = np.linspace(-clip, clip, args.bins + 1)

    ax_top.hist(
        [score[w_mask], score[d_mask], score[b_mask]],
        bins=bins, stacked=True, zorder=3, alpha=0.9,
        color=[WIN, DRAW, LOSS], label=["win", "draw", "loss"],
    )

    ax_top.axvline(0, color=sp.MUTE, lw=0.8, alpha=0.6, zorder=2)
    ax_top.set_ylabel("positions", labelpad=8)
    ax_top.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    n_total = len(score)
    pct_w = 100 * w_mask.sum() / n_total
    pct_d = 100 * d_mask.sum() / n_total
    pct_b = 100 * b_mask.sum() / n_total

    counts_w, _ = np.histogram(score[w_mask], bins=bins)
    counts_d, _ = np.histogram(score[d_mask], bins=bins)
    counts_b, _ = np.histogram(score[b_mask], bins=bins)
    counts_total = counts_w + counts_d + counts_b

    with np.errstate(invalid="ignore"):
        frac_w = np.where(counts_total > 0, counts_w / counts_total, 0.0)
        frac_d = np.where(counts_total > 0, counts_d / counts_total, 0.0)
        frac_b = np.where(counts_total > 0, counts_b / counts_total, 0.0)

    centers = (bins[:-1] + bins[1:]) / 2
    width   = 2 * clip / args.bins * 0.95

    # composition bars, dimmed so the calibration curves read on top
    ax_bot.bar(centers, frac_w,                        width=width, color=WIN,  alpha=0.5, zorder=3)
    ax_bot.bar(centers, frac_d, bottom=frac_w,         width=width, color=DRAW, alpha=0.5, zorder=3)
    ax_bot.bar(centers, frac_b, bottom=frac_w + frac_d, width=width, color=LOSS, alpha=0.5, zorder=3)
    _mark_halfway(ax_bot)

    # Calibration: empirical expected score E = win + ½·draw against the logistic
    # the eval should follow, with K (cp per win-prob) fit to the data. How tight
    # E hugs the dashed curve is how well-scaled the engine's score is.
    if (w_mask.sum() + b_mask.sum()) > 0:
        counts_total = counts_w + counts_d + counts_b
        sig = counts_total > 0
        exp_score = frac_w + 0.5 * frac_d
        k = _fit_k(centers[sig], exp_score[sig], counts_total[sig].astype(float))
        grid = np.linspace(-clip, clip, 400)

        ax_bot.plot(grid, 1.0 / (1.0 + 10.0 ** (-grid / k)),
                    color=sp.GOLD, ls=(0, (5, 2)), lw=1.4, alpha=0.9, zorder=4)
        ax_bot.plot(centers[sig], exp_score[sig], color=sp.TEXT, lw=1.7, alpha=0.95, zorder=5)
        ax_bot.text(0.012, 0.94, f"K = {k:.0f}", transform=ax_bot.transAxes, ha="left",
                    va="top", fontsize=8.5, fontweight="bold", color=sp.GOLD)

    ax_bot.set_xlabel("search score (cp)", labelpad=8)
    ax_bot.set_ylabel("fraction", labelpad=8)
    ax_bot.set_ylim(0, 1)
    ax_bot.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax_bot.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

    _finish(ax_top, ax_bot, clip,
            legend=["win", "draw", "loss"], legend_colors=[WIN, DRAW, LOSS])

    n_sent = int(sentinel.sum())
    sent_str = f"   ·   sentinel {n_sent:,}" if n_sent else ""
    sp.title(fig, "score distribution",
             f"{stem}   ·   {n_total:,} samples   ·   "
             f"W {pct_w:.1f}%  D {pct_d:.1f}%  B {pct_b:.1f}%   ·   "
             f"{args.bins} bins   ·   ±{clip} cp{sent_str}")

    plt.subplots_adjust(top=0.90, bottom=0.09, left=0.08, right=0.97, hspace=0.07)
    sp.save(fig, args.output or f"{stem}_scores.png", dpi=200)


def _multi(files: list[str], args: argparse.Namespace) -> None:
    sp.use_theme()
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True,
    )

    clip = args.max_cp
    bins = np.linspace(-clip, clip, args.bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    width = 2 * clip / args.bins * 0.90

    labels, totals, all_results, all_scores, all_counts = [], [], [], [], []

    for i, path in enumerate(files):
        stem, result, score, _ = _load(path)
        labels.append(stem)
        totals.append(len(score))
        all_results.append(result)
        all_scores.append(score)
        c = FILE_COLORS[i % len(FILE_COLORS)]
        ax_top.hist(score, bins=bins, zorder=3, alpha=0.32, color=c, label=stem,
                    histtype="stepfilled", edgecolor=c, linewidth=0.8)
        counts, _ = np.histogram(score, bins=bins)
        all_counts.append(counts.astype(float))

    # WDL if any file carries wins or losses, else relative dataset density
    has_wdl = any(
        (np.abs(r - 1.0) < 0.01).any() or (np.abs(r - 0.0) < 0.01).any()
        for r in all_results
    )
    if has_wdl:
        for i in range(len(files)):
            result, score = all_results[i], all_scores[i]

            c = FILE_COLORS[i % len(FILE_COLORS)]
            w = np.histogram(score[np.abs(result - 1.0) < 0.01], bins=bins)[0]
            d = np.histogram(score[np.abs(result - 0.5) < 0.01], bins=bins)[0]
            b = np.histogram(score[np.abs(result - 0.0) < 0.01], bins=bins)[0]
            tot = w + d + b

            with np.errstate(invalid="ignore"):
                frac_w = np.where(tot > 0, w / tot, 0.0)
            ax_bot.plot(centers, frac_w, color=c, lw=1.6, alpha=0.9, zorder=3 + i)
        ax_bot.set_ylabel("win fraction", labelpad=8)
    else:
        total_counts = np.sum(all_counts, axis=0)

        with np.errstate(invalid="ignore"):
            for i, counts in enumerate(all_counts):
                frac = np.where(total_counts > 0, counts / total_counts, 0.0)
                bottom = (np.where(total_counts > 0,
                                   sum(all_counts[j] for j in range(i)) / total_counts, 0.0)
                          if i else None)
                ax_bot.bar(centers, frac, bottom=bottom, width=width,
                           color=FILE_COLORS[i % len(FILE_COLORS)], alpha=0.85, zorder=3)
        ax_bot.set_ylabel("fraction", labelpad=8)

    ax_top.axvline(0, color=sp.MUTE, lw=0.8, alpha=0.6, zorder=2)
    ax_top.set_ylabel("positions", labelpad=8)
    ax_top.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    _mark_halfway(ax_bot)
    ax_bot.set_xlabel("search score (cp)", labelpad=8)
    ax_bot.set_ylim(0, 1)
    ax_bot.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

    _finish(ax_top, ax_bot, clip,
            legend=labels, legend_colors=[FILE_COLORS[i % len(FILE_COLORS)] for i in range(len(files))])

    total_str = "   ".join(f"{t:,}" for t in totals)
    sp.title(fig, "score distribution",
             f"{', '.join(labels)}   ·   {total_str} samples   ·   {args.bins} bins   ·   ±{clip} cp")

    plt.subplots_adjust(top=0.90, bottom=0.09, left=0.08, right=0.97, hspace=0.07)
    sp.save(fig, args.output or "deltas_compare.png", dpi=200)


def _fit_k(cp: np.ndarray, exp_score: np.ndarray, weight: np.ndarray) -> float:
    """Fit the logistic scale K (cp per win-prob) by weighted least squares.

    Minimizes Σ w·(E − 1/(1+10^(−cp/K)))² over the populated bins via golden
    section, the same calibration constant the eval tuner golden-searches.
    """
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


def _mark_halfway(ax: Axes) -> None:
    """Dashed gold guide at the 50% line, labeled at the right edge."""
    ax.axhline(0.5, color=sp.GOLD, ls="--", lw=0.8, alpha=0.7, zorder=2)
    ax.text(0.995, 0.53, "50%", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color=sp.GOLD, alpha=0.8)


def _finish(ax_top, ax_bot, clip: int, *, legend: list[str], legend_colors: list) -> None:
    """Shared axis limits, styling, and a swatch legend on the top panel."""
    ax_bot.set_xlim(-clip, clip)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    sp.style_axes(ax_top)
    sp.style_axes(ax_bot)

    handles = [Line2D([0], [0], color=c, lw=6, alpha=0.9) for c in legend_colors]
    leg = ax_top.legend(handles, legend, loc="upper right", frameon=True, fontsize=8,
                        facecolor=sp.PANEL, edgecolor=sp.LINE, labelcolor=sp.TEXT)
    leg.get_frame().set_alpha(0.9)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot search-score distribution from a .scores.csv dataset.",
        formatter_class=sp.HelpFormatter,
    )

    ap.add_argument("csv", nargs="+", help="one or more .scores.csv files (comma-separated ok)")
    ap.add_argument("--bins", type=int, default=60, help="histogram bins (default: 60)")
    ap.add_argument("--max-cp", type=int, default=500, help="score cap for the bin range")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    files = [s.strip() for a in args.csv for s in a.split(",") if s.strip()]

    (_single if len(files) == 1 else _multi)(files, args)


if __name__ == "__main__":
    main()
