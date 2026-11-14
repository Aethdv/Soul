# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy", "chess"]
# ///
"""
Opening move preferences from the starting position.

    uv run scripts/plot_opening.py [--engine ./soul] [--nodes 100000] [options]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize, to_rgb
from matplotlib.ticker import FuncFormatter

import soulplot as sp
import uci

try:
    import chess
except ImportError:
    sys.exit("python-chess is required: pip install chess")


@dataclasses.dataclass(frozen=True, slots=True)
class MoveEval:
    san:     str
    eval_cp: int


def evaluate_moves(engine: str, lim: str, limit_label: str,
                   threads: int | None, hash_mb: int | None) -> list[MoveEval]:
    board = chess.Board()
    moves = list(board.legal_moves)
    total = len(moves)

    print(f"Evaluating {total} moves · {limit_label} each …")
    results: list[MoveEval] = []

    with uci.UCIEngine(engine, threads=threads, hash_mb=hash_mb) as eng:
        t0 = time.perf_counter()

        for i, mv in enumerate(moves, 1):
            san = board.san(mv)
            # Search reports Black-to-move after our push, so negate to White-relative cp.
            eng.new_game()
            ev = -eng.search(f"startpos moves {mv.uci()}", lim).cp
            results.append(MoveEval(san, ev))
            el  = time.perf_counter() - t0
            eta = el / i * (total - i)
            print(f"  {i:>2}/{total}  {san:>5}  {ev:+5d} cp  "
                  f"[{el:5.1f}s · ~{eta:.0f}s left]", flush=True)
    results.sort(key=lambda m: m.eval_cp, reverse=True)
    return results


def _draw(results: list[MoveEval], engine_name: str, limit_label: str,
          output: str, dpi: int, show: bool) -> None:
    n        = len(results)
    evals    = [r.eval_cp for r in results]
    ev_min   = min(evals)
    ev_max   = max(evals)
    ev_range = max(ev_max - ev_min, 1)

    # Span the actual spread rather than centering at 0, since opening moves cluster tightly.
    cmap = sp.oklch_ramp([
        (0.58, 0.15, 28.0),   # red
        (0.74, 0.13, 85.0),   # amber
        (0.70, 0.16, 148.0),  # green
    ])
    norm = Normalize(vmin=ev_min, vmax=ev_max)
    ordered = list(reversed(results))

    sp.use_theme()
    fig, ax = plt.subplots(figsize=(10, max(6.0, n * 0.34 + 2.4)))

    bar_h    = 0.52
    baseline = ev_min - ev_range * 0.06
    grad_res = 512
    bg_rgb   = np.array(to_rgb(sp.INK))

    # Fade toward baseline with a 0.62 floor so bar tails darken instead of vanishing.
    frac        = np.linspace(0, 1, grad_res)
    blend       = 0.62 + 0.38 * frac
    alpha_curve = 0.82 + 0.18 * frac
    bg_contrib  = bg_rgb * (1.0 - blend)[:, None]

    for i, mv in enumerate(ordered):
        is_best = (i == n - 1)
        color   = np.array(to_rgb(cmap(norm(mv.eval_cp))))
        bar_b, bar_t = i - bar_h / 2, i + bar_h / 2

        grad = np.empty((1, grad_res, 4))
        grad[0, :, :3] = bg_contrib + color * blend[:, None]
        grad[0, :, 3]  = alpha_curve

        ax.imshow(grad, aspect="auto", interpolation="bilinear",
                  extent=(baseline, mv.eval_cp, bar_b, bar_t), zorder=2)
        ax.plot([mv.eval_cp, mv.eval_cp], [bar_b + 0.03, bar_t - 0.03],
                color=color, alpha=0.95, lw=1.4, solid_capstyle="round", zorder=3)
        ax.text(mv.eval_cp + ev_range * 0.018, i, f"{mv.eval_cp:+d}",
                ha="left", va="center",
                fontsize=8.4 if is_best else 7.6,
                fontweight="bold" if is_best else "normal",
                color=sp.GOLD if is_best else tuple(color), zorder=4)

    ax.set_yticks(range(n))
    labels = ax.set_yticklabels([f"1. {mv.san}" for mv in ordered], fontsize=9.5, color=sp.TEXT)
    labels[-1].set_color(sp.GOLD)
    labels[-1].set_fontweight("bold")

    ax.set_xlabel("centipawns", labelpad=8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.0f}"))
    ax.set_xlim(baseline - ev_range * 0.04, ev_max + ev_range * 0.14)
    ax.set_ylim(-0.6, n - 0.4)

    # Bars carry the y-axis, so hide the left spine and put grid lines on cp.
    sp.style_axes(ax, grid=False, grain=True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.grid(axis="x", color=sp.TEXT, alpha=0.05, lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    for i in range(0, n, 2):
        ax.axhspan(i - 0.5, i + 0.5, color=sp.TEXT, alpha=0.012, zorder=0)

    sp.title(fig, "opening preferences", f"{engine_name}   ·   {limit_label}")
    plt.subplots_adjust(top=0.91, bottom=0.06, left=0.11, right=0.94)
    sp.save(fig, output, show=show, dpi=dpi)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Opening move preference chart.",
        formatter_class=sp.HelpFormatter,
    )

    ap.add_argument("--engine",   "-e", default="./soul", help="engine binary (default: ./soul)")
    ap.add_argument("--nodes",    "-n", type=int, default=None, help="search nodes per move (default: 100000)")
    ap.add_argument("--movetime", "-m", type=int, default=None, help="search ms per move")
    ap.add_argument("--threads",  "-t", type=int, default=None, help="search threads")
    ap.add_argument("--hash",     "-H", type=int, default=None, help="TT size in MB")
    ap.add_argument("--depth",    "-d", type=int, default=None, help="fixed search depth per move")
    ap.add_argument("--output",   "-o", default=None)
    ap.add_argument("--dpi",            type=int, default=sp.DPI)
    ap.add_argument("--show",           action="store_true")
    args = ap.parse_args()

    if args.nodes is None and args.movetime is None and args.depth is None:
        args.nodes = 100_000

    lim = uci.limit_str(nodes=args.nodes, movetime=args.movetime, depth=args.depth)
    limit_label = uci.limit_label(nodes=args.nodes, movetime=args.movetime, depth=args.depth)
    results = evaluate_moves(args.engine, lim, limit_label, args.threads, args.hash)
    _draw(results, Path(args.engine).stem, limit_label,
          args.output or "opening_prefs.png", args.dpi, args.show)


if __name__ == "__main__":
    main()
