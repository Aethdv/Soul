# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy", "chess"]
# ///
"""
Opening move preferences.

Evaluates every legal first move from the starting position and ranks them
by search score, drawn as advantage-colored bars: green for the highest, red
for the lowest, near-neutral in between.

    uv run scripts/plot_opening.py [--engine ./soul] [--nodes 100000] [options]

Options:
    -e, --engine PATH   engine binary (default: ./soul)
    -n, --nodes INT     search node budget per move
    -m, --movetime INT  search time per move, milliseconds
    -t, --threads INT   search threads (default: engine default)
    -H, --hash MB       transposition table size in MB (default: engine default)
    -d, --depth INT     fixed search depth per move
    -o, --output PATH   image path (default: opening_prefs.png)
    --dpi INT           output DPI (default: 300)
    --show              open interactively after saving
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.colors import Normalize, to_rgb
from matplotlib.ticker import FuncFormatter

import soulplot as sp
import uci

try:
    import chess
except ImportError:
    sys.exit("python-chess is required:  pip install chess")


@dataclasses.dataclass(frozen=True, slots=True)
class MoveEval:
    uci:     str
    san:     str
    from_sq: int
    to_sq:   int
    eval_cp: int


def evaluate_moves(engine: str, nodes: int | None, movetime: int | None,
                   depth: int | None, threads: int | None, hash_mb: int | None) -> list[MoveEval]:
    board = chess.Board()
    moves = list(board.legal_moves)
    total = len(moves)

    limit_str = uci.limit_label(nodes=nodes, movetime=movetime, depth=depth)
    lim = uci.limit_str(nodes=nodes, movetime=movetime, depth=depth)
    print(f"Evaluating {total} moves · {limit_str} each …")
    results: list[MoveEval] = []

    with uci.UCIEngine(engine, threads=threads, hash_mb=hash_mb) as eng:
        t0 = time.perf_counter()

        for i, mv in enumerate(moves, 1):
            san = board.san(mv)
            # Fresh search per move; score is reported for the side to move
            # after our move (Black), so negate to White-relative centipawns.
            eng.new_game()
            ev = -eng.search(f"startpos moves {mv.uci()}", lim).cp
            results.append(MoveEval(mv.uci(), san, mv.from_square, mv.to_square, ev))
            el  = time.perf_counter() - t0
            eta = el / i * (total - i)
            print(f"  {i:>2}/{total}  {san:>5}  {ev:+5d} cp  "
                  f"[{el:5.1f}s · ~{eta:.0f}s left]", flush=True)
    results.sort(key=lambda m: m.eval_cp, reverse=True)
    return results


def _draw(results: list[MoveEval], engine_name: str, nodes: int | None, movetime: int | None,
          depth: int | None, output: str, dpi: int, show: bool) -> None:
    n        = len(results)
    evals    = [r.eval_cp for r in results]
    ev_min   = min(evals)
    ev_max   = max(evals)
    ev_range = max(ev_max - ev_min, 1)

    # Color by rank across the actual spread: a vivid red→amber→green ramp that
    # stays saturated through the middle. Opening evals cluster tight near zero,
    # so a zero-centered diverging map would dump every move into its dark center;
    # spanning the data range keeps each bar distinct and legible.
    cmap = sp.oklch_ramp([
        (0.58, 0.15, 28.0),   # worst  : warm red
        (0.74, 0.13, 85.0),   # middle : amber
        (0.70, 0.16, 148.0),  # best   : green
    ])

    norm = Normalize(vmin=ev_min, vmax=ev_max)

    # worst at bottom (y=0), best at top (y=n-1)
    ordered = list(reversed(results))

    sp.use_theme()
    fig_h = max(6.0, n * 0.34 + 2.4)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    bar_h    = 0.52
    baseline = ev_min - ev_range * 0.06
    grad_res = 512
    bg_rgb   = np.array(to_rgb(sp.INK))

    # Bar fill fades toward the baseline but keeps a 0.62 floor so the tail
    # darkens rather than vanishing into the ground.
    frac        = np.linspace(0, 1, grad_res)
    blend       = 0.62 + 0.38 * frac
    alpha_curve = 0.82 + 0.18 * frac
    bg_contrib  = bg_rgb * (1.0 - blend)[:, None]

    for i, mv in enumerate(ordered):
        is_best = (i == n - 1)
        color   = np.array(to_rgb(cmap(norm(mv.eval_cp))))

        bar_l, bar_r = baseline, mv.eval_cp
        bar_b, bar_t = i - bar_h / 2, i + bar_h / 2
        rgb = bg_contrib + color * blend[:, None]

        grad = np.zeros((1, grad_res, 4))
        grad[0, :, :3] = rgb
        grad[0, :, 3]  = alpha_curve

        ax.imshow(grad, aspect="auto", interpolation="bilinear",
                  extent=(bar_l, bar_r, bar_b, bar_t), zorder=2)

        ax.plot([bar_r, bar_r], [bar_b + 0.03, bar_t - 0.03],
                color=color, alpha=0.95, lw=1.4, solid_capstyle="round", zorder=3)

        ax.text(bar_r + ev_range * 0.018, i, f"{mv.eval_cp:+d}",
                ha="left", va="center",
                fontsize=8.4 if is_best else 7.6,
                fontweight="bold" if is_best else "normal",
                color=sp.GOLD if is_best else tuple(color), zorder=4)

    ax.set_yticks(range(n))
    labels = ax.set_yticklabels([f"1. {mv.san}" for mv in ordered], fontsize=9.5)
    best_san = f"1. {results[0].san}"

    for lbl in labels:
        if lbl.get_text() == best_san:
            lbl.set_color(sp.GOLD)
            lbl.set_fontweight("bold")
        else:
            lbl.set_color(sp.TEXT)

    ax.set_xlabel("centipawns", labelpad=8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.0f}"))
    ax.set_xlim(baseline - ev_range * 0.04, ev_max + ev_range * 0.14)
    ax.set_ylim(-0.6, n - 0.4)

    # spines + grain from the kit; bars carry the y-axis, so drop the left spine
    # and grid centipawns vertically instead of the default horizontal grid.
    sp.style_axes(ax, grid=False, grain=True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.grid(axis="x", color=sp.TEXT, alpha=0.05, lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    # faint zebra striping to track rows across the width
    for i in range(0, n, 2):
        ax.axhspan(i - 0.5, i + 0.5, color=sp.TEXT, alpha=0.012, zorder=0)

    limit_str = uci.limit_label(nodes=nodes, movetime=movetime, depth=depth)
    sp.title(fig, "opening preferences", f"{engine_name}   ·   {limit_str}")
    plt.subplots_adjust(top=0.91, bottom=0.06, left=0.11, right=0.94)
    sp.save(fig, output, show=show, dpi=dpi)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Opening move preference chart.",
        formatter_class=sp.HelpFormatter,
    )

    ap.add_argument("--engine",   "-e", default="./soul")
    ap.add_argument("--nodes",    "-n", type=int, default=None)
    ap.add_argument("--movetime", "-m", type=int, default=None)
    ap.add_argument("--threads",  "-t", type=int, default=None)
    ap.add_argument("--hash",     "-H", type=int, default=None)
    ap.add_argument("--depth",    "-d", type=int, default=None)
    ap.add_argument("--output",   "-o", default=None)
    ap.add_argument("--dpi",            type=int, default=sp.DPI)
    ap.add_argument("--show",           action="store_true")
    args = ap.parse_args()

    if args.nodes is None and args.movetime is None and args.depth is None:
        args.nodes = 100_000

    results = evaluate_moves(args.engine, args.nodes, args.movetime, args.depth, args.threads, args.hash)
    _draw(results, Path(args.engine).stem, args.nodes, args.movetime, args.depth,
          args.output or "opening_prefs.png", args.dpi, args.show)


if __name__ == "__main__":
    main()
