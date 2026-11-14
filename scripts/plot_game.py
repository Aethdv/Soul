# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy", "chess"]
# ///
"""
Game evaluation river from a PGN.

    uv run scripts/plot_game.py game.pgn [--engine ./soul] [--nodes 100000]
    uv run scripts/plot_game.py game.pgn -e ./soul --compare ./other
"""

from __future__ import annotations

import argparse
import sys
import time

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from matplotlib.lines import Line2D

import soulplot as sp
import uci

try:
    import chess
    import chess.pgn
except ImportError:
    sys.exit("python-chess is required: pip install chess")


WHITE_A = sp.advantage(0.7)
BLACK_A = sp.advantage(-0.7)
CMP     = "#7d97b8"


def _load_game(path: str) -> chess.pgn.Game:
    with open(path, encoding="utf-8", errors="replace") as f:
        game = chess.pgn.read_game(f)

    if game is None:
        sys.exit(f"No game found in '{path}'")
    return game


def _game_positions(game: chess.pgn.Game) -> tuple[list[str], list[str]]:
    board = game.board()
    fens, sans = [], []

    for mv in game.mainline_moves():
        sans.append(board.san(mv))
        board.push(mv)
        fens.append(board.fen())
    return fens, sans


def _evaluate_game(engine: str, fens: list[str], *, nodes: int | None = None,
                   movetime: int | None = None, threads: int | None = None,
                   hash_mb: int | None = None, fresh: bool = False,
                   white_relative: bool = False) -> list[uci.Score]:
    total = len(fens)
    lim   = uci.limit_str(nodes=nodes, movetime=movetime)
    out: list[uci.Score] = []

    with uci.UCIEngine(engine, threads=threads, hash_mb=hash_mb) as eng:
        t0 = time.perf_counter()

        for i, fen in enumerate(fens, 1):
            stm = fen.split()[1]
            # Mated final position has no legal move so UCI scores 0; pin it to the winner.
            if chess.Board(fen).is_checkmate():
                r = uci.Score(-uci.MATE if stm == "w" else uci.MATE, 0)
            else:
                if fresh:
                    eng.new_game()
                r = eng.search(f"fen {fen}", lim)
                # Non-WR engines report side-to-move scores; flip cp and mate sign on Black.
                if not white_relative and stm == "b":
                    r = uci.Score(-r.cp, -r.mate if r.mate is not None else None)
            out.append(r)
            el  = time.perf_counter() - t0
            eta = el / i * (total - i)
            print(f"  {i:>3}/{total}  {r.cp:+5d} cp  [{el:5.1f}s · ~{eta:.0f}s left]", flush=True)
    return out


def _logistic(cp, scale: float) -> np.ndarray:
    return 2.0 / (1.0 + 10.0 ** (-np.asarray(cp, dtype=float) / scale)) - 1.0


def _wp_pos(cp) -> np.ndarray:
    # 800 cp scale trades near-zero resolution for a gentler slope toward the rails than
    # the 400 cp win-prob scale; clamping pins mate scores just inside [-1, 1].
    return np.clip(_logistic(cp, 800.0), -0.9985, 0.9985)


def _series(raw: list[uci.Score], clip_limit: int) -> tuple[np.ndarray, np.ndarray]:
    evals = np.clip(np.array([s.cp for s in raw], dtype=float), -clip_limit, clip_limit)
    return evals, sp.ema(evals, 0.40)


def _find_swings(evals: np.ndarray, sans: list[str], n: int = 5,
                 min_cp: int = 50, min_gap: int = 3) -> list[tuple[int, int, str]]:
    """Top `n` single-move cp swings spaced at least `min_gap` plies apart."""
    cands = sorted(
        ((i, int(evals[i] - evals[i - 1]), sans[i])
         for i in range(1, len(evals)) if abs(evals[i] - evals[i - 1]) >= min_cp),
        key=lambda s: -abs(s[1]),
    )

    picked: list[tuple[int, int, str]] = []

    for s in cands:
        if all(abs(s[0] - p[0]) >= min_gap for p in picked):
            picked.append(s)

        if len(picked) >= n:
            break
    return picked


def _move_label(idx: int, san: str) -> str:
    """'1. e4' or '1… e5' style label."""
    full = idx // 2 + 1
    sep  = "…" if idx % 2 == 1 else "."
    return f"{full}{sep} {san}"


def plot_game(
    pgn_path: str,
    engine:   str        = "./soul",
    compare:  str | None = None,
    nodes:    int | None = None,
    movetime: int | None = None,
    threads:  int | None = None,
    hash_mb:  int | None = None,
    fresh:    bool       = False,
    clamp:    int | None = None,
    output:   str | None = None,
    dpi:      int        = sp.DPI,
    show:     bool       = False,
    white_relative: bool = False,
    compare_white_relative: bool = False,
) -> None:
    game = _load_game(pgn_path)
    fens, sans = _game_positions(game)
    n_pos = len(fens)
    if not n_pos:
        sys.exit(f"No moves found in '{pgn_path}'")

    headers = game.headers
    white   = headers.get("White",  "White")
    black   = headers.get("Black",  "Black")
    result  = headers.get("Result", "*")
    event   = headers.get("Event",  "")
    date    = headers.get("Date",   "")[:4]

    def eval_with(path: str, wr: bool) -> list[uci.Score]:
        return _evaluate_game(
            path, fens, nodes=nodes, movetime=movetime, threads=threads,
            hash_mb=hash_mb, fresh=fresh, white_relative=wr,
        )

    eng_name  = Path(engine).stem
    limit_str = uci.limit_label(nodes=nodes, movetime=movetime)
    print(f"Evaluating {n_pos} positions ({eng_name}, {limit_str}) …")
    raw1 = eval_with(engine, white_relative)

    raw2: list[uci.Score] | None = None
    cmp_name: str | None = None

    if compare:
        cmp_name = Path(compare).stem
        print(f"Evaluating with {cmp_name} …")
        raw2 = eval_with(compare, compare_white_relative)

    clip_limit = clamp if clamp is not None else 30000
    mate1 = [s.mate for s in raw1]
    evals1, sm1 = _series(raw1, clip_limit)
    y1, yr1 = _wp_pos(sm1), _wp_pos(evals1)

    y2 = yr2 = None
    if raw2 is not None:
        evals2, sm2 = _series(raw2, clip_limit)
        y2, yr2 = _wp_pos(sm2), _wp_pos(evals2)

    # Plot in bounded win-prob space on a linear axis; auto-zoom unless --clamp fixes cp.
    reach = max(abs(float(y1.min())), abs(float(y1.max())))

    if y2 is not None:
        reach = max(reach, abs(float(y2.min())), abs(float(y2.max())))
    edge = float(_wp_pos(clamp)) if clamp is not None else min(0.999, max(reach, 0.12) + 0.05)

    xs     = np.arange(1, n_pos + 1)
    swings = _find_swings(evals1, sans)

    sp.use_theme()
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.subplots_adjust(top=0.83, bottom=0.09, left=0.07, right=0.96)

    ax.fill_between(xs, y1, 0, where=(y1 >= 0), color=WHITE_A, alpha=0.14, interpolate=True, zorder=1)
    ax.fill_between(xs, y1, 0, where=(y1 <= 0), color=BLACK_A, alpha=0.14, interpolate=True, zorder=1)
    ax.plot(xs, yr1, color=sp.MUTE, alpha=0.20, lw=0.6, zorder=3)

    river_c = np.array([sp.advantage(float(v)) for v in _logistic(sm1, 400.0)])
    sp.gradient_line(ax, xs, y1, river_c, lw=2.0, zorder=4)

    if y2 is not None and yr2 is not None:
        ax.plot(xs, yr2, color=CMP, alpha=0.22, lw=0.7, zorder=3)
        ax.plot(xs, y2, color=CMP, lw=1.5, ls="--", alpha=0.80, zorder=4)

    ax.axhline(0, color=sp.MUTE, lw=0.9, alpha=0.5, zorder=2)
    ax.set_xlim(0.5, n_pos + max(n_pos * 0.06, 2))

    # Pack swing labels into non-overlapping vertical tiers above/below zero using 7pt mono width.
    ax_w_in = ax.get_position().width * fig.get_size_inches()[0]
    xlo, xhi = ax.get_xlim()
    glyph_dx = (7 * 0.6 / 72) * (xhi - xlo) / ax_w_in
    lanes_up: list[float] = []
    lanes_dn: list[float] = []

    for idx, delta, san in sorted(swings, key=lambda s: s[0]):
        y = float(y1[idx])
        good_for_mover = (delta > 0) if (idx % 2 == 0) else (delta < 0)
        ad = abs(delta)

        if ad >= 100 and not good_for_mover:
            col = BLACK_A
        elif ad >= 70 and good_for_mover:
            col = WHITE_A
        else:
            col = sp.GOLD

        # Mate reads as #N, not the cp delta into the mate sentinel.
        m = mate1[idx]
        detail = f"#{abs(m)}" if m is not None else f"{delta:+d}"
        text = f"{_move_label(idx, san)}  ({detail})"

        lanes = lanes_up if y >= 0 else lanes_dn
        half = len(text) * glyph_dx / 2 + glyph_dx
        left, right = float(xs[idx]) - half, float(xs[idx]) + half
        tier = next((t for t, redge in enumerate(lanes) if left > redge), len(lanes))

        if tier == len(lanes):
            lanes.append(right)
        else:
            lanes[tier] = right

        side = 1 if y >= 0 else -1
        ax.scatter([xs[idx]], [y], color=col, s=18, zorder=7, edgecolors="none")
        ax.annotate(
            text,
            xy=(xs[idx], y), xytext=(0, side * (16 + tier * 13)), textcoords="offset points",
            ha="center", va="bottom" if side > 0 else "top",
            fontsize=7, color=col, zorder=8,
            arrowprops={"arrowstyle": "-", "color": col, "lw": 0.6, "alpha": 0.40},
            path_effects=[pe.withStroke(linewidth=2.0, foreground=sp.PANEL)],
        )

    step  = 20
    ticks = list(range(step, n_pos + 1, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t // 2) for t in ticks], fontsize=8)
    ax.set_xlabel("move", labelpad=8)
    ax.set_ylim(-edge, edge)

    # Keep cp rungs >= 0.07 apart in winprob space so top labels stay readable.
    ladder = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
    pos, last_w = [], 0.0

    for t in ladder:
        w = float(_wp_pos(t))

        if w > edge:
            break

        if w - last_w >= 0.07:
            pos.append(t)
            last_w = w

    cps = [-t for t in reversed(pos)] + [0] + pos
    ax.set_yticks([float(_wp_pos(c)) for c in cps])
    ax.set_yticklabels([f"{v:+d}" if v else "0" for v in cps], fontsize=8)
    ax.set_ylabel("centipawns", labelpad=8)

    # No grain: uniform data-coord grain squashes near the top of the winprob y-scale.
    sp.style_axes(ax, grid=True, grain=False)

    # Proxy handles because the river changes color along x.
    if y2 is not None:
        handles = [Line2D([0], [0], color=sp.TEXT, lw=2.0, label=eng_name),
                   Line2D([0], [0], color=CMP, lw=1.6, ls="--", label=cmp_name)]
        leg = ax.legend(handles=handles, fontsize=8, facecolor=sp.PANEL,
                        edgecolor=sp.LINE, labelcolor=sp.TEXT, loc="upper left")
        leg.get_frame().set_alpha(0.9)

    fig_h = fig.get_size_inches()[1]
    fig.text(0.50, 1.0 - 0.22 / fig_h, f"{white}  vs  {black}",
             ha="center", va="top", fontsize=15, fontweight="bold", color=sp.TEXT)
    eng_label = eng_name + (f"  vs  {cmp_name}" if cmp_name else "")
    fig.text(0.50, 1.0 - 0.52 / fig_h, eng_label,
             ha="center", va="top", fontsize=10, fontweight="bold", color=sp.GOLD)

    meta = [limit_str]

    if event and event != "?":
        meta.append(f"{event} ({date})" if date and date not in ("????", "") else event)
    meta.append(result)
    fig.text(0.50, 1.0 - 0.76 / fig_h, "   ·   ".join(meta),
             ha="center", va="top", fontsize=8, color=sp.MUTE)

    out = output or (Path(pgn_path).stem + "_eval.png")
    sp.save(fig, out, show=show, dpi=dpi)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Game evaluation river chart.",
        formatter_class=sp.HelpFormatter,
    )

    ap.add_argument("pgn", help="PGN file")
    ap.add_argument("--engine", "-e", default="./soul", help="primary engine path")
    ap.add_argument("--compare", default=None, help="comparison engine path")
    ap.add_argument("--nodes", "-n", type=int, default=100_000, help="search nodes per position")
    ap.add_argument("--movetime", "-m", type=int, default=None, help="search ms per position")
    ap.add_argument("--threads", "-t", type=int, default=None, help="search threads")
    ap.add_argument("--hash", "-H", type=int, default=None, help="TT size in MB")
    ap.add_argument("--fresh", action="store_true", help="clear TT/history between positions")
    ap.add_argument("--clamp", type=int, default=None, help="fix y-axis cp range (default: auto)")
    ap.add_argument("--output", "-o", default=None)
    ap.add_argument("--dpi", type=int, default=sp.DPI)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--white-relative", "-wr", action="store_true", default=False,
                    help="treat preceding engine's score as White-relative")
    args = ap.parse_args()

    # -wr / --white-relative pairs with the most recently named --engine or --compare.
    wr_engine = wr_compare = False
    last = ""

    for a in sys.argv[1:]:
        flag = a.split("=", 1)[0]
        if flag in ("--engine", "-e"):
            last = "engine"
        elif flag == "--compare":
            last = "compare"
        elif flag in ("--white-relative", "-wr"):
            if last == "compare":
                wr_compare = True
            else:
                wr_engine = True

    plot_game(
        args.pgn, engine=args.engine, compare=args.compare, nodes=args.nodes,
        movetime=args.movetime, threads=args.threads, hash_mb=args.hash, fresh=args.fresh,
        clamp=args.clamp, output=args.output, dpi=args.dpi, show=args.show,
        white_relative=wr_engine, compare_white_relative=wr_compare,
    )


if __name__ == "__main__":
    main()
