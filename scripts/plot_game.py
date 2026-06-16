# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy", "chess"]
# ///
"""
Game evaluation river.

Walks a PGN, evaluates every position, draws advantage as a colored river:
the line's color tracks who's ahead (green White, red Black, dim near equal),
filled to the center line, the sharpest swings annotated.

    uv run scripts/plot_game.py game.pgn [--engine ./soul] [--nodes 100000]

Two-engine comparison:
    uv run scripts/plot_game.py game.pgn -e ./soul --compare ./other

Options:
    -e, --engine PATH      primary engine (default: ./soul)
        --compare PATH     overlay a second engine's read
    -n, --nodes INT        search nodes per position (default: 100000)
    -m, --movetime INT     search time per position, milliseconds
    -t, --threads INT      search threads (default: engine default)
    -H, --hash MB          transposition table size in MB (default: engine default)
        --fresh            clear TT/history between positions (independent evals)
        --clamp INT        fix the y-axis cp range (default: auto from P98)
    -o, --output PATH      image path (default: <pgn>_eval.png)
        --dpi INT          output DPI (default: 200)
        --show             open interactively after saving
    -wr, --white-relative  treat the engine's score as White-relative
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


# Advantage hues from the shared gradient; the comparison line a subordinate blue.
WHITE_A = sp.advantage(0.7)   # White ahead, green
BLACK_A = sp.advantage(-0.7)  # Black ahead, red
CMP     = "#7d97b8"           # comparison engine
BLUNDER = BLACK_A
MISTAKE = sp.GOLD
GREAT   = WHITE_A


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
            # A checkmated final position has no move for the engine to score (it
            # returns 0), which would yank the river back off the rail. Pin it to
            # the winner's side directly: the mated side is the one to move.
            if chess.Board(fen).is_checkmate():
                r = uci.Score(-uci.MATE if fen.split()[1] == "w" else uci.MATE, 0)
            else:
                # Clear TT/history so each eval stands alone; the default carries
                # it across the walk, mirroring how the engine sees a real game.
                if fresh:
                    eng.new_game()
                r = eng.search(f"fen {fen}", lim)
                # White-relative engines already report White's view; otherwise the
                # score is side-to-move relative, so flip cp *and* mate sign on Black.
                if not white_relative and fen.split()[1] == "b":
                    r = uci.Score(-r.cp, -r.mate if r.mate is not None else None)
            out.append(r)
            el  = time.perf_counter() - t0
            eta = el / i * (total - i)
            print(f"  {i:>3}/{total}  {r.cp:+5d} cp  [{el:5.1f}s · ~{eta:.0f}s left]", flush=True)
    return out


def _winprob(cp: float) -> float:
    """Centipawns → signed advantage in [−1, 1] via the standard logistic.

    Squashes the unbounded cp scale so the river's color saturates smoothly: a
    pawn or two reads decisive, tiny edges stay near-neutral, mate-scores cap out.
    """
    return 2.0 / (1.0 + 10.0 ** (-cp / 400.0)) - 1.0


def _wp_pos(cp, scale: float = 800.0):
    """cp → display position in [−1, 1] via a logistic, clamped inside the rails.

    Plotting the river in this space (on a linear axis) gives the near-zero
    middlegame room while compressing decisive evals; the clamp pins mate-range
    to the edge, visible, instead of running to infinity as a true scale would.
    `scale` is deliberately gentler than the 400 used for true win-probability
    *color*: it trades a little near-zero resolution for a slower flatten, so the
    slide into a winning position reads as a slope rather than slamming the rail.
    """
    w = 2.0 / (1.0 + 10.0 ** (-np.asarray(cp, dtype=float) / scale)) - 1.0
    return np.clip(w, -0.9985, 0.9985)


def _find_swings(evals: np.ndarray, sans: list[str], n: int = 5,
                 min_cp: int = 50, min_gap: int = 3) -> list[tuple[int, int, str]]:
    """The n largest single-move eval swings, as (half_move_idx, delta_cp, san).

    Greedy by magnitude, but keeps picks ≥ `min_gap` half-moves apart so one
    decisive sequence doesn't claim every label and pile them in a corner.
    """
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
    dpi:      int        = 200,
    show:     bool       = False,
    white_relative: bool = False,
    compare_white_relative: bool = False,
) -> None:
    if movetime is not None:
        nodes = None  # movetime overrides nodes
    game = _load_game(pgn_path)
    fens, sans = _game_positions(game)
    n_pos = len(fens)
    headers = dict(game.headers)

    white  = headers.get("White",  "White")
    black  = headers.get("Black",  "Black")
    result = headers.get("Result", "*")
    event  = headers.get("Event",  "")
    date   = headers.get("Date",   "")[:4]

    eng_name  = Path(engine).stem
    limit_str = uci.limit_label(nodes=nodes, movetime=movetime)
    print(f"Evaluating {n_pos} positions ({eng_name}, {limit_str}) …")
    raw1 = _evaluate_game(engine, fens, nodes=nodes, movetime=movetime, threads=threads,
                          hash_mb=hash_mb, fresh=fresh, white_relative=white_relative)

    raw2: list[uci.Score] | None = None
    cmp_name: str | None   = None

    if compare:
        cmp_name = Path(compare).stem
        print(f"Evaluating with {cmp_name} …")
        raw2 = _evaluate_game(compare, fens, nodes=nodes, movetime=movetime, threads=threads,
                              hash_mb=hash_mb, fresh=fresh, white_relative=compare_white_relative)

    clip_limit = clamp if clamp is not None else 30000
    mate1  = [s.mate for s in raw1]
    evals1 = np.clip(np.array([s.cp for s in raw1], dtype=float), -clip_limit, clip_limit)
    sm1    = sp.ema(evals1, 0.40)

    sm2: np.ndarray | None = None
    y2 = yr2 = None

    if raw2 is not None:
        evals2 = np.clip(np.array([s.cp for s in raw2], dtype=float), -clip_limit, clip_limit)
        sm2     = sp.ema(evals2, 0.40)
        y2, yr2 = _wp_pos(sm2), _wp_pos(evals2)

    # Pre-transform everything into winprob space and plot on a linear axis. The
    # view auto-zooms to the data: a quiet game fills the axis, a decisive one
    # spreads to the rails. --clamp fixes the extent in cp if you want it pinned.
    y1, yr1 = _wp_pos(sm1), _wp_pos(evals1)

    reach = max(abs(float(y1.min())), abs(float(y1.max())))

    if y2 is not None:
        reach = max(reach, abs(float(y2.min())), abs(float(y2.max())))
    edge = float(_wp_pos(clamp)) if clamp is not None else min(0.999, max(reach, 0.12) + 0.05)

    xs     = np.arange(1, n_pos + 1)
    swings = _find_swings(evals1, sans)

    sp.use_theme()
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.subplots_adjust(top=0.83, bottom=0.09, left=0.07, right=0.96)

    # advantage fills to the center line; interpolate for clean zero crossings
    ax.fill_between(xs, y1, 0, where=(y1 >= 0), color=WHITE_A, alpha=0.14, interpolate=True, zorder=1)
    ax.fill_between(xs, y1, 0, where=(y1 <= 0), color=BLACK_A, alpha=0.14, interpolate=True, zorder=1)

    # raw ghost trace under the smoothed river
    ax.plot(xs, yr1, color=sp.MUTE, alpha=0.20, lw=0.6, zorder=3)

    # the river: color flows with who's ahead (keyed to the true eval)
    river_c = np.array([sp.advantage(_winprob(v)) for v in sm1])
    sp.gradient_line(ax, xs, y1, river_c, lw=2.0, zorder=4)

    # comparison overlay: single cool hue, dashed, subordinate
    if y2 is not None and yr2 is not None:
        ax.plot(xs, yr2, color=CMP, alpha=0.22, lw=0.7, zorder=3)
        ax.plot(xs, y2, color=CMP, lw=1.5, ls="--", alpha=0.80, zorder=4)

    # zero line
    ax.axhline(0, color=sp.MUTE, lw=0.9, alpha=0.5, zorder=2)

    ax.set_xlim(0.5, n_pos + max(n_pos * 0.06, 2))

    # ── swing annotations ──
    # A decisive sequence clusters swings into a tight x-range and their labels
    # overprint. Pack them into vertical lanes: left to right, each label drops to
    # the lowest tier whose previous label clears it, so a crowded run fans out
    # instead of overprinting on one line. Monospace font makes a label's width a
    # clean char-count × glyph advance, so collisions are known without a trial
    # render. Above- and below-zero labels pack separately (they can't collide),
    # and the leader line keeps each label tied to its dot.
    ax_w_in = ax.get_position().width * fig.get_size_inches()[0]
    xlo, xhi = ax.get_xlim()
    glyph_dx = (7 * 0.6 / 72) * (xhi - xlo) / ax_w_in   # one 7pt mono glyph, in data-x
    lanes_up: list[float] = []   # rightmost x consumed per tier, above zero
    lanes_dn: list[float] = []   # ditto below zero

    for idx, delta, san in sorted((s for s in swings if s[0] < n_pos), key=lambda s: s[0]):
        y = float(y1[idx])
        is_white = (idx % 2 == 0)
        good_for_mover = (delta > 0) if is_white else (delta < 0)
        ad = abs(delta)

        if ad >= 100 and not good_for_mover:
            col = BLUNDER
        elif ad >= 70 and good_for_mover:
            col = GREAT
        else:
            col = MISTAKE

        # Mate reads as #N, not the meaningless cp delta into the sentinel.
        m = mate1[idx]
        detail = f"#{abs(m)}" if m is not None else f"{delta:+d}"
        text = f"{_move_label(idx, san)}  ({detail})"

        # Drop into the lowest free tier on this side, then claim its right edge.
        lanes = lanes_up if y >= 0 else lanes_dn
        half = len(text) * glyph_dx / 2 + glyph_dx
        left, right = xs[idx] - half, xs[idx] + half
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
            arrowprops=dict(arrowstyle="-", color=col, lw=0.6, alpha=0.40),
            path_effects=[pe.withStroke(linewidth=2.0, foreground=sp.PANEL)],
        )

    # axes
    step  = 20
    ticks = list(range(step, n_pos + 1, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t // 2) for t in ticks], fontsize=8)
    ax.set_xlabel("move", labelpad=8)

    # Linear axis in winprob space, symmetric about zero, padded to the data.
    ax.set_ylim(-edge, edge)

    # cp tick ladder placed at winprob positions, thinned so the compressed upper
    # region never collides: keep a rung only a readable gap from the last kept.
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

    # No grain here: it samples uniformly in data coords, which the winprob
    # y-scale would squash into smeared bands up top. The river's fills carry the
    # texture instead.
    sp.style_axes(ax, grid=True, grain=False)

    # legend only when comparing: proxy handles (the river is multi-colored)
    if sm2 is not None:
        handles = [Line2D([0], [0], color=sp.TEXT, lw=2.0, label=eng_name),
                   Line2D([0], [0], color=CMP, lw=1.6, ls="--", label=cmp_name)]
        leg = ax.legend(handles=handles, fontsize=8, facecolor=sp.PANEL,
                        edgecolor=sp.LINE, labelcolor=sp.TEXT, loc="upper left")
        leg.get_frame().set_alpha(0.9)

    # title block: players headline, engine(s), then metadata
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
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--white-relative", "-wr", action="store_true", default=False)
    args = ap.parse_args()

    # --white-relative pairs with the most recently named engine/compare:
    #   --engine ./soul -wr --compare ./other -wr
    wr_engine = wr_compare = False
    last = ""

    for a in sys.argv[1:]:
        if a in ("--engine", "-e"):
            last = "engine"
        elif a == "--compare":
            last = "compare"
        elif a in ("--white-relative", "-wr"):
            if last == "compare":
                wr_compare = True
            else:
                wr_engine = True

    plot_game(
        args.pgn, args.engine, args.compare, args.nodes, args.movetime, args.threads,
        args.hash, args.fresh, args.clamp, args.output, args.dpi, args.show, wr_engine, wr_compare,
    )


if __name__ == "__main__":
    main()
