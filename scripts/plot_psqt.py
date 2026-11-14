# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Piece-square table heatmaps from an evaltune checkpoint.

    uv run scripts/plot_psqt.py <checkpoint.json> [piece] [options]
"""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec

import soulplot as sp


PIECES      = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]
FILE_LABELS = list("abcdefgh")
RANK_LABELS = list("87654321")

# Files e-h mirror onto d-a (4 files x 8 ranks = 32 params per phase).
MIRROR_FILE = [0, 1, 2, 3, 3, 2, 1, 0]


def _load_values(cp: Path) -> dict[str, float]:
    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"Error: '{cp}' not found.")
    except json.JSONDecodeError as e:
        # The run log sits beside the checkpoint; json flags remaining lines as Extra data.
        if cp.suffix == ".jsonl" or "Extra data" in str(e):
            sys.exit(f"Error: '{cp}' is a run log. PSQTs come from the checkpoint, e.g. evaltune_checkpoint.json.")
        sys.exit(f"Error: bad JSON ({e})")

    if not isinstance(data, dict):
        sys.exit("Error: checkpoint carries no parameters.")

    # Optimizer checkpoints store {name: {value, momentum, ...}}; older ones store {name: number}.
    params = data.get("params")
    values = (
        {k: float(v["value"]) for k, v in params.items() if isinstance(v, dict) and "value" in v}
        if isinstance(params, dict)
        else data.get("values")
    )

    if not isinstance(values, dict) or not values:
        sys.exit("Error: checkpoint carries no parameters.")
    return values


def _reconstruct(values: dict[str, float], piece_idx: int, phase: str) -> np.ndarray:
    """Reconstruct an 8x8 board from the 32-element mirrored parameter half."""
    name  = PIECES[piece_idx].upper()
    board = np.zeros((8, 8))

    for rank in range(8):
        for file in range(8):
            sq = (rank << 2) + MIRROR_FILE[file]
            board[rank, file] = values.get(f"{phase}_{name}[{sq}]", 0.0)
    return board


def _draw_board(
    ax:       Axes,
    board:    np.ndarray,
    title:    str | None,
    cmap,
    norm,
    annot_fs: float = 7.5,
) -> ScalarMappable:
    # pcolormesh avoids nearest-neighbor edge drift across non-integer cell pixel grids.
    edges = np.arange(-0.5, 8.0)
    im = ax.pcolormesh(edges, edges, board, cmap=cmap, norm=norm, shading="flat")
    ax.set_aspect("equal")
    ax.invert_yaxis()

    ax.hlines(edges, xmin=-0.5, xmax=7.5, color=sp.LINE, lw=0.5, zorder=2)
    ax.vlines(edges, ymin=-0.5, ymax=7.5, color=sp.LINE, lw=0.5, zorder=2)

    stroke = [pe.withStroke(linewidth=0.5, foreground=sp.INK)]
    for r in range(8):
        for f in range(8):
            ax.text(
                f, r, f"{board[r, f]:+.0f}",
                ha="center", va="center", fontsize=annot_fs, fontweight="bold",
                color=sp.TEXT, zorder=2, path_effects=stroke,
            )

    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(FILE_LABELS, fontsize=7, color=sp.MUTE)
    ax.set_yticklabels(RANK_LABELS, fontsize=7, color=sp.MUTE)
    ax.tick_params(which="both", length=0, pad=4)

    if title:
        ax.set_title(title, pad=8, fontsize=10, color=sp.TEXT)

    for spine in ax.spines.values():
        spine.set_edgecolor(sp.LINE)
        spine.set_linewidth(0.6)

    return im


def plot_psqt(
    checkpoint_path: str,
    piece_arg:       str,
    *,
    output:       str | None = None,
    mg_only:      bool = False,
    eg_only:      bool = False,
    material:     bool = False,
    shared_scale: bool = False,
    dpi:          int  = sp.DPI,
    show:         bool = False,
) -> None:
    if mg_only and eg_only:
        sys.exit("Error: --mg-only and --eg-only are mutually exclusive.")

    values = _load_values(Path(checkpoint_path))
    piece_name = piece_arg.strip().capitalize()

    if piece_name == "All":
        sel, single = list(range(6)), False
    elif piece_name in PIECES:
        sel, single = [PIECES.index(piece_name)], True
    else:
        sys.exit(f"Unknown piece '{piece_arg}'. Choose: {', '.join(PIECES)}, or 'all'.")

    phases = []
    if not eg_only:
        phases.append(("MG", "middlegame"))
    if not mg_only:
        phases.append(("EG", "endgame"))
    n_ph   = len(phases)
    n_rows = len(sel)

    sp.use_theme()
    cmap = sp.diverging_cmap()

    board_size = 3.2 if n_rows <= 2 else 2.5
    annot_fs   = 7.5 if n_rows <= 2 else 6.0
    fig_w = n_ph * board_size + 1.2
    fig_h = n_rows * (board_size + 0.6) + 1.6

    fig = plt.figure(figsize=(fig_w, fig_h))
    # Reserve header room for 3 title lines down to ~1.02in.
    gs = GridSpec(
        n_rows, n_ph + 1, figure=fig,
        width_ratios=[1] * n_ph + [0.04],
        top=1.0 - 1.6 / fig_h, bottom=0.04,
        wspace=0.22,
        hspace=0.50 if n_rows > 1 else 0.25,
    )

    # Subtract table mean by default so material baseline does not flatten positional deltas.
    boards: list[list[np.ndarray]] = []
    for pidx in sel:
        row_boards = []
        for tag, _ in phases:
            b = _reconstruct(values, pidx, tag)
            if not material:
                b -= b.mean()
            row_boards.append(b)
        boards.append(row_boards)

    # Symmetric +-vmax puts zero at center; MG and EG stay independent even under --shared-scale.
    shared_vmax: dict[int, float] = {}
    if shared_scale and not single:
        for col in range(n_ph):
            shared_vmax[col] = max(max(float(np.abs(row_b[col]).max()) for row_b in boards), 1.0)

    for row, (pidx, row_boards) in enumerate(zip(sel, boards)):
        name = PIECES[pidx].lower()
        im = None

        for col, ((_, phase_label), b) in enumerate(zip(phases, row_boards)):
            vmax = shared_vmax[col] if shared_vmax else max(float(np.abs(b).max()), 1.0)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

            if single:
                board_title = phase_label if n_ph == 2 else None
            elif n_ph == 2:
                board_title = f"{name}  ·  {phase_label}"
            else:
                board_title = name

            im = _draw_board(fig.add_subplot(gs[row, col]), b, board_title, cmap, norm, annot_fs)

        assert im is not None
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[row, -1]))
        cb.ax.tick_params(colors=sp.MUTE, labelsize=6.5, length=2)
        cb.outline.set_edgecolor(sp.LINE)
        cb.outline.set_linewidth(0.4)

    label = PIECES[sel[0]].lower() if single else "all pieces"
    meta_parts = []
    if n_ph == 1:
        meta_parts.append(phases[0][1])
    meta_parts.append("absolute" if material else "positional δ")
    if shared_scale and not single:
        meta_parts.append("shared scale")

    fig.text(0.50, 1.0 - 0.35 / fig_h, "piece-square tables",
             ha="center", va="top", fontsize=15, fontweight="bold", color=sp.TEXT)
    fig.text(0.50, 1.0 - 0.70 / fig_h, label,
             ha="center", va="top", fontsize=10, fontweight="bold",
             color=sp.GOLD if single else sp.MUTE)
    fig.text(0.50, 1.0 - 1.02 / fig_h, "  ·  ".join(meta_parts),
             ha="center", va="top", fontsize=8, color=sp.MUTE)

    out = output or (f"psqt_{PIECES[sel[0]].lower()}.png" if single else "psqt_all.png")
    sp.save(fig, out, show=show, dpi=dpi)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render PSQT heatmaps from an evaltune checkpoint.",
        formatter_class=sp.HelpFormatter,
    )

    ap.add_argument("checkpoint", help="evaltune checkpoint (.json)")
    ap.add_argument("piece", nargs="?", default="all",
                    help="Pawn/Knight/Bishop/Rook/Queen/King/all (default: all)")
    ap.add_argument("-o", "--output", default=None)
    phase_group = ap.add_mutually_exclusive_group()
    phase_group.add_argument("--mg-only", action="store_true")
    phase_group.add_argument("--eg-only", action="store_true")
    ap.add_argument("--material", action="store_true",
                    help="show absolute values (default: positional delta)")
    ap.add_argument("--shared-scale", action="store_true",
                    help="one color scale per phase across all pieces")
    ap.add_argument("--dpi", type=int, default=sp.DPI)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    plot_psqt(
        args.checkpoint, args.piece,
        output=args.output, mg_only=args.mg_only, eg_only=args.eg_only,
        material=args.material, shared_scale=args.shared_scale,
        dpi=args.dpi, show=args.show,
    )


if __name__ == "__main__":
    main()
