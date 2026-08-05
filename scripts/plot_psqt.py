# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Piece-square table heatmaps.

Reads an evaltune checkpoint JSON and renders MG / EG PSQT heatmaps for
one or all pieces as annotated 8×8 board grids, colored by the diverging
gradient: green where the value is high, red where low, dark-neutral near zero.

Usage:
    uv run scripts/plot_psqt.py <checkpoint.json> [piece] [options]

Options:
    -o, --output PATH     Output image path  (default: psqt_<piece>.png)
    --mg-only             Middlegame table only
    --eg-only             Endgame table only
    --material            Show absolute values (default: positional delta)
    --shared-scale        One color scale per phase across all pieces (all view)
    --dpi INT             Output DPI (default: 300)
    --show                Open interactively after saving
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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec

import soulplot as sp


PIECES      = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]
FILE_LABELS = list("abcdefgh")
RANK_LABELS = list("87654321")

# files e–h mirror onto d–a  (4 files × 8 ranks = 32 params per phase)
MIRROR_FILE = [0, 1, 2, 3, 3, 2, 1, 0]


def _reconstruct(values: dict[str, float], piece_idx: int,
                 is_eg: bool) -> np.ndarray:
    """Full 8×8 from the 32-element compressed half."""
    phase = "EG" if is_eg else "MG"
    name  = PIECES[piece_idx].upper()
    board = np.zeros((8, 8))

    for rank in range(8):
        for file in range(8):
            sq = (rank << 2) + MIRROR_FILE[file]
            board[rank][file] = values.get(f"{phase}_{name}[{sq}]", 0.0)
    return board


def _draw_board(
    ax:       Axes,
    board:    np.ndarray,
    title:    str | None,
    cmap,
    norm,
    annot_fs: float = 7.5,
) -> plt.cm.ScalarMappable:
    """Render one 8×8 heatmap with annotations and subtle grid."""
    # Cell fill via pcolormesh: each cell is a standalone polygon, so cell
    # boundaries are exact regardless of pixel-grid alignment. (imshow with
    # nearest-neighbor can mis-sample at cell edges when the display pixel
    # grid doesn't perfectly divide the cell count.)
    x_edges = np.arange(-0.5, 8.0)
    y_edges = np.arange(-0.5, 8.0)
    X, Y = np.meshgrid(x_edges, y_edges)
    im = ax.pcolormesh(X, Y, board, cmap=cmap, norm=norm,
                       shading="flat")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # origin='upper' equivalent: row 0 at top


    inner = np.arange(0.5, 7.0)   # 0.5 … 6.5
    outer = np.array([-0.5, 7.5]) # board edges
    all_h = np.concatenate([outer, inner])
    ax.hlines(all_h, xmin=-0.5, xmax=7.5,
              color=sp.LINE, lw=0.5, zorder=2)
    ax.vlines(all_h, ymin=-0.5, ymax=7.5,
              color=sp.LINE, lw=0.5, zorder=2)

    for r in range(8):
        for f in range(8):
            v = board[r, f]
            ax.text(
                f, r, f"{v:+.0f}",
                ha="center", va="center",
                fontsize=annot_fs, fontweight="bold",
                fontfamily="monospace",
                color=sp.TEXT,
                zorder=2,
                path_effects=[pe.withStroke(
                    linewidth=0.5, foreground=sp.INK,
                )],
            )

    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(FILE_LABELS, fontsize=7, color=sp.MUTE,
                       fontfamily="monospace")
    ax.set_yticklabels(RANK_LABELS, fontsize=7, color=sp.MUTE,
                       fontfamily="monospace")
    ax.tick_params(which="both", length=0, pad=4)

    if title:
        ax.set_title(title, pad=8, fontsize=10, color=sp.TEXT,
                     fontfamily="monospace")

    for spine in ax.spines.values():
        spine.set_edgecolor(sp.LINE)
        spine.set_linewidth(0.6)

    return im


def plot_psqt(
    checkpoint_path: str,
    piece_arg:       str,
    *,
    output:       str | None = None,
    mg_only:      bool  = False,
    eg_only:      bool  = False,
    material:     bool  = False,
    shared_scale: bool  = False,
    dpi:          int   = sp.DPI,
    show:         bool  = False,
) -> None:

    cp = Path(checkpoint_path)

    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"Error: '{cp}' not found.")
    except json.JSONDecodeError as e:
        # The run log sits beside the checkpoint under a name four characters
        # away, and json reports a whole file of objects as trailing garbage.
        if cp.suffix == ".jsonl" or "Extra data" in str(e):
            sys.exit(f"Error: '{cp}' is a run log. PSQTs come from the checkpoint, e.g. evaltune_checkpoint.json.")
        sys.exit(f"Error: bad JSON ({e})")

    # A checkpoint stores per-parameter optimizer state, `{name: {value, momentum,
    # …}}`; the flat `{name: number}` is what older ones carried.
    params = data.get("params")
    values = ({k: v["value"] for k, v in params.items() if isinstance(v, dict) and "value" in v}
              if isinstance(params, dict) else data.get("values"))

    if not isinstance(values, dict) or not values:
        sys.exit("Error: checkpoint carries no parameters.")

    piece_name = piece_arg.strip().capitalize()

    if piece_name == "All":
        sel, single = list(range(6)), False
    elif piece_name in PIECES:
        sel, single = [PIECES.index(piece_name)], True
    else:
        sys.exit(f"Unknown piece '{piece_arg}'. "
                 f"Choose: {', '.join(PIECES)}, or 'all'.")

    show_mg = not eg_only
    show_eg = not mg_only

    if not (show_mg or show_eg):
        show_mg = show_eg = True

    n_ph   = int(show_mg) + int(show_eg)
    n_rows = len(sel)

    sp.use_theme()
    cmap = sp.diverging_cmap()

    board_size = 3.2 if n_rows <= 2 else 2.5
    annot_fs = 7.5 if n_rows <= 2 else 6.0

    fig_w = n_ph * board_size + 1.2
    fig_h = n_rows * (board_size + 0.6) + 1.6

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(sp.INK)

    # Reserve enough header that the first row's board title clears the meta
    # line below the subtitle (the three title lines run down to ~1.02in).
    gs_top = 1.0 - 1.6 / fig_h
    gs = GridSpec(
        n_rows, n_ph + 1, figure=fig,
        width_ratios=[1] * n_ph + [0.04],
        top=gs_top, bottom=0.04,
        wspace=0.22,
        hspace=0.50 if n_rows > 1 else 0.25,
    )

    # Reconstruct every board up front; centering (the default) subtracts the
    # table mean so the colormap spans positional structure, not the material
    # baseline that would otherwise flatten every square to one shade.
    boards = []

    for pidx in sel:
        mg = _reconstruct(values, pidx, False)
        eg = _reconstruct(values, pidx, True)

        if not material:
            mg -= mg.mean()
            eg -= eg.mean()
        boards.append((mg, eg))

    # Color endpoints are symmetric ±max(|x|) so zero lands neutral-center.
    # --shared-scale ties one vmax across all boards within a phase; MG and EG
    # stay independent: their magnitudes differ too much to share a scale.
    shared_mg = shared_eg = None
    if shared_scale and not single:
        shared_mg = max(max((np.abs(m).max() for m, _ in boards), default=1.0), 1.0)
        shared_eg = max(max((np.abs(e).max() for _, e in boards), default=1.0), 1.0)

    for row, (pidx, (mg, eg)) in enumerate(zip(sel, boards)):
        vmax_mg = shared_mg if shared_mg is not None else max(np.abs(mg).max(), 1.0)
        vmax_eg = shared_eg if shared_eg is not None else max(np.abs(eg).max(), 1.0)
        norm_mg = TwoSlopeNorm(vmin=-vmax_mg, vcenter=0, vmax=vmax_mg)
        norm_eg = TwoSlopeNorm(vmin=-vmax_eg, vcenter=0, vmax=vmax_eg)
        name = PIECES[pidx].lower()

        # Board titles; avoid repeating info already in the suptitle
        if single:
            mg_t = "middlegame" if n_ph == 2 else None
            eg_t = "endgame"   if n_ph == 2 else None
        elif n_ph == 2:
            mg_t = f"{name}  ·  middlegame"
            eg_t = f"{name}  ·  endgame"
        else:
            mg_t = eg_t = name

        col, im = 0, None

        if show_mg:
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor(sp.INK)
            im = _draw_board(ax, mg, mg_t, cmap, norm_mg, annot_fs)
            col += 1

        if show_eg:
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor(sp.INK)
            im = _draw_board(ax, eg, eg_t, cmap, norm_eg, annot_fs)
            col += 1

        assert im is not None  # at least one phase always renders
        cax = fig.add_subplot(gs[row, -1])
        cb  = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(colors=sp.MUTE, labelsize=6.5, length=2)
        cb.outline.set_edgecolor(sp.LINE)
        cb.outline.set_linewidth(0.4)

    label = PIECES[sel[0]].lower() if single else "all pieces"
    meta_parts = []

    if mg_only:
        meta_parts.append("middlegame")
    elif eg_only:
        meta_parts.append("endgame")
    meta_parts.append("absolute" if material else "positional δ")

    if shared_scale and not single:
        meta_parts.append("shared scale")

    fig.text(
        0.50, 1.0 - 0.35 / fig_h, "piece-square tables",
        ha="center", va="top", fontsize=15,
        fontfamily="monospace", fontweight="bold", color=sp.TEXT,
    )
    fig.text(
        0.50, 1.0 - 0.70 / fig_h,
        label,
        ha="center", va="top", fontsize=10,
        fontfamily="monospace", fontweight="bold",
        color=sp.GOLD if single else sp.MUTE,
    )

    if meta_parts:
        fig.text(
            0.50, 1.0 - 1.02 / fig_h,
            "  ·  ".join(meta_parts),
            ha="center", va="top", fontsize=8,
            fontfamily="monospace", color=sp.MUTE,
        )

    out = output or (f"psqt_{PIECES[sel[0]].lower()}.png" if single else "psqt_all.png")
    sp.save(fig, out, show=show, dpi=dpi)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render PSQT heatmaps from an evaltune checkpoint.",
        formatter_class=sp.HelpFormatter,
    )

    ap.add_argument("checkpoint",
                    help="evaltune checkpoint (.json)")
    ap.add_argument("piece", nargs="?", default="all",
                    help="Pawn/Knight/Bishop/Rook/Queen/King/all (default: all)")
    ap.add_argument("-o", "--output",     default=None)
    ap.add_argument("--mg-only",          action="store_true")
    ap.add_argument("--eg-only",          action="store_true")
    ap.add_argument("--material",         action="store_true",
                    help="show absolute values (default: positional delta)")
    ap.add_argument("--shared-scale",     action="store_true",
                    help="one color scale per phase across all pieces")
    ap.add_argument("--dpi",    type=int, default=sp.DPI)
    ap.add_argument("--show",             action="store_true")
    args = ap.parse_args()

    if args.mg_only and args.eg_only:
        ap.error("--mg-only and --eg-only are mutually exclusive")

    plot_psqt(
        args.checkpoint, args.piece,
        output=args.output, mg_only=args.mg_only, eg_only=args.eg_only,
        material=args.material, shared_scale=args.shared_scale,
        dpi=args.dpi, show=args.show,
    )


if __name__ == "__main__":
    main()
