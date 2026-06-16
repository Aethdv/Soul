# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Training and validation loss curves.

Reads an evaltune JSONL log; EMA-smoothed train and val curves over faint
raw traces, the overfit gap hatched, curves labeled at their ends, the
best-val epoch in gold, warm-restart lines. A fixed-K reference is dashed
when the log carries `ref_loss`.

    uv run scripts/plot_loss.py <evaltune.jsonl> [options]

Options:
    -o, --output PATH   image path (default: <log>_loss.png)
    -a, --alpha FLOAT   EMA factor, (0, 1]  (default: 0.06)
    --no-raw            hide raw traces, smoothed only
    --dpi INT           output DPI (default: 200)
    --width / --height  figure size in inches (default: 13 × 7)
    --show              open interactively after saving
"""

from __future__ import annotations

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

import soulplot as sp

from pathlib import Path


TRAIN = "#86b06a"    # sage
VAL = "#e0935a"      # terracotta
REF = "#7d97b8"      # dusty-blue
RESTART = "#b79be0"  # plum


def parse_log(path: str) -> dict:
    """Parse Soul's evaltune JSONL into arrays + best/final summary."""
    epochs, train, val, ref = [], [], [], []
    restarts: list[int] = []
    best_idx, best_val = 0, float("inf")

    for e in sp.iter_log(path):
        if e.get("event") == "epoch":
            ep, t, v = e.get("epoch"), e.get("train_loss"), e.get("val_loss")

            if ep is None or t is None or v is None:
                continue

            epochs.append(int(ep))
            train.append(float(t))
            val.append(float(v))

            if (r := e.get("ref_loss")) is not None:
                ref.append(float(r))

            if e.get("is_best") and v < best_val:
                best_idx, best_val = len(epochs) - 1, v
        elif e.get("event") == "restart":

            if (ep := e.get("epoch")) is not None:
                restarts.append(int(ep))

    if not epochs:
        return {}

    epochs = np.array(epochs, dtype=np.int64)
    train = np.array(train, dtype=np.float64)
    val = np.array(val, dtype=np.float64)
    out = {
        "epochs": epochs,
        "train": train,
        "val": val,
        "restarts": restarts,
        "best_epoch": int(epochs[best_idx]),
        "best_val": best_val,
        "final_val": float(val[-1]),
    }

    if ref:
        out["ref"] = np.array(ref, dtype=np.float64)
    return out


def plot_loss(
    log_path: str,
    output: str | None = None,
    alpha: float = 0.06,
    show_raw: bool = True,
    dpi: int = 200,
    fig_width: float = 13.0,
    fig_height: float = 7.0,
    show: bool = False,
) -> None:
    data = parse_log(log_path)

    if not data:
        print(f"error: no epoch entries in '{log_path}'.", file=sys.stderr)
        sys.exit(1)

    epochs = data["epochs"]
    t_loss, v_loss = data["train"], data["val"]
    restarts = data["restarts"]
    best_epoch, best_val, final_val = data["best_epoch"], data["best_val"], data["final_val"]
    n = len(epochs)

    t_smooth = sp.ema(t_loss, alpha)
    v_smooth = sp.ema(v_loss, alpha)

    sp.use_theme()
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # raw traces: the actual per-epoch wobble under the smoothed curve
    if show_raw and n > 1:
        ax.plot(epochs, t_loss, color=TRAIN, alpha=0.42, lw=0.8, ls=(0, (1, 2)), zorder=1)
        ax.plot(epochs, v_loss, color=VAL, alpha=0.42, lw=0.8, ls=(0, (1, 2)), zorder=1)

    # Generalization gap, both directions. Hatched warm where val rides above
    # train (the overfit gap); a quiet unhatched cool fill where val dips below
    # train (val set easier, or early-epoch noise), not overfit, so it reads
    # calmer. The min/max clamps make the two fills complementary, never overlap.
    sp.band(ax, epochs, t_smooth, np.maximum(t_smooth, v_smooth), VAL, alpha=0.08, hatch="////", zorder=2)
    sp.band(ax, epochs, np.minimum(t_smooth, v_smooth), t_smooth, TRAIN, alpha=0.06, hatch=None, zorder=2)

    # restart lines, with the first one labeled at the top
    for i, rx in enumerate(restarts):
        ax.axvline(rx, color=RESTART, ls=(0, (1, 2)), lw=1.0, alpha=0.40, zorder=2)

        if i == 0:
            ax.text(rx, 1.0, " restart", transform=ax.get_xaxis_transform(), va="bottom",
                    ha="left", fontsize=7.5, color=RESTART, alpha=0.7)

    # Focus the y-axis on the train/val curves, the data actually under study.
    # A reference baseline in a far-off magnitude regime (an untuned ref tens of
    # times the tuned loss) would otherwise own the scale and flatten the curves
    # to one dead line. So ref votes on the scale only when it's in the same
    # neighborhood; otherwise it's tagged at the edge, off-scale but acknowledged.
    core = [t_smooth, v_smooth, np.array([best_val])]

    if show_raw and n > 1:
        core += [t_loss, v_loss]

    core = np.concatenate(core)
    c_lo, c_hi = float(core.min()), float(core.max())
    c_span = (c_hi - c_lo) or 1.0
    y_lo, y_hi = c_lo - c_span * 0.10, c_hi + c_span * 0.10

    # smoothed curves; ref dashed and quiet
    r_end = None
    ref_offscale: tuple[float, bool] | None = None  # (value, is_above) when out of view

    if (r_loss := data.get("ref")) is not None:
        r_smooth = sp.ema(r_loss, alpha)
        r_end = float(r_smooth[-1])
        r_lo, r_hi = float(r_smooth.min()), float(r_smooth.max())

        if r_lo >= c_lo - c_span and r_hi <= c_hi + c_span:  # same neighborhood
            ax.plot(epochs, r_smooth, color=REF, lw=1.5, ls=(0, (5, 2)), alpha=0.85, zorder=3)
            y_lo, y_hi = min(y_lo, r_lo - c_span * 0.10), max(y_hi, r_hi + c_span * 0.10)
        else:
            ref_offscale = (r_end, r_end > c_hi)
    ax.plot(epochs, t_smooth, color=TRAIN, lw=2.0, zorder=4)
    ax.plot(epochs, v_smooth, color=VAL, lw=2.5, zorder=5)

    # best-val marker: gold dot + crosshair vline (value in the subtitle)
    ax.axvline(best_epoch, color=sp.GOLD, ls="--", lw=0.7, alpha=0.35, zorder=2)
    ax.scatter([best_epoch], [best_val], s=34, color=sp.GOLD, edgecolors="none", zorder=7)

    ax.set_xlabel("epoch", labelpad=8)
    ax.set_ylabel("loss", labelpad=8)
    ax.margins(x=0.01)
    ax.set_ylim(y_lo, y_hi)
    sp.style_axes(ax)

    # direct end-labels instead of a legend; all dodge each other
    entries = [(float(t_smooth[-1]), "train", TRAIN), (float(v_smooth[-1]), "val", VAL)]

    if r_end is not None and ref_offscale is None:
        entries.append((r_end, "ref", REF))
    sp.end_labels(ax, epochs[-1], entries)

    # Off-scale reference: tag it at the edge it ran off, value and direction,
    # so it's acknowledged without dragging the scale away from the curves.
    if ref_offscale is not None:
        rv, above = ref_offscale
        ax.text(0.995, 0.985 if above else 0.015,
                f"ref {'↑' if above else '↓'} {rv:.4f}",
                transform=ax.transAxes, ha="right", va="top" if above else "bottom",
                fontsize=8, color=REF, alpha=0.9)

    sp.title(
        fig, "training loss",
        f"{Path(log_path).stem}   ·   {sp.format_count(int(epochs[-1]))} epochs   ·   "
        f"final {final_val:.6f}   ·   best {best_val:.6f} @ {best_epoch}   ·   ema α={alpha}",
    )

    plt.subplots_adjust(top=0.89, bottom=0.09, left=0.08, right=0.90)

    out = output or Path(log_path).with_suffix("").as_posix() + "_loss.png"
    sp.save(fig, out, show=show, dpi=dpi)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot training and validation loss curves.",
        formatter_class=sp.HelpFormatter,
    )

    ap.add_argument("log", help="path to evaltune .jsonl log")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("-a", "--alpha", type=float, default=0.06, help="EMA factor (0, 1] (default: 0.06)")
    ap.add_argument("--no-raw", action="store_true", help="hide raw traces, smoothed only")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--width", type=float, default=13.0)
    ap.add_argument("--height", type=float, default=7.0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not (0.0 < args.alpha <= 1.0):
        ap.error("--alpha must be in (0, 1]")

    plot_loss(
        log_path=args.log, output=args.output, alpha=args.alpha, show_raw=not args.no_raw,
        dpi=args.dpi, fig_width=args.width, fig_height=args.height, show=args.show,
    )


if __name__ == "__main__":
    main()
