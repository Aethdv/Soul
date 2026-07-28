# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Training and validation loss curves.

One log draws EMA-smoothed train and val over faint raw traces, the best-val
epoch in gold, warm-restart lines, and a dashed fixed-K reference where the log
carries `ref_loss`. A reference too far off to share the scale moves to a strip
along the frame.

Several logs draw a comparison. One color per run down a purple→coral→gold ramp,
train and val in their own panels, and `--baseline` holds one run in dashed grey
for the rest to be read against.

    uv run scripts/plot_loss.py <evaltune.jsonl> [more.jsonl ...] [options]

Options:
    -o, --output PATH   image path (default: <log>_loss.png; _loss_compare.png for several)
    -a, --alpha FLOAT   EMA factor, (0, 1]  (default: 0.06)
    --val-only          validation curves only
    --train-only        training curves only
    --no-raw            hide raw traces, smoothed only
    --baseline PATH     draw this run dashed, as the line to beat
    --ylim LO,HI        pin the loss axis, so two renders share one scale
    --run N             which run in an append-only log (default: the longest completed)
    --dpi INT           output DPI (default: 300)
    --width / --height  figure size in inches (default: 13 × 7)
    --show              open interactively after saving
"""

from __future__ import annotations

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

import soulplot as sp

from dataclasses import dataclass, field
from pathlib import Path
from matplotlib.lines import Line2D


TRAIN = "#d8ba8a"      # wheat
VAL = "#f06e6b"        # coral
BEST_VAL = "#ff9d8f"   # light salmon
BEST_TRAIN = "#f4dcaa" # light wheat
REF = "#6fa5e0"        # cornflower
BASE = "#8b8b8b"       # gray
RESTART = "#b79be0"    # plum

# Purple → coral → gold, the ramp comparative runs are spread over. The first
# hue is negative because oklch_ramp lerps H linearly and the wheel is circular:
# written as 305 it would travel the long way round, through cyan and green.
RUN_STOPS = ((0.55, 0.17, -55.0), (0.68, 0.17, 25.0), (0.84, 0.13, 85.0))

RAW_TRIM = 5.0         # percent off each tail of a raw trace before it votes on the y-range
GUTTER = (0.90, 0.98)  # axes fractions an off-scale trace is redrawn between
GUTTER_RULE = 0.885    # where the line dividing that strip from the plot sits
GUTTER_CLEAR = 0.05    # axes fractions kept clear of tick labels below that line
AFFIX = "-_. "         # where a shared prefix or suffix may be cut off a run's name


@dataclass(slots=True)
class Run:
    path: str
    epochs: np.ndarray
    train: np.ndarray
    val: np.ndarray
    best_epoch: int
    best_val: float
    final_val: float
    best_train_epoch: int
    best_train: float
    restarts: list[int] = field(default_factory=list)
    ref: np.ndarray | None = None
    seed: int | None = None
    split_seed: int | None = None
    label: str | None = None

    @property
    def name(self) -> str:
        return self.label or Path(self.path).stem


def split_runs(events: list[dict]) -> list[list[dict]]:
    """Every run in an append-only log.

    A `run` header marks a start exactly. Older logs have none, so they are cut
    where the epoch number stops advancing, which is right for a sweep and wrong
    for a resume. Both rules apply, so a log that gained headers partway still
    splits the runs written before them.
    """
    runs: list[list[dict]] = []
    last_ep = 0

    for e in events:
        ep = e.get("epoch") if e.get("event") == "epoch" else None
        ep = int(ep) if ep is not None else None

        if e.get("event") == "run" or (ep is not None and (not runs or ep <= last_ep)):
            runs.append([])
            last_ep = 0

        if ep is not None:
            last_ep = ep

        if runs:
            runs[-1].append(e)

    return runs


def pick_run(runs: list[list[dict]]) -> int:
    """Index of the longest completed run, ties to the last.

    A sweep writes several runs of one length, so taking the last is harmless
    there. A probe run appended after a real one is two lines long.
    """
    def rank(i: int) -> tuple[bool, int, int]:
        run = runs[i]
        return (any(e.get("event") == "final" for e in run),
                sum(1 for e in run if e.get("event") == "epoch"), i)

    return max(range(len(runs)), key=rank)


def parse_log(path: str, want: int | None = None) -> Run | None:
    """Parse one run out of Soul's evaltune JSONL, or None if it holds no epochs.

    `want` is 1-based and picks a run outright; the default comes from `pick_run`.
    """
    runs = split_runs(list(sp.iter_log(path)))

    if not runs:
        return None

    idx = pick_run(runs) if want is None else max(0, min(want - 1, len(runs) - 1))

    if len(runs) > 1:
        counts = [sum(1 for e in r if e.get("event") == "epoch") for r in runs]
        print(f"  {Path(path).name}: {len(runs)} runs {counts}, plotting #{idx + 1}", file=sys.stderr)

    epochs, train, val, ref = [], [], [], []
    restarts: list[int] = []
    best_idx, best_val = 0, float("inf")
    final: dict | None = None

    for e in runs[idx]:
        event = e.get("event")

        if event == "epoch":
            ep, t, v = e.get("epoch"), e.get("train_loss"), e.get("val_loss")

            if ep is None or t is None or v is None:
                continue

            t, v = float(t), float(v)
            # Drop diverged/NaN epochs so they don't poison EMA and ylim
            if not (np.isfinite(t) and np.isfinite(v)):
                continue

            epochs.append(int(ep))
            train.append(t)
            val.append(v)
            # NaN-padded: appending only the finite ones desyncs it from epochs.
            r = e.get("ref_loss")
            ref.append(float(r) if r is not None else np.nan)

            if e.get("is_best") and v < best_val:
                best_idx, best_val = len(epochs) - 1, v
        elif event == "restart":
            if (ep := e.get("epoch")) is not None:
                restarts.append(int(ep))
        elif event == "final":
            final = e

    if not epochs:
        return None

    # Fallback if no epoch was ever explicitly marked is_best
    if not np.isfinite(best_val):
        best_idx = int(np.argmin(val))
        best_val = float(val[best_idx])

    epochs = np.array(epochs, dtype=np.int64)
    train = np.array(train, dtype=np.float64)
    val = np.array(val, dtype=np.float64)
    train_idx = int(np.argmin(train))

    run = Run(
        path=path,
        epochs=epochs,
        train=train,
        val=val,
        restarts=restarts,
        # Provisional; the final record overrides both below when the log has one.
        best_epoch=int(epochs[best_idx]),
        best_val=best_val,
        final_val=float(val[-1]),
        best_train_epoch=int(epochs[train_idx]),
        best_train=float(train[train_idx]),
    )

    ref = np.array(ref, dtype=np.float64)

    if np.isfinite(ref).any():
        run.ref = np.nan_to_num(ref, nan=float(np.nanmean(ref)))

    if final is not None:
        run.seed = final.get("seed")
        run.split_seed = final.get("split_seed")

        # Authoritative: selection is smoothed, so the epoch that shipped need
        # not be the raw argmin the loop above settled on. Logs written before
        # the trainer reported the train pair keep that argmin.
        for key, series, attrs in (("best_val_epoch", val, ("best_epoch", "best_val")),
                                   ("best_train_epoch", train, ("best_train_epoch", "best_train"))):
            if (be := final.get(key)) is None:
                continue

            if (hit := np.flatnonzero(epochs == int(be))).size:
                setattr(run, attrs[0], int(be))
                setattr(run, attrs[1], float(series[hit[0]]))

    return run


def plot_single(run: Run, *, alpha: float, metrics: tuple[str, ...], show_raw: bool,
                ylim: tuple[float, float] | None, fig_width: float, fig_height: float):
    epochs = run.epochs
    n = epochs.size
    want_train, want_val = "train" in metrics, "val" in metrics

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    t_smooth = v_smooth = None

    if want_train:
        t_smooth = _curve(ax, epochs, run.train, alpha, color=TRAIN, lw=2.0, raw=show_raw, zorder=4)

    if want_val:
        v_smooth = _curve(ax, epochs, run.val, alpha, color=VAL, lw=2.5, raw=show_raw, zorder=5)

    for i, rx in enumerate(run.restarts):
        ax.axvline(rx, color=RESTART, ls=(0, (1, 2)), lw=1.0, alpha=0.40, zorder=2)

        if i == 0:
            ax.text(rx, 1.0, " restart", transform=ax.get_xaxis_transform(), va="bottom",
                    ha="left", fontsize=7.5, color=RESTART, alpha=0.7)

    core = [s for s in (t_smooth, v_smooth) if s is not None]

    if show_raw and n > 1:
        core += [_envelope(run.train)] if want_train else []
        core += [_envelope(run.val)] if want_val else []

    if want_val:
        core.append(np.array([run.best_val]))

    c_lo, c_hi = _y_range(core)

    if ylim is not None:  # a pinned window is the one the ref gets judged against too
        c_lo, c_hi = ylim

    c_span = (c_hi - c_lo) or 1.0
    y_lo, y_hi = c_lo - c_span * 0.10, c_hi + c_span * 0.10

    r_end = None
    off_scale: tuple[np.ndarray, bool] | None = None  # (smoothed ref, is_above) for the gutter

    # The ref is scored on the validation split, so it means nothing beside a train
    # curve alone. An untuned one sits an order of magnitude off and would flatten
    # the curves to a line, so it shares the scale only from nearby.
    if run.ref is not None and want_val:
        r_smooth = sp.ema(run.ref, alpha)
        r_end = float(r_smooth[-1])
        r_lo, r_hi = float(r_smooth.min()), float(r_smooth.max())

        if r_lo >= c_lo - c_span and r_hi <= c_hi + c_span:
            ax.plot(epochs, r_smooth, color=REF, lw=1.5, ls=(0, (5, 2)), alpha=0.85, zorder=3)
            y_lo, y_hi = min(y_lo, r_lo - c_span * 0.10), max(y_hi, r_hi + c_span * 0.10)
        else:
            off_scale = (r_smooth, r_end > c_hi)

    # Placed on the smoothed curve; the raw values are in the subtitle.
    # searchsorted, because a resumed log can skip epochs.
    def on_curve(smooth: np.ndarray, epoch: int) -> float:
        return float(smooth[min(int(np.searchsorted(epochs, epoch)), len(smooth) - 1)])

    if want_val:
        ax.axvline(run.best_epoch, color=BEST_VAL, ls="--", lw=0.7, alpha=0.35, zorder=2)

    if t_smooth is not None:
        sp.dot(ax, [run.best_train_epoch], [on_curve(t_smooth, run.best_train_epoch)], BEST_TRAIN, size=46, zorder=6)

    if v_smooth is not None:
        sp.dot(ax, [run.best_epoch], [on_curve(v_smooth, run.best_epoch)], BEST_VAL, size=52, zorder=7)

    ax.set_xlabel("epoch", labelpad=8)
    ax.set_ylabel("loss", labelpad=8)
    ax.margins(x=0.01)
    # Grow the window so the curves stop at the rule and the strip is left to the
    # ref. A pinned window is the caller's, so it stays as given.
    if off_scale is not None and ylim is None:
        grown = (y_hi - y_lo) / GUTTER_RULE
        y_lo, y_hi = (y_lo, y_lo + grown) if off_scale[1] else (y_hi - grown, y_hi)

    ax.set_ylim(*(ylim or (y_lo, y_hi)))
    sp.style_axes(ax, grain=False)

    if off_scale is not None:
        _gutter(ax, epochs, off_scale[0], above=off_scale[1])

    entries = [(float(s[-1]), label, color) for s, label, color
               in ((t_smooth, "train", TRAIN), (v_smooth, "val", VAL)) if s is not None]

    if r_end is not None and off_scale is None:
        entries.append((r_end, "ref", REF))
    sp.end_labels(ax, epochs[-1], entries)

    facts = [
        *([] if run.name == "evaltune" else [run.name]),
        f"{sp.format_count(int(epochs[-1]))} epochs",
        *([f"final {run.final_val:.6f}", f"val {run.best_val:.6f} @{run.best_epoch}"] if want_val else []),
        *([f"train {run.best_train:.6f} @{run.best_train_epoch}"] if want_train else []),
        f"ema α={alpha}",
    ]

    # Twenty digits apiece; on the same row they swamp the numbers above.
    seeds = [f"{label} {value}" for value, label in ((run.seed, "seed"), (run.split_seed, "split"))
             if value is not None]

    sp.title(fig, "evaltune", "   ·   ".join(facts), "   ·   ".join(seeds) or None)
    fig.subplots_adjust(top=0.89, bottom=0.09, left=0.08, right=0.90)
    return fig


def plot_compare(runs: list[Run], *, alpha: float, metrics: tuple[str, ...], show_raw: bool,
                 baseline: Run | None, ylim: tuple[float, float] | None,
                 fig_width: float, fig_height: float):
    """Runs side by side, one panel per metric, one ramp color per run.

    Train and val split apart because stacking both for every run puts twice the
    curves on one axis and the colors stop identifying anything. Each panel then
    scales to its own metric: the two sit further apart than either varies, so
    one shared axis would spend most of both panels drawing empty space.
    """
    styled = _style_runs(runs, baseline)
    fig, grid = plt.subplots(1, len(metrics), figsize=(fig_width, fig_height), squeeze=False)
    axes = list(grid[0])
    # All the traces share one axis, so per-trace opacity falls as runs pile up
    # and the total ink stays near where two runs put it.
    raw_alpha = min(0.13, 0.26 / len(runs))

    for ax, metric in zip(axes, metrics):
        panel = []

        for i, (run, pen) in enumerate(styled):
            y = run.train if metric == "train" else run.val
            panel.append(_curve(ax, run.epochs, y, alpha, raw=show_raw, raw_alpha=raw_alpha,
                                zorder=3 + 0.01 * i, **pen))

            if show_raw and y.size > 1:
                panel.append(_envelope(y))

        lo, hi = _y_range(panel)
        span = (hi - lo) or 1.0
        ax.set_ylim(*(ylim or (lo - span * 0.10, hi + span * 0.10)))
        ax.set_title(metric, loc="left", fontsize=9.5, color=sp.MUTE, pad=10)
        ax.set_xlabel("epoch", labelpad=8)
        ax.margins(x=0.01)
        sp.style_axes(ax, grain=False)

    axes[0].set_ylabel("loss", labelpad=8)

    facts = [
        f"{len(runs)} runs",
        *([f"baseline {baseline.name}"] if baseline is not None else []),
        f"{sp.format_count(int(max(run.epochs[-1] for run in runs)))} epochs",
        f"ema α={alpha}",
    ]

    sp.title(fig, "evaltune", "   ·   ".join(facts))
    # Before the keys go down, so they read the geometry the panels end up with.
    fig.subplots_adjust(top=0.87, bottom=0.10, left=0.07, right=0.97, wspace=0.16)

    width = max(len(run.name) for run in runs)

    for ax, metric in zip(axes, metrics):
        _panel_legend(ax, styled, metric, width)
    return fig


def _curve(ax, x, y, alpha: float, *, color, lw: float, raw: bool, raw_alpha: float = 0.16,
           ls="-", zorder: float = 4) -> np.ndarray:
    """A faint raw trace under the EMA-smoothed line; returns the smoothed curve.

    The trace is drawn solid. Past a thousand epochs there are more samples than
    pixel columns, so the page already shows a per-column smear of the noise, and
    a dash pattern chops that into static.
    """
    if raw and y.size > 1:
        # Halo under hairline: one pass is either too faint to see or solid black
        # wherever the samples crowd.
        ax.plot(x, y, color=color, alpha=raw_alpha * 0.5, lw=2.4, zorder=0.9)
        ax.plot(x, y, color=color, alpha=raw_alpha * 1.9, lw=0.45, zorder=1)

    smooth = sp.ema(y, alpha)
    ax.plot(x, smooth, color=color, lw=lw, ls=ls, zorder=zorder)
    return smooth


def _gutter(ax, x, y, *, above: bool) -> None:
    """Redraw an off-scale trace as a sparkline in a strip along the frame.

    A ref sitting far off would flatten the tuned curves into one line if it
    shared their scale, and its shape is still worth reading. The strip keeps that
    same y-scale where it fits, since a strip normalized to its own span draws a
    ref that barely moves as a mountain; only a swing too tall to fit is
    compressed.

    Runs after the axis is final, because it prunes the ticks the strip covers: a
    tick label level with the sparkline reads as the sparkline's value.
    """
    lo, hi = float(np.min(y)), float(np.max(y))
    band = GUTTER if above else (1.0 - GUTTER[1], 1.0 - GUTTER[0])
    depth = band[1] - band[0]
    y0, y1 = ax.get_ylim()
    room = depth * (y1 - y0)

    if hi - lo <= room:
        fraction = 0.5 * (band[0] + band[1]) + (y - 0.5 * (lo + hi)) / (y1 - y0)
    else:
        fraction = band[0] + (y - lo) / (hi - lo) * depth

    rule = GUTTER_RULE if above else 1.0 - GUTTER_RULE
    clear = GUTTER_CLEAR if above else -GUTTER_CLEAR
    cut = y0 + (rule - clear) * (y1 - y0)
    ax.set_yticks([t for t in ax.get_yticks() if (t <= cut if above else t >= cut)])

    # Over the raw traces and under everything else, so a spike reaching up from
    # the curves cannot be read as part of the sparkline.
    ax.fill_between([0.0, 1.0], rule, float(above), transform=ax.transAxes,
                    color=sp.INK, lw=0, zorder=1.5)
    # The only mark separating the strip's scale from the plot's, kept subtle
    # enough to read as a gridline.
    ax.plot([0.0, 1.0], [rule, rule], transform=ax.transAxes, color=sp.MUTE, lw=0.9,
            ls=(0, (6, 3)), alpha=0.65, zorder=2)
    # x in data, y in axes fractions, so the strip stays put whatever the axis does.
    ax.plot(x, fraction, transform=ax.get_xaxis_transform(), color=REF, lw=1.2, ls=(0, (1, 1.8)),
            alpha=0.95, zorder=3)
    # Outside the frame, because the sparkline spans the full width and would run
    # straight through a label placed inside it.
    ax.text(1.008, sum(band) / 2, f"ref {'↑' if above else '↓'} {float(y[-1]):.6f}",
            transform=ax.transAxes, ha="left", va="center", fontsize=7.5, color=REF, zorder=4)


def _envelope(y: np.ndarray) -> np.ndarray:
    """The central band of a raw trace, which is what sizes the y-range.

    Min and max would hand the scale to a handful of spikes and squash the
    smoothed curves into a ribbon across the middle. Trimmed, the hair fills
    roughly the band the EMA does and the outliers clip at the frame.
    """
    finite = y[np.isfinite(y)]

    if finite.size == 0:
        return finite
    return np.percentile(finite, [RAW_TRIM, 100.0 - RAW_TRIM])


def _style_runs(runs: list[Run], baseline: Run | None) -> list[tuple[Run, dict]]:
    """Pair each run with the pen it's drawn with, the baseline first so it sits underneath.

    A pen is plot kwargs; one dict paints the curve and its legend proxy, so
    neither can drift from the other.
    """
    others = [run for run in runs if run is not baseline]
    colors = _run_colors(len(others))
    styled = [(run, {"color": color, "lw": 1.8}) for run, color in zip(others, colors)]

    if baseline is not None:
        styled.insert(0, (baseline, {"color": BASE, "ls": (0, (6, 3)), "lw": 1.5}))
    return styled


def _run_colors(n: int) -> list[tuple[float, float, float, float]]:
    """`n` colors spread down the run ramp; a lone run takes the coral middle."""
    ramp = sp.oklch_ramp(RUN_STOPS)

    if n <= 1:
        return [ramp(0.5)]
    return [ramp(float(t)) for t in np.linspace(0.06, 0.96, n)]


def _short_labels(runs: list[Run]) -> None:
    """Label each run with what's left after the part every filename shares.

    A sweep names its logs alike, `4k-evaltune` beside `2k-evaltune`, and the
    shared half carries nothing. Cut only at a separator, and only while every
    name keeps at least two characters.
    """
    stems = [Path(run.path).stem for run in runs]

    if len(stems) < 2:
        return

    head = stems[0]

    for stem in stems[1:]:
        while head and not stem.startswith(head):
            head = head[:-1]

    while head and head[-1] not in AFFIX:
        head = head[:-1]

    if head and all(len(stem) - len(head) >= 2 for stem in stems):
        stems = [stem[len(head):] for stem in stems]

    tail = stems[0]

    for stem in stems[1:]:
        while tail and not stem.endswith(tail):
            tail = tail[1:]

    cut = [stem[:len(stem) - len(tail)] for stem in stems]

    if tail and all(len(part) > 1 and part[-1] in AFFIX for part in cut):
        stems = [part[:-1] for part in cut]

    for run, stem in zip(runs, stems):
        run.label = stem


def _panel_legend(ax, styled: list[tuple[Run, dict]], metric: str, width: int) -> None:
    """Key the runs inside `ax`, each row carrying this panel's own best loss.

    One key per panel, because the number beside a run has to be the number that
    panel draws; a train panel keyed with val losses would be wrong with nothing
    on the page to contradict it. Names pad to a column so mono stacks the losses.
    """
    entries = []

    for run, pen in styled:
        best = run.best_val if metric == "val" else run.best_train
        entries.append((Line2D([0], [0], **pen), f"{run.name:<{width}}  {best:.6f}"))

    corner = _quiet_corner(ax, rows=len(entries) + 1, chars=max(len(t) for _, t in entries))
    sp.legend(ax, entries, anchor=None, loc=corner, pad=1.2, title="best")


def _quiet_corner(ax, *, rows: int, chars: int, size: float = 8.5) -> str:
    """The corner of `ax` holding the least ink, probed with the key's own footprint.

    Measured off the lines already drawn, so it needs no plumbing from the caller.
    Loss curves decay left to right and usually leave the top right open, but a run
    that diverges late fills exactly that corner. Ties go to the first box listed.
    """
    box = ax.get_window_extent()
    em = size * ax.figure.dpi / 72.0                        # one font size, in pixels
    h = min(0.9, (rows * 1.5 + 1.2) * em / box.height)      # rows at their pitch, plus the pad
    w = min(0.9, (chars * 0.62 + 3.4) * em / box.width)     # mono advance, plus handle and pad
    boxes = {                                               # x0, x1, y0, y1, in axes fractions
        "upper right": (1.0 - w, 1.0, 1.0 - h, 1.0),
        "upper left": (0.0, w, 1.0 - h, 1.0),
        "lower right": (1.0 - w, 1.0, 0.0, h),
        "lower left": (0.0, w, 0.0, h),
    }
    data = [line.get_xydata() for line in ax.lines if line.get_xydata().size]

    if not data:
        return "upper right"

    points = np.vstack(data)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    fx = (points[:, 0] - x0) / ((x1 - x0) or 1.0)
    fy = (points[:, 1] - y0) / ((y1 - y0) or 1.0)

    counts = {loc: int(np.count_nonzero((fx >= bx0) & (fx <= bx1) & (fy >= by0) & (fy <= by1)))
              for loc, (bx0, bx1, by0, by1) in boxes.items()}
    return min(counts.items(), key=lambda kv: kv[1])[0]


def _y_range(curves: list[np.ndarray]) -> tuple[float, float]:
    """The y-range a compare panel is scaled to.

    A cold start ramps down from a random eval over its first few hundred epochs,
    covering a thousand times what the tails being compared differ by, so a range
    taken from its extremes draws every other curve as one flat line.

    The tightest run sets the initial range and the rest widen it while they sit
    within one span of slack. A run outside that contributes its trimmed envelope
    instead, and its head clips at the frame.
    """
    usable = [np.asarray(c, dtype=np.float64) for c in curves if np.size(c) and np.isfinite(c).any()]
    # A lone marker value cannot seed the range: its span is zero, and a zero-span
    # range admits everything on the slack test below.
    ranked = sorted((c for c in usable if c.size > 1), key=lambda c: np.ptp(c[np.isfinite(c)]))

    if not ranked:
        return _bounds(usable)

    lo, hi = _bounds([ranked[0]])

    for curve in ranked[1:] + [c for c in usable if c.size <= 1]:
        span = (hi - lo) or 1.0
        c_lo, c_hi = _bounds([curve])

        if c_lo < lo - span or c_hi > hi + span:
            c_lo, c_hi = _bounds([_envelope(curve)])
        lo, hi = min(lo, c_lo), max(hi, c_hi)

    return lo, hi


def _bounds(series) -> tuple[float, float]:
    if not series:
        return 0.0, 1.0

    values = np.concatenate([np.asarray(s, dtype=np.float64).ravel() for s in series])
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return 0.0, 1.0
    return float(finite.min()), float(finite.max())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot training and validation loss curves.",
        formatter_class=sp.HelpFormatter,
    )

    ap.add_argument("logs", nargs="+", help="evaltune .jsonl logs; two or more compare")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("-a", "--alpha", type=float, default=0.06, help="EMA factor (0, 1] (default: 0.06)")
    metric = ap.add_mutually_exclusive_group()
    metric.add_argument("--val-only", action="store_true", help="validation curves only")
    metric.add_argument("--train-only", action="store_true", help="training curves only")
    ap.add_argument("--no-raw", action="store_true", help="hide raw traces, smoothed only")
    ap.add_argument("--baseline", default=None, help="draw this run dashed, as the line to beat")
    ap.add_argument("--ylim", default=None, help="pin the loss axis, lo,hi (default: fit the data)")
    ap.add_argument("--run", type=int, default=None, metavar="N",
                    help="which run in an append-only log, 1-based (default: the longest completed)")
    ap.add_argument("--dpi", type=int, default=sp.DPI)
    ap.add_argument("--width", type=float, default=13.0)
    ap.add_argument("--height", type=float, default=7.0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not (0.0 < args.alpha <= 1.0):
        ap.error("--alpha must be in (0, 1]")

    ylim = None

    if args.ylim:
        try:
            lo, hi = (float(v) for v in args.ylim.split(","))
        except ValueError:
            ap.error("--ylim takes two numbers, lo,hi")

        if not lo < hi:
            ap.error("--ylim takes lo < hi")
        ylim = (lo, hi)

    paths = list(args.logs)
    base_path = Path(args.baseline).resolve() if args.baseline else None

    # A baseline named but not listed is still drawn; it selects nothing.
    if base_path is not None and not any(Path(p).resolve() == base_path for p in paths):
        paths.insert(0, args.baseline)

    runs = []

    # One unreadable path out of five shouldn't cost the other four their plot.
    for path in paths:
        try:
            run = parse_log(path, args.run)
        except OSError as exc:
            print(f"  [warn] {path}: {exc.strerror or exc}", file=sys.stderr)
            continue

        if run is None:
            print(f"  [warn] {path}: no epoch entries", file=sys.stderr)
            continue

        runs.append(run)

    if not runs:
        print("error: no run to plot.", file=sys.stderr)
        sys.exit(1)

    baseline = next((run for run in runs if Path(run.path).resolve() == base_path), None)
    metrics = ("val",) if args.val_only else ("train",) if args.train_only else ("train", "val")

    _short_labels(runs)
    sp.use_theme()

    if len(runs) == 1:
        fig = plot_single(runs[0], alpha=args.alpha, metrics=metrics, show_raw=not args.no_raw,
                          ylim=ylim, fig_width=args.width, fig_height=args.height)
        suffix = "_loss.png"
    else:
        fig = plot_compare(runs, alpha=args.alpha, metrics=metrics, show_raw=not args.no_raw,
                           baseline=baseline, ylim=ylim, fig_width=args.width, fig_height=args.height)
        suffix = "_loss_compare.png"

    # Named off the first log the caller listed, ignoring the baseline inserted above.
    out = args.output or Path(args.logs[0]).with_suffix("").as_posix() + suffix
    sp.save(fig, out, show=args.show, dpi=args.dpi)


if __name__ == "__main__":
    main()
