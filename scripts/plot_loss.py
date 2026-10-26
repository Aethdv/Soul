# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Training and validation loss curves.

    uv run scripts/plot_loss.py <evaltune.jsonl> [more.jsonl ...] [options]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

import soulplot as sp


TRAIN = "#d8ba8a"      # muted yellow-tan
VAL = "#f06e6b"        # warm coral red
BEST_VAL = "#ff9d8f"   # soft salmon pink
BEST_TRAIN = "#f4dcaa" # pale warm yellow
REF = "#6fa5e0"        # soft blue
BASELINE = "#8b8b8b"   # neutral gray
RESTART = "#b79be0"    # soft lavender-purple
DELTA = "#7dc2a7"      # muted seafoam green

RUN_COLOR_STOPS = ((0.66, 0.15, -55.0), (0.70, 0.16, 25.0), (0.74, 0.13, 85.0))

RANGE_TRIM_PERCENTILE = 5.0
REF_GUTTER_BAND = (0.90, 0.98)
REF_GUTTER_RULE = 0.885
REF_GUTTER_TICK_CLEAR = 0.05
LABEL_SEPARATORS = "-_. /"


@dataclass(slots=True)
class Run:
    path: str
    epochs: np.ndarray
    train_loss: np.ndarray
    val_loss: np.ndarray
    best_val_epoch: int | None = None
    best_val_loss: float | None = None
    best_train_epoch: int | None = None
    best_train_loss: float | None = None
    restart_epochs: list[int] = field(default_factory=list)
    ref_loss: np.ndarray | None = None
    moved_params: np.ndarray | None = None
    param_count: int | None = None
    dataset: str | None = None
    seed: int | None = None
    split_seed: int | None = None
    label: str | None = None

    @property
    def name(self) -> str:
        return self.label or Path(self.path).stem

    @property
    def has_val_loss(self) -> bool:
        return bool(self.val_loss.size and np.isfinite(self.val_loss).all())


def split_log_runs(events: list[dict]) -> list[list[dict]]:
    """Split append-only events into logical runs, stitching checkpoint continuations."""
    runs: list[list[dict]] = []
    run_header: dict | None = None
    last_epoch: int | None = None

    for entry in events:
        event = entry.get("event")

        if event == "run":
            start_epoch = entry.get("start_epoch")
            continues_run = (
                bool(runs)
                and run_header is not None
                and last_epoch is not None
                and start_epoch is not None
                and int(start_epoch) == last_epoch + 1
                and all(run_header.get(key) == entry.get(key)
                        for key in ("seed", "split_seed", "dataset", "params"))
            )

            if continues_run:
                runs[-1].append(entry)
            else:
                runs.append([entry])
                run_header = entry
                last_epoch = None
            continue

        epoch = int(entry["epoch"]) if event == "epoch" and entry.get("epoch") is not None else None

        # Headerless logs can only expose a run boundary by resetting the epoch.
        if epoch is not None and (
            not runs or (run_header is None and last_epoch is not None and epoch <= last_epoch)
        ):
            runs.append([])
            run_header = None

        if runs:
            runs[-1].append(entry)

        if epoch is not None:
            last_epoch = epoch

    return runs


def default_run_index(runs: list[list[dict]]) -> int:
    """Longest completed run, ties to the last."""
    def rank(index: int) -> tuple[bool, int, int]:
        entries = runs[index]
        completed = bool(entries) and entries[-1].get("event") == "final"
        epoch_count = sum(1 for entry in entries if entry.get("event") == "epoch")
        return completed, epoch_count, index

    return max(range(len(runs)), key=rank)


def parse_log(path: str, run_number: int | None = None) -> Run | None:
    logical_runs = split_log_runs(list(sp.iter_log(path)))

    if not logical_runs:
        return None

    if run_number is None:
        run_index = default_run_index(logical_runs)
    elif 1 <= run_number <= len(logical_runs):
        run_index = run_number - 1
    else:
        raise IndexError(f"{Path(path).name}: --run {run_number} outside 1..{len(logical_runs)}")

    if len(logical_runs) > 1:
        epoch_counts = [sum(1 for entry in entries if entry.get("event") == "epoch")
                        for entries in logical_runs]
        print(f"  {Path(path).name}: {len(logical_runs)} runs {epoch_counts}, plotting #{run_index + 1}",
              file=sys.stderr)

    entries = logical_runs[run_index]
    run_header = next((entry for entry in entries if entry.get("event") == "run"), {})
    epochs, train_loss, val_loss, ref_loss, moved_params = [], [], [], [], []
    restart_epochs: list[int] = []
    best_val_epoch: int | None = None
    best_val_loss: float | None = None
    best_train_epoch: int | None = None
    best_train_loss: float | None = None
    final_event: dict | None = None

    for entry in entries:
        event = entry.get("event")

        if event == "epoch":
            epoch, train_value = entry.get("epoch"), entry.get("train_loss")
            if epoch is None or train_value is None:
                continue

            train_value = float(train_value)
            if not np.isfinite(train_value):
                continue

            val_value = entry.get("val_loss")
            val_value = float(val_value) if val_value is not None else np.nan
            epoch = int(epoch)

            epochs.append(epoch)
            train_loss.append(train_value)
            val_loss.append(val_value if np.isfinite(val_value) else np.nan)

            ref_value, moved_value = entry.get("ref_loss"), entry.get("moved")
            ref_loss.append(float(ref_value) if ref_value is not None else np.nan)
            moved_params.append(float(moved_value) if moved_value is not None else np.nan)

            if entry.get("is_best"):
                if np.isfinite(val_value):
                    best_val_epoch, best_val_loss = epoch, val_value
                else:
                    best_train_epoch, best_train_loss = epoch, train_value
        elif event == "run":
            # A stitched continuation makes an earlier final an intermediate checkpoint boundary.
            final_event = None
        elif event == "restart":
            if (epoch := entry.get("epoch")) is not None:
                restart_epochs.append(int(epoch))
        elif event == "final":
            final_event = entry

    if not epochs:
        return None

    run = Run(
        path=path,
        epochs=np.asarray(epochs, dtype=np.int64),
        train_loss=np.asarray(train_loss, dtype=np.float64),
        val_loss=np.asarray(val_loss, dtype=np.float64),
        best_val_epoch=best_val_epoch,
        best_val_loss=best_val_loss,
        best_train_epoch=best_train_epoch,
        best_train_loss=best_train_loss,
        restart_epochs=restart_epochs,
        seed=run_header.get("seed"),
        split_seed=run_header.get("split_seed"),
        param_count=run_header.get("params"),
        dataset=run_header.get("dataset"),
    )

    ref_loss = np.asarray(ref_loss, dtype=np.float64)
    moved_params = np.asarray(moved_params, dtype=np.float64)

    # Missing telemetry is not zero or an average of its neighbors. Omit the trace.
    if ref_loss.size and np.isfinite(ref_loss).all():
        run.ref_loss = ref_loss
    if moved_params.size and np.isfinite(moved_params).all():
        run.moved_params = moved_params

    if final_event is not None:
        run.seed = final_event.get("seed", run.seed)
        run.split_seed = final_event.get("split_seed", run.split_seed)

        if run.param_count is None and isinstance(
            params := final_event.get("best_val_params") or final_event.get("params"), list
        ):
            run.param_count = len(params)

        if (epoch := final_event.get("best_val_epoch")) is not None and (
            loss := final_event.get("best_val_loss")
        ) is not None:
            run.best_val_epoch = int(epoch)
            run.best_val_loss = float(loss)

        if (epoch := final_event.get("best_train_epoch")) is not None and (
            loss := final_event.get("best_train_loss")
        ) is not None:
            run.best_train_epoch = int(epoch)
            run.best_train_loss = float(loss)

    return run


def plot_single(run: Run, *, alpha: float, metrics: tuple[str, ...], show_raw: bool,
                show_deltas: bool, ylim: tuple[float, float] | None,
                fig_width: float, fig_height: float):
    epochs = run.epochs
    plot_train, plot_val = "train" in metrics, "val" in metrics
    moved_params = run.moved_params if show_deltas else None

    if moved_params is None:
        fig, loss_ax = plt.subplots(figsize=(fig_width, fig_height))
        delta_ax = None
    else:
        fig, (loss_ax, delta_ax) = plt.subplots(2, 1, figsize=(fig_width, fig_height),
                                                height_ratios=[4, 1], sharex=True)

    train_smooth = val_smooth = None

    if plot_train:
        train_smooth = _plot_series(loss_ax, epochs, run.train_loss, alpha,
                                    color=TRAIN, lw=2.0, raw=show_raw, raw_alpha=0.12, zorder=4)
    if plot_val:
        val_smooth = _plot_series(loss_ax, epochs, run.val_loss, alpha,
                                  color=VAL, lw=2.5, raw=show_raw, raw_alpha=0.12, zorder=5)

    for restart_index, epoch in enumerate(run.restart_epochs):
        loss_ax.axvline(epoch, color=RESTART, ls=(0, (1, 2)), lw=1.0, alpha=0.40, zorder=2)
        if restart_index == 0:
            loss_ax.text(epoch, 1.0, " restart", transform=loss_ax.get_xaxis_transform(), va="bottom",
                         ha="left", fontsize=7.5, color=RESTART, alpha=0.7)

    def point_at_epoch(epoch: int | None, loss: float | None) -> tuple[int, float] | None:
        if epoch is None or loss is None:
            return None

        index = int(np.searchsorted(epochs, epoch))
        if index >= epochs.size or epochs[index] != epoch:
            return None

        return epoch, loss

    best_train_point = point_at_epoch(run.best_train_epoch, run.best_train_loss) if plot_train else None
    best_val_point = point_at_epoch(run.best_val_epoch, run.best_val_loss) if plot_val else None

    range_series = [series for series in (train_smooth, val_smooth) if series is not None]
    if show_raw and epochs.size > 1:
        if plot_train:
            range_series.append(_trimmed_bounds(run.train_loss))
        if plot_val:
            range_series.append(_trimmed_bounds(run.val_loss))

    loss_lo, loss_hi = _loss_range(range_series)
    if ylim is not None:  # the pinned window is also the reference's comparison scale.
        loss_lo, loss_hi = ylim

    loss_span = _range_span(loss_lo, loss_hi)
    y_lo, y_hi = loss_lo - loss_span * 0.10, loss_hi + loss_span * 0.10
    ref_end = None
    gutter_ref: np.ndarray | None = None
    ref_above = False

    if run.ref_loss is not None and plot_val:
        ref_smooth = sp.ema(run.ref_loss, alpha)
        ref_end = float(ref_smooth[-1])
        ref_lo, ref_hi = float(ref_smooth.min()), float(ref_smooth.max())

        if ref_lo >= loss_lo - loss_span and ref_hi <= loss_hi + loss_span:
            loss_ax.plot(epochs, ref_smooth, color=REF, lw=1.5, ls=(0, (5, 2)), alpha=0.85, zorder=3)
            y_lo = min(y_lo, ref_lo - loss_span * 0.10)
            y_hi = max(y_hi, ref_hi + loss_span * 0.10)
        else:
            gutter_ref = ref_smooth
            ref_above = ref_hi - loss_hi >= loss_lo - ref_lo

    if best_val_point is not None:
        epoch, y = best_val_point
        loss_ax.axvline(epoch, color=BEST_VAL, ls="--", lw=0.7, alpha=0.35, zorder=2)
        sp.dot(loss_ax, [epoch], [y], BEST_VAL, size=52, zorder=7)

    if best_train_point is not None:
        epoch, y = best_train_point
        sp.dot(loss_ax, [epoch], [y], BEST_TRAIN, size=46, zorder=6)

    (loss_ax if delta_ax is None else delta_ax).set_xlabel("epoch", labelpad=8)
    loss_ax.set_ylabel("loss", labelpad=8)
    loss_ax.margins(x=0.01)

    if gutter_ref is not None and ylim is None:
        grown = (y_hi - y_lo) / REF_GUTTER_RULE
        y_lo, y_hi = (y_lo, y_lo + grown) if ref_above else (y_hi - grown, y_hi)

    loss_ax.set_ylim(*(ylim or (y_lo, y_hi)))
    sp.style_axes(loss_ax, grain=False)

    if gutter_ref is not None:
        _plot_ref_gutter(loss_ax, epochs, gutter_ref, above=ref_above)

    end_labels = [(float(series[-1]), label, color)
                  for series, label, color in ((train_smooth, "train", TRAIN), (val_smooth, "val", VAL))
                  if series is not None]
    if ref_end is not None and gutter_ref is None:
        end_labels.append((ref_end, "ref", REF))
    sp.end_labels(loss_ax, epochs[-1], end_labels)

    if moved_params is not None:
        _plot_delta_strip(delta_ax, run, moved_params, alpha, raw=show_raw,
                          best_val_epoch=run.best_val_epoch if best_val_point is not None else None)

    first_epoch, last_epoch = int(epochs[0]), int(epochs[-1])
    epoch_fact = (f"{sp.format_count(last_epoch)} epochs" if first_epoch == 1
                  else f"epochs {sp.format_count(first_epoch)}–{sp.format_count(last_epoch)}")
    facts = [] if run.name == "evaltune" else [run.name]
    facts.append(epoch_fact)

    if plot_val:
        facts.append(f"last val {float(run.val_loss[-1]):.6f}")
        if run.best_val_loss is not None:
            facts.append(f"best val {run.best_val_loss:.6f} @{run.best_val_epoch}")
    if plot_train and run.best_train_loss is not None:
        facts.append(f"best train {run.best_train_loss:.6f} @{run.best_train_epoch}")
    if moved_params is not None and run.param_count:
        facts.append(f"{run.param_count} params")
    facts.append(f"ema α={alpha}")

    # Twenty digits apiece; keep them off the already dense facts row.
    seed_facts = [f"{label} {value}" for value, label in ((run.seed, "seed"), (run.split_seed, "split"))
                  if value is not None]

    sp.title(fig, "evaltune", "   ·   ".join(facts), "   ·   ".join(seed_facts) or None)
    fig.subplots_adjust(top=0.89, bottom=0.09, left=0.08, right=0.90, hspace=0.09)
    return fig


def plot_compare(runs: list[Run], *, alpha: float, metrics: tuple[str, ...], show_raw: bool,
                 baseline: Run | None, comparison_note: str | None, ylim: tuple[float, float] | None,
                 fig_width: float, fig_height: float):
    styled_runs = _style_runs(runs, baseline)
    fig, grid = plt.subplots(1, len(metrics), figsize=(fig_width, fig_height), squeeze=False)
    axes = list(grid[0])

    raw_alpha = min(0.13, 0.26 / len(runs))

    for ax, metric in zip(axes, metrics):
        range_series = []

        for draw_index, (run, style) in enumerate(styled_runs):
            loss = run.train_loss if metric == "train" else run.val_loss
            range_series.append(_plot_series(ax, run.epochs, loss, alpha, raw=show_raw,
                                              raw_alpha=raw_alpha, zorder=3 + 0.01 * draw_index, **style))
            if show_raw and loss.size > 1:
                range_series.append(_trimmed_bounds(loss))

        loss_lo, loss_hi = _loss_range(range_series)
        loss_span = _range_span(loss_lo, loss_hi)
        ax.set_ylim(*(ylim or (loss_lo - loss_span * 0.10, loss_hi + loss_span * 0.10)))
        ax.set_title(metric, loc="left", fontsize=9.5, color=sp.MUTE, pad=10)
        ax.set_xlabel("epoch", labelpad=8)
        ax.margins(x=0.01)
        sp.style_axes(ax, grain=False)

    axes[0].set_ylabel("loss", labelpad=8)

    max_epoch = int(max(run.epochs[-1] for run in runs))
    epoch_fact = (f"{sp.format_count(max_epoch)} epochs" if all(run.epochs[0] == 1 for run in runs)
                  else f"through epoch {sp.format_count(max_epoch)}")
    facts = [f"{len(runs)} runs"]
    if baseline is not None:
        facts.append(f"baseline {baseline.name}")
    if comparison_note is not None:
        facts.append(comparison_note)
    facts.extend((epoch_fact, f"ema α={alpha}"))

    sp.title(fig, "evaltune", "   ·   ".join(facts))
    fig.subplots_adjust(top=0.87, bottom=0.10, left=0.07, right=0.97, wspace=0.16)

    label_width = max(len(run.name) for run in runs)
    for ax, metric in zip(axes, metrics):
        _panel_legend(ax, styled_runs, metric, label_width)
    return fig


def _plot_series(ax, epochs, values, alpha: float, *, color, lw: float, raw: bool,
                 raw_alpha: float = 0.16, ls="-", zorder: float = 4) -> np.ndarray:
    if raw and values.size > 1:
        ax.plot(epochs, values, color=color, alpha=raw_alpha * 0.5, lw=2.4, zorder=0.9)
        ax.plot(epochs, values, color=color, alpha=raw_alpha * 1.9, lw=0.45, zorder=1)

    smooth = sp.ema(values, alpha)
    ax.plot(epochs, smooth, color=color, lw=lw, ls=ls, zorder=zorder)
    return smooth


def _plot_ref_gutter(ax, epochs, ref_curve, *, above: bool) -> None:
    ref_lo, ref_hi = float(np.min(ref_curve)), float(np.max(ref_curve))
    band = REF_GUTTER_BAND if above else (1.0 - REF_GUTTER_BAND[1], 1.0 - REF_GUTTER_BAND[0])
    band_depth = band[1] - band[0]
    y_lo, y_hi = ax.get_ylim()
    room = band_depth * (y_hi - y_lo)

    if ref_hi - ref_lo <= room:
        y_frac = 0.5 * (band[0] + band[1]) + (ref_curve - 0.5 * (ref_lo + ref_hi)) / (y_hi - y_lo)
    else:
        y_frac = band[0] + (ref_curve - ref_lo) / (ref_hi - ref_lo) * band_depth

    rule = REF_GUTTER_RULE if above else 1.0 - REF_GUTTER_RULE
    tick_clear = REF_GUTTER_TICK_CLEAR if above else -REF_GUTTER_TICK_CLEAR
    tick_cutoff = y_lo + (rule - tick_clear) * (y_hi - y_lo)
    ticks = [tick for tick in ax.get_yticks()
             if y_lo <= tick <= y_hi and (tick <= tick_cutoff if above else tick >= tick_cutoff)]

    # set_yticks() can expand view limits; the locator leaves the finalized loss scale untouched.
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.fill_between([0.0, 1.0], rule, float(above), transform=ax.transAxes,
                    color=sp.INK, lw=0, zorder=1.5)
    ax.plot([0.0, 1.0], [rule, rule], transform=ax.transAxes, color=sp.MUTE, lw=0.9,
            ls=(0, (6, 3)), alpha=0.65, zorder=2)
    # x remains in epoch coordinates; y is pinned to the reserved axes strip.
    ax.plot(epochs, y_frac, transform=ax.get_xaxis_transform(), color=REF, lw=1.2,
            ls=(0, (1, 1.8)), alpha=0.95, zorder=3)
    ax.text(1.008, sum(band) / 2, f"ref {'↑' if above else '↓'} {float(ref_curve[-1]):.6f}",
            transform=ax.transAxes, ha="left", va="center", fontsize=7.5, color=REF, zorder=4)


def _plot_delta_strip(ax, run: Run, moved_params: np.ndarray, alpha: float, *, raw: bool,
                      best_val_epoch: int | None) -> None:
    epochs = run.epochs
    if raw:
        # Raw Δp is per-epoch activity, not a continuous signal.
        ax.vlines(epochs, 0.0, moved_params, color=DELTA, alpha=0.07, lw=0.45, zorder=0.7)

    moved_smooth = sp.ema(moved_params, alpha)
    ax.fill_between(epochs, 0.0, moved_smooth, color=DELTA, alpha=0.10, lw=0, zorder=0.8)
    ax.plot(epochs, moved_smooth, color=DELTA, lw=1.6, zorder=4)

    if best_val_epoch is not None:
        ax.axvline(best_val_epoch, color=BEST_VAL, ls="--", lw=0.7, alpha=0.35, zorder=2)
    for epoch in run.restart_epochs:
        ax.axvline(epoch, color=RESTART, ls=(0, (1, 2)), lw=1.0, alpha=0.40, zorder=2)

    total_moved = float(moved_params.sum())
    if total_moved > 0:
        p99_index = int(np.searchsorted(np.cumsum(moved_params), 0.99 * total_moved))
        p99_epoch = int(epochs[min(p99_index, epochs.size - 1)])

        if p99_epoch < epochs[-1]:
            align_right = p99_epoch > epochs[0] + 0.75 * (epochs[-1] - epochs[0])
            ax.axvline(p99_epoch, color=DELTA, ls=(0, (1, 2)), lw=0.9, alpha=0.45, zorder=3)
            ax.text(p99_epoch, 0.94, f"99% of Δp @{p99_epoch} " if align_right else f" 99% of Δp @{p99_epoch}",
                    transform=ax.get_xaxis_transform(), va="top", ha="right" if align_right else "left",
                    fontsize=7.5, color=DELTA, alpha=0.8)

    ax.set_ylim(0.0, float(moved_smooth.max()) * 1.18 or 1.0)
    ax.set_xlim(epochs[0], epochs[-1])
    ax.set_ylabel("Δp", labelpad=8)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(3, integer=True))
    sp.style_axes(ax, grain=False)


def _trimmed_bounds(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return finite
    return np.percentile(finite, [RANGE_TRIM_PERCENTILE, 100.0 - RANGE_TRIM_PERCENTILE])


def _style_runs(runs: list[Run], baseline: Run | None) -> list[tuple[Run, dict]]:
    comparison_runs = [run for run in runs if run is not baseline]
    colors = _run_colors(len(comparison_runs))
    styled_runs = [(run, {"color": color, "lw": 1.8}) for run, color in zip(comparison_runs, colors)]

    if baseline is not None:
        styled_runs.insert(0, (baseline, {"color": BASELINE, "ls": (0, (6, 3)), "lw": 1.5}))
    return styled_runs


def _run_colors(count: int) -> list[tuple[float, float, float, float]]:
    ramp = sp.oklch_ramp(RUN_COLOR_STOPS)
    if count == 0:
        return []
    if count == 1:
        return [ramp(0.5)]
    return [ramp(float(position)) for position in np.linspace(0.06, 0.96, count)]


def _short_labels(runs: list[Run]) -> None:
    paths = [Path(run.path) for run in runs]
    labels = [path.stem for path in paths]

    if len(labels) < 2:
        return
    if len(set(labels)) != len(labels):
        labels = [path.with_suffix("").as_posix() for path in paths]

    prefix = labels[0]
    for label in labels[1:]:
        while prefix and not label.startswith(prefix):
            prefix = prefix[:-1]
    while prefix and prefix[-1] not in LABEL_SEPARATORS:
        prefix = prefix[:-1]
    if prefix and all(len(label) - len(prefix) >= 2 for label in labels):
        labels = [label[len(prefix):] for label in labels]

    suffix = labels[0]
    for label in labels[1:]:
        while suffix and not label.endswith(suffix):
            suffix = suffix[1:]
    while suffix and suffix[0] not in LABEL_SEPARATORS:
        suffix = suffix[1:]
    if suffix and all(len(label) - len(suffix) >= 2 for label in labels):
        labels = [label[:-len(suffix)] for label in labels]

    for run, label in zip(runs, labels):
        run.label = label


def _panel_legend(ax, styled_runs: list[tuple[Run, dict]], metric: str, label_width: int) -> None:
    entries = []

    for run, style in styled_runs:
        best_loss = run.best_val_loss if metric == "val" else run.best_train_loss
        loss_text = f"{best_loss:.6f}" if best_loss is not None else "—"
        entries.append((Line2D([0], [0], **style), f"{run.name:<{label_width}}  {loss_text}"))

    corner = _least_occupied_corner(ax, rows=len(entries) + 1, chars=max(len(text) for _, text in entries))
    sp.legend(ax, entries, anchor=None, loc=corner, pad=1.2, title="best")


def _least_occupied_corner(ax, *, rows: int, chars: int, font_size: float = 8.5) -> str:
    box = ax.get_window_extent()
    font_px = font_size * ax.figure.dpi / 72.0
    height = min(0.9, (rows * 1.5 + 1.2) * font_px / box.height)  # row pitch + pad
    width = min(0.9, (chars * 0.62 + 3.4) * font_px / box.width)  # mono advance + handle/pad
    corners = {
        "upper right": (1.0 - width, 1.0, 1.0 - height, 1.0),
        "upper left": (0.0, width, 1.0 - height, 1.0),
        "lower right": (1.0 - width, 1.0, 0.0, height),
        "lower left": (0.0, width, 0.0, height),
    }
    line_points = [line.get_xydata() for line in ax.lines if line.get_xydata().size]

    if not line_points:
        return "upper right"

    points = np.vstack(line_points)
    (x_lo, x_hi), (y_lo, y_hi) = ax.get_xlim(), ax.get_ylim()
    x_frac = (points[:, 0] - x_lo) / ((x_hi - x_lo) or 1.0)
    y_frac = (points[:, 1] - y_lo) / ((y_hi - y_lo) or 1.0)

    occupancy = {corner: int(np.count_nonzero((x_frac >= x0) & (x_frac <= x1) &
                                              (y_frac >= y0) & (y_frac <= y1)))
                 for corner, (x0, x1, y0, y1) in corners.items()}
    return min(occupancy.items(), key=lambda item: item[1])[0]


def _range_span(lower: float, upper: float) -> float:
    span = upper - lower
    if span > 0.0:
        return span
    return max(abs(lower), abs(upper), 1e-3) * 1e-3


def _loss_range(series: list[np.ndarray]) -> tuple[float, float]:
    usable_series = [np.asarray(values, dtype=np.float64) for values in series
                     if np.size(values) and np.isfinite(values).any()]
    # Seed from the tightest series. A cold-start outlier can be orders of magnitude
    # wider than the tails being compared and should not flatten every other run.
    by_span = sorted((values for values in usable_series if values.size > 1),
                     key=lambda values: np.ptp(values[np.isfinite(values)]))

    if not by_span:
        return _bounds(usable_series)

    lo, hi = _bounds([by_span[0]])
    for values in by_span[1:] + [values for values in usable_series if values.size <= 1]:
        span = _range_span(lo, hi)
        value_lo, value_hi = _bounds([values])
        if value_lo < lo - span or value_hi > hi + span:
            value_lo, value_hi = _bounds([_trimmed_bounds(values)])
        lo, hi = min(lo, value_lo), max(hi, value_hi)

    return lo, hi


def _bounds(series: list[np.ndarray]) -> tuple[float, float]:
    if not series:
        return 0.0, 1.0

    values = np.concatenate([np.asarray(item, dtype=np.float64).ravel() for item in series])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(finite.min()), float(finite.max())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training and validation loss curves.",
        formatter_class=sp.HelpFormatter,
    )

    parser.add_argument("logs", nargs="+", help="evaltune .jsonl logs; multiple logs are compared")
    parser.add_argument("-o", "--output", help="image path (default: beside first log)")
    parser.add_argument("-a", "--alpha", type=float, default=0.06,
                        help="EMA factor (0, 1] (default: 0.06)")
    metric_group = parser.add_mutually_exclusive_group()
    metric_group.add_argument("--val-only", action="store_true", help="validation curves only")
    metric_group.add_argument("--train-only", action="store_true", help="training curves only")
    parser.add_argument("--no-raw", action="store_true", help="hide raw traces, smoothed only")
    parser.add_argument("--no-deltas", action="store_true", help="hide the Δp strip")
    parser.add_argument("--baseline", help="draw this run dashed, as the line to beat")
    parser.add_argument("--ylim", help="pin the loss axis, lo,hi (default: fit the data)")
    parser.add_argument("--run", type=int, metavar="N",
                        help="logical run in an append-only log, 1-based (default: longest completed)")
    parser.add_argument("--dpi", type=int, default=sp.DPI, help=f"output DPI (default: {sp.DPI})")
    parser.add_argument("--width", type=float, default=13.0, help="figure width in inches (default: 13)")
    parser.add_argument("--height", type=float, default=7.0, help="figure height in inches (default: 7)")
    parser.add_argument("--show", action="store_true", help="show interactively after saving")
    args = parser.parse_args()

    if not (0.0 < args.alpha <= 1.0):
        parser.error("--alpha must be in (0, 1]")

    ylim = None
    if args.ylim:
        try:
            lo, hi = (float(value) for value in args.ylim.split(","))
        except ValueError:
            parser.error("--ylim takes two numbers, lo,hi")
        if not np.isfinite((lo, hi)).all() or not lo < hi:
            parser.error("--ylim takes finite lo < hi")
        ylim = lo, hi

    log_paths = list(args.logs)
    baseline_path = Path(args.baseline).resolve() if args.baseline else None

    # A baseline need not also be listed as a comparison run.
    if baseline_path is not None and not any(Path(path).resolve() == baseline_path for path in log_paths):
        log_paths.insert(0, args.baseline)

    runs: list[Run] = []
    for path in log_paths:
        try:
            run = parse_log(path, args.run)
        except OSError as exc:
            print(f"  [warn] {path}: {exc.strerror or exc}", file=sys.stderr)
            continue
        except IndexError as exc:
            parser.error(str(exc))

        if run is None:
            print(f"  [warn] {path}: no epoch entries", file=sys.stderr)
            continue
        runs.append(run)

    if not runs:
        print("error: no run to plot.", file=sys.stderr)
        sys.exit(1)

    baseline = next((run for run in runs if Path(run.path).resolve() == baseline_path), None)
    all_have_val = all(run.has_val_loss for run in runs)
    any_have_val = any(run.has_val_loss for run in runs)

    if args.val_only:
        if not all_have_val:
            parser.error("--val-only requires validation loss in every selected run")
        metrics = ("val",)
    elif args.train_only:
        metrics = ("train",)
    elif all_have_val:
        metrics = ("train", "val")
    elif not any_have_val:
        metrics = ("train",)
    else:
        parser.error("some selected runs have no validation loss; use --train-only")

    comparison_note = None
    if len(runs) > 1:
        datasets = {run.dataset for run in runs if run.dataset is not None}

        if len(datasets) > 1:
            comparison_note = "mixed datasets"
            print("  [warn] losses use different datasets; they are not directly comparable", file=sys.stderr)
        elif any(run.dataset is None for run in runs):
            comparison_note = "dataset unverified"
            print("  [warn] dataset missing from an old log; loss comparability is unverified", file=sys.stderr)
        elif "val" in metrics:
            split_seeds = {run.split_seed for run in runs if run.split_seed is not None}
            if len(split_seeds) > 1:
                comparison_note = "mixed splits"
                print("  [warn] validation losses use different split seeds; they are not directly comparable",
                      file=sys.stderr)
            elif any(run.split_seed is None for run in runs):
                comparison_note = "split unverified"
                print("  [warn] split seed missing from an old log; validation comparability is unverified",
                      file=sys.stderr)

    _short_labels(runs)
    sp.use_theme()

    if len(runs) == 1:
        fig = plot_single(runs[0], alpha=args.alpha, metrics=metrics, show_raw=not args.no_raw,
                          show_deltas=not args.no_deltas, ylim=ylim,
                          fig_width=args.width, fig_height=args.height)
        suffix = "_loss.png"
    else:
        fig = plot_compare(runs, alpha=args.alpha, metrics=metrics, show_raw=not args.no_raw,
                           baseline=baseline, comparison_note=comparison_note, ylim=ylim,
                           fig_width=args.width, fig_height=args.height)
        suffix = "_loss_compare.png"

    # Keep the default output anchored to the caller's first log, not an inserted baseline.
    output = args.output or Path(args.logs[0]).with_suffix("").as_posix() + suffix
    sp.save(fig, output, show=args.show, dpi=args.dpi)


if __name__ == "__main__":
    main()
