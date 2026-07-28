# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Elo progression.

Auto-detects its input:

  • ordo / rating log: a cutechess+ordo dump with [Date "…"] blocks and
    rating tables (rating + error per engine). Absolute rating over time,
    error-banded, anchor-relative. Single engine → hero line; many → comparison.

  • SPRT sequence:     fastchess / cutechess / OpenBench result blocks, one
    merged patch each, in order. Sums the measured Elo into a cumulative curve
    inside a 95% cone, a per-patch panel below sized symlog so tiny and huge
    gains both read. --initial-elo shifts the axis to absolute.

  • odds (--odds):     Elo against a time or thread handicap (header
    [Odds "2x"] / [Time …] / [Threads …]), fitting Elo per doubling.

    uv run scripts/plot_elo.py <log-or-results.txt> [options]

Options:
    -o, --output PATH   image path (default: <stem>_elo.png)
    --initial-elo N     SPRT mode: starting rating, axis reads absolute from N
    --odds              odds-scaling mode (Elo vs time/thread handicap)
    --ref-slope N       odds mode: reference at ±N Elo/doubling (0 disables; default 80)
    --title TEXT        override the headline
    --dpi INT           output DPI (default: 300)
    --show              open interactively after saving
"""

from __future__ import annotations

import argparse
import re
import sys

from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

import soulplot as sp

PALETTE = ["#86b06a", "#e0935a", "#7d97b8", "#b79be0", "#d98aa0", "#6fb6a0", "#e6b450"]


_DATE   = re.compile(r'\[Date\s+"([^"]+)"\]')
# Dotted is day-first (dd.mm.yy/yyyy) as you'd hand-write it, falling back to
# PGN's year-first yyyy.mm.dd that cutechess/ordo emits; the field widths keep
# them unambiguous (a 4-digit head can't be a day, a 4-digit tail can't be yy).
# OpenBench slashes are US month-first. Display is always dd.mm.yy regardless.
_DATE_FORMATS = ("%d.%m.%y", "%d.%m.%Y", "%Y.%m.%d",    # dotted; day-first, then PGN
                 "%Y-%m-%d",                            # ISO
                 "%m/%d/%Y", "%m/%d/%y")                # OpenBench, month-first
_DISPLAY_DATE = "%d.%m.%y"

# Odds header; the keyword picks the title, the value's number picks the x-pos.
_ODDS = re.compile(r'\[(Odds|Time|Threads|TC)\s+"([^"]+)"\]')

_RATING = re.compile(r'^\s*\d+\s+(.+?)\s+:\s+([-+]?\d+\.\d+)\s+(----|\d+\.\d+)')
# One Elo line per test. The (?<![A-Za-z]) guards against matching "nElo".
_ELO    = re.compile(r'(?<![A-Za-z])Elo[^:|\n]*[:|]\s*([-+]?[\d.]+)\s*(?:\+/-|\+-|±)\s*([\d.]+)')
_LABEL  = re.compile(r'Results of\s+(.+?)\s+vs\s+', re.I)
_PENTA  = re.compile(r'(?:Ptnml\(0-2\)|Penta)\s*[:|]?\s*\[([\d,\s]+)\]')
_LLR    = re.compile(r'LLR\s*[:|]\s*([-+]?[\d.]+)')


def _parse_date(s: str) -> datetime | None:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def _odds_mult(label: str) -> float:
    """Pull the odds multiplier from a label: '2x' → 2, 'base'/'1x' → 1."""
    m = re.search(r'([\d.]+)', label)
    return float(m.group(1)) if m else 1.0


def _is_anchor(ratings: list[float]) -> bool:
    """An anchor sits pinned at 0 the whole log; every other engine is rated against it."""
    return all(abs(r) < 0.01 for r in ratings)


def parse_ordo(text: str) -> dict[str, dict]:
    """Dated rating tables → {engine: {dates, ratings, errors}} (chronological)."""
    data: dict[str, dict] = {}
    cur_date: datetime | None = None

    for line in text.splitlines():
        if (m := _DATE.search(line)):
            cur_date = _parse_date(m.group(1))
            continue

        if cur_date and (m := _RATING.match(line)):
            name = m.group(1).strip()
            rating = float(m.group(2))
            error = 0.0 if m.group(3) == "----" else float(m.group(3))
            d = data.setdefault(name, {"dates": [], "ratings": [], "errors": []})

            if not d["dates"] or d["dates"][-1] != cur_date:
                d["dates"].append(cur_date)
                d["ratings"].append(rating)
                d["errors"].append(error)
    return data


def parse_sprt(text: str) -> list[dict]:
    """Result blocks → ordered [{label, date, elo, error, llr, penta}], one per test.

    A leading `[Date "…"]` per block is optional; when every block carries one,
    the progress plot lays patches on a real timeline instead of a bare index.
    """
    tests: list[dict] = []
    pending = None
    cur_date: datetime | None = None
    cur_odds: tuple[str, str] | None = None
    cur: dict | None = None

    for line in text.splitlines():
        if (m := _DATE.search(line)):
            cur_date = _parse_date(m.group(1))
            continue

        if (m := _ODDS.search(line)):
            cur_odds = (m.group(1), m.group(2))
            continue

        if (m := _LABEL.search(line)):
            pending = m.group(1).strip()

        if (m := _ELO.search(line)):
            cur = {"label": pending, "date": cur_date, "elo": float(m.group(1)),
                   "error": float(m.group(2)), "llr": None, "penta": None,
                   "odds": cur_odds[1] if cur_odds else None,
                   "odds_kw": cur_odds[0] if cur_odds else None}
            tests.append(cur)
            pending = None
            continue

        if cur is not None:
            if cur["llr"] is None and (m := _LLR.search(line)):
                cur["llr"] = float(m.group(1))

            if cur["penta"] is None and (m := _PENTA.search(line)):
                cur["penta"] = [int(x) for x in m.group(1).replace(" ", "").split(",")]
    return tests


def plot_ordo(data: dict[str, dict], out: str, *, title: str | None, dpi: int, show: bool) -> None:
    anchors = [n for n, d in data.items() if _is_anchor(d["ratings"])]
    active  = [n for n in data if n not in anchors]

    if not active:
        sys.exit("error: no non-anchor engine to plot.")
    active.sort(key=lambda n: data[n]["ratings"][-1], reverse=True)
    comparison = len(active) > 1

    sp.use_theme()
    fig, ax = plt.subplots(figsize=(13, 7))

    entries = []

    for i, name in enumerate(active):
        d = data[name]
        x = mdates.date2num(d["dates"])
        y = np.array(d["ratings"], float)
        e = np.array(d["errors"], float)

        color = PALETTE[i % len(PALETTE)] if comparison else sp.advantage(0.6)

        # ±error cone, densest on the rating and fading to the edge: what the
        # interval means, where a flat fill would read every value in it as
        # equally likely.
        sp.density_band(ax, x, y, e, color, zorder=1)
        ax.plot(x, y, color=color, lw=2.0 if comparison else 2.4, zorder=4)
        # Core fades where the rating is young and the error wide; firms up as games accrue.
        sp.dot(ax, x, y, color, size=30, core_alpha=sp.confidence_alpha(e), zorder=5)
        entries.append((float(y[-1]), f"{name}  {y[-1]:+.0f}", color))

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter(_DISPLAY_DATE))
    ax.set_ylabel("Elo (anchor-relative)", labelpad=8)
    ax.margins(x=0.02, y=0.15)
    sp.style_axes(ax)
    sp.end_labels(ax, ax.get_xlim()[1], entries)

    sub = f"{anchors[0]} = 0" if anchors else None
    sp.title(fig, title or ("Elo comparison" if comparison else active[0]), sub)
    plt.subplots_adjust(top=0.88, bottom=0.10, left=0.09, right=0.86)
    sp.save(fig, out, show=show, dpi=dpi)


def plot_sprt(tests: list[dict], out: str, *, initial: float = 0.0,
              title: str | None, dpi: int, show: bool) -> None:
    elo = np.array([t["elo"] for t in tests], float)
    err = np.array([t["error"] for t in tests], float)
    cum = np.cumsum(elo)
    cum_err = np.sqrt(np.cumsum(err ** 2))  # 95% half-width of the running total
    x = np.arange(1, len(tests) + 1)

    # --initial-elo shifts 0 → the engine's known starting rating, so the axis
    # reads absolute. lo/hi are the 95% CI band (per-test ± are 95% half-widths,
    # so √(Σerr²) is the sum's 95% half-width; the 1.96σ cancels).
    y = initial + cum
    lo, hi = y - cum_err, y + cum_err
    absolute = initial != 0.0

    def fmt(v: float) -> str:
        # Absolute reads as a plain rating; relative keeps the sign of the delta.
        return f"{v:.0f}" if absolute else f"{v:+.0f}"

    # Shade each patch by how much it moved the needle: vivid for a big gain,
    # dim for a marginal one, red below zero. A zero patch floors to a faint mark.
    span = max(float(np.max(np.abs(elo))), 1.0)
    mag = np.sign(elo) * (0.2 + 0.65 * np.abs(np.clip(elo / span, -1, 1)))
    shades = [sp.advantage(m or 0.2) for m in mag]

    sp.use_theme()
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(13, 8), height_ratios=[3, 1], sharex=True)

    # ── top; cumulative total inside a 95% cone of uncertainty
    # Zero is the origin of a cumulative total and belongs on the axis. A starting
    # rating is not: the axis is a rating scale, the first patch lands hundreds of
    # Elo above the mark, and drawing it there spends a third of the panel saying
    # what the subtitle says in words.
    if not absolute:
        ax.axhline(0.0, color=sp.MUTE, lw=0.9, alpha=0.5, zorder=2)

    sp.density_band(ax, x, y, cum_err, sp.GOLD, zorder=1)
    ax.plot(x, y, color=sp.GOLD, lw=2.0, zorder=4)
    # Cumulative cores stay solid; the cone carries the growing uncertainty here.
    sp.dot(ax, x, y, shades, size=22, zorder=5)
    ax.set_ylabel("Elo" if absolute else "cumulative Elo (Σ measured)", labelpad=8)

    # ── bottom; each patch's measured gain, symlog so tiny and huge both read
    ax.set_xlim(0.5, len(tests) + 0.5)
    axd.axhline(0, color=sp.MUTE, lw=0.8, alpha=0.5, zorder=1)
    axd.vlines(x, 0, elo, colors=shades, lw=1.6, zorder=3)
    # Each patch stands alone, so its core opacity tracks its own measurement noise.
    sp.dot(axd, x, elo, shades, size=22, core_alpha=sp.confidence_alpha(err), zorder=4)
    axd.set_yscale("symlog", linthresh=2.0)
    axd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}"))
    axd.set_ylabel("patch Δ", labelpad=8)

    dates = [t["date"] for t in tests]

    if all(d is not None for d in dates):
        # Six upright labels rather than ten leaning ones. Patches are not evenly
        # spaced in time, so the gaps between dates are real and a denser axis
        # only makes them look like a stutter.
        step = max(1, round(len(tests) / 6))
        idx = list(range(0, len(tests), step))
        axd.set_xticks([x[i] for i in idx])
        axd.set_xticklabels([dates[i].strftime(_DISPLAY_DATE) for i in idx], fontsize=7.5)
    else:
        axd.set_xlabel("patch", labelpad=8)

    ax.margins(y=0.12)
    sp.style_axes(ax)
    sp.style_axes(axd, grain=False)

    sp.end_labels(ax, ax.get_xlim()[1],
                  [(float(y[-1]), f"{fmt(y[-1])} ± {cum_err[-1]:.0f}", sp.GOLD)])

    # Bracket derived from the displayed center ± error so it always reconciles
    # (rounding each exact bound separately can disagree with the shown ± by one).
    er = round(float(cum_err[-1]))
    ci_lo, ci_hi = round(float(y[-1])) - er, round(float(y[-1])) + er
    gained = f"from {initial:.0f}   ·   +{cum[-1]:.0f}" if absolute else fmt(y[-1])

    sp.title(fig, title or "Elo progress",
             f"{len(tests)} patches   ·   {gained} ± {er}   ·   95% CI [{fmt(ci_lo)}, {fmt(ci_hi)}]"
             f"   ·   Σ measured (SPRT-biased)")
    plt.subplots_adjust(top=0.89, bottom=0.12, left=0.09, right=0.88, hspace=0.08)
    sp.save(fig, out, show=show, dpi=dpi)


def plot_odds(tests: list[dict], out: str, *, ref_slope: float = 80.0,
              title: str | None, dpi: int, show: bool) -> None:
    # The header value can be either a relative multiplier ("2x", "4T") or the
    # actual config ("16+0.16", "32"). Either way, doublings are relative to the
    # smallest entry, the base, so the x-axis is log₂(raw / base).
    raw = np.array([_odds_mult(t["odds"] or "1") for t in tests], float)
    order = np.argsort(raw)
    raw = raw[order]
    elo = np.array([tests[i]["elo"] for i in order], float)
    err = np.array([tests[i]["error"] for i in order], float)
    labels = [tests[i]["odds"] or f"{r:g}" for i, r in zip(order, raw)]
    base_label = labels[0]
    d = np.log2(raw / raw[0])   # doublings from the base entry

    sp.use_theme()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axhline(0, color=sp.MUTE, lw=0.9, alpha=0.5, zorder=2)

    # Linear fit through the data points (NOT an ideal reference). Slope is the
    # measured scaling, Elo per doubling. Deviation of points from this line is
    # diminishing returns. Sign comes from the direction of testing (positive for
    # dev-vs-base, negative for base-vs-dev); magnitude is the worth of a doubling.
    slope = float("nan")
    keys = []   # legend entries, each the real line so its style matches verbatim

    if len(d) >= 2 and np.ptp(d) > 0:
        slope, intercept = np.polyfit(d, elo, 1)
        gx = np.linspace(d.min(), d.max(), 50)

        (fit_line,) = ax.plot(gx, slope * gx + intercept, color=sp.GOLD, ls=(0, (5, 3)),
                              lw=1.6, alpha=0.85, zorder=3)
        keys.append((fit_line, "fit"))

    # Reference at ±ref_slope through the origin, the "typical doubling" the
    # measured fit is read against. Muted dotted so it stays a backdrop, not a
    # second line fighting the gold fit for the eye.
    if ref_slope and np.ptp(d) > 0:
        ref = np.copysign(ref_slope, 1.0 if np.isnan(slope) else slope)
        (ref_line,) = ax.plot([0.0, float(d.max())], [0.0, ref * float(d.max())],
                              color=sp.MUTE, ls=(0, (1, 3)), lw=1.2, alpha=0.7, zorder=2)
        keys.append((ref_line, "ref"))

    # Magnitude-shaded markers over translucent 95% CI columns (no I-bars).
    # Shade tracks handicap magnitude: vivid green/red far out, dim near base.
    span = max(float(np.max(np.abs(elo))), 1.0)
    for di, ei, ee in zip(d, elo, err):
        if ee > 0:
            sp.glow_column(ax, di, ei, ee, sp.GOLD, zorder=4)
    shades = [sp.advantage(np.clip(e / span, -1, 1) * 0.85) for e in elo]
    sp.dot(ax, d, elo, shades, size=110, core_alpha=sp.confidence_alpha(err), zorder=6)

    for di, ei, ee in zip(d, elo, err):
        txt = f"{ei:+.0f}" if ee == 0 else f"{ei:+.0f} ± {ee:.0f}"
        ax.annotate(txt, xy=(di, ei), xytext=(0, -14), textcoords="offset points",
                    ha="center", va="top", fontsize=7.5, color=sp.TEXT, alpha=0.85)

    kw = tests[0].get("odds_kw") or "Odds"
    ax.set_xticks(d)
    ax.set_xticklabels(labels, fontsize=8.5)  # the header value verbatim: "2x", "4T", …
    ax.set_xlabel({"Threads": "threads", "Time": "time", "TC": "time"}.get(kw, "odds"), labelpad=8)
    ax.set_ylabel("Elo (95% CI)", labelpad=8)
    ax.margins(x=0.08, y=0.22)
    sp.style_axes(ax)

    if keys:
        sp.legend(ax, keys)  # visual key for which dashed line is which

    head = title or {"Time": "time-odds scaling", "Threads": "thread scaling",
                     "TC": "time-odds scaling"}.get(kw, "odds scaling")
    pieces = [f"base: {base_label}"]
    if not np.isnan(slope):
        pieces.append(f"fit: {slope:+.1f} Elo / doubling")

    if ref_slope:
        pieces.append(f"ref: ±{ref_slope:.0f}")
    sp.title(fig, head, "   ·   ".join(pieces))
    plt.subplots_adjust(top=0.87, bottom=0.12, left=0.10, right=0.95)
    sp.save(fig, out, show=show, dpi=dpi)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot Elo progression from an ordo log or a sequence of SPRT results.",
        formatter_class=sp.HelpFormatter,
    )

    ap.add_argument("log", help="ordo rating log or SPRT results file")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--initial-elo", type=float, default=0.0,
                    help="starting rating for SPRT mode; the curve reads absolute from here")
    ap.add_argument("--odds", action="store_true",
                    help="odds-scaling mode: Elo vs time/thread handicap, Elo-per-doubling fit")
    ap.add_argument("--ref-slope", type=float, default=80.0,
                    help="odds mode: reference at ±N Elo/doubling (0 disables; default 80)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi", type=int, default=sp.DPI)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    out = args.output or Path(args.log).with_suffix("").as_posix() + "_elo.png"

    if args.odds:
        tests = parse_sprt(text)

        if not tests:
            sys.exit("error: no SPRT result blocks found for --odds.")
        plot_odds(tests, out, ref_slope=args.ref_slope, title=args.title, dpi=args.dpi, show=args.show)
        return

    ordo = parse_ordo(text)

    if any(not _is_anchor(d["ratings"]) for d in ordo.values()):
        plot_ordo(ordo, out, title=args.title, dpi=args.dpi, show=args.show)
        return
    tests = parse_sprt(text)

    if tests:
        plot_sprt(tests, out, initial=args.initial_elo, title=args.title, dpi=args.dpi, show=args.show)
        return
    sys.exit("error: could not parse an ordo rating log or any SPRT result blocks.")


if __name__ == "__main__":
    main()
