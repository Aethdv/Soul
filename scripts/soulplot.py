# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Shared plotting toolkit for Soul's scripts.

Building blocks, not a fixed theme: each plotter picks its own data colors
and composition over one palette and one set of helpers.

  use_theme()          rcParams, warm-ink canvas, monospace
  style_axes(ax)       spines / grid / ticks, the per-file boilerplate once
  glow_line(...)       line with a soft halo under a crisp stroke
  dot(...)             crisp marker: colored core + ink moat ring
  gradient_fill(...)   area under a curve fading to transparent
  density_band(...)    CI band, opaque at the estimate, fading to the edge
  glow_column(...)     vertical error column with no hard caps
  gradient_line(...)   polyline colored per-vertex
  band(...)            hatched fill between two curves
  advantage(t)         OkLCH win↔loss gradient, mirrors soul::color
  confidence_alpha(e)  error width → opacity, tight solid / noisy soft
  diverging_cmap()     OkLCH diverging colormap, red↔green
  oklch_ramp(stops)    sequential OkLCH colormap
  legend(...)          frameless key, placed off the plot bed
  title / save         headline + subtitle; write + confirm
  ema / format_count   smoothing + K/M/G formatting
  iter_log(path)       stream a JSONL file as dicts
  HelpFormatter        argparse help: rose <metavar>s, aligned
"""

from __future__ import annotations

import sys
import json
import math
import argparse
import re

from collections.abc import Iterator

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import to_rgb, to_rgba_array, LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

INK = "#14110b"   # figure ground, warm near-black
PANEL = "#1b1710" # plot bed, lifted a hair off INK
TEXT = "#e7ddc8"  # warm cream
MUTE = "#8a7e68"  # labels, ticks; warm taupe, one dim
LINE = "#332c20"  # spines, grid, warm
GOLD = "#e6b450"  # markers, highlights

_ANSI_GREEN = "\x1b[38;2;127;176;105m"
_ANSI_DIM = "\x1b[38;2;138;126;104m"
_ANSI_ROSE = "\x1b[38;2;193;125;125m"
_ANSI_RESET = "\x1b[0m"


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """argparse help, Soul-styled: lowercase `<metavar>`s in dusty rose, a help
    column wide enough that long flags don't wrap, and the description left raw.

    The rose is painted last, by a regex over the finished help text, so its
    invisible ANSI bytes never reach argparse's column-width arithmetic. Color is
    a tty-only courtesy; piped or redirected, the help stays plain.
    """

    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=34)

    def _get_default_metavar_for_optional(self, action: argparse.Action) -> str:
        return f"<{action.dest.replace('_', '-')}>"

    def format_help(self) -> str:
        text = super().format_help()
        if sys.stdout.isatty():
            text = re.sub(r"<[\w-]+>", lambda m: f"{_ANSI_ROSE}{m.group()}{_ANSI_RESET}", text)
        return text


def use_theme(font: str = "monospace") -> None:
    """Apply the warm-ink monospace canvas to matplotlib globally."""
    plt.rcParams.update({
        "figure.facecolor": INK,
        "savefig.facecolor": INK,
        "axes.facecolor": PANEL,
        "font.family": font,
        "font.weight": "medium",
        "text.color": TEXT,
        "axes.labelcolor": MUTE,
        "xtick.color": MUTE,
        "ytick.color": MUTE,
        "axes.titlecolor": TEXT,
        "figure.dpi": 110,
        "lines.solid_capstyle": "round",
        "lines.antialiased": True,
        "path.simplify": True,
    })


def style_axes(ax, *, grid: bool = True, grain: bool = True) -> None:
    """Two warm spines, dimmed ticks, a faint y-grid, and film grain."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for side in ("bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(LINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTE, labelsize=8, length=3, width=0.7)

    if grid:
        ax.grid(True, axis="y", color=TEXT, alpha=0.04, lw=0.5)
        ax.set_axisbelow(True)

    if grain:
        _grain(ax)


def _grain(ax, *, alpha: float = 0.012, seed: int = 0) -> None:
    """Faint film grain over the plot bed.

    Drawn in data coords spanning the current view, with the limits restored
    after; a data-coord image stays put under a linear axis. (Drawing it under
    transAxes instead balloons the tight bbox unless the extent is pinned, and
    on a nonlinear axis the data-coord version smears in the compressed region;
    callers on such axes pass grain=False.)
    """
    xl, yl = ax.get_xlim(), ax.get_ylim()
    rng = np.random.default_rng(seed)
    noise = rng.random((320, 560))
    ax.imshow(noise, cmap="gray", aspect="auto", alpha=alpha, zorder=0,
              extent=(xl[0], xl[1], yl[0], yl[1]), interpolation="bilinear", clip_on=True)
    ax.set_xlim(xl)
    ax.set_ylim(yl)


def end_labels(ax, x, entries, *, size: float = 9.0, pad_frac: float = 0.012, min_frac: float = 0.05) -> None:
    """Direct labels at the curves' right ends, in place of a legend.

    `entries` is a list of `(y, text, color)`. Labels spread vertically so none
    collide, each staying as close to its curve as the crowding allows.
    """
    if not entries:
        return

    lo, hi = ax.get_ylim()
    span = (hi - lo) or 1.0
    xlo, xhi = ax.get_xlim()
    xpad = (xhi - xlo) * pad_frac

    items = sorted(entries, key=lambda e: e[0])
    yf = [(y - lo) / span for y, _, _ in items]

    for i in range(1, len(yf)):  # push apart, bottom-up
        if yf[i] - yf[i - 1] < min_frac:
            yf[i] = yf[i - 1] + min_frac

    if (over := yf[-1] - 1.0) > 0:  # whole block slid up past the top → shift down
        yf = [f - over for f in yf]

    for (_, text, color), f in zip(items, yf):
        ax.text(x + xpad, lo + f * span, text, color=color, fontsize=size,
                fontweight="bold", va="center", ha="left", clip_on=False)


def legend(ax, entries, *, anchor=(1.0, 1.0), loc: str = "lower right", size: float = 8.5):
    """Frameless key in the plot's top-right corner, off the bed.

    `entries` is a list of `(artist, label)`: proxy `Line2D`s or markers styled
    to match the plot. Companion to `end_labels`, which labels line ends; this
    keys a line with no end to label, a fit or a reference.
    """
    leg = ax.legend([a for a, _ in entries], [t for _, t in entries], loc=loc,
                    bbox_to_anchor=anchor, frameon=False, fontsize=size,
                    labelcolor=TEXT, handlelength=1.8, handletextpad=0.6, borderaxespad=0.0)
    return leg


def glow_line(ax, x, y, color, *, lw: float = 2.0, layers: int = 4, label=None, zorder: float = 3, **kw):
    """Line with a soft halo: widening, fading copies under a stroke."""
    for i in range(layers, 0, -1):
        ax.plot(x, y, color=color, lw=lw + i * 1.1, alpha=0.05, zorder=zorder - 0.01, solid_capstyle="round")
    return ax.plot(x, y, color=color, lw=lw, label=label, zorder=zorder, solid_capstyle="round", **kw)


def dot(ax, x, y, color, *, size: float = 80, core_alpha: float | np.ndarray = 1.0, ring: str | None = INK,
        ring_lw: float = 1.0, zorder: float = 5):
    """A colored core ringed by a thin ink moat that lifts it off
    whatever band or line sits behind.

    `color` is one color for every dot or a per-point sequence. `core_alpha`
    (scalar or per-point) dims only the fill, never the moat, so a noisy estimate
    reads softer than a tight one and the edge stays sharp: confidence without
    lying through size. Pass `confidence_alpha(err)` straight in.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cols = to_rgba_array(color)            # (1, 4) broadcast, or (N, 4) per-point

    if cols.shape[0] == 1:
        cols = np.repeat(cols, x.size, axis=0)

    core = cols.copy()
    core[:, 3] = core_alpha                # broadcasts a scalar, assigns per-point
    marker = ax.scatter(x, y, s=size, c=core, edgecolors="none", zorder=zorder)

    # The moat rides full-opacity on top, so a soft core never blurs the edge.
    if ring is not None:
        ax.scatter(x, y, s=size, facecolors="none", edgecolors=ring, linewidths=ring_lw, zorder=zorder + 0.01)
    return marker


def gradient_fill(ax, x, y, color, *, base: float | None = None, top_alpha: float = 0.30, zorder: float = 2) -> None:
    """Fill under `y` with a vertical fade from `color` (at the curve) to nothing.

    `base` sets the floor the fade reaches (default: the curve's own minimum).
    Pass `base=0` to fill all the way to a zero axis. The ramp is lightly
    dithered so it stays smooth on 8-bit / wide-gamut displays instead of banding.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rgb = to_rgb(color)
    y0 = float(base) if base is not None else float(y.min())
    y1 = float(y.max())

    h, w = 320, 160
    t = np.linspace(0.0, 1.0, h)[:, None]  # 0 at the curve, 1 at the floor
    fade = top_alpha * (1.0 - t) ** 1.5    # eased: rich near the line, long faint tail
    dither = (np.random.default_rng(0).random((h, w)) - 0.5) * (top_alpha * 0.12)
    ramp = np.zeros((h, w, 4))
    ramp[..., :3] = rgb
    ramp[..., 3] = np.clip(fade + dither, 0.0, 1.0)
    im = ax.imshow(ramp, aspect="auto", origin="upper", extent=(x.min(), x.max(), y0, y1),
                   zorder=zorder, interpolation="bilinear")

    # Clip the ramp to the region under the curve.
    verts = np.vstack([np.column_stack([x, y]), [x[-1], y0], [x[0], y0]])
    clip = Polygon(verts, closed=True, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    im.set_clip_path(clip)


def density_band(ax, x, center, half, color, *, max_alpha: float = 0.16, shells: int = 22, zorder: float = 2) -> None:
    """Confidence band, opaque at `center`, fading to nothing at `center ± half`.

    A flat-alpha box implies every value in the interval is equally likely; this
    says what a CI actually is, densest at the estimate and thinning to the edge.
    Built from nested translucent shells: a point near the estimate is painted by
    every shell, one near the ±edge by only the outermost. Alpha-over compositing
    bends the linear stack into a soft bell, most probable in the middle.
    """
    x = np.asarray(x, dtype=float)
    center = np.asarray(center, dtype=float)
    half = np.asarray(half, dtype=float)
    a = max_alpha / shells

    for k in range(shells, 0, -1):
        f = k / shells
        ax.fill_between(x, center - half * f, center + half * f, color=color, alpha=a, lw=0, zorder=zorder)


def glow_column(ax, x, center, half, color, *, width: float = 8, max_alpha: float = 0.34,
                shells: int = 16, zorder: float = 4) -> None:
    """Vertical error column at one `x`, opaque at `center`, fading out at
    `center ± half` so the uncertainty smears into the ground instead of stopping
    at a wall. The single-column twin of `density_band`: same nested shells, and
    the round caps soften each tip, so the bar has no hard end to read as a bound.
    """
    a = max_alpha / shells

    for k in range(shells, 0, -1):
        f = k / shells
        ax.plot([x, x], [center - half * f, center + half * f], color=color,
                alpha=a, lw=width, solid_capstyle="round", zorder=zorder)


def gradient_line(ax, x, y, colors, *, lw: float = 2.0, zorder: float = 4, capstyle: str = "round"):
    """Polyline colored per-vertex: `colors` is an (N, 3|4) array, one per point.

    Each segment takes the mean of its endpoint colors, so hue flows along the
    line. Used for the eval river, where the color tracks who's ahead.
    """
    points = np.column_stack([np.asarray(x, float), np.asarray(y, float)]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    c = np.asarray(colors, dtype=float)
    lc = LineCollection(segments, colors=0.5 * (c[:-1] + c[1:]), linewidths=lw,
                        zorder=zorder, capstyle=capstyle)
    ax.add_collection(lc)
    return lc


def band(ax, x, lo, hi, color, *, alpha: float = 0.06, hatch: str | None = "////", zorder: float = 2) -> None:
    """Hatched fill between two curves, e.g. the train/val overfit gap."""
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0, hatch=hatch, zorder=zorder)


def title(fig, main: str, sub: str | None = None) -> None:
    """Centered headline with an optional dim subtitle beneath."""
    fig.text(0.5, 0.975, main, ha="center", va="top", fontsize=15, fontweight="bold", color=TEXT)

    if sub:
        fig.text(0.5, 0.93, sub, ha="center", va="top", fontsize=9, color=MUTE)


def save(fig, path: str, *, show: bool = False, dpi: int = 200) -> None:
    """Write the figure; print a colored confirmation line."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=INK, pad_inches=0.3)
    tty = sys.stdout.isatty()
    arrow = f"{_ANSI_GREEN}→{_ANSI_RESET}" if tty else "→"
    name = f"{_ANSI_DIM}{path}{_ANSI_RESET}" if tty else path
    print(f"  saved {arrow} {name}")

    if show:
        plt.show()
    plt.close(fig)


def iter_log(path: str) -> Iterator[dict]:
    """Yield parsed JSON objects from a JSONL file, skipping blank/malformed lines."""
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()

            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"  [warn] line {lineno}: {exc}", file=sys.stderr)


def confidence_alpha(err, *, soft: float = 0.5, solid: float = 1.0) -> np.ndarray:
    """Per-point opacity from error width: tight CI solid, noisy CI soft.

    Normalized within the set, so it reads *relative* confidence with no
    hard-coded error scale baked in. Feeds `dot`'s `core_alpha`. A flat
    spread (all errors equal, or all zero) is all-solid, no false gradient.
    """
    err = np.asarray(err, dtype=float)

    if err.size == 0:
        return err

    span = float(np.ptp(err))

    if span == 0:
        return np.full(err.shape, solid)
    t = (err - err.min()) / span  # 0 tightest, 1 loosest
    return solid - (solid - soft) * t


def ema(data: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average; `alpha` is the weight on the newest sample."""
    data = np.asarray(data, dtype=float)

    if data.size == 0:
        return data

    out = np.empty_like(data)
    out[0] = data[0]

    for i in range(1, data.size):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
    return out


def format_count(v: float, sig: int = 3) -> str:
    """Hooman counts"""
    if v == 0:
        return "0"

    sign = "-" if v < 0 else ""
    v = abs(v)
    suffixes = ("", "K", "M", "G", "T")
    mag = min(int(math.log10(v)) // 3, len(suffixes) - 1)

    if mag == 0:
        return f"{sign}{int(v)}"

    scaled = v / 1000**mag
    places = max(0, sig - int(math.log10(scaled)) - 1)
    s = f"{scaled:.{places}f}"

    if float(s) >= 1000 and mag < len(suffixes) - 1:
        mag += 1
        s = f"{scaled / 1000:.{sig - 1}f}"
    return f"{sign}{s}{suffixes[mag]}"


# Mirrors soul::color::advantage, so plots and the terminal share one palette.
_WIN = ((0.80, 0.13, 92.0), (0.76, 0.16, 145.0), (0.74, 0.155, 162.0)) # gold→green→teal
_LOSS = ((0.78, 0.11, 45.0), (0.72, 0.16, 35.0), (0.64, 0.17, 22.0))   # peach→orange→brick


def advantage(t: float) -> tuple[float, float, float]:
    """Signed advantage → RGB (0–1). t in [−1, 1]: −1 deep loss, +1 deep win."""
    t = max(-1.0, min(1.0, t))
    m = abs(t)
    seq = _WIN if t >= 0 else _LOSS
    lo, hi, frac = (seq[0], seq[1], m / 0.5) if m < 0.5 else (seq[1], seq[2], (m - 0.5) / 0.5)

    L = lo[0] + (hi[0] - lo[0]) * frac
    C = lo[1] + (hi[1] - lo[1]) * frac
    H = lo[2] + (hi[2] - lo[2]) * frac

    return _oklch_to_srgb(L, C, H)


def diverging_cmap(*, neg_hue: float = 25.0, pos_hue: float = 150.0, l_hi: float = 0.62,
                   l_lo: float = 0.15, c_hi: float = 0.15, power: float = 0.75, n: int = 256):
    """Perceptual diverging colormap in OkLCH: negative red, positive green.

    Lightness is V-shaped (dark center, bright extremes) and chroma tapers to
    zero at the center, so the zero crossing reads as a dark neutral that
    recedes into the ink rather than a hard red/green seam.
    """
    stops = []

    for t in np.linspace(0.0, 1.0, n):
        u = (1.0 - 2.0 * t) if t <= 0.5 else (2.0 * t - 1.0)  # 1 at edges, 0 at center
        hue = neg_hue if t <= 0.5 else pos_hue
        stops.append(_oklch_to_srgb(l_lo + (l_hi - l_lo) * u**power, c_hi * u**power, hue))
    return LinearSegmentedColormap.from_list("soul_div", stops, N=n)


def oklch_ramp(stops, *, n: int = 256):
    """Sequential colormap interpolated through OkLCH waypoints.

    `stops` is a list of (L, C, H) control points. Interpolation is linear in
    OkLCH, so lightness and chroma stay even across the ramp: colors hold their
    saturation through the middle instead of greying out as an RGB lerp would.
    For rank/sequence coloring, where every step must stay legible (unlike
    `diverging_cmap`, whose center is meant to recede).
    """
    knots = [i / (len(stops) - 1) for i in range(len(stops))]
    cols = []

    for t in np.linspace(0.0, 1.0, n):
        k = min(len(stops) - 2, max(0, np.searchsorted(knots, t) - 1))
        f = (t - knots[k]) / (knots[k + 1] - knots[k])
        s0, s1 = stops[k], stops[k + 1]
        cols.append(_oklch_to_srgb(*(a + (b - a) * f for a, b in zip(s0, s1))))
    return LinearSegmentedColormap.from_list("soul_ramp", cols, N=n)


def _oklch_to_srgb(L: float, C: float, H: float) -> tuple[float, float, float]:
    a, b = C * math.cos(math.radians(H)), C * math.sin(math.radians(H))
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3

    r = 4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    bl = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_

    def enc(x: float) -> float:
        x = max(0.0, min(1.0, x))
        return 12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055

    return (enc(r), enc(g), enc(bl))
