# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Runs `./engine bench` in N parallel processes and pools their nps, mirroring how an
OpenBench worker measures speed: N independent single-threaded benches contending
for cache and memory bandwidth, not one clean solo run. That contended mean is the
figure comparable to what OB reports; a lone `soul bench` already gives peak
single-threaded nps, so re-running it sequentially would only re-measure the peak.

    uv run scripts/bench.py [workers] [sets]

  workers  concurrent benches per batch  (default: cpu_count // 2)
  sets     batches to pool               (default: 1)

The default halves the logical CPU count to land on physical cores in the common
two-way-hyperthreading case, the way an OB worker's concurrency is set; on a box
without HT, pass the core count. The spread color flags a choice running too hot.

Single file, standard library only: drop it next to any engine and it works.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import statistics
import subprocess
import sys

from concurrent.futures import ThreadPoolExecutor

ENGINE = "./soul"

# `<nodes> nodes <nps> nps`, with or without a leading `Bench:`.
# Take the last match so any info/progress lines printed ahead of it don't win.
_SUMMARY = re.compile(r"(\d+)\s+nodes\s+(\d+)\s+nps")

_TTY = sys.stdout.isatty()
_GOLD = "1;38;2;230;180;80" # the headline nps
_DIM  = "38;2;138;126;104"  # labels and secondary numbers


def paint(text: str, code: str) -> str:
    """ANSI-wrap `text` on a tty; leave it plain when piped or redirected."""
    return f"\x1b[{code}m{text}\x1b[0m" if _TTY else text


def spread_color(spread: float, *, good: float = 0.025, noisy: float = 0.12) -> str:
    """ANSI truecolor for a spread fraction: green when tight, red when noisy.

    Interpolated in OkLCH, so the hue sweeps through amber as the spread grows
    instead of snapping between buckets. The anchors span the range that matters:
    `good` is a realistic-best spread, `noisy` is a batch worth distrusting.
    """
    t = min(max((spread - good) / (noisy - good), 0.0), 1.0)
    green, red = (0.76, 0.16, 145.0), (0.64, 0.17, 22.0)
    L, C, H = (a + (b - a) * t for a, b in zip(green, red))
    r, g, b = _oklch_to_srgb(L, C, H)
    return f"38;2;{round(r * 255)};{round(g * 255)};{round(b * 255)}"


def _oklch_to_srgb(L: float, C: float, H: float) -> tuple[float, float, float]:
    """OkLCH (lightness, chroma, hue°) → sRGB in [0, 1].

    Oklab's perceptual space: a straight line between two colors reads as an even
    gradient, where the same lerp in raw sRGB would muddy through grey.
    """
    a, b = C * math.cos(math.radians(H)), C * math.sin(math.radians(H))
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3

    r  =  4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g  = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    bl = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_

    def gamma(x: float) -> float:
        x = max(0.0, min(1.0, x))
        return 12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055

    return gamma(r), gamma(g), gamma(bl)


def bench_once(engine: str) -> tuple[int, int] | None:
    """One `soul bench` → (nodes, nps), or None if it didn't run or didn't parse."""
    try:
        out = subprocess.run([engine, "bench"], capture_output=True, text=True, timeout=120).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    matches = _SUMMARY.findall(out)
    return (int(matches[-1][0]), int(matches[-1][1])) if matches else None


def run_batch(engine: str, workers: int) -> list[tuple[int, int]]:
    """`workers` benches at once; the contention between them is the measurement."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [r for r in pool.map(lambda _: bench_once(engine), range(workers)) if r]


def main() -> None:
    ap = argparse.ArgumentParser(description="Concurrent bench, OpenBench-style.")
    ap.add_argument("workers", nargs="?", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="concurrent benches per batch (default: cpu_count // 2)")
    ap.add_argument("sets", nargs="?", type=int, default=1, help="batches to pool (default: 1)")
    ap.add_argument("--engine", default=ENGINE, help=f"engine binary (default: {ENGINE})")
    args = ap.parse_args()
    engine, workers, sets = args.engine, args.workers, args.sets

    bench_once(engine)  # warm the binary into page cache so the first timed run isn't cold
    runs = [r for _ in range(sets) for r in run_batch(engine, workers)]

    if not runs:
        sys.exit(f"error: no bench completed — is {engine} built?")

    # A deterministic bench reports the same node count every run; a split means
    # the search is non-deterministic, which is a real bug worth surfacing.
    nodes = {n for n, _ in runs}
    if len(nodes) != 1:
        sys.exit(f"error: non-deterministic bench, node counts differ: {sorted(nodes)}")

    speeds = sorted(nps for _, nps in runs)
    mean = sum(speeds) // len(speeds)
    median = statistics.median(speeds)
    spread = (speeds[-1] - speeds[0]) / median  # run-to-run range; the trust signal
    label = f"{workers}-wide" + (f" × {sets} sets" if sets > 1 else "")

    detail = (f"median {median:,.0f}   min {speeds[0]:,}   max {speeds[-1]:,}"
              f"   ·   {next(iter(nodes)):,} nodes")
    print(f"{paint(f'{mean:,} nps', _GOLD)}   {paint(f'mean of {len(speeds)} runs, {label}', _DIM)}")
    print(f"  {paint(f'spread {spread:.1%}', spread_color(spread))}   {paint(detail, _DIM)}")


if __name__ == "__main__":
    main()
