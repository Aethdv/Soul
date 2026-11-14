# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Concurrent `bench` runner, matching how an OpenBench worker measures speed.

Runs N single-threaded benches in parallel so they contend for cache and memory
bandwidth (`workers` defaults to `cpu_count // 2` for physical cores under
2-way SMT). Stdlib-only so it can be copied next to any engine.

    uv run scripts/bench.py [workers] [sets]
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

# Take the last match so earlier progress/info lines do not win.
_SUMMARY = re.compile(r"(\d+)\s+nodes\s+(\d+)\s+nps")

_TTY = sys.stdout.isatty()
_GOLD = "1;38;2;230;180;80"
_DIM  = "38;2;138;126;104"


def paint(text: str, code: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if _TTY else text


def spread_color(spread: float, *, good: float = 0.025, noisy: float = 0.12) -> str:
    """ANSI truecolor for spread fraction: green at `good`, red at `noisy`."""
    t = min(max((spread - good) / (noisy - good), 0.0), 1.0)
    green, red = (0.76, 0.16, 145.0), (0.64, 0.17, 22.0)
    L, C, H = (a + (b - a) * t for a, b in zip(green, red))
    r, g, b = _oklch_to_srgb(L, C, H)
    return f"38;2;{round(r * 255)};{round(g * 255)};{round(b * 255)}"


def _oklch_to_srgb(L: float, C: float, H: float) -> tuple[float, float, float]:
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
    try:
        res = subprocess.run([engine, "bench"], check=False, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode != 0:
        return None
    matches = _SUMMARY.findall(res.stdout)
    return (int(matches[-1][0]), int(matches[-1][1])) if matches else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Concurrent bench, OpenBench-style.")
    ap.add_argument("workers", nargs="?", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="concurrent benches per batch (default: cpu_count // 2)")
    ap.add_argument("sets", nargs="?", type=int, default=1, help="batches to pool (default: 1)")
    ap.add_argument("--engine", default=ENGINE, help=f"engine binary (default: {ENGINE})")
    args = ap.parse_args()
    engine, workers, sets = args.engine, args.workers, args.sets

    if workers < 1 or sets < 1:
        ap.error("workers and sets must be >= 1")

    # Warm the binary in page cache before timing.
    if bench_once(engine) is None:
        sys.exit(f"error: no bench completed; is {engine} built?")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        runs = [r for _ in range(sets) for r in pool.map(lambda _: bench_once(engine), range(workers)) if r]

    if not runs:
        sys.exit(f"error: no bench completed; is {engine} built?")

    nodes = {n for n, _ in runs}
    if len(nodes) != 1:
        sys.exit(f"error: non-deterministic bench, node counts differ: {sorted(nodes)}")

    speeds = sorted(nps for _, nps in runs)
    mean = sum(speeds) // len(speeds)
    median = statistics.median(speeds)
    spread = (speeds[-1] - speeds[0]) / median
    label = f"{workers}-wide" + (f" × {sets} sets" if sets > 1 else "")

    detail = (f"median {median:,.0f}   min {speeds[0]:,}   max {speeds[-1]:,}"
              f"   ·   {next(iter(nodes)):,} nodes")
    print(f"{paint(f'{mean:,} nps', _GOLD)}   {paint(f'mean of {len(speeds)} runs, {label}', _DIM)}")
    print(f"  {paint(f'spread {spread:.1%}', spread_color(spread))}   {paint(detail, _DIM)}")


if __name__ == "__main__":
    main()
