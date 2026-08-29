"""What correction history does to the score, measured against a build without it.

Builds the engine twice from one tree, the second with `nocorr`, which sends
`corrected_eval` back to the raw evaluation. Searches every position in a suite to
one depth with both and reports the per-position difference.

    python3 scripts/corrcost.py            # depth 20 over src/data/bench.fens
    python3 scripts/corrcost.py 24 30      # depth 24, first 30 positions
"""

import os
import re
import statistics
import subprocess
import sys

FENS = "src/data/bench.fens"


def build(features):
    target = "target/quick/soul"
    cmd = ["cargo", "build", "--quiet", "--profile", "quick"]
    if features:
        cmd += ["--features", features]
    subprocess.run(cmd, check=True, env={**os.environ, "RUSTFLAGS": "-C target-cpu=native"})
    out = f"/tmp/soul-{features or 'base'}"
    subprocess.run(["cp", target, out], check=True)
    return out


def score(binary, fen, depth):
    out = subprocess.run(
        [binary, f"position fen {fen}", f"go depth {depth}"], capture_output=True, text=True
    ).stdout
    final = re.findall(r"score cp (-?\d+) (?!lowerbound|upperbound)", out + " ")
    return int(final[-1]) if final else None


def main():
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    base, nocorr = build(""), build("nocorr")
    fens = [f.strip() for f in open(FENS) if f.strip()][:limit]

    deltas = []
    print(f"  {'on':>6} {'off':>6} {'delta':>6}   position")
    for fen in fens:
        on, off = score(base, fen, depth), score(nocorr, fen, depth)
        if on is None or off is None:
            continue
        deltas.append(on - off)
        print(f"  {on:>6} {off:>6} {on - off:>+6}   {fen.split()[0][:32]}")

    n = len(deltas)
    ranked = sorted(deltas)
    cut = n // 10
    trimmed = ranked[cut : n - cut] if n > 2 * cut else ranked
    spread = statistics.stdev(deltas) if n > 1 else 0.0
    error = spread / (n**0.5) if n else 0.0
    print(f"\ndepth {depth}, {n} positions")
    print(f"  mean   {statistics.mean(deltas):+.2f} +- {2 * error:.2f}   (2 standard errors)")
    print(f"  trim   {statistics.mean(trimmed):+.2f}   (10% off each end, {len(trimmed)} positions)")
    print(f"  median {statistics.median(deltas):+.1f}   spread {spread:.1f}   max |d| {max(abs(d) for d in deltas)}")
    print(f"  sign   {sum(1 for d in deltas if d > 0)} up, {sum(1 for d in deltas if d < 0)} down, {sum(1 for d in deltas if d == 0)} equal")


if __name__ == "__main__":
    main()
