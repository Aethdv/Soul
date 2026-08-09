"""What XorBoard costs, measured against a build without it.

Builds the engine twice from one tree, the second with `nostore`, which drops
the store and sends `checkers` and the picker's threat map back to their
from-scratch forms. Runs both under perf, interleaved so clock drift cannot
favor one, and reports per node.

    make storecost          # 5 runs each
    make storecost RUNS=9

Instructions are deterministic, so a difference of zero between the two
binaries means the feature gate did not take. Cycles are not: the minimum is
the honest estimator, since interference only ever adds, and the median is
printed beside it to show whether one lucky run carried the verdict.
"""

import os
import re
import statistics
import subprocess
import sys

BUILDS = [("with", []), ("without", ["--features", "nostore"])]
COUNTERS = re.compile(r"^\s*([\d,]+)\s+(instructions|cycles)", re.MULTILINE)
NODES = re.compile(r"Bench: (\d+) nodes")


def build(label, flags):
    subprocess.run(
        ["cargo", "build", "--release", "--quiet", *flags],
        env={**os.environ, "RUSTFLAGS": "-C target-cpu=native"},
        check=True,
    )
    path = f"/tmp/soul-{label}"
    subprocess.run(["cp", "target/release/soul", path], check=True)
    return path


def measure(binary):
    run = subprocess.run(
        ["perf", "stat", "-e", "instructions,cycles", binary, "bench"],
        capture_output=True,
        text=True,
        check=True,
    )
    text = run.stdout + run.stderr
    counts = {name: int(value.replace(",", "")) for value, name in COUNTERS.findall(text)}
    bench = NODES.search(text)
    if bench is None:
        raise SystemExit(f"no bench line from {binary}:\n{text}")

    nodes = int(bench.group(1))
    return counts["instructions"] / nodes, counts["cycles"] / nodes, nodes


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    binaries = {label: build(label, flags) for label, flags in BUILDS}
    samples = {label: [] for label in binaries}
    nodes = 0

    for _ in range(runs):
        for label, path in binaries.items():
            insn, cycles, nodes = measure(path)
            samples[label].append((insn, cycles))

    print(f"\nbench {nodes} nodes, {runs} runs interleaved, per node\n")
    print(f"{'':9}{'insn':>10}{'cyc min':>10}{'cyc med':>10}")

    stats = {}
    for label, rows in samples.items():
        stats[label] = (
            min(row[0] for row in rows),
            min(row[1] for row in rows),
            statistics.median(row[1] for row in rows),
        )
        print(f"{label:9}{stats[label][0]:10.1f}{stats[label][1]:10.1f}{stats[label][2]:10.1f}")

    cost = [stats["with"][i] - stats["without"][i] for i in range(3)]
    print(f"\n{'store':9}{cost[0]:+10.1f}{cost[1]:+10.1f}{cost[2]:+10.1f}")

    if abs(cost[0]) < 0.05:
        print("\ninstructions identical: the nostore gate did not take")


if __name__ == "__main__":
    main()
