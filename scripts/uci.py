# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
uci - a small UCI engine harness for the engine-driven plotters.

A line-buffered subprocess with a background reader thread, the UCI handshake,
and one search primitive. Engine-agnostic: it speaks the protocol and nothing
more, so it drives any UCI engine. Score sign conventions (white-relative,
side-to-move) are left to the callers.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time

from pathlib import Path
from typing import NamedTuple

MATE = 29_000


class Score(NamedTuple):
    """A search result. `cp` is stm-relative centipawns with mate mapped to ±MATE
    (so it positions and sorts sanely); `mate` is the signed moves-to-mate from
    the engine (stm view, + = stm mates), or None for a normal eval."""
    cp: int
    mate: int | None = None


def limit_str(*, nodes: int | None = None, movetime: int | None = None,
              depth: int | None = None) -> str:
    """UCI `go` limit clause from whichever budget is set (depth > movetime > nodes)."""
    if depth is not None:
        return f"depth {depth}"

    if movetime is not None:
        return f"movetime {movetime}"

    if nodes is not None:
        return f"nodes {nodes}"
    raise ValueError("limit_str: supply one of nodes / movetime / depth")


def limit_label(*, nodes: int | None = None, movetime: int | None = None,
                depth: int | None = None) -> str:
    """Hooman readable search budget: 'depth 8' / '200ms' / '100,000 nodes'."""
    if depth is not None:
        return f"depth {depth}"

    if movetime is not None:
        return f"{movetime}ms"

    if nodes is not None:
        return f"{nodes:,} nodes"
    return "?"


class UCIEngine:
    """A UCI engine subprocess; use as a context manager.

    `search` returns the engine's reported score in centipawns from the
    side-to-move's perspective, mate mapped to ±MATE. Callers apply their own
    perspective convention (negate for opponent-to-move, etc.).
    """

    def __init__(self, path: str, *, threads: int | None = None, hash_mb: int | None = None,
                 timeout: float = 120.0) -> None:
        exe = Path(path).resolve()

        if not exe.exists():
            raise FileNotFoundError(f"Engine not found: {exe}")

        self._timeout = timeout
        self._proc = subprocess.Popen(
            [str(exe)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )

        self._q: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self._cmd("uci");     self._wait("uciok")

        if threads is not None:
            self._cmd(f"setoption name Threads value {threads}")

        if hash_mb is not None:
            self._cmd(f"setoption name Hash value {hash_mb}")
        self._cmd("isready"); self._wait("readyok")

    def new_game(self) -> None:
        """Reset search state between unrelated positions (clears TT/history)."""
        self._cmd("ucinewgame")
        self._cmd("isready")
        self._wait("readyok")

    def search(self, position: str, limit: str) -> Score:
        """`position <position>` then `go <limit>`; return the stm-relative Score."""
        self._cmd(f"position {position}")
        self._cmd(f"go {limit}")
        cp, mate = 0, None

        while True:
            line = self._get()
            tokens = line.split()

            if "score" in tokens:
                try:
                    score_i = tokens.index("score")
                    kind, val = tokens[score_i + 1], int(tokens[score_i + 2])

                    if kind == "cp":
                        cp, mate = val, None
                    elif kind == "mate":
                        cp, mate = (MATE if val > 0 else -MATE), val
                except (ValueError, IndexError):
                    pass

            if line.startswith("bestmove"):
                return Score(cp, mate)

    def _read(self) -> None:
        assert self._proc.stdout

        for line in self._proc.stdout:
            self._q.put(line.rstrip())

    def _cmd(self, s: str) -> None:
        assert self._proc.stdin
        self._proc.stdin.write(s + "\n")
        self._proc.stdin.flush()

    def _get(self) -> str:
        try:
            return self._q.get(timeout=self._timeout)
        except queue.Empty:
            raise TimeoutError("Engine unresponsive") from None

    def _wait(self, tok: str) -> None:
        deadline = time.monotonic() + self._timeout

        while True:
            remaining = deadline - time.monotonic()

            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for '{tok}'")
            try:
                if tok in self._q.get(timeout=remaining):
                    return
            except queue.Empty:
                raise TimeoutError(f"Timed out waiting for '{tok}'") from None

    def close(self) -> None:
        try:
            self._cmd("quit")
            self._proc.wait(timeout=3)
        except Exception:
            self._proc.kill()
            self._proc.wait()

    def __enter__(self) -> UCIEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
