# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Minimal UCI engine subprocess wrapper."""

from __future__ import annotations

import queue
import subprocess
import threading
import time

from pathlib import Path
from typing import NamedTuple

MATE = 29_000


class Score(NamedTuple):
    """Stm-relative search score (`mate` mapped to `±MATE` in `cp`)."""
    cp: int
    mate: int | None = None


def limit_str(*, nodes: int | None = None, movetime: int | None = None,
              depth: int | None = None) -> str:
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
    """UCI engine subprocess; use as a context manager. `search` returns stm-relative scores."""

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

        self._q: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

        try:
            self._cmd("uci");     self._wait("uciok")

            if threads is not None:
                self._cmd(f"setoption name Threads value {threads}")

            if hash_mb is not None:
                self._cmd(f"setoption name Hash value {hash_mb}")
            self._cmd("isready"); self._wait("readyok")
        except Exception:
            self.close()
            raise

    def new_game(self) -> None:
        self._cmd("ucinewgame")
        self._cmd("isready")
        self._wait("readyok")

    def search(self, position: str, limit: str) -> Score:
        """Send `position <position>` and `go <limit>`; return stm-relative `Score`."""
        self._cmd(f"position {position}")
        self._cmd(f"go {limit}")
        cp, mate = 0, None

        while True:
            tokens = self._get().split()
            if not tokens:
                continue

            if tokens[0] == "info" and "score" in tokens:
                try:
                    score_i = tokens.index("score")
                    kind, val = tokens[score_i + 1], int(tokens[score_i + 2])

                    if kind == "cp":
                        cp, mate = val, None
                    elif kind == "mate":
                        cp, mate = (MATE if val > 0 else -MATE), val
                except (ValueError, IndexError):
                    pass

            if tokens[0] == "bestmove":
                return Score(cp, mate)

    def _read(self) -> None:
        assert self._proc.stdout
        try:
            for line in self._proc.stdout:
                self._q.put(line.rstrip())
        finally:
            self._q.put(None)

    def _cmd(self, s: str) -> None:
        assert self._proc.stdin
        self._proc.stdin.write(s + "\n")
        self._proc.stdin.flush()

    def _get(self, timeout: float | None = None) -> str:
        wait = self._timeout if timeout is None else timeout
        try:
            line = self._q.get(timeout=wait)
        except queue.Empty:
            raise TimeoutError("Engine unresponsive") from None
        if line is None:
            raise EOFError(f"Engine exited (rc={self._proc.poll()})")
        return line

    def _wait(self, tok: str) -> None:
        deadline = time.monotonic() + self._timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for '{tok}'")
            if self._get(remaining).split()[:1] == [tok]:
                return

    def close(self) -> None:
        try:
            if self._proc.poll() is None:
                self._cmd("quit")
                self._proc.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            self._proc.kill()
            self._proc.wait()
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass

    def __enter__(self) -> UCIEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
