"""
LWF Workspace -- the bounded executive memory (Tier 1).

Implements the fast-weight associative memory from mechanism M1:

    Hebbian:  M_t = decay * M_{t-1} + outer(v_t, k_t)
    Delta:    M_t = M_{t-1} + outer(v_t - M_{t-1} k_t, k_t)   (DeltaNet / online LS)

State is a fixed d_v x d_k matrix. Per-step cost and memory are O(d_v*d_k),
INDEPENDENT of how many associations have streamed through it (task duration).

The whole point of this file is to *expose*, not hide, the rank ceiling:
M has rank <= min(d_v, d_k), so it can represent at most that many linearly
independent key->value maps exactly. Everything past that is interference.
That ceiling is what mechanism M1 predicts and what exp_m1 measures.
"""

from __future__ import annotations
import numpy as np


def unit_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize along the last axis. Keys must be unit-norm for the
    associative read M@k to return the stored value with unit gain."""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


class FastWeightWorkspace:
    """Fixed-size associative working state. O(1) in task length by construction."""

    def __init__(self, d_k: int, d_v: int, mode: str = "hebb", decay: float = 1.0):
        assert mode in ("hebb", "delta")
        self.d_k = d_k
        self.d_v = d_v
        self.mode = mode
        self.decay = float(decay)
        self.M = np.zeros((d_v, d_k), dtype=np.float64)
        self.n_writes = 0

    @property
    def state_bytes(self) -> int:
        """Hot-state footprint. Constant. This is the quantity Goal #1 bounds."""
        return self.M.size * self.M.itemsize

    def write(self, k: np.ndarray, v: np.ndarray) -> None:
        k = k.astype(np.float64)
        v = v.astype(np.float64)
        if self.mode == "hebb":
            self.M = self.decay * self.M + np.outer(v, k)
        else:  # delta rule: subtract current prediction before writing
            pred = self.M @ k
            self.M = self.decay * self.M + np.outer(v - self.decay * pred, k)
        self.n_writes += 1

    def read(self, q: np.ndarray) -> np.ndarray:
        """Associative recall. Returns v_hat = M @ q."""
        return self.M @ q.astype(np.float64)

    def rank(self, tol: float = 1e-9) -> int:
        return int(np.linalg.matrix_rank(self.M, tol=tol))
