"""
Experiment M1 -- the Workspace rank-capacity ceiling.

HYPOTHESIS (from mechanism M1): a fixed d_v x d_k fast-weight Workspace can store
at most ~min(d_v,d_k) linearly-independent associations exactly; recall degrades
predictably beyond that. This is the *provable* failure mode we designed the
Ledger (M2) to catch.

FALSIFICATION LOGIC:
  - With ORTHONORMAL keys, capacity should be an exact wall at N = d (can't fit
    more than d orthonormal vectors in R^d). Delta-rule recall should be ~perfect
    for N<=d, then fall.
  - With RANDOM keys, recall should degrade smoothly as N approaches/exceeds d
    (crosstalk grows like sqrt(N/d)).
  - If instead recall collapses far BELOW d (random keys), the rank model of
    Workspace capacity is WRONG and M1 is falsified.

Metric: top-1 retrieval accuracy. For each stored key we read v_hat = M@k and check
whether v_hat's nearest stored value (cosine) is the correct one. Chance = 1/N.
"""

from __future__ import annotations
import numpy as np
from workspace import FastWeightWorkspace, unit_rows

SEED = 0


def make_keys(N: int, d: int, rng: np.random.Generator, orthonormal: bool) -> np.ndarray:
    if orthonormal:
        A = rng.standard_normal((d, d))
        Q, _ = np.linalg.qr(A)            # d orthonormal columns
        if N <= d:
            return Q[:, :N].T.copy()
        # past d: reuse directions (forces linear dependence, exposes the wall)
        extra = rng.standard_normal((N - d, d))
        return unit_rows(np.vstack([Q.T, extra]))
    return unit_rows(rng.standard_normal((N, d)))


def top1_accuracy(ws: FastWeightWorkspace, keys: np.ndarray, vals: np.ndarray) -> float:
    V = vals                                    # (N, d_v), assumed unit-norm rows
    hits = 0
    for j in range(keys.shape[0]):
        v_hat = ws.read(keys[j])
        sims = V @ v_hat                        # cosine up to a positive scale
        if int(np.argmax(sims)) == j:
            hits += 1
    return hits / keys.shape[0]


def run(d: int = 64, n_grid=None, seeds=(0, 1, 2)):
    if n_grid is None:
        n_grid = [4, 8, 16, 32, 48, 64, 80, 96, 128, 192, 256]
    print(f"\n=== M1: Workspace rank-capacity ceiling (d_k=d_v={d}) ===")
    print("top-1 retrieval accuracy (mean over seeds); chance = 1/N\n")
    header = f"{'N':>5} | {'N/d':>5} | {'hebb-rand':>10} {'delta-rand':>11} " \
             f"{'hebb-orth':>10} {'delta-orth':>11} | {'chance':>7}"
    print(header)
    print("-" * len(header))
    results = []
    for N in n_grid:
        acc = {("hebb", False): [], ("delta", False): [],
               ("hebb", True): [], ("delta", True): []}
        for s in seeds:
            rng = np.random.default_rng(SEED + s)
            for mode in ("hebb", "delta"):
                for orth in (False, True):
                    keys = make_keys(N, d, rng, orthonormal=orth)
                    vals = unit_rows(rng.standard_normal((N, d)))
                    ws = FastWeightWorkspace(d, d, mode=mode)
                    for j in range(N):
                        ws.write(keys[j], vals[j])
                    acc[(mode, orth)].append(top1_accuracy(ws, keys, vals))
        row = {k: float(np.mean(v)) for k, v in acc.items()}
        results.append((N, row))
        print(f"{N:>5} | {N/d:>5.2f} | "
              f"{row[('hebb', False)]:>10.3f} {row[('delta', False)]:>11.3f} "
              f"{row[('hebb', True)]:>10.3f} {row[('delta', True)]:>11.3f} | "
              f"{1.0/N:>7.3f}")

    # Verdict on the orthonormal delta wall
    below = [r for (N, r) in results if N <= d]
    above = [r for (N, r) in results if N > d]
    do_below = np.mean([r[("delta", True)] for r in below]) if below else float("nan")
    do_above = np.mean([r[("delta", True)] for r in above]) if above else float("nan")
    print(f"\ndelta+orthonormal: mean acc for N<=d = {do_below:.3f}, N>d = {do_above:.3f}")
    verdict = "SUPPORTED" if (do_below > 0.95 and do_above < do_below - 0.1) else "CHECK"
    print(f"M1 rank-wall prediction (exact <=d, degrade >d): {verdict}")
    return results


if __name__ == "__main__":
    run()
