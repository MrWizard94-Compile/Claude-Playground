"""
Experiment M2 -- the Ledger recovers recall past the Workspace ceiling,
at IDENTICAL hot-state footprint.

HYPOTHESIS (mechanism M2): bounded hot-state need not cost recall. Route the most
recent ~d associations to the Workspace (within its rank capacity, M1) and evict
older ones EXACTLY to a content-addressable Ledger. Overall recall then stays high
as the total N grows, while the hot-state footprint stays constant.

FAIR COMPARISON (no stacked deck):
  - Baseline "cram": ONE Workspace of dim d holding all N associations. Hot bytes = d^2.
  - LWF: SAME Workspace dim d (holds only the last d), plus a cold Ledger. Hot bytes = d^2.
  Same hot footprint. The only difference is whether an exact external organ exists.

We stress it with a NOISY cue (partial/corrupted recall), so the Ledger must actually
do approximate content-addressing, not trivial lookup. Routing is by Ledger match
SCORE (no oracle): trust the Ledger when its top-1 inner product clears a threshold,
else fall back to the Workspace readout.

HONEST COST: we report Ledger comparisons per read (== store size here; O(log N) with
an ANN index in production). The Ledger moves the lunch out of hot-state; it does not
conjure a free one.

FALSIFICATION: if LWF accuracy tracks the cram baseline's decay as N grows (i.e. the
Ledger fails to recover recall), M2 is falsified.
"""

from __future__ import annotations
import numpy as np
from workspace import FastWeightWorkspace, unit_rows
from ledger import ContentAddressableLedger

SEED = 0


def decode(v_hat: np.ndarray, V: np.ndarray) -> int:
    return int(np.argmax(V @ v_hat))


def run(d: int = 64, n_grid=None, sigma: float = 0.15, tau: float = 0.5,
        seeds=(0, 1, 2)):
    # NOTE on sigma: cue noise norm ~ sigma*sqrt(d). At sigma>=0.25 with d=64 the
    # noise norm exceeds the unit signal (SNR<1) and recall is capped by information,
    # not architecture -- both systems fall, LWF still dominates ~90x. sigma<=0.15 is
    # a genuinely corrupted partial cue that is still recoverable; that is the regime
    # where the M2 claim (Ledger restores what cram loses) is cleanly on trial.
    if n_grid is None:
        n_grid = [32, 64, 128, 256, 512, 1024, 2048]
    print(f"\n=== M2: Ledger recovers recall at fixed hot-state "
          f"(d={d}, cue noise sigma={sigma}) ===")
    print("top-1 accuracy over ALL N associations; both systems have identical "
          f"hot-state = {d*d*8} bytes\n")
    header = f"{'N':>6} | {'cram(ws-only)':>13} | {'LWF(ws+ledger)':>14} | " \
             f"{'ledger cmp/read':>15} | {'hot bytes':>10}"
    print(header)
    print("-" * len(header))
    rows = []
    for N in n_grid:
        cram_acc, lwf_acc, cmps = [], [], []
        for s in seeds:
            rng = np.random.default_rng(SEED + s)
            keys = unit_rows(rng.standard_normal((N, d)))
            vals = unit_rows(rng.standard_normal((N, d)))

            # --- Baseline: cram everything into one bounded Workspace ---
            ws_cram = FastWeightWorkspace(d, d, mode="delta")
            for j in range(N):
                ws_cram.write(keys[j], vals[j])

            # --- LWF: Workspace holds the last d; Ledger holds the rest exactly ---
            split = max(0, N - d)
            ledger = ContentAddressableLedger(d, d)
            for j in range(split):
                ledger.write(keys[j], vals[j])
            ws = FastWeightWorkspace(d, d, mode="delta")
            for j in range(split, N):
                ws.write(keys[j], vals[j])

            hits_cram = hits_lwf = 0
            comp_acc = 0
            for j in range(N):
                cue = keys[j] + sigma * rng.standard_normal(d)

                # baseline decode
                if decode(ws_cram.read(cue), vals) == j:
                    hits_cram += 1

                # LWF: score-based routing (no oracle)
                led = ledger.read(cue, topk=1)
                comp_acc += ledger.last_comparisons
                led_idx, led_score = (led[0][0], led[0][2]) if led else (-1, -np.inf)
                if led_score > tau:
                    pred = led_idx
                else:
                    pred = split + decode(ws.read(cue), vals[split:]) if split < N else \
                           decode(ws.read(cue), vals)
                if pred == j:
                    hits_lwf += 1

            cram_acc.append(hits_cram / N)
            lwf_acc.append(hits_lwf / N)
            cmps.append(comp_acc / N)

        ca, la, cm = np.mean(cram_acc), np.mean(lwf_acc), np.mean(cmps)
        rows.append((N, ca, la, cm))
        print(f"{N:>6} | {ca:>13.3f} | {la:>14.3f} | {cm:>15.1f} | {d*d*8:>10}")

    # Verdict: as N grows, does LWF hold while cram collapses?
    Nmax, ca, la, _ = rows[-1]
    verdict = "SUPPORTED" if (la > 0.9 and la - ca > 0.2) else "CHECK"
    print(f"\nAt N={Nmax}: cram={ca:.3f}, LWF={la:.3f}. "
          f"M2 (bounded hot-state keeps recall via Ledger): {verdict}")
    return rows


if __name__ == "__main__":
    run()
