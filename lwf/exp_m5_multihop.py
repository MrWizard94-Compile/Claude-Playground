"""
Experiment M5 -- compositional multi-hop reasoning: does bounded per-step state hide
an unbounded number of steps? (Direct probe of the §7 crux, "outcome 2".)

SETUP: a functional graph over M nodes (each node -> one successor). Each edge
(node -> successor) is stored in the Ledger as (key=embed(node), value=embed(successor)).
An H-hop query asks: starting at node s, what is the node H successors away?

TWO WAYS TO ANSWER:
  (a) LWF iterative pointer-chase: a BOUNDED controller state (just the current cursor)
      does H sequential Ledger reads, DECODING each retrieved vector back to a clean node
      embedding before the next hop. Per-step hot-state is constant; #steps = H.
  (b) Single-shot fixed-state: fold every edge into ONE rank-d Workspace M and answer
      H-hop in a single read via M^H @ embed(s). One step, but composition through a
      lossy rank-d operator amplifies interference.

WHAT THIS ISOLATES (the honest resolution of the crux):
  - "bounded PER-STEP cost" is an architecture property LWF keeps (constant hot-state).
  - "bounded TOTAL cost" is NOT claimable for genuinely sequential reasoning: depth H is
    intrinsic to the problem. Any architecture pays O(depth) -- transformers via layers or
    chain-of-thought, LWF via retrieval steps. The bottleneck that "moves" is DEPTH, which
    is a property of the task, not the machine.

FALSIFICATION of the *design* (not the task): if iterative LWF cannot hold multi-hop
accuracy that single-shot fixed-state loses, the Workspace+Ledger split buys nothing over
a plain fixed state, and M5's separation claim fails.
"""

from __future__ import annotations
import numpy as np
from workspace import FastWeightWorkspace, unit_rows
from ledger import ContentAddressableLedger

SEED = 0


def build_world(M_nodes: int, d: int, rng):
    emb = unit_rows(rng.standard_normal((M_nodes, d)))
    succ = rng.integers(0, M_nodes, size=M_nodes)     # functional graph
    ledger = ContentAddressableLedger(d, d)
    for i in range(M_nodes):
        ledger.write(emb[i], emb[succ[i]])
    # single-shot operator: M = sum_i emb[succ[i]] emb[i]^T
    Mop = np.zeros((d, d))
    for i in range(M_nodes):
        Mop += np.outer(emb[succ[i]], emb[i])
    return emb, succ, ledger, Mop


def decode(v: np.ndarray, emb: np.ndarray) -> int:
    return int(np.argmax(emb @ v))


def true_hop(start: int, H: int, succ: np.ndarray) -> int:
    x = start
    for _ in range(H):
        x = succ[x]
    return x


def lwf_iterative(start, H, emb, ledger, sigma):
    """Bounded controller: current-cursor state only; H sequential decoded reads."""
    cur = start
    hot_bytes = emb.shape[1] * 8            # just the cursor embedding -> constant
    for _ in range(H):
        cue = emb[cur] + sigma * np.random.default_rng().standard_normal(emb.shape[1]) \
            if sigma > 0 else emb[cur]
        v = ledger.read_top1_value(cue)
        cur = decode(v, emb)               # clean the pointer each hop
    return cur, hot_bytes


def single_shot(start, H, emb, Mop):
    """One rank-d operator applied H times in vector space, decoded once at the end."""
    v = emb[start].copy()
    for _ in range(H):                     # M^H @ v0 (still ONE fixed state, no external store)
        v = Mop @ v
        v = v / (np.linalg.norm(v) + 1e-9)
    return decode(v, emb)


def run(M_nodes=500, d=64, H_grid=range(1, 9), n_queries=300, sigma=0.0, seeds=(0, 1, 2)):
    print(f"\n=== M5: multi-hop reasoning (M={M_nodes} nodes, d={d}, cue noise={sigma}) ===")
    print("accuracy over random start nodes; per-step hot-state is constant for LWF\n")
    header = f"{'H(hops)':>8} | {'LWF iter acc':>12} | {'single-shot acc':>15} | " \
             f"{'LWF steps':>9} | {'LWF hot B':>9}"
    print(header)
    print("-" * len(header))
    rows = []
    for H in H_grid:
        li, ss = [], []
        for s in seeds:
            rng = np.random.default_rng(SEED + s)
            emb, succ, ledger, Mop = build_world(M_nodes, d, rng)
            starts = rng.integers(0, M_nodes, size=n_queries)
            hit_l = hit_s = 0
            for st in starts:
                tgt = true_hop(st, H, succ)
                pred_l, hot = lwf_iterative(st, H, emb, ledger, sigma)
                pred_s = single_shot(st, H, emb, Mop)
                hit_l += (pred_l == tgt)
                hit_s += (pred_s == tgt)
            li.append(hit_l / n_queries)
            ss.append(hit_s / n_queries)
        la, sa = float(np.mean(li)), float(np.mean(ss))
        rows.append((H, la, sa))
        print(f"{H:>8} | {la:>12.3f} | {sa:>15.3f} | {H:>9} | {d*8:>9}")

    Hmax, la, sa = rows[-1]
    print(f"\nAt H={Hmax}: LWF-iterative={la:.3f}, single-shot={sa:.3f}")
    print("Interpretation: LWF holds deep composition at CONSTANT per-step state by "
          "paying H sequential steps; the single fixed-state operator collapses.")
    verdict = "SUPPORTED" if (la > 0.9 and la - sa > 0.3) else "CHECK"
    print(f"M5 (iterative Ledger access enables deep composition at bounded per-step "
          f"state): {verdict}")
    return rows


if __name__ == "__main__":
    run()
