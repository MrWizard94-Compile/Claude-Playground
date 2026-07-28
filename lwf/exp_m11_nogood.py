"""
Experiment M11 -- nogoods: intrinsic verification via learned constraints (from EP-GRM).

EP-GRM's verification is richer than LWF's replay-only M3: it records NOGOODS -- jointly
inconsistent literal sets -- so a contradiction is never re-derived. We imported nogoods into
the Ledger as a first-class record type (ledger.write_nogood / nogood_violated) and here show
they do measurable work.

TASK: graph K-colouring by backtracking search. The Workspace holds the current partial
assignment (bounded); the Ledger stores learned nogoods. When a variable has no consistent
colour (dead end), we extract the conflict set -- one blocking neighbour-literal per eliminated
colour -- and record it as a nogood. Before extending any assignment we check the Ledger: if the
current partial assignment already contains a stored nogood, prune immediately (dependency-
directed backtracking / nogood recording, Stallman-Sussman).

HYPOTHESIS: nogood learning prunes redundant search -- fewer nodes/consistency-checks than plain
backtracking on hard instances -- WITHOUT changing the answer (soundness: a recorded nogood is a
genuine no-extension set, so pruning never discards a real solution).

FALSIFICATION: (a) nogood and plain search disagree on ANY instance's colourability (unsound), or
(b) nogood search explores >= plain on hard instances (no benefit). Either kills the import.
"""

from __future__ import annotations
import numpy as np
from ledger import ContentAddressableLedger

SEED = 0


def random_graph(n, avg_deg, rng):
    p = min(1.0, avg_deg / (n - 1))
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i].add(j); adj[j].add(i)
    return adj


def color_search(adj, K, use_nogood, node_budget=2_000_000):
    n = len(adj)
    ledger = ContentAddressableLedger(1, 1)          # used here only as the nogood store
    assignment = {}
    stats = dict(nodes=0, checks=0, prunes=0, nogoods=0, budget_hit=False)

    def consistent(v, c):
        for u in adj[v]:
            stats["checks"] += 1
            if assignment.get(u) == c:
                return False
        return True

    def bt(idx):
        if stats["nodes"] > node_budget:
            stats["budget_hit"] = True
            return False
        if idx == n:
            return True
        # nogood prune: does the current partial assignment entail a known contradiction?
        if use_nogood and ledger.nogood_violated(frozenset(assignment.items())):
            stats["prunes"] += 1
            return False
        v = idx                                       # fixed order (isolate nogood effect)
        blocker = {}                                  # colour -> a neighbour literal blocking it
        for c in range(K):
            stats["nodes"] += 1
            if consistent(v, c):
                assignment[v] = c
                if bt(idx + 1):
                    return True
                del assignment[v]
                if stats["budget_hit"]:
                    return False
            else:
                for u in adj[v]:                      # remember one blocker of colour c
                    if assignment.get(u) == c:
                        blocker[c] = (u, c); break
        if use_nogood and len(blocker) == K:          # v uncolourable -> learn the conflict set
            if ledger.write_nogood(blocker.values()) >= 0:
                stats["nogoods"] += 1
        return False

    sat = bt(0)
    stats["n_nogoods_stored"] = ledger.n_nogoods
    return sat, stats


def run(n=26, avg_deg=4.7, K=3, seeds=range(12)):
    print(f"\n=== M11: nogood-learning search (n={n}, avg_deg={avg_deg}, {K}-colouring) ===")
    print("graph K-colouring by backtracking; Ledger = nogood store. hard random instances "
          "near the 3-colouring threshold.\n")
    hdr = f"{'seed':>4} | {'sat':>4} | {'plain nodes':>11} | {'nogood nodes':>12} | " \
          f"{'speedup':>7} | {'nogoods':>7} | {'sound?':>6}"
    print(hdr); print("-" * len(hdr))
    tot_plain = tot_ng = 0
    all_sound = True
    speedups = []
    for s in seeds:
        rng = np.random.default_rng(SEED + s)
        adj = random_graph(n, avg_deg, rng)
        sat_p, sp = color_search(adj, K, use_nogood=False)
        sat_n, sn = color_search(adj, K, use_nogood=True)
        sound = (sat_p == sat_n) and not sp["budget_hit"] and not sn["budget_hit"]
        all_sound &= sound
        spd = sp["nodes"] / max(sn["nodes"], 1)
        speedups.append(spd)
        tot_plain += sp["nodes"]; tot_ng += sn["nodes"]
        print(f"{s:>4} | {str(sat_p):>4} | {sp['nodes']:>11} | {sn['nodes']:>12} | "
              f"{spd:>6.2f}x | {sn['nogoods']:>7} | {str(sound):>6}")

    print(f"\ntotals: plain {tot_plain} nodes vs nogood {tot_ng} nodes "
          f"-> {tot_plain/max(tot_ng,1):.2f}x fewer; median per-instance speedup "
          f"{np.median(speedups):.2f}x")
    ok = all_sound and (tot_plain > tot_ng)
    print(f"M11 (nogoods prune redundant search, soundly): {'SUPPORTED' if ok else 'CHECK'}")
    if not all_sound:
        print("  !! SOUNDNESS VIOLATION -- nogood and plain disagreed; import is UNSOUND")
    return None


if __name__ == "__main__":
    run()
