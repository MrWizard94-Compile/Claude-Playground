"""
Experiment M21 -- complete search with BOUNDED executive state (the search+verification angle).

M19/M20 deflated the "bounded state can't COMPUTE" crux. The real bottleneck for intelligent
computation (esp. autonomous synthesis, Goal #7) is SEARCH + VERIFICATION. M16 exposed the crux of
THAT: local energy-descent is complete-in-memory but INCOMPLETE (local minima). Systematic
backtracking is complete -- but its trail/stack is O(n), which VIOLATES bounded executive state
(the whole point of LWF). So the genuine open question, unifying the search angle with LWF's core:

  Can complete search run with EXECUTIVE state bounded INDEPENDENT of problem size, by keeping the
  assignment / trail / nogoods in the LEDGER (persistent store, allowed to grow) and only the active
  dependency FRONTIER in the WORKSPACE (bounded)?

This is the LWF Workspace/Ledger split (and TSAM's active window, EP-GRM's focus, M10's frontier)
applied to search. Task: graph K-colouring (hard instances near the threshold), backtracking + nogood
learning (M11). We run the SAME search two ways and instrument memory:
  full      -- everything (assignment + trail + nogoods) in one executive working set: O(n).
  lwf       -- assignment/trail/nogoods in the Ledger; the Workspace at each step holds only the
               current variable + its neighbours read for the consistency check (the frontier).

=====================================================================================
PRE-REGISTERED VERDICT (fixed before run; reported verbatim; NOT reframable post-hoc).
  Over a sweep of n (sparse, bounded-degree graphs so frontier is bounded):
  P1 completeness preserved: lwf solves EXACTLY the same colourable instances as full (identical
     SAT verdicts) -- the offload changes memory location, not the search.
  P2 executive-state boundedness: peak WORKSPACE (active window) stays bounded and does NOT grow
     with n -- Pearson corr(peak_workspace, n) < 0.3 AND peak_workspace <= 3*avg_degree at max n,
     WHILE the Ledger assignment size == n (grows). full's executive == n.
  P3 search preserved: node counts identical between lwf and full (same algorithm).
  ALSO REPORTED (the honest cost): Ledger reads/step (data movement to the persistent store).
  PASS = P1 and P2 and P3. Else HONEST NEGATIVE.
=====================================================================================
CPU, deterministic. Sparse graphs (bounded degree) so the frontier -- hence the Workspace -- is
bounded; dense/high-frontier graphs are the known boundary (M10), reported separately.
"""

from __future__ import annotations
import numpy as np

SEED = 0


def random_graph(n, avg_deg, rng):
    adj = [set() for _ in range(n)]
    m = int(n * avg_deg / 2)
    trials = 0
    while sum(len(a) for a in adj) < 2 * m and trials < 20 * m:
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i != j and j not in adj[i]:
            adj[i].add(j); adj[j].add(i)
        trials += 1
    return adj


def search(adj, K, mode, node_budget=1_500_000):
    """Backtracking + conflict-directed nogood learning. `mode` controls memory accounting:
    'full' = executive holds assignment+trail; 'lwf' = Ledger holds them, Workspace = frontier."""
    n = len(adj)
    assignment = {}                      # in LWF this lives in the Ledger (persistent)
    nogoods = []                         # Ledger: learned conflict sets
    nogood_index = {}
    stats = dict(nodes=0, peak_workspace=0, ledger_reads=0, nogoods=0, budget_hit=False)

    def nogood_violated(assign_items_set):
        cands = set()
        for lit in assign_items_set:
            cands.update(nogood_index.get(lit, ()))
        for nid in cands:
            if nogoods[nid] <= assign_items_set:
                return True
        return False

    def add_nogood(literals):
        ng = frozenset(literals)
        if ng and not any(e <= ng for e in nogoods):
            nid = len(nogoods); nogoods.append(ng)
            for lit in ng:
                nogood_index.setdefault(lit, []).append(nid)
            stats["nogoods"] += 1

    def bt(idx):
        if stats["nodes"] > node_budget:
            stats["budget_hit"] = True; return False
        if idx == n:
            return True
        v = idx
        # WORKSPACE at this step = {v} + v's already-assigned neighbours read for the check.
        # In 'lwf' mode these neighbour colours are READ FROM THE LEDGER (data movement); the
        # active window is frontier-sized. In 'full' mode the whole assignment is 'in hand'.
        nbrs_assigned = [u for u in adj[v] if u in assignment]
        if mode == "lwf":
            workspace = 1 + len(nbrs_assigned)              # current var + frontier read
            stats["ledger_reads"] += len(nbrs_assigned)
        else:
            workspace = 1 + len(assignment)                 # full executive holds everything
        stats["peak_workspace"] = max(stats["peak_workspace"], workspace)
        blocker = {}
        for c in range(K):
            stats["nodes"] += 1
            conflict = next(((u, c) for u in nbrs_assigned if assignment[u] == c), None)
            if conflict is None:
                assignment[v] = c
                # nogood pruning is part of the SEARCH -- applied in BOTH modes so they run the
                # identical algorithm (only the memory ACCOUNTING differs). [bugfix: was lwf-only]
                if nogood_violated(frozenset(assignment.items())):
                    del assignment[v]; continue
                if bt(idx + 1):
                    return True
                del assignment[v]
                if stats["budget_hit"]:
                    return False
            else:
                blocker[c] = conflict
        if len(blocker) == K:
            add_nogood(blocker.values())
        return False

    sat = bt(0)
    stats["assignment_size"] = n            # Ledger assignment footprint
    return sat, stats


def run(n_grid=(20, 40, 60, 80), avg_deg=4.5, K=3, seeds=range(8)):
    print(f"\n=== M21: complete search with BOUNDED executive state (K={K}-colouring, "
          f"avg_deg={avg_deg}) ===")
    print("same backtracking+nogood search, two memory accountings: full (O(n)) vs lwf "
          "(Workspace=frontier, Ledger=assignment/trail/nogoods)\n")
    hdr = f"{'n':>4} | {'sat%':>5} | {'nodes(full)':>11} {'nodes(lwf)':>10} | " \
          f"{'peakWS full':>11} {'peakWS lwf':>10} | {'ledgerRd/node':>13}"
    print(hdr); print("-" * len(hdr))
    all_ok_p1 = all_ok_p3 = True
    peak_ws_by_n, avg_deg_seen = {}, 0.0
    for n in n_grid:
        sat_ct = 0; nf = nl = 0; wf = wl = 0; lr = 0; nn = 0
        for s in seeds:
            rng = np.random.default_rng(SEED + s + n * 100)
            adj = random_graph(n, avg_deg, rng)
            avg_deg_seen = max(avg_deg_seen, max((len(a) for a in adj), default=0))
            satf, sf = search(adj, K, "full")
            satl, sl = search(adj, K, "lwf")
            if satf != satl or sf["budget_hit"] or sl["budget_hit"]:
                all_ok_p1 = False
            if sf["nodes"] != sl["nodes"]:
                all_ok_p3 = False
            sat_ct += satf
            nf += sf["nodes"]; nl += sl["nodes"]
            wf = max(wf, sf["peak_workspace"]); wl = max(wl, sl["peak_workspace"])
            lr += sl["ledger_reads"]; nn += sl["nodes"]
        ns = len(list(seeds))
        peak_ws_by_n[n] = wl
        print(f"{n:>4} | {100*sat_ct/ns:>4.0f}% | {nf//ns:>11} {nl//ns:>10} | "
              f"{wf:>11} {wl:>10} | {lr/max(nn,1):>13.2f}")

    # verdict
    ns_list = list(n_grid); ws_list = [peak_ws_by_n[n] for n in ns_list]
    corr = float(np.corrcoef(ns_list, ws_list)[0, 1]) if len(set(ws_list)) > 1 else 0.0
    max_ws = peak_ws_by_n[n_grid[-1]]
    print("\n--- PRE-REGISTERED VERDICT (fixed before run) ---")
    p1 = all_ok_p1
    p2 = (abs(corr) < 0.3) and (max_ws <= 3 * avg_deg)
    p3 = all_ok_p3
    print(f"  P1 completeness preserved (lwf==full SAT verdicts): {p1}")
    print(f"  P2 Workspace bounded & flat in n: corr(peakWS,n)={corr:.2f} (<0.3), "
          f"peakWS@maxN={max_ws} <= 3*avg_deg={3*avg_deg:.0f} -> {p2}  "
          f"[full executive @maxN grows to ~{n_grid[-1]}]")
    print(f"  P3 search identical (same nodes): {p3}")
    ok = p1 and p2 and p3
    print(f"  M21 VERDICT: {'PASS' if ok else 'FAIL (honest negative)'}")
    print("  [Complete search with executive state bounded by the FRONTIER, independent of problem")
    print("   size -- the assignment/trail/nogoods offloaded to the Ledger. Cost = Ledger reads/node.]")
    return ok


if __name__ == "__main__":
    run()   # pre-registered config (avg_deg=4.5, near the 3-colouring threshold -- HARD)
    print("\n" + "=" * 78)
    print("FOLLOW-UP (post-hoc; pre-registered verdict above unchanged): easier, COMPLETABLE")
    print("instances (avg_deg=3.0, below the hard threshold) to isolate the bounded-MEMORY claim")
    print("from the exponential-TIME wall the pre-registered hard config hit.")
    print("=" * 78)
    run(n_grid=(20, 40, 80, 160), avg_deg=3.0, seeds=range(8))
