# Cross-analysis: EP-GRM (sibling project) vs LWF

**Source:** `C:\WPAI\AI-Research\Grok_Playground` — a parallel autonomous-research project by a
different agent (Grok), given the *same* brief as LWF (novel architecture vs the von Neumann
bottleneck; bounded executive state; minimal data movement; explicit verification; persistent
cognition; 1660 Ti target; start 2026-07-01). Inspected read-only 2026-07-01. Nothing modified.

EP-GRM = **Explicit Persistent Graph Rewrite Machine** — the *symbolic / graph-rewriting*
attack on the problem LWF attacks *neurally / associatively*. Two agents, one brief, two
architectures, convergent conclusions. That convergence is the headline finding.

## Architecture correspondence
| EP-GRM (symbolic) | LWF (neural) | Note |
|-------------------|--------------|------|
| bounded **focus** (executive state) | **Workspace** (bounded fast-weight state) | same role |
| persistent **versioned graph** (single source of truth) | **Ledger** (content-addressable store) | EP-GRM unifies knowledge+working state in one graph; LWF splits them |
| focus holds only node **IDs + metadata** | Workspace holds cursor/control state | both keep pointers, not payloads (cf. M5) |
| **local rewrites only**, movement charged on load | minimal-data-movement thesis | LWF M4/M9 |
| **justifications + nogoods** (truth maintenance), versioned rollback | audit log / bit-exact replay (M3) | EP-GRM's verification is richer (learned constraints) |
| deterministic transitions | deterministic step semantics (M3) | agree |

## Convergent findings (two substrates, same result — the important part)
- **Bounded executive state holds as the problem grows.** EP-GRM: focus stays ≤ small max
  across 96+ runs up to 2000 nodes. LWF: M1 rank ceiling, M2.
- **Working fraction SHRINKS with scale.** EP-GRM information-locality 0.27 → **0.054** at
  2000 nodes. LWF: M2 (recall at fixed hot-state), M5 (bounded per-step at depth).
- **A small bound suffices — knee.** EP-GRM EXP-017: no benefit past focus≈4. LWF M1: past
  rank-d capacity, extra state doesn't help.
- **Minimal data movement via locality.** EP-GRM ~1 node touched/step. LWF M4/M9 movement thesis.
- **A real saturation/ceiling exists.** EP-GRM: focus saturates at max under heavy load. LWF:
  M1 rank ceiling; M8 H=3 composition ceiling.

## The highest-value import: Active Dependency Frontier (F)
EP-GRM's sharpest result: the required executive state / locality tracks the **dependency
frontier width F** (active 2-hop constraining predecessors), **NOT total size N**. Evidence:
- EXP-019: fixed N, F 14→212, locality 0.2→0.3 (higher F → higher L).
- EXP-018: dense graphs (wide frontier) locality 0.22 vs linear 0.029 (**8× worse**).
- Knee (EXP-017): L stable across max_focus once ≥ F.

Why this matters for LWF:
1. It is the answer to LWF's biggest open question — *what determines required Workspace
   capacity, and where does bounded state break?* Answer: **frontier width F relative to the
   rank-d ceiling (M1).** Break when F > d.
2. It gives **independent empirical support for the LWF §7 kill-shot**: "irreducibly global
   reasoning" = wide dependency frontier = where bounded state degrades. EP-GRM's dense-graph
   locality elevation IS that boundary, measured in a different substrate. LWF's M8 H=3 ceiling
   is the same phenomenon (the H-hop task's frontier grows with H).
3. It reframes the LWF cost law: **state ~ O(frontier width F); steps ~ O(reasoning depth).**
   M5 showed depth is free in state (frontier 1, any depth). M10 (new) tests the width axis.

## EP-GRM's honest negatives that keep LWF honest
- **EXP-022/023: the model `L ≈ c·F/N` OVERPREDICTS.** L stayed ~flat despite F varying
  190→500 at fixed N — frontier alone doesn't determine locality; scheduler/rule-firing matters.
  Lesson for LWF: do **not** build a naive "Workspace capacity = working-set size" law; the
  admission/eviction controller (which F you actually load) is a first-class factor.
- **Their movement claim is self-labeled `[BASELINE DEPENDENT]`** — "with better baselines the
  advantage disappears" (their baselines reload everything). LWF's M9 (real CIM/CAM per-op
  silicon energy) is the more grounded version of the same claim.

## Where each project leads
- **LWF ahead of EP-GRM:** real energy/silicon cost model (M4/M9); learned/neural instantiation
  (M6–M8); the Ledger = modern-Hopfield = attention unification. EP-GRM is all symbolic sim.
- **EP-GRM ahead of LWF:** the frontier variable; truth-maintenance verification (**nogoods** =
  learned constraints that prevent re-deriving conflicts — richer than replay-only M3);
  versioned rollback.

## Synthesis (the two are duals of one machine)
A graph-rewrite executor whose **focus/frontier** lives in a bounded associative **Workspace**,
whose **versioned graph** is the content-addressable **Ledger**, running on a **CIM/CAM Fabric**,
with **nogoods stored in the Ledger as first-class learned constraints** and **frontier width
setting the Workspace rank**. Both projects independently triangulated the same three invariants
(bounded state, locality, explicit verification); merging gives EP-GRM's symbolic verifiability
plus LWF's energy grounding and learned recall.

## Concrete imports actioned in LWF
- **M10 (`exp_m10_frontier.py`) — DONE, SUPPORTED.** Adopted frontier width F as an explicit
  controlled variable. Result: LWF-frontier recall = 1.000 for F≤d, flat across N=256→8192
  (across-N std 0.004), degrades past F≈d (0.75 at F=4d); cram tracks N (→0.02 at N=8192);
  Ledger exact for all (F,N) = the overflow escape hatch. The EP-GRM frontier law reproduced in
  the neural-associative substrate. Completes the LWF cost law: **O(depth) steps × O(frontier
  width) hot state** (depth free in state per M5; width costs state per M10).
- **M11 (`exp_m11_nogood.py`) — DONE, SUPPORTED.** Added a first-class `nogood` record type to
  the Ledger (EP-GRM's truth-maintenance verification). Conflict-directed backtracking on hard
  3-colouring: **8.9× fewer search nodes** (median 5.4×, up to 13.4×), **sound** (12/12 identical
  SAT verdicts). Verification extended from replay-only (M3) to active learned-constraint pruning.
- **M12 (`exp_m12_controller.py`) — DONE, SUPPORTED.** Built the frontier-aware admission/eviction
  controller EXP-022 demanded. Two regimes: at C≥F any decent policy ≈ oracle (confirms M10); at
  C<F the LWF-native **belief-decay** (frequency + decay gate) beats un-aged LFU by 0.42 and beats
  random by exploiting frequency the set-oracle ignores. Recommendation: belief-decay or LRU. The
  naive M10 frontier→capacity law is confirmed for C≥F and now paired with a real controller for
  C<F — closing the EXP-022 gap.

All three sibling imports (frontier M10, nogoods M11, controller M12) are now actioned and
supported. Remaining cross-pollination candidates: EP-GRM could import LWF's energy/silicon cost
model (M9) and learned instantiation (M6–M8); LWF could import EP-GRM's versioned-graph rollback.
