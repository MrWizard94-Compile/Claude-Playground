# LWF changelog

Reverse-chronological. Dates are session dates. Each entry: what changed + why.

## 2026-07-01 — Search+verification angle (M21): bounded-executive search
- **Added** `exp_m21_bounded_search.py` (M21): the "attack from another angle" move — complete
  constraint search (3-colouring, backtracking + nogoods) with the LWF split applied to SEARCH:
  Workspace = active frontier, Ledger = assignment/trail/nogoods. → **pre-reg FAIL** (hard near-
  threshold instances hit the node budget; a P3 bug — nogood check was lwf-only — fixed & disclosed).
  Bugfixed completable follow-up: **P3 holds (identical search); executive memory demonstrably bounded
  (peakWS(lwf)≈7 flat vs peakWS(full)=n)** — but pre-reg still FAIL (P2 corr threshold too strict at
  4 points; P1 budget hit). FINDING: executive MEMORY is boundable for search, but search TIME is
  exponential & ORTHOGONAL to the memory architecture. SESSION SYNTHESIS: LWF is a MEMORY +
  DATA-MOVEMENT architecture (validated: M2/M4/M9/M14/M21), NOT a solution to computational
  bottlenecks (M19/M20/M21). Updated OVERVIEW §0 with the session-level thesis.

## 2026-07-01 — The crux, done right (M20): rank-bottlenecked Workspace
- **Added** `exp_m20_crux_bottlenecked.py` (M20): after M19 showed the crux couldn't be bit-bottlenecked,
  M20 uses the CORRECT capacity model (RANK, per M1): sweep the Workspace rank dh below N. → **pre-reg
  INCONCLUSIVE**: a **trained rank-2 Workspace sorts 24 items at 0.95** — training overcomes the rank
  bottleneck. THE CENTRAL FINDING: the M1 rank ceiling governs UNTRAINED arbitrary-association storage,
  NOT trained structured computation; bounded recurrent states compute global functions cheaply. ⇒ the
  crux is largely illusory for COMPUTATION at reachable scale; LWF's Workspace+Ledger split is justified
  by STORAGE (I1, confirmed M2/M14), not computational limits. Thesis-clarifying and partly deflating.
  Updated OVERVIEW.md with a §0 "Corrections & honest state"; updated GAPS G2.1.

## 2026-07-01 — "Do all three" batch: write policy (M17), full-stack-vs-baseline (M18), crux (M19)
All three used PRE-REGISTERED verdicts (G0.1 discipline); M17 & M18 both returned FAIL, reported as-is.
- **M17** (`exp_m17_write_policy.py`, G1.4): verified-only Ledger writes. Import BENEFICIAL (+0.33
  downstream accuracy at 40% corruption, rejects 90% of corrupt writes at ~5% valid-write cost); pre-reg
  FAIL only on P3 (purity ≥0.90 at 60% corruption) — a ceiling bounded by verifier_quality×corruption.
- **M18** (`exp_m18_fullstack_vs_baseline.py`, G0.2): full fast-weight Workspace + Ledger vs a WORKING
  transformer baseline on MQAR. Pre-reg FAIL (config invalid: bounded not stressed at D=48). KEY
  correction: found fast-weight Workspace ≡ linear attention (Schlag 2021) — falsifies GAPS G1.2
  ("M7 used a stand-in"); M7 already tested the real stack in a valid config. Fixed 2 bugs (missing
  shifted-value + normalizer) in the process.
- **M19** (`exp_m19_hard_reasoning.py`, G2.1): the crux — irreducibly-global reasoning via in-context
  SORTING (order statistics). Two-sided pre-registered verdict (LWF can FAIL, which would be the most
  informative outcome). Transformer baseline confirmed viable (sorts to 0.99). → **INCONCLUSIVE**:
  the bounded Workspace SORTED N=16 at 0.96 (didn't fail). Deep finding: the crux is structurally
  UNTESTABLE at toy scale — a bounded state (~6000 floats) dwarfs a toy task's working info (~64 bits),
  so it never enters the failure regime. Reframes the whole open question (needs real-scale tasks).

## 2026-07-01 — First end-to-end integrated LWF (M15)
- **Added** `exp_m15_integrated.py` (M15): the first assembled, learned LWF (trained fast-weight
  Workspace + keyed Ledger + LEARNED-query read + compositional head) on mutable variable
  tracking + comparison, with a **pre-registered verdict**. → **FAIL/PARTIAL (honest)**: integration
  works (learned LWF 0.70 COMPARE, oracle 1.00 — plumbing correct; G1.1/G1.2 closed), but learned
  query formation lags oracle (0.70 vs 1.00 → SECONDARY fails; G1.3 confirmed hard) AND the "strong"
  transformer baseline failed to train (0.44 ≈ chance → PRIMARY vacuous; G0.2 re-opened, G0.3
  re-confirmed). Pre-registered FAIL reported verbatim. See GAPS.md status update.

## 2026-07-01 — Sibling FLEET survey + lexicographic-energy import (M16)
- **Inspected** (read-only) the whole `C:\WPAI` constellation. Beyond EP-GRM: **TSAM** (Tensor-
  State Associative Manifold, RVP-validated deterministic code synthesis), **VeriForge/RHDF** (running
  Rust correct-by-construction MVP), **Topological Hydro-Computational Engine** (ungrounded GPE
  physics→AST), and **JanusPrime** (production multi-AI orchestrator). Wrote
  `references/SIBLING_FLEET.md` (classification + rigor gradient + transferable ideas).
- **CORRECTION:** the sibling "convergence" is a SHARED PRIOR (same operator/doctrine/brief/hardware),
  not independent corroboration. Retracted the "cross-substrate corroboration" framing; upgraded
  `GAPS.md` G3.2 from "weak" to "actively misleading."
- **Added** `exp_m16_energy_termination.py` (M16): imports TSAM's lexicographic-energy termination
  guarantee, with a **pre-registered verdict** (practicing the G0.1 discipline). Outcome: the
  termination guarantee is VALIDATED (P1-P3 = 300/300); the pre-registered verdict FAILED (298/300)
  because greedy descent is incomplete (local minima) — reported as FAIL, **bar not moved**. Found &
  fixed a harness solvability-mislabeling bug mid-way (disclosed). A compound rewrite recovers
  completeness (82/82) while preserving the guarantee.

## 2026-07-01 — Knowledge scales in the Ledger, learned + at scale (M14)
- **Added** `exp_m14_scaling_ledger.py` (M14): tests M9's nonparametric-knowledge bet LEARNED,
  using the GPU (train small models) + 48 GB host RAM (Ledger up to 5M facts / 2.56 GB). → **SUPPORTED**:
  a fixed 108K-param MLP trains fine at 1K facts (0.864) but collapses to chance (0.009) by 100K;
  a small model + host-RAM Ledger holds 0.951 at 100K (95× parametric) and degrades gracefully to
  0.786 at 5M, at constant per-query model cost. Turns M9's assumption into empirical evidence:
  fixed weights can't hold growing knowledge; the external store can. (Calibration first ruled out
  a bigger-model M8 run — on a 1660 Ti a bigger model is slower AND needs more steps; dead end.)
- **Note:** brute-force retrieval is O(N) here (0.02→0.56 ms/q to 5M); ANN index would make it O(log N).

## 2026-07-01 — Controller on a REAL frontier (M13)
- **Added** `exp_m13_live_frontier.py` (M13): wires the controller onto an actual computation DAG
  (frontier = live values = register pressure), with Belady MIN as the true optimum. → **SUPPORTED**:
  at frontier-sized cache (C≥MaxLive) a **liveness-aware controller = Belady optimum** (~0 misses)
  vs recency/frequency 51–200 (which waste slots on dead values). KEY FINDING: real cognition needs
  **dependency-derived liveness (reference-counting)** — the same justification bookkeeping behind
  M11's nogoods — not just recency/frequency; provision C ≥ frontier. Unifies controller (M12/M13)
  with the dependency/verification machinery (M11). Honest limit: below frontier no online policy
  matches Belady (fundamental online gap). Refines M12: belief-decay for drift; liveness for cognition.

## 2026-07-01 — Nogood import (M11) + controller quality (M12)
- **Extended** `ledger.py` with a first-class **nogood** record type (`write_nogood`,
  `nogood_violated`, literal-indexed, subset-minimal) — EP-GRM's truth-maintenance verification,
  now native to the Ledger alongside associative (key,value) knowledge.
- **Added** `exp_m11_nogood.py` (M11): conflict-directed backtracking 3-colouring. → **SUPPORTED**:
  **8.9× fewer** search nodes vs plain backtracking (median 5.4×, up to 13.4×), **sound** (12/12
  identical SAT verdicts). Verification goes from replay-only (M3) to active learned-constraint
  pruning.
- **Added** `exp_m12_controller.py` (M12): Workspace-as-cache admission/eviction quality, dropping
  M10's oracle frontier. → **SUPPORTED** (two-regime): C≥F any decent policy ≈ oracle (confirms
  M10); C<F belief-decay beats un-aged LFU by 0.42 (decay cures staleness) and beats random by
  exploiting frequency. Recommendation: belief-decay or LRU. Addresses EP-GRM EXP-022 directly.
  Found: the set-oracle isn't optimal at C<F (ignores intra-frontier frequency) — reported honestly.

## 2026-07-01 — Cross-pollination from sibling EP-GRM project (M10)
- **Inspected** (read-only) `C:\WPAI\AI-Research\Grok_Playground` — a parallel autonomous-research
  project by another agent (Grok), same brief, that converged on EP-GRM (Explicit Persistent
  Graph Rewrite Machine), the symbolic dual of LWF. Two agents, one brief, convergent invariants.
- **Added** `references/SIBLING_EPGRM.md`: full cross-analysis. Key import = the **Active
  Dependency Frontier**: required executive state tracks frontier width F, not total size N.
- **Added** `exp_m10_frontier.py` (M10): reproduces the frontier law in LWF's associative
  substrate. → VERDICT **SUPPORTED**: LWF-frontier recall is 1.000 for F≤d flat across N=256→8192
  (std 0.004), degrades past F≈d (0.75 at F=4d); cram tracks N (→0.02); Ledger exact everywhere.
  Completes the two-axis cost law: O(depth) steps (free in state, M5) × O(frontier) hot state.
  Independent cross-substrate corroboration of EP-GRM's central finding.

## 2026-07-01 — Stage 2: Fabric cost model on real silicon (M9)
- **Added** `exp_m9_fabric.py` (M9): replaces M4's generic constants with cited per-op silicon
  energy (CAM 0.5 fJ/bit, CIM cell 0.02 pJ, ADC 3 pJ **explicit**, DRAM 10 pJ/B, digital MAC
  0.5 pJ). Runs three designs' actual op-traces — transformer/vonNeumann, LWF/vonNeumann
  (algorithm only), LWF/Fabric (algorithm+substrate) — to separate the two claims. → VERDICT
  **SUPPORTED (model)**: transformer decode is 98% data movement; LWF's win is eliminating it.
  Result made honest via explicit P_core knob: **~33× energy/token at RETRO-like P/25**, **1× if
  knowledge isn't offloaded**. Survives ±10× on every constant. Fixed a self-contradictory
  hardcoded "ADC dominates" line → now computes the true dominant term (cells at d_ws=2048).
- **Added** verified CAM/TCAM, ADC, and CIM-MAC energy numbers to `references/BIBLIOGRAPHY.md`.

## 2026-07-01 — GPU enabled + in-context multi-hop crux (M8)
- **Environment:** swapped global torch 2.12.1+cpu → **2.12.1+cu126** (in-place, --no-deps).
  GPU verified: GTX 1660 Ti, compute 7.5, 6.44 GB, ~3.15 TFLOP/s FP32. `exp_m7`/`exp_m8`
  auto-detect CUDA and switch to larger presets.
- **Added** `exp_m8_multihop_incontext.py` (M8): the GLOBAL-reasoning crux, not flat recall.
  A random functional graph is listed in-context as shuffled edge pairs; the model must walk
  H hops from a start node — genuine composition over in-context facts, graph re-randomised per
  example (only the algorithm generalises). Compares linear / attn / hybrid_topk at equal depth.
  De-risk (900 steps, H=2): hybrid_topk 0.256 [40 reads] ≈ attn 0.275 [205 reads] ≫ linear 0.156
  — bounded-cost retrieval composes 2-hop about as well as full attention at ~5× fewer reads.
  Full H=1..3 sweep (2 seeds) running → `results_m8_full.txt`.

## 2026-07-01 — Stage 0.5: bounded-cost crux (M7)
- **Added** `exp_m7_scaleup.py` (M7): hardware-agnostic (auto CUDA/CPU) upgrade of the crux
  probe. Key change: the Ledger read is now **top-k sparse** (k reads/step, independent of
  context) instead of dense softmax — answers the fair criticism that M6's dense read "was
  just attention." Adds multi-seed error bars, multi-layer support, push-to-breakage D sweep,
  and explicit reads/step cost accounting (attn O(T) vs hybrid-topk = k). Auto-scales to a
  larger preset on GPU. → VERDICT **SUPPORTED**: hybrid-topk holds 1.000 (= full attention)
  reading only 8 entries/step across D=8→64, while bounded linear degrades 0.974→0.541 and
  attention's reads grow 24→192. Attention-level accuracy at bounded per-step read cost;
  confirms the M6 recovery is content-addressing, not dense attention. See `results_m7_full.txt`.

## 2026-07-01 — Stage 1 + crux probes + reference dossier
- **Added** `verify.py` (M3): checkpointable, bit-exact-replayable cognitive step with a
  content-hashed audit trace; tamper-evident; log growth O(steps·k).
- **Added** `exp_m3_replay.py` (M3): builds a real Workspace+Ledger cognitive loop, proves
  bit-exact replay, mid-run checkpoint/resume, tamper-evidence, and linear log growth.
  → VERDICT **SUPPORTED** (253 bytes/step, flat).
- **Added** `exp_m5_multihop.py` (M5): compositional multi-hop reasoning over a Ledger graph;
  isolates *bounded per-step state* from *intrinsic reasoning depth*. Iterative pointer-chase
  holds 1.000 accuracy at 8 hops on 512-byte state; single fixed-state operator collapses to
  chance. → VERDICT **SUPPORTED**. Resolves the crux's "outcome 2": depth is a task property,
  not an architecture failure.
- **Added** `exp_m6_crux_learned.py` (M6): PyTorch/CPU learned MQAR probe — linear vs delta
  (bounded) vs attn vs hybrid(Workspace+Ledger). Directly tests whether external retrieval
  recovers recall that bounded state loses. Fixed a real MQAR bug (values sit one position
  after keys → single-layer attention needs the shifted-value construction).
- **Added** `Ledger.read_hopfield` — one-step modern-Hopfield/softmax read; unifies the Ledger
  with attention per Ramsauer et al. 2020 (attention *is* Hopfield retrieval).
- **Added** `references/BIBLIOGRAPHY.md` — annotated, source-verified (Horowitz energy numbers,
  DeltaNet, Zoology/MQAR, modern Hopfield capacity, RRAM CIM TOPS/W, Landauer, RETRO/kNN-LM).
- **Added** `LAB_NOTEBOOK.md`, `CHANGELOG.md`; wired M3/M5 into `run_all.py`.

## 2026-07-01 — Stage 0 (initial)
- **Added** `workspace.py` (fast-weight bounded associative state), `ledger.py`
  (content-addressable exact store), experiments M1/M2/M4, `run_all.py`, `README.md`,
  `references/INVARIANTS.md`.
- M1 rank-capacity ceiling **SUPPORTED**; M2 Ledger recovery **SUPPORTED** (after calibrating
  cue-noise to the recoverable regime; σ≥0.25 is SNR<1 and capped by information, not design);
  M4 energy scaling **SUPPORTED** (transformer grows with context, LWF flat; robust to ±10×).
