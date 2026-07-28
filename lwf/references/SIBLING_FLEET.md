# The sibling fleet — cross-analysis and honest meta-critique

Inspected read-only 2026-07-01 (`C:\WPAI\*`). Nothing modified anywhere. This supersedes
the pairwise framing in `SIBLING_EPGRM.md`: LWF is one of a *fleet* of same-brief agents, not
one of two independent efforts. That fact changes the epistemics (see §3).

## 1. The research fleet (novel-architecture brief) — rigor gradient
| Project | Substrate | Realization | Rigor |
|---------|-----------|-------------|-------|
| **LWF** (Claude / mine) | bounded-state neural-associative (Workspace/Ledger/Fabric) | Python + torch sim; toy/synthetic | falsifiable, measured; GAPS.md self-audit |
| **EP-GRM** (Grok) | symbolic graph-rewrite, bounded focus | Python/NetworkX sim only (explicit `SIMULATION_LIMITS` doc) | honestly-scoped, but "ALL TIERS COMPLETE" status-theater |
| **TSAM** | deterministic constraint-driven code synthesis; tensor-state manifold | Python; **RVP-validated Stage 0** (Fabric→NeoForge) | **most disciplined** (refuses to invent constraints to hit numbers) |
| **VeriForge / RHDF** (Grok) | Holographic TN + Spiking NN + Reversible Composer + Formal Verifier | **running Rust binary** (<10ms, correct-by-construction) | grounded (emits only from verified atoms; nothing if verify fails) |
| **Topological Hydro-Computational Engine** | superfluid GPE on hex lattice → braid topology → AST | spec only | **ungrounded** — no validation, no falsification plan |

Non-research (production/orchestration, same doctrine):
- **JanusPrime** — multi-AI orchestrator (Claude plans, Grok executes), deterministic validation
  kernel, Smart-Library semantic memory, **verified-only memory seeding**. Real, tested (131+ tests).
- **AutomationLab** — Obsidian design vault for a deterministic mod-porting pipeline (components
  named *Convergence Controller*, *Coverage Ledger*, *Integration Gate*). Design stage.

## 2. Shared invariants (the "convergence")
All five research siblings independently land on: **bounded active window/workspace/focus** over a
**persistent read-only store**; **LRU/liveness eviction** with **locality metrics**; **bounded-hop /
frontier-limited resolution** (work ∝ neighbourhood, not size); **deterministic hashable/replayable
traces**; **energy/constraint-driven termination**; **verification co-located with rewrite**;
**topology-family benchmarks** (linear/tree/hub/sparse/dense). TSAM, EP-GRM, and AutomationLab even
use near-identical vocabulary (focus/window, ledger, frontier, eviction, energy).

## 3. THE honest meta-point — convergence here is a SHARED PRIOR, not evidence
Earlier (`SIBLING_EPGRM.md`) I called EP-GRM's agreement with LWF "strong cross-substrate
corroboration." **The fleet shows that was wrong.** Every agent shares:
- the **same operator** (one person),
- the **same `CLAUDE.md` doctrine** — the identical 8 invariants (bounded state, verification-
  before-mutation, one-shot completeness, zero-warnings, research-over-guesswork, …) appear verbatim
  in JanusPrime §6 and in LWF's own doctrine,
- the **same brief** (explicitly prescribing "bounded state / minimal data movement / explicit
  verification"),
- the **same hardware** (1660 Ti) and **same domain** (Minecraft/NeoForge porting).

When N agents all "discover" the invariants the prompt told them to value, that is **the shared
prompt expressed in N notations, not independent convergence.** Independence is the whole basis of
corroboration, and it is absent. **Finding more siblings does not strengthen the convergence claim —
it falsifies it as evidence.** This upgrades LWF's own `GAPS.md` G3.2 from "weak evidence" to
"actively misleading if cited as corroboration." The invariants may still be *correct* — but their
correctness must be earned by measurement in each project, not inferred from agreement across the fleet.

## 4. Genuinely transferable ideas (ranked; these stand on their own merit, not on convergence)
1. **Lexicographic energy termination (TSAM).** Energy = ordered tuple (H≻S≻C≻Q); every committed
   rewrite strictly decreases it ⇒ bounded-iteration fixed point, no oscillation. **Validated 150/150
   monotone.** LWF's cognitive loop (M3/M5) has **no termination guarantee** — this supplies one.
   → actioned as **M16** (`exp_m16_energy_termination.py`), with a pre-registered verdict.
   OUTCOME (honest): the termination guarantee itself is VALIDATED (P1/P2/P3 = 300/300 strict-
   descent, bounded, no oscillation). The pre-registered verdict nonetheless FAILED (298/300) —
   greedy lexicographic descent is INCOMPLETE (local minima); termination ≠ completeness. Reported
   as FAIL (bar not moved); a compound rewrite recovers completeness (82/82) while preserving the
   guarantee. See LAB_NOTEBOOK.md M16 for the full chain (incl. a harness bug I found and fixed).
2. **Correct-by-construction from verified atoms (VeriForge).** Never emit unverified output; compose
   only from a closed set of atoms carrying proven invariants (SAFETY/PROGRESS/COVERAGE); if
   verification fails, emit *nothing*. This is the strongest form of LWF's nogood idea (M11): not just
   *prune* contradictions but *only ever produce* verified structures. The route to a verifiable
   software-synthesis instantiation of Goal #7. Plus reversible snapshot/revert = concrete rollback.
3. **Verified-only write policy (JanusPrime).** Only validated successes seed the persistent store;
   failed validations never write. Directly addresses LWF **G1.4** (untested write policy): the Ledger
   should admit only verification-passing writes. → candidate M17.
4. **Simulation-fidelity ladder (EP-GRM).** Tier 1 proxy → memory-model → reference impl → hardware,
   with claims graduating only when reproduced at higher fidelity. A concrete plan for moving LWF
   past "everything is toy/synthetic" and for grounding M9 (cost model, not silicon).
5. **Topological-invariant determinism (THCE, the one real nugget).** Encode logic in topological
   invariants (robust to microscopic perturbation) rather than explicit state → determinism without
   rounding sensitivity. Speculative, but genuinely relevant to LWF **G3.4** (analog-CIM precision):
   topologically-encoded state could tolerate analog noise. Flag, don't import yet.

## 5. Which siblings to trust (a rubric, since the fleet spans the whole range)
- **Trust the discipline, not the vision.** TSAM (refuses to fake constraint counts), EP-GRM's
  simulation-limits doc, VeriForge's running binary + honest limitations, and LWF's own GAPS.md are
  trustworthy *because they self-limit*. THCE's ungrounded elaboration and EP-GRM's "ENVELOPE PUSHED /
  ALL COMPLETE" status-stacking are the tells of completeness-theater — the same anti-pattern LWF's
  G0.1 flags in itself.
- **Grounded > elaborate.** VeriForge (9 real rules, running, correct-by-construction) is worth more
  than THCE (elaborate GPE physics, nothing runs), despite THCE looking more impressive on paper.

## 6. Net effect on LWF
- **Retract** the "cross-substrate corroboration" framing wherever it appears; replace with "shared-
  prior, must be earned per-project." (GAPS.md G3.2 updated.)
- **Keep** M10/M11/M12/M13 (they were validated in LWF on their own merit, not on convergence).
- **Add** M16 (lexicographic termination) — the one load-bearing thing LWF genuinely lacked.
- **Queue** verified-only write policy (M17) and the fidelity ladder as the honest path past toy scale.
