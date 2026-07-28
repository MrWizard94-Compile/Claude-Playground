# LWF — Ledger / Workspace / Fabric

A Stage-0 investigation of a bounded-state cognitive architecture that attacks the
von Neumann bottleneck for intelligent computation. Full design rationale, invariants,
and the mechanism ledger live in [`references/INVARIANTS.md`](references/INVARIANTS.md).

**Thesis in one line:** state should be a bounded *sufficient statistic* of history,
not a growing *log* of it — so split cognition into a fixed-size Workspace (fast, lossy),
an unbounded content-addressable Ledger (exact, out of the hot path), and a compute-in-
memory Fabric — and the algorithmic fix for unbounded state turns out to be the same
shape as the hardware fix for data movement.

## What Stage 0 tests (and what it deliberately does not)
Stage 0 falsifies the **information-routing** claims — the parts that need no training
and run on a laptop. It does **not** test the learned-model capacity question (the §7
crux), which needs a GPU and is Stage 0.5+. Quarantining the cheap-to-falsify claims
from the expensive one is the methodology, not an oversight.

## Run
```
python run_all.py            # all three experiments, prints falsifiable verdicts
python run_all.py --plots    # also writes figures/*.png
python exp_m1_capacity.py    # individually
python exp_m2_ledger_recovery.py
python exp_m4_cost_model.py
```
Deterministic (fixed seeds). Requires only `numpy` (+ `matplotlib` for `--plots`).

## Results — fourteen mechanisms (full numbers + honest caveats in `LAB_NOTEBOOK.md`)
- **M1 — Workspace rank ceiling.** A d×d fast-weight state recalls exactly up to N=d
  orthonormal associations, then degrades predictably (rank(M) ≤ d). *Finding:* the DeltaNet
  update only beats plain Hebbian when keys are near-orthogonal; on correlated keys its
  unit-rate correction overshoots and Hebbian is more graceful — the delta advantage needs
  *learned* gating (seen in M6), not raw single-pass.
- **M2 — Ledger recovers recall at fixed hot-state.** Identical 32 KB hot-state, corrupted
  cue (σ=0.15): route recent to Workspace, evict rest to Ledger → **0.97 recall @ N=2048 vs
  0.02** for cramming. At σ≥0.25 (SNR<1) recall is information-capped, not architecture-capped;
  LWF still ~90×. Honest cost: Ledger comparisons/read (O(N) here, O(log N) with ANN).
- **M3 — verifiable replay.** A real Workspace+Ledger loop is **bit-exact replayable**,
  checkpoint/resumable, and **tamper-evident** (1e-6 input change flips the run digest); audit
  log is **flat at 253 bytes/step** (O(steps), not O(n²)). Goal #3, concretely.
- **M4 — energy scaling.** 7B transformer grows **×5.9** over context 128→131072 (KV-cache
  streaming); LWF stays **×1.00**. Slope conclusion survives ±10× on every energy constant.
  E_DRAM=20 pJ/B is conservative vs verified Horowitz HBM (~200 pJ/32b) → verdict understated.
- **M5 — multi-hop composition.** Iterative pointer-chase over the Ledger holds **1.000 at 8
  hops on a constant 512-byte state**; the single fixed-state operator collapses to chance.
  Separates *bounded per-step state* (LWF keeps) from *reasoning depth* (task property, O(depth)
  for everyone). The honest resolution of the crux's "cost-reconverges" worry.
- **M6 — the learned §7 crux (MQAR, PyTorch/CPU), directional.** Across a state-pressure sweep
  (D=8→40) the bounded **linear state degrades monotonically 0.969→0.656**, the **delta rule
  resists (~0.98–0.99)** (confirming the DeltaNet frontier + the M1 nuance that delta needs
  *learned* gating), and **hybrid (Workspace+Ledger) pins 1.000 across the sweep, matching full
  attention** — retrieval recovers what bounded state loses, learned from scratch. Toy scale,
  single layer, one seed; the softmax Ledger read is dense here (deployment: ANN top-k).
  Directional, not a frontier claim.
- **M7 — bounded-cost crux (top-k Ledger), directional.** The honest version: the Ledger read
  attends to only **k=8 entries/step** (constant, context-independent). Across D=8→64 the
  **hybrid holds 1.000 (= full attention) reading 8/step**, while attention's reads grow 24→192
  and the bounded state degrades 0.974→0.541. Proves the recovery is content-addressing, not
  dense attention in disguise: **attention-level accuracy at bounded per-step read cost.**
  Hardware-agnostic (auto CUDA/CPU, auto-scales the preset).
- **M8 — in-context multi-hop, the global-reasoning crux (GPU, 1660 Ti). SUPPORTED at H=2;
  H=3 shared ceiling.** Beyond flat recall: the model must walk H hops through a re-randomized
  in-context graph (genuine composition). At **H=2, hybrid_topk (40 reads) = attn (205 reads),
  both ~2× the bounded state** — bounded-cost retrieval composes, not just looks up. At H=3 all
  models collapse *including attention* (0.161 < hybrid 0.180), so H=3 is a depth/scale ceiling
  of the toy, **not** an LWF-specific failure. The honest edge of what's shown; deeper
  composition at scale is the open frontier.
- **M9 — Fabric cost model on real silicon numbers (Stage 2).** Runs each design's actual
  op-trace through cited per-op energy (CAM 0.5 fJ/bit, CIM cell 0.02 pJ, **ADC 3 pJ modelled
  explicitly**, DRAM 10 pJ/B). The transformer's decode energy is **98% data movement** (7B
  weight + KV streaming); LWF's win is eliminating movement, not cheaper MACs. Result hinges on
  one bet, made explicit: at **P_core=P/25 (RETRO-grounded) → ~33×** energy/token reduction;
  at P_core=P (no knowledge offload) → **1×, no win**. Survives ±10× on every constant. Analytic
  model, not measured silicon.
- **M10 — frontier-width scaling (imported from the sibling EP-GRM project).** Required Workspace
  capacity tracks **dependency frontier width F** (degrades past F≈d), **flat in total N**; cram
  tracks N; the Ledger absorbs overflow at O(F) exact reads. Reproduces EP-GRM's central law in a
  different substrate. Completes the LWF cost law: **O(depth) steps × O(frontier width) hot state**
  (depth free in state per M5; width costs state per M10). See `references/SIBLING_EPGRM.md`.
- **M11 — nogoods: learned-constraint verification (imported from EP-GRM).** The Ledger gains a
  first-class nogood record type; conflict-directed backtracking on hard 3-colouring uses **8.9×
  fewer** search nodes than plain backtracking (median 5.4×, up to 13.4×) with **identical, sound
  verdicts** on all 12 instances. Extends verification from replay (M3) to active pruning.
- **M12 — controller quality: Workspace↔Ledger admission/eviction.** Drops M10's oracle: the
  Workspace is a cache and the controller must discover the drifting frontier. At C≥F any decent
  policy ≈ oracle (confirms M10); at C<F **belief-decay beats un-aged LFU by 0.42** (decay cures
  staleness) and beats random by exploiting frequency. Recommendation: **belief-decay or LRU**.
- **M13 — controller on a REAL frontier (computation DAG live values).** Replaces synthetic drift
  with actual computation; true optimum = **Belady**. At frontier-sized cache (C≥MaxLive) a
  **liveness-aware controller = Belady optimum** (~0 misses) while recency/frequency take 51–200
  (they waste slots on dead values). Finding: real cognition needs **dependency-derived liveness**
  (reference-counting — the same bookkeeping behind M11's nogoods), not just recency/frequency.
  Provision C ≥ frontier. Unifies the controller with the dependency/verification machinery.
- **M14 — knowledge scales in the Ledger, not the model (learned; GPU + 48 GB host RAM).** Tests
  M9's nonparametric-knowledge bet empirically. A fixed 108K-param MLP trains fine at 1K facts
  (0.864) but **collapses to chance (0.009) by 100K**; a small model + host-RAM Ledger holds
  **0.951 at 100K (95×) and degrades only gracefully to 0.786 at 5M facts (2.56 GB RAM)**, at
  constant per-query model cost. A fixed model can't hold growing knowledge; the external store can.

## Files
```
workspace.py                 Tier 1: gated fast-weight associative memory (bounded)
ledger.py                    Tier 2: content-addressable store (+ Hopfield read); auditable cost
verify.py                    M3: checkpoint/replay/audit-trace verifiability layer
exp_m1_capacity.py           M1: rank-capacity ceiling
exp_m2_ledger_recovery.py    M2: Ledger recovers recall at fixed hot-state
exp_m3_replay.py             M3: bit-exact replay + O(steps) audit log
exp_m4_cost_model.py         M4: data-movement/energy vs context length
exp_m5_multihop.py           M5: compositional multi-hop reasoning
exp_m6_crux_learned.py       M6: LEARNED MQAR crux probe (torch, CPU)
exp_m7_scaleup.py            M7: Stage 0.5 -- bounded-cost (top-k) crux, hardware-agnostic
exp_m8_multihop_incontext.py M8: Stage 0.5 -- in-context multi-hop (global-reasoning crux), GPU
exp_m9_fabric.py             M9: Stage 2 -- Fabric cost model on real CIM/CAM silicon numbers
exp_m10_frontier.py          M10: frontier-width scaling (capacity ~ F not N; EP-GRM import)
exp_m11_nogood.py            M11: nogoods -- learned-constraint verification (EP-GRM import)
exp_m12_controller.py        M12: Workspace<->Ledger admission/eviction controller quality
exp_m13_live_frontier.py     M13: controller on a real computation frontier (vs Belady optimum)
exp_m14_scaling_ledger.py    M14: knowledge scales in the Ledger, not the model (GPU + host RAM)
exp_m15_integrated.py        M15: first end-to-end integrated LWF, learned (pre-registered verdict)
exp_m16_energy_termination.py M16: lexicographic-energy termination guarantee (TSAM import, pre-registered)
exp_m17_write_policy.py      M17: verified-only Ledger write policy (JanusPrime import, pre-registered)
exp_m18_fullstack_vs_baseline.py M18: full fast-weight stack vs working transformer baseline (MQAR)
exp_m19_hard_reasoning.py    M19: irreducibly-global reasoning (in-context sorting) -- the crux test
exp_m20_crux_bottlenecked.py M20: the crux via a RANK-bottlenecked Workspace (the decisive reframing)
exp_m21_bounded_search.py    M21: complete search with bounded executive state (the search+verification angle)
exp_m22_fabric_sim.py        M22: Fabric simulator -- component-level energy+latency (supersedes M9)
exp_m23_measured_energy.py   M23: MEASURED decode energy vs context (NVML); run on a rented H100
run_all.py                   driver for M1–M5 (+ optional figures/)
references/SIBLING_EPGRM.md   cross-analysis vs the parallel EP-GRM project (Grok_Playground)
references/SIBLING_FLEET.md   whole-fleet survey + the honest "convergence = shared prior" correction
GAPS.md                      adversarial self-audit (integrity/architecture/crux gaps)
references/INVARIANTS.md     diagnosis, invariants, mechanism ledger
references/BIBLIOGRAPHY.md    annotated, source-verified citations + constants
LAB_NOTEBOOK.md              dated run log with all numbers
CHANGELOG.md                 what changed and why
```

## Next stages
- **Stage 0.5 (GPU):** scale M6 — multi-layer, deeper reasoning, error bars; does the
  retrieval recovery hold when reasoning is genuinely global, or is there a residual class
  it can't fix? This is still the program's kill-shot.
- **Stage 2:** M4 with per-op CIM/CAM energy from published silicon, on real op-traces.
- **Stage 3+:** long-horizon agentic software synthesis; then Fabric FPGA emulation.
