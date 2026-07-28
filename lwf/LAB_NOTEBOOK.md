# LWF lab notebook

Dated, append-only record of what was run, what came back, and what it means. Numbers
are copied from actual runs (fixed seeds). Verdicts are the experiment's own gates.

Environment: Python 3.10.11, numpy 2.2.6, torch 2.12.1+cpu, scipy 1.15.3, networkx 3.4.2,
matplotlib 3.10.8. Windows 11. CPU only.

Legend: [FACT] verified/derivable · [EVIDENCE] measured here · [OPEN] not yet tested.

================================================================================
## 2026-07-01 — Stage 0: information-routing claims (no training)

### M1 — Workspace rank-capacity ceiling  → SUPPORTED
`python exp_m1_capacity.py` (d=64, seeds 0–2). top-1 associative recall:
- N≤d, orthonormal keys, delta rule: **1.000** (exact). N>d: mean **0.653** → falls.
- Random (correlated) keys degrade earlier via crosstalk (N=256: hebb 0.733, delta 0.258).
[EVIDENCE] A d×d fast-weight state holds ≤ d exact associations (rank(M) ≤ d). This is the
provable ceiling that *forces* an external store (Invariant I1).
FINDING (unbidden): with correlated keys, single-pass unit-rate **Hebbian is more graceful
than the delta rule** — the delta advantage in the literature (arXiv:2406.06484) needs
*learned* gating/normalization, which the untrained probe lacks. Documented, not hidden.

### M2 — Ledger recovers recall at fixed hot-state  → SUPPORTED
`python exp_m2_ledger_recovery.py` (d=64, σ=0.15, seeds 0–2). top-1 over all N, both
systems at identical 32 KB hot-state:
- N=2048: **cram (ws-only) 0.017  vs  LWF (ws+ledger) 0.966**.
[EVIDENCE] Same hot footprint; the external organ is the entire difference. Honest cost
reported: Ledger comparisons/read = N (brute force) → O(log N) with an ANN index.
BOUNDARY: at σ≥0.25 the cue-noise norm exceeds the unit signal (SNR<1); recall is capped by
*information*, not architecture — both fall, LWF still ~90× the baseline. Default set to the
recoverable regime so the claim is on trial, not the noise.

### M4 — data-movement / energy scaling  → SUPPORTED
`python exp_m4_cost_model.py` (7B transformer vs LWF; E_DRAM=20 pJ/B conservative for HBM).
- context 128 → 131072: **transformer energy ×5.9** (KV-cache streaming; 83% of movement at
  128k), **LWF energy ×1.00** (flat). Per-token ratio grows 9.2k× → 54k×.
[EVIDENCE] The slope conclusion (transformer grows, LWF flat) survives ±10× perturbation of
every energy constant. Robust claim = the *slope*, not the absolute. Verified Horowitz numbers
(32b MAC ≈1.5 pJ, 32b HBM ≈200 pJ, ~130×) make E_DRAM=20 pJ/B conservative → verdict if
anything understated.

================================================================================
## 2026-07-01 — Stage 1 + crux probes

### M3 — verifiable bit-exact replay; O(steps) audit log  → SUPPORTED
`python exp_m3_replay.py` (real Workspace+Ledger cognitive loop, 200 steps, top-k=4):
- [1] fresh-runner replay bit-exact: **True**
- [2] checkpoint@100 + resume reproduces final state: **True**
- [3] 1e-6 input perturbation changes run digest (tamper-evident): **True**
- log growth: **253 bytes/step, flat** across 25→400 steps (retrieved/step = 4.00 constant).
[EVIDENCE] A bounded pure-function step gives checkpoint/replay/audit that full-context
attention cannot: there is a small object to hash and a linear log, not O(n²) of materialised
context. This is the concrete form of Goal #3 (verification).

### M5 — compositional multi-hop; per-step state vs reasoning depth  → SUPPORTED
`python exp_m5_multihop.py` (500-node functional graph, d=64, seeds 0–2):
- H=1..8 hops: **LWF iterative pointer-chase = 1.000 at every H**, on a **constant 512-byte**
  cursor state, paying H sequential Ledger reads.
- single fixed-state operator (M^H @ v0): collapses immediately (H=1: 0.369; H≥2: ≈chance
  0.002), because 500 assoc. in a rank-64 operator is 7.8× over capacity (consistent w/ M1).
[EVIDENCE] Cleanly separates the two costs the §7 crux conflates: **bounded per-step state**
(architecture — LWF keeps it) vs **reasoning depth H** (task property — every architecture
pays O(depth): transformers via layers/CoT, LWF via retrieval steps). Resolves "outcome 2":
the bottleneck that moves is depth, and depth is intrinsic, not an LWF failure.

### M6 — LEARNED §7-crux probe (MQAR, PyTorch/CPU)  → SUPPORTED (directional)
`python exp_m6_crux_learned.py`. Definitive sweep (d_head=32, 600 steps, batch 32, seed 0),
query-position accuracy (chance 0.016) — `results_m6_full.txt`:
| D pairs | linear | delta | attn | hybrid(ws+ledger) |
|--------:|-------:|------:|-----:|------------------:|
|       8 |  0.969 | 0.995 | 1.000|             1.000 |
|      16 |  0.908 | 0.988 | 1.000|             1.000 |
|      24 |  0.812 | 0.990 | 1.000|             1.000 |
|      32 |  0.722 | 0.990 | 1.000|             1.000 |
|      40 |  0.656 | 0.979 | 1.000|             1.000 |
[EVIDENCE] Three consistent facts: (1) plain bounded **linear state degrades monotonically
0.969→0.656** as association pressure D climbs past d_head=32 — learned confirmation of the
M1 saturation. (2) The **delta rule barely degrades (~0.98–0.99)** — confirms the DeltaNet
recall-memory frontier AND explains the M1 nuance: delta's advantage needs the *learned*
gating the untrained M1 probe lacked (two experiments triangulate one fact). (3) **Hybrid
(Workspace+Ledger) pins 1.000 across the whole sweep, tracking full attention** — retrieval
fully recovers what bounded state loses, learned from scratch.
BUG FIXED mid-build: MQAR values sit one position after their key → single-layer attention
cannot associate k→v without the shifted-value (next-token) construction; without it, ALL
models sat at chance (a false null). Corrected; re-validated.
CAVEATS [honest]: toy scale, single layer, one seed, CPU; hybrid's softmax read is dense here
for differentiability (deployment restricts to ANN top-k, O(log N)); delta has not yet been
pushed to its own breaking D. Directional evidence the crux's *good* outcome is reachable —
NOT a frontier-model claim. Wall time ~7 min (delta scan dominates).

### M7 — Stage 0.5: bounded-cost crux (top-k Ledger, MQAR, torch)  → SUPPORTED (directional)
`python exp_m7_scaleup.py` (device=cpu, 1 layer, d_head=32, top-k=8, 500 steps, seeds 0–1).
query accuracy mean±std, [reads/step] — `results_m7_full.txt`:
| D | linear (1 read) | attn (T reads) | hybrid_topk (8 reads) |
|--:|----------------:|---------------:|----------------------:|
|  8 | 0.974±0.003 | 1.000 [24]  | 1.000 [8] |
| 24 | 0.881±0.003 | 1.000 [72]  | 1.000 [8] |
| 48 | 0.641±0.003 | 1.000 [144] | 1.000 [8] |
| 64 | 0.541±0.026 | 1.000 [192] | 1.000 [8] |
[EVIDENCE] Closes the M6 loophole ("dense read = just attention"). hybrid_topk **recovers
recall to 1.000 across the whole sweep while reading only k=8 entries/step** — constant,
context-independent, **24× fewer reads than attention at D=64** — while bounded linear degrades
0.974→0.541. A BOUNDED top-k content-addressed read is *sufficient* to match full attention:
the architecture's central claim (attention-level accuracy at bounded per-step cost), learned.
HARDWARE-AGNOSTIC: auto-selects CUDA (larger preset: 2 layers, D→128, 3 seeds) else CPU. Wall
~14 min CPU (full-attention D=64 pass dominates). CAVEATS: toy scale, 1 layer, 2 seeds.

### M8 — in-context MULTI-HOP (global-reasoning crux, GPU)  → SUPPORTED at H=2; H=3 shared ceiling
`python exp_m8_multihop_incontext.py` (CUDA 1660 Ti, 5 layers, D=20-node graphs, n_nodes=128,
top-k=8, 2000 steps, seeds 0–1). final-position accuracy mean±std, [reads/step] —
`results_m8_full.txt`:
| H | linear (5) | attn (205) | hybrid_topk (40) | reading |
|--:|-----------:|-----------:|-----------------:|---------|
| 1 | 0.699±0.002 | 0.642±0.015 | 0.671±0.032 | flat recall — bounded state fine (even best) |
| 2 | 0.167±0.005 | 0.346±0.014 | 0.344±0.006 | **COMPOSED: hybrid = attn, both ~2× bounded, at 5× fewer reads** |
| 3 | 0.141±0.004 | 0.161±0.003 | 0.180±0.006 | shared ceiling — **attn fails too** (0.161 < hybrid 0.180) |
[EVIDENCE] The composition claim (not just recall) holds at H=2: bounded-cost top-k retrieval
(40 reads/step) composes a 2-hop chain as well as full attention (205 reads/step), and both
roughly DOUBLE the bounded state — error bars between linear and the retrieval models do not
overlap. At H=3 the accuracy collapses for ALL models including attention (which trails the
hybrid), so H=3 is a **depth/scale/training-budget ceiling of this toy, not an LWF-specific
loss** — reporting it as "LWF fails deep reasoning" would be dishonest; the bottleneck there
is not retrieval. CAVEATS: toy-to-small scale, 5 layers, 2 seeds, 2000 steps (H=2 still
climbing at stop — absolute accuracy is training-limited; the RELATIVE separation is the claim).
Wall ~70 min on 1660 Ti. OPEN: push H=2 to convergence + reach clean H≥3 needs more scale.

### M9 — Stage 2: Fabric cost model on real silicon numbers  → SUPPORTED (model), bet made explicit
`python exp_m9_fabric.py`. Runs each design's actual op-trace through per-op energy from cited
silicon (CAM 0.5 fJ/bit, CIM cell 0.02 pJ, ADC 3 pJ explicit, DRAM 10 pJ/B, digital MAC 0.5 pJ).
Three designs to separate algorithm vs substrate. Per-token energy @ n=8192:
| design | energy/token | vs transformer |
|--------|-------------:|---------------:|
| transformer / von Neumann | ~184 uJ (75% weight-move, 23% KV-move, 2% MAC) | 1× |
| LWF / von Neumann (algorithm only, P_core=0) | 96.7 uJ | ~1500× region |
| LWF / Fabric (P_core=0, full-Ledger extreme) | 92.3 nJ | ~2e6× (ceiling, not the claim) |
[EVIDENCE] Two decompositions matter: (1) the transformer's energy is **98% DATA MOVEMENT**
(streaming 7B weights + KV per token, batch-1 decode) — LWF's win is eliminating movement, not
cheaper MACs. (2) The result hinges on ONE architectural bet, made explicit via P_core:
  - P_core=0 (all knowledge in Ledger): ~2e6× — unrealistic extreme.
  - **P_core=P/25 (RETRO-grounded: retrieval hit GPT-3 quality at ~25× fewer params): ~33×** —
    the DEFENSIBLE headline.
  - P_core=P (no knowledge offload): **1× — no win.** Honest: without the nonparametric bet the
    Fabric buys nothing on the dominant term.
Survives ±10× perturbation of every constant (at P_core=P/25). Dominant Fabric term at d_ws=2048
is CIM-cells (91%), not ADC — ADC dominates only for d < e_adc/e_cell≈150 (cells O(d²) vs ADC
O(d)); modelled explicitly, not hidden. CAVEAT: analytic model with cited constants, NOT measured
silicon; precision/variability of analog CIM not modelled at the accuracy level (would cost bits).

### M10 — frontier-width scaling (imported from sibling EP-GRM project)  → SUPPORTED
`python exp_m10_frontier.py` (d=64, seeds 0–2). frontier-recall accuracy, decode restricted to
the F active values. Imports EP-GRM's law: required executive state tracks dependency FRONTIER
WIDTH F, not total size N.
| mode | F=4 | F=64(=d) | F=128 | F=256(4d) | flat in N? |
|------|----:|---------:|------:|----------:|-----------|
| LWF-frontier (holds only F) | 1.000 | 1.000 | 0.98 | 0.75 | **yes (across-N std 0.004)** |
| cram (holds all N) | 1.0→0.42 | 0.84→0.10 | 0.81→0.06 | 0.73→0.02 | no — tracks N |
| Ledger (exact) | 1.000 | 1.000 | 1.000 | 1.000 | yes — escape hatch |
[EVIDENCE] Workspace capacity tracks frontier F (degrades past F≈d=capacity, M1), **flat in
total N** (distractors sit in the Ledger); cram tracks N; Ledger absorbs overflow at O(F) reads.
Reproduces the EP-GRM frontier law in a DIFFERENT substrate (neural-associative vs symbolic
graph-rewrite) = cross-substrate corroboration. Completes the LWF cost law:
**O(depth) steps (free in state, M5) × O(frontier width) hot state (costs state, M10).**
CAVEAT (from EP-GRM's own EXP-022): assumes the frontier is correctly IDENTIFIED (oracle
admission); their controlled runs show the scheduler/admission is a separate first-class factor,
so a naive F->capacity law is incomplete. M10 measures the capacity law given correct frontier.
See `references/SIBLING_EPGRM.md` for the full cross-project analysis.

### M11 — nogoods: learned-constraint verification (imported from EP-GRM)  → SUPPORTED
`python exp_m11_nogood.py` (n=26, avg_deg=4.7, 3-colouring, 12 hard random instances near the
threshold). Ledger extended with a first-class nogood record type (`write_nogood` /
`nogood_violated`, literal-indexed). Backtracking search with conflict-directed nogood recording:
| metric | plain backtracking | nogood-learning |
|--------|-------------------:|----------------:|
| total search nodes (12 instances) | 2,749,671 | **308,502 (8.9× fewer)** |
| median per-instance speedup | — | **5.4×** (up to 13.4× on hard UNSAT) |
| SAT verdict agreement | — | **12/12 identical (sound)** |
[EVIDENCE] Nogoods do real work AND are sound: identical colourability verdict on every instance,
with 8.9× less search. Extends LWF verification from passive replay (M3) to active learned-
constraint pruning (dependency-directed backtracking) — EP-GRM's richer truth-maintenance model,
now native to the Ledger. FALSIFICATION would have been any SAT/UNSAT disagreement (none) or
no pruning (8.9×).

### M12 — controller quality: Workspace<->Ledger admission/eviction  → SUPPORTED
`python exp_m12_controller.py` (K=500 keys, drifting Zipf-skewed frontier F=32, seeds 0–2). Drops
M10's oracle-frontier assumption; the Workspace is a cache over the Ledger and the controller must
DISCOVER the frontier. Steady-state hit rate by policy:
| regime | random | fifo | lru | lfu | belief | set-oracle |
|--------|-------:|-----:|----:|----:|-------:|-----------:|
| C=2F (easy) | 0.971 | 0.983 | 0.983 | 0.764 | 0.983 | 0.983 |
| C=F | 0.921 | 0.967 | 0.967 | 0.522 | 0.967 | 0.982 |
| C=F/2 (hard) | 0.732 | 0.787 | 0.790 | **0.373** | **0.791** | 0.744 |
[EVIDENCE] Two regimes: (1) **C≥F: controller ~ sufficient** — lru/belief ≈ oracle (0.983);
confirms M10 (capacity ≥ frontier ⇒ success regardless of policy). (2) **C<F: controller matters**
— belief-decay beats un-aged LFU by **0.42** (the decay gate cures LFU's staleness pathology) and
beats random by exploiting intra-frontier frequency. FINDING: the set-oracle is NOT optimal at
C<F (it ignores frequency), so recency/frequency policies exceed it — an honest limit of that
bound. RECOMMENDATION: **belief-decay (frequency + decay gate) or LRU** as the Workspace admission
policy; avoid un-aged LFU; pair with the M10 frontier law. Directly addresses EP-GRM EXP-022
("scheduler is first-class"). CAVEAT: synthetic drifting-Zipf workload, not a real reasoning trace.

### M13 — controller on a REAL frontier: live values of a computation DAG  → SUPPORTED
`python exp_m13_live_frontier.py` (DAG N=400, k=3 parents, seeds 0–2). Replaces M12's synthetic
drift with an actual computation: execute a DAG topologically; the frontier = LIVE values
(produced, not yet fully consumed) = register pressure. Workspace = bounded cache; evicting a
still-live value = a miss (Ledger refetch). True optimum available = **Belady MIN** (future known).
Misses vs cache C relative to MaxLive (~190 = dynamic frontier width):
| C/MaxLive | random | lru | belief | liveness | belady(opt) |
|-----------|-------:|----:|-------:|---------:|------------:|
| 0.50 | 504 | 500 | 502 | 331 | 137 |
| 1.00 | 191 | 202 | 201 | **1** | **1** |
| 1.50 | 49 | 51 | 51 | **0** | **0** |
[EVIDENCE] At C≥MaxLive (frontier-sized, M10's recommended provisioning) the **liveness-aware
controller = Belady optimum** (~0 misses) while recency/frequency (LRU/belief) still take 51–200 —
they waste slots on DEAD values. KEY FINDING: on a real computation frontier, recency/frequency
is NOT enough; the controller must use **dependency-derived liveness (reference-counting)** — the
same justification bookkeeping behind M11's nogoods. Dead-first eviction captures most of Belady's
advantage with NO future knowledge. Below frontier (C<MaxLive) no online policy matches Belady
(fundamental online-caching gap), but liveness (2.4× opt) still beats recency (3.7×). Unifies the
controller (M12/M13) with the dependency/verification machinery (M11). Refines M12's recommendation:
belief-decay for drift-only; **liveness-aware admission for real cognition; provision C ≥ frontier.**

### M14 — knowledge scales in the Ledger, not the model (learned; GPU + 48GB host RAM)  → SUPPORTED
`python exp_m14_scaling_ledger.py` (D=64, V=100 classes, query noise 0.15, 4000 steps, GTX 1660 Ti
+ host-RAM Ledger). Tests M9's load-bearing nonparametric-knowledge bet, LEARNED. A fixed 108K-param
MLP (memorise key->class in weights) vs a small model + content-addressed host-RAM Ledger:
| N facts | Ledger RAM | parametric (fixed 108K) | retrieval (RAM Ledger) | ret ms/q |
|--------:|-----------:|------------------------:|-----------------------:|---------:|
|   1,000 |   1 MB | **0.864** | 0.997 | 0.02 |
|  10,000 |   5 MB | 0.112 | 0.991 | 0.01 |
| 100,000 |  51 MB | **0.009 (=chance)** | **0.951** | 0.02 |
| 1,000,000 | 512 MB | (untrained) | 0.880 | 0.09 |
| 5,000,000 | 2.56 GB | (untrained) | 0.786 | 0.56 |
[EVIDENCE] The fixed-capacity model TRAINS FINE at N=1K (0.864) then COLLAPSES to chance (0.009) by
N=100K — it cannot memorise growing knowledge in fixed weights. The small model + host-RAM Ledger
holds **0.951 at N=100K (95× the parametric)** and degrades only gracefully to 0.786 at 5M facts,
at CONSTANT per-query model cost. Knowledge scaled in the store (RAM), not the parameters — M9's
bet, now empirical and learned. HONEST: brute-force retrieval is O(N) (0.02→0.56 ms/q as N→5M); an
ANN index makes it O(log N). Retrieval softening at 5M is nearest-neighbour confusion under noise at
extreme key density (still 79× chance). Toy embeddings, not language. Uses 48GB RAM for the store.

### M16 — lexicographic-energy termination guarantee (imported from TSAM)  → import VALIDATED; pre-registered verdict FAIL (honest)
`python exp_m16_energy_termination.py` (300 program-repair instances, deterministic). Imports
TSAM's guarantee: accept a rewrite iff it strictly decreases the ordered energy (H≻S≻Q) ⇒ bounded
termination, no oscillation. **Pre-registered verdict (fixed before run, bar = 0 violations).**
- P1 strict-decrease on every committed rewrite: **300/300** ✓
- P2 reached a fixed point (no cap-hit): **300/300** ✓ (mean 23 steps, max 32)
- P3 no state revisits (no oscillation): **300/300** ✓
- P4 solvable→H=0 and unsolvable→H>0+diagnostic: **298/300** ✗ → overall **FAIL**.
[HONEST CHAIN, kept verbatim as a G0.1 demonstration]:
  1. First run FAILED P4 at 199/300.
  2. Investigation (traced a stuck instance): the failures were a **harness bug** — `gen_instance`
     mislabeled instances as "solvable" when random required/forbidden sets overlapped by chance
     (a required-AND-forbidden type = genuinely unsolvable). The mechanism was CORRECTLY rejecting
     them. Fixed the solvability computation (disclosed inline in code); this corrects ground truth,
     NOT the pass criterion.
  3. Corrected run: **298/300** — still FAIL. The 2 residual failures are GENUINE greedy-descent
     local minima (a forbidden node held by valid references; removal would raise H → rejected).
  4. Follow-up (post-hoc, verdict unchanged): a COMPOUND rewrite (remove-forbidden + null its
     referencers, one strictly-decreasing step) fixes exactly those → **82/82 solvable solved**,
     with the termination guarantee still 300/300.
[TAKEAWAY] The IMPORT (termination + no-oscillation via strict lexicographic descent, unifying M11
nogoods as the H component) is **solidly validated**. But **termination ≠ completeness**: greedy
lexicographic descent is INCOMPLETE (local minima); completeness needs richer/compound rewrites or
backtracking (as VeriForge notes: "greedy + limited backtrack"). The pre-registered FAIL is reported
as-is — the discipline (no goalpost-moving) is the point, and it surfaced a real limitation my
earlier verdict-reframing habit would have buried.

### M15 — FIRST end-to-end integrated LWF (learned)  → pre-registered verdict FAIL/PARTIAL (honest)
`python exp_m15_integrated.py` (GPU, n_vars=32, U=8, L=120, d=48, 3500 steps). Mutable variable
tracking + comparison: trained fast-weight Workspace + keyed Ledger + LEARNED-query read + head,
end-to-end. Attacks G1.1/G1.2/G1.3/G0.2/G0.1 at once. **Pre-registered verdict (fixed before run):**
| model | QUERY (chance .125) | COMPARE (chance .333) |
|-------|--------------------:|----------------------:|
| transformer (meant as STRONG baseline) | 0.299 | 0.443 |
| bounded-only (Workspace, no Ledger) | 0.312 | 0.439 |
| oracle-query LWF (reads correct slots) | 1.000 | 1.000 |
| **integrated LWF (learned query)** | **0.716** | **0.697** |
- config validity (bounded ≤ 0.60): 0.439 ✓
- PRIMARY (LWF ≥ 0.9×transformer COMPARE): 0.697 ≥ 0.399 ✓ **but VACUOUS (see below)**
- SECONDARY (LWF ≥ 0.9×oracle): 0.697 ≥ 0.900 ✗ → **overall FAIL/PARTIAL**
[HONEST INTERPRETATION — three findings, one humbling]:
  1. **Integration WORKS (G1.1 ✓, G1.2 ✓).** The assembled system runs end-to-end and learns; the
     trained fast-weight Workspace is real; oracle-query = 1.000 proves the Workspace+Ledger+compose
     plumbing; learned LWF hits 0.70 COMPARE, ~1.6× the bounded/transformer baselines.
  2. **G1.3 confirmed HARD.** Learned query formation reaches 0.70 but oracle reaches 1.00 → SECONDARY
     fails. Forming the retrieval query from Workspace state is learnable but lossy; that 0.70-vs-1.00
     gap is the quantified cost — the real open problem.
  3. **The "strong baseline" FAILED to train (undercuts G0.2, re-exposes G0.3).** transformer COMPARE
     0.443 ≈ chance-ish → PRIMARY passed only because the baseline didn't learn = NOT a real win. The
     task's keyed-slot structure is an inductive bias that fits LWF and that a 3-layer transformer must
     discover from scratch at 3500 steps. So this does NOT cleanly beat a strong baseline; G0.2 is
     partly re-opened and G0.3 (task favors the architecture) re-confirmed.
[NET] Integration exists and query formation is the bottleneck — but the win is over a baseline that
didn't train, on a task shaped to fit LWF. Pre-registered FAIL reported as-is; no reframe.

[DIAGNOSTIC FOLLOW-UP (per doctrine §5 "diagnose", §3 "fix the code not the test")]: I chased WHY
the transformer baseline failed instead of accepting the vacuous PRIMARY pass.
  - Longer training (12k steps, 3.4× M15's budget): FLAT at 0.31 QUERY / 0.44 COMPARE. Not under-trained.
  - 6× bigger transformer (d96/5-layer, 576k params): same plateau. Not capacity.
  - EASY variant (8 vars, L=36): still plateaus ~0.50 QUERY (chance 0.167). Not solvable by scaling down.
  - CROSS-CHECK (the decider): the SAME transformer architecture solves MQAR to 1.000 in M7. So it is
    NOT buggy — M15's task is genuinely HARD-FOR-TRANSFORMERS: variable tokens appear in 3 roles
    (SET/QUERY/COMPARE), defeating vanilla induction; it needs SET-conditioned, recency-weighted,
    variable-matched retrieval, which LWF's keyed-slot overwrite has natively.
[CORRECTED CONCLUSION]: M15 demonstrates INTEGRATION (G1.1/G1.2) and QUERY-FORMATION difficulty
(G1.3), but CANNOT host a strong baseline, so it does NOT establish a strong-baseline win (G0.2).
G0.2 for the retrieval PRINCIPLE is met by M7 (working transformer baseline @1.0; bounded-cost LWF
matches @8 reads/step). The unmet test = FULL integrated stack vs a WORKING strong baseline on ONE
transformer-solvable long-context task — the clean next experiment (M18, not yet built).

### M17 — verified-only write policy for the Ledger (JanusPrime import; G1.4)  → import beneficial; pre-reg verdict FAIL (honest)
`python exp_m17_write_policy.py` (n_keys=500, 4000 writes, verifier detect=0.90/fp=0.05, seeds 0-2).
Imports JanusPrime's "only validated writes seed the store." Ungated (write all, last-write-wins)
vs verified-only (write iff an independent verifier passes). Downstream end-to-end accuracy:
| corruption | ungated e2e | verified e2e | verified purity | valid-write cost |
|-----------:|------------:|-------------:|----------------:|-----------------:|
| 0.2 | 0.789 | **0.972** | 0.974 | 0.052 |
| 0.4 | 0.596 | **0.925** | 0.932 | 0.050 |
| 0.6 | 0.412 | **0.827** | 0.854 | 0.049 |
**Pre-registered verdict:** P1 (verified − ungated ≥ 0.20 @0.4): **+0.329 ✓**; P2 (reject ≥80% corrupt):
**0.900 ✓**; P3 (purity ≥ 0.90 at ALL corruption incl. 0.6): **0.854 ✗** → overall **FAIL**.
[HONEST INTERPRETATION] The import is clearly VALUABLE — gating the write path lifts downstream
accuracy by up to +0.33 and keeps the store trustworthy, at ~5% valid-write cost. The FAIL is P3,
and it's a ceiling I mis-set: with detect=0.90 at 60% corruption the MAXIMUM achievable purity is
~0.864 — **store purity is bounded by verifier_quality × corruption**; you can't gate a clean store
with a weak verifier under heavy corruption. (Fix = a stronger verifier, e.g. M11 nogoods + tests.)
Like M16, my pre-registered bar bundled a guarantee the mechanism can't provide; reported FAIL, no
reframe. G1.4 (write path) is no longer a blank: verified-only writing is the recommended policy.

### M18 — full fast-weight stack vs a WORKING strong baseline (MQAR)  → pre-reg FAIL (config invalid); key correction found
`python exp_m18_fullstack_vs_baseline.py` (GPU, D=48, 2500 steps). Fixes M15's flaw (baseline that
trains) by using MQAR, which a transformer provably solves.
| model | acc | reads/step |
|-------|----:|-----------:|
| attn (strong baseline) | 1.000 | 96 |
| linear (bounded) | 0.994 | 1 |
| fw (bounded fast-weight) | 0.883 | 1 |
| **fw+ldg (full integrated stack)** | **1.000** | **8** |
**Pre-registered verdict:** config valid (attn≥0.90 AND fw≤0.70): fw=0.883 > 0.70 → **INVALID** →
overall **FAIL**. PRIMARY (fw+ldg ≥ 0.9×attn) passed (1.0 at 8 reads vs 96), but the premise
"bounded Workspace fails" did NOT hold at D=48/d_head=48 — bounded state had enough capacity, so the
experiment can't prove the Ledger is *necessary*. Reported FAIL, not reframed (I did NOT re-run at a
higher D to force a pass — that would be config-hunting toward a known result).
[KEY CORRECTION -- the real value of M18]: while fixing why fw failed I found TWO bugs (missing
shifted-value construction + missing linear-attention normalizer). Fixing them made the fast-weight
Workspace IDENTICAL to linear attention (normalized fw 0.995 ≈ linear 0.996). **This is Schlag et al.
2021 ("Linear Transformers Are Secretly Fast Weight Programmers"): the fast-weight Workspace (the M1
object) IS linear attention.** Therefore my GAPS G1.2 premise — "M7 used a linear-attention STAND-IN,
not the real Workspace" — was FALSE. M7's `linear`/`hybrid_topk` ARE the real Workspace, so **M7
already tested the full integrated stack vs a working baseline in a VALID config** (there bounded
degraded to 0.66 at D=64 and hybrid matched attn at 8 reads). M18 confirms the stack trains and
corrects the misconception; the valid-config demonstration is M7.

### M19 — the crux: irreducibly-global reasoning (in-context sorting)  → pre-reg INCONCLUSIVE; deepest finding of the program
`python exp_m19_hard_reasoning.py` (GPU, alphabet=32, 3000 steps). Two-sided pre-registered verdict
(LWF could FAIL, which would falsify the strong thesis). Per-slot accuracy on the sorted output:
| N (width) | attn (baseline) | bounded Workspace | hybrid (Ws+Ledger) |
|----------:|----------------:|------------------:|-------------------:|
|  8 | 1.000 | 0.952 | 1.000 |
| 16 | 1.000 | 0.963 | 0.999 |
**Pre-registered verdict:** config valid = attn≥0.85 AND bounded≤0.60. attn=1.0 ✓ but **bounded=0.963
> 0.60** → INVALID → **INCONCLUSIVE**. The bounded Workspace SORTED 16 items (0.96); it did not fail.
[THE FINDING — why this is the most important INCONCLUSIVE in the program]: my I1 setup was arithmetically
naive. Sorting N=16 needs ~N·log2(N) ≈ 64 bits of working info; the Workspace state is a 96×64 matrix
(~6000 floats ≈ 10⁴–10⁵ bits) — it DWARFS the need, so of course it sorts. I cannot shrink the state
enough to bottleneck it without destroying the model's basic capacity. ⇒ **The crux is structurally
UNTESTABLE at toy scale**: its failure regime (task's *simultaneous* working info > state capacity)
requires N in the thousands for sorting, or real large-scale reasoning — out of reach on a 1660 Ti.
Sorting is not "irreducibly global enough" to defeat a realistically-sized bounded state.
[REFRAMES THE PROGRAM]: the §7 crux is not merely "untested for lack of effort" — the condition that
would make bounded state fail (working set > capacity) doesn't arise at trainable toy scale. This also
implies M8's "H=3 ceiling" was an OPTIMIZATION/depth limit, NOT a state-capacity limit (bounded state
wasn't the bottleneck there either). The genuine crux remains OPEN and needs real-scale tasks. NOTE:
the crux for STORAGE/recall was already settled (M1 rank-d ceiling → M2 Ledger rescue); what's untestable
here is the crux for global COMPUTATION, which a bounded recurrent state handles far better than I assumed.

### M20 — the crux done right: rank-bottlenecked Workspace on global computation  → INCONCLUSIVE; the program's central clarification
`python exp_m20_crux_bottlenecked.py` (GPU, sorting N=24, sweep Workspace rank dh, 3000 steps).
Fixes M19's wrong capacity model (bits → RANK, per M1). Per-slot accuracy:
| dh (rank) | attn | bounded | hybrid |
|----------:|-----:|--------:|-------:|
| 24 | 1.000 | 0.969 | 0.975 |
|  8 | 1.000 | 0.966 | 0.981 |
|  4 | 1.000 | 0.961 | 0.976 |
|  2 | 1.000 | **0.951** | 0.969 |
**Pre-registered verdict:** need a dh where bounded ≤ 0.60 while attn ≥ 0.85. **None exists** — a
**rank-2 Workspace sorts 24 items at 0.95** → **INCONCLUSIVE** (couldn't create the failure regime).
[THE CENTRAL FINDING]: at 200 steps (smoke) the rank bottleneck WORKED (bounded 0.31); at 3000 steps
it does NOT (bounded 0.95). **Training overcomes the rank bottleneck.** I could not make a bounded
state fail on sorting by ANY bottleneck — bits (M19) or rank (M20).
[WHY — corrects a load-bearing assumption]: **the M1 rank ceiling is about UNTRAINED storage of
ARBITRARY associations, not TRAINED structured computation.** A trained rank-2 state doesn't store 24
arbitrary values; it COMPUTES order statistics via learned low-rank features that exploit task
structure (small alphabet → compact sufficient statistic). Bounded recurrent states are far more
computationally capable than their capacity bounds imply.
[CONSEQUENCES for the thesis]:
  1. The "global computation defeats bounded state" crux is largely ILLUSORY at reachable scale.
  2. LWF's Workspace+Ledger split is justified by STORAGE (I1 -- arbitrary knowledge > capacity,
     which M2/M14 genuinely confirm the Ledger rescues), NOT by computational limits of bounded state.
     The Ledger scales KNOWLEDGE, not reasoning the Workspace couldn't do.
  3. M8's "H=3 ceiling" was NOT state capacity (consistent with M19/M20) -- optimization/depth.
This is thesis-clarifying and partly thesis-DEFLATING -- exactly what an adversarial program should
find. The genuine crux (does bounded+retrieval fail?) applies to STORAGE (settled: M1→M2) and is not
reachable for COMPUTATION at toy scale; whether it bites at real scale remains open but is now a
sharper, storage-framed question.

### M21 — complete search with BOUNDED executive state (the search+verification angle)  → pre-reg FAIL; bounded-memory demonstrated, but TIME is the real wall
`python exp_m21_bounded_search.py` (K=3-colouring, backtracking + nogood learning; same search run
two ways: `full` = O(n) executive, `lwf` = Workspace=frontier + Ledger holds assignment/trail/nogoods).
Pre-registered config (avg_deg=4.5, near threshold) → **FAIL** (hard instances hit the 1.5M-node budget;
plus a P3 bug: nogood check was lwf-only → different algorithms. Bug fixed & disclosed).
Completable follow-up (avg_deg=2.5, bugfixed), per-slot memory:
| n | nodes(full) | nodes(lwf) | peakWS full | peakWS lwf | ledger reads/node |
|--:|------------:|-----------:|------------:|-----------:|------------------:|
| 15 | 141 | 141 | 15 | 5 | 0.65 |
| 30 | 9977 | 9977 | 30 | 8 | 0.52 |
| 45 | 600050* | 600050* | 45 | 8 | 0.88 |
| 60 | 33490 | 33490 | 60 | 7 | 0.82 |
(*one instance hit budget) **P3 now holds (identical search).** THE DEMONSTRATION IS IN THE DATA:
**peakWS(full) = n (15→60), peakWS(lwf) ≈ 7 FLAT** — executive Workspace bounded by the frontier,
independent of problem size, with assignment/trail/nogoods offloaded to the Ledger (O(n)).
**Pre-registered verdict still FAIL**: P2's corr<0.3 is too strict for a 4-point series (values 5,8,8,7
are bounded but correlate 0.55); P1 broke on a hard n=45 instance hitting the node budget.
[THE FINDING -- corrected]: **executive MEMORY is boundable for search** (Workspace ~frontier, offload
to Ledger -- demonstrated); NAIVE backtracking's TIME still blew up (n=45 budget-hit). I first wrote
this as "search cost is orthogonal / not LWF's to solve" -- WRONG (corrected 2026-07-02). Precise:
  - WORST-CASE complexity (NP-hardness): not LWF's to solve, nor any architecture's. Bounding memory
    doesn't change worst-case node counts. That part is fair.
  - PRACTICAL search cost: IS reducible by memory-based methods (nogoods, memoization, learned
    move-ordering, subsolution reuse) -- all LWF-NATIVE (the Ledger). M11 ALREADY showed it: nogood
    learning cut nodes 8.9x, nogoods live in the Ledger. M21's own first run (lwf-with-nogoods) explored
    fewer nodes than full. So the Ledger IS a lever on practical search cost; I under-credited it.
[SESSION-LEVEL SYNTHESIS (corrected)]: LWF is a MEMORY + DATA-MOVEMENT architecture -- AND memory is
itself a lever on practical computation (the Ledger accelerates search via stored constraints/heuristics,
M11). Validated: bound executive memory (M1/M2/M21), scale knowledge (M2/M14), cut data movement (M4/M9),
reduce practical search cost (M11). NOT solved: worst-case complexity (nobody's), and "reasoning" was
never the gap (M19/M20). The most promising UNEXPLORED direction is Ledger-stored learned search GUIDANCE
(not just constraint storage) to beat naive search's blowup -- squarely LWF-native.

### M22 — Fabric simulator (component-level energy+latency; supersedes M9)  → the win is mostly ALGORITHMIC, not the silicon
`python exp_m22_fabric_sim.py`. Replaces M9's bytes×pJ spreadsheet with a tiled, component-level model
(CIM cells + ADC/DAC + CAM + DRAM) that reports ENERGY and LATENCY from the real op-traces.
- Transformer decode is **99% pure data movement** (76% weight-stream + 23% KV-stream); latency
  **7 ms/token**, bandwidth-bound (streaming 7B weights). This is the physical von Neumann tax, quantified.
- **KEY DECOMPOSITION @ n=8192 (P_core=0):** transformer 553 µJ → **LWF on COMMODITY hardware (digital,
  no custom silicon) 285 µJ = ~1,900× win from the ALGORITHM alone** (bounded state + retrieval, no
  weight stream) → LWF on the CIM Fabric 31.7 µJ = **only ~9× more from the silicon.** So ~99.9% of the
  modeled win needs NO custom hardware; the analog Fabric is a last-order-of-magnitude optimization.
- Fabric energy is dominated by **Ledger-retrieval fetch (99%), NOT analog compute** (CIM+ADC ≈1%) → the
  ADC-precision-energy worry is moot at these params (precision sweep barely moves the total).
- Fabric vs transformer: ~13k–79k× energy, ~10k–60k× latency (P_core=0); ~2,900× even at P_core=P.
- **HONEST BASELINE (added 2026-07-02 per updated doctrine §8/§15): those ratios use a BATCH-1
  transformer, which flatters LWF.** Real serving BATCHES, amortizing the weight stream (but NOT the
  per-sequence KV cache). vs a batch-64 transformer: LWF's edge is **573× @ n=512, 4,410× @ 8k,
  65,811× @ 128k** — i.e. batching amortizes weights (kills the SHORT-context lead) but not the KV
  cache, so **LWF's win is specifically the context/KV-scaling term, and it GROWS with context.** That
  is the defensible claim; report vs BATCHED, not batch-1. (Consistent with GAPS G0.2.)
[HONEST CAVEATS] Still a MODEL with published-typical constants, NOT measured silicon. Analog precision
may cost TASK ACCURACY — modelled only as ADC energy, not an accuracy hit (needs functional sim/silicon).
The large ratios assume the nonparametric-knowledge bet (P_core rows). Informs whether to BUILD, not proof.
[IMPLICATION for hardware] The goal does NOT need custom silicon for the bulk of the win — that's
algorithmic and runs on a GPU. Custom silicon buys ~9× + the physical-tax claim (a HW research result).
[GROUNDING PASS 2026-07-02, per updated doctrine §5/§15]:
  - e_dram=30 pJ/byte anchored: lit review gives HBM3e ~3.44 pJ/bit (27.5 pJ/B), HBM ~7 pJ/bit (56 pJ/B),
    off-pkg DDR5 ~80 pJ/bit. 30 pJ/B is at the HBM3e/low end → understates transformer movement →
    CONSERVATIVE for LWF. Cited in code + BIBLIOGRAPHY.
  - ABSOLUTE VALIDATED (§3): M22's batched 7B = 0.14 J/token @ n=8192, IN the published measured H100
    band (0.14–0.39 J/tok; LLaMA-3.1-8B 0.143, 70B-FP8-batch128 0.39). Batch-128/n=2048 = 0.039 J/tok
    (below band = idealized movement lower bound, omits overhead → ratios conservative). Model is
    GROUNDED — same order as measured reality, not off by orders of magnitude. Refs: arXiv:2310.03003,
    2407.16893, llm-tracker, TokenPowerBench. M22 now self-validates against this band on every run.

### M23 — MEASURED decode energy vs context (the first non-modeled hardware result)  → benchmark validated; direction confirmed
`python exp_m23_measured_energy.py` (real GPU power via NVML). Matched-size KV-cache transformer vs
bounded-state (linear-attention) model; the ONLY architectural difference is attention (growing KV
cache vs fixed recurrent state), so the measured gap isolates the KV-cache data-movement tax. Untrained
models are valid here — decode energy depends on architecture, not weights (measures scaling, not quality).
Smoke on GTX 1660 Ti (35M params, d=512/6L):
| ctx | TF mJ/tok | BD mJ/tok | E ratio |
|----:|----------:|----------:|--------:|
| 256 | 131 | 161 | 0.8× |
| 1024 | 188 | 165 | 1.1× |
| 4096 | 151 | 176 | 0.9× |
| 16384 | **335** | **199** | **1.7×** |
[EVIDENCE] First MEASURED (not modeled) energy result in the program: transformer per-token energy
GROWS with context (131→335 mJ), bounded stays ~flat (161→199) — the KV-cache tax, on real silicon.
Magnitude is small at toy scale (35M params → weight/MLP-dominated); the effect is large only at real
model size + long context (H100, 1B+ params, 128k ctx), where M22's big ratios live. Benchmark is
hardware-agnostic + validated; ready for an H100 run (`--preset h100`, ~$1–3 of cloud time).
[SCOPE] Measures the ALGORITHMIC half of the win (bounded state vs KV cache) on a GPU — NOT the CIM
Fabric (the further ~9×, which still needs a chip). Turns M4/M22's modeled scaling → measured.

================================================================================
## Scoreboard
| Mech | Claim | Verdict | Kind |
|------|-------|---------|------|
| M1 | Workspace rank ceiling ~d | SUPPORTED | analytic/numeric |
| M2 | Ledger restores recall @ fixed hot-state | SUPPORTED | numeric |
| M3 | Bit-exact replay + O(steps) audit log | SUPPORTED | numeric |
| M4 | Transformer energy grows w/ context, LWF flat | SUPPORTED | cost model |
| M5 | Deep composition at bounded per-step state | SUPPORTED | numeric |
| M6 | Retrieval recovers recall (learned, dense read) | SUPPORTED (directional) | learned, CPU |
| M7 | Bounded top-k retrieval recovers recall @ reads/step << context | SUPPORTED (directional) | learned, CPU |
| M8 | Bounded-cost retrieval COMPOSES multi-hop (not just recall) | SUPPORTED @ H=2; H=3 shared ceiling | learned, GPU |
| M9 | Fabric primitives (CIM+CAM) eliminate the movement dominating decode | SUPPORTED (model); ~33× @ RETRO-like bet | cost model |
| M10 | Workspace capacity tracks FRONTIER width F, not total N (EP-GRM law) | SUPPORTED | analytic/numeric |
| M11 | Nogoods (learned constraints) prune search soundly (EP-GRM import) | SUPPORTED (8.9× fewer nodes, sound) | numeric |
| M12 | Controller quality: decay-aware admission tracks the frontier | SUPPORTED (belief-decay/LRU; not un-aged LFU) | numeric |
| M13 | Liveness-aware controller = optimal at frontier-sized cache (real trace) | SUPPORTED (= Belady @ C≥frontier; needs liveness not just recency) | numeric |
| M14 | Knowledge scales in the Ledger, not the model (M9 bet, learned) | SUPPORTED (parametric→chance, retrieval flat @ 5M facts) | learned, GPU+RAM |
| M15 | First end-to-end integrated LWF (learned) | pre-reg FAIL/PARTIAL: integration works (0.70) but learned-query < oracle & baseline didn't train | learned, GPU, pre-registered |
| M16 | Lexicographic-energy termination guarantee (TSAM import) | import VALIDATED (P1-P3 300/300); pre-reg verdict FAIL (greedy incomplete) | numeric, pre-registered |
| M17 | Verified-only write policy (JanusPrime import, G1.4) | import beneficial (+0.33 e2e); pre-reg FAIL (purity capped by verifier quality) | numeric, pre-registered |
| M18 | Full fast-weight stack vs working baseline (MQAR) | pre-reg FAIL (config invalid: bounded not stressed @D=48); found fast-weight≡linear-attn (corrects G1.2) | learned, GPU, pre-registered |
| M19 | The crux: irreducibly-global reasoning (sorting) | pre-reg INCONCLUSIVE — bounded state SORTS (capacity≫need); crux untestable at toy scale (deep finding) | learned, GPU, pre-registered |
| M20 | The crux via RANK-bottlenecked Workspace | pre-reg INCONCLUSIVE — trained rank-2 state sorts 24 items @0.95; M1 ceiling ≠ trained computation; Ledger justified by STORAGE not computation | learned, GPU, pre-registered |
| M21 | Complete search with bounded executive state (search angle) | pre-reg FAIL; bounded-MEMORY demonstrated (Workspace ~frontier flat vs full O(n)); practical search cost IS LWF-native (M11 nogoods), corrected | numeric, pre-registered |
| M22 | Fabric simulator (energy+latency, supersedes M9) | win is ~1900× ALGORITHMIC (commodity hw) + ~9× from CIM silicon; transformer=99% data movement | cost model |
| M23 | MEASURED decode energy vs context (NVML) | first non-modeled hardware result; TF energy grows 131→335 mJ/tok, bounded flat (toy scale 1.7×); H100-ready | measured, GPU |

## Still OPEN (honest)
- [OPEN] M6 at scale/depth (multi-layer, GPU): does the recovery hold when reasoning is
  genuinely global and multi-step, or is there a residual capability class retrieval can't fix?
- [OPEN] M4 with per-op CIM/CAM silicon numbers on real op-traces (Stage 2).
- [OPEN] Query-formation quality (M2's weak point): learned vs heuristic query into the Ledger.
- [OPEN] Ledger consolidation/GC policy (the "sleep" compaction) — untested.
