# LWF — design invariants & mechanism ledger

Working name: **Ledger / Workspace / Fabric (LWF)** — a bounded-state cognitive
architecture aimed at the von Neumann bottleneck for intelligent computation.

Organizing principle (control-theoretic): *state is a sufficient statistic of
history, not a log of it.* Transformers conflate the two (the KV cache **is** the
history), which is the root of the unbounded-state and data-movement walls.

## The four walls (diagnosis)
- **W1 Physical data-movement tax** [FACT]. Off-chip access ≈ 200–700× a FLOP in
  energy (Horowitz, ISSCC 2014). The literal von Neumann tax.
- **W2 Bandwidth-bound decode** [FACT]. Batch-1 transformer decode has arithmetic
  intensity ≈ 1–2 FLOP/byte; it streams all weights + KV cache per token.
- **W3 State-is-a-log** [FRAME]. Working state = full history (KV cache): O(n) mem,
  O(n²) accumulated compute, unbounded in task duration. An architectural choice.
- **W4 Knowledge/cognition fusion** [FRAME]. No first-class writable persistent store
  the cognition manipulates at inference; everything routes through attention.

## Invariants (the fence — what no design may promise)
- **I1 Bounded-state capacity** [FACT/derivable]. A B-bit state distinguishes ≤ 2^B
  histories ⇒ exact recall of an unbounded growing set from fixed state is impossible
  ⇒ **external memory is mandatory, not optional.** Forces the three tiers.
- **I2 Data-movement asymmetry** [FACT]. The physical win is in *not moving weights*,
  not in fewer FLOPs.
- **I3 Landauer floor** [FACT]. Irreversible erase ≥ kT·ln2 ≈ 2.75 zJ @ 300 K; we run
  ~10⁶–10⁷ above it (headroom, but not free). Reversibility = distant lever.
- **I4 Lossy no-free-lunch** [FACT]. Fixed state must discard info; the only question
  is whether discarded info is task-irrelevant (survivable) or task-relevant (fatal).
  Empirical, per task distribution. This is the program's crux.

## The three tiers
- **Workspace** (Tier 1) — bounded executive memory. Gated fast-weight associative
  matrix M ∈ ℝ^{d_v×d_k}; O(1) size/step in task length. Lossy on purpose. → Goals 1,2,5.
- **Ledger** (Tier 2) — persistent, content-addressable knowledge + episodic log. Read
  cost independent of total size (ANN O(log N)); the exact-recall organ I1 mandates.
  → Goals 4,7. Also the audit log → verifiability (Goal 3).
- **Fabric** (Tier 3) — hardware abstraction: Workspace matvec → SRAM/CIM tile; Ledger
  search → CAM/associative-search-in-array. Weights never stream. → Goals 6,8.

**Co-design thesis:** the algorithm that dodges W3/W4 (bounded state + content-addressable
store) is *natively the same shape* as the hardware that dodges W1/W2 (in-array matvec +
in-array search). Software and hardware reinforce, not fight.

## Execution model
A **cognitive step** replaces fetch-decode-execute:
`(S_t, x_t, R_t) → (S_{t+1}, y_t, W_t)`, with `R_t = Ledger.read(query(S_t,x_t))`.
Fixed cost per step, independent of history length. Only nondeterminism: ANN approx +
analog/FP noise — both boundable (log retrieved IDs; error-margin the Fabric) ⇒ replayable.

## Mechanisms (falsifiable)
| ID | Claim | Test | Status |
|----|-------|------|--------|
| M1 | Workspace holds ≤ ~min(d_v,d_k) exact associations; predictable degrade past it | `exp_m1_capacity.py` | **SUPPORTED** (exact ≤ d orthonormal; smooth decay past d) |
| M2 | External Ledger restores recall past the ceiling at identical hot-state | `exp_m2_ledger_recovery.py` | **SUPPORTED** (0.97 vs 0.02 @ N=2048, 32 KB hot both) |
| M3 | Bounded pure-function step ⇒ bit-exact replay + O(steps) audit log | `exp_m3_replay.py` | **SUPPORTED** (replay/resume/tamper-evident; 253 B/step flat) |
| M4 | Transformer per-token energy grows with context; LWF flat | `exp_m4_cost_model.py` | **SUPPORTED** (xf ×5.9 over ctx 128→131072; LWF ×1.00; robust to ±10× constants) |
| M5 | Deep composition at bounded per-step state (via iterative retrieval) | `exp_m5_multihop.py` | **SUPPORTED** (1.000 @ 8 hops, 512 B state; fixed-op → chance) |
| M6 | Retrieval recovers recall bounded state loses (LEARNED, MQAR) | `exp_m6_crux_learned.py` | **SUPPORTED, directional** (D=24: linear 0.795 → hybrid 1.000 = attn) |

**Ledger read = modern-Hopfield/attention** (Ramsauer et al. 2020): a store of N patterns
retrieves in one softmax step with exponentially small error, and that update rule IS
attention. So attention is Hopfield lookup over context; the LWF Ledger is Hopfield lookup
over a *persisted, indexed* store. Exponential capacity needs storing the N patterns (O(N·d)) →
consistent with I1 (bounded *state* ≠ bounded store). Implemented as `Ledger.read_hopfield`.

## The §7 crux (can kill the program) — NOT tested at Stage 0
Is long-horizon reasoning capacity preserved under (bounded Workspace + top-k Ledger),
or is some reasoning irreducibly global (all-pairs over a large active set)? This needs
a *learned* model (GPU, Stage 0.5+): a capability ladder (assoc-recall → variable-binding
→ multi-hop → agentic synthesis) vs full attention, measuring accuracy AND per-step cost.
Outcomes: (1) tracks at flat cost → strong support; (2) tracks but cost reconverges via
hop-explosion → partial; (3) accuracy cliff retrieval can't fix → falsified for that class.

## What is NOT claimed
Not replacing transformers/CPUs/GPUs/OSes. Not unbounded exact recall from bounded state
(I1 forbids it). Parts (SSMs, external memory, retrieval, CIM) are prior art; the
contributions are the I1/I4-*forced* split, the rank-argument evict boundary, the
co-design isomorphism (M4), and the verifiable step semantics (M3).

## Key references (reference implementations, not endpoints)
- Horowitz, "Computing's Energy Problem," ISSCC 2014 — energy constants (W1, I2).
- Kalman 1960 — sufficient-statistic state (organizing principle).
- Landauer 1961 — thermodynamic erase floor (I3).
- S4/Mamba, RWKV, GLA, DeltaNet, mLSTM — bounded-state recurrence (Workspace / M1).
- NTM, DNC, Memory Networks, Product-Key Memory — external memory (Ledger / M2).
- RETRO, kNN-LM, RAG — nonparametric knowledge (Ledger).
- Hopfield / modern associative memory — capacity theory (M1 rank ceiling).
- ReRAM/PCM/SRAM compute-in-memory; CAM/TCAM — Fabric primitives (M4).
- Jamba/Griffin (attention–SSM hybrids) — evidence the §7 crux is live.
