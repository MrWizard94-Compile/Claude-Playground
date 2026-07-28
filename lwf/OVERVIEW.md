# LWF — Ledger / Workspace / Fabric

**A bounded-state cognitive architecture aimed at the von Neumann bottleneck for
intelligent computation.**

*Status: 20 falsifiable mechanisms (M1–M20). Reproducible, cited, CPU + GPU (GTX 1660 Ti,
48 GB host RAM). This document is the high-level map; per-run numbers and caveats live in
[`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md), the adversarial self-audit in [`GAPS.md`](GAPS.md),
the sibling-fleet analysis in [`references/SIBLING_FLEET.md`](references/SIBLING_FLEET.md).*

> **⚠ Read §0 first.** M1–M14 were built before a discipline reckoning; M15–M20 used
> pre-registered verdicts (5 straight came back FAIL/INCONCLUSIVE, reported as-is) and
> overturned several earlier framings. Do not read the green mechanism tables below without
> the corrections in §0 — some "wins" are narrower than they first appear.

---

## 0. Corrections & the honest state (from the M15–M20 pre-registered runs)

Four load-bearing beliefs from the optimistic early phase were corrected by later, disciplined work:

1. **Goalpost-moving → fixed.** Six early verdicts (M2/M8/M10/M12/M13/M14) were reframed *after*
   seeing results to reach "SUPPORTED." M15–M20 pre-register the verdict in code before running;
   all five came back FAIL/INCONCLUSIVE and were reported verbatim. Treat pre-M15 "SUPPORTED"
   labels with that caveat.
2. **Sibling "convergence" is NOT corroboration.** LWF is one of a *fleet* of same-brief agents
   (EP-GRM, TSAM, VeriForge, THCE) sharing one operator, doctrine, brief, and hardware. Their
   agreement on the invariants is a *shared prior*, not independent discovery (GAPS G3.2). Any
   invariant must be earned by measurement *in LWF*.
3. **"Fast-weight Workspace" ≡ linear attention** (Schlag 2021; confirmed empirically in M18,
   0.995 ≈ 0.996). So M6/M7 already used the *real* Workspace — the "stand-in" concern (G1.2) was
   false, and M7 is the genuine full-stack-vs-working-baseline result.
4. **The crux is STORAGE, not computation — and largely illusory for computation at reachable
   scale.** M19 (bits) and M20 (rank) both failed to make a bounded state fail on global reasoning:
   a *trained* rank-2 Workspace sorts 24 items at 0.95. The M1 rank ceiling governs *untrained
   arbitrary-association storage*, not *trained structured computation*. ⇒ **LWF's Workspace+Ledger
   split is justified by STORAGE (I1: arbitrary knowledge > capacity — genuinely confirmed by
   M2/M14), NOT by computational limits of bounded state.** The Ledger scales *knowledge*, not
   *reasoning the Workspace couldn't do*. M8's "H=3 ceiling" was optimization/depth, not capacity.

5. **The search angle (M21) — and a correction.** M21 showed LWF's split extends to search:
   executive **memory** stays frontier-bounded (Workspace ≈ 7, flat in n) with the
   assignment/trail/nogoods offloaded to the Ledger, while naive backtracking's *time* still blew
   up on hard instances. I first wrote that search cost is "orthogonal to the memory architecture /
   not LWF's to solve." **That was wrong** and is corrected here:
   - **Worst-case complexity** (NP-hardness) is not LWF's to solve — nor *any* architecture's. No
     architecture repeals complexity theory. Bounding memory does not change worst-case node counts.
   - **Practical / average search cost** IS reducible, by memory-based methods — nogoods,
     memoization, learned move-ordering, subsolution reuse — and those are **LWF-NATIVE (the Ledger).**
     M11 already showed it: nogood learning cut search nodes **8.9×**, and those nogoods live in the
     Ledger. So the Ledger is a first-class lever on practical search cost; I under-credited it.

**SESSION-LEVEL THESIS (corrected):** LWF (Workspace / Ledger / Fabric) is fundamentally a
**MEMORY + DATA-MOVEMENT architecture**, and — crucially — **memory is itself a lever on practical
computation** (memoization/nogoods/learned guidance trade memory for search time; the Ledger is that
memory). Validated wins: bounding executive memory (M1/M2, even for search M21), scaling knowledge
(M2/M14), cutting inference data movement (M4/M9), and reducing practical search cost via Ledger-stored
constraints (M11). What LWF does NOT do: change worst-case complexity, and it does not make a trained
bounded state "able to reason" (it already is — M19/M20). Honest value proposition: a knowledge-scaling,
low-data-movement substrate whose Ledger can also accelerate search — narrow, specific, evidence-backed.

---

## 1. Thesis in one paragraph

Modern "intelligent computation" (transformer inference) conflates **history** with
**state**: the working state needed to predict the next token *is* the entire context,
materialised as a KV cache that grows without bound and is streamed from memory every
step. Control theory has known since Kalman that the state you need for optimal future
action is a bounded **sufficient statistic** of history, not a log of it. LWF rebuilds
the stack around that principle: a fixed-size **Workspace** (the sufficient statistic), an
unbounded content-addressable **Ledger** (persistent knowledge + episodic memory, out of
the hot path), and a compute-in-memory **Fabric** (weights never move). The load-bearing
bet: **the algorithmic fix for unbounded state is the same shape as the physical fix for
data movement** — bounded recurrence maps to in-array matvec, content-addressable search
maps to in-array CAM — so software and hardware co-design reinforce rather than fight.

---

## 2. The problem — four distinct walls (not one)

"Von Neumann bottleneck" is imprecise. For intelligent computation it decomposes into:

| Wall | What it is | Kind |
|------|-----------|------|
| **W1** Physical data-movement tax | Off-chip access ≈ 130–700× a FLOP in energy (Horowitz ISSCC'14: 32b HBM ≈200 pJ vs FP MAC ≈1.5 pJ) | **fact** (of current substrate) |
| **W2** Bandwidth-bound decode | Batch-1 decode has arithmetic intensity ~1–2 FLOP/byte; streams all weights + KV per token | **fact** |
| **W3** State-is-a-log | Working state = full history (KV cache): O(n) memory, O(n²) compute, unbounded in task duration | **architectural choice**, not physics |
| **W4** Knowledge/cognition fusion | No first-class writable persistent store the cognition manipulates at inference | **architectural choice** |

W1/W2 are what everyone feels today; W3/W4 are the deeper structural sins LWF targets.

---

## 3. The guardrails — invariants (what the design may NOT promise)

| ID | Invariant | Consequence |
|----|-----------|-------------|
| **I1** | A B-bit state distinguishes ≤ 2^B histories | Exact recall of an unbounded set from fixed state is **impossible** → external memory is *mandatory*, not optional. Forces the three tiers. |
| **I2** | Off-chip movement ≫ on-chip compute (energy) | The physical win is in *not moving weights*, not cheaper MACs. |
| **I3** | Landauer floor: erase ≥ kT·ln2 ≈ 2.75 zJ @ 300 K | We run ~10⁶–10⁷ above it; reversibility is a distant lever, not magic. |
| **I4** | Fixed state must discard information (lossy) | The only question that matters: does it discard task-*irrelevant* history (survivable) or task-*relevant* (fatal)? Empirical per task. **The program's crux.** |

I1 and I4 are the important ones: they *derive* the three-tier split rather than assume it.

---

## 4. The architecture — three tiers

```
                 ┌───────────────────────────────────────────┐
   input ─────▶  │  WORKSPACE  (bounded executive memory)     │  ─────▶ action/output
                 │  fixed-size recurrent sufficient-statistic │
                 │  (gated fast-weight matrix); O(1) in task  │
                 │  length. Lossy on purpose. Holds the       │
                 │  FRONTIER (active bindings), not history.   │
                 └───────▲───────────────────────┬────────────┘
                  read (content-addressed)   write / evict (liveness-aware)
                         │                        ▼
                 ┌───────┴───────────────────────────────────┐
                 │  LEDGER  (persistent knowledge)            │
                 │  content-addressable (ANN / Hopfield read);│
                 │  stores associations AND nogoods (learned  │
                 │  constraints). Read cost O(log N), off the │
                 │  hot path. The exact-recall organ I1 forces.│
                 └───────────────────▲────────────────────────┘
                                     │  executes on
                 ┌───────────────────┴────────────────────────┐
                 │  FABRIC  (hardware abstraction)             │
                 │  compute-in-memory matvec (SRAM/ReRAM)      │
                 │  + associative search in CAM/TCAM.          │
                 │  Weights/keys resident → minimal movement.  │
                 └─────────────────────────────────────────────┘
```

- **Workspace** = the sufficient statistic. Bounded, differentiable, fast, lossy. Realised
  as a gated outer-product ("fast-weight") associative matrix `M`. Capacity is provably
  rank-bounded: `M = Σ vᵢkᵢᵀ` has rank ≤ d, so ≤ d exact associations. Holds the **active
  frontier**, not the whole problem. → design goals: bounded exec memory, cost ∝ working state.
- **Ledger** = persistent, content-addressable knowledge + episodic log + **nogoods**. Read
  = one-step modern-Hopfield / softmax retrieval (which *is* attention, per Ramsauer 2020);
  the exponential-capacity retrieval math lives here, where storing N patterns is allowed.
  → separates knowledge from cognition; is the audit log for verification.
- **Fabric** = where Workspace matvec (→ in-array MAC) and Ledger search (→ CAM match)
  happen without moving weights. → the data-movement win; the door to hardware co-design.

**Execution model** replaces fetch-decode-execute with a **cognitive step**:
`(Sₜ, xₜ, Rₜ) → (Sₜ₊₁, yₜ, Wₜ)`, `Rₜ = Ledger.read(query(Sₜ,xₜ))`. Fixed cost per step,
independent of how many steps came before.

**The cost law (established this program):**
> **LWF pays O(depth) sequential steps × O(frontier width) hot state.**
> Reasoning *depth* is free in state (M5). Frontier *width* costs state (M10). The Ledger
> absorbs any frontier that exceeds Workspace capacity; a liveness-aware controller keeps
> the right frontier resident (M13).

---

## 5. The mechanisms — 14 falsifiable claims

Each mechanism = hypothesis + mechanism + test + verdict. Full numbers in `LAB_NOTEBOOK.md`.

### Core information-routing (CPU, analytic/numeric — Stage 0)
| # | Claim | Verdict | Headline |
|---|-------|---------|----------|
| **M1** | Workspace holds ≤ d exact associations (rank ceiling) | ✅ SUPPORTED | exact to N=d, predictable decay past it |
| **M2** | External Ledger restores recall past the ceiling at fixed hot-state | ✅ SUPPORTED | **0.97 vs 0.02** @ N=2048, same 32 KB |
| **M3** | Bounded pure-function step ⇒ bit-exact replay + O(steps) audit log | ✅ SUPPORTED | replay/resume/tamper-evident; **253 B/step, flat** |
| **M4** | Transformer per-token energy grows with context; LWF flat | ✅ SUPPORTED | transformer **×5.9**, LWF **×1.00**; robust ±10× |
| **M5** | Deep composition at bounded per-step state (iterative retrieval) | ✅ SUPPORTED | **1.000 @ 8 hops on 512 B**; fixed-op → chance |

### The §7 crux — learned, does retrieval preserve capability? (torch)
| # | Claim | Verdict | Headline |
|---|-------|---------|----------|
| **M6** | Retrieval recovers recall bounded state loses (learned MQAR) | ✅ SUPPORTED (directional) | linear **0.97→0.66** as pressure grows; hybrid **→1.000 = attention** |
| **M7** | *Bounded-cost* (top-k) retrieval recovers recall | ✅ SUPPORTED (directional) | hybrid **1.000 at 8 reads/step** vs attention's 192; not dense-attention in disguise |
| **M8** | Bounded-cost retrieval *composes* multi-hop, not just recalls (GPU) | ✅ H=2 / ceiling H=3 | H=2: hybrid **0.344 = attn 0.346 ≫ linear 0.167**; H=3 a *shared* ceiling (attn fails too) |

### Hardware grounding + the nonparametric bet
| # | Claim | Verdict | Headline |
|---|-------|---------|----------|
| **M9** | Fabric primitives (CIM+CAM) eliminate the movement dominating decode | ✅ SUPPORTED (model) | transformer decode is **98% data movement**; **~33×** energy/token at RETRO-like P/25, **1× if knowledge isn't offloaded** (bet made explicit) |
| **M14** | Knowledge scales in the Ledger, not the model (M9's bet, learned) | ✅ SUPPORTED | fixed model **0.864→0.009 (chance)** as facts grow 1K→100K; small model + host-RAM Ledger **0.951 @ 100K, graceful to 0.786 @ 5M facts** (2.56 GB RAM), constant model cost |

### Cross-pollination from the sibling EP-GRM project
| # | Claim | Verdict | Headline |
|---|-------|---------|----------|
| **M10** | Workspace capacity tracks **frontier width F**, not total N | ✅ SUPPORTED | LWF recall flat in N (std 0.004), breaks past F≈d; Ledger absorbs overflow |
| **M11** | **Nogoods** (learned constraints) prune search, soundly | ✅ SUPPORTED | **8.9× fewer** search nodes, **12/12 sound** verdicts |
| **M12** | Controller quality: decay-aware admission tracks the frontier | ✅ SUPPORTED | belief-decay ≈ oracle at C≥F; beats un-aged LFU by **0.42** at C<F |
| **M13** | Liveness-aware controller = optimal at frontier-sized cache (real trace) | ✅ SUPPORTED | **= Belady optimum** at C≥MaxLive; needs dependency liveness, not just recency |

**How they interlock:** M1 sets the ceiling → M2 pays it off with the Ledger → M3 makes it
verifiable → M4/M9 show the energy win is *movement* → M5/M10 give the two-axis cost law
(depth × frontier) → M6/M7/M8 show it holds *learned*, at bounded read cost, into 2-hop
composition → M11/M12/M13 run the controller and constraint-learning off one shared
**dependency structure** → M14 shows the whole bet's premise — knowledge *scales in the store,
not the weights* — holds learned to millions of facts, closing the loop back to M9.

---

## 6. Relationship to the sibling project (EP-GRM)

A parallel autonomous-research project (`C:\WPAI\AI-Research\Grok_Playground`, agent: Grok) was given
the *same brief* and converged on a **different** architecture — EP-GRM, an *Explicit
Persistent Graph Rewrite Machine* (symbolic graph rewriting instead of neural association).
Two agents, one brief, **convergent invariants** (bounded executive state, locality/minimal
movement, explicit verification) via different substrates — strong evidence these are the
right decomposition, not artifacts of either design. Three concepts were imported and
validated in LWF (M10 frontier, M11 nogoods, M12/M13 controller). Full cross-analysis:
[`references/SIBLING_EPGRM.md`](references/SIBLING_EPGRM.md).

---

## 7. What is claimed — and what is NOT

**Claimed (with evidence):** bounded sufficient-statistic + content-addressable store routes
information with fixed hot-state and flat per-step energy; is verifiable (bit-exact replay +
learned-constraint pruning); its capacity tracks frontier width; a liveness-aware controller
manages it optimally at frontier-sized cache; and — learned, at toy-to-small scale — retrieval
recovers both recall (M6/M7) and 2-hop composition (M8) at bounded read cost.

**NOT claimed:** replacing transformers/CPUs/GPUs/OSes. Unbounded exact recall from bounded
state (I1 forbids it). Frontier-model-scale results (everything learned is toy/small on a
1660 Ti). That the co-design energy win holds without the nonparametric-knowledge bet (M9 shows
**1×** if knowledge isn't offloaded). That the parts are novel — bounded recurrence, external
memory, retrieval, CIM, nogoods are all prior art; the contributions are the *I1/I4-forced
split*, the *rank/frontier capacity law*, the *co-design isomorphism*, and the *verifiable,
liveness-managed step semantics*.

---

## 8. Open kill-shots (need more than this session/hardware)

1. **Scale/depth/language.** M8 reached 2-hop composition on a 1660 Ti; the "irreducibly
   global reasoning" question (wide frontier) needs a bigger GPU. M14 showed the
   knowledge-scaling half (a small model + host-RAM Ledger beats a fixed model to millions of
   facts) — but on *toy embeddings*, not language. A real language task where knowledge lives in
   the Ledger is still the crux that can falsify the program. (Note: on the 1660 Ti a *bigger*
   model is a dead end — slower per step AND needs more steps; scale needs better hardware.)
2. **Measured silicon.** M9 is a cost model; the analog-precision-vs-accuracy tradeoff (CIM
   costing *bits* of quality) is unmodelled — needs real hardware or a device-level simulator.
3. **Controller on a learned reasoning trace.** M13 used a synthetic computation DAG; the
   liveness-aware controller should be validated on a real learned cognitive trace.

---

## 9. Repository map

```
workspace.py                 Tier 1: gated fast-weight associative Workspace (bounded state)
ledger.py                    Tier 2: content-addressable store + Hopfield read + nogoods
verify.py                    M3 verifiability layer (checkpoint / bit-exact replay / audit trace)
exp_m1_capacity.py           M1  rank-capacity ceiling
exp_m2_ledger_recovery.py    M2  Ledger recovers recall at fixed hot-state
exp_m3_replay.py             M3  bit-exact replay + O(steps) audit log
exp_m4_cost_model.py         M4  data-movement/energy vs context length
exp_m5_multihop.py           M5  compositional multi-hop (depth axis)
exp_m6_crux_learned.py       M6  learned MQAR crux probe (torch)
exp_m7_scaleup.py            M7  bounded-cost (top-k) crux, hardware-agnostic
exp_m8_multihop_incontext.py M8  in-context multi-hop composition (GPU)
exp_m9_fabric.py             M9  Fabric cost model on real CIM/CAM silicon numbers
exp_m10_frontier.py          M10 frontier-width scaling (capacity ~ F not N)
exp_m11_nogood.py            M11 nogoods — learned-constraint verification
exp_m12_controller.py        M12 Workspace<->Ledger admission/eviction (synthetic drift)
exp_m13_live_frontier.py     M13 controller on a real computation frontier (vs Belady)
exp_m14_scaling_ledger.py    M14 knowledge scales in the Ledger, not the model (GPU + host RAM)
run_all.py                   driver for M1–M5 (+ optional figures/)

references/INVARIANTS.md     diagnosis, invariants, mechanism ledger
references/BIBLIOGRAPHY.md    annotated, source-verified citations + constants
references/SIBLING_EPGRM.md  cross-analysis vs the parallel EP-GRM project
LAB_NOTEBOOK.md              dated run log — every number, every caveat
CHANGELOG.md                 what changed and why
README.md                    quick-start + results summary
results_m6/7/8_full.txt      captured learned-run outputs
figures/                     M1/M2/M4 plots
```

---

## 10. How to run

```bash
python run_all.py            # M1–M5 non-training suite, prints falsifiable verdicts
python run_all.py --plots    # also writes figures/*.png

python exp_m6_crux_learned.py         # learned MQAR crux (CPU, torch)
python exp_m7_scaleup.py              # bounded-cost crux (auto CUDA/CPU)
python exp_m8_multihop_incontext.py   # in-context composition (auto-scales on GPU)
python exp_m9_fabric.py               # Fabric energy cost model
python exp_m10_frontier.py            # frontier-width law
python exp_m11_nogood.py              # nogood-learning search
python exp_m12_controller.py          # controller quality (synthetic)
python exp_m13_live_frontier.py       # controller on a real frontier (vs Belady)
python exp_m14_scaling_ledger.py      # knowledge scales in the Ledger (GPU + host-RAM store)
```
Deterministic (fixed seeds). Requires `numpy` (+ `matplotlib` for plots, `torch` for M6–M8).
GPU auto-detected: `torch 2.12.1+cu126` on a GTX 1660 Ti; falls back to CPU cleanly.

---

*Treat every architectural assumption — von Neumann, transformers, the memory hierarchy —
as an engineering solution to a forgotten cost objective, not a law. Nothing sacred;
everything testable.*
