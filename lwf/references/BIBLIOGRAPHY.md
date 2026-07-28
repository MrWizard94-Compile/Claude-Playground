# LWF — annotated bibliography & verified constants

Every entry is labeled with (a) what it establishes, (b) [FACT]/[EVIDENCE]/[METHOD],
and (c) which LWF tier/mechanism it grounds. Numbers here are pulled from sources this
session (2026-07-01), not from memory. Where an arXiv ID was directly confirmed in a
retrieved result it is given; otherwise the entry is cited by canonical title/venue so
it stays findable without asserting a possibly-wrong ID.

--------------------------------------------------------------------------------
## A. Physical energy / thermodynamics (Fabric, Invariants I2/I3)

**Horowitz, "Computing's Energy Problem (and what we can do about it)," ISSCC 2014.**
[FACT] The canonical energy table. Verified figures pulled this session:
  - 32-bit float MAC ≈ 1.5 pJ; float ops span 0.4–3.7 pJ by type/precision.
  - 32-bit access from off-chip HBM ≈ 200 pJ; 64-bit off-chip DRAM ≈ 1300–2600 pJ.
  - On-chip cache fetch ≈ 20 pJ.
  - Headline ratio: HBM move vs MAC ≈ **130×** for the same 32 bits.
  Grounds **Wall W1**, **Invariant I2**, and the M4 constant E_DRAM. Note our default
  E_DRAM=20 pJ/byte (=80 pJ/32b) is *below* the 200 pJ/32b HBM figure → conservative
  toward the transformer; DDR (~160 pJ/byte) would widen LWF's margin further.
  Mirror: https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf

**Landauer, "Irreversibility and Heat Generation in the Computing Process," IBM J. 1961.**
[FACT] Irreversible bit erase dissipates ≥ kT·ln2. At 300 K ≈ **2.75×10⁻²¹ J** (~10⁻²¹ J,
confirmed this session). Grounds **Invariant I3**; current logic runs ~10⁶–10⁷ above it.
Experimental confirmation: Bérut et al., Nature 2012 (Rutgers mirror in refs).

**HBM / DRAM data-movement energy (grounding pass 2026-07-02, feeds M22 `e_dram`).** [FACT]
  - **HBM3e ≈ 3.44 pJ/bit** total (≈27.5 pJ/byte); common HBM ≈ **7 pJ/bit** (≈56 pJ/byte).
  - HBM3 row-activation ≈ 0.18 pJ/bit; intra-die routing ≈ 0.2 pJ/bit/mm.
  - **Off-package DDR5 ≈ 80 pJ/bit** (≈640 pJ/byte) — ~10–20× worse than HBM.
  - M22 uses e_dram = 30 pJ/byte = 3.75 pJ/bit ≈ HBM3e / low end → UNDERSTATES transformer movement
    → conservative for LWF. Sources: HBM3e/HBM4 vendor data; NVIDIA "Energy-Efficient DRAM for GPUs"
    HPCA 2017; Horowitz ISSCC 2014.

**Measured LLM-inference energy per token (grounding pass 2026-07-02, validates M22's ABSOLUTE).**
  [FACT/EVIDENCE] Published measured H100 figures: **LLaMA-3.1-8B high-throughput ≈ 0.143 J/token**;
  **Llama-3.3-70B FP8, batch 128 ≈ 0.39 J/token**; older A100/V100 65B ≈ 3–4 J/token. Band used to
  validate M22: **0.14–0.39 J/token**. M22's batched 7B lands at **0.14 J/token @ n=8192** (in-band)
  and 0.039 @ n=2048/batch-128 (below-band = idealized movement lower bound, omits activation/overhead
  → ratios conservative). Sources: arXiv:2310.03003 ("Benchmarking Energy Costs of LLM Inference"),
  arXiv:2407.16893 ("The Price of Prompting"), llm-tracker.info, TokenPowerBench (arXiv:2512.03024).
  → M22 is GROUNDED: same order as measured reality, on the conservative side.

--------------------------------------------------------------------------------
## B. Bounded-state recurrence (Workspace, Mechanism M1)

**Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv:2312.00752.**
[EVIDENCE] Fixed-size selective state, linear-time. Establishes the bounded-state
regime the Workspace lives in. Known limitation (below) is the whole reason the Ledger exists.

**Dao & Gu, "Transformers are SSMs: ... Structured State Space Duality (Mamba-2)," arXiv:2405.21060.**
[FACT/METHOD] Formal duality: attention and SSMs are two views of one operation. Directly
supports the LWF claim that "state vs log" is a *choice*, not a divide.

**Yang et al., "Parallelizing Linear Transformers with the Delta Rule over Sequence Length"
(DeltaNet), arXiv:2406.06484, NeurIPS 2024.**
[EVIDENCE] Delta rule > additive linear-attention on associative recall; "given a fixed-size
recurrent state, the delta rule achieves a better frontier of the recall-memory tradeoff."
CAVEAT for our M1: this advantage is realized under *learned* gating/normalization. Our
exp_m1 (untrained, unit learning-rate, single pass) finds plain Hebbian more graceful on
*correlated* keys — consistent, because the frontier gain needs the learned machinery.

**Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule," arXiv:2412.06464.**
[EVIDENCE] Gating + delta rule; state-of-the-art recall-memory frontier for fixed state.
Reference implementation for a production Workspace update.

**Arora et al., "Zoology: Measuring and Improving Recall in Efficient Language Models," ICLR 2024.**
[EVIDENCE] Introduces **MQAR** (multi-query associative recall) — the synthetic task that
isolates fixed-state recall capacity. Basis for our exp_m6 learned crux probe.

**Arora et al., "Based: Simple Linear Attention Language Models Balance the Recall-Throughput
Tradeoff," ICML 2024.** [EVIDENCE] Quantifies recall vs state-size; motivates hybridization.

--------------------------------------------------------------------------------
## C. Associative-memory capacity theory (M1 ceiling, Ledger read)

**Ramsauer et al., "Hopfield Networks is All You Need," arXiv:2008.02217, ICLR 2021.**
[FACT/METHOD] Continuous modern Hopfield net stores **exponentially many** patterns
(~2^(N/2) in the associative-space dimension), retrieves in **one update** with
exponentially small error — and its update rule **is the transformer attention mechanism.**
KEY for LWF: the **Ledger read = one-step modern-Hopfield retrieval**; softmax-attention is
Hopfield lookup over context. Exponential capacity still requires *storing* the N patterns
(O(N·d)) → does NOT violate I1 (bounded *state*), it just says the retrieval math is cheap
and error-tolerant *given* the store. Implemented as `Ledger.read_hopfield`.

**Classical Hopfield capacity ≈ 0.138·N patterns (Amit–Gutfreund–Sompolinsky).**
[FACT] Linear-in-N ceiling for the classical outer-product associative memory — the
theoretical sibling of our rank-≤d Workspace bound (exp_m1). Modern Hopfield breaks this
via stronger nonlinearity, which is precisely the classical-Workspace vs Hopfield-Ledger split.

--------------------------------------------------------------------------------
## D. External / content-addressable memory (Ledger, Mechanism M2)

**Graves et al., "Neural Turing Machines," arXiv:2014; "Hybrid computing using a neural
network with dynamic external memory" (Differentiable Neural Computer), Nature 2016.**
[METHOD] The原型 of a controller + differentiable content-addressable external store.
Direct ancestor of Workspace(controller)+Ledger(store). Known issue: unstable addressing /
hard to train — LWF sidesteps by using a *non-differentiable* ANN index for hard retrieval
and a differentiable Hopfield read only for the soft path.

**Lample et al., "Large Memory Layers with Product Keys," NeurIPS 2019.**
[METHOD] Sub-linear ( ~√N ) key lookup into a huge learned memory — engineering proof that
a large external store can be read at scale without O(N) cost. Grounds the Ledger's O(log N)
per-step claim.

**Borgeaud et al., "Improving Language Models by Retrieving from Trillions of Tokens" (RETRO),
DeepMind 2021/2022.** [EVIDENCE] Frozen retriever + chunked cross-attention over a 2T-token
DB reaches GPT-3-class quality with **25× fewer parameters** — direct evidence for offloading
knowledge to a nonparametric store (Goal #4). The Ledger is the native version of this.

**Khandelwal et al., "Generalization through Memorization: Nearest Neighbor Language Models"
(kNN-LM), ICLR 2020.** [EVIDENCE] Nonparametric kNN cache over past contexts improves LMs
with no retraining — the Ledger's read primitive in prior art.

**"Memory-Augmented Transformers: A Systematic Review," arXiv:2508.10824.**
[REFERENCE] Recent survey mapping the whole memory-augmentation design space; use as an
index into the field when scoping Stage 0.5+.

--------------------------------------------------------------------------------
## E. Compute-in-memory / associative hardware (Fabric, Mechanism M4)

**Wan et al., "A compute-in-memory chip based on resistive random-access memory," Nature 2022
(s41586-022-04992-8).** [FACT] Fully-integrated RRAM CIM performing MACs in-array; weights
never move. Existence proof for the Fabric's Workspace-matvec-in-array primitive.

**RRAM CIM binary MVM, arXiv:2501.10702; analog CIM macros reporting 26–150 TOPS/W** (e.g.
40nm binary macro 26.56 TOPS/W; charge-domain ACIM ~123.8 TOPS/W; up to ~150 TOPS/W with
stochastic binarization). [EVIDENCE] Order-of-magnitude energy headroom vs von-Neumann MAC.
Feeds a future Stage-2 per-op Fabric cost model.

**Content-Addressable Memory (CAM/TCAM) for in-array associative search.** [METHOD] Parallel
in-memory match = the Ledger's hardware read primitive; analog CAM can even approximate
softmax (the Hopfield read) in-array. Ties the Ledger to silicon.

**CAM/TCAM search energy (verified 2026-07-01, feeds M9 `e_cam_bit`).** [FACT]
  - CMOS 16T CAM array search: **0.590 fJ/bit**.
  - SRAM-based BCAM/TCAM: **0.44 fJ/bit**.
  - FeCAM (ferroelectric): **0.182 fJ/bit** digital, **0.069 fJ/bit** analog 3-bit (3.2–8.6×
    below CMOS TCAM). arXiv:2004.01866.
  - FeSQUID cryogenic TCAM: 1.36 aJ/bit (exotic; excluded from the model).
  M9 default e_cam_bit = 0.5 fJ/bit (CMOS, conservative).

**ADC overhead in analog CIM (verified 2026-07-01, feeds M9 `e_adc`).** [FACT] ADCs/DACs are
the dominant peripheral cost: **>50% of power** in RRAM macros, up to **58% energy / 81% area**;
energy grows exponentially with precision; a 32-level (5-bit) ADC ≈ **5.2 pJ** (ACM TODAES 2021,
voltage-controlled ADC). Cells scale O(N²), ADCs O(N) → large arrays amortise peripheral cost.
M9 models e_adc explicitly (default 3.0 pJ) so the crossbar is not made to look free.

**Analog CIM MAC energy (verified, feeds M9 `e_cell_cim`).** [FACT/EVIDENCE] Sub-fJ per cell
MAC; 26–150 TOPS/W across recent macros → **0.007–0.038 pJ/MAC** (cell only). M9 default
e_cell_cim = 0.02 pJ. Precision-limited to 3–8 bits by device variability/nonlinearity/IR-drop.

**"Memory Is All You Need: CIM Architectures for Accelerating LLM Inference," arXiv:2406.08413.**
[REFERENCE] Survey framing CIM specifically for the decode-time memory-bandwidth wall (M4/M9).

--------------------------------------------------------------------------------
## F. Hybrids & the §7 crux (why the crux is live, not settled)

**Jamba (Lieber et al., 2024); Griffin (De et al., 2024); B'MOJO (arXiv:2407.06324).**
[EVIDENCE] Production systems re-inject a *few* attention layers into SSM stacks specifically
to recover recall/global reasoning. This is the empirical shadow of the §7 crux: fixed state
alone loses something; the open question LWF bets on is whether an *external content-addressed
store* (rather than re-added dense attention) fully recovers it. exp_m6 probes this at toy scale.

**Mimetic Initialization (arXiv:2410.11135).** [EVIDENCE] SSMs *can* learn to recall/copy when
initialized to mimic attention — evidence the capability gap is partly optimization, not just
representational. Relevant to interpreting exp_m6 negative results (is it capacity or training?).

--------------------------------------------------------------------------------
## G. Foundational framing

**Kalman, "A New Approach to Linear Filtering and Prediction Problems," 1960.** [FACT]
The state as a *sufficient statistic* of history — the organizing principle of the Workspace.

**von Neumann, "First Draft of a Report on the EDVAC," 1945.** [REFERENCE] The original
stored-program tradeoff LWF treats as an engineering choice, not a law (project thesis).
