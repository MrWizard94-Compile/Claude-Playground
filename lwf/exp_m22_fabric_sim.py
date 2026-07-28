"""
Experiment M22 -- Fabric simulator (supersedes M9's single-number spreadsheet).

M9 modelled energy = bytes x pJ/byte. That hid the two things that actually decide whether the
Fabric wins: (1) the ADC/DAC peripheral cost of analog compute-in-memory (which the literature says
DOMINATES), and (2) latency/throughput (the von Neumann bottleneck is a BANDWIDTH bottleneck, not
just an energy one). M22 is a component-level, tiled, energy+latency model that runs the ACTUAL
op-trace of each design:

  transformer decode step  -> von Neumann: stream ALL weights + KV cache from DRAM, digital MACs.
  LWF step / von Neumann   -> bounded Workspace matvec + Ledger read, but digital + DRAM.
  LWF step / Fabric        -> Workspace matvec in a CIM crossbar (weights resident, analog MAC +
                              ADC readout) + Ledger search in CAM/TCAM (keys resident, parallel match).

It reports per-token ENERGY and LATENCY with a component breakdown, a PRECISION sweep (ADC bits ->
energy, the accuracy/energy knob), and a +/-sensitivity pass. This is a HIGHER-FIDELITY MODEL, not
measured silicon -- it exposes where the cost and uncertainty actually live (ADC, precision, the
nonparametric-knowledge assumption), so it can tell us whether real hardware is worth building.

CONSTANTS (energy pJ, latency ns) -- published typicals, cited; swept in sensitivity:
  e_cell    0.02 pJ/MAC     analog CIM cell (26-150 TOPS/W; Wan et al., Nature 2022 s41586-022-04992-8)
  e_adc/bit 1.0 pJ/bit      SAR ADC ~ linear in resolution (~5-6 pJ @ 5-6 bit; ACM TODAES 2021)
  e_dac     0.1 pJ/input    input driver
  e_cam_bit 5e-4 pJ/bit     CMOS 16T CAM search 0.59 fJ/bit; SRAM-CAM 0.44 fJ/bit
  e_dmac    0.5 pJ/MAC      digital INT8-ish MAC (Horowitz ISSCC'14: 0.9-3.7 pJ FP; lower for INT)
  e_dram    30 pJ/byte      HBM streamed = 3.75 pJ/bit. GROUNDED (lit review 2026-07-02): HBM3e
                            measured ~3.44 pJ/bit (=27.5 pJ/B); common HBM ~7 pJ/bit (=56 pJ/B);
                            off-package DDR5 ~80 pJ/bit (=640 pJ/B). 30 pJ/B is at the HBM3e/low end
                            -> UNDERSTATES transformer movement -> conservative for LWF. Refs:
                            Horowitz ISSCC'14; HBM3e/HBM4 vendor data; NVIDIA DRAM-energy HPCA'17.
  e_sram    0.1 pJ/byte     on-chip SRAM
  HBM_BW    2e12 B/s        modern HBM bandwidth (latency = bytes / BW, i.e. bandwidth-bound)
  t_settle  5 ns  | t_adc 10 ns/conv (pipelined over accum) | t_cam 2 ns
"""

from __future__ import annotations
import argparse

C = dict(e_cell=0.02, e_adc_per_bit=1.0, e_dac=0.1, e_cam_bit=5e-4, e_dmac=0.5,
         e_dram=30.0, e_sram=0.1, HBM_BW=2e12, t_settle=5.0, t_adc=10.0, t_cam=2.0)
BYTES = 2                     # fp16
TILE = 128                   # CIM crossbar tile dimension (128x128)


def dram_latency_ns(nbytes):
    return nbytes / C["HBM_BW"] * 1e9


def transformer_step(n, P, L, d_model, batch=1):
    """von Neumann decode, PER TOKEN. Batching amortizes the WEIGHT stream over `batch` sequences
    (one weight load serves B tokens) but NOT the KV cache (each sequence has its own, size n).
    So per-token weight energy = 2P*e_dram / batch; KV + MAC per-token are batch-independent. This
    is the honest baseline: at batch 1 the transformer is weight-move-dominated; at serving batch
    sizes the weight term amortizes and the (context-growing, non-amortizable) KV term dominates."""
    w_bytes = BYTES * P
    kv_bytes = BYTES * (2 * L * d_model * n)               # K and V for n tokens, per sequence
    e_weight = w_bytes * C["e_dram"] / batch               # amortized across the batch
    e_kv = kv_bytes * C["e_dram"]                           # NOT amortized (per-sequence)
    e_mac = (P + 2 * L * d_model * n) * C["e_dmac"]
    lat = dram_latency_ns(w_bytes / batch + kv_bytes)      # bandwidth-bound, per token
    return dict(total=e_weight + e_kv + e_mac, latency=lat,
                comp=dict(weight_move=e_weight, kv_move=e_kv, mac=e_mac))


def lwf_vonneumann_step(d_ws, N_store, n_probe, bits, P_core):
    """Same LWF algorithm on conventional hw: state + probed Ledger candidates stream from DRAM."""
    ws_move = BYTES * d_ws * d_ws
    ledger_move = BYTES * n_probe * d_ws
    core_move = BYTES * P_core
    move = ws_move + ledger_move + core_move
    e_mac = (d_ws * d_ws + n_probe * d_ws) * C["e_dmac"]
    return dict(total=move * C["e_dram"] + e_mac, latency=dram_latency_ns(move),
                comp=dict(move=move * C["e_dram"], mac=e_mac))


def lwf_fabric_step(d_ws, N_store, n_probe, bits, P_core):
    """Workspace matvec in a CIM crossbar (resident weights, analog MAC + ADC) + CAM Ledger search."""
    tiles_c = -(-d_ws // TILE)                             # ceil: accumulation groups along inputs
    cim_cells = d_ws * d_ws * C["e_cell"]                  # O(d^2) analog MACs, no weight movement
    cim_adc = d_ws * tiles_c * (bits * C["e_adc_per_bit"])  # one ADC per output row per tile-group
    cim_dac = d_ws * C["e_dac"]
    core_cells = P_core * C["e_cell"]                      # nonparametric-knowledge core, if any (resident)
    core_adc = (P_core / d_ws if P_core else 0) * (bits * C["e_adc_per_bit"])
    cam = n_probe * (d_ws * bits) * C["e_cam_bit"]         # parallel in-array match, no key movement
    fetch = BYTES * n_probe * d_ws * C["e_dram"]           # only the retrieved records move
    total = cim_cells + cim_adc + cim_dac + core_cells + core_adc + cam + fetch
    lat = C["t_settle"] + C["t_adc"] * tiles_c + C["t_cam"] + dram_latency_ns(BYTES * n_probe * d_ws)
    return dict(total=total, latency=lat,
                comp=dict(cim_cells=cim_cells, cim_adc=cim_adc + core_adc, cim_dac=cim_dac,
                          cam=cam, fetch=fetch, core_cells=core_cells))


def fmt_e(pj):
    return f"{pj/1e6:.2f} uJ" if pj >= 1e6 else (f"{pj/1e3:.2f} nJ" if pj >= 1e3 else f"{pj:.1f} pJ")


def fmt_t(ns):
    return f"{ns/1e6:.3f} ms" if ns >= 1e6 else (f"{ns/1e3:.2f} us" if ns >= 1e3 else f"{ns:.1f} ns")


def run(P=7e9, L=32, d_model=4096, d_ws=2048, N_store=10_000_000, n_probe=256, bits=6,
        P_core=0.0, n_grid=(1, 4096, 32768, 131072)):
    print("\n=== M22: Fabric simulator (component-level energy + latency; supersedes M9) ===")
    print(f"transformer P={P:.0e} L={L} d_model={d_model} | LWF d_ws={d_ws} N={N_store:.0e} "
          f"n_probe={n_probe} adc={bits}b | P_core={P_core:.0e}")
    print("[HIGHER-FIDELITY MODEL, not measured silicon. Adds ADC/DAC + latency + tiling over M9.]\n")

    hdr = f"{'n(ctx)':>8} | {'xformer E':>11} {'xformer t':>10} | {'Fabric E':>10} {'Fabric t':>9} " \
          f"| {'E x':>7} {'t x':>8}"
    print(hdr); print("-" * len(hdr))
    for n in n_grid:
        xf = transformer_step(n, P, L, d_model)
        lf = lwf_fabric_step(d_ws, N_store, n_probe, bits, P_core)
        print(f"{n:>8} | {fmt_e(xf['total']):>11} {fmt_t(xf['latency']):>10} | "
              f"{fmt_e(lf['total']):>10} {fmt_t(lf['latency']):>9} | "
              f"{xf['total']/lf['total']:>6.0f}x {xf['latency']/lf['latency']:>7.0f}x")

    # HONEST BASELINE (per doctrine s15: skepticism on extraordinary claims). Batch-1 decode streams
    # all weights per token -> flatters LWF. Real serving BATCHES, amortizing the weight stream (but
    # NOT the per-sequence KV cache). Show LWF vs a BATCHED transformer so the win isn't overstated.
    Bserve = 64
    print(f"\n-- vs BATCHED serving transformer (batch={Bserve}; weights amortized, KV not) --")
    print(f"  {'n(ctx)':>8} | {'xf batch=1':>11} {'xf batch=%d' % Bserve:>12} | {'Fabric E':>10} | "
          f"{'LWF vs b1':>9} {'LWF vs b%d' % Bserve:>10}")
    for n in (512, 8192, 131072):
        x1 = transformer_step(n, P, L, d_model, batch=1)['total']
        xb = transformer_step(n, P, L, d_model, batch=Bserve)['total']
        lfe = lwf_fabric_step(d_ws, N_store, n_probe, bits, P_core)['total']
        print(f"  {n:>8} | {fmt_e(x1):>11} {fmt_e(xb):>12} | {fmt_e(lfe):>10} | "
              f"{x1/lfe:>8.0f}x {xb/lfe:>9.0f}x")
    print("  -> batching amortizes the WEIGHT stream (shrinks LWF's SHORT-context lead) but NOT the "
          "KV\n     cache (per-sequence, grows with context) -> LWF's win is specifically the "
          "CONTEXT/KV-scaling\n     term, which batching does not fix. Report ratios vs BATCHED, not "
          "batch-1, as the honest baseline.")

    n0 = 8192
    xf = transformer_step(n0, P, L, d_model)
    lf = lwf_fabric_step(d_ws, N_store, n_probe, bits, P_core)
    print(f"\n-- component breakdown @ n={n0} (P_core={P_core:.0e}) --")
    print(f"  transformer: " + ", ".join(f"{k} {fmt_e(v)} ({v/xf['total']:.0%})"
                                          for k, v in xf['comp'].items()))
    print(f"  LWF/Fabric : " + ", ".join(f"{k} {fmt_e(v)} ({v/lf['total']:.0%})"
                                          for k, v in lf['comp'].items() if v > 0))
    dom = max(lf['comp'].items(), key=lambda kv: kv[1])
    print(f"  -> dominant Fabric term: {dom[0]} ({fmt_e(dom[1])}). "
          f"Transformer is {xf['comp']['weight_move']/xf['total']:.0%} weight-move + "
          f"{xf['comp']['kv_move']/xf['total']:.0%} KV-move = pure DATA MOVEMENT (the physical tax).")

    # ALGORITHM vs SUBSTRATE decomposition: is the win from bounded-state+retrieval, or from analog CIM?
    lv = lwf_vonneumann_step(d_ws, N_store, n_probe, bits, P_core)
    print(f"\n-- algorithm vs substrate (energy @ n={n0}, P_core={P_core:.0e}) --")
    print(f"  transformer / von Neumann : {fmt_e(xf['total']):>10}")
    print(f"  LWF / von Neumann (digital): {fmt_e(lv['total']):>10}  "
          f"(algorithm win = no 7B weight stream): {xf['total']/lv['total']:.0f}x")
    print(f"  LWF / Fabric (CIM+CAM)    : {fmt_e(lf['total']):>10}  "
          f"(substrate win = Workspace resident, not streamed): {lv['total']/lf['total']:.0f}x more")
    print(f"  -> the ALGORITHM (bounded state + retrieval) does most of it; the CIM SUBSTRATE adds a "
          f"further ~{lv['total']/lf['total']:.0f}x by keeping the Workspace resident. Residual floor = Ledger fetch.")

    # the nonparametric-knowledge bet, made explicit (as in M9/M14)
    print(f"\n-- knowledge-is-nonparametric bet (P_core), energy @ n={n0} --")
    for lbl, Pc in (("P_core=0 (full Ledger)", 0.0), ("P_core=P/25 (RETRO-like)", P / 25),
                    ("P_core=P (no offload)", P)):
        lf2 = lwf_fabric_step(d_ws, N_store, n_probe, bits, Pc)
        print(f"  {lbl:<28}: {fmt_e(lf2['total']):>10}  -> {xf['total']/lf2['total']:>7.0f}x vs transformer")

    # ADC precision sweep -- the analog accuracy/energy knob (honest: higher bits = more energy)
    print("\n-- ADC precision sweep (Fabric energy @ n=8192; higher bits -> more energy, better accuracy) --")
    for b in (2, 4, 6, 8, 10):
        lfb = lwf_fabric_step(d_ws, N_store, n_probe, b, P_core)
        adc_frac = lfb['comp']['cim_adc'] / lfb['total']
        print(f"  {b}-bit ADC: {fmt_e(lfb['total']):>10} (ADC = {adc_frac:.0%} of Fabric energy)")

    # sensitivity: does "Fabric << transformer (energy)" survive +/-10x on each constant? (P_core=P/25)
    print("\n-- sensitivity: +/-10x each constant (P_core=P/25, n=8192) --")
    survived = True
    base = {k: C[k] for k in ("e_cell", "e_adc_per_bit", "e_cam_bit", "e_dram")}
    for name in base:
        for f in (0.1, 10.0):
            C[name] = base[name] * f
            xt = transformer_step(n0, P, L, d_model)['total']
            lt = lwf_fabric_step(d_ws, N_store, n_probe, bits, P / 25)['total']
            survived &= (xt > lt)
            C[name] = base[name]
    print(f"  Fabric < transformer held under all single-constant +/-10x perturbations: {survived}")
    # VALIDATION vs published MEASURED LLM-inference energy (doctrine s3: validate against real data).
    print("\n-- ABSOLUTE VALIDATION vs published measured GPU inference (J/token) --")
    PUB_LO, PUB_HI = 0.14, 0.39   # H100: LLaMA-3.1-8B high-throughput 0.143; 70B FP8 batch128 0.39
    for lbl, b, n in (("batch=1 @ n=8192", 1, 8192), ("batch=64 @ n=8192", 64, 8192),
                      ("batch=128 @ n=2048", 128, 2048)):
        j_per_tok = transformer_step(n, P, L, d_model, batch=b)['total'] / 1e12  # pJ -> J
        print(f"  M22 transformer {lbl:<22}: {j_per_tok*1e3:>7.1f} mJ/tok ({j_per_tok:.3f} J/tok)")
    print(f"  published measured band (H100, 7-70B, FP8/high-throughput): {PUB_LO}-{PUB_HI} J/tok "
          f"[arXiv:2310.03003, arXiv:2407.16893, llm-tracker]")
    jb = transformer_step(8192, P, L, d_model, batch=64)['total'] / 1e12
    verdict = "SAME ORDER (idealized lower bound; omits activation/overhead -> ratios conservative)" \
        if 0.02 <= jb <= PUB_HI * 2 else "OUT OF RANGE -- model likely miscalibrated"
    print(f"  -> M22 batched ~{jb:.2f} J/tok vs published {PUB_LO}-{PUB_HI}: {verdict}")

    print("\nHONEST CAVEATS: (1) constants are published typicals, not a measured chip. (2) analog CIM "
          "precision (2-8 bit) may COST task accuracy -- modelled here only as ADC energy, NOT as an\n"
          "accuracy hit (that needs functional simulation / silicon). (3) the win requires the "
          "nonparametric-knowledge bet (see P_core rows). This model informs whether to BUILD, not proof.")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()
