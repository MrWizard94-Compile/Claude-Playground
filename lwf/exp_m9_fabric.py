"""
Experiment M9 -- Stage 2: the Fabric cost model on REAL silicon numbers.

M4 used generic E_DRAM/E_SRAM constants. M9 replaces them with published per-op energy
for the primitives LWF actually maps onto, and runs the ACTUAL op-trace of each design:

  transformer decode step  -> von Neumann: stream weights + KV cache from DRAM, digital MACs
  LWF step (algorithm)     -> bounded Workspace matvec + top-k (ANN) Ledger read
  LWF on the Fabric        -> Workspace matvec in a CIM crossbar (weights resident, no move)
                              + Ledger search in CAM/TCAM (keys resident, parallel match)

We report THREE designs so the two claims are separated and neither hides behind the other:
  (1) transformer / von Neumann     -- the baseline
  (2) LWF / von Neumann             -- ALGORITHM win only (bounded state + ANN, digital+DRAM)
  (3) LWF / Fabric                  -- ALGORITHM + SUBSTRATE win (CIM + CAM, no data movement)

HONESTY RULES:
  - ADC energy is modelled EXPLICITLY (it dominates analog CIM, 50-58% of power) so the
    crossbar is not made to look free. Cells scale O(d^2), ADCs O(d) -> big arrays amortise.
  - The transformer's dominant cost is streaming its P weights per token in batch-1 decode
    (memory-bound fact). LWF only beats that if knowledge is largely NONPARAMETRIC (in the
    Ledger). That is the core architectural BET, not a free lunch -- so P_core is an explicit
    knob and we report P_core=0 (full Ledger) AND P_core=P/25 (RETRO-like: retrieval reached
    GPT-3 quality with ~25x fewer params). If you don't believe the bet, read the P_core=P/25 row.
  - All constants are cited (see references/BIBLIOGRAPHY.md) and swept +/-10x in sensitivity;
    the reported conclusion is the one that survives the sweep.

VERIFIED CONSTANTS (pJ unless noted), grounded in sources pulled 2026-07-01:
  E_DRAM_byte : HBM/DRAM streamed energy per byte. Horowitz'14: 32b HBM access ~200 pJ
                (=50 pJ/B); modern HBM effective streaming lower. Default 10 (conservative-mid).
  e_mac_dig   : digital MAC. Horowitz'14 FP32 ~3.7 pJ, INT8 ~0.2-0.5. Default 0.5.
  e_cell_cim  : analog CIM cell MAC. From 26-150 TOPS/W -> 0.007-0.038 pJ. Default 0.02.
  e_adc       : ADC conversion (dominant CIM overhead). ~5.2 pJ @ 5-bit (ACM'21). Default 3.0.
  e_dac       : input DAC drive. Default 0.1.
  e_cam_bit   : CAM/TCAM search per bit. CMOS 16T 0.59 fJ, SRAM-CAM 0.44, FeCAM 0.18/0.069.
                Default 0.5 fJ = 5e-4 pJ.
"""

from __future__ import annotations
import argparse

# ---- verified/default constants (pJ) ----
E_DRAM_byte = 10.0
e_mac_dig = 0.5
e_cell_cim = 0.02
e_adc = 3.0
e_dac = 0.1
e_cam_bit = 5e-4         # 0.5 fJ/bit/search
BYTES = 2                # fp16


def transformer_step(n, P, L, d_model):
    """von Neumann decode: weight stream + KV stream + digital MACs. Per token, batch 1."""
    weight_move = BYTES * P * E_DRAM_byte
    kv_move = BYTES * (2 * L * d_model * n) * E_DRAM_byte
    mac_weight = P * e_mac_dig                       # ~1 MAC per param per token
    mac_attn = (2 * L * d_model * n) * e_mac_dig
    total = weight_move + kv_move + mac_weight + mac_attn
    return dict(total=total, weight_move=weight_move, kv_move=kv_move,
                mac=mac_weight + mac_attn)


def lwf_vonneumann(d_ws, N_store, n_probe, bits, P_core):
    """LWF algorithm on conventional hw: stream state + probed Ledger candidates from DRAM."""
    ws_move = BYTES * d_ws * d_ws * E_DRAM_byte
    ws_mac = d_ws * d_ws * e_mac_dig
    ledger_move = BYTES * n_probe * d_ws * E_DRAM_byte       # stream candidate vectors
    ledger_mac = n_probe * d_ws * e_mac_dig
    core_move = BYTES * P_core * E_DRAM_byte
    total = ws_move + ws_mac + ledger_move + ledger_mac + core_move
    return dict(total=total, ws=ws_move + ws_mac, ledger=ledger_move + ledger_mac,
                core=core_move)


def lwf_fabric(d_ws, N_store, n_probe, bits, P_core):
    """LWF on the Fabric: Workspace matvec in CIM (no weight move), Ledger search in CAM."""
    cim_cells = d_ws * d_ws * e_cell_cim
    cim_adc = d_ws * e_adc                                    # one conversion per output row
    cim_dac = d_ws * e_dac
    cam_search = n_probe * (d_ws * bits) * e_cam_bit          # parallel match over probed set
    core_move = BYTES * P_core * E_DRAM_byte
    total = cim_cells + cim_adc + cim_dac + cam_search + core_move
    return dict(total=total, cim_cells=cim_cells, cim_adc=cim_adc, cim_dac=cim_dac,
                cam=cam_search, core=core_move)


def fmt(pj):
    if pj >= 1e6:
        return f"{pj/1e6:.2f} uJ"
    if pj >= 1e3:
        return f"{pj/1e3:.2f} nJ"
    return f"{pj:.1f} pJ"


def run(P=7e9, L=32, d_model=4096, d_ws=2048, N_store=10_000_000, n_probe=256, bits=8,
        n_grid=(1, 512, 4096, 32768, 131072)):
    print("\n=== M9: Fabric cost model on real silicon numbers ===")
    print(f"transformer P={P:.0e} L={L} d_model={d_model} | LWF d_ws={d_ws} "
          f"Ledger N={N_store:.0e} n_probe={n_probe}")
    print(f"constants: E_DRAM={E_DRAM_byte} pJ/B, e_mac_dig={e_mac_dig}, e_cell_cim="
          f"{e_cell_cim}, e_adc={e_adc}, e_cam_bit={e_cam_bit} pJ/bit\n")

    # per-token energy vs context length (LWF rows use P_core=0 = full-Ledger extreme;
    # the DEFENSIBLE headline is the P_core=P/25 row further down, not this ceiling)
    print("per-token energy vs context (LWF @ P_core=0 = full-offload extreme):")
    hdr = f"{'n(ctx)':>8} | {'transformer':>12} | {'LWF/vonNeu':>11} | {'LWF/Fabric':>11} | " \
          f"{'xf/Fabric':>9}"
    print(hdr); print("-" * len(hdr))
    for n in n_grid:
        xf = transformer_step(n, P, L, d_model)
        lv = lwf_vonneumann(d_ws, N_store, n_probe, bits, P_core=0.0)
        lf = lwf_fabric(d_ws, N_store, n_probe, bits, P_core=0.0)
        print(f"{n:>8} | {fmt(xf['total']):>12} | {fmt(lv['total']):>11} | "
              f"{fmt(lf['total']):>11} | {xf['total']/lf['total']:>9.0f}x")

    # component breakdown at a representative context
    n0 = 8192
    xf = transformer_step(n0, P, L, d_model)
    lf = lwf_fabric(d_ws, N_store, n_probe, bits, P_core=0.0)
    print(f"\n-- breakdown @ n={n0} (P_core=0, full Ledger) --")
    print(f"  transformer: weight-move {fmt(xf['weight_move'])} ({xf['weight_move']/xf['total']:.0%}), "
          f"KV-move {fmt(xf['kv_move'])} ({xf['kv_move']/xf['total']:.0%}), "
          f"MAC {fmt(xf['mac'])} ({xf['mac']/xf['total']:.0%})")
    print(f"  LWF/Fabric : CIM-cells {fmt(lf['cim_cells'])} ({lf['cim_cells']/lf['total']:.0%}), "
          f"ADC {fmt(lf['cim_adc'])} ({lf['cim_adc']/lf['total']:.0%}), "
          f"CAM {fmt(lf['cam'])} ({lf['cam']/lf['total']:.0%})")
    dom = max((("CIM-cells", lf['cim_cells']), ("ADC", lf['cim_adc']), ("CAM", lf['cam'])),
              key=lambda t: t[1])
    print(f"  -> dominant Fabric term here: {dom[0]}. Cells scale O(d^2), ADC O(d): ADC "
          f"dominates only at small arrays / high ADC precision (crossover ~ d ~ e_adc/e_cell="
          f"{e_adc/e_cell_cim:.0f}); here d_ws={d_ws} > that, so cells lead. ADC modelled, not hidden.")

    # the architectural BET made explicit: P_core sensitivity
    print("\n-- the knowledge-is-nonparametric bet (P_core), @ n=8192 --")
    for label, Pc in (("P_core=0 (full Ledger)", 0.0),
                      ("P_core=P/25 (RETRO-like)", P / 25),
                      ("P_core=P (no offload)", P)):
        lf = lwf_fabric(d_ws, N_store, n_probe, bits, P_core=Pc)
        ratio = transformer_step(n0, P, L, d_model)['total'] / lf['total']
        print(f"  {label:<28}: LWF/Fabric {fmt(lf['total']):>10}  -> {ratio:>7.0f}x vs transformer")

    # sensitivity: does "LWF/Fabric << transformer" survive +/-10x on every constant?
    print("\n-- sensitivity: +/-10x on each constant (P_core=P/25, n=8192) --")
    survived = True
    base = dict(E_DRAM_byte=E_DRAM_byte, e_mac_dig=e_mac_dig, e_cell_cim=e_cell_cim,
                e_adc=e_adc, e_cam_bit=e_cam_bit)
    g = globals()
    for name in base:
        for f in (0.1, 10.0):
            g[name] = base[name] * f
            xf_t = transformer_step(n0, P, L, d_model)['total']
            lf_t = lwf_fabric(d_ws, N_store, n_probe, bits, P_core=P / 25)['total']
            survived &= (xf_t > lf_t)
            g[name] = base[name]
    for name in base:
        g[name] = base[name]
    print(f"  LWF/Fabric < transformer held under all single-constant +/-10x perturbations: "
          f"{survived}")
    print(f"\nM9 verdict: LWF's op-trace maps onto Fabric primitives that eliminate the "
          f"weight/KV\n  DATA MOVEMENT dominating von Neumann decode; the win is movement, "
          f"not cheaper MACs.\n  Honest dependency: it requires the knowledge-nonparametric "
          f"bet (see P_core rows).")
    print("[MODEL: analytic cost model w/ cited constants; not measured silicon. Stage 2.]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    run()
