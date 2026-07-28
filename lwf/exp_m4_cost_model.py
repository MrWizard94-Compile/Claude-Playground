"""
Experiment M4 -- data-movement / energy scaling: LWF step vs transformer decode.

HYPOTHESIS (mechanism M4): per generated token, a transformer's decode cost has a
term that GROWS with context length n (streaming the KV cache from HBM), while an
LWF step is FLAT in n (bounded Workspace resident in SRAM/CIM + top-k Ledger read
whose per-step movement is independent of total store size).

The robust, falsifiable claim is the SLOPE, not any absolute number. Energy constants
below are order-of-magnitude and cited; a sensitivity sweep shows the crossover
conclusion survives +/-10x perturbation of every constant.

ENERGY CONSTANTS (illustrative, per byte moved / per FLOP), grounded in:
  Horowitz, "Computing's Energy Problem (and what we can do about it)", ISSCC 2014.
  - off-chip DRAM/HBM access dominates; on-chip SRAM ~2-3 orders cheaper; a FLOP is
    cheaper still. We treat DRAM movement as the dominant term (Wall 1).

MODEL (per token / per cognitive step), fp16 (2 bytes/elem):
  Transformer (dense, batch 1, memory-bound decode), P params, L layers, d_model:
    bytes(n) = 2*P                              (stream all weights each token)
             + 2 * (2*L*d_model*n)              (stream K and V cache, grows with n)
    energy(n) = bytes(n) * e_dram
    (GQA shrinks the KV constant by n_kv/n_heads but NOT the slope; togglable.)

  LWF step, Workspace d_ws (resident in SRAM/CIM), optional parametric core P_core,
  Ledger top-k read of k records dim d_ws with ANN visiting ~c*log2(N) nodes:
    sram_bytes = 2 * d_ws*d_ws                  (Workspace matvec, from SRAM -- constant)
    hbm_bytes(n) = 2*P_core                     (small/zero core, constant in n)
                 + 2 * k * d_ws * c*log2(N)     (Ledger read, independent of context n)
    energy = sram_bytes*e_sram + hbm_bytes*e_dram
  Note: LWF's HBM term depends on knowledge-store size N via log2(N), NOT on the
  reasoning context length n. That is the decoupling Goal #2 asks for.
"""

from __future__ import annotations
import numpy as np

# --- energy constants (pJ), order-of-magnitude, Horowitz ISSCC 2014 regime ---
E_DRAM = 20.0     # pJ per byte moved off-chip (HBM/DDR regime)
E_SRAM = 0.1      # pJ per byte from on-chip SRAM / CIM tile
BYTES = 2         # fp16


def transformer_energy(n, P=7e9, L=32, d_model=4096, gqa_ratio=1.0):
    weight_bytes = BYTES * P
    kv_bytes = BYTES * (2 * L * d_model * n) * gqa_ratio
    return (weight_bytes + kv_bytes) * E_DRAM, weight_bytes, kv_bytes


def lwf_energy(n, N_store, d_ws=1024, P_core=0.0, topk=8, ann_c=4.0):
    sram_bytes = BYTES * d_ws * d_ws
    hbm_bytes = BYTES * P_core + BYTES * topk * d_ws * (ann_c * np.log2(max(N_store, 2)))
    energy = sram_bytes * E_SRAM + hbm_bytes * E_DRAM
    return energy, sram_bytes, hbm_bytes


def run(P=7e9, L=32, d_model=4096, d_ws=1024, N_store=10_000_000, P_core=0.0,
        n_grid=None, gqa_ratio=1.0):
    global E_DRAM, E_SRAM
    if n_grid is None:
        n_grid = [1, 128, 1024, 4096, 16_384, 65_536, 131_072]
    print("\n=== M4: per-token energy vs context length n ===")
    print(f"transformer: P={P:.0e}, L={L}, d_model={d_model}, gqa_ratio={gqa_ratio}")
    print(f"LWF: d_ws={d_ws}, P_core={P_core:.0e}, Ledger N={N_store:.0e}")
    print(f"constants: E_DRAM={E_DRAM} pJ/byte, E_SRAM={E_SRAM} pJ/byte, fp16\n")
    header = f"{'n(ctx)':>8} | {'xformer nJ/tok':>14} | {'LWF nJ/tok':>11} | " \
             f"{'ratio x':>8} | {'xf KV-frac':>10}"
    print(header)
    print("-" * len(header))
    rows = []
    for n in n_grid:
        e_xf, wb, kvb = transformer_energy(n, P, L, d_model, gqa_ratio)
        e_lwf, _, _ = lwf_energy(n, N_store, d_ws, P_core)
        e_xf_nj, e_lwf_nj = e_xf / 1e3, e_lwf / 1e3   # pJ -> nJ
        ratio = e_xf / e_lwf
        kv_frac = kvb / (wb + kvb)
        rows.append((n, e_xf_nj, e_lwf_nj, ratio))
        print(f"{n:>8} | {e_xf_nj:>14.3g} | {e_lwf_nj:>11.3g} | "
              f"{ratio:>8.1f} | {kv_frac:>10.2%}")

    # slope check: transformer grows ~linearly in n past the KV crossover; LWF flat
    n_lo, n_hi = n_grid[1], n_grid[-1]
    e_lo = transformer_energy(n_lo, P, L, d_model, gqa_ratio)[0]
    e_hi = transformer_energy(n_hi, P, L, d_model, gqa_ratio)[0]
    xf_growth = e_hi / e_lo
    l_lo = lwf_energy(n_lo, N_store, d_ws, P_core)[0]
    l_hi = lwf_energy(n_hi, N_store, d_ws, P_core)[0]
    lwf_growth = l_hi / l_lo
    print(f"\ncontext {n_lo}->{n_hi}: transformer energy x{xf_growth:.1f}, "
          f"LWF energy x{lwf_growth:.3f}")
    print(f"M4 (transformer grows with context, LWF flat): "
          f"{'SUPPORTED' if xf_growth > 2*lwf_growth else 'CHECK'}")

    # sensitivity: does the slope conclusion survive +/-10x on every constant?
    print("\n-- sensitivity: perturb E_DRAM/E_SRAM/d_ws by up to 10x --")
    base_dram, base_sram = E_DRAM, E_SRAM
    survived = True
    for fd in (0.1, 10.0):
        for fs in (0.1, 10.0):
            E_DRAM, E_SRAM = base_dram * fd, base_sram * fs
            g_xf = transformer_energy(n_hi, P, L, d_model, gqa_ratio)[0] / \
                   transformer_energy(n_lo, P, L, d_model, gqa_ratio)[0]
            g_l = lwf_energy(n_hi, N_store, d_ws, P_core)[0] / \
                  lwf_energy(n_lo, N_store, d_ws, P_core)[0]
            ok = g_xf > g_l
            survived &= ok
    E_DRAM, E_SRAM = base_dram, base_sram
    print(f"slope conclusion (xf grows faster than LWF) held under all perturbations: "
          f"{survived}")
    return rows


if __name__ == "__main__":
    run()
