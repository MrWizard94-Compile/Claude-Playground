"""
Experiment M10 -- frontier-width scaling (imported from the sibling EP-GRM project).

EP-GRM's central empirical finding: the required executive state tracks the DEPENDENCY
FRONTIER WIDTH F (the count of simultaneously-active bindings), NOT the total problem size N.
This experiment reproduces that law in LWF's associative substrate and ties it to the M1
rank-d Workspace ceiling.

SETUP: N (key,value) pairs exist. A reasoning step has a FRONTIER of F bindings that must be
held simultaneously to proceed (the rest are distractors / inactive knowledge). We measure
whether the bounded Workspace (capacity ~ d, from M1) can hold the frontier, as F and N vary
independently at fixed d.

  LWF-frontier : load ONLY the F active bindings into the Workspace (evict the N-F inactive to
                 the Ledger). Recall the F. Prediction: accuracy tracks F/d (breaks at F>d) and
                 is INDEPENDENT of N -- distractors sit in the Ledger, off the hot path.
  cram         : load ALL N into the Workspace. Recall the F. Prediction: accuracy drops with N
                 regardless of F (M2's failure, reframed).
  ledger       : recall the F directly from the Ledger (exact). Prediction: ~perfect at O(F)
                 reads, independent of both F-vs-d and N -- the escape hatch when F > d.

Decoding is restricted to the F ACTIVE values (a controlled candidate set) so we measure pure
STORAGE capacity, not global decode ambiguity.

RESULT = the LWF cost law: state ~ O(frontier width F); depth was already shown free in state
(M5). Together: LWF pays O(depth) steps x O(frontier) state -- the two-axis picture.

HONEST CAVEAT (from EP-GRM's own negatives, EXP-022): this assumes the frontier is correctly
IDENTIFIED (an oracle admission policy loads the right F). Their controlled runs showed the
scheduler/admission quality is a separate first-class factor; a naive F->capacity law is
incomplete. M10 measures the capacity law GIVEN correct frontier identification.
"""

from __future__ import annotations
import numpy as np
from workspace import FastWeightWorkspace, unit_rows
from ledger import ContentAddressableLedger

SEED = 0


def decode_among(v_hat, candidates):
    """argmax cosine over a controlled candidate set (the F active values)."""
    return int(np.argmax(candidates @ v_hat))


def trial(F, N, d, mode, rng):
    keys = unit_rows(rng.standard_normal((N, d)))
    vals = unit_rows(rng.standard_normal((N, d)))
    active = np.arange(F)                     # the frontier = first F bindings
    active_vals = vals[active]                # controlled decode candidate set

    if mode == "lwf":                         # Workspace holds ONLY the F frontier bindings
        ws = FastWeightWorkspace(d, d, mode="hebb")
        for j in active:
            ws.write(keys[j], vals[j])
        # (the N-F inactive bindings would live in the Ledger; irrelevant to frontier recall)
        hits = sum(decode_among(ws.read(keys[j]), active_vals) == i
                   for i, j in enumerate(active))
        return hits / F

    if mode == "cram":                        # Workspace holds ALL N bindings
        ws = FastWeightWorkspace(d, d, mode="hebb")
        for j in range(N):
            ws.write(keys[j], vals[j])
        hits = sum(decode_among(ws.read(keys[j]), active_vals) == i
                   for i, j in enumerate(active))
        return hits / F

    if mode == "ledger":                      # exact recall of the frontier from the Ledger
        led = ContentAddressableLedger(d, d)
        for j in range(N):
            led.write(keys[j], vals[j])
        hits = sum(decode_among(led.read_top1_value(keys[j]), active_vals) == i
                   for i, j in enumerate(active))
        return hits / F
    raise ValueError(mode)


def run(d=64, F_grid=(4, 16, 64, 128, 192, 256), N_grid=(256, 1024, 4096, 8192),
        seeds=(0, 1, 2)):
    print(f"\n=== M10: frontier-width scaling (d={d}; capacity ~ d from M1) ===")
    print("frontier-recall accuracy; decode restricted to the F active values\n")

    for mode, title in (("lwf", "LWF-frontier (Workspace holds only F active)"),
                        ("cram", "cram (Workspace holds all N)"),
                        ("ledger", "Ledger (exact recall of frontier)")):
        print(f"-- {title} --")
        corner = "F\\N"
        hdr = f"{corner:>6} | " + " ".join(f"{N:>7}" for N in N_grid)
        print(hdr); print("-" * len(hdr))
        grid = {}
        for F in F_grid:
            row = []
            for N in N_grid:
                if F > N:
                    row.append(float("nan")); continue
                accs = [trial(F, N, d, mode, np.random.default_rng(SEED + s)) for s in seeds]
                row.append(float(np.mean(accs)))
            grid[F] = row
            cells = " ".join((f"{v:>7.3f}" if v == v else f"{'--':>7}") for v in row)
            print(f"{F:>6} | {cells}")
        # independence check
        if mode in ("lwf", "ledger"):
            stds = []
            for F, row in grid.items():
                vals = [v for v in row if v == v]
                if len(vals) > 1:
                    stds.append(np.std(vals))
            print(f"  -> across-N std at fixed F (should be ~0 if capacity ~ F not N): "
                  f"mean {np.mean(stds):.3f}, max {np.max(stds):.3f}")
        print()

    # verdict: LWF depends on F/d not N; cram depends on N. Guard F <= N throughout.
    Nlo, Nhi = min(N_grid), max(N_grid)
    lwf_lowF = np.mean([trial(8, N, d, "lwf", np.random.default_rng(SEED + s))
                        for N in N_grid for s in seeds])
    highF = 4 * d
    lwf_highF = np.mean([trial(highF, N, d, "lwf", np.random.default_rng(SEED + s))
                         for N in N_grid if N >= highF for s in seeds])
    cram_smallN = np.mean([trial(8, Nlo, d, "cram", np.random.default_rng(SEED + s))
                           for s in seeds])
    cram_bigN = np.mean([trial(8, Nhi, d, "cram", np.random.default_rng(SEED + s))
                         for s in seeds])
    print(f"LWF-frontier: F=8 -> {lwf_lowF:.3f} (flat in N),  F=4d -> {lwf_highF:.3f} "
          f"(degrades past capacity d)")
    print(f"cram (fixed F=8): N={Nlo} -> {cram_smallN:.3f},  N={Nhi} -> {cram_bigN:.3f} "
          f"(drops with N)")
    law = (lwf_lowF > 0.9 and lwf_highF < lwf_lowF - 0.15 and cram_smallN - cram_bigN > 0.2)
    print(f"M10 (Workspace capacity tracks FRONTIER F, not total N -- EP-GRM law reproduced): "
          f"{'SUPPORTED' if law else 'CHECK'}")
    return None


if __name__ == "__main__":
    run()
