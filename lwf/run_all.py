"""
Run the LWF Stage-0 falsification suite: M1, M2, M4.

These test the *information-routing* claims of the Ledger/Workspace/Fabric design.
They deliberately do NOT test the learned-model question (the §7 crux), which needs
a GPU and is Stage 0.5. Keeping them separate is the point: we falsify what is cheap
to falsify first.

Usage:  python run_all.py [--plots]
"""

from __future__ import annotations
import sys
import numpy as np

import exp_m1_capacity as m1
import exp_m2_ledger_recovery as m2
import exp_m3_replay as m3
import exp_m4_cost_model as m4
import exp_m5_multihop as m5


def maybe_plots(r1, r2, r4):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa
        print(f"\n[plots skipped: {e}]")
        return
    import os
    outdir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(outdir, exist_ok=True)

    # M1
    Ns = [n for n, _ in r1]
    d = 64
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, lbl in [(("delta", True), "delta+orthonormal"),
                      (("delta", False), "delta+random"),
                      (("hebb", False), "hebb+random")]:
        ax.plot([n / d for n in Ns], [row[key] for _, row in r1], marker="o", label=lbl)
    ax.axvline(1.0, ls="--", color="k", alpha=.5, label="N=d (rank wall)")
    ax.set_xlabel("N / d"); ax.set_ylabel("top-1 recall"); ax.set_title("M1: Workspace rank ceiling")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{outdir}/m1_capacity.png", dpi=120)

    # M2
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r[0] for r in r2], [r[1] for r in r2], marker="o", label="cram (ws-only)")
    ax.plot([r[0] for r in r2], [r[2] for r in r2], marker="s", label="LWF (ws+ledger)")
    ax.set_xscale("log", base=2); ax.set_xlabel("N associations"); ax.set_ylabel("top-1 recall")
    ax.set_title("M2: Ledger recovers recall @ fixed hot-state"); ax.legend(); fig.tight_layout()
    fig.savefig(f"{outdir}/m2_ledger.png", dpi=120)

    # M4
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r[0] for r in r4], [r[1] for r in r4], marker="o", label="transformer")
    ax.plot([r[0] for r in r4], [r[2] for r in r4], marker="s", label="LWF")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("context length n"); ax.set_ylabel("nJ / token")
    ax.set_title("M4: per-token energy vs context"); ax.legend(); fig.tight_layout()
    fig.savefig(f"{outdir}/m4_energy.png", dpi=120)
    print(f"\n[figures written to {outdir}]")


def main():
    print("#" * 70)
    print("# LWF Stage-0 falsification suite  (M1 rank ceiling / M2 Ledger / M4 energy)")
    print("#" * 70)
    r1 = m1.run()
    r2 = m2.run()
    r3 = m3.run()
    r4 = m4.run()
    r5 = m5.run()
    if "--plots" in sys.argv:
        maybe_plots(r1, r2, r4)
    print("\n" + "=" * 70)
    print("Non-training suite complete (M1/M2/M3/M4/M5). The learned crux probe")
    print("(M6) trains models and runs separately:  python exp_m6_crux_learned.py")
    print("=" * 70)
    print("Read the per-experiment verdicts above.")


if __name__ == "__main__":
    main()
