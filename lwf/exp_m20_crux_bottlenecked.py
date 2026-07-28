"""
Experiment M20 -- the crux, done right: rank-bottlenecked Workspace on global COMPUTATION (G2.1).

M19 failed to test the crux because I used the wrong capacity model: I bounded by BITS, but a
float state holds enormous information, so bounded state sorted N=16 fine. The correct bound on a
fast-weight Workspace is RANK (M1): it holds ~dh recoverable associations. So to genuinely stress
it, make N > dh -- the Workspace cannot hold all N values recoverably.

This isolates the real open question. The STORAGE crux is settled (M1 rank ceiling -> M2 Ledger
rescues recall). The untested one is COMPUTATION: when the Workspace is rank-bottlenecked (dh < N),
can a content-addressable Ledger rescue a GLOBAL computation (sorting = order statistics), or only
recall? Sorting is the probe because emitting the i-th smallest is a global selection, not a lookup.

Task: in-context sorting (reused from M19). Fixed N; SWEEP the Workspace rank dh from >N down to <<N.
Models: attn (full-context, no bottleneck), bounded (fast-weight Workspace only), hybrid (Workspace +
top-k Ledger over the N value positions).

=====================================================================================
PRE-REGISTERED VERDICT (fixed before run; reported verbatim; NOT reframable post-hoc).
  Find the smallest dh where bounded is genuinely bottlenecked: bounded <= 0.60 AND attn >= 0.85.
  At that dh:
    - hybrid >= 0.85*attn                 -> RETRIEVAL RESCUES GLOBAL COMPUTATION (thesis robust)
    - bounded+0.15 < hybrid < 0.85*attn   -> PARTIAL (retrieval helps, doesn't close it)
    - hybrid <= bounded+0.15              -> CRUX CONFIRMED: retrieval rescues recall, NOT global
                                             computation -- a real capability ceiling of the design
  If NO swept dh bottlenecks bounded while attn works -> INCONCLUSIVE (couldn't create the regime).
  Two-sided: LWF can fail here, and that failure would be the most informative outcome of the program.
=====================================================================================
GPU. dh is the fast-weight Workspace rank (key dim); d_model stays fixed so embed/compute capacity is
held constant -- only the recurrent state's associative rank is varied.
"""

from __future__ import annotations
import argparse, time
import numpy as np
import torch
import torch.nn.functional as F

from exp_m19_hard_reasoning import Model, make_batch, DEVICE, VOCAB, IGN, A


def train_eval(kind, N, dh, steps, B, lr, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = Model(kind, d=96, dh=dh, topk=8, layers=2, heads=6).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    for _ in range(steps):
        x, y = make_batch(B, N, rng)
        loss = F.cross_entropy(model(x, N).reshape(-1, VOCAB), y.reshape(-1), ignore_index=IGN)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    xe, ye = make_batch(256, N, rng)
    with torch.no_grad():
        m = ye != IGN
        return (model(xe, N).argmax(-1)[m] == ye[m]).float().mean().item()


def run(N=24, dh_grid=(24, 8, 4, 2), steps=3000, B=128, lr=2e-3, seeds=(0,)):
    print(f"\n=== M20: crux via rank-bottlenecked Workspace (sorting N={N}, device={DEVICE.type}, "
          f"steps={steps}) ===")
    print(f"per-slot accuracy (chance=1/{A}={1/A:.3f}); dh = Workspace rank (values to hold = N={N})\n")
    hdr = f"{'dh (rank)':>10} | {'attn':>7} | {'bounded':>8} | {'hybrid':>7}"
    print(hdr); print("-" * len(hdr))
    res = {}
    # attn has no dh bottleneck; compute once
    attn_acc = float(np.mean([train_eval("attn", N, 64, steps, B, lr, s) for s in seeds]))
    for dh in dh_grid:
        b = float(np.mean([train_eval("bounded", N, dh, steps, B, lr, s) for s in seeds]))
        h = float(np.mean([train_eval("hybrid", N, dh, steps, B, lr, s) for s in seeds]))
        res[dh] = (attn_acc, b, h)
        print(f"{dh:>10} | {attn_acc:>7.3f} | {b:>8.3f} | {h:>7.3f}")

    print("\n--- PRE-REGISTERED VERDICT (fixed before run) ---")
    bottleneck_dh = None
    for dh in dh_grid:
        a, b, h = res[dh]
        if b <= 0.60 and a >= 0.85:
            bottleneck_dh = dh
            break
    if bottleneck_dh is None:
        print("  no swept dh bottlenecked bounded while attn worked -> INCONCLUSIVE")
        print("  M20 VERDICT: INCONCLUSIVE (could not create the failure regime)")
        return res
    a, b, h = res[bottleneck_dh]
    print(f"  bottleneck at dh={bottleneck_dh}: attn={a:.3f}, bounded={b:.3f}, hybrid={h:.3f}")
    if h >= 0.85 * a:
        v = "RETRIEVAL RESCUES GLOBAL COMPUTATION (thesis robust: Ledger recovers sorting)"
    elif h > b + 0.15:
        v = "PARTIAL (retrieval helps but does not close global computation)"
    else:
        v = "CRUX CONFIRMED: retrieval rescues RECALL, not global COMPUTATION (design ceiling)"
    print(f"  hybrid={h:.3f} vs 0.85*attn={0.85*a:.3f}, bounded={b:.3f}")
    print(f"  M20 VERDICT: {v}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if args.smoke:
        run(N=24, dh_grid=(8, 2), steps=200)
    else:
        run()
    print(f"[wall {time.time()-t0:.0f}s]")
