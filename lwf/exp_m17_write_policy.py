"""
Experiment M17 -- verified-only write policy for the Ledger (imported from JanusPrime; G1.4).

LWF tested read (M2), eviction (M12/M13), capacity (M1/M10) -- but WHAT gets written to the Ledger,
and WHEN, was a total blank (GAPS.md G1.4). JanusPrime's production pattern: **only validated
successes seed the store; failed validations never write.** This experiment tests that pattern:
a synthesis/reasoning process emits a stream of candidate facts, some CORRECT and some CORRUPT
(wrong values). Two write policies feed the Ledger:

  ungated       : write every candidate (last-write-wins per key). Corrupt writes pollute the store.
  verified-only : write a candidate ONLY if it passes an independent verifier (models a compile/
                  test/constraint/nogood check -- imperfect: catches corruption at rate `detect`,
                  falsely rejects valid facts at rate `fp`).

Downstream: query every key, read the Ledger, score against ground truth. A polluted store returns
wrong answers; a gated store stays trustworthy -- at the cost of some valid writes lost to false
rejections (coverage). We measure both.

=====================================================================================
PRE-REGISTERED VERDICT (fixed before run; reported verbatim; NOT reframable post-hoc).
Verifier realism: detect=0.90 (catches 90% of corrupt), fp=0.05 (rejects 5% of valid).
  P1 at corruption=40%: verified-only end-to-end accuracy >= ungated + 0.20
  P2 gate rejects >= 80% of corrupt writes (== verifier detect within noise)
  P3 verified-only store PURITY >= 0.90 across corruption in {0.2,0.4,0.6}
  ALSO REPORTED (honest cost): valid-write rejection rate (coverage lost to false positives).
  PASS = P1 and P2 and P3. Anything else = HONEST NEGATIVE.
=====================================================================================
CPU, deterministic (fixed seeds). Connects M11 (nogoods = one kind of verifier) to the write path.
"""

from __future__ import annotations
import numpy as np

SEED = 0
V = 10                 # value classes


def make_ground_truth(n_keys, rng):
    return rng.integers(0, V, size=n_keys)          # key -> correct value class


def verify(is_corrupt, detect, fp, rng):
    """Independent imperfect verifier. Returns True = 'passed verification' (write allowed)."""
    if is_corrupt:
        return rng.random() >= detect               # corrupt passes iff verifier misses it
    return rng.random() >= fp                        # valid passes unless false-positive reject


def run_policy(gt, stream, policy, detect, fp, rng):
    """stream = list of (key, value, is_corrupt). Returns store dict key->value + stats."""
    store = {}
    accepted = rejected = corrupt_accepted = corrupt_total = valid_rejected = valid_total = 0
    for key, value, is_corrupt in stream:
        corrupt_total += is_corrupt
        valid_total += (not is_corrupt)
        write = True
        if policy == "verified":
            write = verify(is_corrupt, detect, fp, rng)
        if write:
            store[key] = value                       # last-write-wins
            accepted += 1
            corrupt_accepted += is_corrupt
        else:
            rejected += 1
            valid_rejected += (not is_corrupt)
    # downstream: end-to-end accuracy over ALL keys (unanswered counts wrong)
    correct = sum(1 for k in range(len(gt)) if store.get(k) == gt[k])
    e2e_acc = correct / len(gt)
    purity = (sum(1 for k, v in store.items() if gt[k] == v) / len(store)) if store else 1.0
    coverage = len(store) / len(gt)
    corrupt_reject_rate = 1.0 - (corrupt_accepted / max(corrupt_total, 1))
    valid_reject_rate = valid_rejected / max(valid_total, 1)
    return dict(e2e_acc=e2e_acc, purity=purity, coverage=coverage,
                corrupt_reject_rate=corrupt_reject_rate, valid_reject_rate=valid_reject_rate)


def build_stream(gt, n_writes, corruption, rng):
    stream = []
    for _ in range(n_writes):
        k = int(rng.integers(len(gt)))
        if rng.random() < corruption:
            wrong = int(rng.integers(V))
            while wrong == gt[k]:
                wrong = int(rng.integers(V))
            stream.append((k, wrong, True))
        else:
            stream.append((k, int(gt[k]), False))
    return stream


def run(n_keys=500, n_writes=4000, detect=0.90, fp=0.05, seeds=(0, 1, 2)):
    print(f"\n=== M17: verified-only write policy for the Ledger (JanusPrime import; G1.4) ===")
    print(f"verifier: detect={detect} (catches corrupt), fp={fp} (false-reject valid). "
          f"n_keys={n_keys}, writes={n_writes}\n")
    hdr = f"{'corruption':>10} | {'ungated e2e':>11} {'ungated pur':>11} | " \
          f"{'verif e2e':>10} {'verif pur':>10} {'verif cov':>10} | {'valid-rej':>9}"
    print(hdr); print("-" * len(hdr))
    rows = {}
    for corruption in (0.2, 0.4, 0.6):
        ung, ver = [], []
        for s in seeds:
            rng = np.random.default_rng(SEED + s)
            gt = make_ground_truth(n_keys, rng)
            stream = build_stream(gt, n_writes, corruption, rng)
            ung.append(run_policy(gt, stream, "ungated", detect, fp, np.random.default_rng(100 + s)))
            ver.append(run_policy(gt, stream, "verified", detect, fp, np.random.default_rng(100 + s)))
        def m(lst, k): return float(np.mean([r[k] for r in lst]))
        rows[corruption] = dict(ung=ung, ver=ver)
        print(f"{corruption:>10.1f} | {m(ung,'e2e_acc'):>11.3f} {m(ung,'purity'):>11.3f} | "
              f"{m(ver,'e2e_acc'):>10.3f} {m(ver,'purity'):>10.3f} {m(ver,'coverage'):>10.3f} | "
              f"{m(ver,'valid_reject_rate'):>9.3f}")

    def mm(corruption, pol, k):
        return float(np.mean([r[k] for r in rows[corruption][pol]]))
    print("\n--- PRE-REGISTERED VERDICT (fixed before run) ---")
    p1_gap = mm(0.4, 'ver', 'e2e_acc') - mm(0.4, 'ung', 'e2e_acc')
    p1 = p1_gap >= 0.20
    p2_rej = np.mean([mm(c, 'ver', 'corrupt_reject_rate') for c in (0.2, 0.4, 0.6)])
    p2 = p2_rej >= 0.80
    p3 = all(mm(c, 'ver', 'purity') >= 0.90 for c in (0.2, 0.4, 0.6))
    valid_rej = np.mean([mm(c, 'ver', 'valid_reject_rate') for c in (0.2, 0.4, 0.6)])
    print(f"  P1 @corruption=0.4: verified e2e - ungated e2e = {p1_gap:.3f} (>=0.20) -> {p1}")
    print(f"  P2 corrupt-write rejection rate = {p2_rej:.3f} (>=0.80) -> {p2}")
    print(f"  P3 verified store purity >=0.90 across corruption: -> {p3}")
    print(f"  honest cost: valid-write rejection (coverage lost to false positives) = {valid_rej:.3f}")
    ok = p1 and p2 and p3
    print(f"  M17 VERDICT: {'PASS' if ok else 'FAIL (honest negative)'}")
    print("  [Ledger write path: gating writes on an independent verifier keeps the store trustworthy;")
    print("   the cost is a small fraction of valid writes lost to false rejection.]")
    return ok


if __name__ == "__main__":
    run()
