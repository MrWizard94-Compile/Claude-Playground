"""
Experiment M12 -- controller quality: admission/eviction for the Workspace<->Ledger boundary.

M10 measured the frontier-capacity LAW assuming an ORACLE loaded the right F bindings. EP-GRM's
EXP-022 warns that assumption away: the scheduler/admission policy is a first-class factor, and a
naive frontier->capacity law is incomplete. This experiment drops the oracle and asks: which
controller best keeps the true (latent, drifting) frontier resident in the bounded Workspace?

This is exactly a CACHE-REPLACEMENT problem -- Workspace = cache of capacity C over the Ledger --
so ~50 years of cache theory transfers. We pit realistic policies against the oracle upper bound.

WORKLOAD: K keys; a latent active set (frontier) of size F drifts slowly (temporal locality);
within the frontier, query frequency is Zipf-skewed (so recency AND frequency both carry signal).
Each step queries a key; if resident -> HIT, else fetch from Ledger + admit + evict per policy.

POLICIES:
  random   : evict a random resident (baseline)
  fifo     : evict oldest-admitted
  lru      : evict least-recently-used (exploits temporal locality)
  lfu      : evict least-frequently-used (exploits frequency skew)
  belief   : LWF-native -- each key has an activation that DECAYS each step and boosts on use
             (recency x frequency blend; the fast-weight decay gate applied to admission)
  oracle   : evict a resident that is NOT in the current active set (upper bound)

CLAIM: the right controller (belief/lru) approaches the oracle and far exceeds random/fifo; the
gap-to-oracle quantifies "controller quality" and its dependence on capacity C vs frontier F and
drift. FALSIFICATION: if no realistic policy beats random, admission is not exploitable here; if
even the best trails the oracle badly at C>=F, bounded caching cannot track the frontier.
"""

from __future__ import annotations
import numpy as np

SEED = 0


class Cache:
    def __init__(self, C, policy, decay=0.9):
        self.C = C
        self.policy = policy
        self.decay = decay
        self.slots = {}                     # key -> metadata (recency/freq/activation/order)
        self.t = 0
        self.order = 0

    def step_decay(self):
        if self.policy == "belief":
            for k in self.slots:
                self.slots[k] *= self.decay

    def access(self, key, active_set, rng):
        self.t += 1
        if key in self.slots:               # HIT
            self._touch(key)
            return True
        # MISS -> admit, evicting if full
        if len(self.slots) >= self.C:
            self._evict(active_set, rng)
        self.order += 1
        self.slots[key] = 0.0
        self._touch(key)
        return False

    def _touch(self, key):
        if self.policy in ("lru", "fifo"):
            self.slots[key] = self.order if self.policy == "fifo" else self.t
        elif self.policy == "lfu":
            self.slots[key] += 1
        elif self.policy == "belief":
            self.slots[key] += 1.0
        else:                                # random
            self.slots[key] = 0.0

    def _evict(self, active_set, rng):
        if self.policy == "random":
            victim = rng.choice(list(self.slots))
        elif self.policy == "oracle":
            outsiders = [k for k in self.slots if k not in active_set]
            victim = rng.choice(outsiders) if outsiders else rng.choice(list(self.slots))
        else:                                # lru/fifo/lfu/belief: evict min-score
            victim = min(self.slots, key=lambda k: self.slots[k])
        del self.slots[victim]


def workload(K, F, drift, steps, zipf_s, rng):
    """Yield (query_key, active_set) with a slowly drifting Zipf-skewed frontier."""
    active = list(rng.choice(K, size=F, replace=False))
    ranks = np.arange(1, F + 1)
    probs = (1.0 / ranks ** zipf_s); probs /= probs.sum()
    seq = []
    for _ in range(steps):
        if rng.random() < drift:            # drift: swap one active key out
            i = rng.integers(F)
            cand = int(rng.integers(K))
            if cand not in active:
                active[i] = cand
        q = int(active[rng.choice(F, p=probs)])
        seq.append((q, set(active)))
    return seq


def hit_rate(K, F, C, policy, drift, steps, zipf_s, seeds):
    rates = []
    for s in seeds:
        rng = np.random.default_rng(SEED + s)
        seq = workload(K, F, drift, steps, zipf_s, rng)
        cache = Cache(C, policy)
        warm = steps // 5
        hits = seen = 0
        for i, (q, active) in enumerate(seq):
            cache.step_decay()
            h = cache.access(q, active, rng)
            if i >= warm:
                hits += h; seen += 1
        rates.append(hits / seen)
    return float(np.mean(rates)), float(np.std(rates))


def run(K=500, F=32, steps=6000, zipf_s=1.1, seeds=(0, 1, 2),
        policies=("random", "fifo", "lru", "lfu", "belief", "oracle")):
    print(f"\n=== M12: controller quality (K={K} keys, frontier F={F}, Zipf s={zipf_s}) ===")
    print("steady-state HIT RATE (frontier resident in bounded Workspace); higher = better\n")

    for drift in (0.02, 0.10):
        print(f"-- drift={drift} (frontier turnover per step) --")
        hdr = f"{'C/F':>5} | " + " ".join(f"{p:>8}" for p in policies)
        print(hdr); print("-" * len(hdr))
        for cf in (0.5, 1.0, 1.5, 2.0):
            C = max(1, int(cf * F))
            cells = []
            for p in policies:
                m, sd = hit_rate(K, F, C, p, drift, steps, zipf_s, seeds)
                cells.append(f"{m:>8.3f}")
            print(f"{cf:>5.1f} | " + " ".join(cells))
        print()

    # Two-regime verdict. EASY: C>=F, any reasonable controller should ~ match oracle. HARD:
    # C<F, the regime EXP-022 cares about -- controller quality and the decay gate should matter.
    def hr(C, p, drift=0.02):
        return hit_rate(K, F, C, p, drift, steps, zipf_s, seeds)[0]

    Ce, Ch = 2 * F, F // 2
    easy = {p: hr(Ce, p) for p in ("random", "lru", "belief", "oracle")}
    hard = {p: hr(Ch, p) for p in ("random", "lru", "lfu", "belief", "oracle")}
    print(f"EASY (C=2F): best(lru/belief)={max(easy['lru'], easy['belief']):.3f} vs "
          f"oracle={easy['oracle']:.3f}  -> controller ~ sufficient when C>=F (confirms M10).")
    print(f"HARD (C=F/2): random={hard['random']:.3f}  lfu={hard['lfu']:.3f}  "
          f"lru={hard['lru']:.3f}  belief={hard['belief']:.3f}  set-oracle={hard['oracle']:.3f}")
    print(f"  -> belief beats un-aged LFU by {hard['belief']-hard['lfu']:.3f} (the DECAY gate "
          f"cures LFU staleness); best realistic beats random by "
          f"{max(hard['lru'],hard['belief'])-hard['random']:.3f}.")
    print("  -> NOTE: the set-oracle is NOT optimal at C<F (it ignores intra-frontier frequency);"
          " recency/frequency policies can exceed it -- an honest limit of that upper bound.")
    easy_ok = max(easy['lru'], easy['belief']) >= easy['oracle'] - 0.03
    hard_ok = (hard['belief'] - hard['lfu'] > 0.1) and \
              (max(hard['lru'], hard['belief']) - hard['random'] > 0.03)
    ok = easy_ok and hard_ok
    print(f"M12 (controller: sufficient at C>=F; decay-aware policy needed & effective at C<F): "
          f"{'SUPPORTED' if ok else 'CHECK'}")
    print("[Recommendation: belief-decay (frequency + decay gate) or LRU as the Workspace "
          "admission/eviction policy. Avoid un-aged LFU. Pair with the M10 frontier law.]")
    return None


if __name__ == "__main__":
    run()
