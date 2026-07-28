"""
Experiment M6 -- the §7 crux at TOY scale, LEARNED (PyTorch, CPU).

THE QUESTION (the one that can kill the program): when you replace full-context
attention with a bounded recurrent state, do you lose recall capacity -- and does an
external content-addressable read (the Ledger) recover it?

TASK: MQAR (multi-query associative recall, Arora et al. "Zoology" ICLR 2024) -- the
canonical synthetic probe of fixed-state recall. A sequence presents D key->value pairs,
then queries a subset of keys; the model must emit the right value at each query position.
Increasing D increases the number of associations that must be held = state pressure.

MODELS (one mixing layer each, matched d_model, shared embed+head):
  linear  : linear-attention / fixed d_head x d_head recurrent state (bounded)   [Workspace]
  delta   : DeltaNet-style fixed state (subtract prediction before write)         [Workspace+]
  attn    : single-head causal softmax attention (full context)                   [ceiling]
  hybrid  : linear (bounded) + a content-addressed softmax read over stored (k,v) [Workspace+Ledger]

PREDICTION (LWF thesis): as D grows, linear/delta degrade (fixed state saturates), while
attn and hybrid hold -- i.e. the external content-addressed read recovers what bounded
state loses. If hybrid tracks the bounded models DOWN instead of attn UP, the Ledger buys
nothing and the thesis is (at this scale) falsified.

HONESTY: this is small-scale, CPU, single-layer -- DIRECTIONAL evidence, not proof. The
hybrid's softmax read is dense here for differentiability; deployment restricts it to a
top-k ANN neighbourhood (O(log N)). Seeds fixed. Not a claim about frontier models.
"""

from __future__ import annotations
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

# vocab layout: [0..KP) keys, [KP..KP+VP) values
KP, VP = 64, 64
VOCAB = KP + VP
IGN = -100


def make_batch(B, D, Q, rng):
    """MQAR batch. Context = interleaved [k,v]*D ; then Q query keys; predict values."""
    T = 2 * D + Q
    x = np.zeros((B, T), dtype=np.int64)
    y = np.full((B, T), IGN, dtype=np.int64)
    for b in range(B):
        keys = rng.choice(KP, size=D, replace=False)
        vals = rng.integers(KP, KP + VP, size=D)
        seq = np.empty(2 * D, dtype=np.int64)
        seq[0::2] = keys
        seq[1::2] = vals
        x[b, :2 * D] = seq
        qidx = rng.choice(D, size=Q, replace=(Q > D))
        for j, qi in enumerate(qidx):
            pos = 2 * D + j
            x[b, pos] = keys[qi]        # query = the key token
            y[b, pos] = vals[qi]        # target = its value (predict AT this position)
    return torch.from_numpy(x), torch.from_numpy(y)


def phi(t):                             # linear-attention feature map (positive)
    return F.elu(t) + 1.0


class Mixer(nn.Module):
    def __init__(self, d_model, d_head, kind):
        super().__init__()
        self.kind = kind
        self.dh = d_head
        self.q = nn.Linear(d_model, d_head, bias=False)
        self.k = nn.Linear(d_model, d_head, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        if kind == "hybrid":
            self.qk2 = nn.Linear(d_model, d_head, bias=False)  # separate head for Ledger read

    def _linear(self, q, k, v):
        # bounded fixed-state recurrence via causal cumulative sum of outer products
        q, k = phi(q), phi(k)
        kv = torch.einsum("bti,btj->btij", v, k)          # (B,T,dv,dh)
        kv = torch.cumsum(kv, dim=1)                       # causal fixed-state summary
        num = torch.einsum("btij,btj->bti", kv, q)
        z = torch.cumsum(k, dim=1)                         # normalizer
        den = torch.einsum("bti,bti->bt", z, q).clamp_min(1e-4).unsqueeze(-1)
        return num / den

    def _delta(self, q, k, v):
        B, T, _ = q.shape
        q, k = phi(q), phi(k)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        S = torch.zeros(B, v.shape[-1], self.dh, dtype=v.dtype)
        outs = []
        for t in range(T):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]
            pred = torch.einsum("bij,bj->bi", S, kt)
            S = S + torch.einsum("bi,bj->bij", vt - pred, kt)
            outs.append(torch.einsum("bij,bj->bi", S, qt))
        return torch.stack(outs, dim=1)

    def _attn(self, q, k, v):
        T = q.shape[1]
        att = (q @ k.transpose(1, 2)) / (self.dh ** 0.5)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        return F.softmax(att, dim=-1) @ v

    def forward(self, h, vsrc):
        # q,k from the token stream; v from the SHIFTED (next-token) stream so that
        # matching a key by content retrieves its value in a single layer (the standard
        # shifted-value MQAR construction -- otherwise one layer cannot associate k->v).
        q, k, v = self.q(h), self.k(h), self.v(vsrc)
        if self.kind == "linear":
            return self._linear(q, k, v)
        if self.kind == "delta":
            return self._delta(q, k, v)
        if self.kind == "attn":
            return self._attn(q, k, v)
        if self.kind == "hybrid":                          # Workspace + Ledger read
            ws = self._linear(q, k, v)
            q2, k2 = self.qk2(h), self.k(h)                # content-addressed (Hopfield) read
            ledger = self._attn(q2, k2, v)
            return ws + ledger
        raise ValueError(self.kind)


class Model(nn.Module):
    def __init__(self, kind, d_model=64, d_head=32):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.mix = Mixer(d_model, d_head, kind)
        self.head = nn.Linear(d_model, VOCAB)

    def forward(self, x):
        e = self.emb(x)
        vsrc = torch.roll(e, shifts=-1, dims=1)   # next-token stream = value carrier
        h = e + self.mix(self.norm(e), vsrc)
        return self.head(h)


def query_accuracy(model, x, y):
    with torch.no_grad():
        logits = model(x)
        mask = y != IGN
        pred = logits.argmax(-1)
        return (pred[mask] == y[mask]).float().mean().item()


def train_eval(kind, D, steps, B, lr, d_head, seed=0):
    rng = np.random.default_rng(seed)
    model = Model(kind, d_head=d_head)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    Q = D
    for _ in range(steps):
        x, y = make_batch(B, D, Q, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), ignore_index=IGN)
        opt.zero_grad(); loss.backward(); opt.step()
    # eval on fresh data
    xe, ye = make_batch(256, D, Q, rng)
    return query_accuracy(model, xe, ye)


def run(D_grid=(8, 16, 32, 48), steps=700, B=32, lr=2e-3, d_head=32,
        kinds=("linear", "delta", "attn", "hybrid"), seeds=(0,)):
    print(f"\n=== M6: learned MQAR crux probe (d_head={d_head}, steps={steps}, "
          f"batch={B}, seeds={len(seeds)}) ===")
    print("query-position accuracy (chance = 1/VP = {:.3f}); D = #key-value pairs "
          "(state pressure)\n".format(1.0 / VP))
    header = f"{'D pairs':>8} | " + " ".join(f"{k:>8}" for k in kinds)
    print(header); print("-" * len(header))
    table = {}
    for D in D_grid:
        accs = {}
        for k in kinds:
            vals = [train_eval(k, D, steps, B, lr, d_head, seed=s) for s in seeds]
            accs[k] = float(np.mean(vals))
        table[D] = accs
        print(f"{D:>8} | " + " ".join(f"{accs[k]:>8.3f}" for k in kinds))

    Dmax = D_grid[-1]
    bounded = min(table[Dmax].get("linear", 1), table[Dmax].get("delta", 1))
    retrieval = max(table[Dmax].get("attn", 0), table[Dmax].get("hybrid", 0))
    print(f"\nAt D={Dmax}: best bounded={bounded:.3f}, best retrieval={retrieval:.3f}")
    verdict = "SUPPORTED" if (retrieval - bounded > 0.2) else "CHECK"
    print(f"M6 (retrieval recovers recall that bounded state loses, learned): {verdict}")
    print("[DIRECTIONAL: toy scale, single layer, CPU -- not a frontier-model claim]")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to check timing")
    args = ap.parse_args()
    t0 = time.time()
    if args.smoke:
        run(D_grid=(8, 16), steps=60, kinds=("linear", "attn", "hybrid"))
    else:
        run()
    print(f"\n[wall time: {time.time() - t0:.1f}s]")
