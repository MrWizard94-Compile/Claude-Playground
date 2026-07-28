"""
Experiment M7 -- Stage 0.5: the crux, harder and honest.

M6 showed a hybrid recovers recall, but its Ledger read was DENSE softmax over the whole
context -- a skeptic rightly says "that's just attention; you never showed the BOUNDED-COST
version." M7 fixes exactly that and adds rigor:

  1. TOP-K SPARSE LEDGER: the Ledger read attends to only the k best entries per step
     (bounded reads/step, independent of context length T) -- the real architecture claim,
     not dense attention. Still differentiable through the selected top-k.
  2. PUSH TO BREAKAGE: sweep association pressure D until even the strong bounded model
     (delta) bends, so the necessity of retrieval is visible, not assumed.
  3. ERROR BARS: multiple seeds -> mean +/- std.
  4. MULTI-LAYER: stack L mixing layers (does depth change the picture?).
  5. COST ACCOUNTING: report reads/step (attn = O(T), hybrid-topk = k = const) alongside
     accuracy, so the claim is "matches attention accuracy at bounded read cost."

HARDWARE-AGNOSTIC: auto-selects CUDA if present (larger preset), else CPU (smaller preset).
Scales up automatically on a GPU with no code change.

FALSIFICATION: if hybrid-topk (bounded reads) tracks the bounded models DOWN instead of
attention UP, then cheap top-k retrieval does NOT recover recall and the bounded-cost claim
fails (dense attention would be doing the work in M6, not content-addressing).
"""

from __future__ import annotations
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp_m6_crux_learned import make_batch, phi, VOCAB, VP, IGN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def causal_neg_inf(T, device):
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


class MixLayer(nn.Module):
    def __init__(self, d_model, d_head, kind, topk):
        super().__init__()
        self.kind, self.dh, self.topk = kind, d_head, topk
        self.q = nn.Linear(d_model, d_head, bias=False)
        self.k = nn.Linear(d_model, d_head, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        if kind == "hybrid_topk":
            self.q2 = nn.Linear(d_model, d_head, bias=False)

    def _linear(self, q, k, v):
        q, k = phi(q), phi(k)
        kv = torch.cumsum(torch.einsum("bti,btj->btij", v, k), dim=1)
        num = torch.einsum("btij,btj->bti", kv, q)
        den = torch.einsum("bti,bti->bt", torch.cumsum(k, 1), q).clamp_min(1e-4).unsqueeze(-1)
        return num / den

    def _attn_full(self, q, k, v):
        T = q.shape[1]
        att = (q @ k.transpose(1, 2)) / (self.dh ** 0.5)
        att = att.masked_fill(causal_neg_inf(T, q.device), float("-inf"))
        return F.softmax(att, dim=-1) @ v

    def _attn_topk(self, q, k, v, kk):
        """Sparse content-addressed read: softmax over only the top-kk causal entries.
        Reads/step = kk, independent of T. Returns (out, effective_reads_per_pos)."""
        B, T, _ = q.shape
        att = (q @ k.transpose(1, 2)) / (self.dh ** 0.5)
        att = att.masked_fill(causal_neg_inf(T, q.device), float("-inf"))
        kk = min(kk, T)
        vals, idx = att.topk(kk, dim=-1)                       # (B,T,kk)
        w = F.softmax(vals, dim=-1)                            # over top-kk only
        vg = v[torch.arange(B, device=q.device).view(B, 1, 1), idx]  # (B,T,kk,dv)
        out = (w.unsqueeze(-1) * vg).sum(dim=-2)
        return out, kk

    def forward(self, h, vsrc):
        q, k, v = self.q(h), self.k(h), self.v(vsrc)
        reads = None
        if self.kind == "linear":
            o = self._linear(q, k, v); reads = 1.0            # fixed state
        elif self.kind == "attn":
            o = self._attn_full(q, k, v); reads = float(h.shape[1])  # O(T)
        elif self.kind == "hybrid_topk":
            ws = self._linear(q, k, v)
            led, kk = self._attn_topk(self.q2(h), k, v, self.topk)
            o = ws + led; reads = float(kk)                    # bounded: k reads/step
        else:
            raise ValueError(self.kind)
        return o, reads


class Net(nn.Module):
    def __init__(self, kind, layers=1, d_model=64, d_head=32, topk=8):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d_model)
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(layers))
        self.mix = nn.ModuleList(MixLayer(d_model, d_head, kind, topk) for _ in range(layers))
        self.head = nn.Linear(d_model, VOCAB)
        self.last_reads = 0.0

    def forward(self, x):
        h = self.emb(x)
        reads = 0.0
        for norm, mix in zip(self.norms, self.mix):
            hn = norm(h)
            vsrc = torch.roll(hn, shifts=-1, dims=1)
            o, r = mix(hn, vsrc)
            h = h + o
            reads += r
        self.last_reads = reads
        return self.head(h)


def acc(model, x, y):
    with torch.no_grad():
        pred = model(x).argmax(-1)
        m = y != IGN
        return (pred[m] == y[m]).float().mean().item()


def train_eval(kind, D, steps, B, lr, layers, d_head, topk, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(kind, layers=layers, d_head=d_head, topk=topk).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    for _ in range(steps):
        x, y = make_batch(B, D, D, rng)
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = F.cross_entropy(net(x).reshape(-1, VOCAB), y.reshape(-1), ignore_index=IGN)
        opt.zero_grad(); loss.backward(); opt.step()
    xe, ye = make_batch(256, D, D, rng)
    return acc(net, xe.to(DEVICE), ye.to(DEVICE)), net.last_reads


def run(D_grid, steps, B, lr, layers, d_head, topk, kinds, seeds):
    print(f"\n=== M7: bounded-cost crux (device={DEVICE.type}, layers={layers}, "
          f"d_head={d_head}, top-k={topk}, steps={steps}, seeds={len(seeds)}) ===")
    print(f"query accuracy mean+/-std (chance={1.0/VP:.3f}); reads/step in [brackets]\n")
    hdr = f"{'D':>4} | " + " ".join(f"{k:>16}" for k in kinds)
    print(hdr); print("-" * len(hdr))
    table = {}
    for D in D_grid:
        cells, accs = [], {}
        for k in kinds:
            res = [train_eval(k, D, steps, B, lr, layers, d_head, topk, s) for s in seeds]
            a = np.array([r[0] for r in res]); reads = np.mean([r[1] for r in res])
            accs[k] = (a.mean(), a.std(), reads)
            cells.append(f"{a.mean():.3f}+/-{a.std():.3f}[{reads:.0f}]")
        table[D] = accs
        print(f"{D:>4} | " + " ".join(f"{c:>16}" for c in cells))

    Dmax = D_grid[-1]
    bounded = min(v[0] for k, v in table[Dmax].items() if k == "linear")
    htk = table[Dmax].get("hybrid_topk", (0, 0, 0))
    attn = table[Dmax].get("attn", (0, 0, 0))
    print(f"\nAt D={Dmax}: linear(bounded)={bounded:.3f}, "
          f"hybrid_topk={htk[0]:.3f} [reads/step={htk[2]:.0f}], attn={attn[0]:.3f} "
          f"[reads/step={attn[2]:.0f}]")
    ok = (htk[0] - bounded > 0.15) and (htk[0] >= attn[0] - 0.1) and (htk[2] < attn[2])
    print(f"M7 (bounded top-k retrieval recovers recall at reads/step << context): "
          f"{'SUPPORTED' if ok else 'CHECK'}")
    print("[DIRECTIONAL: still toy scale; auto-scales on GPU]")
    return table


def presets():
    if DEVICE.type == "cuda":
        return dict(D_grid=(16, 32, 64, 96, 128), steps=1500, B=64, lr=2e-3,
                    layers=2, d_head=48, topk=8,
                    kinds=("linear", "attn", "hybrid_topk"), seeds=(0, 1, 2))
    return dict(D_grid=(8, 24, 48, 64), steps=500, B=32, lr=2e-3,
                layers=1, d_head=32, topk=8,
                kinds=("linear", "attn", "hybrid_topk"), seeds=(0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = presets()
    if args.smoke:
        cfg.update(D_grid=(8, 24), steps=80, seeds=(0,))
    t0 = time.time()
    run(**cfg)
    print(f"\n[wall time: {time.time() - t0:.1f}s]")
