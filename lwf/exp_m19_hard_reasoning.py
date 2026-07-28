"""
Experiment M19 -- genuinely HARD (irreducibly global) reasoning: the crux test (G2.1).

Everything LWF has passed so far is recall (M2/M6/M7/M14), shallow composition (M5/M8), or
streamable aggregation. Those do NOT test the §7 crux, because a bounded state can do any STREAMING
computation. The crux needs an IRREDUCIBLY GLOBAL task -- one whose answer requires the whole input
held/compared simultaneously and is provably impossible for a fixed state past a size threshold.

TASK: in-context SORTING via order statistics. Input = N random values, then N output slots; at
output slot i the model must emit the i-th smallest value. Each slot is a global selection over all
N (an order statistic), and emitting a permutation of N items needs ~N*log2(N) bits of state -- so a
B-bit bounded Workspace MUST fail once N*log2(N) > B (Invariant I1). A full-context transformer has
no such bottleneck. The question: can a bounded Workspace + content-addressable Ledger do it?

Models (shared embed): attn (full-context transformer = strong baseline), bounded (normalized
fast-weight/linear Workspace, no Ledger), hybrid (bounded Workspace + top-k Ledger read over the N
value positions, learned query). Swept over N (the global-reasoning width).

=====================================================================================
PRE-REGISTERED VERDICT (fixed before run; reported verbatim; NOT reframable post-hoc).
  At the largest N tested, with CONFIG VALIDITY (attn >= 0.85 AND bounded <= 0.60):
    - IF hybrid >= 0.85*attn  -> LWF does global reasoning (crux SURVIVED at this scale)
    - IF hybrid < 0.85*attn but > bounded+0.15 -> PARTIAL (retrieval helps, doesn't close it)
    - IF hybrid <= bounded+0.15 -> CRUX CONFIRMED: bounded+retrieval cannot do global reasoning
  This is a genuine two-sided test: LWF can FAIL here, and that failure is the most informative
  possible outcome (it would falsify the strong form of the thesis). Reported whichever way it falls.
=====================================================================================
GPU. Per-position accuracy on the sorted output. Small scale (1660 Ti); a negative here is a
scale-qualified negative, not a universal one.
"""

from __future__ import annotations
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A = 32                 # value alphabet
OUT = A                # output-slot marker token
VOCAB = A + 1
IGN = -100


def phi(x):
    return F.elu(x) + 1.0


def make_batch(B, N, rng):
    """[v1..vN, OUT*N]; target at output slot i = i-th smallest value. seq len = 2N."""
    L = 2 * N
    x = np.full((B, L), OUT, dtype=np.int64)
    y = np.full((B, L), IGN, dtype=np.int64)
    for b in range(B):
        vals = rng.integers(0, A, size=N)
        x[b, :N] = vals
        y[b, N:2 * N] = np.sort(vals)         # slot i predicts i-th smallest
    return torch.from_numpy(x).to(DEVICE), torch.from_numpy(y).to(DEVICE)


def causal_mask(L, device):
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), 1)


class Model(nn.Module):
    def __init__(self, kind, d=96, dh=64, topk=8, layers=2, heads=6):
        super().__init__()
        self.kind, self.d, self.dh, self.topk = kind, d, dh, topk
        self.emb = nn.Embedding(VOCAB, d); self.pos = nn.Embedding(512, d)
        if kind == "attn":
            enc = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True,
                                             activation="gelu", dropout=0.0)
            self.tf = nn.TransformerEncoder(enc, layers)
        else:
            self.Wk = nn.Linear(d, dh, bias=False); self.Wv = nn.Linear(d, d, bias=False)
            self.Wq = nn.Linear(d, dh, bias=False); self.Wg = nn.Linear(d, dh)
            nn.init.constant_(self.Wg.bias, 3.0)
            if kind == "hybrid":
                self.bk = nn.Linear(d, dh, bias=False); self.bv = nn.Linear(d, d, bias=False)
                self.qr = nn.Linear(2 * d, dh)
        nread = 1 if kind == "hybrid" else 0
        base = d if kind == "attn" else (2 + nread) * d
        self.head = nn.Sequential(nn.Linear(base, 2 * d), nn.GELU(), nn.Linear(2 * d, VOCAB))

    def _bounded(self, e):
        B, L, _ = e.shape
        vsrc = torch.roll(e, shifts=-1, dims=1)
        M = torch.zeros(B, self.d, self.dh, device=e.device)
        z = torch.zeros(B, self.dh, device=e.device)
        outs = []
        for t in range(L):
            x = e[:, t]
            k, q = phi(self.Wk(x)), phi(self.Wq(x)); v = self.Wv(vsrc[:, t])
            g = torch.sigmoid(self.Wg(x))
            M = g[:, None, :] * M + torch.einsum("bi,bj->bij", v, k)
            z = g * z + k
            den = (z * q).sum(-1, keepdim=True).clamp_min(1e-4)
            outs.append(torch.einsum("bij,bj->bi", M, q) / den)
        return torch.stack(outs, dim=1)

    def forward(self, x, N):
        B, L = x.shape
        e = self.emb(x) + self.pos(torch.arange(L, device=x.device))[None]
        if self.kind == "attn":
            return self.head(self.tf(e, mask=causal_mask(L, x.device)))
        ws = self._bounded(e)
        if self.kind == "bounded":
            return self.head(torch.cat([e, ws], dim=-1))
        # hybrid: content-addressable Ledger over the N value positions
        bank_k = self.bk(e[:, :N]); bank_v = self.bv(e[:, :N])       # (B,N,*)
        q = self.qr(torch.cat([e, ws], dim=-1))                     # (B,L,dh)
        att = torch.einsum("bld,bnd->bln", q, bank_k) / (self.dh ** 0.5)
        kk = min(self.topk, N)
        vals, idx = att.topk(kk, dim=-1)
        w = F.softmax(vals, dim=-1)
        bv = bank_v.unsqueeze(1).expand(B, L, N, self.d)
        gathered = torch.gather(bv, 2, idx.unsqueeze(-1).expand(B, L, kk, self.d))
        ledger = (w.unsqueeze(-1) * gathered).sum(dim=-2)
        return self.head(torch.cat([e, ws, ledger], dim=-1))


def acc(logits, y):
    with torch.no_grad():
        m = y != IGN
        return (logits.argmax(-1)[m] == y[m]).float().mean().item()


def train_eval(kind, N, steps, B, lr, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = Model(kind).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    for _ in range(steps):
        x, y = make_batch(B, N, rng)
        loss = F.cross_entropy(model(x, N).reshape(-1, VOCAB), y.reshape(-1), ignore_index=IGN)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    xe, ye = make_batch(256, N, rng)
    return acc(model(xe, N), ye)


def run(N_grid=(8, 16), steps=3000, B=128, lr=2e-3, seeds=(0,)):
    print(f"\n=== M19: irreducibly-global reasoning (in-context SORTING, device={DEVICE.type}, "
          f"alphabet={A}, steps={steps}) ===")
    print(f"per-slot accuracy on the sorted output (chance=1/{A}={1/A:.3f})\n")
    hdr = f"{'N (width)':>10} | {'attn':>7} | {'bounded':>8} | {'hybrid':>7}"
    print(hdr); print("-" * len(hdr))
    res = {}
    for N in N_grid:
        row = {k: float(np.mean([train_eval(k, N, steps, B, lr, s) for s in seeds]))
               for k in ("attn", "bounded", "hybrid")}
        res[N] = row
        print(f"{N:>10} | {row['attn']:>7.3f} | {row['bounded']:>8.3f} | {row['hybrid']:>7.3f}")

    Nmax = N_grid[-1]; r = res[Nmax]
    print("\n--- PRE-REGISTERED VERDICT (fixed before run) ---")
    valid = (r["attn"] >= 0.85) and (r["bounded"] <= 0.60)
    print(f"  config valid (attn>=0.85 & bounded<=0.60) @N={Nmax}: "
          f"attn={r['attn']:.3f}, bounded={r['bounded']:.3f} -> {valid}")
    if not valid:
        verdict = "INCONCLUSIVE (config invalid -- baseline didn't work or bounded didn't fail)"
    elif r["hybrid"] >= 0.85 * r["attn"]:
        verdict = "CRUX SURVIVED: bounded Workspace + Ledger does global reasoning at this scale"
    elif r["hybrid"] > r["bounded"] + 0.15:
        verdict = "PARTIAL: retrieval helps but does NOT close global reasoning (residual gap)"
    else:
        verdict = "CRUX CONFIRMED: bounded+retrieval CANNOT do irreducibly-global reasoning"
    print(f"  hybrid={r['hybrid']:.3f} vs 0.85*attn={0.85*r['attn']:.3f}, bounded={r['bounded']:.3f}")
    print(f"  M19 VERDICT: {verdict}")
    print("  [Two-sided test: LWF can FAIL here, and that would be the most informative outcome.]")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    run(N_grid=((6,) if args.smoke else (8, 16)), steps=(200 if args.smoke else 3000))
    print(f"[wall {time.time()-t0:.0f}s]")
