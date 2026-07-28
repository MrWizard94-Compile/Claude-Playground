"""
Experiment M18 -- the FULL integrated stack vs a WORKING strong baseline (closes G0.2 for the stack).

M15 failed to establish a strong-baseline win because its stateful task defeated the transformer.
M7 showed a bounded LINEAR-attention Workspace + top-k Ledger matches full attention on MQAR -- but
that used the linear-attention stand-in, not the real fast-weight Workspace (the M1 object). M18
tests the actual thing: a trained FAST-WEIGHT Workspace + a content-addressable Ledger (top-k read
via a query formed from the Workspace state) as one system, on MQAR -- a task a full-context
transformer provably SOLVES (so the baseline is real, unlike M15), while a bounded Workspace alone
degrades (so the Ledger is needed).

Models (shared embed; matched d_model):
  attn    -- full-context transformer (STRONG baseline; M7 confirms it reaches ~1.0)  [reused from M7]
  linear  -- bounded linear-attention Workspace, no Ledger (should degrade with #pairs D) [M7]
  fw      -- bounded FAST-WEIGHT Workspace, no Ledger (the real M1 object; degrades with D)
  fw+ldg  -- fast-weight Workspace + top-k content-addressable Ledger (the FULL integrated stack)

=====================================================================================
PRE-REGISTERED VERDICT (fixed before run; reported verbatim; NOT reframable post-hoc).
  CONFIG VALIDITY (baseline must actually work AND Ledger must be needed), at the largest D:
     attn >= 0.90   AND   fw(bounded) <= 0.70
  PRIMARY: fw+ldg >= 0.90 * attn   at bounded read cost (k reads/step << context length)
  PASS = config-valid AND primary. Else HONEST NEGATIVE.
=====================================================================================
GPU; auto-scales via exp_m7.DEVICE. On MQAR the Workspace is partly redundant (pure recall) -- noted
honestly; the point is that the full stack WITH a real fast-weight Workspace present matches the
working baseline at bounded cost.
"""

from __future__ import annotations
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp_m6_crux_learned import make_batch, phi, VOCAB, VP, IGN
from exp_m7_scaleup import DEVICE, Net   # Net(kind='attn'|'linear') = proven baselines


class FWIntegrated(nn.Module):
    """Fast-weight Workspace (real M1 object) + content-addressable top-k Ledger + head."""
    def __init__(self, d_model=64, d_head=48, topk=8, use_ledger=True):
        super().__init__()
        self.d_model, self.dh, self.topk = d_model, d_head, topk
        self.use_ledger = use_ledger
        self.emb = nn.Embedding(VOCAB, d_model)
        self.pos = nn.Embedding(512, d_model)
        # fast-weight Workspace projections
        self.Wk = nn.Linear(d_model, d_head, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wq = nn.Linear(d_model, d_head, bias=False)
        self.Wg = nn.Linear(d_model, d_head)
        # init the forget gate HIGH so the fast-weight state RETAINS by default (sigmoid(+3)=0.95),
        # instead of halving every step (sigmoid(0)=0.5 -> vanishes over the sequence). Without this
        # the fast-weight Workspace forgets too fast and fails MQAR even at small D.
        nn.init.constant_(self.Wg.bias, 3.0)
        # Ledger read: learned query over the D content-addressable bank entries
        self.bk = nn.Linear(d_model, d_head, bias=False)      # bank key proj
        self.bv = nn.Linear(d_model, d_model, bias=False)     # bank value proj
        self.qr = nn.Linear(2 * d_model, d_head)              # query from [token, ws_read]
        nread = 1 if use_ledger else 0
        self.head = nn.Sequential(nn.Linear((2 + nread) * d_model, 2 * d_model), nn.GELU(),
                                  nn.Linear(2 * d_model, VOCAB))

    def _workspace(self, e):
        """Gated fast-weight recurrence -> per-position read (B,L,d_model). The VALUE is taken from
        the NEXT-token stream (shifted-value construction) so a key position stores key->next-value
        -- the association MQAR needs. Without this the state stores token->itself and cannot recall."""
        B, L, _ = e.shape
        vsrc = torch.roll(e, shifts=-1, dims=1)               # next-token stream = value carrier
        M = torch.zeros(B, self.d_model, self.dh, device=e.device)
        z = torch.zeros(B, self.dh, device=e.device)          # normalizer state (gated sum of k)
        outs = []
        for t in range(L):
            x = e[:, t]
            k, q = phi(self.Wk(x)), phi(self.Wq(x))
            v = self.Wv(vsrc[:, t])
            g = torch.sigmoid(self.Wg(x))
            M = g[:, None, :] * M + torch.einsum("bi,bj->bij", v, k)
            z = g * z + k
            num = torch.einsum("bij,bj->bi", M, q)
            den = (z * q).sum(-1, keepdim=True).clamp_min(1e-4)   # linear-attention normalizer
            outs.append(num / den)
        return torch.stack(outs, dim=1)

    def forward(self, x, D):
        B, L = x.shape
        e = self.emb(x) + self.pos(torch.arange(L, device=x.device))[None]
        ws = self._workspace(e)                                # (B,L,d_model)
        if not self.use_ledger:
            return self.head(torch.cat([e, ws], dim=-1))
        bank_k = self.bk(e[:, 0:2 * D:2])                      # (B,D,dh) context keys
        bank_v = self.bv(e[:, 1:2 * D:2])                      # (B,D,d_model) context values
        ctx = torch.cat([e, ws], dim=-1)
        q = self.qr(ctx)                                       # (B,L,dh) learned query
        att = torch.einsum("bld,bnd->bln", q, bank_k) / (self.dh ** 0.5)   # (B,L,D)
        kk = min(self.topk, D)
        vals, idx = att.topk(kk, dim=-1)
        w = F.softmax(vals, dim=-1)
        bv_exp = bank_v.unsqueeze(1).expand(B, L, D, self.d_model)
        gathered = torch.gather(bv_exp, 2, idx.unsqueeze(-1).expand(B, L, kk, self.d_model))
        ledger = (w.unsqueeze(-1) * gathered).sum(dim=-2)      # (B,L,d_model), k reads/step
        return self.head(torch.cat([e, ws, ledger], dim=-1))


def acc(logits, y):
    with torch.no_grad():
        m = y != IGN
        return (logits.argmax(-1)[m] == y[m]).float().mean().item()


def train_eval(kind, D, steps, B, lr, d_head, topk, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    if kind in ("attn", "linear"):
        model = Net(kind, d_head=d_head, topk=topk).to(DEVICE)
        need_D = False
    else:
        model = FWIntegrated(d_head=d_head, topk=topk, use_ledger=(kind == "fw+ldg")).to(DEVICE)
        need_D = True
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        x, y = make_batch(B, D, D, rng)
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x, D) if need_D else model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), ignore_index=IGN)
        opt.zero_grad(); loss.backward(); opt.step()
    xe, ye = make_batch(256, D, D, rng)
    xe, ye = xe.to(DEVICE), ye.to(DEVICE)
    logits = model(xe, D) if need_D else model(xe)
    reads = {"attn": float(2 * D), "linear": 1.0, "fw": 1.0, "fw+ldg": float(topk)}[kind]
    return acc(logits, ye), reads


def run(D=48, steps=2500, B=256, lr=2e-3, d_head=48, topk=8, seeds=(0,)):
    print(f"\n=== M18: full fast-weight stack vs working baseline (MQAR, device={DEVICE.type}, "
          f"D={D} pairs, steps={steps}) ===")
    print(f"query accuracy (chance=1/{VP}={1/VP:.3f}); reads/step in [brackets]\n")
    kinds = ("attn", "linear", "fw", "fw+ldg")
    res = {}
    for k in kinds:
        r = [train_eval(k, D, steps, B, lr, d_head, topk, s) for s in seeds]
        res[k] = (float(np.mean([x[0] for x in r])), r[0][1])
        print(f"  {k:>8}: acc={res[k][0]:.3f}  [{res[k][1]:.0f} reads/step]")

    attn, lin, fw, fwl = res["attn"][0], res["linear"][0], res["fw"][0], res["fw+ldg"][0]
    print("\n--- PRE-REGISTERED VERDICT (fixed before run) ---")
    valid = (attn >= 0.90) and (fw <= 0.70)
    primary = fwl >= 0.90 * attn
    print(f"  config valid (attn>=0.90 & fw-bounded<=0.70): attn={attn:.3f}, fw={fw:.3f} -> {valid}")
    print(f"  PRIMARY fw+ldg >= 0.9*attn: {fwl:.3f} >= {0.9*attn:.3f} "
          f"[{res['fw+ldg'][1]:.0f} reads vs attn {res['attn'][1]:.0f}] -> {primary}")
    ok = valid and primary
    print(f"  M18 VERDICT: {'PASS' if ok else 'FAIL/PARTIAL (honest negative)'}")
    print("  [full fast-weight Workspace + top-k Ledger vs a transformer baseline that ACTUALLY works.]")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    run(D=48, steps=(120 if args.smoke else 2500))
    print(f"[wall {time.time()-t0:.0f}s]")
