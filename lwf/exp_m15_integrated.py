"""
Experiment M15 -- the FIRST end-to-end integrated LWF system, learned. (Attacks G1.1/G1.2/
G1.3/G0.2/G0.1 from GAPS.md in one honest experiment.)

Every prior mechanism tested ONE tier in isolation. M15 assembles them: a trained gated
FAST-WEIGHT Workspace (the real M1 object, not a linear-attention stand-in -> G1.2), an
external keyed LEDGER written on SET events and read via a LEARNED query formed from the
Workspace state (no oracle -> G1.3), and a compositional output head -- trained jointly,
end-to-end (-> G1.1). It is measured against a STRONG baseline: a full-context transformer
with no information bottleneck (-> G0.2), and against a bounded-Workspace-only ablation.

TASK (not pure lookup -> addresses G0.3): mutable variable tracking + comparison over a long
token stream. Events: SET v=x (variables are REASSIGNED, latest wins -> tests write policy),
QUERY v (recall current value), COMPARE v w (retrieve TWO current values and compute which is
larger -> composition). The number of distinct live variables exceeds the Workspace capacity,
so the Ledger is NECESSARY; recency matters, so the transformer needs positions.

=====================================================================================
PRE-REGISTERED VERDICT (written BEFORE running; reported verbatim whatever the outcome).
G0.1 discipline: no post-hoc threshold changes. If it fails, it is logged as a HONEST NEGATIVE.

  Let T = full-context transformer, L = integrated LWF (learned query), B = bounded-only.
  Config is chosen so the bounded-only ablation is genuinely bottlenecked.

  PRIMARY  (integration + strong baseline): on COMPARE (the compositional queries), at a config
           where bounded-only B_compare <= 0.60, the integrated LWF reaches
                 L_compare >= 0.90 * T_compare.
  SECONDARY(query formation is learnable, G1.3): learned-query LWF reaches
                 L_compare >= 0.90 * Oracle_compare   (oracle = reads the correct slots directly).
  ALSO REPORTED: QUERY accuracy for all; whether B is actually bottlenecked (else config invalid).

  PASS = PRIMARY and SECONDARY both hold AND B_compare <= 0.60 (config validity).
  Anything else is reported honestly (partial / negative), not reframed.
=====================================================================================
Scale is toy-small (1660 Ti). Single integrated instance; the eviction controller (M12/M13) is
validated separately and NOT re-stressed here (Ledger capacity = #vars). Seeds fixed.
"""

from __future__ import annotations
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# token ids
PAD, SET, QUERY, COMPARE, ANS = 0, 1, 2, 3, 4
BASE = 5

# answer label space: [0..U) = value classes (QUERY); [U..U+3) = relation classes (COMPARE)
IGN = -100


class Task:
    def __init__(self, n_vars=32, U=8, L=120, p_set=0.6):
        self.n_vars, self.U, self.L, self.p_set = n_vars, U, L, p_set
        self.var0 = BASE
        self.val0 = BASE + n_vars
        self.vocab = BASE + n_vars + U
        self.n_ans = U + 3

    def gen(self, B, rng):
        n_vars, U, L = self.n_vars, self.U, self.L
        toks = np.zeros((B, L), np.int64)
        tgt = np.full((B, L), IGN, np.int64)
        wvar = np.full((B, L), -1, np.int64)          # SET-value position -> slot to write
        q1 = np.full((B, L), -1, np.int64)            # ANS position -> queried var (oracle)
        q2 = np.full((B, L), -1, np.int64)            # ANS position -> 2nd var (COMPARE)
        qtype = np.full((B, L), -1, np.int64)         # 0=QUERY, 1=COMPARE at ANS
        for b in range(B):
            cur = {}                                   # var -> value
            t = 0
            while t < L:
                assigned = list(cur.keys())
                r = rng.random()
                if r < self.p_set or len(assigned) < 2:
                    if t + 3 > L: break
                    v = int(rng.integers(n_vars)); x = int(rng.integers(U))
                    toks[b, t] = SET; toks[b, t + 1] = self.var0 + v
                    toks[b, t + 2] = self.val0 + x
                    wvar[b, t + 2] = v                 # write value emb into slot v here
                    cur[v] = x
                    t += 3
                elif r < self.p_set + 0.25:
                    if t + 3 > L: break
                    v = int(rng.choice(assigned))
                    toks[b, t] = QUERY; toks[b, t + 1] = self.var0 + v; toks[b, t + 2] = ANS
                    tgt[b, t + 2] = cur[v]             # value class
                    q1[b, t + 2] = v; qtype[b, t + 2] = 0
                    t += 3
                else:
                    if t + 4 > L: break
                    v, w = [int(x) for x in rng.choice(assigned, size=2, replace=False)]
                    toks[b, t] = COMPARE; toks[b, t + 1] = self.var0 + v
                    toks[b, t + 2] = self.var0 + w; toks[b, t + 3] = ANS
                    rel = 0 if cur[v] > cur[w] else (1 if cur[w] > cur[v] else 2)
                    tgt[b, t + 3] = U + rel
                    q1[b, t + 3] = v; q2[b, t + 3] = w; qtype[b, t + 3] = 1
                    t += 4
        to = lambda a: torch.from_numpy(a).to(DEVICE)
        return to(toks), to(tgt), to(wvar), to(q1), to(q2), to(qtype)


def phi(x):
    return F.elu(x) + 1.0


class LWF(nn.Module):
    """Integrated: trained fast-weight Workspace + keyed Ledger + learned-query read + head."""
    def __init__(self, task, d=48, use_ledger=True, oracle_query=False):
        super().__init__()
        self.task, self.d = task, d
        self.use_ledger = use_ledger
        self.oracle_query = oracle_query
        self.emb = nn.Embedding(task.vocab, d)
        self.pos = nn.Embedding(task.L, d)
        # Workspace (gated fast-weight) projections
        self.Wk = nn.Linear(d, d, bias=False); self.Wv = nn.Linear(d, d, bias=False)
        self.Wq = nn.Linear(d, d, bias=False); self.Wg = nn.Linear(d, d)
        # Ledger: learned per-slot key embeddings; two learned read-query heads
        self.slot_key = nn.Embedding(task.n_vars, d)
        self.Wr1 = nn.Linear(2 * d, d); self.Wr2 = nn.Linear(2 * d, d)
        n_read = 2 if use_ledger else 0
        self.head = nn.Sequential(nn.Linear((2 + n_read) * d, 2 * d), nn.GELU(),
                                  nn.Linear(2 * d, task.n_ans))

    def forward(self, toks, wvar, q1, q2):
        B, L = toks.shape; d = self.d
        e = self.emb(toks) + self.pos(torch.arange(L, device=toks.device))[None]
        M = torch.zeros(B, d, d, device=toks.device)             # fast-weight state (dv,dk)
        slots = torch.zeros(B, self.task.n_vars, d, device=toks.device)
        keys = self.slot_key(torch.arange(self.task.n_vars, device=toks.device))  # (n_vars,d)
        outs = []
        for t in range(L):
            x = e[:, t]
            k, v, qw = phi(self.Wk(x)), self.Wv(x), phi(self.Wq(x))
            g = torch.sigmoid(self.Wg(x))                        # (B,d) gate on dk axis
            M = g[:, None, :] * M + torch.einsum("bi,bj->bij", v, k)
            ws_read = torch.einsum("bij,bj->bi", M, qw)          # (B,d)
            if self.use_ledger:
                # write value-embedding into the slot for SET-value positions
                wt = wvar[:, t]                                  # (B,) slot or -1
                m = (wt >= 0).float()[:, None]
                oh = F.one_hot(wt.clamp(min=0), self.task.n_vars).float() * m  # (B,n_vars)
                slots = slots * (1 - oh[:, :, None]) + oh[:, :, None] * x[:, None, :]
                if self.oracle_query:                            # read correct slots directly
                    idx1 = q1[:, t].clamp(min=0); idx2 = q2[:, t].clamp(min=0)
                    r1 = slots[torch.arange(B, device=toks.device), idx1]
                    r2 = slots[torch.arange(B, device=toks.device), idx2]
                else:                                            # LEARNED query from state
                    ctx = torch.cat([x, ws_read], dim=-1)
                    a1 = torch.softmax(self.Wr1(ctx) @ keys.T, dim=-1)   # (B,n_vars) learned query
                    a2 = torch.softmax(self.Wr2(ctx) @ keys.T, dim=-1)
                    r1 = torch.einsum("bn,bnd->bd", a1, slots)
                    r2 = torch.einsum("bn,bnd->bd", a2, slots)
                outs.append(self.head(torch.cat([x, ws_read, r1, r2], dim=-1)))
            else:
                outs.append(self.head(torch.cat([x, ws_read], dim=-1)))
        return torch.stack(outs, dim=1)                          # (B,L,n_ans)


class Transformer(nn.Module):
    """Strong baseline: full-context causal transformer (no information bottleneck)."""
    def __init__(self, task, d=48, layers=3, heads=4):
        super().__init__()
        self.task = task
        self.emb = nn.Embedding(task.vocab, d); self.pos = nn.Embedding(task.L, d)
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, batch_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, task.n_ans)

    def forward(self, toks, *_):
        B, L = toks.shape
        h = self.emb(toks) + self.pos(torch.arange(L, device=toks.device))[None]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=toks.device), 1)
        return self.head(self.tf(h, mask=mask))


def split_acc(logits, tgt, qtype, U):
    with torch.no_grad():
        pred = logits.argmax(-1)
        out = {}
        for name, code in (("QUERY", 0), ("COMPARE", 1)):
            m = (qtype == code) & (tgt != IGN)
            out[name] = ((pred[m] == tgt[m]).float().mean().item() if m.any() else float("nan"))
        return out


def train_eval(kind, task, steps, Bsz, lr, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    if kind == "transformer":
        model = Transformer(task).to(DEVICE)
    else:
        model = LWF(task, use_ledger=(kind != "bounded"),
                    oracle_query=(kind == "oracle")).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        toks, tgt, wvar, q1, q2, qtype = task.gen(Bsz, rng)
        logits = model(toks, wvar, q1, q2)
        loss = F.cross_entropy(logits.reshape(-1, task.n_ans), tgt.reshape(-1), ignore_index=IGN)
        opt.zero_grad(); loss.backward(); opt.step()
    toks, tgt, wvar, q1, q2, qtype = task.gen(512, rng)
    logits = model(toks, wvar, q1, q2)
    return split_acc(logits, tgt, qtype, task.U)


def run(n_vars=32, U=8, L=120, d=48, steps=2500, Bsz=64, lr=2e-3, seeds=(0,)):
    task = Task(n_vars=n_vars, U=U, L=L)
    print(f"\n=== M15: integrated LWF, end-to-end learned "
          f"(device={DEVICE.type}, n_vars={n_vars}, U={U}, L={L}, d={d}, steps={steps}) ===")
    print(f"chance: QUERY=1/{U}={1/U:.3f}, COMPARE=1/3=0.333\n")
    kinds = ["transformer", "bounded", "oracle", "lwf"]
    res = {}
    for k in kinds:
        accs = [train_eval(k, task, steps, Bsz, lr, s) for s in seeds]
        res[k] = {m: float(np.mean([a[m] for a in accs])) for m in ("QUERY", "COMPARE")}
        print(f"  {k:>12}:  QUERY={res[k]['QUERY']:.3f}   COMPARE={res[k]['COMPARE']:.3f}")

    T, Bd, O, Lw = res["transformer"], res["bounded"], res["oracle"], res["lwf"]
    print("\n--- PRE-REGISTERED VERDICT (thresholds fixed before run) ---")
    config_valid = Bd["COMPARE"] <= 0.60
    primary = Lw["COMPARE"] >= 0.90 * T["COMPARE"]
    secondary = Lw["COMPARE"] >= 0.90 * O["COMPARE"]
    print(f"  config validity (bounded_COMPARE <= 0.60): {Bd['COMPARE']:.3f} -> {config_valid}")
    print(f"  PRIMARY  L>=0.9*T on COMPARE: {Lw['COMPARE']:.3f} >= {0.9*T['COMPARE']:.3f} -> {primary}")
    print(f"  SECONDARY L>=0.9*Oracle     : {Lw['COMPARE']:.3f} >= {0.9*O['COMPARE']:.3f} -> {secondary}")
    verdict = "PASS" if (config_valid and primary and secondary) else "FAIL/PARTIAL (honest negative)"
    print(f"  M15 VERDICT: {verdict}")
    print(f"  [integration works: LWF QUERY {Lw['QUERY']:.3f} vs bounded {Bd['QUERY']:.3f} vs "
          f"transformer {T['QUERY']:.3f}]")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if args.smoke:
        run(n_vars=16, U=8, L=48, d=32, steps=150)
    else:
        run()
    print(f"[wall time: {time.time()-t0:.1f}s]")
