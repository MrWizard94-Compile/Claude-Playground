"""
Experiment M8 -- Stage 0.5: in-context MULTI-HOP reasoning (the global-reasoning crux).

M6/M7 tested flat recall (MQAR): match a key, copy its value. That is NOT the hard part of
the §7 crux. The hard part is GLOBAL / COMPOSITIONAL reasoning: chaining several associations
that are all present in-context. A skeptic's strongest claim is that composition needs
simultaneous all-pairs interaction over a large active set -- exactly what a bounded state +
bounded top-k read is supposed to be unable to do.

TASK: a random functional graph over a domain of D nodes is listed in-context as edge pairs
[a, succ(a)] (order shuffled = distractors relative to any query). A query gives a start node;
the model must output the node H HOPS away. H=1 is MQAR; H>1 requires composing H retrievals.
The graph is re-randomised every example, so nothing is memorised -- only the ALGORITHM
(chase the pointer H times through in-context edges) generalises.

MODELS (equal depth L): linear (bounded state), attn (full O(T) reads), hybrid_topk
(bounded state + k reads/step). Multi-hop needs depth ~ H (each layer composes one hop).

THE QUESTION: does hybrid_topk compose multi-hop AS WELL AS full attention at equal depth,
using only k bounded reads/step?
  - If yes across H: the global-reasoning worry is substantially answered at this scale.
  - If hybrid_topk tracks attn on H=1 but FALLS BEHIND as H grows: we've located the residual
    capability class -- composition needs something per-step bounded retrieval doesn't provide.
Either outcome is a real, reportable finding. This is the experiment the GPU is for.

Hardware-agnostic; auto-scales (bigger preset on CUDA). Directional, toy-to-small scale.
"""

from __future__ import annotations
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp_m7_scaleup import MixLayer, DEVICE

IGN = -100


def make_multihop_batch(B, D, H, n_nodes, rng):
    """Sequence = shuffled edge pairs [a, succ(a)] for a domain of D nodes, then [start].
    Label at the final position = the node H hops from start. Everything else = IGN."""
    T = 2 * D + 1
    x = np.zeros((B, T), dtype=np.int64)
    y = np.full((B, T), IGN, dtype=np.int64)
    for b in range(B):
        domain = rng.choice(n_nodes, size=D, replace=False)
        succ_idx = rng.integers(0, D, size=D)             # successor as index into domain
        succ = domain[succ_idx]                           # functional graph within domain
        order = rng.permutation(D)                        # shuffle edge listing = distractors
        seq = np.empty(2 * D, dtype=np.int64)
        seq[0::2] = domain[order]
        seq[1::2] = succ[order]
        x[b, :2 * D] = seq
        s_local = rng.integers(0, D)                      # start (as domain index)
        cur = s_local
        for _ in range(H):                                # walk H hops
            cur = succ_idx[cur]
        x[b, 2 * D] = domain[s_local]
        y[b, 2 * D] = domain[cur]
    return torch.from_numpy(x), torch.from_numpy(y)


class NetM8(nn.Module):
    def __init__(self, kind, n_nodes, layers, d_model=96, d_head=48, topk=8):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, d_model)
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(layers))
        self.mix = nn.ModuleList(MixLayer(d_model, d_head, kind, topk) for _ in range(layers))
        self.head = nn.Linear(d_model, n_nodes)
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


def final_acc(model, x, y, n_nodes):
    with torch.no_grad():
        pred = model(x).argmax(-1)
        m = y != IGN
        return (pred[m] == y[m]).float().mean().item()


def train_eval(kind, D, H, n_nodes, layers, steps, B, lr, d_head, topk, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    net = NetM8(kind, n_nodes, layers, d_head=d_head, topk=topk).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    for _ in range(steps):
        x, y = make_multihop_batch(B, D, H, n_nodes, rng)
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = F.cross_entropy(net(x).reshape(-1, n_nodes), y.reshape(-1), ignore_index=IGN)
        opt.zero_grad(); loss.backward(); opt.step()
    xe, ye = make_multihop_batch(512, D, H, n_nodes, rng)
    return final_acc(net, xe.to(DEVICE), ye.to(DEVICE), n_nodes), net.last_reads


def run(H_grid, D, n_nodes, layers, steps, B, lr, d_head, topk, kinds, seeds):
    print(f"\n=== M8: in-context multi-hop (device={DEVICE.type}, layers={layers}, D={D} "
          f"nodes/graph, n_nodes={n_nodes}, top-k={topk}, steps={steps}, seeds={len(seeds)}) ===")
    print(f"final-position accuracy mean+/-std (chance={1.0/n_nodes:.3f}); [reads/step]\n")
    hdr = f"{'H':>3} | " + " ".join(f"{k:>16}" for k in kinds)
    print(hdr); print("-" * len(hdr))
    table = {}
    for H in H_grid:
        cells, accs = [], {}
        for k in kinds:
            res = [train_eval(k, D, H, n_nodes, layers, steps, B, lr, d_head, topk, s)
                   for s in seeds]
            a = np.array([r[0] for r in res]); reads = np.mean([r[1] for r in res])
            accs[k] = (a.mean(), a.std(), reads)
            cells.append(f"{a.mean():.3f}+/-{a.std():.3f}[{reads:.0f}]")
        table[H] = accs
        print(f"{H:>3} | " + " ".join(f"{c:>16}" for c in cells))

    # Honest per-H verdict. The composition claim lives at H>1 (H=1 is flat recall).
    # For each multi-hop H classify: does bounded-cost retrieval (hybrid) both (a) beat the
    # bounded state and (b) match full attention? If attention ITSELF is near the bounded
    # baseline at some H, that H is a SHARED depth/scale ceiling, not an LWF-specific loss.
    print()
    composed, shared_ceiling = [], []
    for H in H_grid:
        if H == 1:
            continue
        lin = table[H]["linear"][0]; at = table[H]["attn"][0]; hy = table[H]["hybrid_topk"][0]
        beats_bounded = hy - lin > 0.10
        matches_attn = hy >= at - 0.05
        attn_also_stuck = at - lin < 0.10
        if attn_also_stuck:
            shared_ceiling.append(H)
            tag = "shared ceiling (attn also ~= bounded -> scale/depth limit, not LWF)"
        elif beats_bounded and matches_attn:
            composed.append(H)
            tag = "COMPOSED (hybrid ~= attn, both > bounded, at bounded read cost)"
        elif beats_bounded:
            tag = "PARTIAL (hybrid > bounded but trails attn -> residual class)"
        else:
            tag = "no separation"
        print(f"  H={H}: linear={lin:.3f}  attn={at:.3f}[{table[H]['attn'][2]:.0f}]  "
              f"hybrid={hy:.3f}[{table[H]['hybrid_topk'][2]:.0f}]  -> {tag}")
    if composed:
        v = f"SUPPORTED at H={composed} (bounded-cost retrieval composes as well as attention)"
        if shared_ceiling:
            v += f"; H={shared_ceiling} is a shared depth/scale ceiling (attn fails too)"
    else:
        v = "INCONCLUSIVE (no multi-hop H showed retrieval composing above bounded state)"
    print(f"M8: {v}")
    print("[DIRECTIONAL: toy-to-small scale, 1660 Ti]")
    return table


def presets():
    if DEVICE.type == "cuda":
        # tractable on a 1660 Ti (~50 min): H=1..3, 2 seeds for error bars. H=3 probes the
        # composition-depth ceiling at this scale (may stay low for all models -- honest).
        return dict(H_grid=(1, 2, 3), D=20, n_nodes=128, layers=5, steps=2000, B=384,
                    lr=2e-3, d_head=48, topk=8,
                    kinds=("linear", "attn", "hybrid_topk"), seeds=(0, 1))
    return dict(H_grid=(1, 2, 3), D=16, n_nodes=96, layers=4, steps=800, B=32, lr=2e-3,
                d_head=48, topk=8, kinds=("linear", "attn", "hybrid_topk"), seeds=(0,))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = presets()
    if args.smoke:
        cfg.update(H_grid=(1, 2), steps=150, seeds=(0,))
    t0 = time.time()
    run(**cfg)
    print(f"\n[wall time: {time.time() - t0:.1f}s]")
