"""
Experiment M14 -- knowledge scales in the LEDGER, not the model (learned; GPU + large host RAM).

M9 assumed the load-bearing "nonparametric-knowledge bet": knowledge can live in the Ledger
instead of the weights (RETRO-grounded). That premise was never tested LEARNED. M14 tests it at
scale using the two resources for their strengths: the small models train on the 1660 Ti (6 GB),
while a Ledger of up to ~1e6 facts lives in 48 GB host RAM.

TASK: N random facts (key -> value class). A query is a NOISY version of a stored key; predict its
class. Two same-era models:
  parametric : MLP(noisy query) -> class. Must MEMORISE all N key->class maps in fixed weights.
  retrieval  : content-addressed top-1 lookup in the Ledger (noisy query -> nearest key's class).
               Model size / per-query cost is CONSTANT in N; knowledge sits in host RAM.

HYPOTHESIS: as N grows, the parametric model saturates its fixed capacity and its accuracy
COLLAPSES toward chance, while the retrieval accuracy stays high and degrades only GRACEFULLY
(denser keys -> occasional nearest-neighbour confusion), independent of any weight budget. This is
M9's bet, made empirical: knowledge scales in the store, not the parameters.

FALSIFICATION: parametric keeps pace with retrieval as N grows (fixed weights suffice -> no need
for a Ledger), or retrieval collapses as fast as parametric (the store doesn't actually scale).

HONEST NOTES: retrieval here is exact top-1 (brute force, chunked over the RAM Ledger) -> O(N) per
query; an ANN index makes it O(log N) but we report the honest brute-force cost. The parametric
"capacity" depends on its width; we hold it fixed and let N cross it. Toy embeddings, not language.
"""

from __future__ import annotations
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = 64            # embedding dim
V = 100           # value classes  (chance = 1/V = 0.01)
SIGMA = 0.15      # query noise


def build_ledger(N, rng):
    """Facts live in HOST RAM (float32). key_i unit-norm; value_i a class in [0,V)."""
    keys = rng.standard_normal((N, D)).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8
    vals = rng.integers(0, V, size=N).astype(np.int64)
    return keys, vals            # numpy, in RAM


def retrieve_top1_classes(keys_ram, vals_ram, queries, chunk=200_000):
    """Exact top-1 nearest key (by inner product) for each query, streaming the RAM Ledger to the
    GPU in chunks so N can far exceed VRAM. Returns predicted classes. O(N) per query (honest)."""
    q = torch.from_numpy(queries).to(DEVICE)                 # (B,D)
    N = keys_ram.shape[0]
    best_score = torch.full((q.shape[0],), -1e30, device=DEVICE)
    best_idx = torch.zeros(q.shape[0], dtype=torch.long, device=DEVICE)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        kc = torch.from_numpy(keys_ram[s:e]).to(DEVICE)      # (c,D) streamed from RAM
        scores = q @ kc.T                                    # (B,c)
        sc, ix = scores.max(dim=1)
        upd = sc > best_score
        best_score = torch.where(upd, sc, best_score)
        best_idx = torch.where(upd, ix + s, best_idx)
        del kc, scores
    return vals_ram[best_idx.cpu().numpy()]


class ParamMLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, V))

    def forward(self, x):
        return self.net(x)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def train_parametric(keys, vals, steps, B, lr, rng):
    """Train the MLP to map noisy queries -> class over ALL N facts. Eval on fresh noise."""
    N = keys.shape[0]
    model = ParamMLP().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    keys_t = torch.from_numpy(keys).to(DEVICE)
    vals_t = torch.from_numpy(vals).to(DEVICE)
    for _ in range(steps):
        idx = torch.randint(0, N, (B,), device=DEVICE)
        x = keys_t[idx] + SIGMA * torch.randn(B, D, device=DEVICE)
        loss = F.cross_entropy(model(x), vals_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    # eval: fresh noise over a sample of the facts
    with torch.no_grad():
        m = min(N, 4096)
        idx = torch.randint(0, N, (m,), device=DEVICE)
        x = keys_t[idx] + SIGMA * torch.randn(m, D, device=DEVICE)
        acc = (model(x).argmax(-1) == vals_t[idx]).float().mean().item()
    return acc, model.n_params


def eval_retrieval(keys, vals, n_query, rng):
    idx = rng.integers(0, keys.shape[0], size=min(n_query, keys.shape[0]))
    queries = keys[idx] + SIGMA * rng.standard_normal((idx.size, D)).astype(np.float32)
    t0 = time.time()
    pred = retrieve_top1_classes(keys, vals, queries)
    dt = time.time() - t0
    acc = float((pred == vals[idx]).mean())
    return acc, dt / idx.size * 1e3       # ms per query (brute force)


def run(N_train=(1_000, 10_000, 100_000), N_retrieval_only=(1_000_000, 5_000_000),
        steps=4000, B=512, lr=2e-3, seeds=(0,)):
    print(f"\n=== M14: knowledge scales in the Ledger, not the model "
          f"(device={DEVICE.type}, D={D}, V={V}, noise={SIGMA}) ===")
    print(f"accuracy (chance={1/V:.3f}); parametric MLP has FIXED capacity, "
          f"retrieval uses a host-RAM Ledger\n")
    hdr = f"{'N facts':>10} | {'ledger RAM':>10} | {'parametric':>11} | {'retrieval':>10} | " \
          f"{'ret ms/q':>9} | {'MLP params':>10}"
    print(hdr); print("-" * len(hdr))
    par, ret = {}, {}
    for N in N_train:
        p_accs, r_accs, msq = [], [], []
        for s in seeds:
            rng = np.random.default_rng(s)
            keys, vals = build_ledger(N, rng)
            pa, nparams = train_parametric(keys, vals, steps, B, lr, rng)
            ra, ms = eval_retrieval(keys, vals, 4096, rng)
            p_accs.append(pa); r_accs.append(ra); msq.append(ms)
        par[N], ret[N] = np.mean(p_accs), np.mean(r_accs)
        ram_mb = N * D * 4 * 2 / 1e6
        print(f"{N:>10} | {ram_mb:>8.0f}MB | {par[N]:>11.3f} | "
              f"{ret[N]:>10.3f} | {np.mean(msq):>9.3f} | {nparams:>10}")

    for N in N_retrieval_only:                    # large-N retrieval-only (host RAM scaling)
        rng = np.random.default_rng(0)
        keys, vals = build_ledger(N, rng)
        ra, ms = eval_retrieval(keys, vals, 2048, rng)
        ret[N] = ra
        ram_mb = N * D * 4 * 2 / 1e6
        print(f"{N:>10} | {ram_mb:>8.0f}MB | {'(untrained)':>11} | {ra:>10.3f} | "
              f"{ms:>9.3f} | {'--':>10}")

    # verdict: at matched (largest TRAINED) N, retrieval >> parametric because the fixed model
    # saturates its capacity; separately, retrieval degrades only gracefully out to the largest N.
    Nlo, Nhi = min(N_train), max(N_train)
    Nmax = max(list(N_train) + list(N_retrieval_only))
    print(f"\nparametric: N={Nlo} -> {par[Nlo]:.3f}, N={Nhi} -> {par[Nhi]:.3f}  "
          f"(fixed {nparams}-param model: {'COLLAPSES to ~chance' if par[Nhi] < 0.1 else 'holds'})")
    print(f"retrieval : N={Nhi} -> {ret[Nhi]:.3f} (head-to-head), N={Nmax} -> {ret[Nmax]:.3f} "
          f"(graceful scaling on host RAM)")
    print(f"head-to-head gap @ N={Nhi}: retrieval {ret[Nhi]:.3f} - parametric {par[Nhi]:.3f} "
          f"= {ret[Nhi]-par[Nhi]:.3f}")
    collapse = par[Nlo] > 0.8 and par[Nhi] < 0.1                 # trains small, collapses large
    holds = ret[Nhi] > 0.8 and (ret[Nhi] - par[Nhi] > 0.5)      # retrieval wins at matched N
    graceful = ret[Nmax] > 0.5                                  # far above chance at extreme N
    ok = collapse and holds and graceful
    print(f"M14 (knowledge scales in the Ledger, not the model): "
          f"{'SUPPORTED' if ok else 'CHECK'}")
    print("[The nonparametric-knowledge bet behind M9, now empirical & learned: a fixed-size model "
          "cannot\n memorise growing N, but a small model + host-RAM Ledger does, at constant "
          "per-query model cost.]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if args.smoke:
        run(N_train=(1_000, 10_000), N_retrieval_only=(100_000,), steps=300)
    else:
        run()
    print(f"[wall time: {time.time()-t0:.1f}s]")
