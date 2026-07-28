"""
LWF Ledger -- the persistent, content-addressable knowledge store (Tier 2).

Implements mechanism M2's exact-recall organ. Invariant I1 (a bounded state of
B bits distinguishes <= 2^B histories) says the Workspace *cannot* hold unbounded
exact recall. The Ledger is where that recall is forced to live.

Key architectural property under test: the Ledger's contribution to PER-STEP hot
cost does not scale with total store size N. Reads touch O(k) retrieved records
(+ index traversal). Here we implement an EXACT brute-force nearest-neighbour read
and *count the comparisons* so the O(N) is visible and honest -- a production build
swaps this for an ANN index (HNSW/IVF) at O(log N). We report the count rather than
pretend it away.

No free lunch: the Ledger does not make recall cheap, it makes recall *bounded in
hot-state* by paying an out-of-hot-path retrieval cost. That trade is the claim.
"""

from __future__ import annotations
import numpy as np


class ContentAddressableLedger:
    def __init__(self, d_k: int, d_v: int):
        self.d_k = d_k
        self.d_v = d_v
        self._keys: list[np.ndarray] = []
        self._vals: list[np.ndarray] = []
        self.last_comparisons = 0  # comparisons made on the most recent read
        # Nogood store (imported from sibling EP-GRM truth-maintenance verification).
        # A nogood is a jointly-inconsistent set of literals -- a LEARNED constraint that
        # prevents ever re-deriving the same contradiction. First-class Ledger record type,
        # sitting alongside associative (key,value) knowledge: the Ledger holds both what IS
        # true (associations) and what CANNOT be (nogoods). Indexed by one member literal for
        # sub-linear checking. See exp_m11_nogood.py and references/SIBLING_EPGRM.md.
        self._nogoods: list[frozenset] = []
        self._nogood_index: dict = {}      # literal -> list of nogood ids containing it
        self.last_nogood_checks = 0

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def size(self) -> int:
        return len(self._keys)

    def write(self, k: np.ndarray, v: np.ndarray) -> int:
        self._keys.append(k.astype(np.float64).copy())
        self._vals.append(v.astype(np.float64).copy())
        return len(self._keys) - 1

    def read(self, q: np.ndarray, topk: int = 1):
        """Exact content-addressed read: return top-k values by inner product.
        Records comparison count (== N) to keep the retrieval cost auditable."""
        if not self._keys:
            self.last_comparisons = 0
            return []
        K = np.stack(self._keys)                 # (N, d_k)
        scores = K @ q.astype(np.float64)        # (N,)  -- N comparisons
        self.last_comparisons = K.shape[0]
        idx = np.argsort(-scores)[:topk]
        return [(int(i), self._vals[i].copy(), float(scores[i])) for i in idx]

    def read_top1_value(self, q: np.ndarray) -> np.ndarray | None:
        r = self.read(q, topk=1)
        return None if not r else r[0][1]

    # ---- nogood (learned-constraint) interface ----
    def write_nogood(self, literals) -> int:
        """Record a jointly-inconsistent set of literals, kept SUBSET-MINIMAL (a nogood N forbids
        every assignment superset of N, so a smaller N subsumes any larger one):
          - if an existing nogood is a subset of the new one, the new one is redundant -> drop it;
          - otherwise add it and retract any existing nogoods that are supersets of it.
        Removing a subsumed superset never changes `nogood_violated` results."""
        ng = frozenset(literals)
        if not ng:
            return -1
        if any(existing <= ng for existing in self._nogoods):
            return -1                                   # already have a stronger constraint
        survivors = [e for e in self._nogoods if not (ng <= e)]   # drop subsumed supersets
        survivors.append(ng)
        self._nogoods = survivors
        self._nogood_index = {}                         # rebuild index (counts are small)
        for i, n in enumerate(self._nogoods):
            for lit in n:
                self._nogood_index.setdefault(lit, []).append(i)
        return len(self._nogoods) - 1

    def nogood_violated(self, assignment) -> bool:
        """True if the assignment (set of literals) contains any stored nogood as a subset,
        i.e. it entails a known contradiction. Uses the literal index to avoid scanning all
        nogoods: only those sharing a literal with the assignment can possibly match."""
        a = assignment if isinstance(assignment, frozenset) else frozenset(assignment)
        candidates = set()
        for lit in a:
            candidates.update(self._nogood_index.get(lit, ()))
        self.last_nogood_checks = len(candidates)
        for nid in candidates:
            if self._nogoods[nid] <= a:
                return True
        return False

    @property
    def n_nogoods(self) -> int:
        return len(self._nogoods)

    def read_hopfield(self, q: np.ndarray, beta: float = 8.0):
        """One-step modern-Hopfield / softmax-attention read (Ramsauer et al. 2020).

        v_hat = sum_i softmax(beta * <k_i, q>)_i * v_i

        This is the *soft, differentiable* Ledger read. Modern Hopfield theory says a
        store of N patterns retrieves with exponentially small error in one step and
        that this update rule IS transformer attention. In production the softmax is
        restricted to an ANN top-k neighbourhood (O(log N)); here it is dense for a
        faithful, differentiable reference. Returns (v_hat, entropy_of_attention).
        """
        if not self._keys:
            return np.zeros(self.d_v), 0.0
        K = np.stack(self._keys)                 # (N, d_k)
        V = np.stack(self._vals)                 # (N, d_v)
        s = beta * (K @ q.astype(np.float64))    # (N,)
        s -= s.max()
        w = np.exp(s); w /= w.sum()
        self.last_comparisons = K.shape[0]
        ent = float(-(w * np.log(w + 1e-12)).sum())   # low entropy => confident retrieval
        return V.T @ w, ent
