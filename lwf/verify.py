"""
LWF Tier-0 verifiability layer -- mechanism M3.

Claim: because an LWF cognitive step is a bounded pure function of
(checkpointed Workspace state, input, retrieved Ledger records), the whole
execution is CHECKPOINTABLE and bit-exact REPLAYABLE, and the only nondeterminism
(ANN approximation, analog/FP noise) is boundable by logging the retrieved record
IDs. That is a verification property transformers over full context do not offer:
their "state" is the entire history, so there is no small object to checkpoint or
reason about.

This module wraps a step function with:
  - a canonical, order-independent hash of the Workspace state and inputs,
  - a per-step trace record (state hash in, input hash, retrieved IDs, state hash out),
  - deterministic replay from any checkpoint + the trace.

The trace is the audit log. M3's cost claim is that it grows as O(steps * k) with a
SMALL k (retrieved-records-per-step), not O(n^2) like a materialised context.
"""

from __future__ import annotations
import hashlib
import json
import numpy as np
from dataclasses import dataclass, field, asdict


def hash_array(x: np.ndarray) -> str:
    """Stable content hash of an array (dtype+shape+bytes). Bit-exact by construction."""
    x = np.ascontiguousarray(x)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(x.dtype).encode())
    h.update(str(x.shape).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def hash_obj(o) -> str:
    return hashlib.blake2b(json.dumps(o, sort_keys=True, default=str).encode(),
                           digest_size=16).hexdigest()


@dataclass
class StepRecord:
    step: int
    state_in: str
    input_hash: str
    retrieved_ids: list          # the ONLY nondeterministic quantity -> logged
    output_hash: str
    state_out: str

    def as_dict(self):
        return asdict(self)


@dataclass
class Trace:
    records: list = field(default_factory=list)

    def append(self, r: StepRecord):
        self.records.append(r)

    @property
    def n_steps(self):
        return len(self.records)

    def total_retrieved(self):
        return sum(len(r.retrieved_ids) for r in self.records)

    def bytes_estimate(self):
        """Serialized log size -- the quantity M3 claims is O(steps*k)."""
        return len(json.dumps([r.as_dict() for r in self.records]).encode())

    def digest(self):
        """Single hash over the whole run -- two runs match iff bit-identical."""
        return hash_obj([r.as_dict() for r in self.records])


class VerifiableRunner:
    """Wraps a user step_fn(state, x) -> (new_state, output, retrieved_ids).

    Records a replayable trace. `run` executes; `replay` re-executes from a
    checkpoint and asserts the trace digest matches (bit-exact determinism)."""

    def __init__(self, step_fn):
        self.step_fn = step_fn

    def run(self, state0: np.ndarray, inputs) -> tuple[np.ndarray, Trace]:
        state = state0.copy()
        trace = Trace()
        for i, x in enumerate(inputs):
            s_in = hash_array(state)
            new_state, output, rids = self.step_fn(state, x)
            trace.append(StepRecord(
                step=i,
                state_in=s_in,
                input_hash=hash_array(np.asarray(x, dtype=np.float64)),
                retrieved_ids=list(map(int, rids)),
                output_hash=hash_array(np.asarray(output, dtype=np.float64)),
                state_out=hash_array(new_state),
            ))
            state = new_state
        return state, trace

    def replay(self, state0: np.ndarray, inputs, reference: Trace) -> bool:
        """Re-run and confirm bit-exact identity with the reference trace."""
        _, trace2 = self.run(state0, inputs)
        return trace2.digest() == reference.digest()
