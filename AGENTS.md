# AGENTS — Claude-Playground project entry

**Status:** Level 4 project instructions  
**Canonical Constitution:** https://github.com/MrWizard94-Compile/Agents-Constitution  
**Accepted source commit:** `ded36646c11d61bcfa7e2420b81a278dc98f47e6`

The accepted GitHub repository is the only editable WPAI Constitution source.
Load its `SKILL.md`, `AGENTS.md`, `SOP.md`, `references/always-load.md`, and
applicable `law/*` files from this verified commit. A generated mirror is
usable only when its manifest matches that SHA. The `.claude/AGENTS.md` file
is a compatibility pointer to this project entry, not a second source of law.

## Project mission and evidence boundaries

- `lwf/` investigates the Ledger / Workspace / Fabric architecture. Start
  with [`lwf/README.md`](lwf/README.md), then its
  [`references/INVARIANTS.md`](lwf/references/INVARIANTS.md),
  [`GAPS.md`](lwf/GAPS.md), and [`LAB_NOTEBOOK.md`](lwf/LAB_NOTEBOOK.md)
  before changing a mechanism or interpreting a result.
- Preserve the distinction between isolated Stage-0 experiments, learned
  probes, analytic/simulated energy models, and an integrated working system.
  A successful source check or reproduced toy result does not establish
  production capability, measured silicon efficiency, or general reasoning.
- Keep original pre-registered thresholds, failed verdicts, seeds, baselines,
  and caveats visible. A revised experiment may add evidence but must not
  silently rewrite the original decision rule or erase a negative result.
- `lwf.zip`, local settings, Python caches, and credentials are not delivery
  artifacts. Do not commit raw secrets or unreviewed private research data.

## Verification

For a governance-only change, run the pinned Constitution validator/tests,
`tools/check-adoption.py` against this repository, and a warning-clean Python
syntax check. For a scientific or runtime change, run the relevant experiment
under its declared dependencies and hardware, compare the recorded baseline,
and document seeds and evidence limits. `lwf/run_all.py` covers only M1–M5;
do not present it as a full-suite or GPU acceptance gate.

The retired `.claude/SOUL.md` pointer and 188-line Claude-era variant remain
recoverable from Git commit `da865435ca489135e94f77c0c08d18506ef76a4e`.
The variant's SHA-256 is
`AD3137994B46B5581584B9A43049C4FB31173361D49EB1D872CCDBBCB1E2539E`;
it is identical to the reviewed Research-Crawler historical variant.
