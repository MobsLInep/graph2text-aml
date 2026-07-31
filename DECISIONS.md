# Decision log

Append-only. Newest entries at the bottom. One entry per non-obvious technical choice,
recorded **when the decision is made**, not reconstructed afterwards.

Format:

```
## D-NNN — <title>
**Date:** YYYY-MM-DD · **Phase:** N · **Status:** accepted | superseded by D-MMM
**Decision:** one sentence.
**Rationale:** why this and not the alternative.
**Consequences:** what this forecloses or obliges.
```

---

## D-001 — uv over poetry for dependency management
**Date:** 2026-08-01 · **Phase:** 0 · **Status:** accepted
**Decision:** Use `uv` with a committed `uv.lock`.
**Rationale:** Resolution and install are roughly an order of magnitude faster than
poetry, which matters most in CI where we re-resolve on every push; `uv` also manages the
Python 3.11 toolchain itself, so a clean machine needs only `uv` and nothing else.
**Consequences:** Contributors need `uv` installed. The lockfile is authoritative; CI
runs `uv sync --frozen` and will fail rather than silently re-resolve.

## D-002 — Hydra over argparse for configuration
**Date:** 2026-08-01 · **Phase:** 0 · **Status:** accepted
**Decision:** All entrypoints are Hydra apps composing from `configs/`.
**Rationale:** The project's core output is an ablation matrix across two substrates,
three corpus tiers, and several fusion variants; Hydra's config groups and multirun
express that grid directly, and its per-run output directory plus saved resolved config
is most of invariant 5 for free. argparse would push the same combinatorics into shell
scripts, where they would go unversioned.
**Consequences:** Config composition is now a testable surface (see
`tests/integration/test_hydra_compose.py`). Interpolations must resolve — a `paths` entry
that does not resolve fails the test suite, not a training run three hours in.

## D-003 — mypy --strict scoped to `facts/` and `eval/` only
**Date:** 2026-08-01 · **Phase:** 0 · **Status:** accepted
**Decision:** `mypy` runs in strict mode over `src/g2t_aml/facts/` and
`src/g2t_aml/eval/`, and nowhere else.
**Rationale:** Those two modules are the measurement instrument (invariant 1) — a type
error there corrupts headline numbers silently. Everywhere else, strict typing against
untyped torch/PyG/transformers surfaces buys little and costs a steady stream of
`# type: ignore`. This is a deliberate asymmetry, not an oversight.
**Consequences:** Untyped code elsewhere is tolerated; the two measurement modules have
no escape hatch. If a later phase moves measurement logic out of those directories, the
`files` list in `pyproject.toml` must move with it.

## D-004 — `graph`, `llm` and `human` are optional extras, not core dependencies
**Date:** 2026-08-01 · **Phase:** 0 · **Status:** accepted
**Decision:** The default install is CPU-only. CUDA-dependent packages live behind
`--extra graph` / `--extra llm`.
**Rationale:** Phases 1–6 and 10 need no GPU. Forcing a CUDA install into CI would push
it well past the five-minute budget and make it unusable on contributors' laptops.
**Consequences:** CI cannot exercise GPU code paths. Anything in `models/` needs an
explicitly marked (`@pytest.mark.gpu`) test run on a GPU machine before it is trusted.

## D-005 — Torch 2.4.0 / CUDA 12.1 pinned across the graph and llm stacks
**Date:** 2026-08-01 · **Phase:** 0 · **Status:** accepted
**Decision:** Pin `torch==2.4.0` in both the `graph` and `llm` extras, with
`torch-geometric==2.6.1` and PyG companion wheels built for that exact torch/CUDA pair.
**Rationale:** PyG companion packages (`torch-scatter`, `torch-sparse`, `torch-cluster`)
are compiled against a specific torch **and** CUDA build; a mismatch surfaces as an
opaque symbol-resolution error at import time. `vllm==0.6.2` independently pins
`torch==2.4.0`, so 2.4.0 is the version that satisfies both stacks without a conflict.
**Consequences:** Upgrading torch means moving PyG, vllm, and the wheel index URL
together, in one commit, with a note here. Do not bump torch alone.

## D-006 — Split manifests are committed files, not seeded runtime code
**Date:** 2026-08-01 · **Phase:** 0 · **Status:** accepted
**Decision:** `schemas/splits/<substrate>/{train,val,test}.txt` hold literal ID lists
with a sidecar content hash; `configs/data/*.yaml` reference them and carry no `seed`.
**Rationale:** Invariant 2. A seeded split is reproducible only as long as the upstream
row order, library version, and filtering code all stay fixed — three things that will
not stay fixed across fourteen phases. A committed ID list is reproducible unconditionally.
**Consequences:** Regenerating splits is a visible diff and requires a decision entry.
The split-construction script (Phase 2) is run deliberately, never as part of training.

## D-007 — PyG companion wheels are installed outside the lockfile
**Date:** 2026-08-01 · **Phase:** 0 · **Status:** accepted
**Decision:** `torch-scatter`, `torch-sparse` and `torch-cluster` are **not** listed in
the `graph` extra. `make install-pyg` installs them from
`https://data.pyg.org/whl/torch-2.4.0+cu121.html`.
**Rationale:** Their PyPI distributions are sdists that `import torch` in `setup.py`, so
`uv lock` cannot resolve them at all — resolution fails with `ModuleNotFoundError: No
module named 'torch'` regardless of `no-build-isolation-package`, because locking happens
before any environment exists. Building them from source would also produce wheels
compiled against whatever CUDA the build host has, which is the exact drift D-005 exists
to prevent. The prebuilt index wheels are the only correct artifact.
**Consequences:** `uv.lock` does not pin those three; the Makefile does, and the README
records the index URL. When torch moves, the `PYG_WHEELS` variable in the Makefile moves
with it. `uv sync` will not detect a stale or missing PyG install — Phase 7 should assert
the versions at import time.

## D-008 — transformers pinned to 4.45.2, driven by vllm
**Date:** 2026-08-01 · **Phase:** 0 · **Status:** accepted
**Decision:** `transformers==4.45.2` rather than the 4.44.2 originally intended.
**Rationale:** `vllm==0.6.2` requires `transformers>=4.45.0`; 4.44.2 made the `llm` extra
unsatisfiable. 4.45.2 satisfies vllm, peft 0.12.0 and trl 0.9.6 simultaneously.
**Consequences:** The `llm` extra's version floor is set by vllm, not by our needs. If
vllm is ever dropped from the stack, this pin can relax.

## D-009 — the local pre-commit hook runs as `language: script`, not `language: system`
**Date:** 2026-08-01 · **Phase:** 0 · **Status:** accepted
**Decision:** `scripts/hooks/check_no_data_staged.py` carries a `python3` shebang, is
executable, and is invoked directly.
**Rationale:** `language: system` with `entry: python ...` fails on any machine where the
interpreter is `python3` and no `python` alias exists — which includes a stock Ubuntu
runner. A shebang'd script has no such dependency.
**Consequences:** The hook script must stay executable and must not import anything
outside the standard library, since it runs outside the project environment.
