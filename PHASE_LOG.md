# Phase log

Append-only. One entry per phase, written when the phase gate passes.

Format:

```
## Phase N — <name>
**Date:** YYYY-MM-DD · **Gate:** passed | partial
**Delivered:** what exists now that did not before.
**Deferred:** what was consciously left out, and to which phase.
**Notes:** anything the next phase needs to know.
```

---

## Phase 0 — Scaffold, tooling, conventions, CI
**Date:** 2026-08-01 · **Gate:** passed

**Delivered**

- Full directory tree: `src/g2t_aml/{data,facts,corpus,models,eval,human,utils}`,
  `configs/`, `schemas/`, `scripts/`, `tests/{unit,integration,golden}`, `docs/`,
  gitignored `data/` and `artifacts/` with `.gitkeep` markers.
- Packaging on **uv** with a committed `uv.lock`, Python 3.11, hatchling backend.
  Default install is CPU-only; `graph`, `llm`, `eval` and `human` are optional extras.
- Utilities, all tested:
  - `utils/seeding.py` — `seed_everything` covering `random`, numpy, torch, CUDA;
    sets `torch.use_deterministic_algorithms(True, warn_only=True)` and
    `CUBLAS_WORKSPACE_CONFIG=:4096:8`; returns a loggable record of what it set.
  - `utils/hashing.py` — `hash_file`, `hash_dir`, `hash_dataframe`, `hash_config`,
    `hash_id_list`, `hash_manifest`, all over canonical byte encodings.
  - `utils/io.py` — atomic JSON / JSONL / Parquet via same-directory temp + `Path.replace`.
  - `utils/logging.py` — rich handler, optional file tee, `stage()` context manager that
    logs start config, end summary and elapsed time.
  - `utils/run_context.py` — `RunContext` capturing git SHA/branch/dirty, resolved config
    hash, seeds, data manifest hash, schema versions, platform and library versions;
    serialised to `run_context.json` in every run dir (invariant 5).
- Hydra root at `configs/config.yaml` with groups `paths`, `data`, `encoder`, `fusion`,
  `generator`, `corpus`, `eval`, `experiment`. Run dir is
  `artifacts/runs/${now:%Y-%m-%d}/${now:%H-%M-%S}_${experiment.name}`; the resolved
  config is written into it as `resolved_config.json` alongside Hydra's `.hydra/`.
  Both substrate configs carry an explicit **availability mask** (invariant 4).
- `Makefile` with `install`, `lint`, `format`, `typecheck`, `test`, `smoke`, `clean`, and
  announced placeholders for `data`, `splits`, `facts`, `bronze`, `silver`,
  `train-encoder`, `train-generator`, `eval`, `matrix`, `release`.
- Quality tooling: ruff (line length 100, isort, pydocstyle-google, bugbear, annotations),
  mypy strict scoped to `facts/` + `eval/` only, pytest with coverage and the
  `slow`/`gpu`/`network`/`integration` markers registered, pre-commit including a local
  hook that blocks staging anything under `data/` or `artifacts/`.
- GitHub Actions CI: light install, `make lint typecheck test`, e2e smoke run, uv cache.
- `CLAUDE.md` with all 8 invariants verbatim plus a table of how each is enforced
  mechanically; `DECISIONS.md` with D-001…D-009; `README.md`; Apache-2.0 `LICENSE`;
  `CITATION.cff`; `Dockerfile`.

**Pinned versions**

| | |
|---|---|
| Python | 3.11 |
| torch | **2.4.0** |
| CUDA | **12.1** (`cu121` wheels) |
| torch-geometric | 2.6.1 |
| torch-scatter / torch-sparse / torch-cluster | 2.1.2 / 0.6.18 / 1.6.3 |
| transformers / peft / bitsandbytes / accelerate / trl | 4.45.2 / 0.12.0 / 0.43.3 / 0.33.0 / 0.9.6 |
| vllm | 0.6.2 (pins torch 2.4.0 and transformers >=4.45 — see D-005, D-008) |

The three PyG companion packages are **not** in `uv.lock` — their sdists import torch at
build time and cannot be resolved. `make install-pyg` fetches the prebuilt wheels from
`https://data.pyg.org/whl/torch-2.4.0+cu121.html`. See D-007.

**Deferred**

- `schemas/` holds only the splits directory layout; the `case_facts` JSON Schema itself
  is Phase 3, since its shape is not knowable before the fact families are fixed.
- `RESULTS.md` — created in Phase 10, when there is a first result to record.
- `docs/data_cards/` and `docs/annotation/` have README placeholders; the AMLworld and
  Elliptic2 cards are written in Phase 1, the annotation protocol in Phase 6.
- `tests/golden/` is empty. Golden files arrive with the fact layer in Phase 3, which is
  where invariant 1's ≥90% coverage requirement starts biting.
- GPU code paths are untested by CI by construction (D-004). Phase 7 needs a
  `@pytest.mark.gpu` suite run manually before its gate.

**Gate verification** (run on this machine, 2026-08-01)

- `uv sync --group dev` from a clean checkout: 210 packages resolved, no CUDA.
- `make smoke`: ruff clean, `ruff format --check` clean, `mypy --strict` clean over
  `facts/` + `eval/`, **88 tests passed**, e2e smoke run wrote a populated run directory.
- Coverage **94%** overall; `utils/` modules at 100% except `seeding.py` at 55% — the
  uncovered lines are the torch/CUDA branch, unreachable in the CPU environment by
  construction (D-004). It is covered by the GPU suite from Phase 7.
- `seed_everything(42)` returns a populated, JSON-serialisable dict.
- Hydra composes and overrides compose: `data=elliptic2 experiment=full corpus=silver
  seed=7` produces a different config hash and its own run directory.
- Run directory contains `.hydra/{config,hydra,overrides}.yaml`, `resolved_config.json`,
  `run_context.json` (git SHA, config hash, seeds, schema versions, library versions) and
  `smoke.log`.
- `.gitignore` verified: 65 files tracked; nothing under `data/` or `artifacts/` except
  `.gitkeep`. The pre-commit hook was tested against a deliberately staged
  `data/raw/leak.csv` and rejected it.
- Placeholder targets announce their phase and exit non-zero.

**Notes for Phase 1**

- Nothing in `models/` exists beyond an empty `__init__.py`, deliberately. The
  `_target_` fields in `configs/encoder/` and `configs/fusion/` name classes that do not
  exist yet; they are the contract Phases 7 and 8 must satisfy.
- `data.availability` is already load-bearing in tests. When the fact record is designed
  in Phase 3, its availability mask keys must match
  `REQUIRED_AVAILABILITY_KEYS` in `tests/integration/test_hydra_compose.py`.
- Elliptic2 is access-gated and **not redistributable**. Do not add a download step that
  assumes an open URL.
