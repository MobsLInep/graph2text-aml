# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Two things this project versions separately from the software**, because changing either
invalidates published numbers rather than merely changing behaviour:

- `case_facts` schema — **frozen at 1.0.0** (invariant 9)
- `training_record` schema — **frozen at 1.0.0**

A bump to either is a **breaking change to every derived artifact**, requires a
`DECISIONS.md` entry, and requires regeneration from Phase 3 forward. Neither has been
bumped.

---

## [0.1.0] — 2026-08-06

**First public release.** Phases 0–14. Packaged for submission to *Expert Systems with
Applications*.

### What this release does and does not contain

**It contains a complete, tested pipeline and a corpus. It does not contain a trained
generator.** 17 systems are declared in the experiment matrix; **0 have been trained**, and
Gate 8 — does the fusion layer beat its own shuffled control? — is **open**. Every non-run
is named with its blocker in `RESULTS.md` §2. Read `docs/ETHICS.md` §3.1 before treating
any number here as evidence about the method.

### Added — Phase 14, this release

- **`docs/REPRODUCTION.md`** — raw data to every published number, with per-stage runtimes,
  hardware requirements, expected outputs, and an explicit **tolerance policy** naming the
  sources of GPU non-determinism and what is and is not claimed to reproduce.
- **`docs/ETHICS.md`** — intended use, misuse limits, measured failure rates, fairness
  (including what could not be assessed and why), provenance, and GPU-hours with estimated
  emissions.
- **`docs/model_cards/`** — the GATv2 encoder and the eight comparison/ablation arms, each
  with an explicit misuse section. Plus a card recording what is **not** released: the LoRA
  adapters and the fusion projector do not exist.
- **`docs/dataset_cards/`** — Bronze, Silver, Gold and `case_facts`. Two of the three
  narrative tiers are empty and their cards say so on the first line.
- **`scripts/14_quickstart.py`** and a committed 220-record stratified Bronze fixture —
  reproduces one published result from a clean clone in ~2 s, asserted **exactly** against
  `tests/golden/quickstart_evaluation.json`. No data download, no GPU, no network.
- **`scripts/14_verify_release.py`** — nine release checks against a pristine `git archive`
  export: clean clone, install from the lockfile, quickstart, golden files, every
  documented command, every script's `--help`, a full-history secret scan, no leaked data,
  and licence-NOTICE integrity. Runs on a weekly schedule in CI.
- **`scripts/14_package_release.py`** — packages two **licence-separated** bundles that
  must never be merged (D-098), each with its licence text, NOTICE and a SHA-256 manifest.
- **`scripts/14_reconstruct_elliptic2.py`** — the reconstruction path for the access-gated
  substrate. It reports honestly that there is nothing to reconstruct, because Elliptic2
  access was never requested.
- **`docker/Dockerfile.cpu`** and **`docker/Dockerfile.gpu`** — base images pinned by
  digest, uv pinned by version, dependencies `--frozen` from `uv.lock`. The CPU image runs
  its own quickstart at build time, so a successful build has already reproduced a result.
- **`.github/workflows/verify-release.yml`** — scheduled release verification plus both
  image builds.
- **`.gitleaks.toml`** and a gitleaks pre-commit hook. Every allowlist entry is a
  documented false positive.
- **`.dockerignore`** — the build context was ~8 GB without it, because `.gitignore` does
  not apply to `docker build`.
- **A `stats` optional extra** (scipy, statsmodels, krippendorff — no torch), installed by
  CI and the CPU image. `eval/statistics.py`, `eval/report.py` and
  `experiments/aggregate.py` are measurement code under invariant 1 and need scipy; the
  only other homes for it were the `eval` extra, which pulls torch via bert-score, or a
  skip — and a gate that skips the statistics is not gating them. `make install-stats`.
- **`CHANGELOG.md`** (this file).

### Changed — Phase 14

- **`README.md` rewritten.** The previous version claimed phases 6–14 were not started;
  six of them had landed.
- **`CITATION.cff`** — real repository URL, release date and version.
- **`Dockerfile` moved** to `docker/Dockerfile.cpu` and `docker/Dockerfile.gpu`. The root
  file is removed; the GPU path used to be commented-out lines inside the CPU image.
- **`tests/fixtures/NOTICE` extended** to cover the new quickstart fixture as a second
  CDLA-Sharing-1.0 redistribution, with its §3.2 record of changes. That file is a licence
  obligation, not documentation.
- **`RESULTS.md`** — `sage` and `gcn` now carry their measured AUC-PR figures instead of a
  dash. A dash in that file means "no number, never a zero", and using it where a number
  existed violated the file's own convention.
- `scripts/07c_report_tables.py` now uses `argparse`, so `--help` works on a clean clone
  instead of exiting 1 because the encoder report is absent.
- **uv pinned to 0.10.2** in both images and both CI workflows. The previous pin, 0.4.20,
  predates the `--group` flag every install step uses, so the image and `ci.yml` had never
  successfully run.
- **`pytest.importorskip("torch")` added to four test modules** (`test_fusion`,
  `test_generator_guard`, `test_generator_harness`, `test_generator_pipeline`). Without it
  they failed at *collection* in the CPU-only environment, and a collection error takes the
  whole run down — so `make smoke`, the documented CI gate, did not work in the environment
  `make install` produces.
- **`uv.lock` relocked**, which corrected a drift: it held matplotlib 3.11.1 against a
  `pyproject.toml` pin of 3.9.2, so `--frozen` installs were silently getting an undeclared
  version.
- **`build_optimizer` now guards on the parameters' device, not on `bitsandbytes` being
  importable.** `PagedAdamW8bit` constructs over CPU tensors and raises `Expected a cuda
  device, but got: cpu` only at the first `.step()`, so the failure landed mid-training.
  `torch.cuda.is_available()` is not the right test either — a host can have a working card
  while the model sits on CPU. Anyone running `make install-gpu` on a CPU-only box hit this.
- **The GPU image's base digest corrected to the multi-arch index** (`sha256:21196d81...`).
  It had been pinned to the `linux/arm64` per-platform manifest, so an amd64 runner pulled
  an arm64 image and every `RUN` failed with `exec format error`.
- **`scripts/14_verify_release.py` now scrubs `VIRTUAL_ENV` and friends** from commands run
  inside the clean clone, and installs `--extra stats` there. A clean-clone verification
  that resolves against the caller's environment is verifying the wrong thing.
- Module docstrings added to `corpus/`, `models/` and `utils/` `__init__.py`.

### Licensing decisions — Phase 14 (D-098)

- **The narrative corpus and fact records ship under CDLA-Sharing-1.0, not Apache-2.0.**
  The narratives quote AMLworld account identifiers, timestamps, currencies and amounts
  verbatim and each record embeds its `case_facts`, which is more than a de-minimis portion
  of the source Data. That makes them **Enhanced Data**, not §3.5 *Results*. The
  conservative reading costs nothing; the unconservative one is a breach.
- **Code, model weights, metrics and figures ship under Apache-2.0** as §3.5 Results.
- **Nothing Elliptic2-derived is released**, because none exists.

### Known absences in this release

Named here rather than discovered later. Each has a blocker, and none is a software
problem:

| Absent | Blocker |
|---|---|
| Every trained generator arm; the Gate 8 outcome; the LoRA adapters; the fusion projector | GPU ≥24 GB (D-068) |
| The Silver corpus; systems B3/B4/B5; the Method A/B extractor agreement κ | Teacher-API credentials |
| The Gold corpus; every Layer 1 overlap metric | No annotator recruited |
| The Phase 12 decision-setting study | Ethics approval not yet applied for |
| Everything on Elliptic2 | Access never requested |
| A minted Zenodo DOI | Requires a Zenodo–GitHub link and a published release |

---

## Prior phases

Not versioned individually — the project reached 0.1.0 in one line of development.
`PHASE_LOG.md` is the authoritative record of what each phase delivered and deferred, and
`DECISIONS.md` carries every non-obvious technical choice with its rationale (D-001
onward). Summary:

| Phase | Delivered |
|---|---|
| 0 | Scaffold, tooling, conventions, CI |
| 1 | Both substrate loaders, canonical representation, data cards. AMLworld reproduces every published figure exactly. |
| 2 | Case extraction, frozen temporal splits (10,932 / 2,028 / 3,196), leakage audit |
| 3 | The fact layer — schema (frozen 1.0.0), extractor, three-valued checker, vocabulary, serialiser, independent oracle |
| 4 | Bronze — template engine, 15,707 narratives, the ten-point validation harness |
| 5 | Silver — two-teacher pipeline, verifier loop, discard log. **Machinery only.** |
| 6 | Gold — annotation kit, guidelines, the 350-case reserved sample. **No narrative written.** |
| 7 | The GAT encoder — six arms × three seeds, gate passed at +0.070 over the MLP control |
| 8 | The fusion layer — projectors and the derangement control. **Untrained.** |
| 9 | The QLoRA harness and the guard. **Never run.** |
| 10 | The evaluation harness — three layers, two independent claim extractors |
| 11 | The 17-system experiment matrix — declared, orchestrated, aggregated. **Not run.** |
| 12 | The decision-setting study — built, validated against simulated responses. **Blocked.** |
| 13 | Efficiency — instrument built; 2 of 17 systems measured end to end |
| 14 | This release |

[0.1.0]: https://github.com/MobsLInep/graph2text-aml/releases/tag/v0.1.0
