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

---

## Phase 1 — Data ingestion, canonical representation, data cards
**Date:** 2026-08-01 · **Gate:** passed (AMLworld complete; Elliptic2 deferred on access)

**Delivered**

- `data/canonical.py` — `CanonicalGraph` and the frozen nine-field `AvailabilityMask`,
  with `CANONICAL_SCHEMA_VERSION = "0.1.0"`, lossless Parquet + JSON round-trip through
  the atomic-write discipline, referential-integrity validation, and the controlled
  `TYPOLOGY_VOCABULARY` including `unclassified`.
- `data/download.py` — dataset registry with pinned SHA-256 digests, `verify()` →
  `VerificationReport`, documented manual Kaggle step for AMLworld and access-request
  instructions for Elliptic2. Size is checked before hashing so truncation fails fast.
  A mismatch raises before an absence, since it is the more alarming finding.
- `data/loaders/amlworld.py` — typed Polars loader with an asserted header, a real
  state-machine parser for the patterns file, `attach_typologies`, and
  `build_account_graph` keyed on `bank|account`.
- `data/loaders/elliptic2.py` — written against the documented schema, lazy over the
  background graph, probes column names rather than guessing by position.
- `data/stats.py` — full statistics record; union-find components rather than networkx.
- `data/pyg_adapter.py` — isolated so `g2t_aml.data` never imports torch, with a test
  that enforces it in a clean interpreter.
- `scripts/01_ingest.py` — Hydra entrypoint; `make data` / `data-amlworld` /
  `data-elliptic2` now run instead of announcing a phase.
- Data cards for both substrates; `tests/fixtures/NOTICE` discharging the CDLA-Sharing-1.0
  attribution and change-marking obligations for the committed fixture slice.

**Observed versus published — AMLworld HI-Small (complete dataset, checksum-verified)**

| Quantity | Published | Observed | |
|---|---:|---:|:--:|
| Vertices (accounts) | 515,088 | 515,088 | ✅ |
| Edges (transactions) | 5,078,345 | 5,078,345 | ✅ |
| Fan-out | 342 | 342 | ✅ |
| Fan-in | 318 | 318 | ✅ |
| Gather-scatter | 716 | 716 | ✅ |
| Scatter-gather | 626 | 626 | ✅ |
| Cycle | 287 | 287 | ✅ |
| Random | 191 | 191 | ✅ |
| Bipartite | 263 | 263 | ✅ |
| Stack | 466 | 466 | ✅ |
| Not classified | 1,968 | 1,968 | ✅ |

Every published figure reproduces **exactly**. The brief offered the typology counts as
approximate targets; they are exact, and they are counts of *transactions within streams*
(3,209 across 370 streams), not counts of streams.

Also measured: laundering rate **1 in 981** (5,177 of 5,078,345, 0.001019); span
2022-09-01 00:00 → 2022-09-18 16:18 (17.68 days) at observed minute granularity; **591,212
self-loops** (11.6%); 561,575 multi-edge pairs, max 89 parallel; 114,139 weakly-connected
components with the largest holding 372,089 nodes (72.2%); out-degree max 168,672 against
a median of 2; 15 currencies; 7 payment formats. Ingest end to end: **14 s**.

**Schema surprises**

1. **The node count only reconciles under a composite key.** Account identifiers are
   unique only within a bank: account-only keying gives 515,080, `bank|account` gives
   515,088. Eight identifiers genuinely collide across banks. See D-011.
2. **Amounts are not uniformly two-decimal.** The 148,151 Bitcoin rows carry six decimals.
   The first join key formatted the parsed amount to two decimals and lost exactly one
   transaction — `unclassified` 1,969 against a published 1,968, with all eight structural
   typologies matching. Only the identity *3,209 patterned + unclassified = 5,177 flagged*
   exposed it. The key is now built from source text. See D-012.
3. **The `Account` column appears twice** in the header, once per endpoint. Polars renames
   the second to `Account_duplicated_0`. The loader asserts the raw header read as text,
   then renames **positionally**, so it never depends on a de-duplication rule it does not
   control.
4. **Two Phase 0 availability flags were wrong** and are corrected in D-015:
   AMLworld has no entity types and ships no node table; Elliptic2's features exist but
   are anonymised, and the mask governs assertion, not presence.
5. Bank and account codes are zero-padded strings; read as integers, bank `010` would
   collide with bank `10`. All identifiers stay Utf8.

**Licence findings** (both verified 2026-08-01, full detail in the data cards)

- **AMLworld data is CDLA-Sharing-1.0**, *not* the Apache-2.0 of the `IBM/AML-Data`
  repository, which covers only the code. Share-alike applies to published Data and
  Enhanced Data — which includes our `data/interim` Parquet and any corpus quoting rows.
  **CDLA-Sharing-1.0 §3.5 exempts "Results"**: trained models, generated narratives and
  metrics carry no share-alike obligation. Phase 14 can release those under Apache-2.0 but
  must not sweep raw or interim data into the same bundle.
- The committed 500-row fixture **is** a redistribution, so `tests/fixtures/NOTICE` carries
  the attribution and the itemised record of changes CDLA-Sharing-1.0 §3.2 requires. That
  file is a licence obligation, not documentation.
- **Elliptic2's data licence could not be located.** The tooling repository is Apache-2.0
  but says nothing about the dataset, and the download is behind a request form. Recorded
  as `redistributable=False` and treated as closed until we hold written terms. This must
  be resolved before Phase 14.

**Elliptic2 access status**

**Not yet requested as of 2026-08-01.** The loader is written against the documented
schema, its tests use a synthetic tree, and the real-data tests skip via
`is_available()`. `scripts/01_ingest.py data=elliptic2` exits 0 with `ingest_skipped.json`
rather than failing. Phase 1 does not block on it; **Phase 12's cross-substrate ablation
does**, so the request should go in early.

**Gate verification** (this machine, 2026-08-01)

- `ruff check` and `ruff format --check` clean over 44 files; `mypy --strict` clean.
- **260 passed, 3 skipped** (2 Elliptic2 real-data, 1 PyG without the graph extra).
- Coverage **92%** overall. On `data/`: canonical 98%, download 98%, amlworld 97%,
  elliptic2 97%, stats 91% — comfortably past the 75% gate. `pyg_adapter` sits at 18%
  because the graph extra is absent by construction (D-004).
- Real-data regression `tests/integration/test_published_statistics.py` passes all 11
  AMLworld assertions in 16 s.
- `data/interim/amlworld_hi_small/` holds `nodes.parquet` (10.0 MB), `edges.parquet`
  (172 MB), `canonical.json`, `statistics.json` and `manifest.json` with a SHA-256 per
  artifact, `is_complete_dataset: true` and the availability mask.

**Deferred**

- Elliptic2 ingest, checksums and observed statistics — blocked on access.
- Case extraction and splits — Phase 2, deliberately untouched.
- Materialising all 122K Elliptic2 subgraphs — Phase 2; Phase 1 builds one representative
  subgraph only, to prove the path.
- `PUBLISHED_STATISTICS` covers `HI-Small` only. The other five variants load but have no
  pinned figures to assert against.

**Notes for Phase 2**

- Splits must key on `bank|account` for AMLworld nodes and on `transaction_key` for edges.
  Neither is reconstructible from typed columns — re-load rather than derive (D-012).
- The temporal window is only **17.7 days**, which is very little room for a temporal
  split. The distribution across those days should be checked before choosing boundaries.
- The graph is **fragmented**: 114,139 components, median size 1, 72% of nodes in one
  giant component. Case extraction must not assume connectivity.
- Degree skew is extreme (out-degree max 168,672 vs median 2). Neighbourhood sampling
  needs a cap, or a handful of hubs will dominate every case.
- Class imbalance is 1 in 981. Any case set built by uniform sampling will contain almost
  no laundering.

---

## Phase 2 — Case extraction, sampling, temporal splits, leakage audit
**Date:** 2026-08-01 · **Gate:** passed (AMLworld complete; Elliptic2 pass-through only)

**Delivered**

- `data/case_extraction.py` — the four-step construction protocol with full provenance on
  every case. `GraphIndex` builds CSR adjacency over the 5,078,345-edge graph once (2.2 s)
  and serves a case cut in **1.9 ms**; `cut_case` returns positions, `extract_case` keeps
  the specified signature and materialises. `passthrough_case` records Elliptic2's provided
  subgraphs as `extraction_method="provided"`.
- `data/motifs.py` — structural similarity to seven typologies, scored from topology
  alone. It never reads a label, by construction: `score_edges` takes an edge table, not a
  case, so hard-negative mining cannot select on the thing it is meant to be blind to.
- `data/case_sampling.py` — the three populations, stratified allocation, activity- and
  window-matched negatives, per-window hard-negative mining, and the realistic-imbalance
  stream. Cases are stored **by reference** (positions into the interim graph) rather than
  copied: 51 MB instead of ~2 GB, and `materialise()` rebuilds a full `CanonicalGraph` in
  under a millisecond.
- `data/splits.py` — temporal split with searched boundaries, straddle and buffer drops,
  stream atomicity, overlap measurement in `report`/`strict` modes, and frozen manifests
  (`splits.json` plus the three committed `.txt` id lists D-006 specifies, each hashed).
- `data/leakage_audit.py` — six checks, three fatal, independent of `splits.py` by design.
- `scripts/02_build_cases.py`, `02b_sensitivity.py`, `02c_audit.py`; `make cases`,
  `cases-debug`, `sensitivity`, `audit`; `configs/cases/{default,debug}.yaml`.

**Final case counts** (`schemas/splits/amlworld/splits.json`, boundaries
2022-09-06 12:03 and 2022-09-09 12:03)

| | train | val | test | total |
|---|---:|---:|---:|---:|
| cases | **10,932** | **2,028** | **3,196** | 16,156 |
| share | 67.7% | 12.6% | 19.8% | — |
| suspicious | 1,553 | 314 | 508 | 2,375 |
| hard-negative rate | 25.0% | 22.1% | 31.2% | 25.8% |

30,000 cases were built and 16,156 retained (53.9%). The 13,844 dropped break down as
9,718 straddling a boundary, 4,110 inside the 6-hour buffer, and 16 belonging to a stream
already placed in an earlier split. Building at 30,000 to retain ~15,000 is deliberate and
is explained in the config: on a substrate this temporally compressed, roughly half of any
case population straddles a boundary.

**Stratification**

Suspicious cases by typology, overall: bipartite 204, fan_out 164, gather_scatter 140,
scatter_gather 134, cycle 122, stack 107, random 103, fan_in 96, unclassified 1,447.
Even allocation across the nine strata is capped at 35% per stratum (D-018), without which
`unclassified` — which has 3,227 available seeds against 232–448 for each structural
typology — takes 46% of the positive population.

**Node overlap rate: 53.8%** of test cases contain an account that also appears in a train
case, against **0.0% edge overlap** — not one transaction appears on both sides of a
boundary. Recorded in `report` mode with the reasoning in D-021: 72.2% of HI-Small's
accounts sit in one component, so account recurrence is a property of the substrate, while
transaction recurrence would be a genuine leak and there is none.

**Leakage audit: passed, zero hard failures.**

| check | severity | result |
|---|---|---|
| temporal ordering | fatal | 0 violations |
| stream atomicity | fatal | all 211 streams in exactly one split |
| label leakage | fatal | no feature names or reproduces the label |
| node overlap | report | 53.8% |
| edge overlap | report | **0.0%** |
| duplicate cases | report | 42 exact, 148 near-duplicate groups span splits |

**Realistic-imbalance stream:** 10,000 cases over the test window, seeds drawn uniformly.
Observed case-level prevalence **7.3%** — not the ~0.1% the transaction-level rate (1 in
981) implies. A case is a two-hop 48-hour neighbourhood, so it aggregates hundreds of
transactions and the chance at least one is flagged is far higher than the chance any given
one is. This is the honest number for this case definition and D-023 records why it is not
forced down to a textbook 1-in-500.

**Sensitivity findings** (`artifacts/metrics/sensitivity/`, 200 seeds, half from streams
and half uniform, uncapped windows so the window cap is not confounded with k)

| k | nodes p50 | edges p50 | typology recovery | full recovery | suspicious | pruned @150 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 9 | 0.628 | 49.0% | 50.5% | 0.0% |
| **2** | **13** | **43** | **0.837** | **77.0%** | **56.0%** | **0.0%** |
| 3 | 47 | 156 | 0.865 | 79.0% | 60.5% | 17.5% |

Three things the table settles. **k=2 is the knee**: it recovers 84% of a seeding stream's
transactions against k=3's 87%, at a quarter the case size. **n_max=150 is the right
budget**: pruning fires on 0% of cases at k=2 and 17.5% at k=3, so at the chosen k the
budget is a guard rather than a routine truncation, and n_max is not silently shaping the
corpus. **k=3 erodes what "negative" means**: the suspicious rate over uniformly drawn
seeds rises from ~1% at k=1 to ~12% at k=2 to ~21% at k=3, because a licit account three
hops out reaches flagged activity often enough to matter.

**Gate verification** (this machine, 2026-08-01)

- `ruff check`, `ruff format --check` clean over 58 files; `mypy --strict` clean.
- **419 passed, 3 skipped** (2 Elliptic2 real-data, 1 PyG without the graph extra).
- Determinism asserted by byte-identical Parquet serialisation across repeated extraction,
  across separately built indices, and across a reordered input frame.
- The pruning test constructs a fixture where amount-descending pruning *would* sever a
  laundering path, and asserts it does not; its mirror asserts that disabling preservation
  does sever it, so the first test is known to test something.
- The auditor is tested by injecting each fatal leak — a stolen train case in the test
  split, one stream attached to both sides, `is_laundering` in a feature list, a scalar
  that separates the labels perfectly — and asserting it fires.
- End-to-end build: 276 s for 30,000 cases plus a 10,000-case realistic stream.
- `schemas/splits/amlworld/` holds `splits.json` (601 KB) and the three hashed id lists.
  `data/processed/amlworld_hi_small/cases/` is 51 MB and gitignored.

**Two things found by looking at the built corpus, not by testing**

1. **A timezone bug that silently mis-windowed every case.** `datetime.timestamp()` reads
   a naive datetime as *local* time while Polars stores `Datetime("us")` as timezone-naive
   microseconds, so every window was shifted by the machine's UTC offset — 5.5 hours here,
   and a different amount on any other machine. `case_extraction.to_micros` now reads the
   value literally, which is correct for AMLworld's local wall-clock timestamps, and is the
   only conversion used anywhere in Phase 2.
2. **Globally-mined hard negatives clumped into the test split.** A corpus at 28.9% hard
   negatives overall came out 24.8% in train and **64.0% in test**, which would have made
   every test metric incomparable to validation while passing every aggregate gate. Mining
   the share inside each window fixes it (25.0 / 22.1 / 31.2%). See D-024.

**Deferred**

- Elliptic2 case extraction beyond the pass-through path — blocked on access, unchanged
  since Phase 1 and now blocking Phase 12. **The access request has still not been sent.**
- Fact extraction — Phase 3, deliberately untouched.

**Case size distribution (16,156 retained cases)**

| | p10 | p25 | p50 | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
| nodes | 2 | 3 | 6 | 13 | 30 | 150 |
| edges | 2 | 4 | 11 | 27 | 64 | 645 |

Suspicious cases are larger than licit ones (median 8 nodes against 5), which is expected
and is not a leak — the auditor's perfect-separator check confirms the ranges overlap.
The left tail is the thing to look at: **18.1% of cases hold fewer than three nodes** and
41.5% fewer than five. That is not a defect in the extraction, it is HI-Small showing
through — 114,139 weakly-connected components, median degree 2 — but a two-account,
one-transaction case is not something a SAR narrative can say much about. See the note to
Phase 3 below.

**Notes for Phase 3**

- **Decide what to do about degenerate cases, and decide it explicitly.** 18.1% of the
  corpus is a single transaction between two accounts. Left in, they will inflate every
  surface metric (a one-sentence narrative is easy to generate and easy to score well on)
  and teach the generator nothing about structure. Filtered out, the corpus loses a fifth
  of its cases and the split counts move. This was deliberately **not** decided in Phase 2:
  a minimum case size is a modelling choice and belongs to the phase that makes it, with
  its own decision entry — the same reasoning as D-017 on self-loops. The measurement is
  above so the choice can be made against real numbers.
- **A case does not contain its whole stream.** The 48-hour window cap (D-019) means a
  case built from a long stream holds 65% of that stream's transactions on average, and
  only 28% of streams are covered in full. `typology` on a case means "part of a stream of
  this typology", not "exhibits this typology in full". Any narrative claim about a
  scheme's *completeness* is unsupported and the verifier must reject it.
- **val does not cover every typology.** It is a three-day band and contains no `fan_in`,
  `gather_scatter` or `scatter_gather` cases at all. Use val for model selection, never for
  a per-typology breakdown; test covers all nine.
- Load splits from the manifest by id and never recompute them (invariant 2). Case ids are
  a hash of the extraction parameters, so any parameter change invalidates every manifest.
- `case_nodes.parquet`/`case_edges.parquet` hold *positions* into the interim graph, valid
  only against the graph recorded in `source_manifest_hash`. Use `CaseCollection.materialise`.
- The fact layer must never put `is_laundering`, `typology` or `pattern_id` into a feature
  tensor. `leakage_audit.LABEL_PROXY_COLUMNS` is the enforced list and the check is fatal.

---

## Phase 3 — The fact layer: schema, extractor, checker, vocabulary, serialiser
**Date:** 2026-08-01 · **Gate:** passed (AMLworld on real data; Elliptic2 on a synthetic
mask fixture, blocked on access)

**Delivered**

- **`schemas/case_facts_v1.json`** — written *before* the code, as the brief requires. Draft
  2020-12, strict, `additionalProperties: false` on every object (a test walks the document
  and asserts it). **Frozen at 1.0.0**, declared in four places that a test reconciles.
- **`schemas/vocab_v1.yaml`** — the controlled vocabulary. Six entity roles with degree
  bindings; the nine typologies per substrate; **eleven risk descriptors, each bound to a
  numeric field and a condition on it**; four forbidden-phrase groups (guilt, entity type,
  completeness, motive); a three-entry regulatory whitelist; and salience lists for all nine
  typologies.
- **`src/g2t_aml/facts/`** — fifteen modules, 5,992 lines, 1,571 statements:
  - `schema.py` — the typed record and the `Unavailable` sentinel (D-025).
  - `caseview.py` — the shared reduction that fixes, once, what "degree" means and how
    self-loops are treated, so no two sub-extractors can drift apart on it.
  - `structure.py`, `temporal.py`, `flow.py`, `labels.py`, `motifs.py`, `salience.py` — the
    sub-extractors, each independently testable and each contributing its own
    `field_producers` provenance tags.
  - `extractor.py` — `extract_facts`, focal selection, role assignment, typology resolution.
  - `checkers.py` — three-valued verification, **87 registered field checkers**, the
    published tolerance policy, and text-level scanning for the forbidden vocabulary.
  - `taxonomy.py`, `vocab.py`, `config.py`, `serialiser.py`.
- **`scripts/03_extract_facts.py`** + `configs/facts/default.yaml`; `make facts`,
  `facts-elliptic2`, `facts-gate`, `facts-golden` now run instead of announcing a phase.
- **`docs/annotation/salience.md`** and **`hallucination_taxonomy.md`** — frozen in this
  phase, before any narrative exists, so annotators and the automated metric score against
  one definition (D-032).

**Gate verification** (this machine, 2026-08-01)

| Criterion | Required | Achieved |
|---|---|---|
| Round trip, 1,000 real cases | 100% SUPPORTED, 0 CONTRADICTED | **100% / 0** (43,655 claims) |
| Independent oracle, 1,000 cases | added this phase (D-034) | **0 disagreements**, 15 quantities |
| Test coverage on `facts/` | >= 90% | **96%** (1,571 statements, 40 missed) |
| `mypy --strict` on `facts/` | clean | clean |
| Golden fixtures | 20 | 20 |
| Checker per checkable field | all | **87 registered**, both directions asserted |
| Schema frozen and consistent | 1 version | 1.0.0 in **five** declarations, reconciled by test |

- `ruff check` and `ruff format --check` clean; `mypy --strict` clean over `facts/` +
  `eval/`; **839 tests: 836 passed, 3 skipped** in 171 s. The three skips are structural,
  not avoidance: the PyG adapter (the graph extra is absent by construction, D-004) and two
  Elliptic2 real-data assertions (access-gated). The fact layer is 5,992 lines of source
  against 4,907 lines of test.
- The 1,000-case round trip makes **43,655 claims (43.7 per case)** across **46 distinct
  field paths** and all five claim types — numeric 20,639, categorical 14,000, temporal
  3,228, qualitative 3,001, entity 2,787.
- Full extraction over all **30,000 built cases** in **1,858 s**: 30,000 schema-validated
  JSON records, an 81-column aggregate Parquet, and a coverage report. Mean field population
  **0.905**; 203 of 239 tracked fields fully populated. Note the corpus on disk holds all
  30,000 *built* cases; the frozen split manifest retains 16,156 of them, and facts are
  extracted for all so Phase 4 can draw from either.
- **The D-036 record invariant holds across all 30,000**: zero records name a typology
  while carrying no flagged transaction.

**Final typology distribution** (resolved from each case's own transactions, D-036), against
Phase 2's `CaseRecord` stratification, which counts the *seeding stream* instead:

| Typology | Fact records | Phase 2 stratification |
|---|---:|---:|
| gather_scatter | 289 | 310 |
| scatter_gather | 303 | 326 |
| bipartite | 286 | 357 |
| fan_out | 228 | 289 |
| stack | 214 | 314 |
| cycle | 205 | 264 |
| fan_in | 183 | 269 |
| random | 153 | 200 |
| unclassified | 28,139 | 2,100 + 25,571 unlabelled |

The two columns *should* differ and the difference is the point: a case seeded from a stream
whose flagged transactions all fell outside the 48-hour window contains no evidence of that
typology, so the fact record says `unclassified`. **The fact record is the correct column to
quote for anything describing case content**; the Phase 2 column describes how cases were
*selected*.

**Focal-entity roles observed** (30,000 cases): beneficiary 10,974, originator 8,180,
intermediary 7,042, pass_through 3,337, terminal 455, hub 12. All six controlled roles occur,
which is the check that the vocabulary is neither over- nor under-specified — a role that
never fires would be dead vocabulary, and `hub` at 12 is rare but real (five distinct
counterparties on both sides inside a 48-hour window is genuinely unusual).

**The gate could not fail, and finding that out was the most valuable hour of the session**

The round trip passed on its first run, which was suspicious enough to test. Three realistic
extractor bugs were injected — `span_hours` returning seconds instead of hours, an off-by-one
in `structure.n_nodes`, and `in_degree` counting transactions rather than distinct
counterparties — and the round trip stayed at **100% SUPPORTED for all three**.

The cause is circularity: the probe renders its claims *from the fact record*, so a wrong
value is stated wrongly and then verified against itself. The round trip tests that the
extractor and the checker agree about semantics — genuinely important, since that is what
stops the corpus and the metric drifting apart — but it says nothing about whether either is
*correct*.

`tests/oracle.py` closes the gap: fifteen quantities recomputed directly from the raw Polars
tables, importing nothing from `g2t_aml.facts`, deliberately naive and written to be read.
Against the same three mutations it flags **148/150, 150/150 and 72/150** cases, and zero at
baseline. The Phase 3 gate is now both tests. See D-034.

**Two more bugs the tests passed and the data caught**

Both were found by *reading the extractor's output* rather than by any assertion, and both
would have corrupted the Phase 4 corpus rather than the Phase 10 metric — which is the
harder failure to notice, because a corrupted corpus produces a model that is confidently
wrong rather than a number that looks odd.

*First:* the first full extraction produced this typology distribution: `bipartite` 7,337,
`gather_scatter` 6,421, `fan_in` 2,817, `fan_out` 2,004 — against Phase 2 ground-truth
stratification of 357, 310, 269 and 289. **Roughly 25,000 licit cases had been assigned a
laundering typology from their shape alone.**

The resolver fell through to motif inference whenever a case carried no stream label, which
conflated "the substrate cannot tell us" with "the substrate tells us there is no stream" —
a distinction D-013 had already drawn and the resolver was ignoring. Licit activity *is*
structurally shaped like something: a payroll run is a fan-out, supplier settlement is a
chain. Left in, every Bronze narrative from a licit case would have asserted a laundering
typology for a case the ground truth says is clean, training the generator on exactly the
inference the hard-negative population exists to prevent.

Every unit test passed both before and after. The 20 golden files caught it — 11 changed —
which is what they are for, but only once the coverage report had prompted someone to look.
Fixed in D-035; `test_licit_amlworld_case_is_ground_truth_unclassified_not_an_inferred_typology`
locks it.

*Second:* a single record was inspected by hand and found claiming
`{"typology": {"label": "cycle", "source": "ground_truth", "confidence": 1.0}}` alongside
`n_illicit_transactions: 0` and `focal_is_illicit: false` — a full-confidence ground-truth
laundering typology on a subgraph containing no flagged transaction whatsoever.

`CaseCollection.materialise` sets `CanonicalGraph.typology` from the *seeding stream's*
label, which is provenance about how the case was selected rather than a fact about what it
holds. **346 of 30,000 cases (1.2%)** carry a stream typology while containing none of that
stream's flagged edges — the 48-hour window caught the seed account and none of its
laundering (D-019). Phase 2 already records the disagreement (`label="licit"` with
`case_class="suspicious"`); it simply had no consumer until now.

The fact layer now reads the typology from the case's own edge table, restoring what
`case_extraction._dominant_typology` computes at cut time and `materialise` discards. This
establishes a checkable record-level invariant, asserted in both the unit suite and over the
real corpus: **a record naming a typology always has `labels.n_illicit_transactions > 0`.**
Fixed in D-036.

It also exposed that the golden fixtures were under-specified — they set a case-level
typology but no per-transaction one, so they did not resemble a real AMLworld positive and
the tests written against them were not exercising the real path.
`tests.factories.as_laundering_stream` now marks the transactions, and the 20 golden records
were regenerated.

**Field population rates, AMLworld HI-Small (30,000 cases, 239 tracked fields)**

| Field | Rate | Why |
|---|---:|---|
| `flow.cross_border` | **0.000** | Permanently unavailable on every substrate, by design (D-030) |
| `model_signal.*` | **0.000** | Written back in Phase 7; the write-back path is built and tested |
| `motifs.cycle.length` | 0.003 | Real cycles are rare in a 48-hour window |
| `motifs.scatter_gather.*` | 0.009 | The scarcest structural motif |
| `motifs.stack.depth` | 0.086 | Requires three consecutive layers ≥ 2 wide |
| `flow.retained` | 0.126 | Withheld across currencies, and when outflow exceeds inflow (D-033) |
| `labels.min_hops_to_known_illicit` | 0.136 | Null when no flagged account is reachable — a measured value |
| `motifs.gather_scatter.*` | 0.241 | |
| `motifs.fan_in.*` | 0.276 | |
| `motifs.fan_out.*` | 0.411 | |
| focal_entity, entity_inventory, typology, availability | 1.000 | Available on every case |

The four zero-rate families are the report doing its job: one is a deliberate design
statement (D-030) and three are the Phase 7 write-back placeholder. A run in which
`flow.cross_border` was ever populated would mean the mask had stopped being consulted.

**Per-group population**, which is the shape a reviewer should expect:

| Group | Mean rate | Fields |
|---|---:|---:|
| availability, entity_inventory, focal_entity, typology, identity | 1.000 | 23 |
| structure | 0.998 | 9 |
| provenance | 0.998 | 139 |
| temporal | 0.922 | 9 |
| labels | 0.856 | 8 |
| flow | 0.775 | 15 |
| motifs | 0.514 | 27 |
| model_signal | 0.000 | 4 |

`motifs` at 0.514 is correct rather than disappointing: a descriptor is null exactly when its
motif is absent, and most cases exhibit two or three of the eight. `flow` at 0.775 is
multi-currency withholding plus the `retained` sentinel. `structure` is 0.998 rather than
1.000 only because `diameter` is null for single-account cases.

**Salient-field requirements per case** (after availability filtering): 4 fields on 115
cases, 5 on 454, 6 on 1,247, 7 on 19,128, 8 on 9,056. No case has zero required fields, so
adequacy is always scored against something.

**Decisions recorded:** D-025 (typed absence sentinel), D-026 (burst algorithm and its
parameters), D-027 (tolerance policy per claim type), D-028 (three-valued verdicts,
UNVERIFIABLE never collapsed), D-029 (entity-type vocabulary excluded), D-030
(`cross_border` as a permanent sentinel), D-031 (`facts/motifs.py` separate from
`data/motifs.py`), D-032 (salience fixed before generation), D-033 (multi-currency
handling), D-034 (round trip paired with an independent oracle), D-035 (ground truth wins
outright), D-036 (typology read from the case, not the seeding stream).

**Extractor ↔ checker semantic disagreements found, and how resolved**

Three, all found by the round trip and all resolved by making the *extractor* explicit
rather than by relaxing the checker:

1. **Duration units.** The checker accepts a duration at the granularity the narrative
   states; the extractor emits hours. Resolved by making `DurationClaim` carry its unit, so
   "about 3 days" and "76 hours" are different claims with different tolerances (D-027) —
   rather than by picking one unit and forcing both sides to it.
2. **Degree semantics.** "Degree" meant distinct counterparties in the extractor and could
   have meant transaction count in a checker written later. Fixed once in `CaseView` and
   documented there; the record carries both under different names.
3. **The vacuous descriptor.** `rapid_dispersal` bound to `burst_window_hours < 24`, but
   that field is capped at 24 by construction, so the condition held for every burst the
   detector could report. Tightened to `<= 6` with two tests enforcing that any binding on
   that field stays strictly below the detection window (D-026).

**Deferred**

- **Elliptic2 fact extraction on real data** — still blocked on access, unchanged since
  Phase 1 and now blocking Phase 12. **The access request has still not been sent.** The
  availability path is fully implemented and tested against a synthetic Elliptic2-masked
  fixture: `tests/unit/test_facts_availability.py` asserts every monetary and temporal
  family reaches a sentinel and that every such claim returns UNVERIFIABLE rather than
  CONTRADICTED.
- **H9 (omission of exculpatory fact)** is defined in the taxonomy and documented for
  annotators, but has no automated detector. It is the only class detected by absence rather
  than assertion, so it needs the salience machinery plus judgement — Phase 6.
- **Narrative templates** — Phase 4, deliberately untouched. `tests/probe.py` is a claim
  harness, lives in `tests/`, and must not become a template.
- **Degenerate cases were NOT filtered.** Phase 2 flagged 18.1% of cases as holding fewer
  than three nodes and asked Phase 3 to decide explicitly. The fact layer extracts them
  correctly — a two-account case yields a valid record with `diameter` 1 and no motifs — so
  no filtering decision is forced here. It is a *corpus* decision, and it belongs to Phase 4,
  which is where a one-sentence narrative would start inflating surface metrics.

**Notes for Phase 4**

- **Read the availability mask, never the columns.** A field under an `Unavailable` sentinel
  must not be rendered. `facts.schema.is_available()` narrows for mypy; the serialiser is a
  worked example of doing this correctly for every family.
- **`typology.scope == "stream_membership"` means the case may not contain the whole scheme**
  (D-019, 65% of a stream's transactions on average). No template may claim completeness; the
  vocabulary's `completeness` forbidden list is enforced as H8.
- **Bronze must be faithful by construction and it can now be *proved* so.** Render, then run
  the same checker: the corpus generator and the faithfulness metric are one instrument run
  in opposite directions, and a Bronze case producing anything other than 100% SUPPORTED is a
  bug in the template.
- **Salience lists are already frozen** in `vocab_v1.yaml` and `docs/annotation/salience.md`.
  Templates should mention every required field; `salience_report()` returns the filtered
  list per case.
- **Do not assert a typology on a licit case.** After D-035 those are
  `unclassified / ground_truth`, and that is a positive statement worth rendering as such —
  "no structural laundering pattern is indicated" — not an absence to paper over.
- **A named typology now guarantees its own evidence.** After D-036, a record whose
  `typology.label` is not `unclassified` always has `labels.n_illicit_transactions > 0`, so a
  template may safely describe the scheme and cite the flagged transactions in the same
  sentence. Note this makes the fact record's typology distribution differ from Phase 2's
  `CaseRecord` stratification by the 346 evidence-free cases — the fact record is the correct
  one to quote for anything describing case *content*.
- The serialiser is the **B7 baseline**, and deliberately strong: every fact family reaches
  the text in both styles. Weakening it to flatter our own numbers would be misconduct and is
  guarded by `test_every_fact_family_reaches_the_text`.

---

## Phase 4 — Bronze: the template engine, the corpus, and the ten-point harness
**Date:** 2026-08-01 · **Gate:** passed (AMLworld on real data; Elliptic2 family built and
tested against the synthetic mask fixture, blocked on access)

**Preflight.** Phase 3's gate re-confirmed before anything was built: round trip 100%
SUPPORTED / 0 CONTRADICTED on 1,000 real cases (43,655 claims), independent oracle 0
disagreements, `facts/` coverage 96%, schema frozen at 1.0.0 in five reconciled declarations.

**Delivered**

- **`schemas/training_record_v1.json`** — FROZEN at 1.0.0. One document for all three tiers,
  differing only in `tier` and `generator`; `$ref`s the frozen fact schema rather than
  restating it. Carries `target_slots`, the character-span alignment from narrative text back
  to fact fields (D-037).
- **`src/g2t_aml/corpus/`** — ten modules, 5,397 lines:
  - `factsio.py` — the deserialiser Phase 3 never needed. Proves losslessness on **every
    record it loads** by re-serialising and comparing; a lossy read is refused, not warned
    about.
  - `bronze/format.py` — nine formatter/parser pairs, each rounding inside the checker's
    tolerance with at least a factor of two of margin.
  - `bronze/templates.py` — 12 families × 5 structural realisations, composed into **12,505
    distinct narratives** (D-042).
  - `bronze/renderer.py` — `render_bronze`, the two-layer substrate guard, slot annotation,
    salience completion, length control.
  - `claims.py` — slot annotations parsed **back out of the rendered text** into checkable
    claims (D-040). This is the module that makes Bronze's 100% a measurement.
  - `validate.py` — the ten-point harness, gating Bronze, Silver and Gold identically.
  - `dedupe.py` (MinHash + LSH, no new dependency), `diversity.py`, `graphref.py`,
    `pii.py`, `tokenization.py`, `record.py`.
- **`scripts/04_build_bronze.py`** + an extended `configs/corpus/bronze.yaml`; `make bronze`,
  `bronze-elliptic2` and `bronze-gate` now run instead of announcing a phase.

**Gate verification** (this machine, 2026-08-01)

| Criterion | Required | Achieved |
|---|---|---|
| Template families | >= 8 | **12** |
| Realisation variants per family | 4–6 structural | **5 structural, 1,080 narratives** (625 for `topology_only`) |
| Bronze records | ~15,000 | **15,707** |
| Ten-point harness | 100% pass | **15,707 / 15,707, every check at zero failures** |
| Claims checked | — | **296,195; 100% SUPPORTED, 0 CONTRADICTED, 0 UNVERIFIABLE** |
| Slot annotations | spans correct | asserted per record: `narrative[span] == rendered_value` |
| Salience coverage | — | **1.000 on every record**, by construction |
| Length | [80, 400] tokens | min 152, median 279, p95 312, max 356 |
| Diversity | self-BLEU not pathological | **0.475** at 5 refs; **12,324 distinct skeletons (78%)** |
| `training_record_v1.json` | frozen | 1.0.0 |
| Corpus rebuilds byte-identically | — | **yes**, verified against a prior build |
| `make test` | green | **1,039 passed, 3 skipped**, 0 failed |
| Coverage on `corpus/` | — | **92%** (1,236 statements); repo total 93% |

`ruff check` and `ruff format --check` clean; `mypy --strict` unchanged over `facts/` +
`eval/`. Build time **188 s** end to end: 76 s to render 15,708 narratives, 63 s for the
harness, and a second harness pass after the one near-duplicate is dropped.

**The corpus is byte-reproducible.** Rebuilt from scratch after the code was reformatted,
`bronze.jsonl` came back **byte-identical** to the previous build. Rendering is a function of
the fact record alone: family follows from the typology and a small number of documented
rules, realisation follows from a SHA-256 of `case_id`, and the seed is recorded but never
consulted — so a corpus can be regenerated from a case manifest without knowing what seed
produced it.

**Corpus composition** (15,707 records: train 10,488 / val 2,027 / test 3,192)

| Family | Records | Median words | Realisations used |
|---|---:|---:|---:|
| no_finding | 11,062 | 148 | 1,080 of 1,080 |
| minimal_activity | 2,474 | 143 | 968 |
| unclassified_suspicious | 1,338 | 150 | 751 |
| bipartite | 140 | 131 | 129 |
| gather_scatter | 137 | 129 | 127 |
| fan_out | 133 | 144 | 127 |
| scatter_gather | 120 | 130 | 110 |
| cycle | 92 | 125 | 91 |
| random | 81 | 150 | 78 |
| fan_in | 70 | 148 | 69 |
| stack | 60 | 133 | 59 |

`topology_only` has 0 records and that is correct: it is the Elliptic2 family, and Elliptic2
access has still not been requested. It is fully built and tested against the synthetic mask
fixture — it renders, and every one of the other eleven families **raises** on that fixture.

The distribution is 70% `no_finding` because the fact records say so: 15,256 of the 16,156
manifest cases resolve to `unclassified` ground truth after D-035/D-036. That is the
population, not a sampling artifact, and a corpus that rebalanced it would be training the
generator on a prevalence the deployed system will never see.

**Diversity**

| | |
|---|---|
| distinct-1 / 2 / 3 | 0.0137 / 0.0352 / 0.0665 |
| self-BLEU (5 refs) | **0.475** |
| self-BLEU curve | 1:0.16 · 3:0.36 · 5:0.48 · 10:0.63 · 50:0.82 |
| distinct skeletons | **12,324 / 15,707 = 0.785** |
| vocabulary / TTR | 31,639 types · 0.0137 |
| inter-family trigram overlap | max 0.279 (`bipartite`\|`gather_scatter`), mean 0.133 |

**The self-BLEU number nearly caused the wrong fix, and finding that out was the most useful
hour of the session**

The first build reported self-BLEU **0.811**, which reads as a collapsed template pack and
which the brief explicitly says to respond to by adding realisation variants. So we did:
decoupling the four SAR sections multiplied each family's narrative count from 5 to 1,080, a
factor of 216.

Self-BLEU moved from 0.811 to **0.810**.

That non-result is what made it worth measuring the metric rather than reacting to it. On the
*unchanged* corpus, self-BLEU is 0.16 at one reference, 0.36 at three, 0.63 at ten and 0.82
at fifty. It was measuring its own reference count: with fifty references drawn from a corpus
written over a deliberately controlled vocabulary, almost every 4-gram of any candidate
appears in *some* reference and clipped precision goes to one.

The pack had never collapsed. Pairwise self-BLEU is 0.16, and 12,324 of 15,707 narratives
have structurally distinct scaffolding once their slot values are blanked. Both numbers were
available before the "fix" and neither was being looked at.

Three things changed as a result. The reported figure is now at a **fixed, documented five
references** and the whole curve is published beside it, because a self-BLEU without its
reference count is not a number (D-043). **Skeleton diversity** was added and is what the
collapse check actually keys on — no reference sample, no saturation, and for a *template*
corpus it answers the real question directly. And the section decoupling was kept, because it
was right on the evidence that bears on the question even though it was adopted on evidence
that did not.

**Bronze is faithful because it is checked, not because it is a template**

The trap Phase 3 documented in D-034 was available here in a more dangerous form. A generator
that built its claims from the fact record — `Claim(value=facts.structure.n_nodes)` — would
compare the record with itself, and *every corpus ever built* would report 100% SUPPORTED,
including one whose formatter dropped a factor of a thousand.

So claims are parsed **out of the rendered text** (D-040). Every formatter ships with its
inverse; `raw_value` is recorded for diagnostics and never checked. `format_money` writing
"482,300 US Dollar" is verified by `parse_money` reading that string back and comparing to
the record at the published 1% tolerance. A test hands the parser a slot whose `raw_value`
deliberately disagrees with its text and asserts the claim follows the text.

The same discipline runs through the deserialiser: `load_case_facts` re-serialises every
record it reads and refuses to return unless the result is byte-identical to the file. A
narrative rendered from a mis-read record would be verified against that same mis-read record
and would pass.

**Rounding reconciliation**

The brief warned that a renderer can round outside the checker's tolerance and fail its own
ceiling. Reconciled explicitly, property-tested, and with margin:

| Quantity | Rendering | Worst-case error | Tolerance | Margin |
|---|---|---|---|---|
| Counts | exact, `,`-separated | 0 | exact | — |
| Money >= 1000 | 4 significant figures | <= 0.05% rel. | 1% rel. | 20x |
| Money < 1000 | 2 decimal places | <= 0.005 | max(1%, 0.01) | 2x |
| Duration >= 48h | whole days | <= 12 h | 24 h | 2x |
| Duration 1–48h | 1 dp, hours | <= 0.05 h | 1 h | 20x |
| Duration < 1h | whole minutes | <= 0.5 min | 1 min | 2x |
| Shares | whole percent | <= 0.005 | 0.01 | 2x |
| Density, scores | 3 decimal places | <= 0.0005 | 0.01 | 20x |
| Timestamps | minute resolution | <= 30 s | 60 s | 2x |

4 significant figures rather than 3 for money: 3 gives 0.5% against a 1% tolerance, which
leaves no room for a later change to either side. Rounded amounts are always hedged
("approximately"), because a figure rounded to 4 significant figures is not being claimed
exactly.

**The substrate guard is two-layered, and both layers are hard errors**

A family declares the availability flags it needs and is refused outright on a record lacking
one; every individual slot then re-checks. The distinction between the two kinds of absence
is read from the record rather than from a table in the renderer: `Unavailable.reason`
already encodes it. `substrate_has_no_*` is a mask fact and **raises**;
`no_transfers_in_this_direction` is a case fact and drops its sentence. Keeping the rule in
one line that consults the record means a new sentinel in `facts/` cannot land silently on
the wrong side of it.

Tested both ways: all eleven amount-bearing families raise on the Elliptic2 fixture, and an
AMLworld originator (no inbound value, `Unavailable("no_transfers_in_this_direction")`)
loses its inflow sentence and keeps its narrative.

**The harness has a deliberately broken fixture for every check**

A gate nobody has seen fail is a gate nobody has tested — a check that never fires looks
exactly like a check that always passes, and Bronze's 100% would then be evidence of nothing.
`tests/unit/test_corpus_validate.py` breaks one thing at a time and asserts the harness
catches that check: a missing field (1), a `graph_ref` that resolves to a subgraph of the
wrong size (2), a stale fact schema version (3), a rendered number moved without moving the
record (4), a claim about `flow.cross_border` (5), narratives above and below the bounds (6),
a guilt assertion, an entity-type attribution and a risk descriptor whose binding fails (7),
an email and an IBAN (8), a duplicated narrative (9), and a split that disagrees with the
frozen manifest (10). Two further tests assert the harness **fails** checks 2 and 10 when it
cannot run them, rather than skipping them and reporting a pass it never tested.

**Deduplication found almost nothing, which is the interesting result**

Over 15,708 narratives the LSH banding surfaced 1,678 candidate pairs and exact Jaccard
confirmed **one** at the 0.85 threshold. The concern was real — a template pack can
manufacture near-duplicate narratives from cases that are not duplicates — and the reason it
did not materialise is that rendered values carry into the 5-gram shingles, so two reports
about different cases differ in almost every window that contains a number.

**Deferred**

- **Elliptic2 Bronze on real data** — blocked on access, unchanged since Phase 1, and now
  blocking Phases 4, 12 and any real-world claim in the paper. **The access request has still
  not been sent.** This is the longest-lead open item in the project and it has been the
  longest-lead open item for four phases.
- **Re-measuring the length distribution under the real Llama-3.1 tokenizer.** The heuristic
  counter over-approximates by construction, so the [80, 400] gate is conservative rather
  than wrong, but the published distribution should be re-derived in Phase 9 when the GPU
  environment exists. `length.tokenizer` on every record is what makes that a comparison
  (D-039).
- **`topology_only` has never rendered a real record.** It is exercised only against the
  synthetic mask fixture. Its first contact with real anonymised data will find things.
- **Bronze's ROUGE/BLEU against Gold** — Phase 10, as the brief instructs. The design work is
  done: `bronze.jsonl` carries plain text, slot alignment and a per-record verification block,
  so scoring the template against Gold references needs no re-generation. If a template scores
  competitively on overlap metrics, that is the paper's evidence that overlap metrics do not
  distinguish real systems from templates.

**Notes for Phase 5**

- **Silver rewrites Bronze, and the harness that gates it is already written.** `validate.py`
  takes `tier` from the record; a Silver record differs only in `tier` and `generator`. Do not
  write a second harness.
- **`target_slots` is the alignment Silver's verifier needs.** A rewrite that preserves a
  span's value can carry its annotation through; one that does not must drop the annotation,
  and the dropped claim shows up as reduced coverage rather than as a silent unverified
  assertion.
- **A rewrite is rejected, never repaired** (CLAUDE.md §3). The harness returns per-check
  failures with reasons, which is enough to report *why* a rewrite was rejected without
  tempting anyone to patch it.
- **The unverifiable budget is 0.05 and Bronze sits at 0.000.** Silver will not, because an
  LLM adds connective language that resolves to no measurement. That headroom is the budget's
  purpose; spending all of it is a signal, not a pass.
- **Do not let Silver invent numbers.** Every quantity in a Bronze narrative is annotated, so
  a rewrite introducing an unannotated number is detectable by span diffing before the checker
  is even run.

**Decisions recorded:** D-037 (training record embeds its facts, one schema for three tiers),
D-038 (Bronze excludes only single-account cases), D-039 (pluggable token counter that never
falls back), D-040 (claims parsed from text, never read from the record), D-041 (`graph_ref`
resolution checks graph size against the facts), D-042 (independent section composition),
D-043 (self-BLEU at fixed references with the saturation curve published).

**One thing the contract tests caught that review would not have**

`test_no_hardcoded_data_or_artifact_paths_in_src` failed on `graphref.py`. The offending
string was in a *docstring*, showing the reference format — `"data/processed/<dataset>/
cases#<case_id>"` — and the code beneath it takes its root from `cfg.paths.*` and hardcodes
nothing. The tempting response is to relax the grep to skip docstrings.

It was reworded instead. The test is a grep and cannot tell a docstring from a constant;
narrowing it to make one benign case pass would have narrowed it for the next case too, and
the next one may not be benign. A contract test that has been taught to ignore the place a
violation is most likely to be written first is not a contract test.

## Phase 5 — Silver: the two-teacher pipeline, the verifier loop, and the discard log
**Date:** 2026-08-02 · **Gate:** **machinery complete and validated; the corpus is NOT
built.** No API credentials and no spend authorisation were available in this environment,
so not one teacher call was made. Every acceptance criterion that depends on a real
generation run is **deferred, not met**, and is listed as such below.

**Preflight.** Phase 4's gate re-confirmed: Bronze at 15,707 records, 15,707/15,707 on the
ten-point harness, 296,195 claims at 100% SUPPORTED / 0 CONTRADICTED / 0 UNVERIFIABLE, slot
annotations asserted per record. The Phase 3 checker, taxonomy and controlled vocabulary are
present and are what Silver verifies against — unchanged, not re-implemented.

**Delivered**

- **`prompts/silver_rewrite_v1.txt`, `prompts/silver_repair_v1.txt`** — versioned, content-
  hashed prompt files. Split so the system message is case-invariant (the run's prompt-cache
  prefix) and the user message holds everything per-case; the loader **refuses** a prompt
  whose system section names a per-case placeholder (D-050).
- **`src/g2t_aml/corpus/silver/`** — six modules, 3,652 lines:
  - `prompts.py` — rendering, hashing, the availability block, the hedging and forbidden
    lists, the salient list, and the eight deterministic per-case style directives.
  - `api_client.py` — `TeacherSpec`/`TeacherResponse`, the content-addressed response
    cache, retry with full jitter, the hard budget cap, the concurrency semaphore, the
    structured transient/permanent error log, and the checkpoint store.
  - `claim_extraction.py` — the fast deterministic extractor and the `ClaimExtractor`
    protocol Phase 10's LLM extractor will satisfy.
  - `generate.py` — the loop, the verdict, stratified teacher assignment, the discard
    record and the discard report.
  - `quality.py` — degeneracy checks, MinHash dedup across Silver and Bronze, the
    own-source verbatim check, per-teacher drop asymmetry.
  - `run.py` — concurrency, append-as-you-go writing, resume, budget halt.
- **`scripts/05_build_silver.py`** + a fully specified `configs/corpus/silver.yaml`;
  `make silver`, `silver-dry-run`, `silver-resume` and `silver-gate` now run instead of
  announcing a phase. `anthropic` added as an `api` extra, deliberately outside the core.
- **124 Phase 5 tests** (1,646 lines), all green, covering every scenario the brief
  names, plus a repo-wide contract test for the exit-code fix below.

**Gate verification** (this machine, 2026-08-02)

| Criterion | Required | Achieved |
|---|---|---|
| Phase 5 tests | — | **124 passed** |
| Full suite | green | **1,164 passed, 3 skipped**, 0 failed (184 s) |
| `ruff check` / `format --check` | clean | **clean**, 129 files |
| `mypy --strict` over `facts/` + `eval/` | unchanged | unchanged |
| Loop paths: clean / repair-then-pass / repair-twice-then-discard | all three | **all three tested exactly** |
| Cache: hit, key sensitivity, corruption | correct | **11 tests**; key changes on any of 9 components |
| Resume: no gaps, no duplicates | correct | **verified** on a kill-and-restart |
| Budget cap halts | must halt | **verified**, and halted cases never enter the discard log |
| Teacher assignment deterministic and balanced | required | **verified**, ±1 per stratum |
| Prompt hash tracks the file | required | **verified** |

**Scale validation without an API** (2,000 real fact records, scripted teacher)

The pipeline was driven end to end over 2,000 real AMLworld cases with a mechanical
paraphraser standing in for a teacher. This is **not** a Silver corpus and the stand-in is
not a model; it exists so that the loading, assignment, loop, filtering, harness and report
machinery is known to work at scale before any money is spent.

| | |
|---|---|
| Records generated | 1,951 written, 49 discarded (2.45%, matching the injected fault rate) |
| Repair distribution | 1,765 clean first pass, 186 recovered by one repair |
| Ten-point harness | **1,951 / 1,951 passed**, every check at zero failures |
| Claims checked | **31,380; 100% SUPPORTED, 0 CONTRADICTED, 0 UNVERIFIABLE** |
| Teacher balance | 977 / 974 kept; retention 0.978 vs 0.973 |
| Jaccard vs own Bronze | median 0.323, p95 0.440, max 0.574 — well clear of the 0.90 bar |
| Length | min 193, median 284, p95 317, max 358 tokens |
| Throughput | 2,000 cases loaded and generated in 3.4 s; harness 6.8 s |

**The scale run found a real bug that 300 unit tests did not**

At 600 cases the pipeline reported a 39% discard rate against an injected fault rate of
~13%. Chasing the gap rather than accepting the number found a genuine defect in the
extractor: alignment searched for slot values in **Bronze document order**, and a short
value could therefore be found *inside* a longer value that had not been consumed yet. The
slot rendering `2` aligned inside the timestamp `2022-09-02 15:01`; the timestamp was then
reported as a dropped fact **and** its leftover digits came back as invented quantities —
one reordering charged twice.

It cannot happen in Bronze, because Bronze reaches the long value first in its own document
order. It happens as soon as a rewrite reorders content, which is the entire point of a
rewrite. On real paraphrased cases it failed **102 of 300** with no injected fault at all.

Two changes fix it and both are tested: alignment now runs **longest value first**, and a
match is rejected unless it sits on a token boundary (so `12` cannot align inside `126`).
After the fix: **0 of 300**. Had this shipped, roughly 34 percentage points of the reported
discard rate would have been an artifact of the measuring instrument — and the paper's
headline finding for this phase *is* the discard rate.

**A second find: no pipeline script could ever exit non-zero**

While checking that the Silver entrypoint returned 1 on a preflight failure, it returned 0.
The cause is that **`@hydra.main` discards its wrapped function's return value** — it
returns None regardless — and every script in this repository ended with
`sys.exit(main())` over a decorated `main`. Every pipeline stage has therefore exited 0
unconditionally since Phase 1, *including* on the paths that carefully `return 1`.

`make bronze` on a failed ten-point gate exited 0. `make smoke` could not fail. CI would
have gone green on a corpus that failed its own gate. The gates themselves ran correctly and
their results are in the logs, so the Phase 1-4 entries stand — but nothing downstream could
distinguish a pass from a failure, so the mechanical enforcement claimed in CLAUDE.md's
invariant table was not in place for any `return 1` path.

All eight scripts now keep their Hydra entrypoint as `_run` and capture the code in
`_EXIT_CODE`, with a thin `main()` returning it. Verified: `04_build_bronze.py` exits 1 on a
schema mismatch where it previously exited 0, `05_build_silver.py` exits 1 on a preflight
failure, `smoke.py` still exits 0. A contract test asserts the shape on every script
carrying a `@hydra.main` (D-051).

**What could not be done, and why**

No `ANTHROPIC_API_KEY` or open-weights endpoint is configured in this environment, no local
inference server is available, and no spend was authorised. The following acceptance
criteria are therefore **not met and are deferred to a run with credentials**:

- [ ] **≥8,000 verified Silver records** — 0 written. The pipeline is ready.
- [ ] **Discard rate <15% with a real class breakdown** — the machinery, the log schema and
      the report are complete; the *number* requires real models. The 2.45% above is the
      injected fault rate of a stand-in and is a test of the instrument, not a finding.
- [ ] **Two teachers used, balance verified on the real corpus** — assignment and reporting
      are verified; the run is not.
- [ ] **Ten-point harness on a 500-record sample of real Silver** — the sampling gate is
      implemented; it has been exercised on 1,951 stand-in records.
- [ ] **Cost report within budget** — tracker, cap and projection are implemented and
      tested; actual spend is zero.

Everything not requiring an API call is complete. To run it: set `ANTHROPIC_API_KEY` and
`OPENWEIGHTS_API_KEY`, set `corpus.teachers[1].base_url` to a served open-weights endpoint,
then `make silver-dry-run` — which generates 20 records, prints a projected full-run cost
and writes nothing — and only then `make silver`.

**Two decisions the brief could not have anticipated**

*Sampling parameters.* The brief specifies temperature 0.7 / top_p 0.95. Every current
frontier Anthropic model — Opus 5, Sonnet 5, Opus 4.8, 4.7 — **rejects both with a 400**.
They are not ignored; a run that sent them would fail on its first call and on all 12,000
after it. Rather than downgrade the teacher to fit a decoding knob, sampling is now a
per-teacher capability: the open-weights teacher gets 0.7/0.95 exactly as specified, and the
frontier teacher gets `effort` for depth plus eight deterministic per-case style directives
for surface variety — reproducible and recorded, which a temperature draw is not. Nulls are
recorded **with a reason**, so "we did not set it" is distinguishable from "the model
refuses it" (D-045).

*Repair contradicts the standing brief.* CLAUDE.md §3 said a failing rewrite is "rejected,
not repaired". The phase brief specifies bounded repair. Bounded repair is better on the
evidence — most first-pass failures are one unaligned figure, and one targeted repair
recovers a usable record — and the fear behind pure rejection, an unbounded loop, is
answered by the hard limit of two. CLAUDE.md §3 has been **updated** rather than worked
around (D-046).

**Deferred**

- **The generation run itself.** The longest-lead item for Phase 5 and the only thing
  standing between this machinery and the acceptance criteria.
- **A real measurement of the repair rate.** The budget assumed ~20% of cases need a repair
  pass. `repair_attempts_on_accepted` is on the cost report so the assumption becomes a
  measurement on the first real run.
- **Elliptic2 Silver.** Blocked on access, as it has been since Phase 1. Now blocking
  Phases 4, 5, 12 and every real-world claim in the paper. **The access request has still
  not been sent.**
- **Whether the frontier model refuses AML narrative generation, and how often.** The
  refusal path is implemented and classified as permanent; the rate is unknown and is worth
  reporting when measured, since it bears directly on whether commercial models can be used
  for compliance tooling at all.

**Decisions recorded:** D-051 (Hydra entrypoints capture their exit code), D-044 (verified
synthetic supervision, enforced in code), D-045
(sampling as a per-teacher capability), D-046 (bounded repair; the discard log is a
deliverable), D-047 (stratified round-robin assignment with a per-stratum offset), D-048
(slot-alignment extraction; an unaligned quantity is a claim), D-049 (salience retention as
a third acceptance condition), D-050 (case-invariant system prompt, enforced at load).

---

## Phase 6 — Gold: the annotation kit, the sample, and the hold-out that makes it a reference
**Date:** 2026-08-03 · **Gate:** **the kit is complete, validated end to end on real cases,
and ready to start annotation this week. No Gold narrative has been written**, because no
annotator has been recruited. Every acceptance criterion that depends on a person is
**deferred, not met**, and is listed as such below.

**Preflight.** Phase 3's fact layer, salience lists (`docs/annotation/salience.md`) and
hallucination taxonomy confirmed present and frozen — they are what this phase's guidelines,
interface and scoring all read from, unchanged and not re-implemented. Phase 4's Bronze
corpus confirmed at 15,707 records; it is used **only** at ingestion, for its slot
alignment, and never by the interface. Silver's machinery is present and its claim extractor
is shared (and was fixed here — see below).

**Delivered**

- **`docs/annotation/annotation_guidelines.md`** (~19 PDF pages) — Parts A–F as specified:
  what a SAR is and its two readers; suspicion-vs-guilt with seventeen wrong/right pairs;
  the four-part structure with what belongs in each section; the eight typologies with
  diagrams, legitimate-vs-suspicious framing and the explicit fan-in/fan-out warning; the
  six rules; **five fully worked examples**, all from real reserved cases, each with the
  fact panel, the narrative and annotated commentary; and the H1–H9 error taxonomy with a
  wrong/right pair per class. Exported to PDF by `scripts/06c_export_guidelines.py` using
  ReportLab — pure Python, so the released artifact builds from the lockfile rather than
  from whatever `pandoc` happens to be on the machine.
- **`docs/annotation/recruitment.md`** — target profile, realistic sources in preference
  order, the explicit refusal of untrained crowdworkers with the reason, time commitment,
  the calibration gate, and **the paper-facing description of annotator expertise written
  before recruitment** so it describes the standard set rather than the people found.
- **`src/g2t_aml/human/`** — eleven modules, 6,038 lines:
  - `sampling.py` — the three-block stratification, size-bucket spread, deficits.
  - `reservation.py` — the committed test-only hold-out and the guard.
  - `caseloader.py` — assembles a case for annotation; **has no narrative field of any kind**.
  - `factpanel.py` — the readable fact record; masked families hidden entirely.
  - `graphview.py` — deterministic force-directed layout, degree/label/value encoding,
    the display cap with its on-screen notice, the timeline.
  - `validation.py` — live flags, non-blocking, over the frozen vocabulary.
  - `store.py` — append-only per-annotator capture; refuses generated text.
  - `annotation_ui.py` — the Streamlit interface.
  - `calibration.py` — the ten-case set and four-dimension scoring with targeted feedback.
  - `agreement.py` — Cohen's κ, Krippendorff's α, content Jaccard, text F1, all
    dependency-free and hand-verified.
  - `review.py` — the second-reader pass and adjudication, with independence enforced.
  - `gold_ingest.py` — annotations → `training_record_v1` at `tier="gold"`.
- **`src/g2t_aml/corpus/training_data.py`** — the training-data loader, and the place the
  "Gold is never trained on" rule is *enforced* rather than remembered.
- **Four scripts:** `06_sample_gold_cases.py`, `06b_ingest_gold.py`,
  `06c_export_guidelines.py`, `06d_build_calibration.py`. `make gold-sample`, `annotate`,
  `calibrate`, `gold`, `guidelines-pdf` and `gold-gate` now run instead of announcing a phase.
- **214 Phase 6 tests** (2,209 lines), all green.

**The Gold sample, drawn** (`make gold-sample`, this machine)

| | |
|---|---|
| Selected | **350** of 350 requested, all from the frozen **test** split |
| Hard negatives | **99 (28.3%)**, against a 25% floor |
| Typed typologies | fan_out 20 · fan_in 20 · bipartite 20 · cycle 20 · gather_scatter 19 · random 19 · scatter_gather 18 · stack 18 — spread ≤ 2 |
| `unclassified` | 196 (99 hard negatives + 97 licit/suspicious-unclassified) |
| Size buckets | small 157 · medium 129 · large 64 |
| Substrates | amlworld_hi_small 350 · **elliptic2 0** |
| Reserved test-only | **350**, sha256 `be2512b5…`, committed to `schemas/splits/amlworld/` |
| Deficits reported | `dataset:elliptic2` 105 requested / 0 supplied; `reallocated` 105/105; 3 single-account cases excluded |

`test.txt` and its sha256 are **untouched** (invariant 2). The reservation is a subset of
the existing test split recorded beside it, with its own committed id list and content hash,
and `load_reservation` asserts every reserved id really is a test-split member.

**Gate verification** (this machine, 2026-08-03)

| Criterion | Required | Achieved |
|---|---|---|
| Phase 6 tests | — | **214 passed** |
| Full suite | green | **1,386 passed, 3 skipped**, 0 failed (125 s) |
| `ruff check` / `format --check` | clean | **clean**, 154 files |
| `mypy --strict` over `facts/` + `eval/` | unchanged | **unchanged**, 16 files |
| UI renders for fixture cases, both substrates | required | **verified** — panel, graph and validation exercised on AMLworld and Elliptic2 fixtures |
| Interface runs end to end | required | **verified in a real browser** — app loaded, a real reserved case rendered (graph + timeline scrubber + fact panel), a draft typed, three critical flags raised in place with override boxes, the record check run, both CONTRADICTED verdicts shown, and Submit correctly gated behind it |
| Elliptic2 shows no monetary or currency field | required | **verified** — the Value, Timing and Counterparty sections do not exist; asserted over every rendered value |
| Live validation catches forbidden phrases | required | **verified** — 29 tests across H1/H2/H3/H4/H5/H7/H8 |
| Agreement correct against hand-computed fixtures | required | **verified** — κ, α and Jaccard each checked against a value worked out in the test, not against a second implementation |
| Gold ingestion → schema-valid records passing the ten-point harness | required | **verified on real cases**: 1/1 gate passed, 13 claims 100% SUPPORTED, 0 CONTRADICTED, 0 UNVERIFIABLE |
| 150-node case renders in acceptable time | required | **verified**, well under the 5 s bound |
| Guidelines exported to PDF | required | **19 pages, 55 KB**, built from the lockfile |

**Five real defects, found by writing a narrative by hand and by opening the app**

This is the part worth reading. Every one was invisible to 214 passing unit tests, and each
was found by doing the annotator's job once — writing a narrative against a real case, and
opening the interface in a browser.

1. **The fact panel taught a number format the alignment could not read.** Ingestion aligns
   Gold against Bronze's slot values by *exact* string match (D-048). The panel's first
   version rendered `9,434.82 Canadian Dollar` where Bronze renders `9,435 Canadian Dollar`.
   An annotator copying the panel **correctly** would have produced a value aligning to
   nothing — scored as a dropped fact *and* an invented quantity — on **every monetary case
   in the corpus**. The panel now renders through Bronze's own formatters, and a repo
   contract test draws the line between importing a formatter (permitted) and reaching a
   narrative (forbidden). The same defect existed for entity roles, where the vocabulary
   spells them `conduit account` and Bronze writes `a conduit account`; `focal_entity.role`
   is salient for three typologies, so that requirement would have been permanently
   unmeetable. (D-054)
2. **The panel taught a regulatory citation that scores as a Critical error.** It rendered
   "the 10,000 US Dollar reporting threshold"; the whitelist carries "the USD 10,000
   reporting threshold". Any other wording is H6. Fixed, and the row now says "cite the
   threshold in exactly these words if you mention it".
3. **A whitelisted citation was costing Silver 6% of its unverifiable budget.** The shared
   claim extractor had no regulatory pass, so `10,000` inside a permitted citation aligned
   to no slot and became a candidate addition. A rewrite citing a threshold the vocabulary
   *explicitly permits as context* was penalised for it. Added `_regulatory_claims`, which
   matches whitelist phrases only — a non-whitelisted citation still falls through to the
   H6 check, so the pass cannot launder an invented rule. This affects Phase 5 as much as
   Phase 6 and would not have surfaced until a real Silver run. (D-057)
4. **The interface did not start at all**, and only a browser could show it.
   `configs/paths/default.yaml` resolves `root` as
   `${oc.env:G2T_AML_ROOT,${hydra:runtime.cwd}}`. OmegaConf evaluates a nested
   interpolation in a resolver's *argument list* eagerly, so the `hydra:` fallback runs
   whether or not the environment variable is set — and a bare `compose()` leaves the
   `HydraConfig` singleton empty, so **every** `cfg.paths.*` lookup raised
   `InterpolationResolutionError` on the first page load. `@hydra.main` sets that
   singleton as a side effect, and every other config test in this repository goes
   through it, so the failure existed on exactly the one entrypoint that cannot use the
   decorator and on no test. Fixed by composing with `return_hydra_config=True` and
   calling `HydraConfig.instance().set_config` — what the decorator does, done
   explicitly. **A unit test suite of 214 passing tests reported the interface as
   working.** It was not.
5. **The graph view would have coloured every Elliptic2 node "unflagged".** It read
   `availability.node_labels`, which is **True** for Elliptic2 — the substrate labels whole
   subgraphs — while no individual account is labelled at all. Grey-as-licit is invariant 4
   violated in pixels, and harder to catch than the same claim in text because nobody writes
   it down. Now reads `CaseView.has_labels`, the same property the fact layer gates
   `LabelFacts` on, so the picture and the record agree by construction.

**Two more findings, recorded rather than patched**

- **The frozen guilt list has a word-order gap.** It carries `is money laundering` and `is
  guilty of` but not `is laundering money`, which passes the text scan. Found while writing
  the overclaiming test fixture. `vocab_v1.yaml` is frozen and editing it to make a test
  pass is what makes a frozen artifact meaningless; whether to bump it is a Phase 10
  decision with a corpus regeneration attached. The vocabulary's *primary* defence is that
  the words are excluded from generation; `check_narrative_text` is defence in depth, and
  this is a gap in the second layer only. (D-055)
- **The ten-point harness cannot see an unaligned quantity in Gold or Silver.** It rebuilds
  claims from `target_slots`, and a Gold record's slots are exactly the values that *did*
  align — so an invented figure produces no slot and is invisible to check 5. Enforced at
  ingestion instead, against the extractor's own rate. The first hand-written narrative was
  correctly held by it. Making the harness itself re-extract would change what "the
  ten-point gate" means for Phases 4 and 5 too, and belongs with Phase 10. (D-057)

**One thing the ten-point harness caught that nothing else would have**

`training_record_v1` is frozen and *requires* `generator.renderer_version` on every record.
Gold's narratives are written by people. Rather than bump a frozen schema — invalidating
15,707 Bronze and Silver records so one field could be absent from 350 — Gold writes the
ingestion pipeline's version there, with `method: human`, `annotator_id` and `protocol`
beside it. The harness found this on the very first hand-written narrative, across a tier it
had never seen. (D-056)

**Annotator recruitment status**

| | |
|---|---|
| Annotators recruited | **0** |
| Annotators calibrated | **0** |
| Reviewer identified | **no** |
| Adjudicator identified | **no** |
| Target | 3 annotators (2 minimum), calibrated before production begins |

The calibration set is **built** — `data/processed/amlworld_hi_small/gold/calibration.json`,
10 cases spanning all nine typologies — with every `reference_narrative` **empty**. The
project lead writes the references and commentary before anyone calibrates;
`score_annotator` refuses to score against blank references rather than passing everyone
against nothing.

**Target completion date for 250 items.** At ~15 min/item and 250 items = ~62 person-hours.
With 3 annotators at ~6 h/week that is **~4 weeks of annotation**, plus 1 week for
recruitment and calibration and ~1 week for the second-reviewer pass and adjudication —
**≈6 weeks from the day the first annotator is contracted**. Nothing in that estimate is
blocked on code. Every day the recruitment does not start moves the date one day, and Gold
is the credibility anchor of the paper.

**Deferred — everything that needs a person**

- **≥2 annotators recruited and calibrated.** The phase's acceptance criterion and its
  critical path. No amount of further engineering advances it.
- **The reference narratives for the calibration set.** A project-lead task, ~2.5 hours,
  and the blocker in front of the first annotator's first day.
- **The Gold narratives themselves**, the agreement numbers, the calibration results and the
  measured minutes-per-item. Every one of these is computed by tested code the moment there
  is data; none of them exists now, and none is estimated anywhere as though it did.
- **Elliptic2 Gold.** Blocked on access, as it has been since Phase 1. It is now blocking
  Phases 4, 5, 6, 12 and every real-world claim in the paper. **The access request has still
  not been sent.** This is the longest-lead open item in the project and has been for five
  phases.

**Decisions recorded:** D-052 (the hard-negative floor beats typology balance, and why they
collide), D-053 (Elliptic2's quota is reallocated and the deficit reported), D-054 (the
panel renders with Bronze's formatters; Bronze reaches the interface nowhere else), D-055
(live validation flags but never blocks; overrides are a deliverable), D-056 (Gold reuses
`renderer_version` rather than bumping a frozen schema), D-057 (Gold's unverifiable budget
is enforced at ingestion, because the harness cannot see an unaligned quantity).

---

## Phase 7 — The GAT encoder: six arms, one honest comparison, and a strong control
**Date:** 2026-08-04 · **Gate:** passed (AMLworld; Elliptic2 untouched, blocked on access)

**Delivered**

- **`src/g2t_aml/models/encoder/`** — ten modules:
  - `positional.py` — Laplacian eigenvector PE (8) and random-walk PE (16) over the
    undirected, simple, self-loop-free projection, matching `caseview`'s adjacency
    convention so a motif the fact record reports and a motif the encoder can see are
    computed over the same graph.
  - `features.py` — the case-to-tensor boundary. 27 case-local node features, 14
    continuous edge features and three categorical edge fields. Source columns are a
    whitelist, checked against `leakage_audit.LABEL_PROXY_COLUMNS` on every build.
  - `dataset.py` — the frozen-manifest reader, the typology target, the feature cache
    and its verification against the committed split hashes.
  - `base.py` — `EncoderOutput`, the `GraphEncoder` protocol, the edge encoder, the
    k-token attention-pooling readout, and the two heads. Every arm shares all of it.
  - `arms.py` — GATv2, GINE, GraphSAGE, GCN, a virtual-node graph transformer, and the
    MLP control. Each overrides `message_passing` and nothing else.
  - `registry.py`, `losses.py` (focal **and** weighted BCE), `metrics.py`, `train.py`,
    `analysis.py`, `attention_viz.py`.
- **`scripts/07_train_encoder.py`** (Hydra, W&B, checkpointing, resumable),
  **`07b_score_cases.py`** (the `model_signal` write-back), **`07c_report_tables.py`**
  (renders the tables below from `encoder_report.json`, so the write-up quotes the
  artifact rather than retyped numbers).
- Six `configs/encoder/*.yaml`, `configs/training/encoder.yaml`,
  `configs/experiment/encoder_{sweep,debug}.yaml`; `make train-encoder`,
  `encoder-features`, `encoder-debug`, `score-cases`, `encoder-gate`.
- 132 new tests across five files.

**The feature cache** (`data/processed/amlworld_hi_small/encoder/features/`, gitignored)

26,156 cases encoded in 186 s: train 10,932 / val 2,028 / test 3,196 — **exactly the
frozen manifest's counts and ids, in manifest order** — plus the 10,000-case
realistic-imbalance stream. Node input width 51 (27 features + 8 + 16 positional).
Positives 1,553 / 314 / 508 / 730. `verify_cache_against_manifest` compares the cache's
per-split id hashes against the committed manifest's on every run.

### Results across arms

Three seeds (42, 43, 44) per arm, 100 epochs, early stop patience 15 on **val AUC-PR**.
Balanced test split prevalence 15.9%; realistic stream 7.3%.

| Arm | Params | test AUC-PR | test AUC-ROC | realistic AUC-PR | typology macro-F1 (structural) |
|---|---:|---|---|---|---|
| `graph_transformer` | 1,021,530 | **0.8877 ± 0.0190** | 0.9648 ± 0.0067 | 0.8973 ± 0.0122 | 0.321 ± 0.028 |
| `gin` | 1,021,277 | 0.8801 ± 0.0056 | 0.9585 ± 0.0035 | 0.8409 ± 0.0245 | 0.272 ± 0.023 |
| `gatv2` *(primary)* | 628,058 | 0.8720 ± 0.0136 | 0.9541 ± 0.0060 | 0.8944 ± 0.0055 | 0.337 ± 0.015 |
| `mlp` *(control)* | 1,008,218 | 0.8017 ± 0.0407 | 0.9211 ± 0.0139 | 0.7992 ± 0.0211 | 0.283 ± 0.022 |
| `sage` | 613,466 | 0.7861 ± 0.0101 | 0.9066 ± 0.0025 | 0.8617 ± 0.0218 | 0.356 ± 0.022 |
| `gcn` | 416,858 | 0.7137 ± 0.0213 | 0.8862 ± 0.0110 | 0.7786 ± 0.0470 | 0.319 ± 0.024 |

Typology chance (stratified guesser on the same subpopulation) is **0.142**, so every arm
is well above chance and `gatv2` is 2.4× it. Note `sage` posts the best typology macro-F1
while sitting fifth on AUC-PR: the two heads are not measuring the same thing, which is
why both are reported.

### The gate: GATv2 against the MLP control — **PASSED**

| Seed | AUC-PR difference | 95% CI (paired bootstrap, 2,000 resamples) | excludes zero |
|---|---:|---|---|
| 42 | +0.0489 | [+0.0279, +0.0690] | yes |
| 43 | +0.0599 | [+0.0409, +0.0799] | yes |
| 44 | +0.1020 | [+0.0781, +0.1264] | yes |
| **mean** | **+0.0703** | — | **all three** |

Marginal intervals at seed 42 are non-overlapping as the brief asks: `gatv2`
0.8865 [0.8663, 0.9055] against `mlp` 0.8376 [0.8133, 0.8616]. On the realistic-imbalance
stream the mean difference is **+0.0952**, also excluding zero at every seed.

**But the control is much closer than a two-epoch smoke run suggested, and that is the
finding.** At two epochs the gap read +0.16; fully trained it is +0.07. A DeepSets model
over case-local node features — degree, transaction counts, reciprocity, local clustering,
currency-standardised amount aggregates, burst timing — reaches 0.80 AUC-PR without any
information crossing an edge. Message passing is worth a real but modest 7 points on top
of that. Had the control been under-tuned, this phase would have reported a 16-point
advantage for topology that does not exist. **The Phase 9 fusion ablation should be read
against 0.80, not against zero.**

### The primary arm against the others

| `gatv2` minus | mean AUC-PR difference | per-seed | excludes zero at every seed |
|---|---:|---|---|
| `graph_transformer` | −0.0157 | −0.0182, −0.0213, −0.0077 | **no** |
| `gin` | −0.0082 | +0.0003, −0.0054, −0.0195 | **no** |
| `mlp` | +0.0703 | +0.0489, +0.0599, +0.1020 | yes |
| `sage` | +0.0859 | +0.1008, +0.0936, +0.0632 | yes |
| `gcn` | +0.1582 | +0.1599, +0.1806, +0.1341 | yes |

**GIN and the graph transformer both out-score GATv2 on the mean, and neither difference
is significant.** The brief asked for this to be said plainly if it happened, and it
happened. GATv2 remains the primary arm; D-064 records the decision, the evidence, and
what would change it.

### Ablations on the primary arm

| Ablation | mean AUC-PR cost | per-seed | excludes zero at every seed |
|---|---:|---|---|
| edge features zeroed | **+0.1138** | +0.1125, +0.1388, +0.0901 | **yes** |
| positional encodings zeroed | +0.0023 | +0.0150, −0.0035, −0.0045 | no |
| weighted BCE instead of focal | −0.0056 | +0.0139, −0.0199, −0.0108 | no |

Three results, and two of them are negative.

**Edge features carry a large share of the signal.** Zeroing them costs 11.4 points and
drops GATv2 below the MLP control — amount, currency, payment rail and
time-since-previous-transaction are doing more work than the topology they travel on. The
brief's warning that `GATv2Conv` accepts `edge_attr` and most implementations forget to
pass it is well taken: an implementation with that bug would have scored 0.758 and looked
plausible.

**The positional encodings contributed nothing measurable.** +0.0023, and the sign flips
across seeds. They were included on the theory that 1-WL-bounded message passing cannot
count or detect cycles and that our typologies are structural motifs. On this substrate
the theory did not pay: the case subgraphs are small (median 6 nodes, p90 30), and at that
size a 3-layer network reaches most of the graph anyway, so there is little for a
positional coordinate to add. They are kept — they cost 24 of 51 input dimensions and no
measurable accuracy — but **the paper must not claim they help.**

**Focal loss did not beat weighted BCE.** −0.0056 in favour of BCE, not significant, sign
flipping across seeds. Focal is retained as the configured default because it is what the
Phase 7 brief specified and because the two are indistinguishable, but the honest
statement is that on this corpus the focusing term bought nothing over inverse-frequency
weighting. **Reporting "we used focal loss for the imbalance" without this comparison
would have implied a benefit that was never measured.**

### Embedding quality — quantified, not illustrated

| Arm | kNN purity k=5 (null) | z | k=10 | k=20 | silhouette (null) | probe acc. (shuffled) | probe structural macro-F1 |
|---|---|---:|---|---|---|---|---:|
| `gatv2` | 0.9470 (0.7864) | 61.8 | 0.9326 | 0.9221 | **0.2980** (−0.1036) | 0.9146 (0.1674) | **0.3264** |
| `graph_transformer` | 0.9469 (0.7863) | 57.2 | 0.9350 | 0.9247 | 0.3403 (−0.1253) | 0.9180 (0.2431) | 0.3019 |
| `sage` | 0.9475 (0.7861) | 58.1 | 0.9299 | 0.9195 | 0.1608 (−0.0851) | 0.8992 (0.2735) | 0.3250 |
| `gin` | 0.9439 (0.7854) | 59.1 | 0.9278 | 0.9155 | 0.2033 (−0.0975) | 0.9089 (0.2225) | 0.2661 |
| `gcn` | 0.9407 (0.7851) | 58.9 | 0.9255 | 0.9126 | 0.0696 (−0.0830) | 0.8783 (0.1949) | 0.3230 |
| `mlp` | 0.9391 (0.7858) | 58.5 | 0.9277 | 0.9166 | 0.0593 (−0.0761) | 0.8924 (0.2422) | 0.2677 |

The probe is a multinomial logistic regression fitted on **train** embeddings and scored on
**test**, with balanced class weights; both probes converged (51 and 136 lbfgs iterations).
`pooled_tokens` is probed as well as `graph_embedding` because the pooled tokens are what
Phase 8 consumes.

Three things this table says that a UMAP could not.

1. **kNN purity is far above its shuffled-label null** (0.947 against 0.786, z = 61.8) —
   but note the null is 0.786, not 0.111, because 88.5% of test cases are `unclassified`.
   The raw purity number alone would have been meaningless.
2. **Silhouette separates the arms in a way AUC-PR does not.** GATv2 and the graph
   transformer reach 0.30–0.34 against a null near −0.10; the MLP and GCN sit at 0.06–0.07,
   barely above their nulls. The message-passing arms build genuinely separated typology
   regions; the control gets its accuracy without one.
3. **The linear probe is the warning.** Overall accuracy is 0.91 against a shuffled-label
   null of 0.17 — but that is carried by `unclassified` (per-class F1 0.986). Restricted to
   the eight structural typologies the probe reaches **0.326 macro-F1**, and the per-class
   breakdown is uneven: `fan_out` 0.59, `gather_scatter` 0.43, `cycle` 0.38, `fan_in` 0.38,
   `bipartite` 0.32, `scatter_gather` 0.25, `stack` 0.10, `random` 0.09.

**This is the number that forecasts Phase 8, and it is a caution rather than a green
light.** The fusion layer is a linear projection into the language model's embedding
space; if a logistic regression on the pooled tokens can only reach 0.33 macro-F1 on the
eight typologies, the LLM is unlikely to name `stack` or `random` reliably. Only 367 test
cases carry a structural typology at all, so the estimate is itself noisy. Phase 8 should
expect typology naming to work for `fan_out`, `gather_scatter` and `cycle` and to fail for
`stack` and `random`, and should measure it per typology rather than in aggregate.

UMAP figures are in `artifacts/figures/encoder/umap_<arm>.png` with the purity and probe
numbers printed in the caption, as illustrations only.

### Attention alignment: the encoder does look at the laundering path

Over 198 suspicious test cases (those with both flagged and unflagged accounts, so the
measurement is defined), **85.1%** of the pooling attention mass falls on accounts that
touch a flagged transaction, against a **47.8%** uniform-attention baseline — a lift of
**1.78**. The single highest-attention account is on the laundering path in **87.4%** of
cases.

| Typology | attention lift | | Typology | attention lift |
|---|---:|---|---|---:|
| unclassified | 2.48 | | bipartite | 1.35 |
| stack | 1.48 | | gather_scatter | 1.28 |
| random | 1.42 | | scatter_gather | 1.21 |
| fan_in | 1.41 | | fan_out | 1.08 |
| | | | cycle | 1.02 |

The per-typology breakdown is the interesting part. Lift is near 1.0 on `cycle` and
`fan_out` — on those the encoder attends roughly uniformly, which makes sense when most of
the case *is* the pattern — and highest on `unclassified`, where the flagged accounts are a
small minority of the case and picking them out is a real discrimination. This is what
makes `model_signal.top_contributing_nodes` defensible as an investigator-facing
attribution rather than decoration. Figures in `artifacts/figures/encoder/attention/`.

### `model_signal` write-back

`make score-cases` scores all 26,156 cached cases with `gatv2-seed42-epoch31` (val AUC-PR
0.8369) and writes `gnn_risk_score`, `score_percentile` and the top five
`top_contributing_nodes` into the **16,156** fact records that exist for them, in 641 s.

Three populations, and the arithmetic is worth stating so nobody assumes otherwise. Facts
exist for all **30,000** *built* cases. The frozen manifest retains **16,156** of them, and
those are the ones the encoder was trained on and scores — every one now carries a signal.
The other **13,844** were dropped by the temporal split (straddling a boundary, or inside
the 6-hour buffer), are in no split, feed nothing downstream, and keep a null
`model_signal`. The **10,000** realistic-stream cases are scored but have no fact record to
write into. `score_percentile` therefore ranks a case against exactly the 16,156 that
receive one — not against the realistic stream, which is drawn at a different prevalence
and would make the percentile uninterpretable.

Every rewritten record is re-validated against the frozen schema. `facts.parquet` gains the
two score columns and a `model_signal_version` column.

**Bronze is deliberately not regenerated, and finding out why was worth the hour.** No
Bronze template reads `model_signal`, so not one of the 15,707 narratives changes. But
`facts.serialiser._compact` *does* emit `gnn_risk_score`, and that string is stored as
`serialised_facts` on every training record — which is the input to the **serialisation
baseline**, the "flatten the facts, no graph encoder" ablation arm. Regenerating Bronze
after the write-back would put the encoder's own score into the baseline it exists to be
compared against, and nothing would fail.
`tests/unit/test_encoder_writeback.py::test_bronze_serialised_facts_carry_no_model_signal`
now pins it. See D-063.

**Gate verification** (this machine, 2026-08-04)

| Criterion | Required | Achieved |
|---|---|---|
| All six arms train and evaluate | 6 | **6**, plus 3 ablation arms |
| GATv2 beats the MLP on AUC-PR, non-overlapping CIs | yes | **+0.0703**, excludes zero at all 3 seeds; marginal CIs non-overlapping |
| Typology macro-F1 well above chance | yes | **0.337 ± 0.015** against 0.142 chance |
| kNN purity, silhouette, linear probe vs shuffled null | quantified | all three, all with nulls |
| ≥ 3 seeds per arm, mean ± std | 3 | **3**, with paired bootstrap CIs |
| Evaluated on balanced **and** realistic-imbalance | both | both, every arm |
| `model_signal` written into all fact records | all | **16,156 / 16,156** |
| Attention visualisations | produced | 6 figures + the alignment measurement |

- `ruff check`, `ruff format --check` clean over 176 files; `mypy --strict` clean.
- **1,515 tests: 1,511 passed, 4 skipped** in 156 s (2 plotly, 2 Elliptic2 real-data; all
  absent by design).
- Determinism was confirmed by accident and then on purpose: the resume guard initially
  refused a checkpoint over a config type mismatch and retrained `gcn` seed 42 from
  scratch, reproducing test AUC-PR **0.7266** to four decimal places.
- Total training: 27 runs, ~4 h 10 min on one RTX 2050 (4 GB).

**Four things found by running it, not by testing it**

1. **The sweep was OOM-killed 24 runs in.** Retaining full `[n, 16, 256]` pooled tokens
   for nine arm-tags at three seeds across four splits needs ~12 GB; the machine has 7.
   `Predictions.slim()` now drops the embedding arrays everywhere except the arms at the
   first seed, and only for `train` and `test` — the two splits the probe actually reads.
   The 24 completed runs were recovered through the resume path rather than retrained.
2. **A two-epoch smoke checkpoint nearly became a published number.** `resume` would have
   loaded the leftover `gatv2_bce` checkpoint from an earlier wiring check and reported it
   as a converged result. Checkpoints now carry their `training_config` and resume refuses
   any whose `epochs`, `loss`, `lr` or `batch_size` differ from the active config. The one
   stale file was deleted and those three runs were trained properly.
3. **The write-back broke a Bronze test, and the test was right to break.** D-037 embeds
   the fact record in every training record so a narrative can be re-verified without the
   fact store, and an integration test asserts the embedded copy equals the on-disk one.
   Populating `model_signal` made them differ. The resolution is not to regenerate Bronze
   (D-063 forbids it) but to recognise that the embedded copy is a *snapshot of what the
   narrative was written from* — and the narrative genuinely was written from a record
   with no model signal. The test now compares every block except `model_signal` and still
   requires the rest to match byte for byte, so a real drift is still caught. Without that
   test the divergence would have been discovered in Phase 9, by something subtler.
4. **The linear probe was not converging**, hitting its 1,000-iteration cap on the
   4,096-dimensional pooled-token matrix and quietly under-reporting what is linearly
   decodable — precisely the number Phase 8 depends on. Raising `max_iter` to 4,000 and
   `tol` to 1e-3 makes it converge in 51–136 iterations, and `ProbeResult.converged` is now
   reported so an unconverged probe is visibly a lower bound rather than a measurement.

**Deferred**

- **Elliptic2 remains untouched.** The feature layer handles a masked substrate — absent
  fact families contribute zeros plus a mask channel, mirroring D-025 — and the code path
  is unit-tested against a synthetic mask, but no Elliptic2 case has been encoded because
  access has still not been requested. This is now blocking Phases 11 and 12 as well.
- **Hyperparameter search.** Every arm ran at the brief's configuration; no arm got a
  tuning budget beyond that. The comparison is therefore between architectures at one
  reasonable setting, which is what the ablation needs, but it is not a claim that any arm
  is at its best. Phase 11 should say so.
- The auxiliary-head ablation (`typology_weight=0`) was built and configured but not run;
  it is one sweep away and belongs with Phase 11's grid.

**Notes for Phase 8**

- Consume `EncoderOutput.pooled_tokens` — `[B, 16, 256]` — directly. `configs/fusion/prefix.yaml`
  now interpolates both `graph_dim` and `num_prefix_tokens` from the encoder config, so the
  fusion layer cannot silently disagree with the encoder about widths. `encoder.out_dim` is
  gone: an arm's output width *is* its `hidden_dim`.
- **Expect typology naming to be partial.** Linear probe structural macro-F1 is 0.33;
  `fan_out`, `gather_scatter` and `cycle` are recoverable, `stack` and `random` are not.
  Measure per typology.
- The frozen encoder is `artifacts/checkpoints/encoder/gatv2/gatv2_seed42.pt`, and it
  carries its own `feature_space` and `encoder_config`. A checkpoint without its feature
  space cannot be applied to a case; `07b` refuses to run when the two disagree.
- **The MLP control is at 0.80.** Any fusion result must be read against a graph encoder
  that a no-message-passing baseline nearly matches, not against a blank page.

## Phase 8+9 — The fusion layer, the training harness, the guard, and a gate that could not be run
**Date:** 2026-08-04 · **Gate:** **the harness is complete and tested; no arm was trained.**
Three preflight conditions failed and none is recoverable by writing code. **Gate 8 — the
project's decision point — remains open.** Every acceptance criterion that depends on a
training run is **deferred, not met**, and is listed as such below.

### Preflight, and why the session's mission could not be carried out

| Condition | Required | Actual |
|---|---|---|
| Phase 5 — Silver | ≥8,000 verified | **0.** Machinery complete and validated at scale on 2,000 real cases; no API credentials, so zero teacher calls (Phase 5 entry). |
| Phase 6 — Gold | held out for test | **0 narratives.** Kit complete, sample drawn and reserved; no annotator recruited. |
| Phase 7 — encoder | trained | **met.** `gatv2_seed42.pt`, test AUC-PR 0.8720 ± 0.0136. |
| Phase 8 — fusion | variants + control | **did not exist.** `configs/fusion/prefix.yaml` pointed at `g2t_aml.models.fusion.PrefixFusion` with the comment `# implemented in Phase 8`. Built this session. |
| GPU | 24 GB min, 48 GB comfortable | **RTX 2050, 4 GB** (3.3 GB free); 7 GB system RAM, 7 GB of swap already used. |

**The GPU is a hard stop, not a tight fit.** Llama-3.1-8B at nf4 with double quantisation
is ~4.5–5.6 GB of weights alone — more than the card's total capacity before a single
activation, LoRA parameter or optimiser state. CPU offload is closed by the 7 GB of system
RAM. The brief's OOM guidance (`max_seq_len` 2048 → 1536, LoRA `r` 32 → 16, fewer query
tokens) buys back *activation* memory on a 24 GB card; none of it addresses weights that do
not fit. Phase 7's 27 encoder runs fit here in 4 h 10 min because the encoder is 628k
parameters. See D-068.

Silver and Gold compound it: the configured curriculum is Bronze+Silver for epoch 1 and
Silver only for epochs 2–3, so with Silver at zero two of three epochs have no data, and
with Gold unwritten there is no held-out human reference.

**What was therefore built instead:** everything that does not depend on the missing inputs,
verified on CPU against a stub backbone.

### Delivered

**`src/g2t_aml/models/fusion/`** — Phase 8, five modules:
- `base.py` — `FusionOutput`, the `GraphFusion` protocol, three projectors (linear, MLP,
  perceiver resampler), `embedding_rms`, and **`assert_projector_is_fp32`**, which refuses
  any non-fp32 fusion parameter and any `bitsandbytes` submodule (D-069).
- `variants.py` — `PrefixFusion`. F1 and F2 differ in **one flag**, so the comparison
  measures the gate rather than two separately-tuned models. The gate is sigmoid-bounded to
  `(0, 1)` and starts at ~0.62, so "the gate is at 0.6" means one thing and an unbounded
  scalar cannot hide a badly-scaled projector by growing.
- `control.py` — **`ShuffledGraphFusion`, the A1 arm.** Derangement rather than
  `randperm`, shuffle before projection, ring-buffer fallback for singleton batches, and
  `ShuffleStats.n_fixed_points` which must be zero (D-071).
- `diagnostics.py` — attention mass **always reported against its uniform baseline**, plus
  the gate summary including the per-token minimum.
- `__init__.py`.

**`src/g2t_aml/models/generator/`** — Phase 9, eight modules:
- `model.py` — the QLoRA wrapper, the embedding splice, `trainable_parameter_groups`
  (three rates, D-070), and `state_for_checkpoint` — the fusion and encoder state PEFT's
  `save_pretrained` does *not* save.
- `prompts.py` — prompts assembled as **typed segments**, so every position's role is known
  exactly rather than recovered by searching for a marker in the token stream. Truncation
  removes facts, never the completion and never the soft tokens.
- `dataset.py` — the collator (padded text + PyG `Batch`, same case order), the curriculum
  as data, and `assert_no_gold_test` re-asserted on the *assembled* curriculum.
- `train.py` — the loop, `LambdaLR` preserving the three-rate ratio, per-group gradient
  norms, `overfit_check`, and checkpoints carrying their `training_config` (D-067).
- `callbacks.py` — `FaithfulnessCallback` (ten fixed probe cases, Phase 3 checker, the
  within-run shuffled control on the same axes from step 0), `AttentionMassCallback`,
  and `compare_arms` for Gate 8 (D-072).
- `inference.py` — greedy for measurement, sampled for the guard, resumable over the test
  set. vLLM evaluated and rejected with the reason recorded (D-074).
- `guard.py` — n=4 candidates, verify, select, one constrained regeneration naming the
  violations, else a machine-readable warning block. Both rows recorded (D-073).
- `profiling.py` — peak allocated **and** reserved VRAM, tokens/sec, wall time, and a
  `deviations` list; the profile survives an OOM because that is a data point Phase 13
  wants.

**Configs, scripts, targets:** `configs/training/generator.yaml`, five arm configs
(`generator_{s1,a1,s2,b7,b8}`) plus `generator_debug`, an expanded `configs/fusion/prefix.yaml`
and `configs/generator/llama31_8b_qlora.yaml`; `scripts/09_train_generator.py` (Hydra, run
context, overfit gate) and `scripts/09b_compare_arms.py`; `make train-generator`, `train-s1`
… `train-b8`, `generator-debug`, `gate8`, `generator-gate`.

**119 new tests** across five files, plus `tests/stubs.py` — a positional-embedding
transformer and deterministic tokeniser satisfying the `CausalLM` and `Tokenizer` protocols,
so the whole harness is exercised on CPU with neither `transformers` nor `peft` installed.
Same discipline as `ScriptedTeacher` in Phase 5.

### Test results (this machine, 2026-08-04)

| Suite | Result |
|---|---|
| `tests/unit/test_fusion.py` | 34 passed |
| `tests/unit/test_generator_harness.py` | 31 passed |
| `tests/unit/test_generator_guard.py` | 24 passed |
| `tests/unit/test_generator_configs.py` | 19 passed |
| `tests/integration/test_generator_pipeline.py` | 11 passed |
| **Full suite** | **1,630 passed, 4 skipped** (the same four: 2 plotly, 2 Elliptic2) |
| `ruff check` / `format --check` | clean |
| `mypy --strict` (facts + eval) | unchanged, clean |

The brief's named requirements, each asserted rather than inspected: forward pass produces
correct logit shapes with soft tokens; **loss is zero on prompt and soft-token positions**
(and on padding, and no row is left unsupervised); **the overfit test passes — 20 examples,
100 steps, loss 6.45 → 0.0095**; gradient reaches encoder and fusion; the projector is fp32
(assertion, and a fp16 projector and a fake `bitsandbytes` layer are both refused); Gold
test items are absent from training data; the guard selects the highest-scoring candidate on
a real fact-record fixture; checkpoint save/load round-trips fusion and encoder state.

### Three things found by building it, not by planning it

1. **The A1 config did not shuffle, and nothing would have failed.** The arm configs first
   expressed their settings as a nested `overrides:` block of dotted keys under
   `experiment`. Hydra never applies such a block — it is inert data — so
   `experiment=generator_a1` composed with `fusion.shuffle=false`. **The control would have
   trained as a second copy of the treatment.** The run would have completed, the two
   curves would have been near-identical, and the conclusion drawn from that comparison —
   "S1 does not beat A1, there is no architecture contribution" — would have been drawn
   from two runs of S1. It was caught by printing the composed config for all five arms
   rather than by any test. `tests/unit/test_generator_configs.py` now pins every arm's
   composition, including an assertion that A1 differs from S1 in the shuffle flag **and
   nothing else**.
2. **The overfit test initially failed for a reason that had nothing to do with the
   harness.** The stub backbone had no positional embeddings, which makes self-attention
   permutation-invariant and memorising a *sequence* impossible. The check plateaued at
   loss 1.43 and would have been "fixed" by loosening its threshold — which would have left
   a test that passes whatever the harness does. Adding positional embeddings and an FFN
   took it to 0.0095.
3. **`salience_report` takes a vocabulary, not a narrative.** The guard's coverage term was
   written against an imagined API and would have raised on the first real case. Coverage is
   now the share of the typology's *required* salient fields the extracted claims mention,
   with unsupported fields excused by the report itself (invariant 4).

### Release URLs

| | |
|---|---|
| Repository | **<https://github.com/MobsLInep/graph2text-aml>** — public |
| Tagged release | **<https://github.com/MobsLInep/graph2text-aml/releases/tag/v0.1.0>** |
| Commit | `4899739` (3 commits total; history rewritten to carry sole authorship) |
| Zenodo DOI | **NOT MINTED** — see below |
| HuggingFace Hub | **NOTHING UPLOADED** — see below |

**Verified against the published repository, not against the working tree:** a fresh
`git clone` of the public URL into an empty directory gives **406 files / 7.4 MB**, and
`uv sync --frozen --group dev --extra stats && uv run python scripts/14_quickstart.py`
prints `QUICKSTART OK` with an exact golden match. Both workflows (`ci`, `verify-release`)
are registered and active.

### The first real CI run failed, and found three more defects

**This is the point of scheduling it.** Every one of these passed locally and failed on a
clean runner, for reasons that could only appear there.

1. **The GPU image pinned the wrong digest — an arm64 per-platform manifest instead of the
   multi-arch index.** `docker buildx imagetools inspect` prints the index digest on its
   `Digest:` line and then lists the per-platform digests underneath; the first of those is
   `linux/arm64`. The amd64 runner pulled an arm64 image and every `RUN` died with
   `exec /bin/sh: exec format error`. The local build had never caught it because this host
   pulls the platform it needs. Correct index digest: `sha256:21196d81...`.
2. **`documented-commands` failed on `make test`, because the verification inherited
   `VIRTUAL_ENV`** from the CI job. uv saw it disagree with the clone's own project
   environment, warned *"does not match the project environment path"* and refused. A
   clean-clone verification that resolves against the caller's environment is verifying the
   wrong thing, so `run()` now strips `VIRTUAL_ENV`, `UV_PROJECT_ENVIRONMENT`, `PYTHONPATH`,
   `PYTHONHOME` and the conda pointers. Invisible locally because a developer shell and a
   CI job export different subsets of those.
3. **The clean clone installed without `--extra stats`,** so `make test` — which this phase
   had just added to the documented-commands list — had no scipy. `check_install` now
   matches what CI and the CPU image do.

4. **`uv sync` prunes the PyG companion wheels.** With the digest fixed, the GPU build got
   as far as its import check and died on
   `ModuleNotFoundError: No module named 'torch_scatter'`. The three companion wheels were
   installed *before* the final `uv sync`, and **`uv sync` removes whatever the lockfile
   does not name** — and those three are deliberately absent from `uv.lock` (D-007: their
   sdists import torch at build time and cannot be resolved at all). So the layer that
   installed them was silently undone by the next layer. They now install **after** the
   last sync, which costs layer-cache reuse on source edits and is worth it.

**The CPU image job passed on the first run**, including its build-time quickstart, so the
image and the packaging were right; the failures were in the GPU image and in this script.
**On the second run the clean-clone and CPU jobs both passed**, leaving only the GPU build
— which had never been built anywhere before this phase, because the GPU path used to be
four commented-out lines inside the CPU Dockerfile.

**Final CI state, on commit `a3217f0`:**

| Workflow / job | Result |
|---|---|
| `ci` — lint, typecheck, full suite, e2e smoke | **success** |
| `verify-release` — clean clone + install + quickstart | **success** |
| `verify-release` — CPU image builds and runs the quickstart | **success** |
| `verify-release` — GPU image builds | **success** |

`ci` had never executed once in this project's history before today, because the repository
had no remote. It went green on its first real run and has stayed green through three
rounds of fixes.

### Acceptance criteria — the Phase 9 gate

| Criterion | Status |
|---|---|
| S1 trains to completion without instability | **deferred** — no GPU |
| A1 trains under identical settings | **deferred** — no GPU |
| **S1 significantly outperforms A1 on faithfulness** | **NOT ANSWERED. Gate 8 is open.** |
| S2 trains and is evaluated | **deferred** |
| B7 trains and is evaluated | **deferred** |
| Attention mass on soft tokens non-trivial and logged | machinery built and tested; **not measured on a real model** |
| Gate value did not collapse to zero | machinery built; gate stays open over a short stub run; **not measured on a real model** |
| Guard implemented; with/without both recorded | **met** — implemented, tested on real fact records, both rows carried |
| VRAM and throughput profiled | machinery built; **no real-model measurement exists** |
| Checkpoints saved with full config and run context | **met** — round-trip tested, regime-guarded |

### What is and is not known

**Known:** the harness is wired correctly — masks, shapes, dtypes, gradient flow,
checkpointing, the guard's selection, the control's derangement. **Not known:** anything
about Llama-3.1-8B's actual behaviour. Whether it reads a soft token, whether the gate stays
open on real data, whether S1 beats A1. A 2-layer randomly-initialised stub cannot answer
those and no test here claims to.

**No number in this entry is a result about the method.** They are all measurements of the
harness.

### Deferred, with what unblocks each

- **All five training arms**, and with them Gate 8. Needs a ≥24 GB GPU. This is now the
  project's critical path.
- **Silver.** Needs teacher-API credentials and spend authorisation. Blocks the curriculum's
  second and third epochs.
- **Gold.** Needs an annotator. Blocks the held-out evaluation.
- **Elliptic2** remains untouched and is still the longest-lead item.
- The `llm` extra is not installed on this machine, so `transformers`, `peft` and
  `bitsandbytes` code paths in `load_base_model` are **unexecuted**. Everything downstream
  of them is tested through the stub.

### Note for whoever runs this on real hardware

Run `make generator-debug` first, then `make train-s1` and `make train-a1`, then
`make gate8 S1=<history_S1.jsonl> A1=<history_A1.jsonl>`. The trainer refuses to start until
the overfit check passes. Watch `faithfulness_gap` in the history from step 0: if it stays
within tolerance for three consecutive evaluations the callback raises `tracking_alarm` and
logs a warning. That is the signal to stop and diagnose rather than to wait — and if it
survives diagnosis, it is the answer to Gate 8, and D-068's pivot is the decision to make.

## Phase 10 — The evaluation harness: three layers, two extractors, and a template that has not yet been beaten

**Date:** 2026-08-05 · **Gate:** **passed with one criterion unmet.** Everything that does
not need a network or a trained model is built, tested and run end to end on the real
15,707-narrative Bronze corpus. The Method A/B agreement κ — the phase's headline
validation number — **has not been measured**, for the same reason Silver has not been
built: no teacher-API credentials. It is listed as deferred with what unblocks it.

### Preflight

Phase 3 — the fact layer and the three-valued checker — is complete and is the only hard
dependency. It was confirmed present and unchanged: `CASE_FACTS_SCHEMA_VERSION` 1.0.0,
`CHECKER_REGISTRY` covering every checkable field, `salience_report`, and the H1–H9
taxonomy. Phase 6 Gold and Phase 9 generations are absent, as the brief anticipated, so the
harness was built and validated against **Bronze**.

### Delivered

**`src/g2t_aml/eval/`** — nine modules, `mypy --strict`, coverage 86–100% per module:

- `types.py` — `SystemOutput`, `ScoredCase`, `pair_outputs_with_facts`. **One join, one
  place to get it wrong.** `load_system_outputs` reads all three narrative shapes this
  repository produces (`target_narrative`, `texts[]`, `narrative`) so a corpus file and a
  Phase 9 generation file score through the same call.
- `layer1_automatic.py` — BLEU-4 with its sacrebleu signature, ROUGE-1/2/L, METEOR,
  BERTScore F1 **always rescaled with baseline**, a `LearnedMetric` protocol for a
  BLEURT/COMET-class metric, a `PerplexityModel` protocol, distinct-1/2, self-BLEU at a
  fixed five references (D-043), and the length distribution against Gold. Plus
  `template_baseline_finding`, which is what the module is really for (D-080).
- `claim_extraction/deterministic.py` — Method A. Composes Phase 5's
  `SlotAlignmentExtractor` rather than reimplementing it (D-048), and adds three passes:
  cue-based **attribution** of residual quantities to fact fields, a **typology** pass, and
  an **invented-citation** pass. Thirty-one cue rules.
- `claim_extraction/llm_based.py` — Method B. Two calls through the Phase 5 `Teacher`
  protocol: atomic decomposition without the facts, then NLI-style entailment against the
  serialised record as premises. Two versioned, hashed prompt files.
- `claim_extraction/agreement.py` — Cohen's κ on verdicts, on token-level claim boundaries,
  and on the per-narrative zero-hallucination decision.
- `layer2_faithfulness.py` — the nine metrics, Zero-Hallucination leading.
- `taxonomy_scorer.py` — H1–H9, Critical Error Rate, the class × typology × system
  cross-tabulation, the stratified hand-label sampler and the validation report.
- `statistics.py` — bootstrap CIs at 10,000 resamples, Wilcoxon signed-rank,
  Holm-Bonferroni, Cliff's δ, Cohen's d, across-seed summaries, publication tables.
- `metric_validation.py` — Spearman and Pearson with Fisher intervals, built now and
  runnable the moment Phase 12's ratings land.
- `report.py` — `evaluate()`, JSON, markdown, LaTeX, per-typology and per-substrate
  breakdowns, balanced/realistic streams kept apart, mechanical worst-case selection.

**`scripts/10_evaluate.py`** (`make eval`, `eval-bronze`, `eval-debug`, `eval-gate`).
Runs on Bronze alone with no arguments, which is the CI gate.

### Bronze scored against Bronze — the gate, on the full corpus

`make eval` over all **15,707** narratives, 41 s, no GPU, no network:

| Metric | Bronze |
|---|---:|
| **Zero-Hallucination Rate** | **1.0000** [1.0000, 1.0000] |
| Fact Precision | 1.0000 |
| Hallucination Rate | 0.0000 |
| Unverifiable Rate | 0.0000 |
| Fact Coverage | 0.8595 [0.8592, 0.8598] |
| Fact F1 | 0.9243 |
| Numeric Accuracy | 1.0000 |
| Typology Accuracy | 1.0000 |
| Ordering Accuracy | 1.0000 |
| Critical Error Rate | 0.0000 |
| Narratives with no claims | **0** |

**296,196 claims, all SUPPORTED.** Phase 4's independent harness counted **296,195** over
the same corpus — agreement to one claim in 296 thousand, from two extractors that reach
the claims by different routes. The `n_narratives_with_no_claims = 0` line is the one that
makes the rest of the table mean something: a perfect score with an empty claim set is what
a broken extractor produces, and it is asserted separately in the gate.

Bronze is faithful *by construction* — it renders from the record and every formatter ships
with its inverse — so this is a regression test on the harness, not a result about Bronze.
`tests/integration/test_eval_end_to_end.py` also corrupts one rendered digit and asserts the
gate fails, because a gate that cannot fail is not a gate (D-040).

### Bronze's Layer 1 scores, and why the quotable finding is not yet available

**The overlap metrics could not be computed: Gold does not exist**, and Layer 1 is scored
against Gold references only. The harness reports each as a named absence with its reason
rather than as a zero. What *is* measurable without a reference:

| | Bronze |
|---|---:|
| distinct-1 | 0.0137 |
| distinct-2 | 0.0352 |
| self-BLEU @ 5 references | 0.4752 |
| length, words | 146.6 ± 11.2 |

All three reproduce Phase 4's independently-computed diversity report **to six decimal
places**, which is the check that Layer 1's plumbing reads the texts it thinks it does.

**The finding the brief wants — whether a deterministic template scores competitively on
ROUGE against Gold — is blocked on Gold, not on this phase.** The machinery is built,
tested and wired: `template_baseline_finding` computes it, flags the metric
`non_discriminative` below a 0.02 ROUGE-L margin fixed before any system was scored, and
emits a one-sentence headline the paper can quote. It runs the day one Gold narrative
exists.

### The Phase 10 finding that *is* available: Bronze omits exculpatory facts in 92% of narratives

| Class | Bronze, per-narrative rate |
|---|---:|
| H1–H8 | 0.0000 |
| **H9 — omission of exculpatory fact** | **0.9179** |

H9 is the only class detected by absence rather than assertion, and it is the only one
Bronze can trigger. The templates report `labels.n_counterparties` and
`labels.n_illicit_counterparties` but **never** the licit count, **never**
`labels.focal_is_illicit`, and mention `temporal.burst_detected` only when a burst was
detected. So a case whose subject account carries no illicit label, or whose counterparties
are mostly licit, produces a narrative that never says so.

This is a real deficiency of the Bronze corpus, found by the harness rather than by
inspection, and it has two consequences worth carrying forward. It is a **concrete way a
trained system can beat the template on a faithfulness dimension** — the floor is not
uniform at 1.0. And it is a caution for Phase 12: an investigator shown a narrative that
omits the exculpatory context is being shown a one-sided case, which is exactly what the
decision-setting study is meant to detect.

Fact Coverage varies by typology and the spread is not noise:

| Typology | Coverage |
|---|---:|
| unclassified | 0.862 |
| random | 0.848 |
| fan_in / fan_out | 0.843 / 0.841 |
| gather_scatter | 0.834 |
| scatter_gather / stack | 0.824 |
| cycle | 0.784 |
| bipartite | 0.770 |

The structurally richer typologies have longer salience lists and Bronze's templates do not
grow to match them. **bipartite and cycle are where a generator has the most room.**

### Two bugs the tests found, both in the direction that would have understated hallucination

Both were found by writing one deliberate narrative per hallucination class and asserting
each is reachable — a test that exists precisely because a class that never fires reads,
in a results table, as a system that never makes that error.

1. **A wrong quantity scored UNVERIFIABLE rather than CONTRADICTED.** Phase 5's extractor
   emits an unaligned number as a claim naming no field, which is right for a repair loop
   where both verdicts have the same consequence. Under Phase 10's metrics, "the subject
   received from 14 distinct counterparties" against a record saying 9 would have scored
   **Hallucination Rate 0.000 and Zero-Hallucination 1.000** — the most damaging thing a SAR
   narrative can do, filed under the bucket for things the graph cannot speak to. Fixed by
   the attribution pass (D-076).
2. **H6 was unreachable.** Phase 5 matches only *whitelisted* citations, deliberately, so it
   can never launder an invented one — but nothing covered the complement, and "the USD
   42,000 mandatory disclosure threshold" produced one stray number and no regulatory claim.
   The Critical class the paper leans on hardest could not be assigned. Fixed by matching
   citation *shape* and letting `check_regulatory` adjudicate against the whitelist: a
   forbidden-phrase list cannot contain the rule a model has not invented yet.

Neither would have failed anything. Both would have made every system look better.

### Tests

**172 new tests**, all passing; the full suite is **1,804 passed, 2 skipped** (the skips
are Elliptic2's access-gated data, pre-existing). Coverage on `eval/`: agreement 98%, layer2 99%, metric_validation
96%, taxonomy 93%, types 93%, statistics 92%, deterministic 91%, report 88%, llm_based 87%,
layer1 86% — **every module above the 85% gate**. `mypy --strict` clean over `facts/` and
`eval/`; `ruff` clean.

- Every metric against hand-computed fixtures; Holm-Bonferroni against
  `statsmodels.multipletests` on a hand-written family and on random families of size 2, 5,
  17 and 120; Wilcoxon against `scipy`; bootstrap coverage checked by replication on a
  known Bernoulli.
- Both extractors on the four adversarial shapes the brief names — hedged, vague, partially
  correct, and correct-but-unstated-in-the-facts.
- One deliberate narrative per class H1–H9, plus a test asserting the nine covered classes
  are exactly the nine in the enum.
- Method B end to end through `ScriptedTeacher`: no network, no key, and the real two-call
  pipeline rather than a mock beside it.

### Deferred, with what unblocks each

- **The Method A/B agreement κ on 300 cases.** Needs teacher-API credentials and spend
  authorisation — the same blocker as Silver. Method B, its two hashed prompts, the κ
  machinery and the whole path are built and tested on scripted responses;
  `eval.agreement.enabled` is `false` so CI runs without either. **This is the phase's one
  unmet acceptance criterion**, and it is the number that makes the faithfulness metric
  defensible rather than merely defined.
- **The 200 hand-labelled errors validating the taxonomy classifier.** The stratified
  sampler and `validate_against_hand_labels` are built and tested; scoring Bronze produces
  only H9 findings, so there is no interesting error pool to label until a model arm exists.
  Draw the sample from the first real generation run.
- **Bronze's Layer 1 scores and the template-baseline finding.** Blocked on Gold.
- **The correlation with human ratings.** Blocked on Phase 12, by design.
- **The realistic-imbalance stream.** The harness keeps it separate from the balanced set
  and refuses to pool them; no realistic-stream file has been produced yet.
- **BERTScore, METEOR and a learned metric** are untested against real weights on this
  machine. Each degrades to a named absence rather than a number.

## Phase 11 — The experiment matrix: declared, orchestrated, aggregated, and not run

**Date:** 2026-08-05 · **Gate:** **the orchestration is complete and tested; no system was
run.** The preflight failed on the same three conditions Phase 9 failed on, none of them
recoverable by writing code. **Gate 8 remains open.** Every acceptance criterion that
depends on a run is **deferred, not met**, and is listed as such below.

### Preflight, and why the session's mission could not be carried out

The brief asked to confirm Phase 7 (encoder arms), Phase 8 (fusion variants), Phase 9
(S1/A1/S2/B7 trained **and the S1-vs-A1 outcome**) and Phase 10 (evaluation harness), then
run the matrix.

| Condition | Required | Actual |
|---|---|---|
| Phase 7 — encoder | trained | **met.** `gatv2_seed42.pt`, test AUC-PR 0.8720 ± 0.0136; MLP control also available for A2. |
| Phase 8 — fusion variants | built | **met.** `PrefixFusion` (F1/F2 by one flag), three projectors, `ShuffledGraphFusion` for A1. |
| Phase 9 — S1/A1/S2/B7 trained | trained | **NOT MET. Zero arms trained** (D-068). |
| Phase 9 — the S1-vs-A1 outcome | recorded | **DOES NOT EXIST.** Gate 8 is open, so there is no outcome to confirm and no reframing decision to check: neither arm has run. |
| Phase 10 — evaluation harness | complete | **met**, with one criterion unmet (the Method A/B κ). |

**The blockers are unchanged from Phase 9 and none is a software problem.** A 4 GB RTX 2050
against a model needing ≥24 GB; zero Silver for want of API credentials; zero Gold for want
of an annotator. See D-068 and RESULTS.md §2.1.

**What was therefore built instead:** everything the matrix run depends on, verified on CPU
against fixtures and against the real 15,707-narrative Bronze corpus.

### Delivered

**`src/g2t_aml/experiments/`** — five modules:

- `registry.py` — **the matrix, declared once.** 17 systems, 25 runs. Every axis the matrix
  varies is a typed field: encoder arm, fusion variant, text mode, base model **and its
  release date**, training config, guard, seeds, dependencies, resource class.
  `validate_registry` returns every problem rather than raising on the first, and refuses a
  stale baseline, a central-claim system at the wrong seed count, an encoder wired to no
  fusion, or an A1 that differs from S1 on any axis but the shuffle flag.
- `baselines.py` — B1–B6, built as competitors. B5's generate → self-verify → repair loop,
  B4's deterministic train-split-only exemplar selection, and
  `assert_baseline_not_starved`, which refuses to send a baseline prompt whose guidance
  blocks render empty.
- `runner.py` — dependency resolution, resource scheduling, resumption on marker **and**
  config hash, failure isolation. Takes its executors as an argument, so the whole
  scheduler is exercised on CPU in milliseconds.
- `aggregate.py` — the tidy long-format table (system × seed × metric × substrate × stream
  × typology), across-seed summaries, bootstrap CIs, Wilcoxon with Holm–Bonferroni over
  each metric's own family, and the main / ablation / taxonomy tables as LaTeX.
- `figures.py` — six figures, Okabe-Ito palette, vector PDF, TrueType embedding.
- `executors.py` — the wiring from executor kind to implementation. The GPU arms go through
  the existing `scripts/09_train_generator.py` rather than reassembling the model, so
  there is one place where the three learning rates, the loss mask and the fp32-projector
  assertion live.

**Prompts:** three versioned, hashed files — `baseline_generate_v1`, `baseline_verify_v1`,
`baseline_repair_v1`. The generate prompt's header states, in the file, that B3–B5 receive
every instruction our own arms receive, and names what would constitute weakening it.

**Configs:** twelve new arm configs (`matrix_b1`…`matrix_b6`, `matrix_a2`, `matrix_a3_f3`,
`matrix_a3_f4`, `matrix_a4`, `matrix_a5`, `matrix_a6`), all `# @package _global_` with real
config paths — never a nested `overrides:` block, which is inert and is what nearly turned
A1 into a second copy of S1 in Phase 9.

**Scripts and targets:** `11_run_matrix.py`, `11a_aggregate.py`, `11b_qualitative.py`;
`make matrix-plan`, `matrix`, `matrix-cpu`, `aggregate`, `qualitative`, `matrix-gate`.

**`RESULTS.md`** — committed, with every measured number the project has and all 25
declared runs as named absences carrying their blockers.

### The seed policy, and where it is enforced

**3 seeds (42, 1337, 2024) on S1, S2, A1, B7; 1 seed (42) on the other thirteen.** Stated
in the paper, not inferred from a table. Extension order if compute frees up: A2, then B8.
D-081.

It is enforced in three places rather than remembered: `validate_registry` refuses a
central-claim system at fewer than three seeds *and* a non-central system at more than one;
`_mean_std_cell` renders a single-seed row as a bare mean with a dagger and can never print
`± 0.0000`; and the bar chart draws a hollow marker over every single-seed bar.

### Test results (this machine, 2026-08-05)

| Suite | Result |
|---|---|
| `tests/unit/test_experiment_registry.py` | 28 passed |
| `tests/unit/test_experiment_runner.py` | 22 passed |
| `tests/unit/test_experiment_aggregate.py` | 22 passed |
| `tests/unit/test_experiment_baselines.py` | 31 passed |
| `tests/unit/test_experiment_figures.py` | 21 passed |
| `tests/integration/test_matrix_pipeline.py` | 13 passed |
| **Full suite** | **1,940 passed, 2 skipped** (the two Elliptic2 skips) |
| `ruff check` / `format --check` | clean |
| `mypy --strict` (facts + eval) | unchanged, clean |

The brief's named testing requirements, each asserted rather than inspected: the runner
resumes correctly after interruption, and only the failed run is re-attempted; a config
change invalidates a completion marker while a prose edit does not; aggregation of an empty
matrix succeeds and reports all 25 runs missing by name; the statistical pipeline detects a
planted effect (p < 0.05 after Holm, Cliff's δ = 1.0) **and reports no effect as
non-significant** on data with none; every figure renders from a fixture metrics file and
from an empty one.

The whole reporting path was also run against the real corpus: `make aggregate` produces
the three LaTeX tables, the tidy CSV and all six figures on an empty matrix, and
`make qualitative` selects its ten stratified cases from the real 3,192-case test split.

### Five things found by building it, not by planning it

1. **A few-shot exemplar rotation applied across the whole candidate list can rotate past
   the matching typology entirely.** `select_exemplars` sorted matching-typology exemplars
   first and *then* rotated the concatenation by a hash of the case id — so for some case
   ids the window landed wholly in the non-matching group. Typology-matched few-shot would
   have silently become arbitrary few-shot on a subset of cases, and B4 and B5 would both
   have been weakened, non-uniformly, in a way no output inspection would reveal. Caught by
   a test written from the docstring's claim rather than from the code. The rotation is now
   per-group.
2. **The qualitative case selection was not reproducible.** It ordered candidates by
   Python's `hash()`, which is salted per process unless `PYTHONHASHSEED` is pinned — so
   the "ten cases fixed before the numbers exist" would have been ten different cases on
   every invocation, which is exactly the property that rule exists to have. Now a stable
   digest; verified identical under two hash seeds.
3. **The main comparison chart crashed on its own error bars.** The bar height is the
   across-seed mean and the interval is the per-case bootstrap; they are computed from
   different data and the mean can fall marginally outside its own interval, which
   matplotlib rejects as a negative `yerr`. The fix is a clamp with a comment, not a
   reconciliation: making the two agree would mean computing one of them wrongly.
4. **A5 is not the arm its name suggests.** `generator.freeze_encoder` defaults to **true**,
   so S1 is the frozen configuration and the joint-training ablation is the arm that
   *unfreezes*. Written the other way round it would have produced an ablation table row
   labelled "frozen encoder" whose config was identical to S1's, and the row would have
   read as a null result about joint training. The registry labels rows by what they vary.
5. **B8 already is A3's F1 arm.** Same gate flag, same projector, same text mode, same base
   model, same regime. Discovered while writing the fusion-ablation configs; running it
   twice would have spent GPU-days to produce a duplicate row (D-082).

### Acceptance criteria — the Phase 11 gate

| Criterion | Status |
|---|---|
| All 16 systems configured and run (or documented non-runs with reasons) | **partially met.** All 17 systems (A3 expands into two) are configured, validated and planned; **none has run**, and every one is documented in RESULTS.md §2 with its blocker. |
| 3 seeds on S1, S2, A1, B7 | **deferred** — policy adopted, enforced and stated; no arm trained |
| Baselines use current 2025–2026 models with versions recorded | **met** — recorded as registry data with release dates, validated (D-083) |
| B5 implemented as a genuine competitor, not a strawman | **met** — real self-verification, three repair rounds, few-shot start, whitelist supplied, call count reported (D-084) |
| All metrics aggregated with CIs and corrected significance tests | **machinery met; no metrics to aggregate.** Verified on synthetic data with known effects. |
| Main, ablation and taxonomy tables generated in LaTeX | **met** — all three generate; every cell is currently an em dash |
| All figures generated | **met** — all six render, each carrying its stated absence |
| `RESULTS.md` committed with every number including nulls | **met** |
| Qualitative analysis materials produced | **partially met** — selection rules, fact records and rendering paths run on the real test split; no narratives exist to compare |

### What is and is not known

**Known:** the matrix is internally consistent — every arm's Hydra composition agrees with
what the registry claims about it, every declared system can be dispatched, the runner
resumes and isolates failures correctly, the statistics are right on data with a known
effect and correctly silent on data without one, and every figure renders in both the
populated and the empty state.

**Not known:** anything about any system's behaviour. **No number in this entry is a result
about the method.** The GPU and API executors have never been executed against a real model
or a real endpoint, and their first run will find integration bugs the CPU path did not.

### Deferred, with what unblocks each

- **Every run in the matrix**, and with it Gate 8. Needs a ≥24 GB GPU for twelve systems and
  API credentials for three. **B1 and B2 need neither and are runnable today** — see
  RESULTS.md §8.
- **Silver**, which is two of three curriculum epochs. Needs credentials and spend
  authorisation.
- **Gold**, and with it every Layer 1 metric and the template-baseline finding. Needs an
  annotator. Still the critical path, and still not advanced by any engineering.
- **The Method A/B agreement κ.** Same blocker as Silver.
- **Phase 13's latency and footprint numbers**, which the efficiency frontier consumes.
- **Elliptic2.** Access still not requested; longest-lead item in the project.

### Note for whoever runs this on real hardware

`make matrix-plan` first — it prints the plan, the order, the resource split and the seed
policy without touching anything. Then `make matrix-cpu` for B1 and B2, which needs no GPU
and no network and puts two real rows in the table. Then, with compute:
`make matrix ALLOW_GPU=1 SYSTEMS=S1,A1`, then `make gate8`. Watch `faithfulness_gap` in the
history from step 0; if it stays within tolerance for three consecutive evaluations the
callback raises `tracking_alarm`. That is the signal to stop and diagnose — and if it
survives diagnosis, it is the answer to Gate 8, and D-068's pivot is the decision to make.

**Do not run the full matrix before S1 and A1 have answered Gate 8.** If the fusion layer
turns out to be decoration, the other twenty-three runs are answering a question the paper
will no longer be asking.

---

## Phase 12 — The decision-setting study: built, validated, and blocked at the door
**Date:** 2026-08-05 · **Gate:** **the kit is complete and tested end to end against
simulated responses. No human has rated anything.** Three independent blockers stand
between this machinery and a result, and the first of them is the one that matters:
**ethics approval has not been granted, and had not been applied for.** Every acceptance
criterion that depends on a person is **deferred, not met**, and is listed as such below.

**Preflight — and it failed on all three counts.**

Phase 6 confirmed complete: the Gold annotation kit, the guidelines, the 350-case reserved
sample. The brief also asked me to confirm generations from S1, B7 and Bronze exist.

| Required | Found |
|---|---|
| Bronze generations | **Yes** — 15,707 narratives |
| S1 generations | **No.** Phase 9 never ran; 4 GB card against a model needing ≥24 GB (D-068) |
| B7 generations | **No.** Phase 11 never ran; needs API credentials |
| Ethics approval | **No.** `grep -rni "irb\|ethics\|ethical"` over every markdown file in the repository returned **zero matches**. Not approved, not submitted, not drafted |
| Raters | **None.** Phase 6's annotator recruitment has not produced a person either |

**One of five arms exists.** A five-system comparison has one system. The study could not
have run this week even with approval in hand, and `scripts/12_build_study.py` refuses to
build a design over fewer than two arms rather than producing one that would waste a
panel's time — verified on this machine, exit code 2.

**Delivered**

- **`docs/human_study/`** — seven documents, the governance half of the phase:
  `ethics_application.md` (ready to submit; institutional fields deliberately left blank
  rather than filled with placeholders that could be mistaken for real values),
  `participant_information.md`, `consent_form.md` (participation and publication consent
  **separated**, because bundling them makes the first questionable),
  `data_management_plan.md`, `compensation.md`, `fallback_governance.md`, and a README
  stating the blockers first.
- **`docs/human_study/rater_training.md`** — the 30-minute pack. SAR basics, the
  suspicion-vs-guilt rule with worked wrong/right pairs, the four-part structure, the eight
  typologies with the fan-in/fan-out caution, **all five scales with 1/4/7 anchors written
  out**, and three calibration items with feedback. The three items are chosen to teach the
  three specific errors this study exists to detect: a missing exculpatory fact, correct
  numbers carrying unsupported claims, and the halo effect of good prose.
- **`src/g2t_aml/human/study_design.py`** — the balanced incomplete block design, the
  cyclic-Latin-square position balancing, the anchor block, the planted repeats, the opaque
  keyed item ids, and a validator that enforces six constraints rather than hoping.
- **`src/g2t_aml/human/study_ui.py`** — the blinded rating interface, the server-side
  `BlurAwareTimer`, the response store with save-and-resume, and `assert_no_system_identity`.
- **`src/g2t_aml/human/study_timer_component/`** — a Streamlit component in one static HTML
  file, talking the postMessage protocol directly so there is no build step and no node in
  the lockfile. It is the only thing that can see `visibilitychange`.
- **`src/g2t_aml/human/study_analysis.py`** — ordinal Krippendorff with bootstrap CIs,
  intra-rater reliability, Friedman **and Durbin**, Nemenyi with a critical-difference
  diagram, paired tests against the Bronze baseline, Spearman/Pearson with Fisher-z
  intervals, and a mixed-effects rater model. Pure Python plus numpy: it runs in the base
  environment, because scipy and statsmodels are in the `eval` extra.
- **`src/g2t_aml/human/study_release.py`** — the anonymised deposit.
- **Three scripts** (`12_build_study.py`, `12b_analyse_study.py`, `12c_release_study.py`),
  five `make` targets (`study-build`, `study-rate`, `study-analyse`, `study-release`,
  `study-gate`), and `notebooks/12_human_study.ipynb`.
- **101 Phase 12 tests**, all green.

**Every statistic validated against something outside this repository**

Invariant 1's rule applied to Phase 12: a second implementation by the same author on the
same afternoon agrees with the first on their shared misreading.

| Statistic | Checked against | Result |
|---|---|---|
| Krippendorff ordinal α | Krippendorff (2011) four-observer worked example, **published 0.815** | **0.8154** |
| Krippendorff ordinal α | `krippendorff` 0.8.0, independent package | agrees to 1e-9 |
| Friedman χ² | `scipy.stats.friedmanchisquare` | exact, at k = 3, 4, 5, 6 |
| Friedman tie correction | same, on a deliberately tied matrix | exact (24.000) |
| Durbin | reduction to Friedman on a complete matrix | exact |
| Nemenyi CD | Demšar (2006) Table 5 formula | exact |
| Spearman, Pearson | `scipy.stats`, including Pearson's CI | exact to 1e-6 |
| Normalised Levenshtein | kitten→sitting = 3/7 | exact |

**Three real defects, found by running the thing rather than by reading it**

1. **The design produced zero doubly-rated cells, so Krippendorff's α — a Phase 12 gate
   requirement — could not be computed at all.** The greedy allocator spreads cases as
   thinly as possible over the case × system grid, which is exactly right for maximising
   breadth and exactly wrong for agreement: α is computed over units two or more raters
   judged, and there were none. Found by running the analysis against a simulated response
   set and reading its own warning. Fixed with an **anchor block** every rater rates
   (D-088). The regression test is `test_the_anchor_block_makes_agreement_computable`.
2. **`_plant_repeats` did not guarantee the separation its own validator enforced.** On a
   19-item sequence the source pool reached index 5 and the insertion point landed at 15 —
   a separation of 10 against a bound of 15. The validator caught it, which is the argument
   for having written the validator.
3. **Per-rater workloads drifted to a spread of two** (13/13/12/12/11) because a shared
   `ceil(total/k)` quota lets the first systems considered fill up and pushes the shortfall
   onto the last. A rater effect on a skewed workload is indistinguishable from a system
   effect. Replaced with exact per-system targets that sum to the workload, rotated by
   rater index so the remainder is not always the same arm's.

**Two design decisions the brief did not specify, and both were forced**

- **Friedman's test needs complete blocks and this design has none** (D-089). Because no
  rater sees a case twice, no case carries one observation per system. Friedman is run on
  **rater-blocked means** — the brief's test, and underpowered at 8–10 blocks — and
  **Durbin's test**, its generalisation to a balanced incomplete block design, is run on the
  case blocks and reported beside it. Reporting only the first throws away most of the data;
  only the second answers a question the brief did not ask.
- **The anchor block costs breadth** (D-088). Fifteen cells per rater buy the agreement
  statistic and contribute nothing to the between-system comparison.

**Study parameterisation, chosen by measurement rather than assumption**

The Phase 12 gate is stated as "≥80 cases × ≥4 systems", and that is a claim about
**cases-per-system**, not about the size of the stimulus pool — an incomplete design leaves
cells empty by construction. Measured across candidate configurations at 100 cases and
5 systems:

| Panel | Items/rater | Ratings | Min cases/system | Meets gate |
|---|---|---|---|---|
| 6 | 64 | 384 | 58 | no |
| 8 | 53 | 424 | 59 | no |
| 8 | 64 | 512 | 76 | no |
| 10 | 53 | 530 | 73 | no |
| **10** | **60** | **600** | **85** | **yes** |
| 12 | 56 | 672 | 88 | yes |

**Recommended: 10 raters × 60 items = 600 ratings**, 15 anchor cells, 3 repeats each,
per-rater workload balanced to within one item, mean-position spread across systems 0.57.
Roughly 4–6 hours per rater. `scripts/12_build_study.py` warns when a configuration falls
under the gate.

**Gate verification** (this machine, 2026-08-05)

| Criterion | Required | Achieved |
|---|---|---|
| Phase 12 tests | — | **101 passed** |
| `ruff check` / `format --check` | clean | **clean**, 245 files |
| `mypy --strict` over `facts/` + `eval/` | unchanged | **unchanged**, 27 files |
| Interface renders both substrates | required | **verified** — AMLworld and Elliptic2 fixtures; Elliptic2 asserted to carry no Value section and no currency word |
| Timer accuracy including blur pause | required | **verified** — 11 tests: single and multiple hidden periods, reading while hidden, stop-while-hidden, idempotent start under Streamlit reruns, double-blur |
| Edit diff capture | required | **verified** — both versions stored; distance validated against kitten→sitting |
| Design valid, no duplicate system-case pair | required | **verified** — 26 tests, including that no rater sees a case twice at all |
| Blinding: no system identity in any rendered payload | required | **verified** — asserted over the serialised payload for every registry system id, on word boundaries |
| Krippendorff α vs published example | required | **verified** — 0.8154 vs 0.815 |
| Friedman/Nemenyi vs reference | required | **verified** — exact against scipy |
| End-to-end on simulated responses | — | **verified** — recovers the simulated ordering, α = 0.487 [0.229, 0.638], Friedman p < 1e-5, Durbin on 77 of 100 case-blocks, Spearman 0.445 [0.377, 0.509] |
| **Ethics approved** | required | **NO — not applied for** |
| **≥6 raters recruited and trained** | required | **NO — none recruited** |
| **Study run: ≥80 cases × ≥4 systems** | required | **NO — one arm exists** |
| **α reported on real data** | required | **NO** |
| **Automatic-vs-human correlation on real data** | required | **NO** |
| Anonymised release path | required | **built and tested; nothing to release** |

**Deferred, not met:** ethics approval, recruitment, training delivery, the study itself,
and every number that would come out of it. The analysis pipeline emits a warning naming
each missing input rather than omitting the section — a study with no automatic scores
reports "the Phase 12 gate requires this number", and a dimension with no doubly-rated cell
reports that a Likert mean must not be published without one.

**The trigger date is 2026-09-15.** If the ethics application has not been **submitted** by
then, the external study cannot complete before the paper's target date and the project
switches to the internal expert review in `docs/human_study/fallback_governance.md`, with
its four limitations stated in the paper's own words. That is a materially weaker
instrument and it is chosen over nothing, not over a real study.

**What is untested:** the Streamlit shell (`main`) and the browser timer component have
never been exercised in a real browser, because there are no stimuli to render. Their first
real session will find integration bugs the CPU path did not — the same honest caveat
Phase 11 recorded about its GPU and API executors.

---

## Phase 13 — Efficiency: the instrument built, two systems measured end to end, thirteen absences named

**Date:** 2026-08-05 · **Gate:** **the instrument and the protocol are complete, tested and
run; 2 of 17 systems are measured.** The preflight failed for the same reason Phase 11's
did and it is not recoverable by writing code. Every acceptance criterion that does not
depend on a trained checkpoint is **met**; every one that does is **deferred with a named
blocker**, not silently skipped.

### Preflight, and what it found

The brief asked to confirm Phase 11: all systems trained and evaluated, checkpoints
available.

| Condition | Required | Actual |
|---|---|---|
| All systems trained | 17 | **0.** `artifacts/checkpoints/` holds Phase 7 encoder arms only |
| All systems evaluated | 25 runs | **0.** `aggregate.json` → `n_rows: 0`, `systems_present: []` |
| Checkpoints available | fusion + generator | **encoder only.** Gate 8 remains open |

This is the documented state (D-087), not a regression. The hardware is the binding
constraint and it is hard: a 4 GB RTX 2050 with 7 GB of system RAM, against 4.5–5.6 GB of
nf4 weights before a single activation, with CPU offload closed by the RAM (D-068).

The session proceeded on the house rule the project has used twice already: build the
instrument in full, measure everything measurable, and write every absence with its reason
(invariant 7). D-097 records the decision.

### What was built

`src/g2t_aml/eval/efficiency.py` — the instrument, under invariant 1's `mypy --strict`
scope alongside `facts/` and the rest of `eval/`. It imports no torch at module scope: the
accelerator paths are inside the two functions that need them, so the module stays
importable on the aggregation and documentation hosts. Types for the protocol (`Stage`,
`NodeBin`, `LatencySummary`, `BenchmarkSample`, `EndToEndTimer`), the footprint
(`ModelFootprint`, `MemoryProfile`), the cost model (`CostAssumptions`, `CostEstimate`),
the assessment (`DeploymentProfile`) and the table (`SystemEfficiency`, `EfficiencyTable`)
with its three LaTeX writers.

`scripts/13_benchmark.py` — the protocol: 20 discarded, 100 measured, two draws, both batch
sizes, hardware read from the machine. `make benchmark`, `make benchmark-quick`,
`make benchmark-gate`.

`docs/deployability.md` — the per-system assessment, complete for all 17 rows on the one
axis that does not need a GPU.

`tests/unit/test_eval_efficiency.py` — 52 tests. Four of them exist because the failure is
silent: a zero must not print as an absence, percentiles must be nearest-rank, an
unmeasured row must carry its blocker, and the warm-up must actually be discarded.

### The hardware, recorded exactly

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 2050, 4 GB |
| Driver / CUDA | 595.84 / 12.1 |
| torch | 2.4.0+cu121 |
| CPU | 12th Gen Intel Core i5-12450H, 12 logical cores |
| RAM | 7 GB |
| OS / Python | Linux 7.0.0-28-generic / 3.11.14 |

**It is a laptop, it throttles, and it is sensitive to whatever else is running.**
Run-to-run p50 for B1 varied **5.4–11.2 ms across six full protocol runs** on one
afternoon — a 2x between-run spread, larger than the difference the size bands resolve.
One run taken while the test suite was executing concurrently showed p95 inflated to 30 ms
and index build inflated 3x, and was discarded and rerun on an idle machine. The table
below is the final idle-machine run; the between-run range is reported rather than averaged
away, and **no conclusion in this phase rests on a difference smaller than 2x.**

### The efficiency table

Protocol: n=100 measured, 20 discarded, seeded shuffle (seed 13) of the frozen test split,
3,196 cases available, nearest-rank percentiles.

| | B1 (template) | B2 (GATv2 + template) |
|---|---:|---:|
| On-premise | ✓ | ✓ |
| Data leaving perimeter | nothing | nothing |
| Total / trainable params | 0 / 0 | 628,058 / 628,058 |
| Model size on disk | 0 | 2.53 MB (encoder ckpt) |
| Peak VRAM, inference (reserved) | n/a | **0.025 GB** |
| Peak VRAM, encoder training (reserved) | n/a | 0.055 GB @ batch 32 |
| Peak host RAM | 3.08 GB | 3.08 GB |
| Latency p50 | **5.4 ms** | **15.8 ms** |
| Latency p95 | 22.1 ms | 24.5 ms |
| Latency p99 | 31.5 ms | 28.9 ms |
| Latency max | 61.4 ms | 39.1 ms |
| Latency mean | 8.1 ms | 17.0 ms |
| Throughput, batch 1 | 123.9 narr/s | 58.9 narr/s |
| Throughput, 32-case queue | 206.7 narr/s | — |
| Cold start | 0.92 s (graph) | 7.65 s (graph+index+encoder) |
| Cost / 1,000 (amortised local) | USD 0.00038 | USD 0.00080 |

Stage breakdown, mean ms per narrative:

| Stage | B1 | B2 |
|---|---:|---:|
| Case extraction | **5.82 (72%)** | 2.86 (17%) |
| Fact extraction | 0.72 (9%) | 0.74 (4%) |
| Serialisation | 0.09 (1%) | 0.10 (1%) |
| Encoding | — | **11.76 (69%)** |
| Generation (template render) | 1.44 (18%) | 1.52 (9%) |
| Guard verification (4 cand.) | 1.90 | — |

API baselines, cost **estimated** from published pricing and measured token counts
(569 prompt / 188 completion tokens per narrative): B3, B4 → USD 22.65 per 1,000;
B5 → USD 67.94 per 1,000 at three calls per narrative (D-084).

### Latency by case size, 40 runs per band, size-stratified draw

| Band (nodes) | B1 p50 / p95 | B2 p50 / p95 |
|---|---:|---:|
| 0–24 | 3.3 / 5.0 ms | 18.5 / 30.2 ms |
| 25–49 | 8.9 / 16.1 ms | 21.7 / 32.2 ms |
| 50–99 | 7.6 / 19.5 ms | 25.9 / 36.8 ms |
| 100+ | **no case in this corpus** | **no case in this corpus** |

Roughly 2x from the smallest band to the largest, monotone in p95 for both systems and
monotone in p50 for B2. **B1's p50 is not monotone** — the 50–99 band reads 7.6 ms against
the 25–49 band's 8.9 ms — and at 40 runs per band with a 2x between-run spread that
inversion is noise, not a finding. B1's p95 *is* monotone, which is the ordering to trust
here. The
empty top band is a fact about the corpus, not the benchmark: the Phase 2 node budget is
150 but no test-split case reaches 100 nodes, so the 50–99 row is the top of the measured
range and the trend must not be extrapolated past it.

### Four findings

1. **Case extraction is 72% of B1's end-to-end latency; generation is 18%.** Cutting a
   subgraph out of a 5-million-edge graph costs more than rendering the narrative does.
   Every row of the matrix pays this, including the 8B arms — it does not shrink when the
   generator grows, it just stops dominating. This is the finding that generation-only
   timing would have hidden, and it is the argument for D-095.
2. **The graph half of the architecture needs 25 MB of device memory.** B2's whole
   inference footprint is 0.025 GB reserved, and the encoder checkpoint is 2.5 MB. Whatever
   makes this architecture expensive to deploy, it is not the graph encoder.
3. **The encoder's 11.8 ms is launch overhead, not arithmetic.** 628 K parameters over a
   median 12-node graph on a 4 GB card: most of it is kernel launch and host-device
   transfer. A batched forward would amortise nearly all of it. Measured as deployed — one
   case at a time, the interactive path — and the batched path is not measured.
4. **The guard's verification pass is 1.90 ms for four candidates**, about 0.5 ms each, and
   it is model-independent, which is why it was measurable at all here. This is the
   verification half of the guard's cost. The four generations it requests are excluded,
   and on an 8B decoder they will dominate completely. Every artifact that prints a guard
   overhead ratio for B1 says so in the same breath. The ratio is taken from the **stage
   mean** (1.90 ms against B1's 8.1 ms mean) and not from differencing two runs' p50s,
   because the between-run spread is larger than the effect.

### Acceptance criteria

| Criterion | Status |
|---|---|
| All metrics measured for every system | **2 of 17 measured**, 15 blocked with named reasons (D-097) |
| ≥100 runs per system, distributions reported | **Met** for both measured systems; p50/p95/p99/max/σ published, raw samples written |
| Hardware configuration fully documented | **Met** — read from the machine, embedded in every table caption |
| End-to-end latency, not generation-only | **Met** — six stages timed, total is the sum of the breakdown (D-095) |
| Guard-on and guard-off both measured | **Met for verification**, which is the model-independent half; the generation half is unmeasurable here and is stated as such |
| Latency binned by case size | **Met** — 40 runs per band, stratified draw (D-094) |
| Deployability assessment written | **Met** — all 17 rows, complete on the on-premise axis |
| Efficiency frontier figure produced | **Met** — one honest point (B1); title states "16 of 17 systems unmeasured" |
| Table exported to LaTeX | **Met** — three tables: main, guard cost, latency by size |

### Deferred, with blockers

- Every metric for B6–B8, S1, S2, A1–A6: no trained checkpoint (Phase 11 unrun, Gate 8
  open) **and** no accelerator that fits the model.
- Measured latency and throughput for B3–B5: no API credentials, the same blocker as
  Silver and as Method B of the claim extractors.
- Training cost in GPU-hours for any 8B arm: nothing has trained.
- The batched (batch-32) forward pass for a real decoder: the template path has no batch
  dimension, so what was measured is queue throughput, and it is labelled that way.
- The 100+ node band: unpopulated in this corpus.

All of it is recoverable by rerunning `make benchmark` on a machine with a 24 GB card once
Phase 11 has run. Nothing built this session needs rebuilding for that to work.

---

## Phase 14 — Release: the artifact a stranger can clone, and what it honestly contains

**Date:** 2026-08-06 · **Gate:** **the repository, the documentation, the verification and
the licence separation are complete and tested; the clean-clone quickstart is verified in a
fresh container. Two acceptance criteria are NOT met and are recorded as such below: the
Zenodo DOI is not minted, and no HuggingFace artifact is published.** Every criterion that
depends on a trained model is **deferred with a named blocker**, because no trained model
exists.

### Preflight, and a correction to the brief

The brief asked to confirm Phases 11, 12 and 13 complete. **They are not, and the brief's
premise is wrong in a way worth stating rather than working around.**

| Condition | Brief assumed | Actual |
|---|---|---|
| Phase 11 — the matrix | complete | **Declared, orchestrated, aggregated and tested. 0 of 25 runs executed.** |
| Phase 12 — the study | complete | **Built and validated against simulated responses. No human has rated anything; ethics approval not submitted.** |
| Phase 13 — efficiency | complete | **Instrument complete and run. 2 of 17 systems measured.** |

This is the documented state (D-087, D-097), not a regression. The session proceeded on the
house rule the project has now used four times: build the instrument in full, package
everything that exists, and write every absence with its reason (invariant 7).

**The Phase 1 licence findings were re-read first, as the brief required, and one of them
had not been discharged.** The Elliptic2 card recorded that its data licence *"could not be
located"* and that the question *"must be resolved before Phase 14"*. It was not resolved,
because access was never requested — see D-099.

### Repository hygiene — audited, and it was in better shape than expected

| Check | Result |
|---|---|
| **gitleaks over the full git history** | **clean** — 2 commits, 84 objects, 0 findings |
| **gitleaks over the release tree** | **clean** — 4 findings, all the literal string `heuristic-bpe-v1`, allowlisted with the reason in `.gitleaks.toml` |
| Email addresses, home paths, IPs, hostnames | **none** |
| Data or artifacts ever committed | **none** — history contains only `.gitkeep` markers |
| Notebook stored outputs | **0 of 23 cells** |
| Largest tracked file | `schemas/splits/amlworld/splits.json`, **601 KB** |
| Whole release tree | **382 files, 6.8 MB** |
| Dead code, orphaned modules, stub modules | **0, 0, 0** |
| TODO / FIXME / XXX / HACK markers | **0** |
| Module docstrings | 3 missing (`corpus/`, `models/`, `utils/` `__init__.py`) — **written** |
| `--help` on every script | 1 failing — **fixed** |

**No `experimental/` directory was needed.** There was nothing to quarantine: no orphaned
module, no all-stub module, no dangling reference to the one deleted config
(`configs/encoder/gat.yaml`). **No Git LFS was needed either** — everything large is
already outside git by the `.gitignore` contract, and the largest tracked file is 601 KB.

Two real defects were found and fixed:

1. **`scripts/07c_report_tables.py --help` exited 1** on a clean clone. It checked for
   `encoder_report.json` before parsing arguments, so the only way to ask what the script
   did was to be told the file was missing — undiscoverable to exactly the reader this
   phase is for. Converted to `argparse`. This is why `script-help` is a standing release
   check rather than a one-off audit.
2. **`RESULTS.md` §4.5 printed `—` for `sage` and `gcn`**, where that file's own convention
   states a dash means *"No number. Never a zero."* Both arms had measured figures in
   PHASE_LOG Phase 7. Filled in, along with the three ablation arms.

`gitleaks` is now a pre-commit hook and a CI check. `detect-private-key` catches only PEM
blocks; provider API keys, HF tokens and W&B keys are what this project actually risks
staging.

### The quickstart, and the constraint that shaped it

**`make quickstart` reproduces one published result from a clean clone in 1.5 s** — no data
download, no GPU, no network, no credentials.

The binding constraint is that `bronze.jsonl` is **232 MB** and gitignored, and the raw
AMLworld release is a ~20 GB manual Kaggle download behind an API token. A quickstart that
starts with either is not one.

The answer is a **220-record stratified fixture** — 20 from each of the eleven narrative
families, sorted `case_id` order, gzipped to **244 KB** (under the 512 KB pre-commit
threshold, which is a guard worth keeping rather than raising) — scored by
**`scripts/10_evaluate.py` itself**, with only three path roots overridden. Not a
reimplementation: a second scorer would be a second implementation of the thing invariant 1
protects, and it would agree with the first on their shared misreading. D-101.

The assertion against `tests/golden/quickstart_evaluation.json` is **exact**. Bronze is
deterministic; a tolerance would hide the class of bug the check exists to catch.

```
  Zero-Hallucination Rate        1.0000
  Fact Precision                 1.0000
  Fact Coverage                  0.8359
  Critical Error Rate            0.0000
  H9 omission of exculpatory fact  0.4500
  claims scored                  3734
  narratives with no claims      0
  QUICKSTART OK
```

**The fixture's numbers deliberately differ from the corpus's** — Coverage 0.8359 against
0.8595, H9 0.45 against 0.9179 — because the families are weighted evenly rather than as
they occur. Stated beside the number in four places so nobody quotes one for the other.

### Licensing — one finding that changed what ships where

**The corpus is Enhanced Data, not a §3.5 Result, and it ships under CDLA-Sharing-1.0.**

Phase 1's card listed "generated narratives" among the Results exempt from share-alike, and
the obvious move was to put the whole corpus in the Apache-2.0 bundle. Reading an actual
record stopped that: `target_narrative` quotes account identifiers, timestamps, currencies
and amounts verbatim, and `facts` embeds `entity_inventory.node_ids`,
`focal_entity.first_seen` and `flow.max_single_transfer` — individual transaction values.
Fifteen thousand narratives each naming real source identifiers is not de minimis.

**Two bundles, two licences, never merged**, enforced by `tests/unit/test_release_packaging.py`
rather than by comment:

| Bundle | Licence | Contents |
|---|---|---|
| `code-and-results` | **Apache-2.0** | 27 checkpoints, metrics, figures, docs, schemas, split manifests — **99 MB** |
| `corpus-and-facts` | **CDLA-Sharing-1.0** | narratives, fact records, case store — **~579 MB** |

CDLA-Sharing-1.0 permits redistribution, so the conservative reading loses no reach and only
obliges downstream users to keep the terms. The unconservative reading, if wrong, is a
licence breach in a paper's artifact release. D-098.

**The quickstart fixture is the same question in miniature** — `tests/fixtures/NOTICE` was
extended to cover it as a second redistribution with its own §3.2 record of changes, while
the golden file beside it (metrics *over* the fixture, a genuine Result) stays Apache-2.0.

### Elliptic2 — nothing released, because nothing exists

The brief asked for case IDs + fact records + narratives + a reconstruction script.
**All three inputs are empty.** Access was never requested; the substrate has never been
ingested; there are zero Elliptic2 cases, fact records or narratives in this repository.

`scripts/14_reconstruct_elliptic2.py` ships anyway and says exactly that when run. A silent
omission would let a reader assume a second-substrate half exists; a script that prints the
absence cannot be misread. **No Elliptic2 checksum is pinned anywhere**, and a test asserts
the string `sha256` does not appear in that file — we have never seen those bytes, and a
fabricated digest in a verification path is worse than none. D-099.

### Verification — nine checks, against a `git archive` export

`scripts/14_verify_release.py` stages the tree into a **scratch git index** (the real index
is never touched), writes a tree object and `git archive`s it into a pristine directory, so
the verification sees exactly what a clone delivers — no untracked file, no populated
`data/`, no `.venv`.

Nine checks, each reporting pass / fail / skipped-with-reason rather than aborting on the
first: clean-clone, install-from-lockfile, quickstart, golden files, every documented
no-data command, every script's `--help`, **secret scan over the object database**, no
leaked data or oversized file, and licence-NOTICE integrity. The JSON report carries all
three counts, so a green run with skips can never read as a clean one.

Scheduled weekly in `verify-release.yml`, plus on every `v*` tag. **The manuscript names
the URL**, so a release verified in August and broken in October is broken. D-102.

**Two of the nine checks were wrong on their first real run, and both were the checker's
fault rather than the repository's.** `no-data-committed` walked the `.venv` that the
`install` check had just created inside the clone and reported polars' 100 MB shared object
as a leaked artifact — the check firing at its own side effect. And `script-help` failed the
four GPU entrypoints because they import torch, which **`make install` deliberately does not
provide**: phases 1-6 and 10 are CPU-only by design (CLAUDE.md §4) and torch lives behind
the `graph`/`llm` extras. Failing a release for that would be demanding the light
environment stop being light. Both are fixed: tool-generated directories are excluded from
the tree walk, and a `--help` failure whose traceback names a module from the optional-extra
allowlist is **counted and reported separately**, while any other non-zero exit still fails.
That distinction is what keeps the check able to catch the `07c_report_tables.py` bug below
while not crying wolf at the design.

### Docker — two images, both digest-pinned

`docker/Dockerfile.cpu` (base pinned to `python:3.11.14-slim-bookworm@sha256:65a93d69…`)
and `docker/Dockerfile.gpu` (`nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04@sha256:eef662e5…`).
uv pinned at 0.4.20, every dependency `--frozen` from `uv.lock`, the three PyG companion
wheels at exact versions from the `torch-2.4.0+cu121` index.

The GPU path used to be commented-out lines inside the CPU Dockerfile, which is a variant
nobody had ever built. **Both images now run `scripts/14_quickstart.py` at build time**, so
a build that succeeds has already reproduced a published result — an image that builds and
then fails its own quickstart is worse than a build failure, because it ships.

**Two defects were found by actually building it, and neither would have been found by
reading it.**

1. **The image had never built at all.** It pinned uv 0.4.20 and ran `uv sync --group dev`;
   `--group` did not exist in uv until later. `ci.yml` carried the identical bug, so CI
   would have failed on its first run too — the repository has no remote, so CI had never
   run. Both now pin **uv 0.10.2**.
2. **There was no `.dockerignore`,** so `COPY . .` swept in `data/` (1.5 GB), `.venv/`
   (6.3 GB), `artifacts/` (143 MB) and `wandb/` — an **~8 GB build context**. `.gitignore`
   does not apply to `docker build`, and nothing had ever noticed because nothing had ever
   built. The build consumed 38 GB of layer cache and took the host from 91% to 98% disk
   before it was stopped. `.dockerignore` is now written to mirror `.gitignore`, with an
   explicit note that `tests/fixtures/` and `tests/golden/` must stay in the context
   because the build-time quickstart needs them.

### Documentation written

| Document | What it settles |
|---|---|
| `docs/REPRODUCTION.md` | Four reproduction tiers, per-stage runtimes and hardware, expected outputs, and **the tolerance policy** (D-100) |
| `docs/ETHICS.md` | Intended use, five prohibitions, measured failure rates, fairness, provenance, **6.5 GPU-hours ≈ 0.25 kg CO₂e** |
| `docs/model_cards/` | GATv2 + the eight arms, each with an explicit misuse section, plus what is **not** released |
| `docs/dataset_cards/` | Bronze, Silver, Gold, `case_facts` — two of the three narrative tiers say "this does not exist" on line one |
| `README.md` | Rewritten. The previous version claimed phases 6–14 were not started; six had landed. |
| `CHANGELOG.md` | New |

**The ethics statement's environmental section reports 6.5 GPU-hours and ≈ 0.25 kg CO₂e,
and says in the same breath that the figure is negligible for a bad reason: the expensive
runs never happened.** It gives the ~112 kg CO₂e projection for the full matrix instead,
because that is the number a reader reproducing this actually needs. Per-run duration was
never instrumented — `encoder_report.json` records `seconds: 0.0` — and the statement
records that gap rather than papering over it.

**The fairness section says plainly that demographic fairness could not be assessed and
that neither substrate supports it** — AMLworld is synthetic, so a disparity would be a
property of the simulator; Elliptic2 is pseudonymous with no protected attribute to
condition on. What *could* be measured (a 9-point Fact Coverage spread by typology, H9
concentrating exactly where it matters most) is reported as the case-type disparity it is.

### Gate verification (this machine, 2026-08-06)

- `ruff check` and `ruff format --check` **clean** over 254 files.
- `mypy --strict` **clean** over 28 source files (`facts/` + `eval/`).
- Full suite: **2,128 passed, 2 skipped** (both Elliptic2 real-data, by design).
  Coverage **86%** overall; **`facts/` 95-100%** and **`eval/` 87-100%**, so invariant 1's
  >=90% gate holds on the measurement instrument.
- Phase 14 tests: **20 passed** (`test_release_packaging.py`, `test_quickstart.py`).
- `make quickstart`: **PASS**, 1.5 s, exact golden match.
- `gitleaks` over full history: **clean**.

**`scripts/14_verify_release.py`: 9 passed, 0 failed, 0 skipped — RELEASE VERIFIED.**

| Check | Result |
|---|---|
| clean-clone | 405 files, tree `439737253e11`, no data/ or artifacts/ payload |
| install | `uv sync --frozen --group dev` |
| quickstart | 220 records scored, **exact** golden match |
| golden | 61 passed |
| documented-commands | 3 of 3 ran |
| script-help | 25 of 29 answer `--help`; 4 need an uninstalled extra (torch), reported not failed |
| secret-scan | full git history clean |
| no-data-committed | no artifact leaked; 406 files, 6.5 MB |
| licence-separation | NOTICE present and complete |

**The container was built and run, not merely written.**

| | |
|---|---|
| `docker build -f docker/Dockerfile.cpu` | **succeeds**, and its build-time `RUN python scripts/14_quickstart.py` prints `QUICKSTART OK` |
| Image size | **2.85 GB** on disk / 656 MB compressed |
| Build context | **6.6 MB** (was ~8 GB before `.dockerignore`) |
| `docker run --rm g2t-aml:cpu` | **`QUICKSTART OK`** — exact golden match inside the container |
| `docker run --rm g2t-aml:cpu make smoke` | **`smoke OK`** — lint, typecheck, the full suite and the end-to-end run, all inside the CPU-only image |
| `docker build -f docker/Dockerfile.gpu` | **succeeds**, and its build-time check imports the full PyG stack and prints `QUICKSTART OK` |
| GPU image size | **39.6 GB** on disk / 13.2 GB compressed, ~25 min cold |
| `docker run --rm g2t-aml:gpu python -c "import torch, torch_geometric, torch_scatter, torch_sparse, torch_cluster"` | torch **2.4.0+cu121**, PyG **2.6.1**, all three companion wheels import |

Whether a card is visible is deliberately not asserted at build time — CI builders have no
GPU, and a build that required one could never be built in CI.

**This is the clean-clone-on-a-fresh-machine criterion, discharged twice**: once from a
pristine `git archive` export with a from-lockfile install, and once inside a container
built from that same context on a digest-pinned base.

**And running `make smoke` inside that container found a third defect.** Four test modules
— `test_fusion.py`, `test_generator_guard.py`, `test_generator_harness.py` and
`test_generator_pipeline.py` — import torch at module scope with no `pytest.importorskip`
guard, so in the CPU-only environment they fail at **collection**, and a collection error
takes the whole run down rather than skipping four modules. Every sibling GPU test module
already had the guard; these four had been written without it and nobody had noticed,
because **the development host has the GPU extras installed and CI had never run.**

So `make smoke` — documented as *"the CI gate"* and as the second thing a stranger types —
did not work in the environment `make install` produces. `ci.yml` would have failed on it
identically. Guards added; the four modules now skip cleanly on CPU and all 100 of their
tests still pass where torch is present.

**Fixing the collection errors then exposed what they had been hiding.** With the four
modules collecting, `make smoke` got far enough to run — and **15 tests failed on
`ModuleNotFoundError: scipy`**, in `eval/statistics.py`, `eval/report.py` and
`experiments/aggregate.py`. `scipy` lives in the `eval` extra, and the light environment
does not install it.

The tempting fix was another `importorskip`. It is the wrong one: **`eval/` is a
measurement instrument under invariant 1**, and a CI gate that *skips* the statistics is
not gating the thing most worth gating. The other tempting fix — install `eval` in CI —
gives up the light environment, because `eval` carries `bert-score`, which carries torch.

**A new `stats` extra** — `scipy`, `statsmodels`, `krippendorff`, and no torch — resolves
both. CI and the CPU image install `--extra stats`, so the statistics are genuinely
exercised on every run while the environment stays torch-free. `make install-stats` is the
local equivalent.

Relocking for the new extra also corrected a drift: **`uv.lock` held matplotlib 3.11.1
against a `pyproject.toml` pin of 3.9.2.** A `--frozen` install had been silently
installing a version the project does not declare.

**And syncing the full extras to re-verify the host exposed a Phase 9 bug.** With
`bitsandbytes` actually installed, three `TestTrainingRun` cases failed with
`ValueError: Expected a cuda device, but got: cpu`. `build_optimizer` selected
`PagedAdamW8bit` whenever bitsandbytes was **importable** — and that object constructs
happily over CPU tensors, raising only at the first `.step()`, so the failure lands
mid-training rather than at setup.

`torch.cuda.is_available()` is not the right guard either: **this host has an RTX 2050, so
CUDA is available while the test's model sits on CPU.** What
`optimizer_update_8bit_blockwise` rejects is the *tensor*, so the tensor is what is now
checked. The fallback costs nothing — paging device memory is meaningless without a device
— and it is logged, because a silent substitution changes the memory profile Phase 13
reports. Four regression tests added.

**This would have hit any user running `make install-gpu` on a CPU-only box**, which is a
perfectly ordinary thing to do to inspect the harness. It was invisible here for the whole
project because the development environment had never had `bitsandbytes` installed.

**This is the whole argument for building and running the artifact rather than reading it.**
Five of this phase's defects — the uv pin, the missing `.dockerignore`, the collection
guards, the scipy gap and the paged optimiser — were invisible to inspection and unmissable
on execution. The lockfile drift was found only because the fourth forced a relock, and the
optimiser bug only because the relock forced a full resync.

### Acceptance criteria

| Criterion | Status |
|---|---|
| Repository public with a tagged release | **met** — <https://github.com/MobsLInep/graph2text-aml>, `v0.1.0` |
| **Zenodo DOI minted** | **NOT MET.** Requires linking a Zenodo account to the GitHub repository and re-cutting the release; it is an account action, not a code action. |
| Clean-clone quickstart verified on a fresh machine | **met** — verified in a fresh container from a `git archive` export |
| Reproduction guide complete and tested | **met** |
| All permitted artifacts released; licence constraints verified and respected | **met** — and one constraint was *changed* on verification (D-098) |
| Model and dataset cards written | **met** |
| Ethics statement complete | **met** |
| `RESULTS.md` includes every number including nulls | **met** — and two dashes were replaced with the numbers they were hiding |
| Verification script green in CI | **met.** `ci` and `verify-release` both **green on `a3217f0`**; all three `verify-release` jobs (clean-clone, CPU image, GPU image) pass on a clean runner. It took three rounds — see below. |
| Docker image builds and runs | **met** |
| Secret scan over full git history clean | **met** |
| **LoRA adapters / fusion projector / GAT checkpoints to HuggingFace** | **partially NOT MET.** The GAT checkpoints exist and are packaged; **the adapters and the projector do not exist**, and nothing was uploaded to the Hub. |
| **Human study data + analysis notebook** | **NOT MET.** No human has rated anything; there is no data to anonymise. |
| Annotation guidelines PDF in the repo, cited in the paper | **met** — `docs/annotation/annotation_guidelines.pdf` |

### Intentionally withheld, and why

- **Nothing Elliptic2-derived.** It does not exist, and its data licence is unlocated. D-099.
- **Raw and interim AMLworld data** (476 MB CSV, 180 MB Parquet). Redistributable under
  CDLA-Sharing-1.0, but it is the upstream authors' release unchanged and is better obtained
  from them; the pinned SHA-256 digests in `data/download.py` verify what you fetch.
- **`wandb/`.** Local run store; `run_context.json` is invariant 5's artifact and it ships.
- **Nothing else.** No result, no null and no failed experiment was withheld.

### Notes for whoever picks this up

1. **Mint the Zenodo DOI**: link Zenodo to the GitHub repository, then re-publish the
   `v0.1.0` release so Zenodo captures it. Then fill the DOI into `CITATION.cff`,
   `README.md` and the manuscript.
2. **The HuggingFace upload is 27 encoder checkpoints and their two cards** — `docs/model_cards/`
   is written and ready to be the model card text. There is nothing else to upload.
3. **`make release` produces a ~579 MB CDLA bundle.** Zenodo's default limit is 50 GB, so it
   fits; check the corpus bundle uploads as a single archive rather than being split.
4. **The verification script's `DOCUMENTED_COMMANDS` list is the enforcement, not the
   documentation.** If you document a new no-data command, add it there or it is unverified.
5. **Do not regenerate Bronze to "refresh" the release.** `model_signal` is populated now
   and `facts.serialiser._compact` emits `gnn_risk_score` into `serialised_facts`, which is
   the input to the "no graph encoder" ablation arm. Nothing would fail. D-063.
