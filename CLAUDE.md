# Graph2Text AML — standing brief

**Read this first, every session.** It is the durable context for this repository. If
something here is wrong or stale, fix it here rather than working around it.

---

## 1. What this is

**Graph2Text AML** takes a financial transaction subgraph that has been flagged as
suspicious and generates a human-readable **Suspicious Activity Report (SAR) narrative**
that a financial-crime investigator can use as a first draft.

The contribution is the shift from opaque classification — *"node is illicit, score
0.87"* — to explanation — *"this account received funds from three counterparties
previously associated with illicit activity, then dispersed them to nine fresh accounts
within 22 hours — a pattern consistent with layering."*

**Target venue:** *Expert Systems with Applications* (Elsevier).

**Architecture:** GAT graph encoder → fusion/projection layer → QLoRA-finetuned
Llama-3.1-8B → structured narrative. **The fusion layer is the technical novelty:** how
graph structure gets injected into the language model's embedding space.

---

## 2. The two datasets, and why both

| | **IBM AMLworld** | **Elliptic2** |
|---|---|---|
| Source | Altman et al., NeurIPS D&B 2023 | Bellei et al., KDD MLF 2024 |
| Nature | Synthetic | Real Bitcoin |
| Role | Primary development + evaluation substrate | Real-world demonstration |
| Ground truth | Complete | Partial, subgraph-level |
| Has | timestamps, amounts, bank IDs, entity types, per-stream typology labels | anonymised features only |
| Scale | — | 122K labeled subgraphs over 49M node clusters / 196M edge transactions |
| Access | Open | **Gated**, not redistributable |

AMLworld carries per-stream labels across **8 laundering typologies**: fan-out, fan-in,
gather-scatter, scatter-gather, simple cycle, random, bipartite, stack.

Two substrates deliberately: AMLworld gives complete ground truth to *measure against*,
Elliptic2 shows the method survives contact with real, anonymised data. They have
different fact availability, which is why invariant 4 exists.

---

## 3. The corpus problem, and the three tiers

**The single most important architectural fact: there is no existing corpus of (graph,
SAR narrative) pairs anywhere in the world.** Real SARs are confidential by statute. We
construct the corpus ourselves:

- **Bronze** — deterministic template rendering from the fact record. Faithful by
  construction, stylistically flat. The floor everything else must beat.
- **Silver** — LLM rewrites of Bronze, each gated by an automated verifier. A rewrite that
  asserts an unsupported or masked fact gets **at most two targeted repair attempts, then
  is discarded and logged** (D-046). Two teachers, from different families, assigned
  deterministically and balanced per stratum — one teacher is distillation, and the code
  refuses to run with one. **The discard log is a deliverable**, not a debug artifact.
- **Gold** — human-authored narratives under the protocol in `docs/annotation/`. Small,
  held out, never used for training.

**The verification machinery that makes the corpus trustworthy is the *same code* that
measures faithfulness at evaluation time, run in reverse.** This is why the fact layer is
load-bearing for both the data and the headline numbers, and why invariant 1 is invariant 1.

---

## 4. Pipeline stages and dependency order

```
Phase 1  data      ingest + normalise substrates        -> data/interim
Phase 2  splits    frozen temporal split manifests      -> schemas/splits/   [committed]
Phase 3  facts     extract case_facts records           -> data/processed
                   schemas/case_facts_v1.json + vocab_v1.yaml  [committed, FROZEN]
Phase 4  bronze    deterministic template narratives    -> data/processed
Phase 5  silver    verified LLM rewrites                -> data/processed
Phase 6  (gold / annotation tooling)                    -> docs/annotation
Phase 7  train-encoder     GAT encoder + model_signal write-back [GPU]
Phase 8  fusion            projection into LM embedding space   [GPU] DONE (untrained)
Phase 9  train-generator   QLoRA finetune                       [GPU] BUILT, NOT RUN
Phase 10 eval      faithfulness + surface metrics       -> artifacts/metrics
Phase 11 matrix    the 17-system experiment grid        -> artifacts/matrix
                   aggregate + figures + RESULTS.md     BUILT, NOT RUN
Phase 12 (metric validation / decision-setting study)  BUILT, NOT RUN
Phase 13 efficiency  VRAM, throughput, latency, cost    RUN; 2 of 17 systems measured
Phase 14 release   public repo, docs, verification    DONE; Zenodo DOI + HF upload open
```

Phases 1–6 and 10 are **CPU-only**. Do not introduce a CUDA dependency into them.

---

## 5. Project invariants

These are not style preferences. Violating one silently corrupts the paper.

1. **The fact layer is a measurement instrument.** A bug in `src/g2t_aml/facts/`
   silently corrupts every headline number in the paper. It requires ≥90% test coverage,
   golden-file tests, and **calibration against an independent oracle** — the round-trip
   test alone cannot catch an extractor bug, because the probe reads its claims from the
   record and so verifies a wrong value against itself (D-034). Never change it casually.
2. **Splits are temporal and frozen.** Never regenerate splits from a seed at runtime.
   Split manifests are committed ID lists with content hashes.
3. **Schema versions are pinned and recorded in every derived artifact.** Changing
   `case_facts` schema after corpus generation means regenerating the corpus.
4. **Nothing may assert a fact that does not exist for its substrate.** Elliptic2 has no
   amounts, no currencies, no real timestamps, no entity types. The fact record carries
   an availability mask and generation must respect it.
5. **Every run records:** git SHA, resolved Hydra config, data manifest hash, all seeds,
   library versions.
6. **Never delete or overwrite a results file.** Write to a new timestamped run directory.
7. **Negative and null results are kept and reported.** `RESULTS.md` includes failed
   experiments.
8. **No real-world PII or identifiers ever enter the repo**, including in test fixtures.
   Synthetic IDs only.
9. **`case_facts` is FROZEN at schema version 1.0.0.** `schemas/case_facts_v1.json` is the
   contract. Changing any field — adding, removing, renaming, or altering a type or an
   availability rule — is a **breaking change that invalidates every fact record, every
   generated corpus and every published number derived from them**, and requires a version
   bump, a `DECISIONS.md` entry, and regeneration from Phase 3 forward. The version is
   declared in **five** places and a test asserts they all agree:
   `g2t_aml.facts.schema.CASE_FACTS_SCHEMA_VERSION` (the source of truth), its re-export
   `g2t_aml.CASE_FACTS_SCHEMA_VERSION`, the `schema_version` `const` in the JSON Schema,
   `case_facts_schema_version` in `schemas/vocab_v1.yaml`, and `schema_version.case_facts`
   in `configs/config.yaml`. `scripts/03_extract_facts.py` aborts on a mismatch.
10. **Absence is a typed sentinel, never `0` and never a bare `None`.** A fact family a
    substrate cannot support is `facts.schema.Unavailable`, carrying a reason. A measured
    null (no cycle exists; no illicit node is reachable) *is* a bare `None`, and the two
    mean different things to the checker — UNVERIFIABLE against a sentinel, a real
    comparison against a measured null. Do not conflate them. (D-025)

### How the invariants are enforced mechanically

| Invariant | Enforcement |
|---|---|
| 1 | `mypy --strict` scoped to `facts/` + `eval/`; coverage gate; `tests/golden/`; the 1,000-case round trip **plus** the independent oracle in `tests/oracle.py` (D-034) |
| 2 | `schemas/splits/` committed; `data.split` configs carry no `seed` key |
| 3 | `g2t_aml.CASE_FACTS_SCHEMA_VERSION`, echoed by `RunContext.schema_versions` |
| 4 | `data/*.yaml: availability` mask; asserted by `tests/integration/test_hydra_compose.py` |
| 5 | `utils/run_context.py` → `run_context.json` in every run dir |
| 6 | Hydra run dir is `${paths.runs_dir}/<date>/<time>_<experiment>`; `make clean` never touches `artifacts/` |
| 7 | `RESULTS.md` + `PHASE_LOG.md` |
| 8 | pre-commit `detect-private-key`; synthetic-ID fixtures only |
| 9 | `test_schema_version_is_frozen_and_consistent_everywhere`; `03_extract_facts.py` aborts on a config/code mismatch |
| 10 | `mypy --strict` cannot dereference a union without narrowing; `tests/unit/test_facts_availability.py` |

---

## 6. Code conventions

- **No hardcoded paths, anywhere.** Every directory root lives in `configs/paths/` and is
  reached as `cfg.paths.*`. There is a test that greps `src/` for violations.
- **Notebooks are exploration only.** `src/` never imports from `notebooks/`. There is a
  test for that too.
- **Explicit types across module boundaries.** Dataclasses or pydantic models, not dicts,
  for anything one module hands another.
- **Every public function gets a docstring stating what it returns and what it raises.**
  Google style. Enforced by ruff `D` rules and a contract test.
- **Atomic writes only.** Use `utils/io.py`. A killed job must never leave a half-written
  file that a later stage treats as valid.
- **Hash everything that feeds a result.** `utils/hashing.py` is canonical: sorted keys,
  resolved interpolations, order-independent where order is not meaningful.
- **No stub modules.** Empty `__init__.py` is fine; a file full of `pass` bodies is not.
  Stubs rot.
- Line length 100. Python 3.11. `ruff` for lint + format, `mypy --strict` for the two
  measurement modules only.

---

## 7. How to run things

```bash
make install        # light CPU-only env + dev tools + pre-commit hooks
make install-gpu    # adds graph + llm + human extras (CUDA 12.1)
make smoke          # the CI gate: lint + typecheck + tests + e2e smoke run
make lint format typecheck test
make help           # every target
```

Hydra overrides work on any entrypoint:

```bash
uv run python scripts/smoke.py data=elliptic2 experiment=full corpus=silver seed=7
```

Pipeline targets (`data`, `splits`, `facts`, `bronze`, `silver`, `train-encoder`,
`train-generator`, `eval`, `matrix`, `release`) exist and announce which phase implements
them.

---

## 8. Where things are written down

- **`DECISIONS.md`** — append-only. Every non-obvious technical choice, with rationale.
  Add an entry *when you make the decision*, not later.
- **`PHASE_LOG.md`** — append-only. What each phase delivered, what was deferred.
- **`RESULTS.md`** — created in Phase 10. Includes negative results (invariant 7).
- **`docs/data_cards/`** — one card per substrate: provenance, licence, known limitations.
- **`docs/annotation/`** — the Gold-tier annotation protocol (Phase 6), plus two documents
  frozen in Phase 3 *before* any narrative existed, so annotators and the automated metric
  score against one definition: `salience.md` (what an adequate narrative must mention, per
  typology) and `hallucination_taxonomy.md` (the nine classes and the Critical Error Rate).

---

## 9. Current state

**Phases 0, 1, 2, 3, 4, 7, 10, 13 and 14 complete. Phases 5, 6, 8, 9, 11 and 12 have
complete, validated machinery and no data or no run**, for the same shape of reason in each
case and for different missing inputs.

- **The repository is public and released at `v0.1.0`.**
  <https://github.com/MobsLInep/graph2text-aml>. `make quickstart` reproduces one published
  result from a clean clone in 1.5 s against an exact golden file; `make verify-release`
  runs nine checks against a `git archive` export and is scheduled weekly in CI. Both
  Docker images are digest-pinned and run the quickstart at build time. **Two release
  criteria are open and neither is a code action**: the Zenodo DOI is not minted (needs a
  Zenodo↔GitHub link and a re-published release) and nothing is on the HuggingFace Hub
  (there are 27 encoder checkpoints to upload and nothing else — the adapters and the
  fusion projector do not exist). See PHASE_LOG Phase 14 and D-098…D-102.
- **`make smoke` needs `--extra stats`, and CI and the CPU image install it.** scipy lives
  behind an extra; `eval/statistics.py`, `eval/report.py` and `experiments/aggregate.py`
  need it and are measurement code under invariant 1, so they must be **gated, not
  skipped**. It is a separate extra from `eval` because `eval` pulls bert-score and
  therefore torch, and the light environment's whole point is that it has none. Four
  generator test modules also gained `pytest.importorskip("torch")`: without it they failed
  at *collection*, which aborts the run rather than skipping them, and **`make smoke` did
  not work in the environment `make install` produces**. Both were invisible on this host,
  which has every extra installed. (PHASE_LOG Phase 14)
- **The corpus ships under CDLA-Sharing-1.0, not Apache-2.0 (D-098).** Note 5 below said
  §3.5 exempts narratives; reading an actual record showed it does not — `target_narrative`
  quotes account ids, timestamps and amounts verbatim and `facts` embeds per-transaction
  values, which is Enhanced Data. **The two bundles are licence-homogeneous and must never
  be merged**, and `tests/unit/test_release_packaging.py` is what enforces it.

- **Silver is NOT built** — no API credentials, so zero teacher calls have been made. Run
  `make silver-dry-run` before `make silver`. See PHASE_LOG Phase 5.
- **Gold is NOT written** — no annotator has been recruited. The sample is drawn and
  reserved, the interface runs, and the ingestion path is verified end to end on real
  cases with hand-written narratives; **not one Gold narrative exists**. See PHASE_LOG
  Phase 6. Recruitment is the critical path and no further engineering advances it.
- **No generator arm has been trained, and Gate 8 is OPEN.** The Phase 8 fusion layer, the
  Phase 9 harness and the guard are built and tested on CPU against a stub backbone (119
  tests). **This machine has a 4 GB RTX 2050 and 7 GB of system RAM.** Llama-3.1-8B at nf4
  is ~4.5–5.6 GB of weights alone — more than the card holds before a single activation —
  and CPU offload is closed by the RAM. This is not the `max_seq_len` fallback the Phase 9
  brief describes; that assumes a 24 GB card. **Do not attempt `make train-s1` here.** See
  PHASE_LOG Phase 8+9 and D-068. Nothing is known about whether the model reads a soft
  token; every number in that entry measures the harness, not the method.
- **The Phase 11 matrix is declared, orchestrated and tested, and not one system has run.**
  17 systems, 25 runs, every one documented as a non-run with its blocker in `RESULTS.md`
  §2. `make matrix-plan` prints the plan without touching anything. **B1 and B2 need no GPU
  and no network and are runnable today** — the cheapest way to put two real rows in the
  results table and exercise the whole run→aggregate→figure path on real data.
  **Do not run the full matrix before S1 and A1 have answered Gate 8**: if the fusion layer
  is decoration, the other twenty-three runs answer a question the paper will not be asking.
- **Phase 12's study is built and blocked, and its first blocker is paperwork.** The design,
  the blinded interface, the analysis and the anonymised release are complete and tested
  against simulated responses (101 tests). **Ethics approval has not been applied for** —
  4-8 weeks, and it blocks all data collection. `docs/human_study/` holds the application
  ready to submit. **Only Bronze has generations**, so `make study-build` refuses: a
  five-arm study has one arm. No rater is recruited. The trigger date for falling back to an
  internal expert review is **2026-09-15** (D-092), and it is set on *submission*, not
  approval.
- **Phase 13 ran, and measured two of seventeen systems end to end.** `make benchmark`
  measures B1 and B2 at n=100 with 20 discarded, plus the components every other row shares
  (graph load, index build, encoder inference and training, guard verification), and writes
  the other fifteen as rows carrying their blockers. **Three numbers to carry forward: case
  extraction is 72% of B1's end-to-end latency and generation is 18%; B2's whole inference
  VRAM footprint is 0.025 GB; and the measured guard figure is verification only** — 1.90 ms
  for four candidates, model-independent, taken from the stage mean and not from differencing
  two runs' p50s, because **this host's between-run spread is 2x** (B1 p50 ranged 5.4-11.2 ms
  over six runs) and no Phase 13 conclusion rests on a smaller difference. Any guard overhead
  ratio computed for B1 is a fact about B1 and must never be quoted as the guard's overhead
  in general. `docs/deployability.md`
  covers all seventeen rows on the one axis that needs no GPU. D-093…D-097.

Scaffold and tooling; both substrate loaders, the
canonical representation and the data cards; case construction, the frozen temporal split
manifests and the leakage auditor; the fact layer — schema, extractor, three-valued
checker, controlled vocabulary and serialiser; the Bronze corpus — template engine,
15,707 narratives, and the ten-point validation harness; the Gold annotation kit; and the
graph encoder — six arms, three seeds each, with `model_signal` written back into every
fact record.

`src/g2t_aml/models/encoder/` is Phase 7. Six arms behind one `BaseEncoder`, differing
**only** in `message_passing`: GATv2 (primary), GINE, GraphSAGE, GCN, a virtual-node graph
transformer, and an MLP control with no message passing. `make train-encoder` runs the
whole sweep, `make score-cases` writes `model_signal`, `make encoder-gate` runs the tests.
The four numbers to carry forward:

- **GATv2 beats the MLP control by +0.070 AUC-PR**, excluding zero at all three seeds —
  the gate passed. But **the control reaches 0.80** on its own. Message passing is worth a
  real but modest margin over node-local summary statistics, and any fusion result in
  Phase 9 must be read against 0.80 rather than against zero.
- **GIN (0.880) and the graph transformer (0.888) out-score GATv2 (0.872)**, neither
  significantly. GATv2 stays primary; D-064 records the evidence and what would change it.
- **Edge features are worth +0.114; positional encodings are worth nothing** (+0.002, not
  significant) and **focal loss is no better than weighted BCE** (−0.006, not significant).
  Both nulls are kept and reported (D-065, D-066).
- **The linear probe reaches 0.33 structural macro-F1** on the pooled tokens Phase 8
  consumes. That is the Phase 8 forecast and it is a caution: `fan_out`, `gather_scatter`
  and `cycle` are recoverable, `stack` and `random` are not.

`src/g2t_aml/models/fusion/` is Phase 8 and `src/g2t_aml/models/generator/` is Phase 9.
Five things in them are load-bearing, each documented at its definition: the fusion
projector trains in **fp32** and never quantised (`assert_projector_is_fp32`, D-069); the
loss is on the **completion only**, never the prompt, the soft tokens or the padding; there
are **three learning rates** — LoRA 2e-4, fusion 1e-3, encoder 1e-5 — because a single rate
produces a run indistinguishable from the architecture failing (D-070); the **A1 control is
a derangement** with a zero fixed-point count, not a `randperm` (D-071); and the guarded and
unguarded results are **two separate table rows** (D-073). `make generator-gate` runs the
tests, `make train-s1` … `train-b8` the arms, `make gate8 S1=… A1=…` the decision.

`src/g2t_aml/eval/` is Phase 10, and it is a measurement instrument under invariant 1
alongside `facts/` — `mypy --strict` covers both. Three layers, and the reporting order is
a decision rather than a habit: **Layer 2 faithfulness leads everywhere**, Layer 1 overlap
follows under a heading saying what it is for, and Layer 3 (`metric_validation.py`) is
built and waits on Phase 12. **Zero-Hallucination Rate is the headline** — the
per-narrative binary, not averaged precision, because one fabricated fact makes a SAR
unfileable regardless of the rest (D-077). `make eval` scores every configured system,
`make eval-bronze` runs the Bronze-only gate in 41 s with no GPU and no network, and
`make eval-gate` runs the tests. Four things in it are load-bearing:

- **Two independent claim extractors, and their κ is the validation.** Method A aligns
  against Bronze slots and applies cue rules; Method B decomposes with an LLM and judges
  entailment against the serialised record, and **never calls `check_claim`** — two
  extractors that both consulted the checker would agree by construction (D-078). The κ is
  **not yet measured**: Method B needs API credentials, the same blocker as Silver.
- **Cue attribution can only sharpen a verdict, never soften one** (D-076). Phase 5's
  extractor left a *wrong* quantity as UNVERIFIABLE, which is right for a repair loop and
  wrong for a metric that reports hallucination as *contradicted* over total. An unaligned
  number matching no cue is still not promoted to H2.
- **No new tolerance anywhere** (D-075). Every verdict comes from the Phase 3 checker under
  the frozen policy; `eval/` owns no threshold that decides a verdict.
- **Bronze scores 1.0000 Zero-Hallucination over all 15,707 narratives, 296,196 claims, all
  supported** — one claim different from Phase 4's independent count. That is the CI gate,
  and `n_narratives_with_no_claims = 0` is asserted beside it, because a perfect score over
  an empty claim set is what a broken extractor produces.

`src/g2t_aml/human/` also holds Phase 12: `study_design.py` (the balanced incomplete block
design, the blinding, the anchor block), `study_ui.py` (the rating interface and its two
clocks), `study_timer_component/` (one static HTML file, the only thing that can see
`visibilitychange`), `study_analysis.py` (**the only module that unblinds**) and
`study_release.py`. `make study-gate` runs the tests; `study-build`, `study-rate`,
`study-analyse` and `study-release` are the pipeline. Four things in it are load-bearing:

- **The design reserves an anchor block every rater rates, or there is no Krippendorff
  alpha at all.** The allocator spreads cases as thinly as possible, which is right for
  breadth and leaves *zero* doubly-rated cells — and alpha is defined only over units two
  raters judged. Found by running the analysis on simulated responses and reading its own
  warning, not by inspection. (D-088)
- **Friedman needs complete blocks and this design has none**, because no rater sees a case
  twice. Friedman runs on rater-blocked means and **Durbin** on the case blocks, and both
  are reported with their block counts. Durbin's implementation is validated by asserting it
  reduces to Friedman exactly on a complete matrix. (D-089)
- **Two clocks time each item, and `timing_source` says which one won.** Streamlit does not
  re-run when a tab is hidden, so a server-side clock counts a coffee break as reading time.
  A silent fallback would put those minutes into the headline number. (D-090)
- **Every statistic is validated against something outside this repository** — Krippendorff
  ordinal alpha against the published 0.815 worked example (we get 0.8154), Friedman and the
  correlations against scipy, Nemenyi against Demsar's table. Invariant 1's rule: two
  implementations by one author agree on their shared misreading.

`src/g2t_aml/human/` is Phase 6: the stratified Gold sample, the test-only reservation, the
Streamlit annotation interface (fact panel, graph view, live validation, store),
calibration, agreement, the second-reviewer workflow and ingestion into `tier="gold"`.
**Two rules run through all of it.** An annotator is never shown generated text — no
Bronze, no Silver, no model output — and `tests/integration/test_repo_contract.py` asserts
that no annotator-facing module can reach a narrative. And a Gold test item is never
trained on, enforced by `corpus/training_data.load_training_records` rather than by anyone
remembering. `make gold-gate` runs the Phase 6 tests; `make gold-sample`, `annotate`,
`calibrate`, `gold` and `guidelines-pdf` are the pipeline.

`src/g2t_aml/corpus/silver/` is Phase 5: prompts, teacher clients, claim extraction, the
generate→verify→repair→discard loop, quality filtering and the runner. It is driven entirely
through the `Teacher` protocol, so `ScriptedTeacher` exercises the whole pipeline with no
network and the base environment needs no provider SDK — `anthropic` is the `api` extra.
`make silver-gate` runs the Phase 5 tests.

`src/g2t_aml/corpus/` is Phase 4. `make bronze` renders the corpus in ~90 s and gates it;
`make bronze-gate` runs the renderer, harness and corpus tests. **`training_record_v1.json`
is frozen at 1.0.0 and carries all three tiers** — Silver and Gold differ only in `tier`
and `generator`, and the same ten-point harness gates all three (D-037).

`src/g2t_aml/facts/` is the measurement instrument (invariant 1). It runs in both
directions: `extract_facts` builds a checkable record from a case, and `check_claim`
verifies a narrative's claims against exactly that record using the same field paths. The
schema is **frozen at 1.0.0** (invariant 9). `make facts` extracts, `make facts-gate` runs
the round trip and the oracle.

`data/interim/amlworld_hi_small/` holds the ingested HI-Small graph. It reproduces the
published figures **exactly** — 515,088 nodes, 5,078,345 edges, and all nine typology
counts. `make data` regenerates it in ~14 s.

`schemas/splits/amlworld/` holds the committed split manifest: **10,932 / 2,028 / 3,196**
cases, temporally disjoint, audited. `make cases` regenerates it in ~4.5 min.

`data/processed/amlworld_hi_small/corpus/` holds **bronze.jsonl** — 15,707 records,
train 10,488 / val 2,027 / test 3,192 — plus its validation and diversity reports and a
per-family sample file a human can read. 296,195 claims, **100% SUPPORTED, 0 CONTRADICTED,
0 UNVERIFIABLE**, all ten checks at zero failures.

Twenty-one things that will bite if forgotten:

1. **AMLworld nodes are keyed `"<bank>|<account>"`.** Account ids are unique only within a
   bank; eight collide across banks, and account-only keying gives 515,080 rather than
   515,088. (D-011)
2. **`transaction_key` is built from source text and cannot be rebuilt from typed
   columns.** The Bitcoin rows carry six decimals, so a float round-trip loses
   transactions. Re-load a frame rather than deriving the key. (D-012)
3. **Never use `datetime.timestamp()` on a substrate timestamp.** It reads a naive
   datetime as local time while Polars stores UTC-naive microseconds, which shifts every
   time window by the machine's UTC offset. Use `case_extraction.to_micros`. This bug was
   live in Phase 2 and was silent.
4. **A case does not contain its whole laundering stream.** The 48-hour window cap keeps
   65% of a stream's transactions on average. `typology` means "part of a stream of this
   typology", not "exhibits it in full". (D-019)
5. **AMLworld data is CDLA-Sharing-1.0, not Apache-2.0** — share-alike on data and interim
   artifacts. §3.5 exempts *Results*, which covers models and metrics but **not the
   narratives**: they quote source identifiers, timestamps and amounts verbatim, so the
   corpus is Enhanced Data and ships CDLA-Sharing-1.0 (D-098, and it corrects what the
   Phase 1 data card §2 implied). The two release bundles are licence-homogeneous and must
   never be merged.
6. **Elliptic2 access has still not been requested**, and Phase 14 shipped without it.
   Its loader and case pass-through are written and tested against the documented schema;
   real-data tests skip. **Nothing Elliptic2-derived exists or is released**, and
   `scripts/14_reconstruct_elliptic2.py` says so when run rather than implying a
   second-substrate half. No Elliptic2 checksum is pinned anywhere — we have never seen
   those bytes — and a test enforces that. Phase 12 needs it, and it remains the
   longest-lead open item in the project. (D-099)
7. **The round trip alone cannot catch an extractor bug.** The probe renders its claims
   *from the fact record*, so a wrong value is stated wrongly and verified against itself
   — three injected bugs left it at 100% SUPPORTED. `tests/oracle.py` is what actually
   calibrates the extractor, and it must stay free of any import from `g2t_aml.facts`.
   (D-034)
8. **Degree means distinct counterparties, never transaction count**, everywhere in
   `facts/`. And the node table's `in_degree`/`out_degree` columns are *global* aggregates
   over the whole 515,088-account graph — reading them in the fact layer instead of
   recomputing in-case degree would make every narrative in the corpus unfaithful.
9. **Never sum across currencies.** HI-Small has fifteen currencies, 72,170 cross-currency
   transactions and no exchange rates. Aggregates are withheld as sentinels and the
   per-currency breakdown is always emitted. (D-033)
10. **A qualitative binding must be strictly tighter than the detector it reads.**
    `burst_window_hours` is capped at 24 by construction, so a `rapid_dispersal` binding of
    `"< 24"` would hold for every burst that can exist. It is `"<= 6"`, and two tests
    enforce the relationship. (D-026)
11. **A generated claim is parsed out of the rendered text, never read from the record.**
    Building a claim from the value the formatter started with compares the record with
    itself and reports *any* corpus as 100% SUPPORTED — the same circularity as D-034, in a
    more dangerous place. Every formatter ships with its inverse. (D-040)
12. **Self-BLEU without its reference count is not a number.** On this corpus it reads 0.16
    at one reference and 0.82 at fifty, unchanged in between. It is reported at a fixed five
    and the curve is published beside it; the collapse check keys on distinct scaffolding
    skeletons instead. (D-043)
13. **Every current frontier Anthropic model rejects `temperature` and `top_p` with a 400.**
    Opus 5, Sonnet 5, Opus 4.8, 4.7. They are not ignored — a teacher spec that sets
    `supports_sampling: true` on one of them fails every call in the run. Use `effort` for
    depth and the per-case style directives for surface variety. (D-045)
14. **Silver's slot alignment runs longest-value-first, on token boundaries.** In Bronze's
    document order a long value is always reached before the short values that hide inside
    it; a rewrite reorders content, and then `2` aligns inside `2022-09-02 15:01` — the
    timestamp is scored as dropped *and* its digits come back as invented. It failed 102 of
    300 real paraphrased cases and would have put ~34 spurious points into the discard rate,
    which is this phase's headline finding. Do not "simplify" the ordering back. (D-048)
15. **Anything shown to an annotator must be spelled the way the alignment reads it back.**
    The fact panel renders through *Bronze's* formatters and display maps, not through the
    vocabulary's or its own. A panel showing `9,434.82 Canadian Dollar` against Bronze's
    `9,435 Canadian Dollar` means an annotator who copies it **correctly** produces a value
    aligning to nothing — scored as a dropped fact *and* an invented one, on every monetary
    case. Same trap for roles (`conduit account` vs `a conduit account`) and for the
    threshold citation, where the non-whitelisted wording is H6. (D-054)
16. **`availability.node_labels` is True on Elliptic2.** The substrate labels whole
    subgraphs, so no individual account is labelled at all, and anything reading that flag
    to decide whether accounts are labelled will paint every Elliptic2 node "unflagged" —
    invariant 4 violated in pixels rather than in text. Use `CaseView.has_labels`, which is
    what the fact layer gates `LabelFacts` on.
18. **The MLP control reaches 0.80 AUC-PR with no message passing at all.** It is a
    DeepSets model over case-local node features and it gets within 0.07 of GATv2. Every
    claim about what graph structure contributes — in Phase 8, in Phase 9, in the paper —
    is a claim about that 0.07, not about the whole number. At two epochs the gap read
    0.16, which is what an under-tuned control would have reported.
19. **Never read the interim node table's `in_degree`/`out_degree`/`degree`/
    `total_received`/`total_sent` into a model feature.** They are global aggregates over
    the whole 515,088-account graph, computed across both sides of the temporal boundary,
    so a test-window account's encoding would carry its training-window activity. The fact
    layer has the same prohibition for a different reason (note 8). Everything in
    `models/encoder/features.py` is recomputed from the case's own edges, and a test
    overwrites all five columns with absurd constants and requires the tensor to be
    unchanged. (D-059)
20. **`facts.serialiser._compact` emits `gnn_risk_score` into `serialised_facts`.** Now
    that Phase 7 has populated `model_signal`, regenerating Bronze would push the
    encoder's own score into the **serialisation baseline** — the "no graph encoder"
    ablation arm — and nothing would fail. Bronze is deliberately not regenerated and a
    test pins `gnn_risk_score=none` in `bronze.jsonl`. (D-063)
21. **A checkpoint carries no evidence of how long it trained.** A two-epoch wiring-check
    checkpoint left in `artifacts/checkpoints/` resumes exactly as happily as a converged
    one and its number lands in the results table looking like every other row. This
    nearly happened. Checkpoints now record their `training_config` and resume refuses a
    mismatch. (D-067)
22. **A Hydra `overrides:` block nested under a config key is inert.** The Phase 9 arm
    configs first expressed their settings that way, and `experiment=generator_a1` composed
    with `fusion.shuffle=false` — **the control would have trained as a second copy of the
    treatment**, and the resulting "S1 does not beat A1" would have been two runs of S1
    compared against each other. Nothing failed; it was caught by printing the composed
    config. Arm configs use `# @package _global_` and set real config paths, and
    `tests/unit/test_generator_configs.py` pins every arm — including that A1 differs from
    S1 in the shuffle flag and nothing else.
23. **A causal-LM stub without positional embeddings cannot memorise a sequence.** The
    Phase 9 overfit test plateaued at loss 1.43 for that reason, not because of the harness,
    and the tempting fix was to loosen the threshold — which would have left a test that
    passes whatever the harness does. Adding positional embeddings and an FFN to
    `tests/stubs.py` took it to 0.0095.
24. **Bronze omits an exculpatory fact in 92% of its narratives (H9).** The templates
    report `labels.n_counterparties` and `labels.n_illicit_counterparties` but never the
    licit count, never `labels.focal_is_illicit`, and mention `temporal.burst_detected`
    only when a burst was detected — so a case whose subject carries no illicit label
    produces a narrative that never says so. Found by the Phase 10 harness, not by
    inspection. It is a concrete dimension on which a trained system can beat the template
    (the floor is not uniformly 1.0), and a caution for Phase 12: a narrative that omits
    the exculpatory context is a one-sided case, which is what the decision-setting study
    exists to detect.
25. **A wrong quantity is only CONTRADICTED if a cue rule attributes it to a field.** The
    Phase 10 cue table is a sensitivity surface: a narrative phrased outside every cue has
    its quantities scored UNVERIFIABLE rather than checked, which *understates*
    hallucination. That is the conservative direction and it shows up in the unverifiable
    rate, but it means the rule table has to grow as new phrasings appear, and
    `DeterministicReport.attributed` records which rule fired so a mis-attribution is
    traceable to one rule. (D-076)
17. **The ten-point harness cannot see an unaligned quantity in Silver or Gold.** It
    rebuilds claims from `target_slots`, and those are exactly the values that *did* align
    — so an invented figure produces no slot and is invisible to check 5. Bronze is immune
    by construction; the other two tiers enforce the unverifiable budget at ingestion,
    against the extractor's own rate. (D-057)

`src/g2t_aml/experiments/` is Phase 11: the matrix, declared once and read by everything
downstream. `registry.py` is the source of truth for what the paper's results table
contains — a system is reproducible from its spec plus its Hydra config and nothing else —
and `tests/integration/test_matrix_pipeline.py` asserts that every arm's composed config
agrees with what the registry claims about it, because a registry that says F2 against a
config that composes ungated is a table whose rows are labelled wrong and nothing fails.
Four things in it are load-bearing:

- **The seed asymmetry is three seeds on S1, S2, A1 and B7 and one everywhere else**, stated
  in the paper and enforced in three places: `validate_registry` refuses either violation,
  a single-seed cell renders as a bare mean with a dagger and can never print `± 0.0000`,
  and the bar chart marks every single-seed bar. (D-081)
- **B5 verifies itself, never with our checker, and is given more inference compute than any
  of our arms** — the asymmetry is recorded on `AgenticTrace.n_calls` and reported rather
  than corrected away. Read `baselines.py`'s docstring and `prompts/baseline_verify_v1.txt`
  before touching a baseline. (D-084)
- **Resumption is a completion marker AND a config hash, and the hash is in the path.** A
  marker alone reports an old number under a new configuration; a hash alone re-runs a
  GPU-week. Descriptive fields are excluded from the hash so a prose edit costs nothing.
  (D-085)
- **A partly-failed dependency is not a dependency.** A5 reads S1's checkpoint, so a system
  counts as satisfied only when every one of its seeds succeeded — otherwise A5 reads
  whichever seed survived and produces a number under a row heading that is not true of it.
  (D-086)

`RESULTS.md` is committed and currently contains one measured table (Bronze, Phase 7) and
twenty-five named absences. Regenerate the matrix half with `make aggregate`.

Phase 14 is `scripts/14_*.py` and `docker/`. **Start with `make quickstart`** — it stages a
committed 220-record fixture and scores it with `scripts/10_evaluate.py` *itself*, in 1.5 s,
asserted **exactly** against `tests/golden/quickstart_evaluation.json` (D-101). Four things
in the release are load-bearing:

- **The tolerance policy splits at numbers versus conclusions, not deterministic versus
  stochastic** (D-100). Per-seed AUC-PR gets ±0.02; the *gate outcome* and the ablation
  *signs* get no band at all, because if they flip that is a finding about the method and
  treating it as variance is the specific dishonesty the policy exists to prevent.
- **Verification runs against a `git archive` export, never the working tree** (D-102), so
  an untracked file or a populated `data/` cannot make a broken release look fine. The
  secret scan runs over the **object database**, because a credential committed and later
  removed is still in every clone.
- **The fixture's numbers differ from the corpus's on purpose** — Coverage 0.8359 against
  0.8595, H9 0.45 against 0.9179 — because the eleven families are weighted evenly rather
  than as they occur. Never quote one as the other.
- **`tests/fixtures/NOTICE` is a licence obligation, not documentation.** It covers two
  CDLA-Sharing-1.0 redistributions now, and `14_verify_release.py` asserts it still names
  both. Deleting it is a breach.

See `PHASE_LOG.md` for observed-versus-published tables, the split counts, the sensitivity
findings and the Phase 3 gate results, and `DECISIONS.md` D-010…D-102.

## 10. What the next session should do

Unchanged in priority order from `RESULTS.md` §8 — **the release does not advance any of
it**, and that is the honest reading of where this project stands:

1. `make score-cases`, then **B1 and B2**. CPU-only, runnable today, two real rows.
2. **Credentials** → B3/B4/B5, then Silver, then the Method A/B κ.
3. **A rented ≥24 GB GPU** → S1 and A1 at three seeds, then `make gate8`. Everything else
   in the matrix is read against that comparison.
4. **An annotator.** Nothing else unblocks Gold.

Two Phase 14 leftovers, neither of which is a code action: **mint the Zenodo DOI** (link
Zenodo to the GitHub repo, re-publish `v0.1.0`, then fill the DOI into `CITATION.cff`,
`README.md` and the manuscript), and **upload the 27 encoder checkpoints to the
HuggingFace Hub** using `docs/model_cards/` as the card text.
