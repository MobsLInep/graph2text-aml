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
- **Silver** — LLM rewrites of Bronze, each gated by an automated verifier. A rewrite
  that asserts an unsupported or masked fact is **rejected, not repaired**.
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
Phase 4  bronze    deterministic template narratives    -> data/processed
Phase 5  silver    verified LLM rewrites                -> data/processed
Phase 6  (gold / annotation tooling)                    -> docs/annotation
Phase 7  train-encoder     GAT encoder                          [GPU]
Phase 8  fusion            projection into LM embedding space   [GPU]
Phase 9  train-generator   QLoRA finetune                       [GPU]
Phase 10 eval      faithfulness + surface metrics       -> artifacts/metrics
Phase 12 matrix    ablation grid across substrates
Phase 14 release   package artifacts for submission
```

Phases 1–6 and 10 are **CPU-only**. Do not introduce a CUDA dependency into them.

---

## 5. Project invariants

These are not style preferences. Violating one silently corrupts the paper.

1. **The fact layer is a measurement instrument.** A bug in `src/g2t_aml/facts/`
   silently corrupts every headline number in the paper. It requires ≥90% test coverage
   and golden-file tests. Never change it casually.
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

### How the invariants are enforced mechanically

| Invariant | Enforcement |
|---|---|
| 1 | `mypy --strict` scoped to `facts/` + `eval/`; coverage gate; `tests/golden/` |
| 2 | `schemas/splits/` committed; `data.split` configs carry no `seed` key |
| 3 | `g2t_aml.CASE_FACTS_SCHEMA_VERSION`, echoed by `RunContext.schema_versions` |
| 4 | `data/*.yaml: availability` mask; asserted by `tests/integration/test_hydra_compose.py` |
| 5 | `utils/run_context.py` → `run_context.json` in every run dir |
| 6 | Hydra run dir is `${paths.runs_dir}/<date>/<time>_<experiment>`; `make clean` never touches `artifacts/` |
| 7 | `RESULTS.md` + `PHASE_LOG.md` |
| 8 | pre-commit `detect-private-key`; synthetic-ID fixtures only |

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
- **`docs/annotation/`** — the Gold-tier annotation protocol.

---

## 9. Current state

**Phase 0 complete.** Scaffold, tooling, conventions, CI. No science yet — no data
loading, no models, no fact extraction. See `PHASE_LOG.md`.
