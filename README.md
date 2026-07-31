# Graph2Text AML

**Generating Suspicious Activity Report narratives from flagged transaction subgraphs.**

Anti-money-laundering models tell an investigator *that* an account is suspicious. This
system tells them *why*, in prose they can use as the first draft of a Suspicious
Activity Report:

> *"This account received funds from three counterparties previously associated with
> illicit activity, then dispersed them to nine fresh accounts within 22 hours — a
> pattern consistent with layering."*

**Architecture:** GAT graph encoder → fusion/projection layer → QLoRA-finetuned
Llama-3.1-8B → structured narrative. The fusion layer — how graph structure enters the
language model's embedding space — is the technical contribution.

**Substrates:** [IBM AMLworld](https://github.com/IBM/AMLSim) (synthetic, complete ground
truth, 8 laundering typologies) and [Elliptic2](https://www.elliptic.co/elliptic2) (real
Bitcoin, 122K labeled subgraphs, access-gated and anonymised).

---

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) and nothing else — it fetches Python 3.11
itself.

```bash
git clone <repo-url> graph2text-aml
cd graph2text-aml

make install     # CPU-only environment + dev tools + pre-commit hooks
make smoke       # lint + typecheck + tests + end-to-end smoke run
```

`make smoke` is the CI gate. If it passes on a clean clone, your environment is correct.

A single smoke run writes a fresh directory under `artifacts/runs/<date>/<time>_debug/`
containing `resolved_config.json`, `run_context.json`, `smoke.log`, and Hydra's `.hydra/`
composition trace. Nothing is ever overwritten.

```bash
make help        # all targets
```

### Configuration

Every entrypoint is a [Hydra](https://hydra.cc) app composing from `configs/`. Override
anything from the command line:

```bash
uv run python scripts/smoke.py data=elliptic2 experiment=full corpus=silver seed=7
```

Config groups: `paths`, `data`, `encoder`, `fusion`, `generator`, `corpus`, `eval`,
`experiment`. **No path is hardcoded anywhere in the codebase** — every directory root
lives in `configs/paths/default.yaml` and is reached as `cfg.paths.*`. Point the whole
tree somewhere else with `G2T_AML_ROOT=/scratch/g2t uv run python scripts/...`.

---

## Hardware requirements

**Most of this project runs on a laptop.** Phases 1–6 (ingestion, splits, fact
extraction, Bronze and Silver corpus construction) and Phase 10 (evaluation) are
CPU-only. That is deliberate: the default `make install` pulls no CUDA packages at all,
so CI and everyday fact-layer work stay fast. Budget ~16 GB RAM and ~120 GB disk for the
full AMLworld set; Elliptic2's background graph wants considerably more.

The GPU phases are narrower. The GAT encoder (Phase 7) trains comfortably on any modern
16 GB card. QLoRA finetuning of Llama-3.1-8B (Phase 9) needs **24 GB VRAM minimum** — a
4090 or A5000 will do it at batch size 1 with gradient checkpointing and 4-bit NF4
quantisation — and is **comfortable at 48 GB** (A6000, L40S), where you can raise the
effective batch size and stop fighting sequence length. Inference through vLLM fits in
24 GB.

```bash
make install-gpu   # adds graph + llm + human extras
```

**Pinned stack: torch 2.4.0 / CUDA 12.1 / PyTorch Geometric 2.6.1.** The PyG companion
packages are compiled against that exact pair, so install them from the matching index:

```bash
uv pip install --find-links https://data.pyg.org/whl/torch-2.4.0+cu121.html \
    torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3
```

Do not upgrade torch on its own — `torch-geometric`, the companion wheels, and `vllm`
all move with it. See `DECISIONS.md` D-005.

---

## Repository map

```
configs/          Hydra config groups. All paths live in configs/paths/.
schemas/          case_facts JSON Schema (Phase 3) + committed split manifests.
src/g2t_aml/
  data/           substrate ingestion and normalisation
  facts/          case_facts extraction — the measurement instrument, see invariant 1
  corpus/         Bronze / Silver / Gold narrative construction
  models/         GAT encoder, fusion layer, generator
  eval/           faithfulness + surface metrics
  human/          annotation and adjudication UI
  utils/          seeding, hashing, atomic IO, logging, run provenance
scripts/          one entrypoint per pipeline stage
tests/            unit / integration / golden
docs/             data cards, annotation protocol
artifacts/        run outputs — gitignored, append-only
data/             raw/interim/processed — gitignored
notebooks/        exploration only, never imported by src/
```

---

## Current status

| Phase | | Status |
|---|---|---|
| 0 | Scaffold, tooling, CI | **complete** |
| 1 | Data ingestion | not started |
| 2 | Frozen temporal splits | not started |
| 3 | Fact layer | not started |
| 4 | Bronze corpus | not started |
| 5 | Silver corpus | not started |
| 6 | Gold annotation | not started |
| 7 | GAT encoder | not started |
| 8 | Fusion layer | not started |
| 9 | Generator finetuning | not started |
| 10 | Evaluation | not started |
| 12 | Ablation matrix | not started |
| 14 | Release packaging | not started |

Details in `PHASE_LOG.md`.

---

## Working on this

Read **`CLAUDE.md`** first — it is the standing brief and carries the eight project
invariants that must never be violated. Two worth repeating here:

- **The fact layer is a measurement instrument.** A bug in `src/g2t_aml/facts/` silently
  corrupts every headline number in the paper.
- **No real-world PII or identifiers ever enter the repo**, including in test fixtures.
  Synthetic IDs only.

Non-obvious technical choices go in `DECISIONS.md` when you make them. Phase outcomes,
including what you deferred, go in `PHASE_LOG.md`. Failed experiments go in `RESULTS.md`
— negative results are kept and reported.

---

## Data access and licensing

Code is Apache-2.0 (`LICENSE`). The datasets are **not** redistributed here and carry
their own terms:

- **IBM AMLworld** — obtain from the authors' release (Altman et al., NeurIPS Datasets &
  Benchmarks 2023).
- **Elliptic2** — access-gated (Bellei et al., KDD Workshop on Machine Learning in
  Finance 2024). Requires a signed agreement. Not redistributable.

Place downloads under `data/raw/<substrate>/`, which is gitignored. No real SAR text and
no real financial identifiers exist anywhere in this repository.

## Citation

See `CITATION.cff`.
