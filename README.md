# Graph2Text AML

**Generating Suspicious Activity Report narratives from flagged transaction subgraphs.**

[![licence: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![data: CDLA-Sharing-1.0](https://img.shields.io/badge/corpus-CDLA--Sharing--1.0-orange.svg)](https://cdla.dev/sharing-1-0/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

Anti-money-laundering models tell an investigator *that* an account is suspicious. This
system tells them *why*, in prose they can use as the first draft of a Suspicious Activity
Report:

> *"This account received funds from three counterparties previously associated with
> illicit activity, then dispersed them to nine fresh accounts within 22 hours — a pattern
> consistent with layering."*

**Architecture:** GAT graph encoder → fusion/projection layer → QLoRA-finetuned
Llama-3.1-8B → structured narrative. **The fusion layer — how graph structure enters the
language model's embedding space — is the technical contribution.**

---

## ⚠ Read this before reading the results

**17 systems are declared in the experiment matrix. Zero have been trained.**

The pipeline is complete, tested and reproducible end to end. The corpus exists. The graph
encoder is trained. **But no generator arm has run, and Gate 8 — does the fusion layer beat
its own shuffled control? — is open.** The blockers are a 4 GB GPU against a model needing
≥24 GB, absent API credentials, and an unrecruited annotator. None is a software problem.

Every non-run is named with its blocker in **[`RESULTS.md`](RESULTS.md) §2**. Nothing in
this repository is estimated, projected or extrapolated. If you are evaluating this as a
deployable component, read **[`docs/ETHICS.md`](docs/ETHICS.md) §3.1** and stop there.

---

## Results

**The headline table is the one that exists.** These are measurements on real data by the
code in this repository, not projections.

### Bronze corpus faithfulness — 15,707 narratives, 296,196 claims

| Metric | Bronze (B1) | 95% CI |
|---|---:|---|
| **Zero-Hallucination Rate** | **1.0000** | [1.0000, 1.0000] |
| Fact Precision | 1.0000 | — |
| Hallucination Rate | 0.0000 | — |
| Fact Coverage | 0.8595 | [0.8592, 0.8598] |
| Fact F1 | 0.9243 | — |
| Critical Error Rate (H4+H6+H7) | 0.0000 | — |
| **H9 — omission of exculpatory fact** | **0.9179** | — |
| Narratives with no claims | **0** | asserted separately |

**The 1.0000 is a regression test on the measurement instrument, not an achievement.**
Bronze renders deterministically from the fact record and every formatter ships with its
inverse, so it is faithful by construction. `n_narratives_with_no_claims = 0` is what makes
the rest of the table mean anything — a perfect score over an empty claim set is what a
broken extractor produces.

**The real finding is H9 at 0.9179: 92% of template narratives omit a fact that weakens
the suspicion they describe.** So the template floor is *not* uniformly 1.0, and that is a
concrete dimension on which a trained system can beat it. Found by the harness, not by
inspection.

### Graph encoder — the only trained component

| Arm | Test AUC-PR |
|---|---|
| Graph transformer | 0.8877 ± 0.0190 |
| GIN | 0.8801 ± 0.0056 |
| **GATv2 (primary)** | **0.8720 ± 0.0136** |
| **MLP control — no message passing** | **0.8017 ± 0.0407** |
| GraphSAGE | 0.7861 ± 0.0101 |
| GCN | 0.7137 ± 0.0213 |

**The gate passed: GATv2 beats the MLP control by +0.070, excluding zero at all three
seeds. But the control reaches 0.80 with no message passing at all.** Every claim about
what graph structure contributes is a claim about that 0.07, not about 0.87. Two arms
out-score the primary and neither significantly; D-064 records why GATv2 stays primary.

Two ablations are **null and are reported as such** (invariant 7): positional encodings buy
+0.002, and focal loss is 0.006 *worse* than weighted BCE. Edge features are worth +0.114.

**Everything, including the nulls and the 25 named non-runs: [`RESULTS.md`](RESULTS.md).**

---

## Installation

Requires [`uv`](https://docs.astral.sh/uv/) and nothing else — it fetches Python 3.11
itself.

```bash
git clone https://github.com/MobsLInep/graph2text-aml.git
cd graph2text-aml

make install        # CPU-only: no torch, no CUDA. Everything below except training.
make install-stats  # + scipy/statsmodels. Still no torch. Required by `make smoke`.
make install-gpu    # adds the graph + llm + human extras and the PyG wheels (CUDA 12.1)
```

Or use the container — the base images are digest-pinned and the CPU image runs its own
quickstart at build time:

```bash
docker build -f docker/Dockerfile.cpu -t g2t-aml:cpu .
docker run --rm g2t-aml:cpu
```

**Pinned stack: Python 3.11 / torch 2.4.0 / CUDA 12.1 / PyTorch Geometric 2.6.1.** The
three PyG companion wheels are deliberately not in `uv.lock` — their sdists import torch at
build time, so they cannot be resolved into a lockfile (D-007). Do not upgrade torch on its
own; `torch-geometric`, the companion wheels and `vllm` all move with it (D-005).

---

## Quickstart — reproduce a published result in under a minute

**No data download. No GPU. No network after install. No credentials.**

```bash
make quickstart
```

```
staged 220 Bronze records from bronze_quickstart.jsonl.gz

  Zero-Hallucination Rate        1.0000
  Fact Precision                 1.0000
  Hallucination Rate             0.0000
  Fact Coverage                  0.8359
  Critical Error Rate            0.0000
  H9 omission of exculpatory fact  0.4500
  claims scored                  3734
  narratives with no claims      0

QUICKSTART OK — matches tests/golden/quickstart_evaluation.json exactly
```

It runs the **real** Phase 10 evaluation harness — the same `scripts/10_evaluate.py` the CI
gate runs, not a reimplementation — over a committed 220-record stratified slice of the
Bronze corpus, and asserts the result **exactly** against a committed golden file. Bronze is
deterministic, so any difference is a bug rather than variance.

The numbers differ from the full-corpus table above because the fixture weights the eleven
narrative families evenly rather than as they occur. That is expected and is recorded in
`tests/fixtures/NOTICE`.

Then the full development gate — lint, typecheck, the test suite and an end-to-end smoke
run:

```bash
make smoke
make help          # every target
```

---

## Full reproduction

**[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md)** is the complete guide: data acquisition
including the Elliptic2 access process, exact versions, every pipeline stage with its
command, runtime and hardware requirement, expected outputs and how to verify them, and —
importantly — **the tolerance policy**.

The short version:

| Tier | Needs | Time | Reproduces |
|---|---|---|---|
| **A — quickstart** | nothing | **< 1 min** | the Bronze evaluation gate on a committed fixture |
| **B — full CPU** | AMLworld raw data | **~30 min** | ingestion, splits, facts, the whole corpus, its evaluation |
| **C — GPU** | Tier B + a CUDA card | **~6.5 h** | the nine-arm encoder sweep and its gate |
| **D — blocked** | ≥24 GB VRAM, credentials, an annotator, Elliptic2 access | — | nothing yet |

```bash
make data      # ingest AMLworld HI-Small   ~14 s    -> reproduces every published figure exactly
make cases     # frozen temporal splits     ~4.5 min -> 10,932 / 2,028 / 3,196, audited
make facts     # case_facts extraction               -> 16,156 records
make bronze    # render + the ten-point gate ~90 s   -> 15,707 narratives
make eval-bronze                            # ~41 s  -> the CI gate
```

**On non-determinism:** exact reproduction of every digit is not achievable and we do not
claim it. The CPU stages are byte-exact and content-hashed. The GPU stages are not — scatter
atomics, cuDNN kernel selection and TF32 all move the digits, and a card newer than the
Turing development host will differ visibly. **What must reproduce is the conclusion**: the
encoder gate outcome, the signs of the ablation effects, and which results are null.
Tolerances are tabulated in `docs/REPRODUCTION.md` §6.

Verify the release yourself, against a pristine `git archive` export:

```bash
make verify-release           # nine checks: clean clone, install, quickstart, golden files,
                              # documented commands, --help, secret scan, no leaked data,
                              # licence NOTICEs
make verify-release-docker    # the same, inside a freshly built container
```

---

## Repository map

```
configs/          Hydra config groups. Every path lives in configs/paths/ — none is hardcoded.
schemas/          case_facts_v1.json + vocab_v1.yaml (FROZEN 1.0.0), training_record_v1.json
                  (FROZEN 1.0.0), and the committed temporal split manifests.
src/g2t_aml/
  data/           substrate ingestion, canonical representation, case extraction, splits
  facts/          THE MEASUREMENT INSTRUMENT (invariant 1). Runs both ways: extract_facts()
                  builds a checkable record, check_claim() verifies a narrative against it.
  corpus/         Bronze / Silver / Gold construction + the ten-point harness gating all three
  models/         encoder (trained) / fusion (built, untrained) / generator (built, never run)
  eval/           three-layer evaluation; Zero-Hallucination Rate is the headline (D-077)
  experiments/    the 17-system matrix, declared once and read by everything downstream
  human/          the Gold annotation kit and the Phase 12 decision-setting study
  utils/          atomic IO, canonical hashing, seeding, run provenance
scripts/          one entrypoint per pipeline stage, 01_ through 14_
tests/            unit / integration / golden, plus the independent oracle
docs/             REPRODUCTION · ETHICS · data_cards · dataset_cards · model_cards ·
                  annotation · human_study · deployability
docker/           CPU and GPU images, both digest-pinned
artifacts/        run outputs — gitignored, append-only, never overwritten
data/             raw / interim / processed — gitignored
notebooks/        exploration only; src/ never imports from here, and a test enforces it
```

**Start with [`CLAUDE.md`](CLAUDE.md)** — the standing brief. It carries the ten project
invariants and twenty-odd hard-won notes that will bite if forgotten. `DECISIONS.md`
records every non-obvious technical choice; `PHASE_LOG.md` records what each phase
delivered and deferred.

---

## Documentation

| Document | What it holds |
|---|---|
| [`RESULTS.md`](RESULTS.md) | **Every number, including the nulls and the 25 named non-runs** |
| [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) | Raw data to every published number; the tolerance policy |
| [`docs/ETHICS.md`](docs/ETHICS.md) | Intended use, misuse limits, measured failure rates, fairness, emissions |
| [`docs/model_cards/`](docs/model_cards/) | Per-model cards, each with an explicit misuse section — and what is *not* released |
| [`docs/dataset_cards/`](docs/dataset_cards/) | Bronze, Silver, Gold, `case_facts`. Two tiers are empty and say so first. |
| [`docs/data_cards/`](docs/data_cards/) | Provenance and licence per upstream substrate |
| [`docs/annotation/`](docs/annotation/) | The Gold protocol, the salience definition, the nine-class hallucination taxonomy |
| [`docs/deployability.md`](docs/deployability.md) | Per-system deployment assessment on the axes that need no GPU |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history, and the two frozen schema versions |

---

## Citation

```bibtex
@software{samaddar2026graph2text,
  author  = {Samaddar, Shreyansh},
  title   = {Graph2Text AML: graph-conditioned generation of Suspicious
             Activity Report narratives},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/MobsLInep/graph2text-aml}
}
```

Machine-readable metadata in [`CITATION.cff`](CITATION.cff). **Please also cite the
substrate** — Altman et al., NeurIPS Datasets & Benchmarks 2023.

---

## Licensing

**Two licences, two bundles, and the distinction is load-bearing rather than
bureaucratic.**

| What | Licence | Why |
|---|---|---|
| **Code, model weights, metrics, figures, `RESULTS.md`** | **Apache-2.0** | *Results* under CDLA-Sharing-1.0 §3.5, which imposes no obligations on the publication of Results |
| **Narrative corpus, `case_facts` records, case store** | **CDLA-Sharing-1.0** | **Enhanced Data.** The narratives quote AMLworld account identifiers, timestamps, currencies and amounts verbatim, and every record embeds its fact record — more than a de-minimis portion of the source Data |

`make release` packages the two separately, each with its licence text, its attribution
NOTICE and a SHA-256 manifest. **They must not be merged** (D-098). `make release-plan`
shows what each would contain without writing anything.

### The upstream substrates

- **IBM AMLworld** — **CDLA-Sharing-1.0**, *not* the Apache-2.0 of the `IBM/AML-Data`
  repository, which covers only the code. Share-alike, attribution preserved, changes
  marked, no added restrictions. Obtain it from the authors' Kaggle release (Altman et al.,
  NeurIPS D&B 2023).
- **Elliptic2** — access-gated (Bellei et al., KDD MLF 2024). **Its data licence could not
  be located**, so the substrate is treated as closed. **This project never obtained it**:
  there is no Elliptic2 result anywhere here. `scripts/14_reconstruct_elliptic2.py` is the
  path for rebuilding from *your own* licensed copy, and it reports honestly that there is
  currently nothing to reconstruct against.

Place downloads under `data/raw/<substrate>/`, which is gitignored.

**No real SAR text, no real financial data and no personal data of any kind exists anywhere
in this repository.** Real SARs are confidential by statute — that is precisely why the
corpus had to be constructed. The full git history and the release tree were scanned in
Phase 14: no secrets, no credentials, no identifiers. `make secret-scan` re-runs it.

---

## Acknowledgements

This work depends on two datasets that were released for public research, and on the
maintainers who made that possible:

- **Erik Altman, Jovan Blanuša, Luc von Niederhäusern, Béni Egressy, Andreea Anghel and
  Kubilay Atasu** (IBM Research), for AMLworld — released under CDLA-Sharing-1.0 with the
  per-stream typology labels that make a faithfulness metric possible at all. There is no
  version of this project without complete ground truth to measure against.
- **Claudio Bellei, Muhua Xu and colleagues** (Elliptic and the MIT-IBM Watson AI Lab), for
  Elliptic2 and for documenting its schema publicly enough that a loader could be written
  and tested against it before access was ever requested.

The evaluation machinery leans on published work it validates itself against rather than
trusting: Krippendorff's ordinal alpha against the canonical worked example (we reproduce
0.815 as 0.8154), Friedman and the rank correlations against `scipy`, and Nemenyi against
Demšar's table. Two implementations by one author agree on their shared misreading, which
is why every statistic here is checked against something outside this repository.

Built on `polars`, `hydra`, `pytorch`, `pytorch-geometric`, `transformers`, `peft`,
`streamlit` and `uv`.
