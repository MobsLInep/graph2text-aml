# Reproduction guide

**Every published number in this project, from raw data to the figure it lands in.**

Written Phase 14, 2026-08-06, for release `v0.1.0`.

---

## 0. Read this first

**Most of this project's declared experiments have not run.** 17 systems are declared; **0
have been trained.** What is reproducible is: the data pipeline, the frozen splits, the
fact layer, the 15,707-narrative Bronze corpus, the nine-arm encoder sweep, the Bronze
evaluation, and the two-system efficiency benchmark. Everything else in `RESULTS.md` is a
named non-run with a named blocker.

**Do not read a failure to reproduce an untrained arm as a reproduction failure.** §8 lists
exactly which numbers exist.

### The three tiers of what you can reproduce

| Tier | Needs | Time | Reproduces |
|---|---|---|---|
| **A — Quickstart** | Nothing but the repo | **< 15 min** | The Bronze evaluation gate on a committed fixture |
| **B — Full CPU** | AMLworld raw data (~20 GB download, 476 MB used) | **~30 min** | Ingestion, splits, facts, the whole Bronze corpus, its evaluation, efficiency |
| **C — GPU** | Tier B + a CUDA card | **~6.5 h** (encoder) | The nine-arm encoder sweep and Gate 7 |
| **D — Blocked** | ≥24 GB VRAM, API credentials, an annotator, Elliptic2 access | — | **Nothing yet. Nobody has reproduced these because nobody has produced them.** |

---

## 1. Environment

### 1.1 Exact versions

| | |
|---|---|
| Python | **3.11** (3.11.14 on the development host) |
| Package manager | **uv 0.10.2** — fetches Python itself; nothing else is required |
| Lockfile | `uv.lock`, fully pinned. **Always install with `--frozen`.** |
| torch | **2.4.0+cu121** (GPU tiers only) |
| PyTorch Geometric | **2.6.1** |
| PyG companion wheels | torch-scatter 2.1.2, torch-sparse 0.6.18, torch-cluster 1.6.3 |
| CUDA | **12.1** |
| OS on the development host | Linux 7.0.0-28-generic |

### 1.2 Install

```bash
git clone <repo-url> graph2text-aml && cd graph2text-aml

make install        # CPU-only: no torch, no CUDA. Tiers A and B.
make install-stats  # + scipy/statsmodels/krippendorff. Still no torch. Needed for `make smoke`.
make install-gpu    # adds eval + graph + llm + human extras and the PyG wheels. Tiers C, D.
```

**`make smoke` needs `--extra stats`.** The statistics in `eval/` and `experiments/` use
scipy, and it is deliberately *not* in the base install. It is also deliberately not taken
from the `eval` extra, which carries `bert-score` and therefore torch — the point of the
light environment is that it has none. CI and the CPU image both install `stats`.

**The three PyG companion wheels are deliberately not in `uv.lock`.** Their PyPI sdists
import torch at build time, so they cannot be resolved into a lockfile at all, and building
from source would compile against whatever CUDA the build host happens to have.
`make install-pyg` installs the prebuilt wheels for torch 2.4.0+cu121 exactly (D-007).

**Do not upgrade torch on its own.** `torch-geometric`, the three companion wheels and
`vllm` all move with it (D-005).

### 1.3 Or use the container

```bash
docker build -f docker/Dockerfile.cpu -t g2t-aml:cpu .     # tiers A and B
docker build -f docker/Dockerfile.gpu -t g2t-aml:gpu .     # tiers C and D
docker run --rm g2t-aml:cpu make smoke
```

The CPU image is the one CI verifies on a schedule (`scripts/14_verify_release.py`). See
[`docker/README.md`](../docker/README.md).

### 1.4 Point the tree elsewhere

**No path is hardcoded anywhere in the codebase.** Every directory root lives in
`configs/paths/default.yaml` and is reached as `cfg.paths.*`:

```bash
G2T_AML_ROOT=/scratch/g2t uv run python scripts/01_ingest.py
```

---

## 2. Data acquisition

### 2.1 IBM AMLworld HI-Small — open, and this is the one you need

Distributed through Kaggle, so the download is a **deliberate manual step**.

```bash
# 1. Kaggle API token: Account -> Settings -> API -> Create New Token
#    save to ~/.kaggle/kaggle.json, mode 600
# 2.
uv run pip install kaggle
# 3.
kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml \
    -p data/raw/amlworld --unzip
# 4. verify
uv run python scripts/01_ingest.py     # checksums are verified automatically
```

The full release is ~20 GB across six variants. **Only two files are needed:**

| File | Size | SHA-256 (pinned) |
|---|---:|---|
| `HI-Small_Trans.csv` | 475,664,283 B | `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040` |
| `HI-Small_Patterns.txt` | 323,843 B | `7636c1d712168139ba0ff90f1b45aac9888d0ad46387084560886f204c03d6e6` |

Both belong in `data/raw/amlworld/` (gitignored). They may be symlinked in — symlinks are
resolved before hashing — but that directory is the canonical location every config and
test assumes.

**Size is checked before hashing, so a truncated download fails fast.** A checksum mismatch
raises *before* an absence is reported, because a wrong file is the more alarming finding.

**Licence: CDLA-Sharing-1.0** — not the Apache-2.0 of the `IBM/AML-Data` repository, which
covers only the code. This governs what you may redistribute; see `README.md` § Licensing.

### 2.2 Elliptic2 — access-gated, and we do not have it

> **This project has never obtained Elliptic2. Access was not requested. No Elliptic2
> number exists anywhere in this repository, and none can be reproduced.**

The loader is written against the documented schema and tested against a synthetic tree;
its real-data tests skip. `scripts/01_ingest.py data=elliptic2` exits **0** with
`ingest_skipped.json` rather than failing.

If you hold a licensed copy:

```bash
# 1. Request access:  https://www.elliptic.co/elliptic2
# 2. Official tooling: https://github.com/MITIBMxGraph/Elliptic2
# 3. Unzip so data/raw/elliptic2/ contains:
#      background_edges.csv  background_nodes.csv  connected_components.csv
#      edges.csv  nodes.csv
# 4.
make data-elliptic2
```

**No checksums are pinned for Elliptic2 — we have never seen the files.** `verify()` reports
`MISSING`. Pin them on first ingest.

**Column names are probed, never guessed by position.** The official tooling has used more
than one spelling across releases, so the loader probes a candidate list per role
(`clusterId`/`cluster_id`/`nodeId`/…, `ccId`/`componentId`/…, `ccLabel`/`label`/…) and
raises `Elliptic2SchemaError` — *"Refusing to guess by position"* — rather than silently
reading the wrong column.

**Licence status: unresolved, treated as closed.** The tooling repository is Apache-2.0 but
says nothing about the dataset; the terms presumably arrive with the access grant. Nothing
Elliptic2-derived may be redistributed from here, and there is nothing to redistribute. See
`docs/data_cards/elliptic2.md` §2 and the reconstruction path in §7 below.

---

## 3. Tier A — the quickstart, under 15 minutes, no data download

**This is what a stranger runs from a clean clone to see the project work.**

```bash
make install        # ~2-4 min, network-bound
make quickstart     # ~2 s
```

It scores the committed **220-record** Bronze fixture with the real Phase 10 evaluation
harness — the same `scripts/10_evaluate.py` the CI gate runs — and asserts the result
against a committed golden file. No GPU, no network after install, no dataset download.

**Expected output**

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

The fixture is a stratified slice — 20 records from each of the eleven narrative families —
so **its numbers differ from the full-corpus numbers by construction**: Fact Coverage reads
0.8359 here against 0.8595 over all 15,707 records, and H9 reads 0.45 against 0.9179. Never
quote one as the other.

**Tolerance: exact.** Bronze is deterministic and the fixture is committed. Any difference
is a failure, not variance.

To also run the full gate (lint + typecheck + 1,000+ tests + an end-to-end smoke run):

```bash
make install-stats  # scipy/statsmodels; `make smoke` needs them
make smoke          # ~5-8 min
```

---

## 4. Tier B — the full CPU pipeline

Run in this order; each stage depends on the last.

| # | Command | Runtime | Hardware | Writes |
|---|---|---|---|---|
| 1 | `make data` | **~14 s** | 8 GB RAM | `data/interim/amlworld_hi_small/` |
| 2 | `make cases` | **~4.5 min** | 16 GB RAM | `data/processed/.../cases/`, `schemas/splits/amlworld/` |
| 3 | `make audit` | ~20 s | any | `leakage_audit.json` |
| 4 | `make facts` | ~2 min | 8 GB RAM | `facts.parquet`, `data/processed/.../facts/` |
| 5 | `make facts-gate` | ~1 min | any | — (tests) |
| 6 | `make bronze` | **~90 s** | 8 GB RAM | `corpus/bronze.jsonl` + 3 reports |
| 7 | `make eval-bronze` | **~41 s** | any | `artifacts/metrics/eval/<run>/` |
| 8 | `make benchmark` | ~10 min | idle machine | `artifacts/metrics/phase13/` |

Budget **~120 GB disk** for the full AMLworld set, or ~2 GB if you keep only the two
HI-Small files. Peak RAM is at stage 2.

### 4.1 What each stage should produce

**Stage 1 — ingestion.** Reproduces every published AMLworld figure **exactly**:

| Quantity | Expected |
|---|---:|
| Vertices | 515,088 |
| Edges | 5,078,345 |
| fan_out / fan_in / gather_scatter / scatter_gather | 342 / 318 / 716 / 626 |
| cycle / random / bipartite / stack | 287 / 191 / 263 / 466 |
| not classified | 1,968 |

Verify with `uv run pytest tests/integration/test_published_statistics.py` (11 assertions,
~16 s). `manifest.json` carries a SHA-256 per artifact; `edges.parquet` should be
180,656,462 B with digest `2a17994096…`.

**Stage 2 — splits.** **10,932 / 2,028 / 3,196**, temporally disjoint, 13,844 cases dropped
(9,718 straddling, 4,110 in the 6-hour buffer, 16 stream-assigned). The manifests are
already committed — **a rebuild that changes a content hash is a failure**, not variance.

**Stage 3 — leakage audit.** All three fatal checks pass: temporal ordering, stream
atomicity (211 streams, 0 straddling), label leakage. Edge overlap **0.000**; node overlap
**0.538** and reported, not suppressed (see `docs/dataset_cards/case_facts.md` §4).

**Stage 4–5 — facts.** 16,156 records at `case_facts` 1.0.0. The gate runs the 1,000-case
round trip **and** the independent oracle. **Both must pass**: the round trip alone cannot
catch an extractor bug — three injected bugs left it at 100% SUPPORTED (D-034).

**Stage 6 — Bronze.** 15,707 records (train 10,488 / val 2,027 / test 3,192). The
ten-point harness must report `gate_passed: true`, `pass_rate: 1.0`, **0 failures on all
ten checks**, and 296,195 supported claims.

**Stage 7 — evaluation.** The headline table:

| Metric | Expected |
|---|---:|
| **Zero-Hallucination Rate** | **1.0000** |
| Fact Coverage | 0.8595 |
| Fact F1 | 0.9243 |
| Claims scored | 296,196 |
| **Narratives with no claims** | **0** |
| H9 (omission of exculpatory fact) | 0.9179 |

**Check `n_narratives_with_no_claims = 0` before believing any other row.** A perfect score
over an empty claim set is what a broken extractor produces.

The 296,196 here against 296,195 at stage 6 is **expected**: two extractors reaching the
claims by different routes, agreeing to one claim in 296 thousand.

**Stage 8 — efficiency.** See §6 — this is the one stage with a wide tolerance.

---

## 5. Tier C — the encoder sweep (GPU)

```bash
make install-gpu
make encoder-features    # build the feature cache from the frozen splits
make train-encoder       # 9 arms x 3 seeds
make encoder-gate        # the tests
uv run python scripts/07c_report_tables.py
```

| | |
|---|---|
| Runtime | **~6.5 h** on 1 × RTX 2050 (4 GB) — the development host |
| VRAM | < 4 GB. Any modern card is comfortable. |
| Writes | `artifacts/checkpoints/encoder/`, `artifacts/metrics/encoder/encoder_report.json` |

### Expected

| Arm | Test AUC-PR |
|---|---|
| `graph_transformer` | 0.8877 ± 0.0190 |
| `gin` | 0.8801 ± 0.0056 |
| `gatv2_bce` | 0.8775 ± 0.0106 |
| **`gatv2` (primary)** | **0.8720 ± 0.0136** |
| `gatv2_no_pe` | 0.8696 ± 0.0049 |
| **`mlp` (control)** | **0.8017 ± 0.0407** |
| `sage` | 0.7861 ± 0.0101 |
| `gatv2_no_edge` | 0.7582 ± 0.0236 |
| `gcn` | 0.7137 ± 0.0213 |

**The claim that must reproduce is the gate: GATv2 beats the MLP control by ≈ +0.070,
excluding zero at all three seeds.** The individual digits will not match — see §6.

Then write the scores back into the fact records:

```bash
make score-cases
```

**Do not regenerate Bronze after this.** `facts.serialiser._compact` emits
`gnn_risk_score` into `serialised_facts`, which is the input to the *"no graph encoder"*
ablation arm — regenerating would push the encoder's own score into that baseline and
**nothing would fail** (D-063).

---

## 6. Known non-determinism, and the tolerance policy

**Exact reproduction of every digit is not achievable, and this section says where and by
how much.** Anyone who reports matching to the fourth decimal on the GPU stages has either
been lucky or is not measuring what they think.

### 6.1 The policy

| Stage | Tolerance | Rationale |
|---|---|---|
| Ingestion, splits, facts, Bronze | **Exact.** Byte-identical; content hashes must match. | Fully deterministic. Any difference is a bug. |
| Bronze evaluation (Layer 2 faithfulness) | **Exact** on rates; **± 0.0001** on bootstrap CI bounds | Deterministic scoring; the CI resamples under a fixed seed. |
| Encoder AUC-PR / AUC-ROC, per seed | **± 0.02 absolute** | cuDNN kernel selection, atomics, TF32. |
| Encoder AUC-PR, 3-seed mean | **± 0.01 absolute** | Averaging absorbs part of it. |
| **Encoder gate outcome** (GATv2 − MLP > 0 at every seed) | **Must reproduce exactly** | This is the claim. If it flips, that is a finding, not a tolerance breach. |
| Ablation *signs* (edge features positive; PE and focal-loss nulls) | **Must reproduce** | Ditto. |
| Latency p50 / p95 / p99 | **± 2×** on a laptop; **± 20%** on a dedicated idle host | See §6.3. |
| VRAM, parameter counts, model size on disk | **Exact** | Not stochastic. |
| Cost estimates | **Not reproducible.** Inputs are list prices at 2026-08. | Recompute from `CostAssumptions`. |

### 6.2 Sources of non-determinism, named

1. **GPU floating-point non-associativity.** Atomics in scatter/gather reductions — which
   is every message-passing layer — sum in a nondeterministic order. This is the dominant
   source and it is not fixable by seeding.
2. **cuDNN algorithm selection** varies with card, driver and available memory. A different
   card is a different sequence of kernels.
3. **TF32 on Ampere and later.** The development host is Turing (RTX 2050) and does **not**
   use TF32. **An A100 or a 4090 will produce visibly different digits from the published
   table for this reason alone.**
4. **`torch.use_deterministic_algorithms` is not enabled**, because several PyG scatter
   ops have no deterministic implementation and enabling it raises rather than slowing
   down. Making it deterministic would mean changing the model.
5. **Library versions.** Pin torch 2.4.0+cu121 and PyG 2.6.1 exactly. Different minor
   versions change kernel dispatch.

Seeds *are* set and recorded (`utils/seeding.py`, `run_context.json`), and they remove
data-order and initialisation variance. **They do not remove 1–3.**

### 6.3 The host is a laptop, and the latency numbers say so

Phase 13's between-run spread on the development host was **2×**: B1's p50 varied
**5.4–11.2 ms across seven full protocol runs** on one afternoon. One run taken while the
test suite executed concurrently showed p95 inflated to 30 ms and index build inflated 3×,
and was discarded and rerun idle.

**The published range is reported rather than averaged away, and no Phase 13 conclusion
rests on a difference smaller than 2×.** Reproducing latency within 2× on comparable
hardware is a pass. Anything tighter is not claimed.

### 6.4 What we assert instead of digits

The reproducibility claim this project makes is:

- **Deterministic stages reproduce exactly** — and they are checked by content hash, not by
  eye.
- **Stochastic stages reproduce their *conclusions*** — gate outcomes, signs of effects,
  which results are null — within the tolerances above.
- **Every run is traceable.** `run_context.json` in every run directory carries the git
  SHA, resolved config, data manifest hash, all seeds and library versions (invariant 5),
  and nothing is ever overwritten (invariant 6).

---

## 7. Tier D — what cannot be reproduced, because it was never produced

| Number | Blocker | What unblocks it |
|---|---|---|
| **Gate 8 — does S1 beat A1?** | **GPU ≥24 GB.** Llama-3.1-8B at nf4 is 4.5–5.6 GB before a single activation; the dev host has a 4 GB card and 7 GB RAM, which also closes CPU offload (D-068). | Rented compute. **Run S1 and A1 first**, then `make gate8`. If it comes out null, the reframing decision must be made before more compute is spent. |
| 12 systems / 20 runs in the matrix | same | same. `make matrix-plan` prints the plan without touching a GPU. |
| Silver corpus; systems B3, B4, B5 | **Teacher-API credentials.** 0 calls ever made. | Spend authorisation. `make silver-dry-run` first. |
| Method A / Method B extractor agreement κ | same | The number that makes the faithfulness metric defensible rather than merely defined. |
| Gold corpus; every Layer 1 overlap metric | **No annotator recruited.** 0 narratives. | Recruitment. No engineering advances it. |
| The Phase 12 decision-setting study | **Ethics approval not applied for** (4–8 weeks); only Bronze has generations, so a five-arm study has one arm; no rater. | `docs/human_study/` holds the application, ready to submit. |
| **Everything on Elliptic2** | **Access never requested.** | §7.1. |

### 7.1 The Elliptic2 reconstruction path

Elliptic2 is not redistributable and we hold nothing to redistribute, so the release ships
a **reconstruction script** rather than data:

```bash
uv run python scripts/14_reconstruct_elliptic2.py --raw data/raw/elliptic2 --out data/interim
```

It rebuilds the derived artifacts from **your own licensed copy** — never from ours. The
release bundle would normally carry case IDs, fact records and narratives so the script has
something to rebuild against; **for Elliptic2 those are empty, because the substrate was
never ingested.** The script therefore exits with a clear message and the manifest records
zero cases. It is shipped now so the path is in place, tested and documented the day access
arrives.

---

## 8. Index — which number lives where

| Number | Source of truth | Regenerate with |
|---|---|---|
| AMLworld published statistics | `data/interim/.../statistics.json` | `make data` |
| Split counts, leakage audit | `schemas/splits/amlworld/`, `leakage_audit.json` | `make cases`, `make audit` |
| Bronze ten-point gate | `corpus/bronze_validation.json` | `make bronze` |
| Bronze diversity, self-BLEU curve | `corpus/bronze_diversity.json` | `make bronze` |
| Bronze faithfulness, H9, coverage-by-typology | `artifacts/metrics/eval/<run>/` | `make eval-bronze` |
| Encoder arms, gate, ablations, probes | `artifacts/metrics/encoder/encoder_report.json` | `make train-encoder` → `scripts/07c_report_tables.py` |
| Efficiency, latency, VRAM, cost | `artifacts/metrics/phase13/` | `make benchmark` |
| The matrix and its 25 non-runs | `artifacts/metrics/phase11/missing_runs.md` | `make aggregate` |
| **Everything, including the nulls** | **`RESULTS.md`** | — |

---

## 9. If it does not reproduce

1. **Check the tier.** §7 lists what was never produced. That is the most common cause.
2. **Check `run_context.json`** in the run directory against yours — git SHA, config hash,
   data manifest hash, seeds, library versions. A differing data manifest hash means you
   are not running on the same data.
3. **Check the tolerance in §6.1** before treating a difference as a failure — and check
   whether your card uses TF32 (§6.2.3).
4. **Run `scripts/14_verify_release.py`**, which reproduces the Tier A path in a clean
   container against committed golden files and reports precisely which step diverged.
5. If a **deterministic** stage differs, that is a bug and we want to hear about it. Open an
   issue with your `run_context.json`.
