# Results

**Every number this project has produced, including the failed and the null ones**
(invariant 7). This file is both a reproducibility artifact and a hedge: if a reviewer
asks "did you try X", the answer is a link into this document.

**Status as of 2026-10-28: the first generator arms have trained, on a 16 GB RTX A4000.
The Gate 8 pilot (seed 42) answers "indistinguishable".** S1, A1 and B7 are trained at seed
42; S1, A1, B1 and B2 are scored on the 3,192-case test split (section 4.9). The rest of the
GPU matrix is queued and running (`PROGRESS.md`). The API arms, Silver, Gold and Elliptic2
are unchanged. Nothing here is estimated, projected, extrapolated or filled in. Every
result from a trained arm carries its regime: Bronze-only, one epoch, batch 1 x
accumulation 32 (D-104, D-105).

- Last updated: 2026-11-17 (matrix stopped by the user; §4.10)
- Schema: `case_facts` 1.0.0 (frozen), `training_record` 1.0.0 (frozen)
- Substrate: AMLworld HI-Small. Elliptic2 access has still not been requested.

---

## 1. How to read this file

| Convention | Meaning |
|---|---|
| A number | Measured, on the stated data, by the stated code, at the stated commit. |
| **NOT RUN** | The run is declared in `src/g2t_aml/experiments/registry.py` and has not executed. The blocker is named. |
| **NOT MEASURED** | The machinery exists and is tested; the measurement needs an input that does not exist. |
| † | Single seed. No variance estimate exists, and none is implied. |
| — | No number. Never a zero. |

**A dash and a zero are different claims** and are never interchanged, in this file, in
the LaTeX tables or in the figures.

---

## 2. The experiment matrix, and what has run

17 systems, 25 runs (the seed asymmetry in §3 makes the run count exceed the system
count). **Runs completed: 4** (B1, B2, S1 seed 42, A1 seed 42); B7 seed 42 is trained and
awaiting its decode. Status as of 2026-10-28.

| # | System | Role | Resource | Seeds | Status | Blocker |
|---|---|---|---|---:|---|---|
| B1 | Bronze template | Faithfulness ceiling | CPU | 1 | **DONE** (§4.9) | — |
| B2 | GATv2 classifier + template | Plan's baseline (a) | CPU | 1 | **DONE** (§4.9) | — |
| B3 | Frontier zero-shot | Plan's baseline (b) | API | 1 | **NOT RUN** | no API credentials |
| B4 | Frontier few-shot (k=5) | Stronger (b) | API | 1 | **NOT RUN** | no API credentials |
| B5 | Frontier + agentic verify loop | Closest existing competitor | API | 1 | **NOT RUN** | no API credentials |
| B6 | Llama-3.1-8B zero-shot | Untuned floor | GPU | 1 | queued | first decode pass (D-117) |
| B7 | Llama-3.1-8B QLoRA, serialised | **The threatening baseline** | GPU | 3 | seed 42 trained; 1337, 2024 queued | GPU time |
| B8 | G-Retriever-style (F1) | Standard graph-LLM recipe | GPU | 1 | queued | GPU time |
| S1 | GAT + F2 + text | Full system | GPU | 3 | **seed 42 DONE** (§4.9); 1337, 2024 queued | GPU time |
| S2 | GAT + F2, graph only | **Headline** | GPU | 3 | seed 42 training (first attempt refused by its overfit check, D-118) | GPU time |
| A1 | S1, deranged graph tokens | **Sanity control** | GPU | 3 | **seed 42 DONE** (§4.9); 1337, 2024 queued | GPU time |
| A2 | S1 with MLP encoder | Is topology needed? | GPU | 1 | queued | GPU time |
| A3_F3 | S1 with linear projector | Fusion ablation | GPU | 1 | queued | GPU time |
| A3_F4 | S1 with perceiver projector | Fusion ablation | GPU | 1 | queued | GPU time |
| A4 | S1, encoder unfrozen | Joint-training ablation | GPU | 1 | queued | GPU time |
| A5 | S1 decoded **with the guard on** (D-116) | Guard contribution | GPU (inference) | 1 | queued | S1 at all three seeds |
| A6 | S1 on Qwen3-8B | Generality | GPU | 1 | **NOT RUN** | pinned `transformers==4.45.2` cannot load Qwen3; substitute proposed, awaiting the user (D-115) |

The A3 fusion ablation's **F1 point is B8**, which is S1 with the gate off and nothing
else changed. Running a third arm with those settings under a different name would spend
GPU-days to produce a duplicate row (D-082).

The machine-readable version of this table, regenerated on every aggregation run, is
`artifacts/metrics/phase11/missing_runs.md`.

### 2.1 The blockers, in order of lead time

| Blocker | Blocks | What unblocks it |
|---|---|---|
| **GPU time** | the queued GPU rows | Resolved for hardware: an RTX A4000 (16 GB) runs every 8B arm at batch 1 x 32 (D-105). Cost on that card: ~4 h overfit check + ~20 h training + ~13 h greedy test decode per arm-seed. The queue (`PROGRESS.md`) runs them in the order D-114 sets. |
| **Teacher-API credentials** | B3, B4, B5; Silver; the Method A/B agreement κ | Spend authorisation. |
| **Silver corpus** | Epochs 2–3 of every trained arm's curriculum | The credentials above. Currently **0** verified records. |
| **Gold corpus** | Every Layer 1 overlap metric; the held-out human reference | An annotator. Currently **0** narratives. Recruitment is the critical path and no further engineering advances it. |
| **Elliptic2 access** | Every second-substrate result | An access request, still not made. Longest-lead open item. |

---

## 3. The seed policy, stated rather than hidden

The full matrix at three seeds is roughly 3–5 GPU-weeks, which fits no schedule this
project has. The adopted policy, recorded in D-081 and enforced by
`registry.validate_registry`:

- **3 seeds** — S1, S2, A1, B7. The systems carrying the central claim.
- **1 seed** — everything else.
- **Extension order if compute frees up** — A2, then B8.

Single-seed rows are marked with a dagger in every table and with a hollow marker in every
figure. `SeedSummary.std` is `None` at one seed by construction, so a single-seed row
cannot print `0.0000` and be misread as a zero-variance result.

**This asymmetry is reported in the paper.** The alternative to it is not symmetry; it is
a smaller matrix — dropping ablations to afford variance estimates on arms nobody disputes.

---

## 4. What HAS been measured

These are real numbers on real data. None of them is a result about the method; all of
them are results about the corpus and the measurement instrument.

### 4.1 Bronze corpus, scored by the Phase 10 harness

All **15,707** narratives, 41 s, no GPU, no network. `make eval-bronze` is the CI gate.

| Metric | Bronze | 95% CI |
|---|---:|---|
| **Zero-Hallucination Rate** | **1.0000** | [1.0000, 1.0000] |
| Fact Precision | 1.0000 | — |
| Hallucination Rate | 0.0000 | — |
| Unverifiable Rate | 0.0000 | — |
| Fact Coverage | 0.8595 | [0.8592, 0.8598] |
| Fact F1 | 0.9243 | — |
| Numeric Accuracy | 1.0000 | — |
| Typology Accuracy | 1.0000 | — |
| Ordering Accuracy | 1.0000 | — |
| Critical Error Rate | 0.0000 | — |
| Claims scored | 296,196 | all SUPPORTED |
| Narratives with no claims | **0** | asserted separately |

Bronze is faithful **by construction** — it renders from the record and every formatter
ships with its inverse — so this is a regression test on the harness, not a result about
Bronze. Phase 4's independent harness counted 296,195 claims over the same corpus:
agreement to one claim in 296 thousand from two extractors that reach the claims by
different routes. `n_narratives_with_no_claims = 0` is what makes the rest of the table
mean anything; a perfect score over an empty claim set is what a broken extractor produces.

### 4.2 The one genuine finding available without a trained model

| Class | Bronze, per-narrative rate |
|---|---:|
| H1–H8 | 0.0000 |
| **H9 — omission of exculpatory fact** | **0.9179** |

Bronze omits an exculpatory fact in 92% of its narratives. The templates report
`labels.n_counterparties` and `labels.n_illicit_counterparties` but never the licit count,
never `labels.focal_is_illicit`, and mention `temporal.burst_detected` only when a burst
was detected — so a case whose subject carries no illicit label produces a narrative that
never says so. Found by the harness, not by inspection.

**The template floor is therefore not uniformly 1.0**, and H9 is a concrete dimension on
which a trained system can beat it.

### 4.3 Bronze Fact Coverage by typology

The spread is not noise: structurally richer typologies have longer salience lists and
Bronze's templates do not grow to match them.

| Typology | Coverage |
|---|---:|
| unclassified | 0.862 |
| random | 0.848 |
| fan_in | 0.843 |
| fan_out | 0.841 |
| gather_scatter | 0.834 |
| scatter_gather | 0.824 |
| stack | 0.824 |
| cycle | 0.784 |
| bipartite | 0.770 |

**bipartite and cycle are where a generator has the most room.**

### 4.4 Bronze surface statistics

| | Bronze |
|---|---:|
| distinct-1 | 0.0137 |
| distinct-2 | 0.0352 |
| self-BLEU @ 5 references | 0.4752 |
| length, words | 146.6 ± 11.2 |

All three reproduce Phase 4's independently-computed diversity report to six decimal
places. Self-BLEU without its reference count is not a number (D-043): on this corpus it
reads 0.16 at one reference and 0.82 at fifty.

### 4.5 Phase 7 encoder — the only trained component in the project

| Arm | Test AUC-PR | Note |
|---|---:|---|
| Graph transformer | 0.8877 ± 0.0190 | out-scores GATv2, not significantly |
| GIN | 0.8801 ± 0.0056 | out-scores GATv2, not significantly |
| GATv2 + weighted BCE (ablation) | 0.8775 ± 0.0106 | out-scores GATv2, not significantly |
| **GATv2 (primary)** | **0.8720 ± 0.0136** | the arm every fusion result uses |
| GATv2 without positional encodings | 0.8696 ± 0.0049 | the PE null, see §4.6 |
| **MLP control (no message passing)** | **0.8017 ± 0.0407** | a DeepSets model over node-local features |
| GraphSAGE | 0.7861 ± 0.0101 | |
| GATv2 without edge features | 0.7582 ± 0.0236 | the one large ablation, see §4.6 |
| GCN | 0.7137 ± 0.0213 | |

**The gate passed: GATv2 beats the MLP control by +0.070 AUC-PR, excluding zero at all
three seeds.** But the control reaches 0.80 on its own. **Every claim about what graph
structure contributes is a claim about that 0.07**, not about the whole number. At two
epochs the gap read 0.16, which is what an under-tuned control would have reported.

GATv2 stays primary despite being out-scored; D-064 records the evidence and what would
change it.

### 4.6 Encoder ablations — two null results, kept and reported

| Ablation | Effect on AUC-PR | Significant? | Decision |
|---|---:|---|---|
| Edge features | **+0.114** | yes | kept |
| Positional encodings | +0.002 | **no** | retained anyway, D-066 |
| Focal loss vs weighted BCE | **−0.006** | **no** | focal kept as default, D-065 |

Both nulls are reported rather than dropped.

### 4.7 The Phase 8 forecast, and its caution

A linear probe on the pooled tokens Phase 8 consumes reaches **0.33 structural macro-F1**.
`fan_out`, `gather_scatter` and `cycle` are recoverable from those tokens; **`stack` and
`random` are not.** Read S2's per-typology breakdown before its mean.

---

### 4.8 Efficiency and deployability (Phase 13)

Measured on an NVIDIA RTX 2050 (4 GB), i5-12450H, 7 GB RAM, torch 2.4.0+cu121, driver
595.84. Protocol: 20 runs discarded, 100 measured, nearest-rank percentiles, seeded draw
from the frozen test split. **2 of 17 systems measured**; the other 15 are rows with named
blockers. `make benchmark` regenerates it.

| | B1 (template) | B2 (GATv2 + template) |
|---|---:|---:|
| On-premise / data leaving | ✓ / nothing | ✓ / nothing |
| Total / trainable params | 0 / 0 | 628,058 / 628,058 |
| Model size on disk | 0 | 2.53 MB |
| Peak VRAM, inference (reserved) | n/a | **0.025 GB** |
| End-to-end p50 / p95 / p99 | **5.4 / 22.1 / 31.5 ms** | **15.8 / 24.5 / 28.9 ms** |
| Throughput (batch 1 / 32-queue) | 123.9 / 206.7 narr/s | 58.9 / — narr/s |
| Cold start | 0.92 s | 7.65 s |
| Cost / 1,000, amortised local | USD 0.00038 | USD 0.00080 |

Latency by case size (40 runs per band, stratified), p95: B1 5.0 → 16.1 → 19.5 ms across
the 0–24, 25–49 and 50–99 node bands; B2 30.2 → 32.2 → 36.8 ms. Roughly 2x from the
smallest band to the largest. **No test-split case reaches 100 nodes**, so the top band is
empty and the trend must not be extrapolated.

**The host is a laptop and the between-run spread is 2x.** B1's p50 varied 5.4–11.2 ms
across seven full protocol runs; a run taken while the test suite was executing was
discarded and rerun idle. No conclusion above rests on a difference smaller than 2x.

API baselines, **cost estimated** from published list pricing (2026-08) and token counts
measured from this corpus (569 prompt / 188 completion): B3, B4 USD 22.65 per 1,000;
B5 USD 67.94 per 1,000 at three calls per narrative. These are estimates and are labelled
`api_marginal`, never as measurements of those systems.

Three things in this table are load-bearing and are easy to misread:

- **Case extraction is 72% of B1's end-to-end latency and generation is 18%.** Cutting the
  subgraph costs more than writing the narrative. Every row pays it, including the 8B arms.
- **The guard figure is verification only.** 1.90 ms for four candidates, model-independent,
  taken from the stage mean rather than from differencing two runs. The four generations the
  guard also requests are excluded, so any overhead ratio computed for B1 is a fact about B1
  and not the guard's overhead in general (D-097).
- **The cost columns are different kinds of number.** Local is amortised capital, API is
  marginal. The assumptions are declared in D-093; the breakeven against the frontier API
  is ≈ 2,300 narratives/month, and that is a cost comparison and not a quality one.

Full assessment, including hardware sizing and the regulatory position, in
`docs/deployability.md`.

### 4.9 The first trained arms (2026-10-05 → 2026-10-15)

**Regime, stated with every number below:** one seed (42); Bronze only, one epoch (Silver
has no records, D-104); batch 1 x accumulation 32 with `sdpa` attention on a 16 GB card
(D-105); greedy, unguarded decoding of the full 3,192-case test split. B1 and B2 are
deterministic. Phase 10 evaluator; reports in `artifacts/metrics/eval/19-50-13_debug/`
(S1, A1) and `artifacts/metrics/eval/05-54-44_debug/` (B1, B2), and per run in
`artifacts/matrix/<system>/seed42/*/metrics.json`.

| System | Zero-Halluc. [95% CI] | Precision | Halluc. rate | Unverif. | Coverage | Fact F1 | Critical | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B1 template† | **1.000** | 1.000 | 0.000 | 0.000 | 0.853 | 0.920 | 0 | 3,192 |
| B2 classifier + template† | **1.000** | 0.903 | 0.000 | 0.097 | 0.853 | 0.877 | 0 | 3,192 |
| S1† | **0.979** [0.974, 0.984] | 0.985 | 0.001 | 0.013 | 0.766 | 0.859 | 0 | 3,192 |
| A1† | **0.9715** [0.9655, 0.9771] | 0.985 | 0.002 | 0.014 | 0.778 | 0.867 | 0 | 3,192 |

Hallucination taxonomy, per-narrative rate:

| System | H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 | **H9** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0.903** |
| B2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 | **0.903** |
| S1 | 0.000 | 0.020 | 0.031 | 0 | 0 | 0 | 0 | 0.068 | **0.903** |
| A1 | 0.002 | 0.023 | 0.033 | 0 | 0 | 0 | 0 | 0.074 | **0.903** |

**H9 is the same 2,882 cases in all four systems**, and the same fields within them
(`labels.n_licit_counterparties` 2,732, `labels.focal_is_illicit` 2,685,
`temporal.burst_detected` 1,482). Not one of the 6,384 S1 and A1 narratives mentions a
licit counterparty, an absent burst or the subject's own (un)flagged status. On the full
Bronze corpus H9 is 0.918 (§4.1); 0.903 is its value on the test split. B2's H8 of 1.000
and unverifiable rate of 0.097 are its appended classifier sentence, which the checker
cannot verify on any case — a property of that baseline, to be stated wherever it is
quoted.

**Gate 8 pilot (seed 42)** — `scripts/09b_compare_arms.py`, probe supported rate over six
aligned diagnostic steps, tolerance 0.02
(`artifacts/metrics/generator/gate8_pilot_seed42_20261014-1950.json`):

| | S1 | A1 | Difference | Verdict |
|---|---:|---:|---:|---|
| Probe supported rate | 0.99487 | 0.99490 | −0.00003 | **indistinguishable** |

In-run diagnostics, both arms, every 50 steps: the fusion gate stays at 0.62 → 0.619; the
soft tokens receive 0.0006–0.0010 of attention against a uniform 0.017–0.019 (lift
0.03–0.055); S1's within-run shuffled-graph control matches S1 at every step and raised the
tracking alarm from step 150.

Paired on the same 3,192 test cases (A1 − S1; bootstrap 95% CI; Wilcoxon, Holm-corrected):

| Metric | A1 − S1 | p (Holm) | Cliff's δ |
|---|---|---|---:|
| Zero-Hallucination | −0.0075 [−0.0138, −0.0016] | 0.014 | −0.008 |
| Fact coverage | +0.0114 [+0.0085, +0.0142] | <0.001 | +0.037 |
| Fact F1 | +0.0075 [+0.0054, +0.0096] | <0.001 | +0.037 |
| Hallucination rate | +0.0005 [+0.0002, +0.0009] | 0.002 | +0.008 |
| Fact precision | −0.0009 [−0.0022, +0.0003] | 0.026 | −0.004 |
| Unverifiable rate | +0.0004 [−0.0008, +0.0016] | 0.22 | +0.005 |

Opposite signs, every effect negligible, all inside the 0.02 tolerance. One seed is not the
three Gate 8 specifies; the remaining seeds are queued (D-114).

**B7 seed 42 — the same model fine-tuned on the serialised facts with no graph at all**
(scored 2026-10-30, `artifacts/matrix/B7/seed42/1fe562334aff/metrics.json`):

| System | Zero-Halluc. | Precision | Halluc. rate | Unverif. | Coverage | Fact F1 | H2 | H3 | H8 | H9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B7† | 0.9765 | 0.9892 | 0.0013 | 0.0095 | 0.7716 | 0.8639 | 0.018 | 0.023 | 0.063 | 0.903 |

Paired B7 − S1 on the same 3,192 cases (case bootstrap, 2,000 resamples): Zero-Hallucination
−0.0025 [−0.0066, +0.0016]; coverage +0.0053 [+0.0025, +0.0082]; fact F1 +0.0046
[+0.0025, +0.0068]. **Removing the graph pathway entirely costs nothing**: B7 is
indistinguishable from S1 on the headline and marginally ahead on coverage. Read with the
Gate 8 pilot, both comparisons the paper's central claim rests on say the same thing at
seed 42: the generator's faithfulness comes from the serialised facts, not the graph.

**Phase 7 replication on this machine** (27 runs, 2026-10-05): GATv2 0.857, MLP 0.797,
gate GATv2 − MLP +0.0605 excluding zero at every seed (PASS, as published); GIN 0.861;
graph transformer 0.896, **beating GATv2 by 0.038 with the interval excluding zero at
every seed** — published as not significant (D-113); SAGE 0.760; GCN 0.712. Edge features
+0.092 (significant); positional encodings −0.005 and focal loss −0.017 (both null, as
published).

### 4.10 The matrix as it stood when the user stopped it (2026-11-17)

Every completed run, scored by `scripts/11c_score_matrix.py` (same regime as §4.9; greedy,
unguarded; † = one seed). Full per-run table, every H-class and every run path:
`artifacts/metrics/phase11/results_snapshot_2026-11-17.md`. Aggregated tables:
`artifacts/metrics/phase11/` (written 2026-11-09).

| System | Seeds | Zero-Halluc. mean (sd) | Coverage | Fact F1 | H2 | H3 | H9 |
|---|---|---|---|---|---|---|---|
| B1 template | 42 | 1.0000† | 0.8525† | 0.9200† | 0† | 0† | 0.903† |
| B2 classifier + template | 42 | 1.0000† | 0.8525† | 0.8769† | 0† | 0† | 0.903† |
| B6 untuned Llama-3.1-8B, zero-shot | 42 | 0.6501† | 0.4948† | 0.5863† | 0.144† | 0.898† | 0.903† |
| **S1** graph + facts | 42, 1337, 2024 | 0.9803 (0.0064) | 0.7704 (0.0037) | 0.8621 (0.0024) | 0.019 | 0.038 | 0.903 |
| **A1** shuffled graph | 42, 2024 | 0.9759 (0.0062) | 0.7776 (0.0001) | 0.8669 (0.0001) | 0.021 | 0.038 | 0.903 |
| **B7** facts only, no graph | 42, 1337 | 0.9853 (0.0124) | 0.7731 (0.0021) | 0.8644 (0.0006) | 0.012 | 0.033 | 0.903 |
| **S2** graph only, no facts | 42, 1337, 2024 | **0.0000** (0.0000) | 0.3458 (0.0075) | 0.4015 (0.0068) | 0.605 | 0.992 | 0.903 |
| A2 MLP encoder | 42 | 0.9734† | 0.7758† | 0.8654† | 0.023† | 0.040† | 0.903† |

B6 additionally has a **critical error rate of 0.120** (H5 0.113, H6 0.120) and H8 0.737;
S2 has H1 (entity fabrication) **1.000** and H8 ~1.000 at every seed. Every other system's
critical error rate is 0.

**Gate 8, per seed** (`scripts/09b_compare_arms.py`, probe supported rate, tolerance 0.02):

| Seed | S1 | A1 | Difference | Verdict |
|---|---:|---:|---:|---|
| 42 | 0.99487 | 0.99490 | −0.00003 | indistinguishable |
| 2024 | 0.99476 | 1.00000 | −0.00524 | indistinguishable |
| 1337 | — | — | — | **NOT RUN**: A1 seed 1337 failed its overfit check (§6, D-119) |

**Paired on the same 3,192 test cases** (mean difference, 95% case-bootstrap CI):

| Comparison | Seed | Zero-Halluc. | Coverage | Fact F1 |
|---|---|---|---|---|
| A1 − S1 | 42 | −0.0075 [−0.0135, −0.0016] | +0.0114 [+0.0084, +0.0143] | +0.0075 [+0.0053, +0.0096] |
| A1 − S1 | 2024 | −0.0069 [−0.0103, −0.0038] | +0.0062 [+0.0037, +0.0090] | +0.0040 [+0.0023, +0.0058] |
| B7 − S1 | 42 | −0.0025 [−0.0066, +0.0016] | +0.0053 [+0.0025, +0.0082] | +0.0046 [+0.0025, +0.0068] |
| B7 − S1 | 1337 | +0.0194 [+0.0138, +0.0251] | +0.0010 [−0.0024, +0.0044] | +0.0009 [−0.0013, +0.0031] |
| A2 − S1 | 42 | −0.0056 [−0.0122, +0.0003] | +0.0096 [+0.0070, +0.0121] | +0.0061 [+0.0042, +0.0079] |
| S2 − S1 | 42 / 1337 / 2024 | −0.979 / −0.975 / −0.987 | −0.419 / −0.436 / −0.419 | −0.457 / −0.470 / −0.455 |
| B6 − S1 | 42 | −0.3289 [−0.3468, −0.3117] | −0.2715 [−0.2775, −0.2659] | −0.2731 [−0.2780, −0.2685] |
| B1 − S1 | 42 | +0.0210 [+0.0160, +0.0257] | +0.0862 [+0.0830, +0.0893] | +0.0607 [+0.0581, +0.0633] |

**What the snapshot says.**
1. **With the facts in the prompt, the graph adds nothing.** S1, A1 (wrong graph), B7 (no
   graph) and A2 (no message passing) all sit at ZHR 0.97–0.99 and coverage 0.77–0.78; the
   paired differences change sign across seeds (A1 is below S1 on ZHR at both seeds, B7 is
   below at seed 42 and above at seed 1337) and are all small. Gate 8 is "indistinguishable"
   at both seeds that ran. The trained S1/A1/A2 models give the soft tokens ~1/20–1/12 of
   uniform attention (lift 0.03–0.08) and the shuffled-graph control tracks them at every
   seed.
2. **Without the facts, the graph tokens cannot carry the case.** S2 contradicts the record
   in **every** narrative at every seed (ZHR 0.000; H1 1.000 — the soft tokens cannot carry
   account identifiers, amounts or times) and covers 35% of salient facts. But the tokens
   are not empty: S2's probe supported rate exceeds its own shuffled-graph control at every
   seed (0.460 vs 0.412, 0.457 vs 0.400, 0.482 vs 0.443) and it pays them 3–4x the
   attention S1 does (lift 0.10–0.14). The graph carries some structure; nowhere near a
   narrative's worth.
3. **Fine-tuning is what buys faithfulness**: the untuned model on the baselines' own prompt
   (B6) reaches ZHR 0.650 with a 12% critical-error rate; the same model fine-tuned on the
   same facts (B7) reaches 0.985 with none.
4. **Nothing beats the template**, and H9 is 0.903 in every system on the same cases:
   training on Bronze alone reproduces Bronze's omissions exactly.

**Not run when stopped:** A1 seed 1337, B7 seed 2024, B8, A4, A3_F3 and A3_F4 failed their
overfit checks (memorised, then diverged — not wiring; A4 diverged because the check gave
its unfrozen encoder the projector's rate) and A5 ran out of memory in its guarded decode.
The fixes are in the working tree and tested on the stub, not run (D-119). A6, B3, B4, B5,
Silver and the κ were never queued (D-115; API spend).

## 5. What is NOT MEASURED, and why

| Quantity | Status | Blocker |
|---|---|---|
| **Gate 8 — does S1 beat A1 on faithfulness?** | **Pilot answered at seed 42: indistinguishable** (§4.9). Seeds 1337 and 2024 queued. | GPU time |
| Every Layer 1 overlap metric (BLEU, ROUGE, METEOR, BERTScore) | **NOT MEASURED** | scored against Gold only; Gold does not exist |
| The template-baseline finding (does a template score competitively on ROUGE?) | **NOT MEASURED** | Gold. `template_baseline_finding` runs the day one Gold narrative exists. |
| **Method A / Method B extractor agreement κ** | **NOT MEASURED** | API credentials. This is the number that makes the faithfulness metric defensible rather than merely defined. |
| 200 hand-labelled errors validating the taxonomy classifier | **NOT MEASURED** | Scoring Bronze yields only H9 findings, so there is no interesting error pool until a model arm exists. |
| **Correlation of automatic metrics with human factual correctness** | **NOT MEASURED** | Phase 12 built, not run. Ethics approval **not applied for** (4-8 weeks, blocks all collection); only Bronze has generations, so a five-arm study has one arm; no rater recruited. See PHASE_LOG Phase 12 and D-092. |
| Time-to-usable-draft per system, vs the Bronze template baseline | **NOT MEASURED** | same. This is the strongest deployment evidence the project can produce and it has no number. |
| Edit distance from presented draft to filable version | **NOT MEASURED** | same |
| "Would you file this after review?" filing rate per system | **NOT MEASURED** | same |
| Inter-rater Krippendorff alpha (ordinal) | **NOT MEASURED** | same. The machinery reproduces the published 0.815 worked example to 0.8154; there is no panel to run it on. |
| Intra-rater reliability from repeated items | **NOT MEASURED** | same |
| Attention mass on soft tokens | **Measured** for S1 and A1 at seed 42: ~1/20 of a uniform spread at every probe step (§4.9) | — |
| Whether the F2 gate collapses to zero | **Measured**: it neither collapses nor moves — 0.62 at initialisation, 0.619 at step 300, in both arms | — |
| VRAM, throughput, latency for the 8B arms | **NOT MEASURED** | Phase 13 ran and measured what exists: B1 and B2 end to end at n=100, plus the encoder and guard components every other row shares. The thirteen 8B and API rows have no checkpoint and no card / no credentials. See §4.4 and D-097. |
| The realistic-imbalance stream | **NOT MEASURED** | no realistic-stream file has been produced |
| Everything on Elliptic2 | **NOT MEASURED** | access not requested |

---

## 6. Negative and surprising results so far

Kept under invariant 7, in the order they were found.

1. **Positional encodings contribute nothing measurable** (+0.002, not significant). Retained; reported as a null (D-066).
2. **Focal loss is no better than weighted BCE** (−0.006, not significant). Kept as default; reported as a null (D-065).
3. **The MLP control reaches 0.80 AUC-PR with no message passing at all.** This bounds every downstream claim about topology.
4. **Bronze omits an exculpatory fact in 92% of narratives.** A defect of our own corpus, found by our own harness.
5. **The round-trip test alone cannot catch an extractor bug.** Three injected bugs left it at 100% SUPPORTED, because the probe renders its claims from the record and verifies a wrong value against itself. `tests/oracle.py` is what calibrates the extractor (D-034).
6. **Silver's slot alignment failed 102 of 300 real paraphrased cases** under document-order alignment, which would have put ~34 spurious points into the discard rate. Longest-value-first on token boundaries fixes it (D-048).
7. **A wrong quantity scored UNVERIFIABLE rather than CONTRADICTED**, and **H6 was unreachable.** Both bugs understated hallucination; both would have made every system look better (Phase 10).
8. **The A1 control did not shuffle.** A nested Hydra `overrides:` block is inert, so `experiment=generator_a1` composed with `fusion.shuffle=false` — the control would have trained as a second copy of the treatment, and "S1 does not beat A1" would have been two runs of S1 compared against each other. Nothing failed; it was caught by printing the composed config.
9. **The Phase 9 overfit test plateaued at loss 1.43** because the stub backbone had no positional embeddings, making self-attention permutation-invariant. The tempting fix was to loosen the threshold, which would have left a test that passes whatever the harness does.
10. **A few-shot exemplar rotation applied across the whole candidate list can rotate past the matching typology entirely** — silently turning typology-matched few-shot into arbitrary few-shot on some case ids and not others. Found by a test written before the behaviour was inspected; the rotation is now per-group (Phase 11).
11. **Python's `hash()` is salted per process**, so the "fixed before the numbers exist" qualitative case selection would have drawn a different ten cases every invocation. Now a stable digest (Phase 11).
12. **Case extraction, not generation, dominates end-to-end latency.** Cutting a case out of the 5-million-edge graph is 72% of B1's per-narrative cost and rendering the narrative is 18%. Every row of the matrix pays it, and it does not shrink when the generator grows. Generation-only timing — the number comparable papers report — would have hidden this entirely (Phase 13, D-095).
13. **The graph half of the architecture costs 25 MB of VRAM.** B2's whole inference footprint is 0.025 GB reserved and the encoder checkpoint is 2.5 MB. Whatever makes this architecture expensive to deploy, it is not the graph encoder (Phase 13).

14. **The trained generator does not read the graph.** S1 gives its 16 soft tokens ~1/20 of
    uniform attention, generates the same narrative with a foreign graph, never moves its
    fusion gate, and is indistinguishable from the deranged-graph control A1 (Gate 8 pilot,
    §4.9). Under the pre-registered consequence the paper's contribution is the corpus and
    the evaluation framework.
15. **Both trained arms are less faithful than the template they were trained on**: 2.1-2.9%
    of their narratives carry a contradicted fact (numeric and temporal errors), and they
    cover 77-78% of salient facts against the template's 85%.
16. **H9 is identical for B1, B2, S1 and A1 — 0.903, on exactly the same 2,882 cases.** No
    trained narrative mentions an exculpatory fact Bronze never states. Trained on Bronze
    alone, the models inherit its one-sidedness exactly (D-104).
17. **The graph transformer significantly out-scores GATv2 on this replication** (+0.038, every
    seed), reversing D-064's "not significant". Reported as a changed conclusion (D-113).
18. **Defects that reached the real model and that 119 stub-backed tests could not see**: a
    decoding loop that read the next token from padding (D-107), an entrypoint that had never
    run (D-108), a control arm whose soft tokens were ~100x the treatment's scale (D-111), and
    an aggregator that had never been fed a real run (D-112). Each is recorded with the run
    that found it.

---

## 7. Provenance

Every run directory carries `resolved_config.json`, `run_context.json` (git SHA, resolved
config hash, all seeds, data manifest hash, library versions — invariant 5) and its
completion marker. Nothing is ever overwritten: the run directory is keyed on the config
hash, so a re-run under a changed config writes a new directory and the old number stays
readable (invariant 6).

Regenerate everything in this file that comes from the matrix:

```bash
uv run python scripts/11_run_matrix.py --dry-run     # the plan, no execution
uv run python scripts/11a_aggregate.py               # tables, figures, missing-run report
uv run python scripts/11b_qualitative.py             # side-by-side, worst cases, disagreements
```

---

## 8. What the next session should run first

*(Superseded 2026-10-28: items 1 and 3 are done or running; see `PROGRESS.md` for the live
queue. Kept below as written.)*

In this order, because each one is cheap and each unblocks reading of the next:

1. **`make score-cases`, then B1 and B2.** Both are CPU-only and both are runnable today.
   They put two real rows in the results table and exercise the whole run→aggregate→figure
   path on real data instead of on fixtures.
2. **Credentials → B3, B4, B5, then Silver, then the Method A/B κ.** The κ is the number
   that makes the faithfulness metric defensible.
3. **Rented GPU → S1 and A1 first**, at all three seeds, then `make gate8`. Everything
   else in the matrix is read against that comparison, and if it comes out null the
   reframing decision has to be made before more compute is spent.
4. **An annotator.** Nothing else unblocks Gold, and nothing else in the repository
   advances it.
