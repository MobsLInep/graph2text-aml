# Dataset card — Bronze corpus

**15,707 deterministically-rendered SAR narratives** paired with the fact record each was
rendered from. The faithfulness floor of the project, and the only narrative tier that
exists.

| | |
|---|---|
| Card written | 2026-08-06 (Phase 14), for release `v0.1.0` |
| File | `bronze.jsonl`, 232 MB (each record carries its full `case_facts`) |
| Records | **15,707** |
| Schema | `training_record_v1.json` **frozen at 1.0.0**; `case_facts` **frozen at 1.0.0** |
| Substrate | IBM AMLworld HI-Small (**synthetic**) |
| **Licence** | **CDLA-Sharing-1.0** — share-alike. See §7. |
| Built by | `make bronze` (~90 s, CPU only, no network) |
| Code | `src/g2t_aml/corpus/bronze/` |

---

## 1. What it is

Each record pairs a transaction subgraph — via its `case_facts` record and a `graph_ref` —
with a prose narrative describing it in the register of a Suspicious Activity Report
narrative section.

```
This referral describes activity centred on account 021611|800F41B10, identified as an
intermediary. In scope are 9 accounts and 21 transactions. The transactions fall between
2022-09-07 13:30 and 2022-09-08 08:52, a period of 19.4 hours. Aggregate inflow to the
subject account was around 6,273 US Dollar. [...] 4 counterparties were involved, of which
0 appear on flagged transactions.
```

### Fields

| Field | What it holds |
|---|---|
| `case_id`, `dataset`, `split`, `tier`, `generator` | Identity and provenance |
| `facts` | The complete `case_facts` record — the checkable ground truth |
| `serialised_facts` | The record flattened to text; the input to the "no graph encoder" ablation |
| `target_narrative` | The rendered narrative |
| `target_slots` | Which record values landed in which spans of the text |
| `salience` | What an adequate narrative must mention, per the frozen `docs/annotation/salience.md` |
| `verification` | The checker's verdict over every claim |
| `graph_ref` | Pointer into the case store, `<path>#<case_id>` |
| `length`, `schema_version` | Bookkeeping |

---

## 2. Construction

**Deterministic template rendering from the fact record.** No model, no sampling, no
network, no randomness. `make bronze` rebuilds it byte-identically in ~90 s.

The load-bearing design rule:

> **A generated claim is parsed out of the rendered text, never read from the record.**
> Building a claim from the value the formatter started with compares the record with
> itself and reports *any* corpus as 100% SUPPORTED. **Every formatter ships with its
> inverse** (D-040).

That is the same circularity trap as D-034 in a more dangerous place, and it is why the
1.0000 in §4 means something.

### Splits

Temporal, frozen, committed as ID lists with content hashes (invariant 2). **Never
regenerated from a seed at runtime.**

| Split | Cases | Bronze records |
|---|---:|---:|
| train | 10,932 | **10,488** |
| val | 2,028 | **2,027** |
| test | 3,196 | **3,192** |
| **total** | **16,156** | **15,707** |

The 449-record shortfall is cases that produced no renderable narrative under the length
bounds or the availability mask; it is not a sampling step.

### Composition by narrative family

| Family | Records |
|---|---:|
| `no_finding` | 11,062 |
| `minimal_activity` | 2,474 |
| `unclassified_suspicious` | 1,338 |
| `bipartite` | 140 |
| `gather_scatter` | 137 |
| `fan_out` | 133 |
| `scatter_gather` | 120 |
| `cycle` | 92 |
| `random` | 81 |
| `fan_in` | 70 |
| `stack` | 60 |

**86% of the corpus is `no_finding` or `minimal_activity`.** The eight structural
typologies together are 833 records, 5.3%. This reflects the source graph's 1-in-981
laundering rate and it is the single most important thing to know before training on this
corpus.

---

## 3. Quality gate — the ten-point harness

**All ten checks at zero failures over all 15,707 records. `gate_passed: true`, pass rate
1.0.** The same harness gates Silver and Gold identically (D-037).

| Check | Failures |
|---|---:|
| `schema_valid` | 0 |
| `facts_schema_version` | 0 |
| `graph_ref_resolves` | 0 |
| `split_consistent` | 0 |
| `length_in_bounds` (80–400 tokens) | 0 |
| `vocabulary_clean` | 0 |
| `no_pii_or_identifiers` | 0 |
| `zero_contradicted` | 0 |
| `unverifiable_rate` (≤ 0.05) | 0 |
| `deduplicated` (Jaccard 0.85) | 0 |

Deduplication examined 1,678 candidate pairs and confirmed **0**, so nothing was dropped.

---

## 4. Evaluation

Scored by the independent Phase 10 harness (`make eval-bronze`, 41 s, no GPU, no network).
This is the project's CI gate.

| Metric | Value | 95% CI |
|---|---:|---|
| **Zero-Hallucination Rate** | **1.0000** | [1.0000, 1.0000] |
| Fact Precision | 1.0000 | — |
| Hallucination Rate | 0.0000 | — |
| Unverifiable Rate | 0.0000 | — |
| **Fact Coverage** | **0.8595** | [0.8592, 0.8598] |
| Fact F1 | 0.9243 | — |
| Numeric / Typology / Ordering Accuracy | 1.0000 | — |
| Critical Error Rate (H4+H6+H7) | 0.0000 | — |
| Claims scored | **296,196** | all SUPPORTED |
| Narratives with **no** claims | **0** | asserted separately |

**The 1.0000 is a regression test on the measurement instrument, not an achievement.**
Bronze is faithful by construction. Phase 4's independent harness counted 296,195 claims
over the same corpus — agreement to one claim in 296 thousand between two extractors that
reach the claims by different routes. `n_narratives_with_no_claims = 0` is what makes the
rest of the table mean anything: a perfect score over an empty claim set is what a broken
extractor produces.

### Surface statistics

| | |
|---|---:|
| distinct-1 / distinct-2 | 0.0137 / 0.0352 |
| self-BLEU @ **5** references | 0.4752 |
| Length (words) | 146.6 ± 11.2 |
| Length (tokens) | median 279, p05 244, p95 312, range 152–356 |

**Self-BLEU without its reference count is not a number** (D-043): on this corpus it reads
0.16 at one reference and 0.82 at fifty, flat in between. It is reported at a fixed five
and the curve is published beside it.

---

## 5. Known limitations

### 5.1 The one that matters — 92% omit an exculpatory fact

**H9 (omission of exculpatory fact) fires on 0.9179 of narratives.**

The templates report `labels.n_counterparties` and `labels.n_illicit_counterparties` but
never the *licit* count, never `labels.focal_is_illicit`, and mention
`temporal.burst_detected` only when a burst was detected. A case whose subject carries no
illicit label produces a narrative that never says so. The reader gets a one-sided account.

Found by the Phase 10 harness, not by inspection. **Not fixed in this release.**

Two consequences: **the template floor is not uniformly 1.0**, so H9 is a concrete
dimension on which a trained system can beat it; and it is the failure mode least likely to
be caught by human review, because a reviewer cannot notice the absence of something they
were never shown (`docs/ETHICS.md` §3.3).

### 5.2 Fact Coverage varies 9 points by typology

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
| **bipartite** | **0.770** |

Not noise: structurally richer typologies have longer salience lists and the templates do
not grow to match them. **`bipartite` and `cycle` are where a generator has the most room.**

### 5.3 The rest

- **Stylistically flat by design.** distinct-1 of 0.0137 is what a template corpus looks
  like. Do not read the diversity numbers as a defect being reported; read them as the
  reason Silver exists.
- **Severe class imbalance.** 86% `no_finding` / `minimal_activity`; `stack` has 60
  records and `fan_in` 70.
- **`model_signal.gnn_risk_score` is `none` throughout, deliberately.** Phase 7 has since
  populated `model_signal`, so regenerating Bronze now would push the encoder's own score
  into `serialised_facts` — which is the input to the *"no graph encoder"* ablation arm —
  and nothing would fail. Bronze is deliberately not regenerated and a test pins
  `gnn_risk_score=none` in the file (D-063). **If you rebuild Bronze, you have changed that
  ablation.**
- **The ten-point harness cannot see an unaligned quantity in Silver or Gold.** It rebuilds
  claims from `target_slots`, which are exactly the values that *did* align. Bronze is
  immune by construction; the other tiers enforce an unverifiable budget at ingestion
  instead (D-057). Relevant here only as a warning against assuming the harness generalises.
- Everything in [`README.md`](README.md) § Common limitations.

---

## 6. Intended use and misuse

**Intended.** Training and evaluating graph-to-text SAR narrative generation; as the
faithfulness ceiling baseline (system B1); as a fixture for developing verification
machinery.

**This corpus is not a corpus of real SARs and must not be presented as one.** Real SARs
are confidential by statute and none were consulted. These are template renderings over
synthetic simulator output. A model trained on this corpus has learned the register of
*this corpus*, which no regulator has validated.

**Do not use the H1–H8 zeros as a benchmark result.** They are a property of deterministic
rendering. A system compared against them is being compared against construction, not
against performance.

All prohibitions in `docs/ETHICS.md` §2 apply.

---

## 7. Licence — CDLA-Sharing-1.0, and why not Apache-2.0

The narratives quote AMLworld account identifiers, timestamps, currencies and transaction
amounts verbatim, and `facts` embeds per-transaction values (`max_single_transfer`,
`first_seen`, `entity_inventory.node_ids`). That is **more than a de-minimis portion of the
source Data**, which makes this corpus **Enhanced Data** under CDLA-Sharing-1.0 — not a
§3.5 *Result*.

Consequences for anyone redistributing it:

- It must go out under an **unmodified CDLA-Sharing-1.0**, with the agreement text
  included and attribution to IBM as Data Provider preserved.
- **Modifications must be marked** with prominent notices.
- **No additional restrictions** may be imposed downstream — no commercial-use, platform or
  field-of-use limits.
- It must **not** be bundled with the Apache-2.0 code/weights/metrics release. They are
  separate archives on purpose (D-098).

Metrics computed *over* this corpus (`RESULTS.md`, `bronze_validation.json`,
`bronze_diversity.json`) are Results and are Apache-2.0.

---

## 8. Reproduction

```bash
make data      # ingest AMLworld HI-Small          ~14 s
make cases     # frozen temporal splits            ~4.5 min
make facts     # case_facts extraction
make bronze    # render + the ten-point gate       ~90 s
make eval-bronze                                 # ~41 s, the CI gate
```

**Bronze is fully deterministic: it rebuilds byte-identically, and the tolerance is
exact.** If your rebuild differs at all, something is wrong — see
`docs/REPRODUCTION.md` § Tolerances.

Companion files shipped alongside `bronze.jsonl`: `bronze_validation.json` (the ten-point
report), `bronze_diversity.json` (the self-BLEU curve and distinct-n), and
`bronze_samples.md` (a per-family sample a human can read without loading 232 MB).
