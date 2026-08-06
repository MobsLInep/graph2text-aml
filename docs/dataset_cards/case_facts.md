# Dataset card — `case_facts` records and the frozen splits

**16,156 structured fact records** over AMLworld HI-Small subgraphs, plus the committed
temporal split manifests. This is the substrate every narrative tier is rendered from and
every faithfulness verdict is checked against.

| | |
|---|---|
| Card written | 2026-08-06 (Phase 14), for release `v0.1.0` |
| Records | **16,156** cases |
| Schema | `case_facts` **FROZEN at 1.0.0** — `schemas/case_facts_v1.json` |
| Files | `facts.parquet` (2.3 MB), `cases/cases.jsonl` (33 MB), `schemas/splits/amlworld/` |
| **Licence** | **CDLA-Sharing-1.0** (records) / **Apache-2.0** (split ID manifests and schemas) |
| Built by | `make cases` (~4.5 min) then `make facts` |
| Code | `src/g2t_aml/facts/`, `src/g2t_aml/data/case_extraction.py` |

---

## 1. Why this is the most load-bearing artifact in the project

**The fact layer is a measurement instrument** (invariant 1). It runs in both directions
using the same field paths: `extract_facts` builds a checkable record from a case, and
`check_claim` verifies a narrative's claims against exactly that record.

> **The verification machinery that makes the corpus trustworthy is the *same code* that
> measures faithfulness at evaluation time, run in reverse.**

So a bug here silently corrupts every headline number in the paper — the corpus *and* the
metric, in the same direction, with nothing failing. That is why it carries ≥90% coverage,
`mypy --strict`, golden-file tests, and **calibration against an independent oracle**.

**The round-trip test alone cannot catch an extractor bug.** The probe renders its claims
*from the fact record*, so a wrong value is stated wrongly and verified against itself —
three injected bugs left it at 100% SUPPORTED. `tests/oracle.py` is what actually
calibrates the extractor, and it must stay free of any import from `g2t_aml.facts`
(D-034).

---

## 2. Schema — frozen at 1.0.0

Fact families: `entity_inventory`, `focal_entity`, `flow`, `structure`, `temporal`,
`motifs`, `labels`, `model_signal`, plus an `availability` mask.

**Changing any field — adding, removing, renaming, or altering a type or an availability
rule — is a breaking change that invalidates every fact record, every generated corpus and
every published number derived from them.** The version is declared in **five** places and
a test asserts they all agree; `scripts/03_extract_facts.py` aborts on a mismatch
(invariant 9).

### Absence is typed, and there are two kinds of it

| | Meaning | Checker verdict |
|---|---|---|
| `Unavailable(reason=...)` | The substrate **cannot** supply this fact family | UNVERIFIABLE |
| bare `None` | A **measured** null — no cycle exists; no illicit node is reachable | A real comparison |

**Never `0`, and never conflate the two** (D-025). `mypy --strict` cannot dereference the
union without narrowing, which is what enforces it.

### The availability mask (invariant 4)

Nine flags per substrate. Nothing may assert a fact that does not exist for its substrate.
AMLworld has no entity types; Elliptic2 has no amounts, currencies, real timestamps or
entity types.

**One trap: `availability.node_labels` is `True` on Elliptic2.** The substrate labels whole
subgraphs, so no *individual* account is labelled at all. Anything reading that flag to
decide whether accounts are labelled will paint every Elliptic2 node "unflagged" —
invariant 4 violated in pixels rather than in text. Use `CaseView.has_labels`, which is
what the fact layer gates `LabelFacts` on.

---

## 3. Construction

### Case extraction

```
k_hops: 2                    max_neighbours_per_node: 64      n_max: 150
prune_rule: amount_desc      preserve_laundering_paths: true  seed: 1337
max_window_hours: 48.0       window_pad_hours: 12.0
```

**A case does not contain its whole laundering stream.** The 48-hour window cap keeps 65%
of a stream's transactions on average, so `typology` means *"part of a stream of this
typology"*, not *"exhibits it in full"* (D-019). Every downstream label inherits this.

### Sampling

```
n_cases: 30000 requested      positive_fraction: 0.2       hard_negative_fraction: 0.3
hard_negative_min_score: 0.5  max_stratum_share: 0.35      max_seeds_per_stream: 12
```

The source graph's laundering rate is **1 in 981**, so uniform sampling would produce a
case set containing almost no laundering. Cases are stratified; the resulting prevalence is
**not** the base rate and must never be reported as one.

A separate **realistic-imbalance stream** of 10,000 uniformly-sampled cases (observed
prevalence 0.073) exists precisely so that a number at realistic imbalance can be reported
alongside the stratified one.

### Composition

| | Cases |
|---|---:|
| **Total** | **16,156** |
| by class | licit 10,081 · hard_negative 3,558 · suspicious 2,517 |
| by label | licit 13,781 · suspicious 2,375 |
| hard-negative rate | 0.258 |
| by typology | none 13,639 · unclassified 1,447 · bipartite 204 · fan_out 164 · gather_scatter 140 · scatter_gather 134 · cycle 122 · stack 107 · random 103 · fan_in 96 |

---

## 4. The splits — temporal, frozen, committed, audited

**Never regenerated from a seed at runtime** (invariant 2). Committed as ID lists with
content hashes in `schemas/splits/amlworld/`; the `data.split` configs carry no `seed` key,
so there is nothing to regenerate them *from*.

| Split | Cases | Window |
|---|---:|---|
| train | **10,932** | → 2022-09-06 05:46 |
| val | **2,028** | 2022-09-06 18:22 → 2022-09-09 05:59 |
| test | **3,196** | 2022-09-09 18:31 → |

```
mode: temporal    boundaries: [2022-09-06T12:03, 2022-09-09T12:03]
buffer_hours: 6.0     boundary_snap_hours: 24.0     proportions: [0.7, 0.15, 0.15]
```

**13,844 cases were dropped to make the splits clean**, and the reasons are recorded rather
than absorbed: 9,718 straddled a boundary, 4,110 fell within the 6-hour buffer, and 16 had
a stream already assigned to an earlier split.

### The leakage audit — passed, with one finding reported rather than fixed

`make audit` re-runs it over the committed manifest.

| Check | Severity | Result |
|---|---|---|
| Temporal ordering | fatal | **passed** — every later split begins after the previous ends |
| Stream atomicity | fatal | **passed** — all 211 laundering streams sit in exactly one split |
| Label leakage | fatal | **passed** — no declared feature names or reproduces the label |

**Edge overlap between splits is 0.000 (0 shared edges). Node overlap is 0.538.**

That node overlap is real and is reported, not suppressed: 3,388 accounts appear in both
train and test cases, because a long-lived account transacts on both sides of a temporal
boundary. **It is exactly why encoder features must be recomputed from each case's own
edges.** The interim node table's `in_degree` / `out_degree` / `degree` / `total_received`
/ `total_sent` are *global* aggregates across the whole 515,088-account graph and across
both sides of the boundary — reading them into a model feature would carry a test-window
account's training-window activity into its encoding (D-059), and reading them in the fact
layer would make every narrative unfaithful. A test overwrites all five with absurd
constants and requires the feature tensor to be unchanged.

---

## 5. Limitations

- **17.7 days of source data**, split three ways with a 6-hour buffer. Very little temporal
  room by any deployment standard.
- **Node overlap of 0.538 between splits.** Temporally clean, not entity-disjoint. §4.
- **Stratified, not representative.** The case set's positive rate is a sampling parameter,
  not the base rate. Use the realistic-imbalance stream when the base rate matters.
- **A case is a fragment** (D-019, §3).
- **`model_signal` is populated now, and was not when Bronze was rendered.** Phase 7 wrote
  `gnn_risk_score` back into every record. Bronze is deliberately **not** regenerated,
  because `facts.serialiser._compact` emits `gnn_risk_score` into `serialised_facts` — the
  input to the *"no graph encoder"* ablation arm — and regenerating would push the
  encoder's own score into that baseline with nothing failing (D-063).
- **Degree means distinct counterparties, never transaction count**, everywhere in
  `facts/`. A consumer assuming otherwise will misread every degree field.
- **Never sum across currencies.** 15 currencies, 72,170 cross-currency transactions, no
  exchange rates. Cross-currency aggregates are withheld as sentinels and the per-currency
  breakdown is always emitted (D-033).
- **AMLworld only.** `facts_elliptic2` skips cleanly; the substrate has never been ingested.
- Everything in [`README.md`](README.md) § Common limitations.

---

## 6. Licence

Split ID manifests, the JSON Schema and `vocab_v1.yaml` are **Apache-2.0** — they are lists
of case identifiers and a schema, not data.

**The fact records and case files are CDLA-Sharing-1.0**: they embed AMLworld account
identifiers, timestamps and per-transaction amounts. Same reasoning as
[`bronze.md`](bronze.md) §7, and they ship in the same data bundle.

---

## 7. Intended use and misuse

**Intended.** The checkable ground truth for narrative generation and faithfulness
evaluation; the input to case-level graph classification.

**Misuse.** All of `docs/ETHICS.md` §2. Specific to this artifact: **do not treat a fact
record as a description of a real account.** Every identifier in it is simulator output.
And **do not use the stratified prevalence as a base rate** — it is a sampling parameter,
and quoting it as an AML prevalence figure would misrepresent the source data by more than
an order of magnitude.

---

## 8. Reproduction

```bash
make data          # ingest AMLworld HI-Small                     ~14 s
make cases         # extract, split temporally, audit for leakage ~4.5 min
make audit         # re-run the leakage audit over the manifest
make facts         # extract case_facts records
make facts-gate    # the 1,000-case round trip + the independent oracle
```

**Deterministic. `seed: 1337` is a fixed extraction parameter, not a runtime knob, and the
tolerance is exact** — the split manifests are content-hashed and a rebuild that changes
one is a failure, not a variance. See `docs/REPRODUCTION.md` § Tolerances.
