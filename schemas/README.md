# schemas/

Four kinds of thing live here, all **committed** and all load-bearing.

## `case_facts_v1.json` — *Phase 3* · **FROZEN at 1.0.0**

The JSON Schema for the `case_facts` record: the structured, substrate-agnostic
description of a flagged subgraph that every narrative is generated from and every
faithfulness metric is measured against. Draft 2020-12, strict, `additionalProperties:
false` on every object — a test walks the document and asserts it.

**Invariant 9: this schema is frozen.** Adding, removing, renaming or retyping a field is a
breaking change that invalidates every fact record, every generated corpus and every
published number derived from them. It requires a version bump, a `DECISIONS.md` entry, and
regeneration from Phase 3 forward.

The version is declared in **four** places and
`tests/unit/test_facts_coverage.py::test_schema_version_is_frozen_and_consistent_everywhere`
asserts all four agree:

| Location | Role |
|---|---|
| `g2t_aml.facts.schema.CASE_FACTS_SCHEMA_VERSION` | the source of truth |
| `schema_version.const` in this schema | what a record must declare to validate |
| `case_facts_schema_version` in `vocab_v1.yaml` | which schema the vocabulary targets |
| `schema_version.case_facts` in `configs/config.yaml` | what a run reads or writes |

`g2t_aml.CASE_FACTS_SCHEMA_VERSION` re-exports the first so a consumer that has not imported
the fact layer sees the same number. `scripts/03_extract_facts.py` aborts if the config and
the code disagree.

**Availability is the load-bearing design detail.** Every fact family a substrate may not
support is a `oneOf` between the populated object and an explicit
`{"available": false, "reason": ...}` sentinel. Absence is never `0` and never a bare `null`
that could be read as a measured value (D-025). Each record also carries the substrate's full
`AvailabilityMask` verbatim, so it is self-describing.

## `vocab_v1.yaml` — *Phase 3* · **FROZEN at 1.0.0**

The controlled vocabulary: what any component is *permitted* to assert. Closed lists of
entity roles and typologies; every qualitative intensifier bound to a numeric field and a
condition on it; the hedging allow/forbid lists; the regulatory whitelist; and the per-typology
salience lists.

The methodological contribution is `risk_descriptors`. "Rapid dispersal" is not a stylistic
flourish — it is a claim that `temporal.burst_window_hours <= 6`, and if the record says 400
the narrative is CONTRADICTED. Publishing the binding table is what makes qualitative language
checkable.

Note the **deliberate exclusion** of entity-type terms — no `mixer`, no `exchange`, no
`shell company`. Neither substrate carries an entity-type column, so those claims are
unevidenced by construction, and they are excluded from the vocabulary rather than merely
flagged afterwards. See D-029 and `docs/annotation/hallucination_taxonomy.md`.

## `training_record_v1.json` — *Phase 4* · **FROZEN at 1.0.0**

One (graph, facts, narrative) training example. **The same schema carries all three corpus
tiers** — Bronze, Silver and Gold differ only in `tier` and in the open `generator` block —
so the ten-point harness in `g2t_aml.corpus.validate` gates all three with one
implementation, and "Silver is verified" means exactly what it means for Bronze. Designing
it that way now is what stops Phase 5 needing a migration.

Two details carry the weight:

**The record embeds its fact record** rather than pointing at one, `$ref`-ing
`case_facts_v1.json` so the two can never drift. A narrative must be re-verifiable against
*the record it was written from*, not against whatever the fact store holds later — those
are the same object today and need not be after a Phase 7 write-back or any re-run of
`make facts`. See D-037.

**`target_slots` aligns narrative text back to fact fields** by character span, on every
tier. That turns Phase 10's Layer-2 faithfulness evaluation into an alignment problem
rather than an LLM-extraction problem, and lets Silver's verifier align a rewrite against
the Bronze it came from. The claim a checker verifies is parsed out of `rendered_value` —
the text actually written — never out of `raw_value`, because building it from the record
would compare the record with itself and report every corpus as perfectly faithful. See
D-040.

## `splits/<substrate>/` — *Phase 2*

Frozen temporal split manifests: `train.txt`, `val.txt`, `test.txt`, one case ID per
line, plus a `manifest.json` carrying the content hash of each list
(`utils.hashing.hash_id_list`).

**Invariant 2: splits are temporal and frozen.** These are committed ID lists, never
regenerated from a seed at runtime — a seeded split is only reproducible while the
upstream row order, library versions, and filtering code all hold still, and across
fourteen phases they will not. See `DECISIONS.md` D-006.

Regenerating a split is a visible diff and requires a `DECISIONS.md` entry.
