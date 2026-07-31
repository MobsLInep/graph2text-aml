# schemas/

Two kinds of thing live here, both of them **committed** and both of them load-bearing.

## `case_facts.schema.json` — *Phase 3*

The JSON Schema for the `case_facts` record: the structured, substrate-agnostic
description of a flagged subgraph that every narrative is generated from and every
faithfulness metric is measured against.

Not written yet. Its shape is not knowable until the fact families are fixed in Phase 3.

**Invariant 3: schema versions are pinned and recorded in every derived artifact.** The
version lives in `g2t_aml.CASE_FACTS_SCHEMA_VERSION` and in
`configs/config.yaml: schema_version.case_facts`, and is echoed into every
`run_context.json`. Changing it after corpus generation means regenerating the corpus.

Every record carries an **availability mask** (invariant 4) declaring which fact families
its substrate can support. The mask keys must stay in sync with `data.availability` in
`configs/data/*.yaml`; there is a test asserting the key set.

## `splits/<substrate>/` — *Phase 2*

Frozen temporal split manifests: `train.txt`, `val.txt`, `test.txt`, one case ID per
line, plus a `manifest.json` carrying the content hash of each list
(`utils.hashing.hash_id_list`).

**Invariant 2: splits are temporal and frozen.** These are committed ID lists, never
regenerated from a seed at runtime — a seeded split is only reproducible while the
upstream row order, library versions, and filtering code all hold still, and across
fourteen phases they will not. See `DECISIONS.md` D-006.

Regenerating a split is a visible diff and requires a `DECISIONS.md` entry.
