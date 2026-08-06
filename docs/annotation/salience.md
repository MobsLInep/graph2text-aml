# Salience: what an adequate SAR narrative must mention

**Status:** frozen in Phase 3 (2026-08-01), before any narrative existed. See `DECISIONS.md`
D-032. The machine-readable source of truth is the `salience:` block in
`schemas/vocab_v1.yaml`; this document is the human-facing statement of the same lists, for
annotators working under the Phase 6 protocol.

## Why this was fixed before generation started

Adequacy is scored against these lists. If they were written after inspecting model output,
they would be a description of whatever the model happened to produce, and every adequacy
number measured against them would be circular — the system would be graded against its own
behaviour. Fixing them first is what makes adequacy a test the system can fail.

The same lists are used by the automated metric and by human annotators, so the two are
scoring the same definition rather than two that drifted apart.

## How to apply them

For each case, take the list for its `typology.label`. Every entry is a field path into the
`case_facts` record.

1. **A field under an availability sentinel is excused.** If `flow.total_outflow` is
   `{"available": false, ...}` — as it is on every Elliptic2 case — the narrative is not
   penalised for omitting it. No narrative could mention it faithfully.
2. **A field whose value is a measured `null` is excused.** A narrative cannot be required to
   state a fan width for a fan that is not there.
3. Everything remaining is **required**. `facts.salience.salience_report()` performs steps 1
   and 2 and returns the filtered `required` list plus the `excused` one.

Adequacy is `mentioned / len(required)`. A case whose required list is empty scores no
adequacy rather than a perfect one.

## The lists

Every typology's list is its own entries **plus** the common three.

**Common to every typology**

| Field | Why it is salient |
|---|---|
| `focal_entity.id` | A SAR that does not identify its subject is not a SAR. |
| `structure.n_nodes` | The reader needs the size of what they are looking at. |
| `structure.n_edges` | Distinguishes a two-transaction case from a sixty-transaction one. |

**Per typology**

| Typology | Additional required fields |
|---|---|
| `fan_out` | `motifs.fan_out.width`, `temporal.span_hours`, `flow.total_outflow`, `labels.n_illicit_counterparties` |
| `fan_in` | `motifs.fan_in.width`, `temporal.span_hours`, `flow.total_inflow`, `labels.n_illicit_counterparties` |
| `gather_scatter` | `motifs.gather_scatter.gather_width`, `motifs.gather_scatter.scatter_width`, `flow.total_inflow`, `flow.total_outflow`, `temporal.event_ordering` |
| `scatter_gather` | `motifs.scatter_gather.width`, `flow.total_outflow`, `temporal.span_hours`, `focal_entity.role` |
| `cycle` | `motifs.cycle.length`, `temporal.span_hours`, `flow.total_outflow` |
| `bipartite` | `motifs.bipartite.left_size`, `motifs.bipartite.right_size`, `structure.density`, `flow.total_inflow` |
| `stack` | `motifs.stack.depth`, `temporal.span_hours`, `flow.total_outflow`, `focal_entity.role` |
| `random` | `temporal.span_hours`, `flow.total_inflow`, `flow.total_outflow`, `labels.n_illicit_counterparties` |
| `unclassified` | `temporal.span_hours`, `flow.total_inflow`, `flow.total_outflow`, `labels.n_illicit_counterparties`, `focal_entity.role` |

`unclassified` is also the fallback for any typology without a list of its own. It is the
longest list deliberately: with no structural story to tell, the narrative has to carry its
weight on the quantitative facts.

## Reasoning behind the choices

Each list answers "what would an investigator need in order to decide whether to escalate
this case, given that it is *this* kind of case".

- **The fans** need their width, because the width is the finding. A fan-out narrative that
  says "funds were dispersed" without saying to how many accounts has described nothing an
  analyst can act on. They also need a duration, because nine transfers over three weeks and
  nine over three hours are different cases.
- **The two-sided composites** need *both* widths. A gather-scatter narrative quoting only
  the gather side has described a fan-in. `gather_scatter` additionally requires
  `temporal.event_ordering`, because the ordering is what separates genuine
  collect-then-disperse from ordinary two-way activity that happens to have counterparties on
  both sides — and on real data that ordering is frequently `interleaved`, which is exactly
  the finding a narrative should not quietly omit.
- **`cycle`** needs its length: a three-account round trip and a seven-account loop are
  different scenarios, and the length is the only thing distinguishing them.
- **`bipartite`** needs both side sizes and the density, because bipartiteness is a claim
  about the whole shape rather than about one account, and a reader cannot check it from a
  focal-entity description alone.
- **Everything with money** requires a directional total on the side the typology is about —
  outflow for dispersal patterns, inflow for aggregation ones. Requiring both everywhere
  would penalise narratives for padding.
- **`labels.n_illicit_counterparties`** appears wherever a count of flagged counterparties is
  the primary reason the case is suspicious at all.

## What salience is *not*

It is not a faithfulness check. A narrative can mention every salient field and state all of
them wrongly; that is what `facts.checkers` measures, and the two are reported separately. It
is also not a cap — mentioning fields outside the list is not penalised, provided the claims
are supported.

## Changing a list

A reviewed decision requiring an entry in `DECISIONS.md`, for the reason at the top of this
document. `tests/unit/test_facts_coverage.py::test_every_salience_path_is_a_real_field`
prevents a list from naming a field that does not exist, which would otherwise make a
requirement permanently unsatisfiable and silently depress every adequacy score.
