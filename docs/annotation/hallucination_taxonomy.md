# The hallucination taxonomy: nine classes, three critical

**Status:** frozen in Phase 3 (2026-08-01). The machine-readable source of truth is
`src/g2t_aml/facts/taxonomy.py`; `taxonomy_table()` emits this table, so the paper, the
annotation guidelines and the code cannot drift apart.

Every adverse verdict from `facts.checkers` carries one of these classes. Annotators
adjudicating a disputed automated verdict, or labelling a Gold-tier narrative, assign from
the same nine.

## The classes

| ID | Class | Severity | What it covers |
|---|---|---|---|
| **H1** | Entity fabrication | High | Names a counterparty, account or institution not in the case subgraph. Checkable exactly against `entity_inventory.node_ids`. |
| **H2** | Numeric error | High | A count, amount, degree, width or share disagreeing with the record beyond the declared tolerance for its claim type. |
| **H3** | Temporal error | Medium | Wrong ordering of events, an invented duration, or a timestamp outside the case window. |
| **H4** | **Attribution fabrication** | **Critical** | Assigns a business type or real-world identity — "mixer", "exchange", "shell company". |
| **H5** | Typology error | Medium | Names a typology the record does not carry, or asserts an *inferred* typology without the required hedge. |
| **H6** | **Regulatory fabrication** | **Critical** | Cites a threshold, rule or obligation outside the whitelist in `vocab_v1.yaml`. |
| **H7** | **Guilt overclaim** | **Critical** | Asserts guilt, criminality or proof rather than suspicion. |
| **H8** | Unsupported inference | Medium | Motive, intent, off-graph context, or the completeness of a scheme the case only partly contains. |
| **H9** | Omission of exculpatory fact | Medium | Leaves out a fact in the record that materially weakens the suspicion. |

## Why H4, H6 and H7 are reported separately

They aggregate into a **Critical Error Rate**, published independently of overall
faithfulness rather than averaged into it.

A narrative that gets a count wrong needs an edit. A narrative that calls an address a
"mixer", cites a regulation that does not exist, or states that an account holder *is*
laundering money is one that, filed as-is, would expose the institution — the first two
misrepresent evidence to a regulator, and the third asserts a criminal finding that a SAR is
legally not entitled to make. Rolling these into a single percentage lets a system with a 2%
critical-error rate look identical to one with 0%, and that difference is the difference
between deployable and not.

They also share a mechanism, which is why the vocabulary attacks them directly. Each is an
assertion the substrate cannot license *at all*, rather than a value read off it incorrectly.
There is no case, on either substrate, for which "this address is a mixer" is checkable — so
the words are excluded from the controlled vocabulary rather than merely flagged afterwards
(D-029). The checker still catches them via `check_narrative_text`, because defence in depth
is appropriate at this severity.

## The three verdicts

Adverse classes attach to two of the three verdicts, and the distinction matters when
adjudicating.

- **CONTRADICTED** — the record says otherwise. The claim was checkable and failed.
- **UNVERIFIABLE** — the record cannot speak to it: the field is under an availability
  sentinel, the phrase resolves to no measurement, or the substrate lacks the required flag.
- **SUPPORTED** — verified within the declared tolerance. Carries no class.

**UNVERIFIABLE is never a softer CONTRADICTED.** It is the diagnostically most valuable
bucket, because it collects exactly the compliance-dangerous claims: assertions about masked
facts, unsupported attributions, vague intensifiers that measure nothing. A system with high
SUPPORTED *and* high UNVERIFIABLE has learned to say impressive things the graph cannot back.
Annotators should not resolve an unverifiable claim into either neighbour for tidiness — see
D-028.

## H9 is the odd one

Every other class is detected by something the narrative *asserted*. H9 is detected by
something it left out, which makes it the only class that cannot be found by checking claims
one at a time. It is assessed against the salience lists (`docs/annotation/salience.md`) plus
annotator judgement about materiality: a licit-counterparty majority, an ordinary payment
format, or a benign `event_ordering` are all facts in the record that weaken the suspicion,
and omitting them shapes a reader's conclusion as surely as a false statement would.

Because it is judgement-bound, H9 is the class where inter-annotator agreement should be
watched most closely in Phase 6.

## Tolerance, for adjudicating H2 and H3

An H2 or H3 finding is only valid outside the published tolerance (D-027):

| Claim type | Tolerance |
|---|---|
| Counts | Exact. |
| Monetary amounts | 1% relative, 0.01 absolute floor. |
| Durations | Within one unit of the granularity **the narrative itself states**. |
| Categorical | Exact, against the controlled vocabulary. |

The duration rule is asymmetric on purpose. "About three days" against 76 hours is
SUPPORTED — the narrative claimed a precision of one day and is right at that precision.
"76 hours" against 80 hours is CONTRADICTED. The same four-hour error resolves differently
depending on how precisely it was stated, which is the only rule that neither punishes
appropriate vagueness nor waves through a real error.
