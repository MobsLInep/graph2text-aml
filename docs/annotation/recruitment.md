# Annotator recruitment brief

**Graph2Text AML, Phase 6.** For the Gold tier: 350 human-authored SAR narratives, held out
from training, used as the reference standard the whole evaluation rests on.

This document states who we need, what we are asking of them, and — for the paper — the
honest description of the expertise we will actually have. That last section is written
before recruitment rather than after, so that the description cannot drift toward whoever
we happen to find.

---

## 1. What the work is

Writing short structured narratives describing financial-transaction subgraphs that
automated monitoring has flagged. Each item is one case: an interactive graph, a structured
fact record, and a writing box with a four-part template.

- **Volume:** 350 items total, plus 10 calibration items per annotator.
- **Time per item:** ~15 minutes, including reading the case. Measured per item and
  reported; the estimate will be replaced by a measurement after the first week.
- **Total commitment:** roughly 90 person-hours across the set. With three annotators
  that is about 30 hours each, workable over four to six weeks part-time.
- **Format:** a browser interface, self-paced, no fixed hours. Items are queued per
  annotator and progress is saved per item.
- **Training:** ~2 hours reading the guidelines, then 10 calibration items (~2.5 hours)
  scored against reference answers with written feedback before any real work begins.

Annotators write narratives. They do not label data, do not rate model outputs, and are
never shown any system-generated text — see §5.

---

## 2. Who we need

### Required

- **Working familiarity with AML/CFT concepts.** Specifically: what a SAR is and what it is
  for; the distinction between suspicion and a finding of guilt; why layering, structuring
  and placement are described the way they are. This is the non-negotiable requirement.
- **Ability to write clear, structured English prose** to a specification.
- **Willingness to complete calibration and act on feedback.** An annotator who cannot pass
  calibration cannot annotate — see §4.

### Realistic sources, in rough order of preference

1. **Practising compliance staff** — AML analysts, transaction-monitoring investigators,
   financial-crime officers at a bank or fintech. Closest to the real task. Hardest to
   secure and most expensive; expect part-time availability at best.
2. **Forensic accounting practitioners** — accustomed to writing defensible findings from
   transaction data. Strong on the evidential discipline; may need the typology material.
3. **MSc students in financial crime, criminology, forensic accounting or finance**, given
   the full training pack. The most realistic source at this scale and the one we plan
   around. They have the conceptual grounding and need the domain-specific protocol, which
   is what the guidelines and calibration supply.
4. **AML certification holders** (CAMS or equivalent) outside a current compliance role.

### Not acceptable

- **Untrained crowdworkers.** Not a matter of cost or scale. The task requires knowing why
  "the account holder is laundering money" is a materially different sentence from "the
  activity appears consistent with layering" — and a crowdworker's plausible-looking
  narrative would enter the reference standard indistinguishably from a good one. **A Gold
  set built this way does not survive review**, and it should not: it would be a set of
  guesses used to score a model and reported as expert judgement.
- **Anyone who has worked on the model side of this project.** The Gold set must be
  independent of the system it evaluates.
- **Anyone who cannot commit to the calibration step.**

### How many

**Three annotators, minimum two.** Two is the floor because 15% of items are
double-annotated and inter-annotator agreement over a single annotator is undefined. Three
gives a workable ~120 items each and lets one person's schedule slip without stalling the
phase.

Plus **one reviewer** for the second-reader pass, and **one adjudicator** who is neither an
author nor a reviewer of any item they decide. The project lead writes the calibration
reference answers and can act as adjudicator, but not as reviewer of items whose calibration
they authored.

---

## 3. What we provide

- **`annotation_guidelines.md`** — the full protocol: what a SAR is, the four-part
  structure, the eight typologies with diagrams, the six rules, five fully worked examples
  and the error taxonomy with wrong/right pairs. Also released with the corpus.
- **The interface**, which does a substantial part of the work: it shows the graph and the
  fact record, marks which facts this case's typology requires, counts tokens, flags
  forbidden phrasing as you type, and checks the finished narrative against the fact record
  before you submit.
- **Calibration with written, per-dimension feedback** naming the specific items and the
  specific difference, not a score.
- **A named contact** for questions, and a comment box on every item.

Annotators are compensated at a rate appropriate to the expertise required. Nothing in this
protocol is conditioned on output volume: paying per item creates pressure to write quickly,
and the per-item timings would then measure that pressure rather than the task.

---

## 4. Calibration is a gate, not an orientation

Every annotator completes 10 calibration items before any of their work counts. Their output
is scored against reference answers written by the project lead on four dimensions, scored
and thresholded **separately**:

| Dimension | Threshold | What it catches |
|---|---|---|
| Typology agreement | 0.70 | The systematic error — someone who confuses scatter-gather with gather-scatter does it every time |
| Salience coverage | 0.70 | Fluent narratives that omit the required facts |
| Hedging compliance | **1.00** | Guilt overclaims. No partial credit |
| Factual accuracy | 0.95 | Claims that disagree with the fact record |

Below any threshold: targeted feedback, then re-calibrate. **An annotator who has not passed
calibration does not annotate, and their calibration items never enter the corpus.**

This is a gate because the alternative is discovering in week six that one annotator has been
systematically mislabelling a typology, at which point their items are unusable and there is
no budget to redo them. Ten items costs each person about two and a half hours. Forty wasted
items costs ten, plus the schedule.

Hedging compliance is set at 1.00 deliberately. A guilt overclaim is a critical error, and
somebody who produces one in ten items will produce them in a hundred.

---

## 5. Independence, and why it constrains what we can offer

**Annotators are never shown any system-generated text** — no model output, no template
rendering, no rewrite, not as a starting draft and not as a "reference". This is enforced in
code: the object the interface loads has no field that can carry a narrative, and the
annotation store refuses a record that carries one.

The reason is that Gold's entire value is being an *independent* reference. A narrative
written next to a draft is a set of edits to that draft, and a system evaluated against it
would be scored on how well it matches its own output. That would make the headline
comparison circular and the phase pointless.

The practical cost is that we cannot make the task easier by pre-filling anything. The
interface compensates with structure — the four-part scaffold, the marked salient facts, the
live flags — none of which is prose.

**Pseudonymity.** Annotators are identified in the data only as `annotator-01`,
`annotator-02` and so on. No name, email or affiliation enters the repository at any point;
the repository forbids real-world identifiers in any artifact, and who wrote which narrative
is one.

---

## 6. The honest description for the paper

This section is the point of writing the brief before recruiting. It is the text that will
appear in the paper's data section, and it is written now so it describes the standard we
set rather than the people we found.

> **Draft, to be completed with the actual profile once recruitment closes.**
>
> The Gold tier comprises N human-authored narratives written by K annotators with
> AML/compliance backgrounds [*state the actual composition: how many practitioners, how
> many students, what qualifications*]. Annotators were not practising SAR filers
> [*or were, if so — state it*]. Each completed a 10-item calibration set scored against
> reference narratives written by the project lead across four dimensions, and no annotator
> began production annotation before meeting the threshold on all four. 15% of items were
> independently double-annotated, assigned deterministically by case identifier before
> annotation began. Every item was checked against its fact record by a second reviewer who
> did not author it; disagreements were logged and adjudicated by a third party, with the
> adjudication recorded. Inter-annotator agreement on typology assignment was κ = [*value*]
> (Krippendorff's α = [*value*]); content-selection overlap over salient fields was
> [*value*]; surface similarity between independent narratives of the same case was
> [*value*], which is low by design and is reported as evidence that SAR narrative writing
> admits substantial legitimate variation.

### The limitations to state plainly, whatever the profile turns out to be

1. **These are not real SARs and these annotators are not filing them.** Real SARs are
   confidential by statute and no corpus of them exists to compare against. What Gold
   measures is expert judgement applied to the same evidence a system sees, under a
   published protocol — not agreement with real filings, which is unobtainable.
2. **If annotators are students rather than practitioners, say so in that sentence, not in a
   footnote.** A trained MSc student following a detailed protocol is a defensible reference
   standard for this task and an indefensible substitute for a practitioner if described as
   one. The claim we can support is "trained annotators working to a published protocol",
   and that is the claim we will make.
3. **The set is small.** 350 items, and about 200 held out as test-only. Adequate as a
   reference standard, inadequate as a training corpus, and we do not use it as one.
4. **Coverage is uneven by construction.** AMLworld only, unless Elliptic2 access is granted
   before annotation closes; the 30% Elliptic2 quota is currently an unmet deficit and is
   recorded as such in the sampling report. Typology strata are capped by supply — the
   AMLworld test split holds only 19 `stack` cases — so "balanced across typologies" means
   evenly allocated under capacity, not equal counts.
5. **Annotators saw the fact record, not the raw data.** They wrote from the same extracted
   facts a system reads, which is what makes the comparison fair, and which also means a
   fact-extraction error would propagate identically into both.

---

## 7. Recruitment status

| | |
|---|---|
| Annotators recruited | **0** |
| Annotators calibrated | **0** |
| Reviewer identified | not yet |
| Adjudicator identified | not yet |
| Target | 3 annotators, 2 minimum, calibrated before production begins |

**This is the phase's critical path.** Every piece of machinery — the interface, the
calibration scoring, the agreement analysis, the ingestion pipeline — is built and tested
against real cases. None of it produces a single Gold narrative until people are recruited.
At ~15 minutes an item and 350 items, a start slipping by a month moves the whole corpus by
a month, and Gold is the credibility anchor of the paper.

Approaches to make, in parallel rather than in sequence:

1. MSc programme convenors in financial crime / forensic accounting / criminology — offer
   the training pack and compensation; these programmes often welcome applied work.
2. Professional bodies and AML practitioner networks for part-time practitioner time.
3. Fintech compliance teams, for staff time under an agreement that no proprietary data is
   involved (the substrates are synthetic and public, which makes this an easy conversation).
4. Forensic accounting practices.

---

*Companion documents: [`annotation_guidelines.md`](annotation_guidelines.md) (the protocol),
[`salience.md`](salience.md) (what an adequate narrative must mention),
[`hallucination_taxonomy.md`](hallucination_taxonomy.md) (the nine error classes).*
