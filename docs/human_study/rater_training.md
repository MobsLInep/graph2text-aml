# Rater training pack

**Study:** Decision-setting evaluation of generated SAR narratives
**Version:** 1.0 · **Date:** 2026-08-05
**Time required:** 30 minutes, and it is paid.

---

## Before you start

You will read ~60 short drafts, rate each on five scales, decide whether you would file
it, and edit it until you would. The software times you and records what you changed.

**Two things to know now, because they change how you should work:**

1. **The timer pauses when you switch tab.** You do not need to rush, and stepping away
   does not spoil an item. Read at your natural pace.
2. **Rate before you edit.** The interface shows the draft, then the scales, then the edit
   box, in that order, deliberately. If you fix the text first, your ratings describe the
   text *you* produced rather than the one you were given, and the measurement is lost.

You will not be told which system wrote any draft. Some are from a simple template, some
from a large language model, some from the system this project is building. They are
deliberately indistinguishable to you.

---

## Part 1 — SARs in five minutes

A **Suspicious Activity Report** is filed by a regulated firm when it identifies activity
that may indicate money laundering. It has a structured part (who, what, when, how much)
and a **narrative** — free prose explaining what was observed and why it is suspicious.
The narrative is what this study is about.

It has two readers, and they want different things:

- **The internal reviewer**, who decides whether to file at all. They want to know the
  facts are right and the case holds together.
- **The financial intelligence unit**, who may act on it. They want enough specificity to
  do something with it — which accounts, what pattern, what scale, what window.

A good narrative serves both. A narrative that says "suspicious activity was detected on
the account" serves neither.

### The one rule that matters most

**A SAR reports suspicion. It does not assert guilt.**

The filer is not a court and has not proved anything. The narrative describes *observed
activity* and explains why it is *consistent with* a known laundering pattern. It never
says the subject laundered money, never states intent, and never reaches a legal
conclusion.

| Wrong | Right |
|---|---|
| "The subject laundered £2.4m through nine accounts." | "The subject dispersed £2.4m to nine accounts within 22 hours, a pattern consistent with layering." |
| "This is a shell company." | "The counterparty received funds from six sources and made no outgoing payments during the window." |
| "The account holder was knowingly involved." | "No legitimate business rationale for the pattern is apparent from the transaction data available." |
| "Account X is a mixer." | "Account X received funds from 14 distinct sources and disbursed to 12 distinct destinations within the window." |

The last one matters more than it looks. Calling something a "mixer", an "exchange" or a
"shell company" is a claim about a real-world business identity that the transaction data
cannot support. It is one of the three error classes this project treats as critical.

---

## Part 2 — the four-part structure

Every narrative in this study is meant to have four sections. Some will not. That is
information, and it belongs in your **completeness** rating.

| Section | What belongs in it |
|---|---|
| **SUBJECT & SCOPE** | Which account the report is about, the observation window, and the size of what was examined |
| **ACTIVITY OBSERVED** | What actually happened: counts, volumes, directions, counterparties, timing. Facts only |
| **PATTERN & TYPOLOGY** | What pattern this resembles, hedged appropriately, and why |
| **BASIS & ACTION** | Why this warrants a report and what is recommended |

---

## Part 3 — the eight typologies

You will meet these named in the drafts. You do not need to detect them yourself — the
fact panel tells you what the case is — but you need to know whether a draft has used the
name correctly.

| Typology | Shape | Note |
|---|---|---|
| **fan-out** | One account sends to many | Dispersal. Common in layering |
| **fan-in** | Many accounts send to one | Consolidation |
| **gather-scatter** | Many in, then many out, via one account | The classic layering pass-through |
| **scatter-gather** | Out to many, then back to one | Often placement |
| **cycle** | Funds return to the origin | Strongly suspicious; no ordinary business reason |
| **bipartite** | Two distinct groups transacting across | Often a mule network |
| **stack** | Layered chains through intermediaries | Depth rather than width |
| **random** | No clean motif | The residual class |

**Two cautions that will come up:**

- **Fan-in and fan-out are not suspicious by themselves.** A payroll account fans out
  every month. A retailer fans in all day. The suspicion comes from the combination with
  timing, value, counterparty history and the absence of a business rationale. A draft
  that treats the shape alone as proof is overclaiming — that belongs in your
  **regulatory tone** rating.
- **A case may only be *part* of a laundering stream.** The extraction window caps at 48
  hours and keeps about 65% of a stream's transactions on average. A draft that describes
  the scheme as complete — "the full laundering cycle was executed" — is claiming more than
  the data shows.

---

## Part 4 — the five scales, and what the numbers mean

**This is the most important section of this pack.** A "7" has to mean the same thing to
you as it does to every other rater, or the agreement statistic we compute is measuring
nothing. The anchors below are the definitions. They are identical to the text shown in
the interface — you do not have to remember which document was authoritative.

Use the whole scale. If everything you see is a 5 or a 6, the study learns very little.

### 1. Factual correctness

*Is every factual assertion in this narrative supported by the record shown?*

| | |
|---|---|
| **1** | At least one assertion is contradicted by the record: a wrong amount, a wrong count, a wrong direction of flow, or a counterparty that does not appear. |
| **4** | No assertion is contradicted, but at least one cannot be checked against the record and would have to be verified before filing. |
| **7** | Every assertion is supported by the record, with no rounding, direction or attribution errors. |

Check against the **fact panel on the left**, not against your professional intuition
about what is plausible. If the panel says six counterparties and the draft says six, that
is correct even if six seems low for the pattern.

### 2. Completeness

*Are the material facts present?*

| | |
|---|---|
| **1** | A fact a reviewer would need to reach a decision is missing: the pattern, the volume, the counterparties, or the exculpatory context. |
| **4** | The main pattern is stated but a supporting detail a reviewer would ask for is absent. |
| **7** | Every material fact is present, including the exculpatory ones. A reviewer would ask no follow-up question answerable from the record. |

**"Including the exculpatory ones" is doing real work here.** If the fact panel says the
subject account carries no illicit label, or that no burst was detected, and the draft
never mentions it, the draft is presenting a one-sided case. That is an omission, and it
should cost completeness marks. It is one of the specific things this study exists to
detect.

### 3. Actionability

*Could an investigator act on this?*

| | |
|---|---|
| **1** | Says something happened without saying what, to whom, or in what pattern. No next step follows from it. |
| **4** | The activity is identifiable but the investigator would have to return to the raw data to decide anything. |
| **7** | States the pattern, its participants and its scale precisely enough that the next investigative step is obvious from the text alone. |

Ask yourself: if this landed on your desk with no attachments, would you know what to do
next?

### 4. Readability and professional register

*Does this read as professional financial-crime writing?*

| | |
|---|---|
| **1** | Ungrammatical, repetitive, or written in a register no compliance function would send: marketing language, chat register, or machine-listing style. |
| **4** | Clear and correct but flat or formulaic; a reviewer would rewrite it before sending. |
| **7** | Reads as competent professional prose. Could be sent after a proofread. |

A template-generated narrative that is correct but reads like a filled-in form belongs
around **4**, not **1**. Reserve 1–2 for text that is genuinely hard to read or in the
wrong register entirely.

### 5. Regulatory tone appropriateness

*Does it report suspicion rather than assert guilt?*

| | |
|---|---|
| **1** | Asserts criminality as fact: calls the activity money laundering, names the subject as a launderer, or states intent. |
| **4** | Mostly hedged, with at least one sentence that overstates what the data supports. |
| **7** | Describes observed activity and why it is consistent with a typology, without asserting intent, guilt or a legal conclusion anywhere. |

Also score down here for: attributing a real-world business identity ("mixer", "shell
company"), citing a regulation or threshold that does not apply, and describing an
inferred pattern as established fact.

### 6. Would you file this after review? (yes / no)

A binary, and it is not a summary of the five scales. The question is practical: **after a
reasonable review pass, would this go out?** A draft with one fixable numeric error might
still be a yes. A draft that asserts guilt is a no however well written it is.

### 7 and 8. Time and edits — you do not enter these

The interface records how long you spend, and compares the draft you were shown against
the version you leave in the edit box. You do not have to do anything for either. Just
edit until you would file it, and no further — the measurement is "what did this need",
not "how would I have written it".

**Please do not rewrite to taste.** If the draft is filable as-is, change nothing.

---

## Part 5 — three calibration items

Work these before you start. Rate each, then read the feedback.

### Calibration item 1

> **Fact panel says:** focal account `BANK001|ACC4471`; 6 distinct counterparties received
> funds; total outflow 48,200 USD; window 22 hours; burst detected; typology fan_out
> (from ground truth); subject account carries no illicit label.

> **Draft:** "SUBJECT & SCOPE. This report concerns account BANK001|ACC4471 over a 22-hour
> window. ACTIVITY OBSERVED. The account dispersed 48,200 USD to six distinct
> counterparties. The transactions occurred in a concentrated burst. PATTERN & TYPOLOGY.
> The activity is consistent with a fan-out dispersal pattern. BASIS & ACTION. The
> concentration of outbound transfers to multiple new counterparties within a short window
> warrants further review."

**Rate it now, then read on.**

<details>
<summary>Feedback</summary>

**Suggested: factual 7 · completeness 4 · actionability 6 · readability 6 · tone 7 · file: yes**

- **Factual 7** — every number matches the panel. Nothing is asserted that cannot be
  checked.
- **Completeness 4, not 7** — the panel says *the subject account carries no illicit
  label*, and the draft never mentions it. That is exculpatory context a reviewer needs,
  and its absence makes the case look one-sided. This is the single most common defect in
  this study and the easiest to miss, because nothing in the text looks wrong. **You have
  to read the panel for what is missing, not just check what is present.**
- **Tone 7** — "consistent with", "warrants further review". No guilt asserted.
- **Readability 6** — competent, slightly mechanical.

If you gave completeness a 6 or 7, recalibrate: absence of an exculpatory fact is an
omission even when everything present is correct.
</details>

### Calibration item 2

> **Fact panel says:** focal account `BANK003|ACC8812`; 4 distinct counterparties;
> total inflow 91,000 EUR; window 31 hours; no burst detected; typology gather_scatter
> (**inferred** from motif detection, not ground truth).

> **Draft:** "SUBJECT & SCOPE. Account BANK003|ACC8812 is the subject of this report.
> ACTIVITY OBSERVED. The account received 91,000 EUR from four counterparties and
> redistributed the funds. PATTERN & TYPOLOGY. This is a gather-scatter layering scheme.
> The account is a shell company used to obscure the origin of criminal proceeds. BASIS &
> ACTION. The subject has laundered funds through this structure and the account should be
> frozen."

**Rate it now, then read on.**

<details>
<summary>Feedback</summary>

**Suggested: factual 4 · completeness 4 · actionability 4 · readability 5 · tone 1 · file: no**

Three separate critical problems, all in the last two sections:

- **"is a shell company"** — a real-world business identity claim the transaction data
  cannot support. Critical.
- **"has laundered funds"** and **"criminal proceeds"** — asserts guilt as established
  fact. This is the thing a SAR is legally not entitled to do. Critical.
- **"This is a gather-scatter layering scheme"** — the panel says the typology was
  **inferred**, not read from ground truth. Asserting it flatly, with no hedge, overstates
  what is known.

**Tone must be 1.** Any one of these would put it at 1–2; all three together leave no
argument.

**Factual 4, not 1** — this is the distinction that matters. The *numbers* are all correct:
91,000 EUR, four counterparties. Nothing is contradicted by the record. The problem is
that unsupported claims are made *on top of* correct facts, and those are unverifiable
rather than contradicted. Factual correctness asks "is what it says supported"; regulatory
tone asks "is it entitled to say it". Keep the two separate — a draft can be numerically
perfect and still be unfilable.

**File: no.** Not close.
</details>

### Calibration item 3

> **Fact panel says:** focal account `BANK002|ACC1039`; 9 distinct counterparties sent
> funds; total inflow 12,400 USD; window 6 hours; burst detected; typology fan_in;
> 3 of 9 counterparties carry an illicit label.

> **Draft:** "Account BANK002|ACC1039 received 12,400 USD from nine counterparties over
> six hours. Three of the nine have previously been associated with illicit activity. The
> rapid consolidation of funds from multiple sources, several with adverse history, is
> consistent with a fan-in consolidation pattern and merits review."

**Rate it now, then read on.**

<details>
<summary>Feedback</summary>

**Suggested: factual 7 · completeness 5 · actionability 6 · readability 6 · tone 7 · file: yes**

- **Factual 7** — all four figures match, including the 3-of-9 illicit count, which is the
  most material fact in the case and is correctly stated.
- **Completeness 5** — everything material is present, but the four-part structure is
  absent: no headings, and BASIS & ACTION is compressed into "merits review". A reviewer
  would want the basis stated. Structure is part of completeness in this study.
- **Readability 6** — genuinely well written, arguably better prose than item 1. Do not
  let good prose pull the other scales up with it; that is the halo effect this study is
  designed to detect, and it is the main reason ratings across dimensions are collected
  separately rather than as one overall score.
- **Tone 7** — "associated with", "consistent with", "merits review". Correctly hedged
  throughout.

**If you rated readability high and let completeness follow it, that is worth noticing.**
The dimensions are meant to move independently.
</details>

---

## Part 6 — practicalities

- **Save and resume:** your progress saves on every submission. Close the window whenever
  you like; the link returns you to the next unrated item.
- **Sessions:** we suggest no more than an hour at a time.
- **Repeated items:** a small number of items appear twice. This is deliberate and it
  checks whether the *panel* is consistent — it is not a test of you, and no individual
  result is computed or reported. Rate the second showing as you find it; do not try to
  remember what you said the first time.
- **Do not type anything confidential** into the edit box or the comment field. An
  automated scanner checks every edit and withholds anything that looks like a real
  identifier.
- **Questions:** [RESEARCHER NAME], [EMAIL]. Ask at any point, including mid-study.

Thank you. The two measurements this study exists to produce — how long a draft takes to
make usable, and how much of it has to change — cannot be obtained any other way.
