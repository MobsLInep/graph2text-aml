# Annotation guidelines: writing Gold-tier SAR narratives

**Graph2Text AML, Phase 6.** Version 1.0.0, 2026-08-03.

This is the document you work from. It is also released with the corpus and cited in the
paper, because a human-annotated set whose protocol is not published is a set nobody can
evaluate or reproduce.

Read Parts A to D before your calibration items. Keep Part C (the typologies), Part D (the
rules) and Part F (the common errors) open while you write.

**The two documents this one rests on were frozen before any narrative existed**, in Phase
3: [`salience.md`](salience.md) — what an adequate narrative must mention, per typology —
and [`hallucination_taxonomy.md`](hallucination_taxonomy.md) — the nine error classes. They
were fixed first so that the standard you are held to and the standard the automated metric
applies are one standard rather than two that drifted apart. Nothing in this document
overrides either of them.

---

## Contents

- [Part A — What a SAR is, and the one rule that matters most](#part-a)
- [Part B — The four-part structure](#part-b)
- [Part C — The eight typologies](#part-c)
- [Part D — The rules](#part-d)
- [Part E — Five worked examples](#part-e)
- [Part F — Common errors, wrong and right](#part-f)
- [Appendix — Allowed hedges, salience lists, quick reference](#appendix)

---

<a name="part-a"></a>

## Part A — What a SAR is, and the one rule that matters most

### A.1 What you are writing

A **Suspicious Activity Report** is the document a financial institution files when
monitoring — automated, human, or both — surfaces activity that may indicate financial
crime. In most jurisdictions filing is a legal obligation, filing decisions are made under
time pressure, and the institution is protected from liability for filing in good faith.

The narrative section is the part of a SAR that a human being actually reads. Structured
fields carry account numbers and amounts; the narrative carries the *account of what
happened* — what the institution observed, in what order, and why it thought the activity
warranted a referral.

**It has two readers, and they want different things.**

The **investigator**, at a financial intelligence unit, may see thousands of these. They
are deciding, in a few minutes, whether this one is worth opening. They need the shape of
the activity, the size of it, the timing, and what specifically made it look wrong — fast,
and in a form they can act on. A narrative that says "unusual patterns were observed" has
told them nothing and has wasted their few minutes.

The **regulator**, potentially, in a supervisory examination years later. They are asking
whether the institution's filings were justified, consistent and accurate. Everything you
write may be re-read by someone checking it against the underlying data, in a context where
being wrong is a supervisory finding.

Those two readers give you the standard: **specific enough to act on, and defensible enough
to re-read.**

### A.2 The critical distinction: suspicion is not guilt

**A SAR reports suspicion. It never reports guilt.**

This is not a matter of tone, politeness or hedging style. It is what the document legally
is. A SAR is a *referral for investigation* — a statement that activity was observed which
may warrant further enquiry. It is not a finding, not an accusation, and not evidence of a
crime. The institution filing it has not established that anything unlawful occurred and is
not entitled to say that it has.

Writing that an account holder *is* laundering money does three things, all bad. It states
a criminal finding the filer has no authority to make. It misrepresents the evidential
weight of the filing to whoever reads it next. And, if the person is later found to have
done nothing wrong, it is a written record of the institution having accused them.

In this project's taxonomy that is **H7, Guilt overclaim, rated Critical**, and it is
reported in the Critical Error Rate independently of everything else.

**The pattern to internalise: describe the activity as fact, and the interpretation as
possibility.**

| ✗ Do not write | ✓ Write instead |
|---|---|
| The account holder is laundering money. | The observed activity appears consistent with layering. |
| These transfers prove structuring. | The transfer amounts are consistent with structuring and warrant further review. |
| The subject is clearly a money mule. | The account received funds from three flagged counterparties and dispersed them within 22 hours, a pattern that merits further enquiry. |
| This confirms the account is part of a laundering network. | The account is one hop from a counterparty flagged in the source data. |
| The criminal moved 40,000 USD through nine accounts. | The subject account moved 40,000 US Dollar to nine accounts. |
| Funds were deliberately structured to evade reporting. | Nine transfers fell below the USD 10,000 reporting threshold. |
| The account is definitely part of the scheme. | The account's activity is indicative of the pattern described. |

Notice what changes in the right-hand column. The **facts stay exactly as strong** — the
amounts, the counts, the timings are stated flatly and without hedging. What softens is
only the **inference**. Hedging your facts is a different error (it makes the narrative
useless); hedging your conclusions is the requirement.

**Hedge the conclusion. Never hedge the number.**

| ✗ Over-hedged, useless | ✓ Correct |
|---|---|
| The account may have received approximately some funds. | The account received 296,100 Mexican Peso. |
| It is possible that several accounts were involved. | Eight accounts received funds from the subject. |
| There might have been a transfer around some point. | The largest transfer, 7,021 Canadian Dollar, occurred on 2022-09-10. |

You may hedge **only** with a phrase from the allowed list (Appendix A.1). That list is
closed. It exists so that "suspicion" has one vocabulary rather than fifty, and so the
distinction between a report and an accusation is machine-checkable rather than a matter of
someone's ear.

### A.3 Two more things about *these* SARs specifically

**You are describing a window, not a scheme.** Each case is a subgraph extracted around a
seed account within a bounded time window. On average a case contains about **65%** of the
transactions in the laundering stream it came from, and only about 28% of streams are
captured in full. So even where the source data labels a case as, say, `fan_out`, the case
in front of you may show only part of that shape. Never describe what you see as the
complete scheme, the whole network, or the full extent of anything.

**Everything is synthetic or anonymised.** AMLworld is a synthetic dataset; Elliptic2 is
real Bitcoin data with anonymised identifiers. No real person or institution appears in any
case, and the account identifiers are not real accounts. This does not lower the standard —
the narratives are the reference standard an automated system is measured against, so an
error here becomes an error in the published numbers.

---

<a name="part-b"></a>

## Part B — The four-part structure

Every narrative has four parts, in this order, each with its own heading:

```
[1] Subject & Scope

[2] Activity Observed

[3] Pattern & Typology

[4] Basis & Action
```

Write the headings exactly as shown, in title case. (Not upper case: the automated
identifier scanner treats long runs of capitals as possible bank codes, and `ACTIVITY
OBSERVED` trips it.)

**Target length: 80–400 tokens**, roughly 60–300 words. The interface counts for you. A
narrative outside those bounds fails validation at ingestion — under 80 it is a caption,
over 400 it is not being read by a busy investigator.

### [1] Subject & Scope

*Who this is about and how much you are looking at.*

**Belongs here:** the focal account identifier; its role inside the case (originator,
beneficiary, intermediary, pass-through, hub, isolated); the number of accounts and
transactions in the reviewed subgraph; the number of institutions, if available.

**Does not belong here:** any interpretation, any conclusion, any amount, the typology.

> Account 004403|80F962B10 is the subject of this report and acts as the originating
> account within the reviewed activity. The reviewed subgraph comprises 9 accounts
> connected by 12 transactions across 9 institutions.

### [2] Activity Observed

*What happened, in facts, with no interpretation at all.*

**Belongs here:** counterparty counts in and out; transaction counts; amounts, per
currency; the largest single transfer; the observed time span and the window; any burst;
the ordering of phases; payment formats.

**Does not belong here:** the typology name, the word "suspicious", any conclusion, any
comparison to typical behaviour. If a sentence here could not be checked against the fact
panel line by line, it is in the wrong section.

This section is where an over-cautious annotator does the most damage. State the numbers
flatly.

> Over an observed period of 26.9 hours, from 2022-09-13 08:22 to 2022-09-14 11:16, the
> subject account sent 30,060 Euro across 4 transfers and 29,600 US Dollar across 4
> transfers, to 8 distinct counterparties, and received nothing within the reviewed window.
> Ten of the twelve transactions fall inside a 23.7-hour burst. All settlement used ACH.

### [3] Pattern & Typology

*What shape the activity has, and what that shape is consistent with.*

**Belongs here:** which structural detectors fired and with what dimensions; **which ones
did not**; the typology label and whether it is ground truth or inferred; a hedged
statement of what the pattern is consistent with.

**Does not belong here:** anything about intent or motive; any business-type attribution;
any claim about the completeness of a scheme.

This is the section where hallucinations concentrate. Everything here must trace to a
detector result or to the typology field in the panel.

> The subject account dispersed funds to 8 distinct recipients within the reviewed window,
> a fan-out of width 8. No cycle, stack, gather-scatter or scatter-gather structure was
> detected at the configured thresholds. The case carries the ground-truth typology label
> fan_out, and the observed dispersal appears consistent with that pattern.

### [4] Basis & Action

*Why this was flagged, what weakens it, and what should happen next.*

**Belongs here:** counterparty label counts — flagged **and unflagged**; flagged
transaction counts; proximity to flagged accounts; **any fact in the record that weakens
the suspicion**; a hedged recommendation.

**Does not belong here:** a filing decision, a regulatory finding, a statement about what
the institution will do, anything about a real filing obligation.

**Omitting an exculpatory fact is an error in its own right** — class H9. If the majority of
counterparties are unflagged, if the payment formats are ordinary, if the event ordering is
benign, say so. A narrative that presents only the incriminating half of a record shapes the
reader's conclusion as surely as a false statement would.

> All 8 counterparties appear on at least one transaction flagged in the source data, and 8
> transactions in the reviewed subgraph carry that flag; the subject account is itself
> flagged. No transfers fell close to the USD 10,000 reporting threshold, and settlement
> used a single ordinary payment rail. The activity warrants further review by an
> investigator with access to customer records not represented in this subgraph.

---

<a name="part-c"></a>

## Part C — The eight typologies

Eight named laundering typologies, plus `unclassified` for everything else. On AMLworld the
label is **ground truth**, carried on the case's own transactions. On Elliptic2 there is no
ground truth and any typology is **inferred** from structure alone — and must be hedged.

### C.0 The warning that matters most

> **Fan-in and fan-out appear in perfectly normal activity.** A payroll account fans out to
> every employee, every fortnight. A retail merchant's settlement account fans in from
> hundreds of customers. A landlord's account fans in monthly from tenants and fans out to a
> mortgage lender.
>
> **Structural presence is not evidence of anything on its own.** What makes a fan-out
> suspicious is the *combination* — the counterparties are freshly created, or flagged, or
> the funds arrive and leave within hours, or the amounts sit just under a reporting
> threshold, or the recipients have no other activity. The shape is where you start
> looking, never where you finish.
>
> Roughly a quarter of the cases you will annotate are **hard negatives**: legitimate
> activity deliberately selected *because* it has a suspicious-looking shape. They are in
> the set precisely to see whether a system — and a human — can decline to escalate. Writing
> a confident suspicion narrative about a payroll run is the single most damaging error
> available to you.

### C.1 The eight, plus unclassified

#### fan_out — one account to many

```
              ┌──▶ B
      A ──────┼──▶ C
              ├──▶ D
              └──▶ E
```

One account disperses to several recipients. **In the panel:** `fan-out: detected`, with
`width` (how many recipients) and `window hours`. The focal account's *out-degree* is high
and its in-degree is often zero.

**Legitimately:** payroll, supplier payments, dividend distribution, an individual paying
bills.
**Suspiciously:** the dispersal phase of layering — funds arriving from one source and being
split across many accounts to break the audit trail.
**What separates them:** timing (hours rather than a monthly cycle), the recipients'
characteristics, and whether the funds came in immediately beforehand.

#### fan_in — many accounts to one

```
      B ──────┐
      C ──────┼──▶ A
      D ──────┤
      E ──────┘
```

The mirror image: several accounts pay into one. **In the panel:** `fan-in: detected`, with
`width`. The focal account's *in-degree* is high.

**Legitimately:** merchant settlement, rent collection, crowdfunding, a joint account.
**Suspiciously:** aggregation — collecting the proceeds of many small placements before
moving them onward as one.
**What separates them:** whether the funds then leave promptly and as a block.

#### gather_scatter — collect, then disperse

```
      B ──┐                  ┌──▶ F
      C ──┼──▶  A  ────────▶ ┼──▶ G
      D ──┘                  └──▶ H
```

One hub collects from several sources and then disperses to several destinations. **In the
panel:** `gather-scatter: detected`, with `gather width` and `scatter width`.

**The thing that distinguishes it from ordinary two-way activity is the *ordering*.** Look
at `Event ordering`. A genuine collect-then-disperse reads `inflow phase then outflow
phase`. If it reads `interleaved`, the account was doing both at once — which is what an
ordinary business account does all day, and is a materially weaker finding. **Salience
requires you to state the ordering** for this typology, precisely so it cannot be quietly
omitted. On real data it is frequently `interleaved`, and saying so is the honest report.

#### scatter_gather — split, then recombine

```
              ┌──▶ B ──┐
      A ──────┼──▶ C ──┼──▶ E
              └──▶ D ──┘
```

One origin splits funds across several intermediaries which then recombine at one
destination. **In the panel:** `scatter-gather: detected`, with `width`.

**Legitimately:** rare in ordinary retail banking; occurs in structured settlement.
**Suspiciously:** a classic layering construction — the split-and-rejoin exists to make the
path between origin and destination hard to follow.
**Do not confuse it with gather-scatter.** Gather-scatter is *many-in then many-out through
one hub*; scatter-gather is *one-out through many then many-in at one point*. This is the
confusion pair that calibration exists to catch.

#### cycle — funds return to their origin

```
      A ──▶ B ──▶ C ──▶ D
      ▲                 │
      └─────────────────┘
```

A directed loop. **In the panel:** `cycle: detected`, with `length` (how many accounts in
the loop). Salience requires the length: a three-account round trip and a seven-account loop
are different scenarios and the length is the only thing separating them.

**Legitimately:** intra-group treasury movements; an individual moving money between their
own accounts.
**Suspiciously:** round-tripping to create apparent transaction history or apparent
legitimacy.

#### bipartite — two groups, links only between them

```
      A ──┐  ┌──▶ D
      B ──┼──┼──▶ E
      C ──┘  └──▶ F
```

The accounts split into two sets with transactions only ever running between the sets, never
within one. **In the panel:** `bipartite: detected`, with `left size`, `right size` and
`score`.

**Legitimately:** any two-sided market — customers and merchants, employers and employees.
It is extremely common and is weak evidence on its own.
**Suspiciously:** a layer of a structured scheme, where one set of accounts exists only to
feed another.
Salience requires **both side sizes and the density**, because bipartiteness is a claim
about the whole shape and a reader cannot check it from a focal-account description.

#### stack — consecutive layers

```
      A ──┐        ┌──▶ D ──┐        ┌──▶ G
      B ──┼──▶ C ──┼──▶ E ──┼──▶ F ──┼──▶ H
          └        └──▶ ... └        └
```

Layers of accounts in sequence, each layer at least a minimum width. **In the panel:**
`stack: detected`, with `depth`.

**Suspiciously:** depth is the point — each layer adds a hop between the origin of funds and
their destination. A depth of 3 or more supports the controlled phrase "consistent with
layering".

#### random — no coherent structure, but labelled

Activity that the source data labels as part of a laundering stream but which exhibits no
clean shape. **In the panel:** typology `random`; usually few or no detectors fire.

Salience requires the quantitative facts — span, both flow directions, flagged counterparty
count — because there is no structural story to carry the narrative.

#### unclassified — no typology

The largest group by far, and it covers three quite different situations:

1. **Licit activity.** No laundering stream, nothing flagged. Most of these.
2. **Hard negatives.** Licit activity that *looks* structured. Roughly a quarter of your
   items.
3. **Suspicious activity with no clean shape.** Flagged transactions are present but no
   typology applies.

`unclassified` has the **longest** salience list, deliberately: with no structural story to
tell, the narrative has to carry its weight on the quantitative facts.

### C.2 Recognising a typology from the panel

**Read the panel before you read the graph.** The graph shows you the shape; the panel tells
you what the detectors actually found, at the thresholds the metric uses. Where your eye and
the panel disagree, the panel wins — it is what the narrative will be checked against.

Three habits worth forming:

**Check whether the focal account is the one the shape is about.** A case can carry the
label `fan_out` while the focal account is a *recipient* of the fan-out, with the hub being
some other account. The panel names the hub in the detector line. Worked Example 1 is exactly
this situation, and describing the focal account as "dispersing funds" there would be simply
false.

**Read the "not detected" lines.** They are facts about the case, not blanks. `cycle: not
detected` means the detector ran and found no cycle — that is evidence, and stating it is
often the most useful sentence in Section 3.

**Distinguish "not detected" from "not available".** A detector that did not fire is a
measured absence. A section that is *missing entirely* — no Value section at all — means the
substrate cannot support those facts, and you must not write about them at all. The panel
lists masked families once, at the top, in a warning box.

---

<a name="part-d"></a>

## Part D — The rules

Six rules. Each has an error class attached; the ones marked **Critical** are reported
separately from everything else because they are the ones that would make a filing
indefensible.

### D.1 Assert only what the fact record supports

Every claim must trace to a line in the panel. Not to your knowledge of AML, not to what is
usually true of accounts like this, not to what the graph looks like it implies.

If you find yourself writing a sentence and cannot point at the panel line behind it, delete
the sentence.

- Numbers: exact, as the panel renders them. → wrong number is **H2**.
- Timing: the observed span and window as given. → **H3**.
- Off-graph context, motive, intent: never. → **H8**.

**Copy numbers in the panel's own format.** The panel deliberately renders every value the
way the verification machinery expects to read it back. Writing "roughly 9,400 Canadian
Dollars" where the panel says `9,435 Canadian Dollar` is scored as both a dropped fact and
an unsupported quantity — you will have said something true and been marked down for it. If
you want to round, round *and* give the figure: "9,435 Canadian Dollar, close to ten
thousand".

### D.2 Never name an entity type — **Critical, H4**

**No substrate carries an entity-type column.** AMLworld's schema has a bank code and
nothing else about an account. Elliptic2's node features are anonymised. There is *no case,
on either substrate*, for which "this address is a mixer" is a checkable statement.

Forbidden without exception: **mixer, tumbler, exchange account, cryptocurrency exchange,
darknet market, shell company, front company, casino account, money service business,
hawala** — and any other business-type term.

This is rated Critical because it is the most plausible-looking wrong sentence available. It
reads like expertise. It is an attribution with no evidence of any kind behind it, and in a
document that may be read by a regulator it misrepresents what the institution knows.

**Describe what the account did, never what it is.**

| ✗ | ✓ |
|---|---|
| Funds were sent to a mixer. | Funds were dispersed to 9 accounts within 4 hours. |
| The counterparty is a shell company. | The counterparty has no inbound activity within the reviewed window. |
| The account belongs to a money service business. | The account received funds from 40 distinct counterparties. |

You **may** say that an account acts as a hub, an intermediary, a pass-through, an
originator, a beneficiary or an isolated account — these are the six roles in the controlled
vocabulary, and each is defined by degree, which the record measures.

### D.3 Use only allowed hedging phrases — **Critical, H7**

The complete allowed list (Appendix A.1):

> appears consistent with · is indicative of · may reflect · warrants further review · is
> consistent with · suggests · may indicate · raises the possibility of · merits further
> enquiry · could not be verified

Anything asserting guilt, proof or certainty is forbidden: *is money laundering, was money
laundering, the criminal, the launderer, the fraudster, proves, proven, confirms, confirmed
that, is guilty of, guilty, clearly laundering, definitely laundering, beyond doubt,
conclusively.*

Also forbidden, for their own reasons:

- **Completeness** (H8): *the entire scheme, the complete laundering operation, the full
  extent of, all transactions in the scheme, the whole network.* You have 65% of a stream on
  average. You cannot describe the whole of anything.
- **Motive** (H8): *in order to evade, with the intention of, deliberately structured,
  knowingly, attempting to conceal, in an effort to disguise.* Intent is off-graph. Nothing
  in a subgraph evidences why anyone did anything.

**Inferred typologies must be hedged.** Where the panel says a typology was *inferred* rather
than read from ground truth — every Elliptic2 case — the typology must appear inside one of:
*appears consistent with, is indicative of, may reflect, structurally resembles*. An unhedged
inferred typology is **H5**.

### D.4 Mention every salient field for the typology

Each typology has a list of fact fields an adequate narrative must mention (Appendix A.2, and
`salience.md` in full). **The interface marks them in the panel** — a salient row carries a
marker — and shows you which are still missing as you write.

Two things excuse a field, and the interface applies both for you: a field the substrate
cannot support, and a field whose value is a measured null (you cannot be required to state a
fan width for a fan that is not there). What remains is required.

Mentioning fields *outside* the list is not penalised, provided the claims are supported.

### D.5 Never sum across currencies

HI-Small has fifteen currencies, over seventy thousand cross-currency transactions, and no
exchange rates. **A total across currencies is undefined**, and the record refuses to encode
one — where a case is multi-currency you will see per-currency lines and no aggregate.

Report each currency separately. Never add them, never convert, never write "approximately
60,000 in total" across two currencies.

> ✓ The subject sent 30,060 Euro across 4 transfers and 29,600 US Dollar across 4 transfers.
>
> ✗ The subject sent approximately 60,000 across 8 transfers.

### D.6 On Elliptic2: no amounts, no currencies, no institutions, no clock

Elliptic2 is real Bitcoin data with anonymised features. It carries **no monetary amounts, no
currencies, no wall-clock timestamps, no institution identity and no per-account labels.** An
Elliptic2 panel has no Value section, no Timing section and no Counterparty labels section at
all — those facts do not exist for that substrate.

Write about **topology only**: how many accounts, how many transactions, the degrees, the
density, which structural detectors fired and which did not, the shape. And hedge the
typology, because on Elliptic2 every typology is inferred.

The interface flags an amount-shaped or clock-shaped token in an Elliptic2 draft as a
critical error. Worked Example 4 shows what a good topology-only narrative reads like.

### D.7 Regulatory references

You may cite a reporting threshold only from the whitelist, and only **in exactly the
whitelisted words**:

- *the USD 10,000 reporting threshold* · *the currency transaction reporting threshold*
- *the USD 5,000 SAR filing threshold*
- *the EUR 10,000 declaration threshold*

Cite one only where the case's currency matches, and only as **context** — never as a finding
about a real filing obligation, because AMLworld is synthetic and carries no jurisdiction.
Any other citation, or the same threshold in different words, is **H6, Critical**. The panel
shows you the exact phrase in the threshold row.

---

<a name="part-e"></a>

## Part E — Five worked examples

Each shows the fact record as the interface presents it, the graph in outline, the narrative,
and commentary on the choices. All five are real cases from the Gold sample.

---

### Example 1 — A clean fan-out (and a trap)

**Case** `amlworld_hi_small-fcee452677f08f77` · typology `fan_out` (ground truth) · 9
accounts

**Fact panel (abridged)**

```
SUBJECT
 * Focal account: 004403|80F962B10
   Role in this case: the originating account
   Counterparties sending in: 0        Counterparties receiving out: 8
   Inbound transactions: 0             Outbound transactions: 8

SCOPE
 * Accounts in scope: 9               * Transactions: 12
   Density: 0.111                       Largest out-degree: 8
   Self-transactions: 4   (counted in transactions, excluded from degrees and motifs)

TIMING
   Window start: 2022-09-13 08:22       Window end: 2022-09-14 11:16
 * Observed span: 26.9 hours
   Burst span: 23.7 hours               Transactions in burst: 10
   Event ordering: outflow phase

VALUE
   Sent in Euro: 30,060 Euro over 4 transfers
   Sent in US Dollar: 29,600 US Dollar over 4 transfers
   Currencies involved: Euro, US Dollar  (a cross-currency total is undefined and not shown)
   Institutions involved: 9             Payment formats: ACH

COUNTERPARTY LABELS
 * Counterparties on a flagged transaction: 8
   Counterparties with no flag: 0       Flagged transactions: 8
   Hops to the nearest flagged account: 0    Focal account itself flagged: yes

STRUCTURAL PATTERNS
   fan-out: detected — hub 004403|80F962B10, width 8, window hours 26.9
 * fan-out: width: 8
   fan-in / chain / cycle / bipartite / stack / gather-scatter / scatter-gather: not detected
```

**Graph:** one central node with eight outbound edges, no inbound edges, no structure among
the recipients. A textbook star.

**Narrative**

> **[1] Subject & Scope**
> Account 004403|80F962B10 is the subject of this report and acts as the originating account
> within the reviewed activity. The reviewed subgraph comprises 9 accounts connected by 12
> transactions across 9 institutions.
>
> **[2] Activity Observed**
> Over an observed period of 26.9 hours, from 2022-09-13 08:22 to 2022-09-14 11:16, the
> subject account sent funds to 8 distinct counterparties across 8 outbound transactions and
> received nothing from any account within the reviewed window. Outbound value comprised
> 30,060 Euro across 4 transfers and 29,600 US Dollar across 4 transfers; no cross-currency
> total is defined. Ten of the twelve transactions fall within a 23.7-hour burst beginning
> 2022-09-13 10:42. All settlement used ACH. Four transactions are self-transactions.
>
> **[3] Pattern & Typology**
> The subject account dispersed funds to 8 distinct recipients, a fan-out of width 8, with
> no inbound activity in the reviewed window and no structure among the recipients. No
> fan-in, chain, cycle, bipartite, stack, gather-scatter or scatter-gather structure was
> detected at the configured thresholds. The case carries the ground-truth typology label
> fan_out, and the observed dispersal is consistent with that pattern.
>
> **[4] Basis & Action**
> All 8 counterparties appear on at least one transaction flagged in the source data, 8
> transactions in the reviewed subgraph carry that flag, and the subject account is itself
> flagged. No transfers fell close to the USD 10,000 reporting threshold, and settlement
> used a single ordinary payment rail. The concentration of the dispersal within roughly a
> day, combined with the absence of any offsetting inbound activity, warrants further review
> by an investigator with access to customer records not represented in this subgraph.

**Commentary**

- **Section 2 states no total across currencies**, because none is defined (rule D.5). Saying
  "no cross-currency total is defined" is better than silence: it tells the reader the
  omission is principled rather than an oversight.
- **The width, 8, is stated explicitly.** It is salient for `fan_out`, and "dispersed funds
  to several accounts" would fail adequacy. Section 3 states it again in the structural
  framing; repetition across sections is fine.
- **`flow.total_outflow` is salient for `fan_out` but is excused here**, because it is
  withheld under the multi-currency rule. The interface marks it excused; the narrative is
  not penalised for its absence.
- **Section 3 lists what did *not* fire.** Cheap, and it forecloses a reader assuming a
  structure that is not there.
- **Section 4 includes the exculpatory material** — no near-threshold transfers, one ordinary
  rail — even though the case is strongly incriminating. That is H9 avoidance and it is not
  optional.
- **The trap this case avoids:** the focal account *is* the hub here (the detector line says
  `hub 004403|80F962B10`, the same identifier). Always check. Where the hub named in the
  detector line is a *different* account, the focal account is a recipient of the fan-out and
  writing that it "dispersed funds" would be false.

---

### Example 2 — A gather-scatter, and why the ordering is the finding

**Case** `amlworld_hi_small-0694950debcc58d6` · typology `gather_scatter` (ground truth) · 3
accounts

**Fact panel (abridged)**

```
SUBJECT
 * Focal account: 0048309|811C599A0
   Role in this case: an intermediary
   Counterparties sending in: 2        Counterparties receiving out: 2

SCOPE
 * Accounts in scope: 3               * Transactions: 5
   Density: 0.667                       Reciprocity: 1.000
   Self-transactions: 1

TIMING
   Observed span: 43.1 hours            Burst: none detected
 * Event ordering: interleaved

VALUE
 * Total received by focal: 116,500 Saudi Riyal
 * Total sent by focal: 81,560 Saudi Riyal
   Retained: 34,910 Saudi Riyal         Largest single transfer: 60,780 Saudi Riyal

COUNTERPARTY LABELS
   Counterparties on a flagged transaction: 2   Counterparties with no flag: 0
   Flagged transactions: 5              Focal account itself flagged: yes

STRUCTURAL PATTERNS
   gather-scatter: detected — gather width 2, scatter width 2
```

**Graph:** three nodes, dense, with edges running both ways between the focal account and
each counterparty.

**Narrative**

> **[1] Subject & Scope**
> Account 0048309|811C599A0 is the subject of this report and acts as an intermediary within
> the reviewed activity. The reviewed subgraph comprises 3 accounts connected by 5
> transactions across 3 institutions.
>
> **[2] Activity Observed**
> Over an observed period of 43.1 hours the subject account received 116,500 Saudi Riyal
> across 2 inbound transfers from 2 counterparties and sent 81,560 Saudi Riyal across 2
> outbound transfers to the same 2 counterparties, retaining 34,910 Saudi Riyal. The largest
> single movement was 60,780 Saudi Riyal. Inbound and outbound activity was interleaved
> rather than falling into a collection phase followed by a dispersal phase. No burst was
> detected. All settlement used ACH, and one transaction is a self-transaction.
>
> **[3] Pattern & Typology**
> A gather-scatter structure was detected, with a gather width of 2 and a scatter width of 2
> across a subgraph whose density is 0.667 and whose reciprocity is 1.000 — every pair of
> accounts transacts in both directions. No fan-in, fan-out, cycle, stack or scatter-gather
> structure was detected. The case carries the ground-truth typology label gather_scatter.
> The interleaved ordering means the observed activity does not show a completed collection
> phase preceding dispersal, and the structure is therefore weaker evidence of layering than
> the typology label alone would suggest.
>
> **[4] Basis & Action**
> Both counterparties appear on flagged transactions and all 5 transactions in the reviewed
> subgraph carry that flag; the subject account is itself flagged. Against this, the subgraph
> is small, fully reciprocal, and the ordering is interleaved — a profile also consistent with
> ordinary two-way activity between three parties. On the balance of what this subgraph
> shows, the activity merits further enquiry, with the flagged status of every transaction as
> the primary basis rather than the structure itself.

**Commentary**

- **The ordering is stated, in plain words, twice.** `temporal.event_ordering` is on the
  salience list for `gather_scatter` for exactly this reason. `interleaved` is the honest and
  awkward finding, and it is what a narrative would most like to omit.
- **Both widths appear.** A gather-scatter narrative quoting only the gather side has
  described a fan-in.
- **Section 3 argues *against* the label's implication.** This is correct and is what a
  competent analyst does. The ground-truth label says the case belongs to a gather_scatter
  stream; the case as observed does not show the phase separation the typology is about, and
  saying so is more useful than repeating the label.
- **Reciprocity 1.000 is used as evidence**, not just quoted. Every pair transacting both
  ways is ordinary business behaviour, and naming it is what makes Section 4 balanced.
- **Section 4 names its own primary basis** — the flags, not the shape. An investigator
  reading a hundred of these needs to know which part of the case is load-bearing.

---

### Example 3 — A hard negative: legitimate activity that looks structured

**Case** `amlworld_hi_small-006e89f5bc1f04b6` · typology `unclassified` · hard negative · 4
accounts

**Fact panel (abridged)**

```
SUBJECT
 * Focal account: 019|8090C7E20
 * Role in this case: the originating account
   Counterparties sending in: 0        Counterparties receiving out: 2

SCOPE
 * Accounts in scope: 4               * Transactions: 4
   Density: 0.250

TIMING
 * Observed span: 15.7 hours           Burst: none detected
   Event ordering: outflow phase

VALUE
 * Total sent by focal: 9,435 Canadian Dollar
   Largest single transfer: 7,021 Canadian Dollar
   Transfers near the USD 10,000 reporting threshold: 0
   Institutions involved: 3            Payment formats: ACH, Cheque

COUNTERPARTY LABELS
 * Counterparties on a flagged transaction: 0
   Counterparties with no flag: 2      Flagged transactions: 0
   Nearest flagged account: none reachable in this case
   Focal account itself flagged: no

STRUCTURAL PATTERNS
   bipartite: detected — left size 2, right size 2, score 1.0
   everything else: not detected
```

**Graph:** four nodes in two clean pairs. It looks like a structured layer. It is not one.

**Narrative**

> **[1] Subject & Scope**
> Account 019|8090C7E20 is the subject of this report and acts as the originating account
> within the reviewed activity. The reviewed subgraph comprises 4 accounts connected by 4
> transactions across 3 institutions.
>
> **[2] Activity Observed**
> Over an observed period of 15.7 hours the subject account paid out 9,435 Canadian Dollar
> across 2 outbound transactions to 2 distinct counterparties and received nothing within the
> reviewed window. The largest single movement was 7,021 Canadian Dollar. Settlement used ACH
> and Cheque. No transfers fell close to the USD 10,000 reporting threshold, and no burst was
> detected.
>
> **[3] Pattern & Typology**
> The subgraph is bipartite, with sides of 2 and 2 accounts and a density of 0.250. No
> fan-in, fan-out, cycle, stack, gather-scatter or scatter-gather structure was detected at
> the configured thresholds. Bipartiteness at this scale is a property of almost any
> four-account subgraph with no within-side transactions and carries no independent weight.
> The case carries the typology label unclassified, and no structure in the reviewed window
> supports assigning a more specific one.
>
> **[4] Basis & Action**
> No transaction in the reviewed subgraph is flagged in the source data, neither counterparty
> appears on a flagged transaction, the subject account is not itself flagged, and no flagged
> account is reachable within the case. The activity comprises two outbound payments over
> fifteen hours on two ordinary payment rails, none close to a reporting threshold. On the
> evidence available in this subgraph the activity appears consistent with ordinary outbound
> settlement, and no basis for escalation is identified.

**Commentary**

- **Section 4 says there is no basis.** That is the correct output, and writing it takes
  more nerve than writing a suspicion narrative. A quarter of your items are like this.
- **The bipartite detection is named and then discounted, with a reason.** Ignoring it would
  be dishonest; reporting it without the discount would be misleading. Explaining *why* a
  four-node bipartite structure is unremarkable is the analytic content of this narrative.
- **The exculpatory facts are stated in full and together**: no flags anywhere, no flagged
  account reachable, ordinary rails, nothing near a threshold. Individually each is thin;
  together they are the finding.
- **No hedged suspicion is manufactured.** "Appears consistent with ordinary outbound
  settlement" is a hedge pointing at the innocent reading, which is what the record supports.
  Writing "warrants further review" here to be safe would be *worse* than useless — it is the
  false positive the hard negatives exist to detect.
- **The threshold is cited in the whitelisted words**, taken from the panel row.

---

### Example 4 — Elliptic2: topology only

**Case** an Elliptic2 subgraph · typology inferred · 3 accounts

**Fact panel — note what is absent**

```
Not available on this substrate:
  - timing — this substrate has no wall-clock timestamps
  - value — this substrate has no monetary amounts or currencies
  - counterparty labels — this substrate labels whole subgraphs, not accounts

SUBJECT
 * Focal account: 001|80000001
 * Role in this case: the originating account
   Counterparties sending in: 0        Counterparties receiving out: 2

SCOPE
 * Accounts in scope: 3               * Transactions: 3
   Density: 0.500                       Reciprocity: 0.000
   Largest in-degree: 2                 Largest out-degree: 2

STRUCTURAL PATTERNS
   everything: not detected
```

There is no Timing section, no Value section and no Counterparty labels section. Not empty —
absent.

**Narrative**

> **[1] Subject & Scope**
> Cluster 001|80000001 is the subject of this report and acts as the originating node within
> the reviewed activity. The reviewed subgraph comprises 3 nodes connected by 3 transactions.
> This substrate carries no monetary amounts, no wall-clock timing and no per-node labels, so
> this report describes topology only.
>
> **[2] Activity Observed**
> The subject node sends to 2 distinct counterparties across 2 outbound transactions and
> receives from none within the reviewed subgraph. A third transaction runs between the two
> counterparties. The largest in-degree in the subgraph is 2 and the largest out-degree is 2.
>
> **[3] Pattern & Typology**
> The subgraph has a density of 0.500 and a reciprocity of 0.000: every transaction runs in
> one direction only. No fan-in, fan-out, chain, cycle, bipartite, stack, gather-scatter or
> scatter-gather structure was detected at the configured thresholds. This substrate carries
> no typology ground truth, and the observed topology does not resemble any of the eight
> typologies closely enough to support even a hedged assignment; the case is unclassified.
>
> **[4] Basis & Action**
> This substrate provides no per-node labels, so no statement can be made about whether any
> node in this subgraph has been associated with illicit activity. The reviewed structure is
> a three-node directed subgraph with no detected pattern. On the topology alone no basis for
> escalation is identified, and any assessment would require attributes this substrate does
> not carry.

**Commentary**

- **Not one number that is not a count or a ratio.** No amounts, no times, no currencies.
- **The masking is stated explicitly in Section 1**, and again in Section 4. This is not
  padding: a reader who does not know the substrate would otherwise read the absence of a
  value discussion as an omission.
- **"Cluster" and "node" rather than "account".** Elliptic2 nodes are clusters of Bitcoin
  addresses, not bank accounts. Either wording is accepted, but the substrate-appropriate one
  is better.
- **Section 4 declines to conclude, and says why.** "No per-node labels exist, so no statement
  can be made" is a substantive sentence — it distinguishes *we checked and found nothing*
  from *we cannot check*, and the two are different facts about the case.
- **No typology is asserted, hedged or otherwise.** Where an Elliptic2 case *does* resemble a
  typology, the assignment must sit inside one of the four required hedges (rule D.3). Here
  nothing fires, so the honest answer is `unclassified`.

---

### Example 5 — Ambiguous: several detectors fire and none of them settles it

**Case** `amlworld_hi_small-a151c3239e11a9f9` · typology `unclassified` · flagged activity
present · 6 accounts

**Fact panel (abridged)**

```
SUBJECT
 * Focal account: 001669|80126CD40
 * Role in this case: an intermediary
   Counterparties sending in: 3        Counterparties receiving out: 1
   Inbound transactions: 3             Outbound transactions: 2

SCOPE
 * Accounts in scope: 6               * Transactions: 7
   Density: 0.167

TIMING
 * Observed span: 11.9 hours           Burst span: 11.9 hours (7 transactions)
   Event ordering: outflow phase then inflow phase

VALUE
 * Total received by focal: 3,568 Euro
 * Total sent by focal: 302.52 Euro
   Retained: 3,266 Euro                Largest single transfer: 2,150 Euro

COUNTERPARTY LABELS
 * Counterparties on a flagged transaction: 1
   Counterparties with no flag: 3      Counterparties in total: 4
   Flagged transactions: 1
   Share of inbound value from flagged counterparties: 1%
   Focal account itself flagged: yes

STRUCTURAL PATTERNS
   fan-in: detected — hub 001669|80126CD40, width 3, window hours 0.4
   chain: detected — max length 3
   bipartite: detected — left size 4, right size 2, score 0.5
   fan-out / cycle / stack / gather-scatter / scatter-gather: not detected
```

**Narrative**

> **[1] Subject & Scope**
> Account 001669|80126CD40 is the subject of this report and acts as an intermediary within
> the reviewed activity. The reviewed subgraph comprises 6 accounts connected by 7
> transactions across 6 institutions.
>
> **[2] Activity Observed**
> Over an observed period of 11.9 hours the subject account received 3,568 Euro across 3
> inbound transactions from 3 distinct counterparties and sent 302.52 Euro across 2 outbound
> transactions to 1 counterparty, retaining 3,266 Euro. The largest single movement was 2,150
> Euro. The three inbound transfers fall within a 0.4-hour window. Settlement used ACH, Cheque
> and Credit Card. The observed phase sequence is an outflow phase followed by an inflow
> phase.
>
> **[3] Pattern & Typology**
> Three structures were detected: a fan-in of width 3 into the subject account within 0.4
> hours, a chain of maximum length 3, and a bipartite split of 4 and 2 accounts with a score
> of 0.5. No fan-out, cycle, stack, gather-scatter or scatter-gather structure was detected.
> The case carries no typology label and is unclassified. The detected structures do not
> combine into a single pattern: the fan-in is not followed by a corresponding dispersal, the
> chain does not run through the subject account's inbound counterparties, and a bipartite
> score of 0.5 over six accounts is weak. On the evidence in this subgraph no typology can be
> assigned.
>
> **[4] Basis & Action**
> One of the 4 counterparties appears on a flagged transaction and 1 of the 7 transactions in
> the subgraph carries that flag; the subject account is itself flagged. The remaining 3
> counterparties carry no flag, and flagged counterparties account for 1% of inbound value.
> Three payment rails are in use. The rapid arrival of three inbound transfers within 0.4
> hours is the only feature of the reviewed activity that is not routine, and against a 1%
> flagged share of inbound value it is thin. The case merits further enquiry at low priority,
> and the basis is the compression of the inbound transfers rather than the counterparty
> profile.

**Commentary**

- **Ambiguity is reported as ambiguity.** Section 3 says the structures do not combine, and
  says why for each one. Picking whichever detector fired most impressively and writing a
  confident story around it is the failure this example exists to prevent.
- **The 1% figure is the most important number in the case** and it is exculpatory. Three
  detectors fired and one counterparty is flagged, which reads alarming; 1% of inbound value
  is what actually calibrates it. Omitting it would be **H9**.
- **Section 4 states its priority and its basis.** "Low priority, on the compression of the
  inbound transfers" is something an investigator can triage on.
- **`retained` is stated.** 3,568 in and 302.52 out means most of the money stayed. That is
  informative and is not on any salience list — mentioning fields outside the list is fine
  when the claims are supported.
- **No qualitative intensifier is used loosely.** The narrative says "within 0.4 hours"
  rather than "rapid dispersal": in the controlled vocabulary "rapid dispersal" is a *claim*
  that `temporal.burst_window_hours <= 6`, and using it where the binding does not hold is
  H2. Where you are unsure whether an intensifier applies, give the number instead.

---

<a name="part-f"></a>

## Part F — Common errors, wrong and right

One pair per class in the taxonomy. Read this before your calibration items and again after
your feedback.

### H1 — Entity fabrication (High)

Naming an account, counterparty or institution not in the case subgraph.

> ✗ Funds moved from 019|8090C7E20 through 019|8090C7E21 to three further accounts.
>
> ✓ Funds moved from 019|8090C7E20 to 2 further accounts.

*Usually a transcription slip. Account identifiers are long and differ by one character. The
interface checks every identifier you type against the case's inventory and flags one that is
not there — take that flag seriously; it is exactly decidable.*

### H2 — Numeric error (High)

A count, amount, degree, width or share that disagrees with the record.

> ✗ The account dispersed funds to approximately nine counterparties.
>
> ✓ The account dispersed funds to 8 counterparties.

*Counts are exact — there is no tolerance and no rounding. Amounts have a 1% tolerance.
"Approximately nine" against eight is wrong, and "approximately" does not rescue it.*

A second form, easy to miss:

> ✗ The rapid dispersal of funds across 8 accounts…  (burst window: 23.7 hours)
>
> ✓ The dispersal of funds across 8 accounts over 23.7 hours…

*"Rapid dispersal" is a controlled phrase that claims `burst_window_hours <= 6`. At 23.7
hours the claim is false and the verdict is CONTRADICTED, not "a bit florid".*

### H3 — Temporal error (Medium)

Wrong ordering, an invented duration, or a timestamp outside the case window.

> ✗ Funds were received and then immediately dispersed the same afternoon.
>   (event ordering: interleaved)
>
> ✓ Inbound and outbound activity was interleaved across the 43.1-hour window.

*Note the duration rule is asymmetric. "About three days" against 76 hours is SUPPORTED — you
claimed a precision of one day and are right at that precision. "76 hours" against 80 is
CONTRADICTED. Be as precise as you can be correctly, not as precise as possible.*

### H4 — Attribution fabrication (**Critical**)

Assigning a business type or real-world identity.

> ✗ The subject account transferred funds to what appears to be a cryptocurrency exchange.
>
> ✓ The subject account transferred funds to an account with 40 inbound counterparties and no
>   outbound activity within the reviewed window.

*"Appears to be" does not save it. No substrate carries an entity-type column, so the claim is
not weakly supported — it is unsupported by anything at all. Describe the degree profile; let
the investigator, who has the customer records, draw the conclusion.*

### H5 — Typology error (Medium)

Naming a typology the record does not carry, or asserting an inferred one without a hedge.

> ✗ This is a scatter-gather structure.   (panel: gather-scatter detected)
>
> ✓ A gather-scatter structure was detected, with a gather width of 2 and a scatter width of 2.

*The two are mirror images and are the most-confused pair in calibration. Gather-scatter:
many in, through one hub, then many out. Scatter-gather: one out, through many, then back
together at one point.*

And on Elliptic2:

> ✗ The subgraph is a fan-out.   (typology inferred)
>
> ✓ The subgraph structurally resembles a fan-out.

### H6 — Regulatory fabrication (**Critical**)

Citing a threshold, rule or obligation outside the whitelist.

> ✗ Transfers fell below the reporting threshold under 31 CFR 1010.313.
>
> ✓ No transfers fell close to the USD 10,000 reporting threshold.

*Also wrong for a subtler reason — the right threshold in the wrong words:*

> ✗ …close to the 10,000 US Dollar reporting threshold.
>
> ✓ …close to the USD 10,000 reporting threshold.

*The whitelist is a list of exact phrases. The panel's threshold row shows you the permitted
one; copy it.*

### H7 — Guilt overclaim (**Critical**)

Asserting guilt, criminality or proof rather than suspicion.

> ✗ The subject deliberately structured these transfers to evade reporting requirements.
>
> ✓ Nine transfers fell below the USD 10,000 reporting threshold, a pattern consistent with
>   structuring, which warrants further review.

*Two errors in one sentence, which is typical: "deliberately" is motive (H8) and "to evade" is
both motive and an assertion of criminal purpose (H7). The corrected version states the
observable fact, names the pattern with an allowed hedge, and stops.*

### H8 — Unsupported inference (Medium)

Motive, intent, off-graph context, or the completeness of a scheme.

> ✗ These accounts were opened for the purpose of receiving the dispersed funds and form the
>   full extent of the network.
>
> ✓ The 8 recipient accounts have no inbound activity from any other account within the
>   reviewed window. The reviewed subgraph is a bounded extract and may not contain the whole
>   of the related activity.

*Two separate failures. Account-opening purpose is off-graph — the subgraph contains
transactions, not account applications. And "the full extent" claims completeness that a case
holding 65% of its stream on average cannot support.*

### H9 — Omission of exculpatory fact (Medium)

Leaving out a fact in the record that materially weakens the suspicion.

> ✗ **[4] Basis & Action**
>   One counterparty appears on a flagged transaction and the subject account is itself
>   flagged. Three structural patterns were detected. The case warrants further review.
>
> ✓ **[4] Basis & Action**
>   One of the 4 counterparties appears on a flagged transaction and 1 of the 7 transactions
>   carries that flag; the subject account is itself flagged. The remaining 3 counterparties
>   carry no flag, and flagged counterparties account for 1% of inbound value. The case merits
>   further enquiry at low priority.

*Nothing in the ✗ version is false. Every sentence would pass a claim-by-claim check. It is
still a bad narrative, because the reader would escalate on it and the record does not support
that. **H9 is the only class you cannot catch by re-reading what you wrote — you have to
re-read the panel.** Before you submit, look specifically for the facts that cut the other
way, and put them in.*

*It is also the class where annotators disagree most, so it is watched most closely in the
agreement analysis. If you are unsure whether a fact is materially exculpatory, include it.*

---

<a name="appendix"></a>

## Appendix

### A.1 Allowed hedging phrases

> appears consistent with · is indicative of · may reflect · warrants further review · is
> consistent with · suggests · may indicate · raises the possibility of · merits further
> enquiry · could not be verified

Required for an **inferred** typology (one of these must wrap it):

> appears consistent with · is indicative of · may reflect · structurally resembles

### A.2 Salience lists — what each typology must mention

Every list is its own entries **plus** the common three. Full detail, and the reasoning, in
[`salience.md`](salience.md). The interface marks these rows in the panel and tracks which
you have covered.

**Common to every typology:** `focal_entity.id` · `structure.n_nodes` · `structure.n_edges`

| Typology | Additional required fields |
|---|---|
| `fan_out` | fan-out width · observed span · total outflow · flagged counterparty count |
| `fan_in` | fan-in width · observed span · total inflow · flagged counterparty count |
| `gather_scatter` | gather width · scatter width · total inflow · total outflow · **event ordering** |
| `scatter_gather` | scatter-gather width · total outflow · observed span · focal role |
| `cycle` | cycle length · observed span · total outflow |
| `bipartite` | left size · right size · density · total inflow |
| `stack` | stack depth · observed span · total outflow · focal role |
| `random` | observed span · total inflow · total outflow · flagged counterparty count |
| `unclassified` | observed span · total inflow · total outflow · flagged counterparty count · focal role |

A field the substrate cannot support is **excused**. A field whose value is a measured null is
**excused**. The interface applies both for you.

### A.3 The six entity roles

The only role words you may use. Each is defined by degree, which the record measures.

| Role | Definition | Write it as |
|---|---|---|
| originator | sends inside the case, receives nothing inside it | the originating account · the sending account |
| beneficiary | receives inside the case, sends nothing inside it | the receiving account · the destination account |
| pass_through | exactly one counterparty in and one out | a pass-through account · a conduit account |
| hub | at least 5 distinct counterparties on **both** sides | a hub account · a central account |
| intermediary | sends and receives, but neither pass-through nor hub | an intermediary · an intermediate account |
| terminal | no counterparties inside the case at all | an isolated account |

There is deliberately no *mixer*, *exchange*, *merchant* or *gambling service*, and there
never will be: no substrate carries an entity-type column.

### A.4 Pre-submission checklist

Before you press Submit:

1. All four sections present, in order, with headings in title case.
2. Between 80 and 400 tokens.
3. Every salient row for this typology mentioned — check the panel markers.
4. Every number copied in the panel's own format.
5. No entity type named anywhere.
6. Every conclusion inside an allowed hedge; no number inside one.
7. **Section 4 contains the facts that cut against escalation.**
8. Nothing about intent, motive or the completeness of a scheme.
9. On Elliptic2: no amount, no currency, no clock time, no institution.
10. Run **Check against the record** and read what it says.

### A.5 Where to raise a problem

If the panel and the graph disagree, if a rule seems wrong for a case in front of you, or if
you cannot write a faithful narrative within the length bounds, **do not work around it** —
record it in the comment box and raise it. A rule that fires on correct writing is a rule
that needs changing, and the override rates are analysed for exactly that. Working around a
problem silently is the one response that loses the information.

---

*Graph2Text AML annotation guidelines v1.0.0. Released with the corpus under the repository's
licence. The machine-readable sources of truth are `schemas/vocab_v1.yaml` (controlled
vocabulary, salience lists, hedging) and `src/g2t_aml/facts/taxonomy.py` (the nine classes);
where this document and those disagree, they are correct and this document is a bug.*
