# Dataset cards

One card per **corpus tier** produced by this project. For the two upstream *substrates*
we consume but did not create, see [`../data_cards/`](../data_cards/) instead — that is a
different question (provenance and licence of someone else's data) from this one
(construction, splits and limitations of ours).

Written Phase 14, 2026-08-06, for release `v0.1.0`.

---

## The tiers

| Tier | Card | Records | Status |
|---|---|---:|---|
| **Bronze** — deterministic template rendering | [`bronze.md`](bronze.md) | **15,707** | **released** |
| **Silver** — verified LLM rewrites | [`silver.md`](silver.md) | **0** | **not built** — no API credentials |
| **Gold** — human-authored | [`gold.md`](gold.md) | **0** | **not written** — no annotator recruited |
| `case_facts` records + frozen splits | [`case_facts.md`](case_facts.md) | 16,156 cases | **released** |

**Two of the three narrative tiers are empty.** That is the project's state, not an
omission from this directory. The cards for the empty tiers document what the machinery
produces, what has been validated against scripted and simulated inputs, and exactly what
is missing — because a reader deciding whether to build them needs that, and a reviewer
checking whether the paper's corpus claims are honest needs it more.

---

## Why the corpus exists at all

**There is no corpus of (graph, SAR narrative) pairs anywhere in the world.** Real
Suspicious Activity Reports are confidential by statute. Every narrative in this project is
constructed, and the three tiers are three different constructions with the same contract:

- Bronze is **faithful by construction** — rendered from the fact record, every formatter
  shipping with its inverse. It is the floor everything else must beat.
- Silver is **faithful by verification** — LLM rewrites gated by the Phase 3 checker, at
  most two targeted repairs, then discard-and-log (D-046).
- Gold is **faithful by authorship** — written by a person under `docs/annotation/`,
  small, held out, never trained on.

`training_record_v1.json` is **frozen at 1.0.0 and carries all three tiers.** Silver and
Gold differ from Bronze only in `tier` and `generator`, and the same ten-point validation
harness gates all three identically (D-037).

---

## Licensing — read this before redistributing anything here

**The corpus and the code are under different licences and the distinction is
load-bearing.**

Every released narrative quotes AMLworld account identifiers, timestamps, currencies and
transaction amounts directly (`"...account 021611|800F41B10 ... between 2022-09-07 13:30
and 2022-09-08 08:52 ... around 6,273 US Dollar"`). That embeds more than a de-minimis
portion of the source Data, so the corpus is **Enhanced Data** under CDLA-Sharing-1.0, not
a §3.5 *Result*.

| Bundle | Licence |
|---|---|
| Narrative corpus, fact records, case files, split manifests | **CDLA-Sharing-1.0** (share-alike, attribution, changes marked) |
| Code, model weights, metrics, `RESULTS.md`, figures | **Apache-2.0** (Results, §3.5 exempt) |

The two are released as **separate bundles and must stay separate**. See `README.md`
§ Licensing and D-098.

---

## Common limitations

These apply to every tier and are not repeated in full on each card.

- **One substrate, and it is synthetic.** AMLworld HI-Small only. Elliptic2 has never been
  ingested, so no tier has a second-substrate half.
- **17.7 days of source data.** The temporal split has very little room.
- **A case is a fragment.** The 48-hour extraction window keeps ~65% of a laundering
  stream's transactions on average (D-019). `typology` means "part of a stream of this
  typology", not "exhibits it in full", and every narrative describes the fragment.
- **No demographic attribute exists** in the source, so no tier supports fairness analysis
  on protected attributes. `docs/ETHICS.md` §4.
- **Never sum across currencies.** HI-Small has 15 currencies, 72,170 cross-currency
  transactions and no exchange rates. Cross-currency aggregates are withheld as typed
  sentinels and the per-currency breakdown is always emitted (D-033). A consumer that
  reconstructs a total by adding the breakdown is producing a number the corpus refused to
  produce.
- **Absence is typed.** A fact family the substrate cannot support is an `Unavailable`
  sentinel carrying a reason — never `0`, never a bare `None`. A *measured* null (no cycle
  exists) is a bare `None`, and the two mean different things to the checker (D-025).
