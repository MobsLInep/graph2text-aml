# Ethics statement

**Graph2Text AML** · written Phase 14, 2026-08-06 · covers release `v0.1.0`

This document states what the system is for, what it must not be used for, how it is known
to fail, what we could and could not assess, where the data came from, and what the
compute cost.

It is written against **the state of the artifact as released**, which is not the state the
project's design anticipated. The single most important sentence in this document is in §3:

> **No generator has been trained, so this release contains no system whose hallucination
> rate has been measured.** Every faithfulness number below describes the deterministic
> template corpus and the evaluation instrument, not a learned model.

Anyone reading this as due diligence on a deployable component should stop at §3.

---

## 1. Intended use

**Investigator draft-assist, with human-in-the-loop review, inside a regulated financial
institution's existing SAR workflow.**

The intended operator is a trained financial-crime investigator who has already decided a
case warrants review. The system's output is a *first draft* of the narrative section of a
Suspicious Activity Report: a prose account of what the transaction subgraph shows,
assembled from facts the investigator can check against the same case data.

The design commitment behind that framing is that **the investigator remains the author**.
The draft exists to remove transcription and structuring work from a task that is
otherwise done from a blank page. It does not exist to decide whether the case is
suspicious — that decision is upstream and is made by a person and, usually, by a separate
detection model.

Three properties of the design follow from this and are not negotiable in a deployment:

- **Every claim in the output is checkable.** The narrative is generated against a
  structured `case_facts` record and verified with the same code, in reverse
  (`src/g2t_aml/facts/`). An investigator reviewing a draft can be shown the record.
- **Absence is typed.** A fact the substrate cannot supply is a sentinel carrying a
  reason, never a zero and never silence (invariant 10). A narrative cannot quietly assert
  something the data does not license.
- **The output is scoped to suspicion, never to guilt.** Asserting criminality is a
  critical error class (H7) with a dedicated detector, because a SAR is legally not
  entitled to make a criminal finding.

### Out of scope by construction

The system was never designed for, tested for, or evaluated on: transaction monitoring,
alert triage, customer risk scoring, sanctions screening, KYC/CDD decisions, account
closure, law-enforcement targeting, or any use where the output reaches a person who is
the subject of the analysis.

---

## 2. Misuse limits

These are prohibitions, not cautions. Each names a use the artifact could technically
support and must not be put to.

**It must not be used to make or support a determination about an individual without human
review.** The output is unverified prose about a named account holder. A pipeline that
routes it to any decision — filing, freezing, closing, reporting, escalating — without a
person reading and taking responsibility for it is outside intended use, regardless of how
the confidence is presented.

**It must not be used to generate volume filings.** Defensive over-filing is a known
pathology of AML compliance, and it has real costs: it degrades the signal reaching
Financial Intelligence Units, and every filed SAR attaches a suspicion record to a real
person. A system that lowers the marginal cost of writing a narrative lowers the marginal
cost of filing one. Deployments must not treat throughput as the success metric, and the
project deliberately measures *time-to-usable-draft* and *would-you-file-this* rather than
narratives-per-hour (Phase 12 design; **not yet run** — see §3).

**It must not be presented to a regulator, an FIU, or in a legal proceeding as
machine-authored without disclosure.** If a filed narrative originated as a model draft,
that provenance belongs in the institution's record. Concealing it misrepresents the
evidentiary basis of the filing.

**It must not be used to justify a determination after the fact.** Generating an
explanation for a decision already made, and presenting it as the reasoning behind that
decision, is rationalisation with a machine in the loop. The system produces a description
of a subgraph; it does not reconstruct anyone's reasoning.

**It must not be repurposed to profile individuals or groups.** Nothing in the fact layer
carries demographic attributes, and nothing should be added to it that does. See §5.

**The corpus must not be treated as a corpus of real SARs.** It is not. Real SARs are
confidential by statute and none were consulted. The Bronze narratives are template output
and the Gold tier does not exist yet. A model trained on this corpus has learned the
register of this corpus, which no regulator has validated.

---

## 3. Known failure modes, with the measured rates

### 3.1 What has and has not been measured

| | |
|---|---|
| Systems in the declared experiment matrix | 17 |
| Systems trained | **0** |
| Systems with a measured faithfulness score | **1** (B1, the deterministic template) |
| Gate 8 — does the fusion layer beat its own shuffled control? | **OPEN. Not answered.** |
| Human evaluation of any output | **None.** Zero people have rated anything. |

The blockers are hardware, credentials and recruitment, documented in `RESULTS.md` §2.1
and D-068. They are not framed here as excuses; they are framed here because **an ethics
statement that reports a template's failure rates as if they were a system's failure rates
would be the most misleading thing in this repository.**

### 3.2 Measured failure rates — the Bronze template corpus (15,707 narratives, 296,196 claims)

| Failure class | Rate | Note |
|---|---:|---|
| H1 Entity fabrication | 0.0000 | |
| H2 Numeric error | 0.0000 | |
| H3 Temporal error | 0.0000 | |
| **H4 Attribution fabrication** (critical) | 0.0000 | |
| H5 Typology error | 0.0000 | |
| **H6 Regulatory fabrication** (critical) | 0.0000 | |
| **H7 Guilt overclaim** (critical) | 0.0000 | |
| H8 Unsupported inference | 0.0000 | |
| **H9 Omission of exculpatory fact** | **0.9179** | **see below** |
| Zero-Hallucination Rate | 1.0000 | per-narrative binary |
| Critical Error Rate | 0.0000 | H4 + H6 + H7 |
| Fact Coverage | 0.8595 | |

**The zeros in H1–H8 are not an achievement and must not be quoted as one.** Bronze renders
deterministically from the fact record and every formatter ships with its inverse, so it is
faithful *by construction*. That table is a regression test on the measurement instrument.
The only reason it appears in an ethics statement is that it establishes the floor against
which a trained system would be read.

**H9 at 0.9179 is a real measured defect, and it is the ethically significant one.**

### 3.3 The exculpatory-omission failure, stated plainly

**92% of the released Bronze narratives omit a fact in the record that materially weakens
the suspicion they describe.**

The templates report how many counterparties a subject has and how many of those appear on
flagged transactions, but never the *licit* count; never `labels.focal_is_illicit`; and
mention `temporal.burst_detected` only when a burst was found. So a case whose subject
carries **no** illicit label produces a narrative that never says so. The reader sees a
one-sided account.

This matters more than a numeric error would. A wrong count is caught on review. A missing
exculpatory fact is invisible on review — the investigator cannot notice the absence of
something they were never shown, and the draft's framing has already anchored them. It is
the failure mode most likely to survive human-in-the-loop review, which is the control this
system's entire safety case rests on.

It was found by the Phase 10 harness, not by inspection, and it is not fixed in this
release. It is documented here, in `RESULTS.md` §4.2, and in the Bronze dataset card.

### 3.4 Failure modes anticipated but not measured

Each of these is a design concern with detection machinery built and no number attached,
because no model has run. They are listed so that a deployer knows what to look for.

- **Fluent unfaithfulness.** The characteristic LLM failure: a well-formed narrative
  asserting a value the record does not carry. The three-valued checker exists to catch it
  and the guard exists to reject it at generation time. **Unmeasured on any model.**
- **Critical assertions (H4/H6/H7).** Calling an address a "mixer", citing a regulation
  that does not exist, or stating that an account holder *is* laundering. These are
  attacked at the vocabulary level — the words are excluded rather than flagged afterwards
  (D-029) — with `check_narrative_text` as defence in depth. **Unmeasured on any model.**
- **Substrate-inappropriate assertion.** Elliptic2 has no amounts, no currencies, no real
  timestamps and no entity types. A model trained on AMLworld and run on Elliptic2 would
  be strongly disposed to assert all four. Invariant 4 and the availability mask exist for
  exactly this. **Untested: Elliptic2 has never been ingested.**
- **A case is not its whole scheme.** The 48-hour extraction window keeps ~65% of a
  laundering stream's transactions on average (D-019). A narrative describing a case is
  describing a fragment, and "consistent with layering" is a statement about the fragment.
  A generator that writes as though the case were complete is overclaiming, and no metric
  currently isolates this.
- **Automation bias.** A plausible draft makes a reviewer likelier to accept its framing.
  This is a property of the deployment, not the model, and it is the thing Phase 12's
  decision-setting study was designed to detect. **The study has not run.**

### 3.5 Failure modes of the measurement, not the model

Stated because a metric that flatters is an ethical problem in its own right.

- **Hallucination is currently *understated* by the cue-attribution design.** A wrong
  quantity is scored CONTRADICTED only if a cue rule attributes it to a field; a narrative
  phrased outside every cue has its quantities scored UNVERIFIABLE instead of checked
  (D-076). That is the conservative direction — it shows up in the unverifiable rate — but
  a reader must not treat the hallucination rate as an upper bound.
- **The round-trip test alone cannot catch an extractor bug.** Three injected bugs left it
  reporting 100% SUPPORTED, because the probe renders its claims from the record and
  verifies a wrong value against itself (D-034). `tests/oracle.py` is the independent
  calibration and it is what the faithfulness numbers actually rest on.
- **The two-extractor agreement κ, which is what makes the faithfulness metric defensible
  rather than merely defined, has not been measured.** Method B needs API credentials.

---

## 4. Fairness — what we could and could not assess

**We could not assess demographic fairness, and neither substrate supports it.**

This is the honest answer and it is worth more than a hedge.

| Substrate | Why demographic fairness cannot be assessed |
|---|---|
| **AMLworld** | **Fully synthetic.** Generated by an agent-based simulator. Its accounts correspond to no real people, so a disparity measured across any attribute would be a property of the simulator's generative process and not evidence about the world. Measuring it and reporting it as a fairness result would be worse than not measuring it. |
| **Elliptic2** | **Pseudonymous and anonymised.** Nodes are clusters of Bitcoin addresses with unnamed, obfuscated features. There is no protected attribute to condition on, by design and by the licence under which it is distributed. |

Neither substrate carries age, gender, nationality, ethnicity, residence, or any proxy the
project is aware of. AMLworld carries a `bank` identifier and a currency; neither is a
demographic attribute, and treating a currency as a nationality proxy would be an
inference the data does not license.

### What we could assess, and did

Disparity across the axes the data *does* support was measured, and one result is
non-uniform:

- **Fact Coverage varies by typology**, from 0.770 (bipartite) to 0.862 (unclassified) —
  a 9-point spread. Structurally richer typologies have longer salience lists and the
  templates do not grow to match them. Reported in `RESULTS.md` §4.3.
- **The H9 omission rate is highest exactly where it matters most**: cases whose subject
  carries no illicit label are the ones whose narrative never says so.
- Encoder performance by typology: `fan_out`, `gather_scatter` and `cycle` are recoverable
  from the pooled tokens; **`stack` and `random` are not** (0.33 structural macro-F1).

These are *case-type* disparities, not demographic ones, and they are reported as such.

### What a deployer must do that we could not

A real deployment runs on real customers, and the fairness question becomes answerable at
that point and remains unanswered by anything in this repository. Before deployment:
measure narrative quality, coverage and critical-error rate stratified by whatever
protected attributes the institution lawfully holds, and treat a disparity in *coverage* as
seriously as one in error rate — an under-described case is an under-defended person.

**Nothing in this release constitutes evidence that the system is fair. It constitutes
evidence that we did not have the data to find out.**

---

## 5. Data provenance and privacy

### Provenance

| Substrate | Origin | Nature | Status here |
|---|---|---|---|
| IBM AMLworld (HI-Small) | Altman et al., NeurIPS D&B 2023 | **Synthetic**, agent-based simulator | Ingested; reproduces every published figure exactly |
| Elliptic2 | Bellei et al., KDD MLF 2024 | **Real Bitcoin**, clustered and anonymised | **Never obtained.** Access was not requested. |

Licences, obligations and what is redistributable are in `docs/data_cards/` and
summarised in `README.md`. The short form: AMLworld data is CDLA-Sharing-1.0 with
share-alike, its §3.5 exempts Results; Elliptic2's data licence **could not be located**
and the substrate is treated as closed.

### Privacy

**No real financial data, no real Suspicious Activity Report text, and no personal data of
any kind is present in this repository or in any released artifact.**

- Every narrative, fact record, case and identifier in the released corpus derives from
  **AMLworld, which is synthetic.** The account identifiers that appear in the text
  (`021611|800F41B10`) are simulator output and correspond to no person or institution.
- Real SARs are confidential by statute. None were obtained, read, or used. This is the
  reason the corpus had to be constructed at all.
- Invariant 8 forbids real-world identifiers anywhere in the repository, including in test
  fixtures. It is enforced by pre-commit hooks (`detect-private-key`, `gitleaks`) and a
  full-history secret scan in CI.
- The Phase 14 audit scanned the complete git history and the full release tree: **no
  secrets, no credentials, no email addresses, no personal identifiers.**

### Human-subject data

The Phase 12 decision-setting study would have collected data from professional raters.
**It has not run, no person has been recruited, and no human-subject data exists.**

Ethics approval has been **prepared but not submitted** (`docs/human_study/`). The consent
form, participant information sheet, compensation policy and data-management plan are
written. Should the study run, the released response data is anonymised by
`src/g2t_aml/human/study_release.py` and rater identities never leave the collecting host.
The Gold annotation protocol carries the same commitment.

---

## 6. Environmental cost

**Reported as measured, and the measurement is coarse — per-run duration was not
instrumented (`seconds` is recorded as 0.0 in `encoder_report.json`), which is a gap this
statement records rather than papers over.**

### What was actually spent

| | |
|---|---|
| GPU training performed | The Phase 7 encoder sweep only — 9 arms × 3 seeds |
| Hardware | 1 × NVIDIA RTX 2050 laptop GPU, 4 GB; i5-12450H; 7 GB RAM |
| Wall-clock span of the sweep | **6.5 h** (2026-08-03 20:29 → 2026-08-04 03:02), from the 93 run directories |
| Estimated GPU-hours | **≈ 6.5** on one 4 GB laptop GPU |
| Assumed system draw under load | ~80 W (mobile GPU ~45 W + host ~35 W); no datacentre, PUE 1.0 |
| Estimated energy | **≈ 0.52 kWh** |
| Estimated emissions | **≈ 0.25 kg CO₂e** at the IEA global-average 0.475 kg/kWh; ≈ 0.37 kg at an Indian-grid 0.71 kg/kWh |

Everything else in the project — ingestion (14 s), case extraction (4.5 min), fact
extraction, the 15,707-narrative Bronze corpus (~90 s), evaluation (41 s) and the Phase 13
benchmark — is **CPU-only by design**, and was not separately instrumented. It is bounded
above by intermittent laptop use over roughly one week and is small against the figure
above.

**This project's environmental cost is negligible, and it is negligible for a bad reason:
the expensive runs never happened.** Stating 0.25 kg CO₂e without that sentence would be
misleading by omission.

### What reproduction would cost

The figure a reader actually needs. The full 17-system, 25-run matrix at the declared seed
policy is estimated at **3–5 GPU-weeks** on a ≥24 GB card (`RESULTS.md` §3):

| | |
|---|---|
| Compute | ~4 GPU-weeks ≈ **672 GPU-hours** on an A100-40GB or equivalent |
| Assumed draw | 250 W average, datacentre PUE 1.4 |
| Estimated energy | **≈ 235 kWh** |
| Estimated emissions | **≈ 112 kg CO₂e** at global-average grid intensity |

This is an estimate from published TDP and a declared PUE, not a measurement, and it is
labelled as such wherever it appears. Anyone reproducing should run the priority order in
`RESULTS.md` §8 — S1 and A1 first, then Gate 8 — precisely so that **if the fusion layer
turns out to be decoration, the remaining twenty-three runs are never spent.** That is an
environmental argument as much as a scientific one.

The API baselines (B3, B4, B5) carry a marginal cost of USD 22.65–67.94 per 1,000
narratives at 2026-08 list pricing (`RESULTS.md` §4.8); their inference emissions are not
disclosed by the providers and are therefore not estimated here rather than guessed.

---

## 7. Accountability

- Failure modes, negative results and non-runs are recorded in `RESULTS.md` under
  invariant 7. Nothing is dropped for being unflattering.
- Every run writes `run_context.json` — git SHA, resolved config, data manifest hash, all
  seeds, library versions (invariant 5) — so any published number is traceable to the code
  and data that produced it.
- Results files are never overwritten (invariant 6).
- Corrections and material changes to this statement will be recorded in `CHANGELOG.md`
  and in `DECISIONS.md`.

**If you are evaluating this artifact for deployment, the answer is that it is not ready
for one, and §3.1 is why.**

---

## 8. See also

| Document | What it holds |
|---|---|
| `RESULTS.md` | Every number, including the nulls and the 25 named non-runs |
| `docs/data_cards/` | Provenance, licence and limitations per substrate |
| `docs/dataset_cards/` | Construction, splits and limitations per corpus tier |
| `docs/model_cards/` | Per-model intended use, evaluation, limitations, misuse |
| `docs/annotation/hallucination_taxonomy.md` | The nine classes and the Critical Error Rate |
| `docs/human_study/` | The unsubmitted ethics application, consent form and DMP |
| `docs/deployability.md` | Per-system deployment assessment on the non-GPU axes |
| `docs/REPRODUCTION.md` | How to reproduce every number, and the tolerance policy |
