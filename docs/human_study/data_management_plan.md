# Data management plan — Phase 12 decision-setting study

**Version:** 1.0 · **Date:** 2026-08-05
**Applies to:** the human rating study only. The project's corpus and model artifacts are
covered by `docs/data_cards/`.

---

## 1. What is collected

### 1.1 Research dataset (the responses)

Per participant, once:

| Field | Example | Why |
|---|---|---|
| `rater_id` | `rater-03` | Pseudonymous key. Assigned by the researcher; never derived from a name |
| background category | `compliance analyst, 5-10 years` | So the panel is described honestly in the paper |

Per item, ~60 times per participant:

| Field | Why it is needed |
|---|---|
| `item_id` | Opaque key joining the response to its condition |
| `case_id` | Which case |
| `position`, `is_repeat` | Order effects; intra-rater reliability |
| five ordinal ratings (1–7) | The rating scales |
| `would_file` | The binary decision |
| `seconds_to_usable_draft` | **Headline measure.** Active seconds, tab-hidden time excluded |
| `timing_source`, `hidden_seconds`, `n_blurs` | Whether that exclusion actually happened |
| `presented_narrative`, `corrected_narrative` | **Headline measure.** Edit distance is computed between them |
| `comment` | Optional free text. Internal only — never published (§5) |
| `submitted_at` | Internal only — never published (§5) |

### 1.2 Administrative data (held separately)

Name, contact details, payment details, and the signed consent form. Held by
[FINANCE/ADMIN FUNCTION] for payment and audit. **Never joined to the research dataset**,
and not accessible to the analysis pipeline.

### 1.3 What is deliberately not collected

Employer; job title beyond the broad category; anything about real cases the participant
has worked on; any special-category data under UK GDPR Art. 9; IP addresses; browser
fingerprints; keystroke-level telemetry.

## 2. What the participants are shown

Synthetic and anonymised public data only:

- **IBM AMLworld (HI-Small)** — published *simulated* transaction data, CDLA-Sharing-1.0.
- **Elliptic2** — anonymised public blockchain data, gated licence, not redistributable.

No real SAR, customer or identifiable person appears at any point. This is project
invariant 8 ("no real-world PII or identifiers ever enter the repo"), enforced by a
pre-commit scanner and by tests over the fixtures.

## 3. Storage and access

| Data | Location | Access |
|---|---|---|
| Responses, during collection | `artifacts/human_study/responses/` on [INSTITUTIONAL SERVER / ENCRYPTED PROJECT VOLUME] | PI and named co-investigators |
| Blind key (`key.json`) | Same volume, separate directory, **not readable by the rating interface** | PI only, until the study closes |
| Signed consent forms | [INSTITUTIONAL SECURE STORE], separate from the above | PI and [ADMIN] |
| Payment details | [FINANCE SYSTEM] | [FINANCE] |
| Rater re-identification map (`*_rater_map.PRIVATE.json`) | Stored **with the consent forms**, never with the release | PI only |

Encryption at rest per institutional policy. Access by named individuals only. The
response files are append-only and written atomically, so an interrupted session leaves a
complete previous file rather than a truncated one.

## 4. Retention

| Data | Retention | Then |
|---|---|---|
| Signed consent forms | 10 years from publication, per [INSTITUTIONAL POLICY] | Securely destroyed |
| Contact and payment details | 7 years, per financial-audit requirement | Securely destroyed |
| Rater re-identification map | Destroyed on publication of the anonymised dataset — it has no further research purpose and its only effect afterwards is to make re-identification possible | Destroyed |
| Internal response dataset (with comments) | 10 years from publication | Securely destroyed |
| Published anonymised dataset | Indefinite, as a permanent research record | Retained |

## 5. Anonymisation before publication

Implemented in `src/g2t_aml/human/study_release.py`; tested in
`tests/unit/test_study_release.py`. Four steps, each addressing a distinct re-identification
route:

1. **Identifiers are re-pseudonymised.** `rater-03` already appears in the recruitment
   records, the payment schedule and the consent forms, so anyone holding those could join
   them to the release. The published labels are re-derived through a keyed digest under a
   salt **not shared with the study design**, and sorted by digest so the ordering carries
   no information. The mapping is written outside the release directory, marked PRIVATE,
   and destroyed on publication.

2. **Free text is dropped, not filtered.** Comments are deleted entirely. A comment such as
   "I saw this pattern at my last employer" is identifying, and no pattern-matching rule
   catches the general case, so the field does not survive at all.

3. **Corrections are scanned and withheld if flagged.** The corrected narratives *are*
   published, because the edit distance is a headline measurement and a release that
   cannot reproduce it is not reproducible. Each is passed through the Phase 4 identifier
   scanner first; anything flagged is withheld, and its item id and the reason appear in
   the public manifest, so the released count reconciles against the design and the
   exclusion is visible rather than silent.

4. **Timestamps are removed.** A sequence of submission times is a record of when a named
   professional was working.

**System labels are revealed at this point**, and only at this point. The blind key is an
input to the release and to the analysis, and to nothing else.

## 6. Sharing

Anonymised dataset deposited in [ZENODO / INSTITUTIONAL REPOSITORY] under
[CC-BY-4.0 or CC0], with a DOI cited in the paper. It contains the responses, the block
design, the verbatim scale anchors, a manifest with content hashes, and a README stating
the two caveats a re-analyst must know: which rows were timed by the fallback clock, and
that Likert means must not be reported without an agreement statistic.

The analysis notebook (`notebooks/12_human_study.ipynb`) is released with it, so every
published statistic is recomputable from the released files rather than taken on trust.

Participants who decline Part B of the consent form are excluded from the deposit and
included in the analysis.

## 7. Responsibilities

| Role | Person | Responsible for |
|---|---|---|
| Data controller | [INSTITUTION] | Legal compliance |
| Principal investigator | [NAME] | Everything in this plan |
| Data protection officer | [NAME] | Review and advice |

**Legal basis (UK GDPR):** Art. 6(1)(e), task in the public interest (scientific
research), with Art. 89(1) safeguards. No special-category data is processed, so Art. 9
does not apply.

## 8. Breach procedure

Any suspected breach is reported to [DPO] within 24 hours of discovery and handled under
[INSTITUTIONAL BREACH POLICY], including assessment of whether the ICO and the affected
participants must be notified within 72 hours.
