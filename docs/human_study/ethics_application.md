# Research ethics application — decision-setting evaluation of generated SAR narratives

**Status: DRAFT, NOT SUBMITTED.** Prepared 2026-08-05. Submission is the critical path for
Phase 12; approval typically takes 4–8 weeks and blocks all data collection.

Complete the bracketed institutional fields before submission — they are left blank
deliberately rather than filled with placeholders that could be mistaken for real values.

---

## 1. Administrative

| Field | Value |
|---|---|
| Project title | Graph2Text AML: generating Suspicious Activity Report narratives from transaction subgraphs |
| Study title | Decision-setting evaluation of generated SAR narratives |
| Principal investigator | [NAME, POSITION, DEPARTMENT] |
| Co-investigators | [NAMES] |
| Institution | [INSTITUTION] |
| Sponsor / funder | [FUNDER, GRANT REF, or "unfunded doctoral research"] |
| Anticipated start | On approval |
| Anticipated end | Approval + 10 weeks |
| Risk category applied for | Low risk — competent adults, professional judgement task, no sensitive personal data |

## 2. Summary in plain language

Financial institutions are legally required to file Suspicious Activity Reports (SARs)
when they detect potential money laundering. Writing the narrative section of a SAR is
slow, manual work performed by financial-crime investigators. This project builds a system
that drafts that narrative automatically from the transaction data.

Automatic text-quality metrics cannot tell us whether such a draft is *useful to an
investigator*. This study asks people with anti-money-laundering knowledge to read drafts
produced by several different systems, rate them, correct them to a state they would be
willing to file, and be timed while doing so. We measure how long each draft takes to
make usable and how much of it has to be changed.

Participants judge software output. They are not asked about their own institutions,
their own cases, or anything confidential.

## 3. Aims and value

1. Establish whether generated drafts reduce investigator drafting time against a
   template baseline. This is the deployment-relevant claim.
2. Establish how much correction each system's output requires.
3. Validate the project's automatic faithfulness metric against expert judgement, so that
   the remaining (much larger) automatic evaluation can be trusted.

Without (3), every automatic number in the resulting paper rests on an unvalidated proxy.

## 4. Design

Blinded, balanced incomplete block design.

- **Participants:** 6–10 AML-literate adults.
- **Stimuli:** 80–120 anonymised, synthetic transaction cases, each rendered by 4–5
  systems. Participants see one system's draft per case and are never told which.
- **Per participant:** approximately 60 items, expected 4–6 hours, split across
  self-paced sessions with save-and-resume. No session need exceed 60 minutes.
- **Blinding:** system identity is held in a separate file the rating interface never
  loads. Enforced in code and asserted by an automated test.
- **Balance:** no participant sees the same case twice; system-to-position assignment is
  balanced by Latin square; 5% of items are repeats, for intra-rater reliability.

Full technical specification: `src/g2t_aml/human/study_design.py`.

## 5. Participants

### 5.1 Inclusion criteria

Adults (18+) with demonstrable working knowledge of anti-money-laundering concepts:
current or former financial-crime investigators, compliance analysts, AML consultants,
financial-intelligence-unit staff, regulators, or postgraduate students on a specialised
financial-crime programme who have completed relevant coursework.

### 5.2 Exclusion criteria

- Anyone unable to give informed consent.
- Anyone in a supervisory or assessment relationship with the investigator (see §8).
- **Untrained crowdworkers are excluded categorically.** The task requires professional
  judgement about regulatory tone and evidential sufficiency; a panel without it would
  produce numbers that look like results and are not.

### 5.3 Recruitment

Professional networks, AML practitioner associations, LinkedIn professional groups, and
university financial-crime programmes. No cold contact of individuals at their employers.
Recruitment materials state the time commitment and compensation up front.

### 5.4 Honest description of expertise

The panel will be described in the resulting publication **by what it actually is**,
including the proportion who are students rather than practitioners and their median years
of relevant experience. The paper will not use the unqualified word "expert" unless every
panel member is a practising professional. Overstating a panel is both an integrity
failure and, in our judgement, a reviewing risk.

## 6. Consent

Written informed consent obtained before any data is collected, via
`consent_form.md`, after the participant has read `participant_information.md` and had an
opportunity to ask questions.

Consent is sought separately for:

1. Participation.
2. **Publication of anonymised response data in a public research repository.**

(2) is optional and refusing it does not prevent participation; a participant who declines
has their data used for the analysis but excluded from the public release. This is
separated deliberately because "your data will be published" is a materially different
proposition from "you will take part", and bundling them makes the consent to the first
questionable.

## 7. Data

Summarised here; full detail in `data_management_plan.md`.

**Collected:** pseudonymous participant identifier; a self-reported professional background
category; per item — five ordinal ratings, one binary filing decision, active time in
seconds, the presented and corrected narrative texts, and tab-visibility counts.

**Not collected:** name, contact details beyond what is needed to pay the participant
(held separately by [FINANCE/ADMIN FUNCTION], never in the research dataset), employer,
any information about real cases, any special-category data under UK GDPR Art. 9.

**Stimuli contain no real personal data.** Cases are drawn from IBM AMLworld, a published
*synthetic* dataset, and optionally Elliptic2, which is anonymised public blockchain data.
No real SAR, no real customer, and no real transaction of an identifiable person is shown
at any point. This is enforced in the codebase by a project invariant and a pre-commit
scanner.

**Legal basis (UK GDPR):** Art. 6(1)(e) task in the public interest (scientific research),
with appropriate safeguards under Art. 89(1).

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Fatigue from a repetitive multi-hour task | Self-paced, save-and-resume, no session need exceed an hour, participants told to stop whenever they wish |
| Participant feels professionally evaluated | Explicitly stated in the information sheet that the *systems* are under evaluation and no individual performance measure is computed or reported. Intra-rater reliability is reported at panel level only |
| Re-identification from the public dataset | Free-text comments dropped entirely; identifiers re-pseudonymised under a separate salt; corrections passed through an automated identifier scanner and withheld if flagged. See `data_management_plan.md` §5 |
| Participant inadvertently types confidential information into a correction box | Instructed not to; automated scanner withholds any response that trips it; withheld items are counted in the public manifest so the exclusion is visible |
| Coercion where the investigator knows the participant professionally | Anyone in a supervisory or assessment relationship with the investigator is excluded. Others are told in writing that declining has no consequence and that they may withdraw without giving a reason |
| Time burden disproportionate to compensation | See `compensation.md`: paid at a professional rate against a piloted time estimate, and paid for time actually spent if it exceeds the estimate |

**No deception is used.** Participants are not told which system produced a draft, and this
withholding is disclosed to them in advance as a necessary feature of a blinded comparison.
That is blinding, not deception, and the information sheet says so explicitly.

## 9. Withdrawal

A participant may stop at any point without giving a reason and without any effect on
compensation for work already done. They may withdraw their data up to the point of
publication of the anonymised dataset, after which withdrawal is not technically possible;
this cut-off is stated in the information sheet and the consent form, with the date to be
supplied once the analysis schedule is fixed.

## 10. Dissemination

Aggregate results in a peer-reviewed journal article (*Expert Systems with Applications*),
and the anonymised item-level response data in a public repository under
`data_management_plan.md` §5. No participant is identifiable in either.

## 11. Declarations

- [ ] No participant will be approached before written approval is received.
- [ ] The study will be conducted as described, and any change resubmitted as an amendment.
- [ ] Data will be handled as described in `data_management_plan.md`.
- [ ] The PI has completed the institution's research-integrity and data-protection training.

**Signed:** [PI] **Date:** [DATE]

---

## Attachments

1. `participant_information.md`
2. `consent_form.md`
3. `data_management_plan.md`
4. `compensation.md`
5. `rater_training.md`
6. Screenshots of the rating interface — capture from `make study-rate RATER=rater-01`
   once stimuli exist. **Not yet producible:** only the Bronze arm has generations, so
   `make study-build` currently refuses to construct a design (see `README.md`).
