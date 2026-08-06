# Fallback: internal expert review

**Version:** 1.0 · **Date:** 2026-08-05

What happens if ethics approval will not arrive in time, and the honest terms on which the
weaker thing is reported.

---

## The trigger date: 2026-09-15

If the ethics application has **not been submitted** by 2026-09-15, the external study
cannot realistically complete before the paper's target submission, and the project
switches to the fallback.

The arithmetic behind that date:

| Step | Duration |
|---|---|
| Committee review and approval | 4–8 weeks |
| Recruitment of 6–10 AML-literate participants | 3–4 weeks, overlappable with the above |
| Training and data collection | 3 weeks |
| Analysis and writing | 1 week |
| **Total from submission** | **9–13 weeks** |

The date is recorded here so that the decision is made on a calendar rather than in the
week the deadline arrives, when the temptation is to run something ungoverned and describe
it generously.

**Note that the trigger is submission, not approval.** An application submitted on time but
approved late leaves the project waiting with everything else ready, which is a
recoverable position. An application never submitted is not.

## The fallback: what it is

An **internal expert review** conducted under a different governance model.

- **Reviewers:** 2–4 members of the research group and immediate academic collaborators
  with financial-crime domain knowledge. Not recruited from outside; not paid a
  participation fee; acting within their normal research role.
- **Governance:** internal peer review of research outputs, not human-subjects research.
  No personal data is collected about the reviewers beyond a role label, no publication of
  individual-level data, and therefore no ethics approval required. **This exemption holds
  only because the individual-level data is not published.** If any of it were to be
  released, the fallback would need approval like anything else.
- **Instrument:** the same interface, the same blinding, the same scales, the same anchored
  training pack. The measurements are identical; only who is making them, and under what
  governance, differs.
- **Design:** the same balanced incomplete block design, at whatever panel size is
  available. With 2–4 reviewers the anchor block becomes proportionally larger, so the
  agreement statistic remains computable.

## What it is not

It is **not** a substitute for the external study, and the paper must not present it as
one. Its limitations are real and are stated in the paper's limitations section verbatim:

1. **The reviewers are not independent of the authors.** They know the project, they may
   be able to infer which arm is the proposed method from its style, and blinding does not
   remove an incentive it cannot detect. This is the most serious limitation and it is
   listed first.
2. **The panel is small.** With 2–4 reviewers the Friedman test is blocked on 2–4 blocks
   and is severely underpowered; the case-blocked Durbin test carries most of the
   inferential weight, and both are reported with that stated.
3. **The panel is academic, not practitioner.** They are not the population the deployment
   claim is about. A time-to-draft reduction measured on researchers is evidence that the
   drafts are easier to work with; it is **not** a measurement of investigator drafting
   time in a compliance function, and the paper will not call it one.
4. **No participant-level data is released**, so this part of the evaluation is not
   externally reproducible in the way the rest of the project is.

## How it is reported

- Described as "internal expert review" throughout. **The words "user study",
  "expert evaluation" and "practitioner evaluation" are not used of it.**
- Reported in its own subsection, with the four limitations above stated in that
  subsection rather than deferred to a general limitations paragraph at the end.
- The reviewers' actual backgrounds given in the same honest terms the main protocol
  requires — number, roles, and years of relevant experience.
- Presented as **preliminary evidence motivating a properly governed external study**,
  which is what it is.

A reviewer at *Expert Systems with Applications* who reads "internal expert review, four
academic reviewers, here are its four limitations" is being told the truth and can weigh
it. A reviewer who reads "expert evaluation" and discovers the panel were the authors'
colleagues will not be charitable about anything else in the paper either.

## What is still required under the fallback

Everything that does not depend on external participants still applies:

- [x] The blinding holds, and is asserted by the same test.
- [x] The design is validated by the same code.
- [x] Time-to-usable-draft and edit distance are measured the same way.
- [x] Krippendorff's α is reported, with its small-panel caveat.
- [x] The automatic-versus-human correlation is computed — this is the fallback's single
      most valuable output, because validating the automatic metric is what licenses the
      rest of the paper's numbers, and even a small panel contributes to it.

## If both routes fail

If the fallback also cannot be run — most likely because there are still no stimuli beyond
Bronze — then Layer 3 is reported as **not performed**, with the reason, in `RESULTS.md`
and in the paper. That is invariant 7 applied to a phase that produced no results, and it
is what the project has already done for Phases 9 and 11.

It is a materially weaker paper. It is not a dishonest one.
