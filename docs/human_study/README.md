# Phase 12 — the decision-setting study

This directory is the governance half of Phase 12. The engineering half is
`src/g2t_aml/human/study_*.py`.

## Status, as of 2026-08-05

**Ethics approval has not been granted. It has not been applied for.** No participant may
be approached and no data may be collected until it is. Everything in this directory
except `ethics_application.md` is written to be *submitted*, not to be used yet.

Two further blockers stand behind the ethics one, and they are the reason the study could
not have run this week even with approval in hand:

| Blocker | State |
|---|---|
| Ethics approval | Not applied for. 4–8 weeks from submission. |
| Stimuli | **Only Bronze exists.** S1/S2 need a GPU the project does not have (D-068); B7/B3 and Silver need API credentials. A five-arm study has one arm. |
| Raters | None recruited. Phase 6's annotator recruitment has not produced a person either. |

A study comparing one system against itself is not a study. See `PHASE_LOG.md` Phase 12.

## Files

| File | What it is | For |
|---|---|---|
| `ethics_application.md` | The application, ready to submit | The committee |
| `participant_information.md` | What a participant is told before consenting | Participants |
| `consent_form.md` | What they sign, including publication consent | Participants |
| `data_management_plan.md` | What is collected, where it lives, how long, how anonymised | Committee + DPO |
| `compensation.md` | What participants are paid and on what basis | Committee + finance |
| `rater_training.md` | The 30-minute training pack with anchored scales | Raters |
| `fallback_governance.md` | What happens if approval will not arrive in time | The project |

## The decision that has to be made now

`fallback_governance.md` sets a **trigger date of 2026-09-15**. If the ethics application
has not been submitted by then, the external study cannot complete before the paper's
target submission, and the project switches to the internal expert review described in
that document — reported honestly as such, with its limitations stated in the paper.

That is a materially weaker piece of evidence. It is chosen over nothing, and it is not
chosen over a real study unless the calendar forces it.
