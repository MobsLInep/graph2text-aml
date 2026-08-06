# Dataset card — Gold corpus

## **This corpus does not exist. Not one narrative has been written.**

| | |
|---|---|
| Card written | 2026-08-06 (Phase 14), for release `v0.1.0` |
| Narratives | **0** |
| Blocker | **No annotator has been recruited.** Recruitment is the critical path and no further engineering advances it. |
| Machinery | Complete, validated end to end on real cases with hand-written narratives |
| **Sample** | **350 cases, drawn, stratified, reserved test-only, and committed** |
| Code | `src/g2t_aml/human/` · gate: `make gold-gate` |

The sample is real and committed. The narratives are not written. Those are two different
facts and this card keeps them apart.

---

## 1. What it would be

Human-authored SAR narratives written by trained annotators under the protocol in
[`../annotation/`](../annotation/), against the fact record and the case subgraph — and
**never against a generated draft**.

Gold is the project's reference tier: **small, held out, and never used for training.**
Every Layer 1 overlap metric (BLEU, ROUGE, METEOR, BERTScore) is scored against it, which
is why all of them are currently unmeasured.

---

## 2. The sample — this part is real

Drawn by `make gold-sample`, stratified in three blocks, and **reserved test-only** with
its own committed ID list and content hash.

| | |
|---|---|
| Selected | **350** of 350 requested, all from the frozen **test** split |
| Hard negatives | **99 (28.3%)**, against a 25% floor |
| Typed typologies | fan_out 20 · fan_in 20 · bipartite 20 · cycle 20 · gather_scatter 19 · random 19 · scatter_gather 18 · stack 18 (spread ≤ 2) |
| `unclassified` | 196 (99 hard negatives + 97 licit / suspicious-unclassified) |
| Size buckets | small 157 · medium 129 · large 64 |
| Substrates | amlworld_hi_small 350 · **elliptic2 0** |
| Reservation | sha256 `be2512b5…`, committed to `schemas/splits/amlworld/` |

**A deficit is recorded rather than hidden**: 105 Elliptic2 cases were requested, 0
supplied, and all 105 were reallocated to AMLworld. Three single-account cases were
excluded. `test.txt` and its sha256 are **untouched** — the reservation is a subset of the
existing test split recorded beside it, and `load_reservation` asserts every reserved ID
really is a test-split member (invariant 2).

---

## 3. Two rules that run through the whole kit

**An annotator is never shown generated text.** No Bronze, no Silver, no model output.
`caseloader.py` has no narrative field of any kind, `store.py` refuses generated text, and
`tests/integration/test_repo_contract.py` asserts that **no annotator-facing module can
reach a narrative**. This is enforced structurally, not by instruction, because an
annotator who has read a draft is writing a paraphrase and the tier stops being independent.

**A Gold test item is never trained on**, enforced by
`corpus/training_data.load_training_records` rather than by anyone remembering.

### And one trap that would have corrupted every monetary case

**Anything shown to an annotator must be spelled the way the alignment reads it back**
(D-054). The fact panel renders through *Bronze's* formatters and display maps — not the
vocabulary's, and not its own. A panel showing `9,434.82 Canadian Dollar` against Bronze's
`9,435 Canadian Dollar` means an annotator who copies it **correctly** produces a value
that aligns to nothing: scored as a dropped fact *and* an invented one, on every monetary
case. The same trap applies to roles (`conduit account` vs `a conduit account`) and to the
threshold citation, where the non-whitelisted wording is a critical H6 error.

---

## 4. What is verified, and what that does not establish

`make gold-gate` runs 214 Phase 6 tests over sampling, reservation, the fact panel, the
graph view, live validation, the store, calibration, agreement and ingestion. The
**ingestion path is verified end to end on real cases with hand-written narratives** — so
the pipeline from an annotator's text to `tier="gold"` records demonstrably works.

**This establishes that the kit is correct. It establishes nothing about inter-annotator
agreement, calibration pass rates, or the quality or register of human narratives on this
task**, because no annotator has used it.

---

## 5. Blockers

| Blocker | Detail |
|---|---|
| **An annotator** | Recruitment. `docs/annotation/recruitment.md` states the target profile, the sources in preference order, and the **explicit refusal of untrained crowdworkers** with the reason. The paper-facing description of annotator expertise was written *before* recruitment, so it describes the standard set rather than the people found. |

### What this blocker also blocks

- **Every Layer 1 overlap metric.** BLEU, ROUGE, METEOR and BERTScore are all scored
  against Gold.
- **The template-baseline finding** — does a deterministic template score competitively on
  ROUGE against a trained model? `template_baseline_finding` runs the day one Gold
  narrative exists, and it is one of the more interesting questions the project can ask
  cheaply.
- The held-out human reference for the Phase 12 study.

---

## 6. Limitations this tier would have

- **350 cases is small**, and it is small deliberately — it is a reference, not training
  data. Every metric scored against it carries that sample size.
- **AMLworld only.** The 105-case Elliptic2 block was requested and could not be supplied,
  so the reference has no second-substrate half.
- **Human agreement is unknown.** The machinery computes Cohen's κ, Krippendorff's α,
  content Jaccard and text F1, all dependency-free and hand-verified; there is nothing to
  compute them over.
- **Annotator expertise will be a real constraint.** The protocol refuses untrained
  crowdworkers because SAR narrative writing is a professional skill, which narrows the
  recruitable pool sharply and is the reason this blocker has not cleared.
- Everything in [`README.md`](README.md) § Common limitations.

---

## 7. Licence

**CDLA-Sharing-1.0.** Gold narratives describe AMLworld cases and would carry the same
identifiers, timestamps and amounts as Bronze — Enhanced Data on the reasoning in
[`bronze.md`](bronze.md) §7.

Should the tier ever be written, the annotators' **authorship and any consent terms** are a
second, independent constraint on top of the data licence, and must be settled with the
annotators before release. `docs/human_study/` holds the consent and data-management
templates.

---

## 8. Intended use and misuse

If written: the held-out human reference for all overlap metrics, and the target register
for generation. **Never training data** — enforced in code.

`docs/ETHICS.md` §2 applies in full. Specific to this tier: **Gold must not be moved into
training to improve a number.** The moment it is, every metric scored against it becomes
meaningless, and the enforcement lives in `training_data.py` precisely because that
temptation arrives late, under deadline, when someone is looking for one more point.

---

## 9. Reproduction

```bash
make gold-sample                        # draw and reserve the 350 cases (already committed)
make guidelines-pdf                     # render the annotation guidelines + recruitment brief
make calibrate ANNOTATOR=annotator-01   # the ten-case calibration gate
make annotate  ANNOTATOR=annotator-01   # the annotation interface
make gold                               # ingest reviewed annotations -> gold.jsonl + the gate
make gold-gate                          # the tests
```

`make gold-sample` and `make gold-gate` run today. The middle three need a person.
