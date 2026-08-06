# Annotation

## Written in Phase 3, and deliberately so

These two are frozen alongside the fact layer, **before any narrative exists**. Both are
generated from the same source of truth the automated metric uses, so human annotators and
the machine score against one definition rather than two that drifted apart.

- `salience.md` — the per-typology lists of fields an adequate narrative must mention.
  Fixed now because salience decided after seeing model output describes whatever the model
  produced, and every adequacy number measured against it afterwards is circular (D-032).
- `hallucination_taxonomy.md` — the nine hallucination classes, their severities, and why
  H4/H6/H7 aggregate into a separately-reported Critical Error Rate.

## Written in Phase 6

- `annotation_guidelines.md` — the protocol, released with the corpus and cited in the
  paper: what a SAR is and why suspicion is not guilt, the four-part structure, the eight
  typologies, the six rules, **five fully worked examples** from real reserved cases, and
  the H1–H9 error taxonomy with a wrong/right pair for each. Exported to PDF by
  `make guidelines-pdf`.
- `recruitment.md` — the annotator profile, the sources, the refusal of untrained
  crowdworkers, the calibration gate, and the honest description of expertise that goes
  into the paper.

**Phase 0 planned three files here — `protocol.md`, `rubric.md` and `adjudication.md` —
and they are one document and some code instead.** The protocol and the rubric are one
thing written twice: an annotator who is told what to write and then scored against a
separate rubric is being held to a standard they were not given. They are Parts B–D and
the Appendix of the guidelines. Adjudication is not prose at all — it is
`g2t_aml.human.review`, which *refuses* a disputed review that carries no adjudication and
refuses an adjudicator who was party to the dispute. A markdown file describing that
process would have been a description of a rule nothing enforced.

**Agreement** is reported as Cohen's κ on typology over the double-annotated subset,
Krippendorff's α over the pooled judgements, Jaccard over salient-field selection, and
token F1 between narratives — the last expected low, and reported as evidence that SAR
narrative writing has legitimate variance. Phase 0 said "every case gets at least two
independent annotators"; that is **15% of cases**, assigned deterministically by case id
before annotation begins. Double-annotating all 350 would double the cost of the phase's
scarcest resource to improve a statistic that 52 pairs already estimate adequately.
**Every** case gets a second-reviewer pass against its fact record, which is the stronger
guarantee and the one that survived.

**Invariant 8: no real-world PII or identifiers ever enter the repo**, and that includes
anything shown to annotators. All account identifiers presented in the annotation UI are
synthetic.
