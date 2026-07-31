# Annotation

The Gold-tier protocol, written in Phase 6:

- `protocol.md` — task definition, what an annotator sees, what counts as a faithful
  narrative, how to handle facts the substrate cannot support (invariant 4).
- `rubric.md` — the scoring rubric: faithfulness, completeness, investigator utility.
- `adjudication.md` — how disagreements between the two required annotators are resolved.

Agreement is reported as Krippendorff's alpha. Every case gets at least two independent
annotators plus adjudication.

**Invariant 8: no real-world PII or identifiers ever enter the repo**, and that includes
anything shown to annotators. All account identifiers presented in the annotation UI are
synthetic.
