"""The three-tier narrative corpus: Bronze, Silver, Gold.

There is no existing corpus of (graph, SAR narrative) pairs anywhere -- real SARs are
confidential by statute -- so this package constructs one, in three tiers that differ in
how the text was produced and not in what it must satisfy:

- :mod:`~g2t_aml.corpus.bronze` renders deterministically from the fact record. Faithful
  by construction, stylistically flat, and the floor every other system must beat.
- :mod:`~g2t_aml.corpus.silver` rewrites Bronze with two teacher models from different
  families, each rewrite gated by the Phase 3 checker: at most two targeted repair
  attempts, then discard-and-log (D-046). The discard log is a deliverable.
- Gold is human-authored under ``docs/annotation/``, small, held out, never trained on.

:mod:`~g2t_aml.corpus.validate` is the ten-point harness, and it gates all three tiers
identically -- Silver and Gold differ from Bronze only in ``tier`` and ``generator``
(D-037). ``training_record_v1.json`` is frozen at 1.0.0 and carries all three.
:mod:`~g2t_aml.corpus.training_data` is what keeps a Gold test item out of training, so
that no one has to remember to.

Submodules are imported explicitly; this package deliberately re-exports nothing, so that
importing :mod:`g2t_aml.corpus` never drags in the Silver teacher clients (and with them
the optional ``api`` extra).
"""
