"""The three learned components, in the order signal flows through them.

- :mod:`~g2t_aml.models.encoder` (Phase 7) -- six message-passing arms behind one
  ``BaseEncoder``, differing *only* in ``message_passing``. GATv2 is primary. The MLP
  control has no message passing at all and reaches 0.80 AUC-PR, so every claim about
  what graph structure contributes is a claim about the 0.07 margin above it, not about
  the whole number.
- :mod:`~g2t_aml.models.fusion` (Phase 8) -- the projection of pooled graph tokens into
  the language model's embedding space. This is the technical contribution. The projector
  trains in fp32 and is never quantised (D-069).
- :mod:`~g2t_aml.models.generator` (Phase 9) -- the QLoRA finetuning harness, the guard,
  and the arm definitions. Loss is on the completion only, never the prompt, the soft
  tokens or the padding; three learning rates, not one (D-070).

Everything here imports torch. Phases 1-6 and 10 are CPU-only by design and must never
reach into this package -- ``tests/integration/test_repo_contract.py`` enforces it.

**No generator arm has been trained.** The fusion layer and the training harness are
built and tested on CPU against a stub backbone; Gate 8 is open. See D-068 and
``PHASE_LOG.md`` Phase 8+9 before reading any number attached to these modules.
"""
