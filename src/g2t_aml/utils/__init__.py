"""Cross-cutting machinery that the invariants are enforced through.

- :mod:`~g2t_aml.utils.io` -- atomic writes. A killed job must never leave a half-written
  file that a later stage reads as valid. Nothing in this repository writes a result any
  other way.
- :mod:`~g2t_aml.utils.hashing` -- the canonical hash: sorted keys, resolved Hydra
  interpolations, order-independent where order is not meaningful. Everything that feeds
  a result is hashed with it, including the config hash in a Phase 11 run path (D-085).
- :mod:`~g2t_aml.utils.run_context` -- invariant 5's artifact. Every run writes
  ``run_context.json`` carrying the git SHA, the resolved config, the data manifest hash,
  every seed and the library versions.
- :mod:`~g2t_aml.utils.seeding` -- the seeds themselves, set in one place.
- :mod:`~g2t_aml.utils.logging` -- structured logging setup.

This package is CPU-only and imports no torch at module scope, so it stays importable on
an aggregation or documentation host.
"""
