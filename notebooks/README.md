# notebooks/

**Exploration only. Never load-bearing.**

`src/` must never import from here — there is a test asserting it. Notebooks are for
looking at data and forming hypotheses; anything another module depends on gets promoted
into `src/g2t_aml/` with tests, and the notebook keeps only the call.

Notebooks are excluded from ruff and from coverage. Strip outputs before committing.
