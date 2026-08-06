"""Phase 11: the experiment matrix -- declaration, orchestration, aggregation, figures.

Four modules, in the order they run:

- :mod:`~g2t_aml.experiments.registry` declares every system. A system is reproducible
  from its spec plus its Hydra config and nothing else.
- :mod:`~g2t_aml.experiments.baselines` implements B1-B6, including B5's agentic
  generate-verify-repair loop. Read its docstring before touching a baseline.
- :mod:`~g2t_aml.experiments.runner` resolves dependencies, schedules by resource class,
  resumes from completion markers and isolates failures.
- :mod:`~g2t_aml.experiments.aggregate` collects everything into one tidy table, runs the
  significance battery and emits the LaTeX; :mod:`~g2t_aml.experiments.figures` draws from
  the same table so a figure cannot disagree with a table.

Nothing here imports torch, transformers or matplotlib at module scope. The aggregation
and reporting path runs on a machine that will never train anything -- which is, at the
time of writing, the only machine this project has.
"""

from g2t_aml.experiments.registry import (
    CENTRAL_CLAIM_SYSTEMS,
    SEEDS_CENTRAL,
    SEEDS_SINGLE,
    Executor,
    FusionVariant,
    Resource,
    SystemSpec,
    TextMode,
    all_systems,
    expand_runs,
    get_system,
    matrix_summary,
    resolution_order,
    system_ids,
    validate_registry,
)

__all__ = [
    "CENTRAL_CLAIM_SYSTEMS",
    "SEEDS_CENTRAL",
    "SEEDS_SINGLE",
    "Executor",
    "FusionVariant",
    "Resource",
    "SystemSpec",
    "TextMode",
    "all_systems",
    "expand_runs",
    "get_system",
    "matrix_summary",
    "resolution_order",
    "system_ids",
    "validate_registry",
]
