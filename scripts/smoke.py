#!/usr/bin/env python
"""End-to-end smoke test: compose Hydra, seed, capture provenance, write a run dir.

This is the registered smoke entrypoint that `make smoke` and CI run. It exercises the
Phase 0 contract and nothing else: config composition resolves, the resolved config is
persisted, seeding is reproducible, and `run_context.json` lands in the run directory.

Usage:
    uv run python scripts/smoke.py
    uv run python scripts/smoke.py experiment=full data=elliptic2
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from g2t_aml.utils.hashing import hash_config, short
from g2t_aml.utils.io import write_json
from g2t_aml.utils.logging import configure_logging, get_logger, stage
from g2t_aml.utils.run_context import RunContext
from g2t_aml.utils.seeding import seed_everything

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


#: Exit code captured out of the Hydra-decorated entrypoint.
#:
#: ``@hydra.main`` **discards its wrapped function's return value** — it returns None
#: regardless — so the long-standing ``sys.exit(main())`` always exited 0 and every
#: documented "exits non-zero when the gate fails" in this repository was silently untrue.
#: A failing gate looked identical to a passing one to CI, to ``make``, and to any caller
#: checking ``$?``. Capturing the code out of a module-level cell is the smallest fix that
#: keeps the Hydra entrypoint shape. Found in Phase 5; see PHASE_LOG.
_EXIT_CODE: list[int] = []


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def _run(cfg: DictConfig) -> None:
    """Run the Phase 0 smoke check.

    Args:
        cfg: Composed Hydra configuration.

    Returns:
        0 on success.

    Raises:
        RuntimeError: If the resolved config or run context could not be written, which
            means the provenance guarantees in invariant 5 do not hold.
    """
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    configure_logging(log_file=run_dir / "smoke.log")
    log = get_logger(__name__)

    with stage(
        "smoke",
        log,
        experiment=cfg.experiment.name,
        data=cfg.data.name,
        encoder=cfg.encoder.name,
        fusion=cfg.fusion.name,
        corpus_tier=cfg.corpus.tier,
        run_dir=str(run_dir),
    ) as summary:
        seeds = seed_everything(cfg.seed, deterministic=cfg.deterministic)

        # Persist the fully-resolved config next to the results. Hydra already writes
        # .hydra/config.yaml, but that copy is unresolved; this one has interpolations
        # baked in and is what the config hash is taken over.
        resolved = OmegaConf.to_container(cfg, resolve=True)
        write_json(run_dir / "resolved_config.json", resolved)

        ctx = RunContext.capture(
            experiment_name=cfg.experiment.name,
            cfg=cfg,
            seeds=seeds,
            repo_root=Path(__file__).resolve().parents[1],
            phase="0",
        )
        ctx_path = ctx.save(run_dir)

        if not ctx_path.exists() or not (run_dir / "resolved_config.json").exists():
            raise RuntimeError(f"provenance files missing from run dir {run_dir}")

        summary["run_id"] = ctx.run_id
        summary["git_sha"] = ctx.git_sha
        summary["config_hash"] = short(hash_config(cfg))
        summary["seeded_backends"] = [k for k in ("numpy", "torch") if seeds.get(k)]
        summary["run_dir"] = str(run_dir)

    _EXIT_CODE.append(0)


def main() -> int:
    """Run the Hydra entrypoint and return its exit code.

    Returns:
        The code ``_run`` produced, or 0 when it produced none.
    """
    _run()
    return _EXIT_CODE[-1] if _EXIT_CODE else 0


if __name__ == "__main__":
    sys.exit(main())
