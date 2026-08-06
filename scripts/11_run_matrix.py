#!/usr/bin/env python
"""Phase 11: plan and run the full experiment matrix.

Resolves the registry into (system, seed) runs, orders them so dependencies precede
dependents, skips runs that are already complete under the same config hash, and executes
the rest -- serialising GPU work and parallelising the CPU and API work. One system
failing does not abort the matrix.

**Start with ``--dry-run``.** It prints the plan, the resource split and the seed policy
without touching a GPU or spending an API dollar, and it is how you check that the matrix
you are about to run is the matrix you meant.

Usage:
    uv run python scripts/11_run_matrix.py --dry-run
    uv run python scripts/11_run_matrix.py --systems B1,B2
    uv run python scripts/11_run_matrix.py --systems S1,A1 --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from g2t_aml.experiments.registry import (
    Executor,
    SystemSpec,
    UnknownSystemError,
    all_systems,
    get_system,
    matrix_summary,
    validate_registry,
)
from g2t_aml.experiments.runner import RunStatus, plan_matrix, run_matrix
from g2t_aml.utils.io import write_json
from g2t_aml.utils.logging import configure_logging, get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]


def _not_implemented(reason: str):  # noqa: ANN202 -- returns a closure, typed at use
    """Build an executor that fails with a stated reason.

    Every arm the matrix declares has an executor registered, and an arm that cannot run
    here fails with WHY rather than being absent from the plan. A system missing from the
    plan disappears from the results table; a system that fails with "needs a >=24 GB GPU"
    appears in RESULTS.md as a documented non-run, which is what invariant 7 requires.

    Args:
        reason: Why this executor cannot run in this environment.

    Returns:
        An executor callable that raises with that reason.
    """

    def _executor(spec: SystemSpec, seed: int, directory: Path) -> dict[str, Any]:  # noqa: ARG001 -- the signature is the ExecutorFn contract
        raise RuntimeError(f"{spec.system_id} (seed {seed}) not runnable here: {reason}")

    return _executor


def build_executors(*, allow_gpu: bool, allow_api: bool) -> dict[str, Any]:
    """Assemble the executor table for this environment.

    Args:
        allow_gpu: Whether a GPU large enough for the generator is available.
        allow_api: Whether teacher-API credentials and spend authorisation exist.

    Returns:
        Executor name to callable. The CPU arms are wired to their real implementations;
        the GPU and API arms are wired either to their implementations or to a failing
        executor carrying the blocker's name.
    """
    from g2t_aml.experiments.executors import (
        run_api_baseline,
        run_local_zero_shot,
        run_template_baseline,
        run_trained_generator,
    )

    gpu_blocker = (
        "requires a GPU with >=24 GB (D-068: this machine has a 4 GB RTX 2050 and 7 GB "
        "of system RAM; Llama-3.1-8B at nf4 is ~4.5-5.6 GB of weights alone)"
    )
    api_blocker = (
        "requires teacher-API credentials and spend authorisation (the same blocker as "
        "Silver and as the Method A/B agreement kappa)"
    )
    return {
        str(Executor.TEMPLATE): run_template_baseline,
        str(Executor.CLASSIFIER_TEMPLATE): run_template_baseline,
        str(Executor.API_ZERO_SHOT): (
            run_api_baseline if allow_api else _not_implemented(api_blocker)
        ),
        str(Executor.API_FEW_SHOT): (
            run_api_baseline if allow_api else _not_implemented(api_blocker)
        ),
        str(Executor.API_AGENTIC): (
            run_api_baseline if allow_api else _not_implemented(api_blocker)
        ),
        str(Executor.LOCAL_ZERO_SHOT): (
            run_local_zero_shot if allow_gpu else _not_implemented(gpu_blocker)
        ),
        str(Executor.TRAINED_GENERATOR): (
            run_trained_generator if allow_gpu else _not_implemented(gpu_blocker)
        ),
    }


def _resolve_paths(substrate: str) -> dict[str, str]:
    """Resolve the run-invariant paths every executor needs, out of the Hydra config.

    Args:
        substrate: The data substrate whose corpus the matrix scores.

    Returns:
        The context block written beside every run and folded into its config hash.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    # `paths.root` resolves through `${hydra:runtime.cwd}`, which reads the HydraConfig
    # singleton. This script is an argparse entrypoint, not a Hydra job, so that singleton
    # is unset: compose with return_hydra_config=True and install it -- the same shape
    # tests/integration/test_hydra_compose.py uses, for the same reason.
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[f"data={substrate.split('_')[0]}"],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
        paths = OmegaConf.to_container(cfg.paths, resolve=True)
    GlobalHydra.instance().clear()

    processed = Path(str(paths["processed_dir"]))
    return {
        "repo_root": str(REPO_ROOT),
        "substrate": substrate,
        "test_set": "test",
        "bronze_path": str(processed / substrate / "corpus" / "bronze.jsonl"),
        "metrics_dir": str(paths["metrics_dir"]),
        "checkpoints_dir": str(paths["checkpoints_dir"]),
    }


def main() -> int:
    """Plan and optionally execute the matrix.

    Returns:
        0 when every run completed or was skipped, 2 when any run failed or was blocked,
        and 1 when the registry itself is inconsistent. The non-zero code on a failed run
        is deliberate: a matrix that lost three arms overnight must not look to CI like a
        matrix that finished.
    """
    parser = argparse.ArgumentParser(description="Run the Phase 11 experiment matrix.")
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "matrix",
        help="matrix root directory",
    )
    parser.add_argument(
        "--systems", default="", help="comma-separated system ids; all when omitted"
    )
    parser.add_argument("--dry-run", action="store_true", help="plan only, execute nothing")
    parser.add_argument("--force", action="store_true", help="ignore completion markers and re-run")
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help="this machine has a GPU large enough for the generator arms",
    )
    parser.add_argument(
        "--allow-api",
        action="store_true",
        help="teacher-API credentials and spend authorisation exist",
    )
    parser.add_argument(
        "--substrate", default="amlworld_hi_small", help="which substrate's corpus to score"
    )
    parser.add_argument(
        "--encoder-arms",
        default="gatv2,mlp",
        help="Phase 7 encoder checkpoints that already exist, comma-separated",
    )
    args = parser.parse_args()

    configure_logging()
    log = get_logger("matrix")

    problems = validate_registry()
    if problems:
        for problem in problems:
            log.error("registry: %s", problem)
        return 1

    if args.systems:
        try:
            specs = [get_system(s.strip()) for s in args.systems.split(",") if s.strip()]
        except UnknownSystemError as exc:
            log.error("%s", exc)
            return 1
    else:
        specs = list(all_systems())

    summary = matrix_summary(specs)
    log.info(
        "matrix: %d systems, %d runs (%d GPU-serialised)",
        summary["n_systems"],
        summary["n_runs"],
        summary["runs_by_resource"].get("gpu", 0)
        + summary["runs_by_resource"].get("gpu_inference", 0),
    )
    log.info(
        "seed policy: %s at %s; everything else at %s",
        ", ".join(summary["multi_seed_systems"]),
        summary["seeds_central"],
        summary["seeds_single"],
    )

    # Paths come out of `configs/paths/` and never out of a literal here (CLAUDE.md §6).
    # They are folded into every run's config hash, so pointing the matrix at a different
    # corpus invalidates the completion markers rather than silently reusing old numbers.
    context = _resolve_paths(args.substrate)
    log.info("corpus: %s", context["bronze_path"])
    plan = plan_matrix(specs, root=args.root, overrides=context)
    if plan.problems:
        for problem in plan.problems:
            log.error("plan: %s", problem)
        return 1

    log.info("plan:\n%s", json.dumps(plan.summary(), indent=2))
    if args.dry_run:
        write_json(Path(args.root) / "plan.json", plan.summary())
        return 0

    external = {f"encoder:{arm.strip()}" for arm in args.encoder_arms.split(",") if arm.strip()}
    result = run_matrix(
        plan,
        build_executors(allow_gpu=args.allow_gpu, allow_api=args.allow_api),
        external_artifacts=external,
        force=args.force,
        summary_path=Path(args.root) / "matrix_summary.json",
    )

    for status in RunStatus:
        records = result.by_status(status)
        if records:
            log.info("%s: %s", status, ", ".join(r.run_id for r in records))
    for record in result.by_status(RunStatus.FAILED):
        log.error("%s failed: %s", record.run_id, record.error)
    for record in result.by_status(RunStatus.BLOCKED):
        log.error("%s blocked by %s", record.run_id, record.blocked_by)

    log.info(
        "matrix finished in %.1fs: %s",
        result.duration_s,
        "all runs accounted for" if result.ok else "SOME RUNS FAILED OR WERE BLOCKED",
    )
    return 0 if result.ok else 2


if __name__ == "__main__":
    sys.exit(main())
