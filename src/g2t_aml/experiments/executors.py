"""Phase 11: the executors -- what actually produces each system's narratives.

The runner dispatches on :class:`~g2t_aml.experiments.registry.Executor` and knows nothing
else about how a system works. This module is the other side of that contract: one function
per executor kind, each taking ``(spec, seed, run_directory)`` and returning what belongs
in the completion marker.

Every executor writes ``generations.jsonl`` into its run directory in the shape
:func:`g2t_aml.eval.types.load_system_outputs` reads, so a template baseline, a frontier-API
baseline and a QLoRA arm all score through the same call in Phase 10. That is deliberate:
the alternative is three evaluation paths, and three paths is three places for the headline
metric to be computed slightly differently.

**On this machine, the GPU and API executors will raise.** They are written in full anyway,
against the real Phase 9 and Phase 5 machinery, because the alternative -- stubs -- rots
(see CLAUDE.md §6), and because the point of Phase 11 is that the matrix runs on the first
day compute lands rather than starting a week of integration then. What is untested here is
stated in PHASE_LOG rather than papered over: no executor below the CPU ones has been
executed against a real model or a real endpoint.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from g2t_aml.corpus.factsio import facts_from_dict
from g2t_aml.experiments.registry import Executor, SystemSpec
from g2t_aml.facts.schema import CaseFacts
from g2t_aml.utils.io import read_json, read_jsonl, write_jsonl
from g2t_aml.utils.logging import get_logger

__all__ = [
    "GENERATIONS_FILE",
    "load_bronze_corpus",
    "run_api_baseline",
    "run_local_zero_shot",
    "run_template_baseline",
    "run_trained_generator",
]

log = get_logger(__name__)

#: Every executor writes this, in the shape `eval.types.load_system_outputs` reads.
GENERATIONS_FILE = "generations.jsonl"


class ExecutorError(RuntimeError):
    """Raised when an executor cannot produce outputs.

    Always carries WHY in its message: the runner records it verbatim into the run's
    ``FAILED.json`` and into the matrix summary, and that message is what ends up in
    RESULTS.md as the reason a system did not run.
    """


def _context_from_run_dir(directory: Path) -> Mapping[str, Any]:
    """Read the run-invariant context the runner wrote beside the run.

    Args:
        directory: The run directory.

    Returns:
        The ``context`` block of ``resolved_config.json``, or an empty mapping.
    """
    config_path = Path(directory) / "resolved_config.json"
    if not config_path.is_file():
        return {}
    payload = read_json(config_path)
    if isinstance(payload, dict) and isinstance(payload.get("context"), dict):
        return payload["context"]
    return {}


def load_bronze_corpus(
    path: Path | str, *, limit: int = 0
) -> tuple[dict[str, str], dict[str, CaseFacts], dict[str, str]]:
    """Read the Bronze corpus into narratives, fact records and splits.

    The fact record embedded on a training record is the record the narrative was
    *written from*, which is the one faithfulness is scored against. Re-deriving it from
    the case store would be a second extraction, and a disagreement between the two would
    surface as a hallucination in every narrative rather than as the extractor bug it
    would be -- the same reasoning ``scripts/10_evaluate.py`` uses.

    Args:
        path: The corpus JSONL.
        limit: Stop after this many records, or 0 for all.

    Returns:
        ``(narratives, facts, splits)``, each keyed by case id.

    Raises:
        ExecutorError: If the corpus file is missing.
    """
    corpus = Path(path)
    if not corpus.is_file():
        raise ExecutorError(
            f"Bronze corpus not found at {corpus}; run `make bronze` first (Phase 4)"
        )
    narratives: dict[str, str] = {}
    facts: dict[str, CaseFacts] = {}
    splits: dict[str, str] = {}
    for row in read_jsonl(corpus):
        if not isinstance(row, dict):
            continue
        case_id = str(row["case_id"])
        narratives[case_id] = str(row["target_narrative"])
        facts[case_id] = facts_from_dict(dict(row["facts"]))
        splits[case_id] = str(row.get("split", "unknown"))
        if limit and len(narratives) >= limit:
            break
    return narratives, facts, splits


def _test_case_ids(splits: Mapping[str, str]) -> list[str]:
    """Return the test-split case ids, sorted.

    Args:
        splits: Case id to split name.

    Returns:
        The test cases. Sorted so two runs of a baseline emit the same order and a diff
        between two generation files is a diff in the text.
    """
    return sorted(case_id for case_id, split in splits.items() if split == "test")


def _paths_for(directory: Path) -> tuple[Path, Path]:
    """Return the repository root and the Bronze corpus path for a run.

    **Both come out of the run's own context, never out of a literal in this file.** Every
    directory root lives in ``configs/paths/`` and is reached as ``cfg.paths.*``
    (CLAUDE.md §6); the entrypoint resolves them and the runner writes them beside the
    run, so an executor reading a corpus is reading the corpus its config named.

    Args:
        directory: The run directory.

    Returns:
        ``(repo_root, bronze_path)``.

    Raises:
        ExecutorError: If the run's context does not carry both paths.
    """
    context = _context_from_run_dir(directory)
    root = context.get("repo_root")
    bronze = context.get("bronze_path")
    missing = [name for name, value in (("repo_root", root), ("bronze_path", bronze)) if not value]
    if missing:
        raise ExecutorError(
            f"run at {directory} has no {missing} in its context; the entrypoint resolves "
            "these from cfg.paths and the runner writes them into resolved_config.json. "
            "Executors never guess a path."
        )
    return Path(str(root)), Path(str(bronze))


# ------------------------------------------------------------------- the CPU arms ---


def run_template_baseline(spec: SystemSpec, seed: int, directory: Path) -> dict[str, Any]:
    """Produce B1's or B2's outputs.

    B1 is the committed Bronze corpus, read rather than re-rendered: Phase 7 has since
    populated ``model_signal``, so regenerating Bronze would push the encoder's own score
    into the serialisation baseline and nothing would fail (D-063). B2 appends the Phase 7
    classifier's score and its binary call to the same text, which is what a deployed
    classify-then-template system emits.

    Args:
        spec: The system.
        seed: The seed. Recorded on every row; neither arm is stochastic, so it does not
            change the output, and that invariance is asserted by the matrix tests.
        directory: The run directory.

    Returns:
        Counts and the generations path, for the completion marker.

    Raises:
        ExecutorError: If the corpus or the encoder's scores are missing.
    """
    from g2t_aml.experiments.baselines import (
        render_classifier_template_baseline,
        render_template_baseline,
    )

    repo_root, bronze_path = _paths_for(directory)
    narratives, _facts, splits = load_bronze_corpus(bronze_path)
    case_ids = _test_case_ids(splits)
    if not case_ids:
        raise ExecutorError(f"no test-split cases found in {bronze_path}")

    if spec.executor is Executor.CLASSIFIER_TEMPLATE:
        context = _context_from_run_dir(directory)
        metrics_dir = context.get("metrics_dir")
        if not metrics_dir:
            raise ExecutorError(
                f"{spec.system_id} needs metrics_dir in its run context; it is resolved "
                "from cfg.paths.metrics_dir by the entrypoint"
            )
        scores_path = Path(str(metrics_dir)) / "encoder" / f"scores_{spec.encoder_arm}.jsonl"
        if not scores_path.is_file():
            raise ExecutorError(
                f"{spec.system_id} needs the Phase 7 scores at {scores_path}; run "
                "`make score-cases` first"
            )
        predictions = {
            str(row["case_id"]): float(row["score"])
            for row in read_jsonl(scores_path)
            if isinstance(row, dict) and "case_id" in row and "score" in row
        }
        rows = render_classifier_template_baseline(
            case_ids, narratives, predictions, system=spec.system_id
        )
    else:
        rows = render_template_baseline(case_ids, narratives, system=spec.system_id)

    for row in rows:
        row["seed"] = seed
        row["stream"] = "balanced"
    out = Path(directory) / GENERATIONS_FILE
    write_jsonl(out, rows)
    return {
        "generations": str(out),
        "n_cases": len(rows),
        "executor": str(spec.executor),
        "deterministic": True,
    }


# ------------------------------------------------------------------- the API arms ---


def run_api_baseline(spec: SystemSpec, seed: int, directory: Path) -> dict[str, Any]:
    """Produce B3, B4 or B5's outputs through the Phase 5 teacher client.

    Args:
        spec: The system.
        seed: The seed. Recorded; the frontier models used here reject sampling
            parameters outright (D-045), so a seed does not change the output and the run
            is single-seed by policy anyway.
        directory: The run directory.

    Returns:
        Counts, cost, and for B5 the agentic-trace totals, for the completion marker.

    Raises:
        ExecutorError: If credentials are absent or a prompt fails to load.
    """
    import os

    from g2t_aml.corpus.silver.api_client import APITeacher, TeacherSpec
    from g2t_aml.experiments.baselines import (
        DEFAULT_K_SHOT,
        Exemplar,
        assert_prompts_loadable,
        generate_agentic,
        generate_few_shot,
        generate_zero_shot,
    )
    from g2t_aml.facts.serialiser import serialise_facts
    from g2t_aml.facts.vocab import load_vocabulary

    # Fail before spending a dollar, not after four hundred cases.
    prompt_hashes = assert_prompts_loadable()

    api_key_env = "ANTHROPIC_API_KEY"
    if not os.environ.get(api_key_env):
        raise ExecutorError(
            f"{spec.system_id} needs {api_key_env}; this is the same blocker as Silver "
            "and as the Method A/B agreement kappa"
        )

    repo_root, bronze_path = _paths_for(directory)
    narratives, facts, splits = load_bronze_corpus(bronze_path)
    case_ids = _test_case_ids(splits)
    vocabulary = load_vocabulary()

    teacher_spec = TeacherSpec(
        key="baseline",
        family="frontier",
        provider="anthropic",
        model=str(spec.base_model_version or spec.base_model),
        max_output_tokens=4096,
        supports_sampling=False,
        api_key_env=api_key_env,
    )
    teacher = APITeacher(teacher_spec)

    pool: tuple[Exemplar, ...] = ()
    if spec.executor in (Executor.API_FEW_SHOT, Executor.API_AGENTIC):
        # TRAIN SPLIT ONLY. `select_exemplars` raises on anything else; building the pool
        # from the train split here means that check should never fire, which is exactly
        # the relationship a leak guard should have with its caller.
        pool = tuple(
            Exemplar(
                case_id=case_id,
                split="train",
                typology=facts[case_id].typology.label,
                serialised_facts=serialise_facts(facts[case_id], style="verbose"),
                narrative=narratives[case_id],
            )
            for case_id, split in sorted(splits.items())
            if split == "train"
        )[:200]

    rows: list[dict[str, Any]] = []
    failures = 0
    total_calls = 0
    for case_id in case_ids:
        record = facts[case_id]
        try:
            if spec.executor is Executor.API_AGENTIC:
                output = generate_agentic(
                    record,
                    teacher,
                    pool=pool,
                    system=spec.system_id,
                    k=int(spec.extra.get("k_shot", DEFAULT_K_SHOT)),
                    max_rounds=int(spec.extra.get("max_repair_rounds", 3)),
                    vocabulary=vocabulary,
                )
            elif spec.executor is Executor.API_FEW_SHOT:
                output = generate_few_shot(
                    record,
                    teacher,
                    pool,
                    system=spec.system_id,
                    k=int(spec.extra.get("k_shot", DEFAULT_K_SHOT)),
                    vocabulary=vocabulary,
                )
            else:
                output = generate_zero_shot(
                    record, teacher, system=spec.system_id, vocabulary=vocabulary
                )
        except Exception:
            failures += 1
            log.exception("%s failed on case %s", spec.system_id, case_id)
            continue
        row = output.to_dict()
        row["seed"] = seed
        row["stream"] = "balanced"
        rows.append(row)
        if output.trace is not None:
            total_calls += output.trace.n_calls
        else:
            total_calls += 1

    out = Path(directory) / GENERATIONS_FILE
    write_jsonl(out, rows)
    return {
        "generations": str(out),
        "n_cases": len(rows),
        "n_case_failures": failures,
        "n_model_calls": total_calls,
        "calls_per_narrative": (total_calls / len(rows)) if rows else None,
        "model": teacher_spec.model,
        "model_release_date": spec.base_model_release_date,
        "prompt_hashes": prompt_hashes,
        "executor": str(spec.executor),
    }


# ------------------------------------------------------------------- the GPU arms ---


def run_local_zero_shot(spec: SystemSpec, seed: int, directory: Path) -> dict[str, Any]:
    """Produce B6's outputs: the untuned local base model, one call per case.

    B6 uses the **same prompt file** B3, B4 and B5 use, so the B6-to-B7 delta is what
    QLoRA bought and the B6-to-B3 delta is what the frontier model's capability bought.
    Two comparisons from one arm, and neither is confounded by a prompt difference.

    Args:
        spec: The system.
        seed: The seed.
        directory: The run directory.

    Returns:
        Counts and the generations path.

    Raises:
        ExecutorError: If the LLM extra is not installed or the model cannot be loaded.
    """
    from g2t_aml.experiments.baselines import assert_prompts_loadable

    prompt_hashes = assert_prompts_loadable()
    repo_root, _bronze_path = _paths_for(directory)
    result = _invoke_entrypoint(
        repo_root,
        "09_train_generator.py",
        overrides=[
            f"experiment={spec.experiment_config}",
            f"seed={seed}",
            "generator.use_fusion=false",
            "training.epochs=0",  # inference only: B6 is the UNTUNED floor
            f"hydra.run.dir={Path(directory).as_posix()}",
        ],
        system_id=spec.system_id,
    )
    out = Path(directory) / GENERATIONS_FILE
    if not out.is_file():
        raise ExecutorError(
            f"{spec.system_id} produced no {GENERATIONS_FILE} in {directory}; the "
            f"entrypoint exited {result}"
        )
    return {
        "generations": str(out),
        "n_cases": sum(1 for _ in read_jsonl(out)),
        "model": spec.base_model,
        "model_release_date": spec.base_model_release_date,
        "prompt_hashes": prompt_hashes,
        "executor": str(spec.executor),
    }


def _invoke_entrypoint(
    repo_root: Path,
    script: str,
    *,
    overrides: Sequence[str],
    system_id: str,
) -> int:
    """Run a pipeline entrypoint in a subprocess and propagate its exit code.

    **The GPU arms go through the existing Phase 9 entrypoint rather than reassembling the
    model here.** ``scripts/09_train_generator.py`` is where the backbone, the encoder
    checkpoint, the fusion layer, the curriculum and the overfit gate are wired together,
    and a second assembly path in this module would be a second place for the three
    learning rates, the loss mask or the fp32 projector assertion to drift out of
    agreement with the first. One path, invoked with the arm's own Hydra config.

    A subprocess rather than an in-process call because ``@hydra.main`` can be initialised
    once per process, and the matrix runs a dozen arms.

    Args:
        repo_root: The repository root.
        script: The entrypoint's filename under ``scripts/``.
        overrides: Hydra overrides.
        system_id: The system, for the error message.

    Returns:
        The entrypoint's exit code.

    Raises:
        ExecutorError: If the entrypoint exited non-zero.
    """
    import subprocess

    command = [sys.executable, str(repo_root / "scripts" / script), *overrides]
    log.info("%s -> %s", system_id, " ".join(command))
    completed = subprocess.run(command, cwd=repo_root, check=False)
    if completed.returncode != 0:
        raise ExecutorError(
            f"{system_id}: {script} exited {completed.returncode}; its own log in the run "
            "directory carries the reason"
        )
    return completed.returncode


def run_trained_generator(spec: SystemSpec, seed: int, directory: Path) -> dict[str, Any]:
    """Train (or reuse) a Phase 9 arm and decode the test set.

    A5 is the one arm here that trains nothing: it is S1's checkpoint decoded with the
    guard disabled, which is why its spec sets ``trained=False`` and depends on S1. Every
    other arm trains, and the trainer refuses to start until the 20-example/100-step
    overfit check passes -- the bugs that check catches (a misaligned loss mask, a
    detached fusion path, a splice on the wrong positions) are all invisible in the loss
    curve of a full run.

    Args:
        spec: The system.
        seed: The seed.
        directory: The run directory.

    Returns:
        The checkpoint path, the generations path, and the profile.

    Raises:
        ExecutorError: If the LLM extra is absent, the encoder checkpoint is missing, or
            training fails.
    """
    repo_root, _bronze_path = _paths_for(directory)
    overrides = [
        f"experiment={spec.experiment_config}",
        f"seed={seed}",
        f"training.guard.enabled={str(spec.guard).lower()}",
        f"hydra.run.dir={Path(directory).as_posix()}",
    ]

    if not spec.trained:
        # A5 reads S1's checkpoint. The dependency is declared in the registry and the
        # runner refuses to schedule this arm until S1 has succeeded at every seed, so a
        # missing checkpoint here means the matrix root moved, not that S1 failed.
        source = next(iter(spec.depends_on), None)
        checkpoint = _locate_checkpoint(Path(directory).parents[2], source, seed)
        if checkpoint is None:
            raise ExecutorError(
                f"{spec.system_id} reuses {source}'s checkpoint at seed {seed} and it "
                "could not be located under the matrix root"
            )
        overrides += [f"generator.resume_from={checkpoint.as_posix()}", "training.epochs=0"]

    _invoke_entrypoint(
        repo_root, "09_train_generator.py", overrides=overrides, system_id=spec.system_id
    )

    generations = Path(directory) / GENERATIONS_FILE
    if not generations.is_file():
        raise ExecutorError(
            f"{spec.system_id} trained but wrote no {GENERATIONS_FILE} in {directory}"
        )
    checkpoint_dir = Path(directory) / "checkpoint"
    return {
        "checkpoint": str(checkpoint_dir) if checkpoint_dir.exists() else None,
        "generations": str(generations),
        "n_cases": sum(1 for _ in read_jsonl(generations)),
        "guard": spec.guard,
        "trained": spec.trained,
        "model": spec.base_model,
        "executor": str(spec.executor),
    }


def _locate_checkpoint(matrix_root: Path, system_id: str | None, seed: int) -> Path | None:
    """Find a completed run's checkpoint under the matrix root.

    Args:
        matrix_root: The matrix root directory.
        system_id: The system whose checkpoint is wanted.
        seed: The seed.

    Returns:
        The checkpoint directory, or None when there is no completed run for that
        (system, seed).
    """
    if not system_id:
        return None
    seed_dir = matrix_root / system_id / f"seed{seed}"
    if not seed_dir.is_dir():
        return None
    completed = [d for d in sorted(seed_dir.iterdir()) if (d / "COMPLETED.json").is_file()]
    for candidate in reversed(completed):
        checkpoint = candidate / "checkpoint"
        if checkpoint.is_dir():
            return checkpoint
    return None


def executor_table() -> dict[str, Any]:
    """Return every executor keyed by its registry name.

    Returns:
        The mapping the runner consumes. Kept here rather than in the script so the
        wiring is importable by a test.
    """
    return {
        str(Executor.TEMPLATE): run_template_baseline,
        str(Executor.CLASSIFIER_TEMPLATE): run_template_baseline,
        str(Executor.API_ZERO_SHOT): run_api_baseline,
        str(Executor.API_FEW_SHOT): run_api_baseline,
        str(Executor.API_AGENTIC): run_api_baseline,
        str(Executor.LOCAL_ZERO_SHOT): run_local_zero_shot,
        str(Executor.TRAINED_GENERATOR): run_trained_generator,
    }


def coverage_of_registry(specs: Sequence[SystemSpec]) -> list[str]:
    """Report any system whose executor has no implementation.

    Args:
        specs: The systems to check.

    Returns:
        One line per uncovered system. Empty when every declared system can be dispatched
        -- which is asserted by a test, because a system with no executor silently
        disappears from the results table.
    """
    table = executor_table()
    return [
        f"{spec.system_id} declares executor {spec.executor} with no implementation"
        for spec in specs
        if str(spec.executor) not in table
    ]
