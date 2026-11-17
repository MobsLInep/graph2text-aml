"""Decode a test split through the inference guard, in three batched passes.

:class:`~g2t_aml.models.generator.guard.InferenceGuard` selects among candidates and calls
back for one constrained regeneration. Calling back into a GPU model one case at a time
would decode the few repairs at batch size one; this module runs the guard's work as three
passes instead, each batched and each resumable from its own file:

1. **candidates** — ``n`` sampled narratives per case (``guard_candidates.jsonl``);
2. **repairs** — for every case whose best candidate still contradicts the record, one
   greedy regeneration from the original prompt followed by the draft and the guard's
   repair instruction (``guard_repairs.jsonl``);
3. **selection** — the guard proper, run over every case with the precomputed repair as
   its regeneration, so its statistics count exactly what a per-case guard would have.

Passes 1 and 3 call the guard's own selection with the same inputs, so the cases that pass
2 repairs are exactly the cases the guard would have asked to repair.

The output row carries the guarded text and, beside it, the first sampled candidate — what
the same sampling would have emitted without verification — and the guard's verdict, so
the guarded row is never read without knowing what the guard did to it (D-073).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from g2t_aml.models.generator.guard import GuardStatistics, GuardWeights, InferenceGuard
from g2t_aml.models.generator.inference import GenerationConfig, run_test_set
from g2t_aml.models.generator.prompts import BuiltPrompt, PromptBuilder, PromptSegment, SegmentRole
from g2t_aml.utils.io import read_jsonl

__all__ = ["GuardDecodeConfig", "guarded_test_set", "repair_prompt"]

CANDIDATES_FILE = "guard_candidates.jsonl"
REPAIRS_FILE = "guard_repairs.jsonl"


@dataclass(frozen=True)
class GuardDecodeConfig:
    """How the guard samples and selects.

    Attributes:
        n_candidates: Sampled narratives per case.
        temperature: Sampling temperature.
        top_p: Nucleus threshold.
        weights: The selection weights.
        allow_regeneration: Whether to attempt the one constrained regeneration.
        max_new_tokens: Generation cap, for candidates and repairs alike.
        batch_size: Cases per batch in the candidate pass. Each case is ``n_candidates``
            rows, so this is a quarter of the greedy batch at the default four.
        seed: Sampling seed.
    """

    n_candidates: int = 4
    temperature: float = 0.6
    top_p: float = 0.9
    weights: GuardWeights = field(default_factory=GuardWeights)
    allow_regeneration: bool = True
    max_new_tokens: int = 512
    batch_size: int = 2
    seed: int = 42


def repair_prompt(
    builder: PromptBuilder, record: Mapping[str, Any], draft: str, instruction: str
) -> BuiltPrompt:
    """Build the constrained-regeneration prompt: the original prompt, the draft, the fix.

    The draft is placed where the model's answer went and the repair instruction follows
    it as a further user turn, so the model sees what it wrote and exactly what was wrong
    with it. The soft-token positions are those of the original prompt, unchanged.

    Args:
        builder: The arm's prompt builder.
        record: The test record.
        draft: The candidate the guard selected and could not clean.
        instruction: :meth:`InferenceGuard.build_repair_prompt`'s text.

    Returns:
        An inference prompt (every label ``-100``).
    """
    base = builder.build(dict(record), for_training=False)
    extra = list(
        builder.tokenizer.encode(
            f"{draft}\n<|user|>\n{instruction}\n<|assistant|>\n", add_special_tokens=False
        )
    )
    return BuiltPrompt(
        input_ids=[*base.input_ids, *extra],
        labels=[*base.labels, *([-100] * len(extra))],
        soft_mask=[*base.soft_mask, *([False] * len(extra))],
        segments=(*base.segments, PromptSegment(SegmentRole.PROMPT, extra)),
        n_facts_truncated=base.n_facts_truncated,
        case_id=base.case_id,
    )


def guarded_test_set(
    generator: Any,
    records: Sequence[Mapping[str, Any]],
    items: Sequence[tuple[BuiltPrompt, Any]],
    *,
    builder: PromptBuilder,
    collator: Any,
    tokenizer: Any,
    config: GuardDecodeConfig,
    case_inputs: Callable[[str], tuple[Any, Any]],
    run_dir: str | Path,
    device: str = "cpu",
    log: Any = None,
) -> tuple[list[dict[str, Any]], GuardStatistics]:
    """Decode every case through the guard.

    Args:
        generator: The trained generator.
        records: The test records, in the same order as ``items``.
        items: ``(prompt, graph)`` pairs built with ``for_training=False``.
        builder: The prompt builder, for the repair prompts.
        collator: The collator.
        tokenizer: The tokeniser.
        config: Sampling and selection settings.
        case_inputs: Case id to ``(facts, extractor)`` - the fact record the guard
            verifies against and the claim extractor bound to that case.
        run_dir: Where the per-pass files are written. An interrupted decode resumes from
            them.
        device: Device to decode on.
        log: Optional logger.

    Returns:
        One output row per case, in input order, and the guard's statistics.

    Raises:
        ValueError: If ``records`` and ``items`` disagree on the cases.
    """
    run_dir = Path(run_dir)
    order = [str(r["case_id"]) for r in records]
    if order != [prompt.case_id for prompt, _ in items]:
        raise ValueError("records and items must list the same cases in the same order")
    graphs = {prompt.case_id: graph for prompt, graph in items}
    by_id = {str(r["case_id"]): r for r in records}

    sampling = GenerationConfig(
        max_new_tokens=config.max_new_tokens,
        do_sample=True,
        temperature=config.temperature,
        top_p=config.top_p,
        num_return_sequences=config.n_candidates,
        seed=config.seed,
    )
    run_test_set(
        generator,
        items,
        collator=collator,
        tokenizer=tokenizer,
        cfg=sampling,
        output_path=run_dir / CANDIDATES_FILE,
        batch_size=config.batch_size,
        device=device,
        log=log,
    )
    candidates = {
        str(r["case_id"]): list(r["texts"]) for r in read_jsonl(run_dir / CANDIDATES_FILE)
    }

    # Which cases the guard will ask to repair: the same selection, with repair disabled.
    probe = InferenceGuard(weights=config.weights, allow_regeneration=False)
    to_repair: dict[str, BuiltPrompt] = {}
    for case_id in order:
        facts, extractor = case_inputs(case_id)
        report = probe.run(case_id, candidates[case_id], facts, extractor)
        if report.warning is not None and config.allow_regeneration:
            to_repair[case_id] = repair_prompt(
                builder,
                by_id[case_id],
                report.selected.text,
                probe.build_repair_prompt(report.selected.violations),
            )
    if log is not None:
        log.info("guard: %d of %d cases need a repair pass", len(to_repair), len(order))

    repairs: dict[str, str] = {}
    if to_repair:
        run_test_set(
            generator,
            [(to_repair[c], graphs[c]) for c in order if c in to_repair],
            collator=collator,
            tokenizer=tokenizer,
            cfg=GenerationConfig.deterministic(max_new_tokens=config.max_new_tokens),
            output_path=run_dir / REPAIRS_FILE,
            batch_size=max(1, config.batch_size * config.n_candidates),
            device=device,
            log=log,
        )
        repairs = {str(r["case_id"]): r["texts"][0] for r in read_jsonl(run_dir / REPAIRS_FILE)}

    guard = InferenceGuard(weights=config.weights, allow_regeneration=config.allow_regeneration)
    rows: list[dict[str, Any]] = []
    for case_id in order:
        facts, extractor = case_inputs(case_id)
        repaired = repairs.get(case_id)
        report = guard.run(
            case_id,
            candidates[case_id],
            facts,
            extractor,
            regenerate=(lambda _violations, text=repaired: text) if repaired is not None else None,
        )
        rows.append(
            {
                "case_id": case_id,
                "texts": [report.text],
                "guard": True,
                "guard_regenerated": report.regenerated,
                "guard_warning": report.warning,
                "guard_selected_score": report.selected.score,
                "unguarded_first_candidate": candidates[case_id][0],
                "n_candidates": len(candidates[case_id]),
            }
        )
    return rows, guard.stats
