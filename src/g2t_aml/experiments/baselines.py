"""Phase 11: the baseline systems B1-B6, built as competitors rather than as foils.

**Deliberately weakening a baseline is research misconduct, and reviewers detect it.** The
guarantees this module is written to hold, each enforced by something other than good
intentions:

- B3, B4 and B5 receive the **same instructions our own arms receive** -- the SAR
  structure, the availability mask, the salience list, the controlled vocabulary's hedging
  and forbidden blocks. ``prompts/baseline_generate_v1.txt`` says so at the top and
  :func:`assert_baseline_not_starved` checks the blocks are actually non-empty at call
  time, because a prompt that renders an empty forbidden list is a strawman produced by
  accident rather than by intent.
- **B5 is a genuine agentic system.** Generate, self-verify against the record, repair,
  up to three rounds. Its verification is *its own model's*, not our Phase 3 checker:
  handing it our instrument would build a competitor that does not exist and cannot be
  cited, and would also hand the baseline the exact tool the evaluation scores it with.
  It is allowed more inference compute than any of our arms, and
  :class:`AgenticTrace` records exactly how much so the asymmetry is reported.
- **B4's exemplars come from the TRAIN split only**, selected deterministically from the
  case id, and :func:`select_exemplars` raises if a candidate is not a train case. A
  few-shot baseline given a test-split exemplar is a leak, and it would flatter the
  baseline rather than weaken it -- which is why the check is here and not left to review.

Everything is driven through the Phase 5 :class:`~g2t_aml.corpus.silver.api_client.Teacher`
protocol, so ``ScriptedTeacher`` exercises the entire agentic loop -- generation, parse,
repair, round limits, parse failures -- with no network and no credentials, exactly as the
Silver pipeline does.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from g2t_aml.corpus.silver.api_client import Teacher, TeacherError
from g2t_aml.corpus.silver.prompts import (
    PromptRenderError,
    RenderedPrompt,
    _availability_blocks,
    _forbidden_block,
    _hedging_block,
    _inferred_hedge_block,
    _salient_block,
    load_prompt,
)
from g2t_aml.facts.schema import CaseFacts
from g2t_aml.facts.serialiser import serialise_facts
from g2t_aml.facts.vocab import ControlledVocabulary, load_vocabulary
from g2t_aml.utils.logging import get_logger

__all__ = [
    "DEFAULT_K_SHOT",
    "GENERATE_PROMPT_NAME",
    "MAX_REPAIR_ROUNDS",
    "REPAIR_PROMPT_NAME",
    "VERIFY_PROMPT_NAME",
    "AgenticTrace",
    "BaselineOutput",
    "Exemplar",
    "SelfReportedViolation",
    "assert_baseline_not_starved",
    "generate_agentic",
    "generate_few_shot",
    "generate_zero_shot",
    "parse_verification",
    "select_exemplars",
]

log = get_logger(__name__)

GENERATE_PROMPT_NAME = "baseline_generate_v1"
VERIFY_PROMPT_NAME = "baseline_verify_v1"
REPAIR_PROMPT_NAME = "baseline_repair_v1"

#: B4's exemplar count, from the brief.
DEFAULT_K_SHOT = 5

#: B5's repair budget. Three, not the two Silver allows, because B5 is a baseline being
#: given its best shot rather than a corpus builder being kept honest -- and because the
#: published agentic approach it stands in for iterates until clean.
MAX_REPAIR_ROUNDS = 3

_MIN_WORDS = 130
_MAX_WORDS = 260

_VIOLATION_TYPES = frozenset(
    {"NUMBER", "ENTITY", "UNAVAILABLE", "REGULATION", "INFERENCE", "OMISSION"}
)
_VERDICT_CLEAN = re.compile(r"^\s*VERDICT:\s*CLEAN\s*$", re.IGNORECASE | re.MULTILINE)
_VERDICT_VIOLATIONS = re.compile(r"^\s*VERDICT:\s*VIOLATIONS\s*$", re.IGNORECASE | re.MULTILINE)
_VIOLATION_LINE = re.compile(r"^\s*-\s*([A-Z]+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*$", re.MULTILINE)


class BaselineError(RuntimeError):
    """Raised when a baseline cannot produce an output at all."""


@dataclass(frozen=True)
class Exemplar:
    """One in-context example for the few-shot baseline.

    Attributes:
        case_id: Which case it came from. Recorded so a leak is traceable.
        split: The split it came from. Asserted to be ``train``.
        typology: Its typology, which is what selection matches on.
        serialised_facts: The record, serialised the same way the target case is.
        narrative: The reference narrative.
    """

    case_id: str
    split: str
    typology: str
    serialised_facts: str
    narrative: str


@dataclass(frozen=True)
class SelfReportedViolation:
    """One finding from B5's own verification pass.

    **Self-reported, and named that way throughout.** These are the baseline model's
    judgements about its own draft, not verdicts from the Phase 3 checker, and the two must
    never be conflated in a table: B5's self-reported clean rate is a property of B5's
    self-assessment, and its measured Zero-Hallucination Rate comes from the same
    instrument every other system is scored with.

    Attributes:
        kind: One of NUMBER, ENTITY, UNAVAILABLE, REGULATION, INFERENCE, OMISSION.
        span: The quoted span from the draft.
        correction: What the model said the record supports instead.
    """

    kind: str
    span: str
    correction: str

    def to_dict(self) -> dict[str, str]:
        """Return the violation as a JSON-serialisable mapping.

        Returns:
            Every field.
        """
        return {"kind": self.kind, "span": self.span, "correction": self.correction}


@dataclass(frozen=True)
class AgenticTrace:
    """What B5 actually did, recorded so its compute advantage is reportable.

    Attributes:
        rounds: How many verify-repair rounds ran.
        n_calls: Total model calls, generation included. **This is the number that makes
            the comparison honest**: B5 spends several calls per narrative where S1 spends
            one forward pass, and a faithfulness table that does not say so is comparing
            two things at different budgets without mentioning it.
        violations_per_round: Self-reported findings at each round.
        converged: Whether the final verification came back clean.
        parse_failures: Verification responses that did not match the required format.
            Recorded, never retried into a better-looking result: a baseline whose
            verifier fails to parse is a baseline with a weaker verification step, and
            hiding that would overstate it.
    """

    rounds: int
    n_calls: int
    violations_per_round: tuple[tuple[SelfReportedViolation, ...], ...] = ()
    converged: bool = False
    parse_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the trace as a JSON-serialisable mapping.

        Returns:
            Every field, with the violations expanded.
        """
        return {
            "rounds": self.rounds,
            "n_calls": self.n_calls,
            "converged": self.converged,
            "parse_failures": self.parse_failures,
            "violations_per_round": [
                [v.to_dict() for v in round_] for round_ in self.violations_per_round
            ],
            "n_self_reported_violations": sum(len(r) for r in self.violations_per_round),
        }


@dataclass(frozen=True)
class BaselineOutput:
    """One baseline's narrative for one case, with its provenance.

    Attributes:
        case_id: The case.
        system: The system id, e.g. ``B5``.
        narrative: The generated text.
        model: The exact model version that produced it.
        prompt_name: Which prompt file.
        prompt_hash: Its content hash.
        rendered_hash: Hash of the exact text sent.
        n_exemplars: How many in-context examples, for the few-shot arm.
        exemplar_case_ids: Which cases they came from, so a leak is traceable after the
            fact rather than only preventable before it.
        trace: The agentic trace, for B5.
        usage: Token counts and cost, when the teacher reports them.
    """

    case_id: str
    system: str
    narrative: str
    model: str
    prompt_name: str
    prompt_hash: str
    rendered_hash: str
    n_exemplars: int = 0
    exemplar_case_ids: tuple[str, ...] = ()
    trace: AgenticTrace | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the output as a JSON-serialisable mapping.

        Returns:
            Every field. The shape matches what
            :func:`g2t_aml.eval.types.load_system_outputs` reads, so a baseline's
            generations file scores through the same call as a Phase 9 arm's.
        """
        return {
            "case_id": self.case_id,
            "system": self.system,
            "narrative": self.narrative,
            "model": self.model,
            "prompt_name": self.prompt_name,
            "prompt_hash": self.prompt_hash,
            "rendered_hash": self.rendered_hash,
            "n_exemplars": self.n_exemplars,
            "exemplar_case_ids": list(self.exemplar_case_ids),
            "agentic_trace": self.trace.to_dict() if self.trace is not None else None,
            "usage": dict(self.usage),
        }


# --------------------------------------------------------------- prompt assembly ---


def assert_baseline_not_starved(values: Mapping[str, str]) -> None:
    """Refuse to send a baseline prompt that is missing its guidance blocks.

    A baseline whose forbidden-phrase list renders empty, or whose salience list renders
    empty, is a baseline being asked to do the task with less information than our own
    systems get. That is the failure mode this function exists to make impossible: it is
    far more likely to arrive as a silent rendering bug than as a decision, and a silent
    rendering bug that flatters the primary system is indistinguishable from misconduct in
    the results table.

    Args:
        values: The placeholder values about to be rendered.

    Raises:
        BaselineError: If any block that must carry content is empty or whitespace.
    """
    required_non_empty = (
        "fact_record",
        "hedging_block",
        "forbidden_block",
        "salient_block",
        "unavailable_block",
        "available_block",
    )
    starved = [
        name for name in required_non_empty if name in values and not str(values[name]).strip()
    ]
    if starved:
        raise BaselineError(
            f"baseline prompt would render with empty {starved}; a baseline given less "
            "guidance than the primary system is not a baseline"
        )


def _exemplar_block(exemplars: Sequence[Exemplar]) -> str:
    """Render the in-context examples.

    Args:
        exemplars: The examples, in the order they are presented.

    Returns:
        The block, or an empty string for the zero-shot arm. Rendered above the target
        case so the model reads the pattern before the problem.
    """
    if not exemplars:
        return ""
    parts = [
        "WORKED EXAMPLES. Each is a fact record and the narrative an investigator wrote "
        "from it. Follow their structure, register and level of hedging.\n"
    ]
    for index, exemplar in enumerate(exemplars, start=1):
        parts.append(
            f"--- EXAMPLE {index} ({exemplar.typology}) ---\n"
            f"FACT RECORD\n\n{exemplar.serialised_facts}\n\n"
            f"NARRATIVE\n\n{exemplar.narrative}\n"
        )
    parts.append("--- END OF EXAMPLES. The case to write about follows. ---\n")
    return "\n".join(parts)


def _generation_values(
    facts: CaseFacts,
    vocabulary: ControlledVocabulary,
    exemplars: Sequence[Exemplar],
    *,
    min_words: int,
    max_words: int,
) -> dict[str, str]:
    """Assemble every placeholder value the generation prompt needs.

    Args:
        facts: The fact record.
        vocabulary: The controlled vocabulary.
        exemplars: In-context examples, empty for zero-shot.
        min_words: Lower end of the length band.
        max_words: Upper end.

    Returns:
        The values, checked by :func:`assert_baseline_not_starved`.

    Raises:
        BaselineError: If a guidance block would render empty.
    """
    unavailable, available = _availability_blocks(facts)
    values = {
        "fact_record": serialise_facts(facts, style="verbose"),
        "unavailable_block": unavailable,
        "available_block": available,
        "typology_label": facts.typology.label,
        "salient_block": _salient_block(facts, vocabulary),
        "hedging_block": _hedging_block(vocabulary),
        "forbidden_block": _forbidden_block(vocabulary),
        "inferred_hedge_block": _inferred_hedge_block(facts, vocabulary),
        "exemplar_block": _exemplar_block(exemplars),
        "min_words": str(min_words),
        "max_words": str(max_words),
    }
    assert_baseline_not_starved(values)
    return values


def select_exemplars(
    case_id: str,
    pool: Sequence[Exemplar],
    *,
    k: int = DEFAULT_K_SHOT,
) -> tuple[Exemplar, ...]:
    """Choose k in-context examples for one case, deterministically.

    Selection is by typology match first, then by a rotation keyed on the case id, so two
    runs of B4 present the same exemplars in the same order and a difference between runs
    is a difference in the model rather than in the draw.

    Args:
        case_id: The target case. Never itself eligible.
        pool: Candidate exemplars. Must all be train-split cases.
        k: How many to select.

    Returns:
        Up to ``k`` exemplars, typology matches first.

    Raises:
        BaselineError: If any candidate is not from the train split. **A test-split
            exemplar is a leak that would flatter this baseline**, so the check runs here
            rather than being left to review.
    """
    leaked = sorted({e.case_id for e in pool if e.split != "train"})
    if leaked:
        raise BaselineError(
            f"few-shot exemplar pool contains non-train cases {leaked}; that is a leak "
            "into a baseline, and it would overstate the baseline rather than weaken it"
        )
    candidates = [e for e in pool if e.case_id != case_id]
    if not candidates:
        return ()

    target_typology = next((e.typology for e in pool if e.case_id == case_id), None)
    matching = sorted(
        (e for e in candidates if e.typology == target_typology), key=lambda e: e.case_id
    )
    others = sorted(
        (e for e in candidates if e.typology != target_typology), key=lambda e: e.case_id
    )

    # A stable rotation rather than a random sample: reproducible without carrying a seed
    # through every call site, and independent of the pool's file order. Rotated WITHIN
    # each group, never across the two -- a rotation over the concatenation can carry the
    # window past the matching typology entirely, which silently turns typology-matched
    # few-shot into arbitrary few-shot on some case ids and not others.
    offset = sum(ord(c) for c in case_id)

    def _rotate(group: list[Exemplar]) -> list[Exemplar]:
        if not group:
            return []
        pivot = offset % len(group)
        return group[pivot:] + group[:pivot]

    ordered = _rotate(matching) + _rotate(others)
    return tuple(ordered[:k])


# ------------------------------------------------------------------- generation ---


def _complete(
    teacher: Teacher, prompt: RenderedPrompt, *, case_id: str, kind: str, attempt: int
) -> tuple[str, dict[str, Any]]:
    """Call a teacher and return its text with whatever usage it reported.

    Args:
        teacher: The model client.
        prompt: The rendered prompt.
        case_id: The case, for cache keying and error attribution.
        kind: What this call is, for the cache key and the error log.
        attempt: Repair round index.

    Returns:
        ``(text, usage)``.

    Raises:
        BaselineError: If the teacher failed.
    """
    try:
        response = teacher.complete(prompt, case_id=case_id, kind=kind, attempt=attempt)
    except TeacherError as exc:
        raise BaselineError(f"{kind} call failed for {case_id}: {exc!r}") from exc
    usage = {
        key: getattr(response, key, None)
        for key in ("input_tokens", "output_tokens", "cost_usd", "from_cache", "model")
        if hasattr(response, key)
    }
    return response.text, usage


def generate_zero_shot(
    facts: CaseFacts,
    teacher: Teacher,
    *,
    system: str = "B3",
    vocabulary: ControlledVocabulary | None = None,
    min_words: int = _MIN_WORDS,
    max_words: int = _MAX_WORDS,
    prompts_dir: str | None = None,
) -> BaselineOutput:
    """Generate one narrative with a single model call and no exemplars.

    Args:
        facts: The fact record. The model's only source of case information.
        teacher: The model client.
        system: The system id to record.
        vocabulary: The controlled vocabulary; loaded when omitted.
        min_words: Lower end of the length band.
        max_words: Upper end.
        prompts_dir: Override for the prompt directory. Tests only.

    Returns:
        The narrative with its provenance.

    Raises:
        BaselineError: If a guidance block is empty or the model call fails.
        PromptRenderError: If the prompt file and this function disagree on placeholders.
    """
    vocab = vocabulary if vocabulary is not None else load_vocabulary()
    template = load_prompt(GENERATE_PROMPT_NAME, prompts_dir)
    values = _generation_values(facts, vocab, (), min_words=min_words, max_words=max_words)
    prompt = template.render(values)
    text, usage = _complete(teacher, prompt, case_id=facts.case_id, kind="baseline", attempt=0)
    return BaselineOutput(
        case_id=facts.case_id,
        system=system,
        narrative=text.strip(),
        model=teacher.spec.model,
        prompt_name=template.name,
        prompt_hash=template.content_hash,
        rendered_hash=prompt.rendered_hash,
        usage=usage,
    )


def generate_few_shot(
    facts: CaseFacts,
    teacher: Teacher,
    pool: Sequence[Exemplar],
    *,
    system: str = "B4",
    k: int = DEFAULT_K_SHOT,
    vocabulary: ControlledVocabulary | None = None,
    min_words: int = _MIN_WORDS,
    max_words: int = _MAX_WORDS,
    prompts_dir: str | None = None,
) -> BaselineOutput:
    """Generate one narrative with k in-context exemplars.

    Args:
        facts: The fact record.
        teacher: The model client.
        pool: Train-split exemplars to select from.
        system: The system id to record.
        k: How many exemplars.
        vocabulary: The controlled vocabulary; loaded when omitted.
        min_words: Lower end of the length band.
        max_words: Upper end.
        prompts_dir: Override for the prompt directory. Tests only.

    Returns:
        The narrative with its provenance, including which exemplars were used.

    Raises:
        BaselineError: If the pool leaks a non-train case, a guidance block is empty, or
            the model call fails.
    """
    vocab = vocabulary if vocabulary is not None else load_vocabulary()
    exemplars = select_exemplars(facts.case_id, pool, k=k)
    template = load_prompt(GENERATE_PROMPT_NAME, prompts_dir)
    values = _generation_values(facts, vocab, exemplars, min_words=min_words, max_words=max_words)
    prompt = template.render(values)
    text, usage = _complete(teacher, prompt, case_id=facts.case_id, kind="baseline", attempt=0)
    return BaselineOutput(
        case_id=facts.case_id,
        system=system,
        narrative=text.strip(),
        model=teacher.spec.model,
        prompt_name=template.name,
        prompt_hash=template.content_hash,
        rendered_hash=prompt.rendered_hash,
        n_exemplars=len(exemplars),
        exemplar_case_ids=tuple(e.case_id for e in exemplars),
        usage=usage,
    )


def parse_verification(text: str) -> tuple[tuple[SelfReportedViolation, ...], bool]:
    """Parse a self-verification response.

    Args:
        text: The model's response.

    Returns:
        ``(violations, parsed)``. ``parsed`` is False when the response matched neither
        required form; the caller records that as a parse failure and stops the loop
        rather than retrying, because a retry until the format is right is a loop that
        selects for whichever answer happens to parse.
    """
    if _VERDICT_CLEAN.search(text):
        return (), True
    if not _VERDICT_VIOLATIONS.search(text):
        return (), False
    violations = tuple(
        SelfReportedViolation(kind=kind.upper(), span=span, correction=correction)
        for kind, span, correction in _VIOLATION_LINE.findall(text)
        if kind.upper() in _VIOLATION_TYPES
    )
    # A VIOLATIONS verdict with no parseable lines is a malformed response, not a clean
    # draft. Treating it as clean would silently convert every formatting failure into a
    # pass and inflate the baseline's self-reported convergence.
    return violations, bool(violations)


def _violation_block(violations: Sequence[SelfReportedViolation]) -> str:
    """Render findings for the repair prompt.

    Args:
        violations: The self-reported findings.

    Returns:
        A bullet list.
    """
    return "\n".join(
        f'- [{v.kind}] "{v.span}" — the record supports: {v.correction}' for v in violations
    )


def generate_agentic(
    facts: CaseFacts,
    teacher: Teacher,
    *,
    pool: Sequence[Exemplar] = (),
    system: str = "B5",
    k: int = DEFAULT_K_SHOT,
    max_rounds: int = MAX_REPAIR_ROUNDS,
    vocabulary: ControlledVocabulary | None = None,
    min_words: int = _MIN_WORDS,
    max_words: int = _MAX_WORDS,
    prompts_dir: str | None = None,
) -> BaselineOutput:
    """Generate, self-verify and repair -- the agentic competitor, run properly.

    The loop is the published agentic-SAR shape: draft, audit the draft against the record
    with the same model, repair what the audit found, re-audit, up to ``max_rounds``. It
    starts from the **few-shot** draft rather than the zero-shot one, because a competitor
    is entitled to its best configuration and there is no version of this method that
    deliberately begins from a weaker draft.

    The verification is the baseline's own. Our Phase 3 checker never runs inside this
    loop -- see the header of ``prompts/baseline_verify_v1.txt`` for why that is a
    correctness requirement and not a courtesy.

    Args:
        facts: The fact record.
        teacher: The model client.
        pool: Train-split exemplars; B5 uses the same ones B4 does.
        system: The system id to record.
        k: How many exemplars.
        max_rounds: Verify-repair rounds allowed.
        vocabulary: The controlled vocabulary; loaded when omitted.
        min_words: Lower end of the length band.
        max_words: Upper end.
        prompts_dir: Override for the prompt directory. Tests only.

    Returns:
        The final narrative with a full :class:`AgenticTrace`. **The narrative returned is
        the last one produced, converged or not** -- returning an earlier draft because it
        scored better under our checker would be selecting the baseline's output with our
        instrument, which is exactly the advantage B5 is not given.

    Raises:
        BaselineError: If the initial generation fails.
    """
    vocab = vocabulary if vocabulary is not None else load_vocabulary()
    initial = (
        generate_few_shot(
            facts,
            teacher,
            pool,
            system=system,
            k=k,
            vocabulary=vocab,
            min_words=min_words,
            max_words=max_words,
            prompts_dir=prompts_dir,
        )
        if pool
        else generate_zero_shot(
            facts,
            teacher,
            system=system,
            vocabulary=vocab,
            min_words=min_words,
            max_words=max_words,
            prompts_dir=prompts_dir,
        )
    )

    verify_template = load_prompt(VERIFY_PROMPT_NAME, prompts_dir)
    repair_template = load_prompt(REPAIR_PROMPT_NAME, prompts_dir)
    unavailable, _available = _availability_blocks(facts)
    fact_record = serialise_facts(facts, style="verbose")
    salient = _salient_block(facts, vocab)
    # The whitelist itself, in the model's own auditing context. Anything outside it is H6
    # (Critical), and a verifier that is not told the whitelist cannot catch an invented
    # citation -- which would understate the one class the paper leans on hardest.
    permitted = (
        "\n".join(
            f"- {ref.citation} (variants: {', '.join(ref.phrase_variants)})"
            for ref in sorted(vocab.regulatory.values(), key=lambda r: r.ident)
        )
        or "- (none)"
    )

    narrative = initial.narrative
    usage: dict[str, Any] = dict(initial.usage)
    rounds = 0
    n_calls = 1
    parse_failures = 0
    converged = False
    history: list[tuple[SelfReportedViolation, ...]] = []

    for round_index in range(max_rounds):
        verify_prompt = verify_template.render(
            {
                "fact_record": fact_record,
                "unavailable_block": unavailable,
                "typology_label": facts.typology.label,
                "salient_block": salient,
                "permitted_regulatory_block": permitted,
                "draft_narrative": narrative,
            }
        )
        try:
            verdict_text, verify_usage = _complete(
                teacher,
                verify_prompt,
                case_id=facts.case_id,
                kind="baseline_verify",
                attempt=round_index,
            )
        except BaselineError:
            # A failed verification call ends the loop with the draft we have. It is not a
            # failure of the baseline's narrative, and discarding the case here would
            # silently shrink B5's test set relative to every other system's.
            log.warning("B5 verification call failed for %s; keeping current draft", facts.case_id)
            break
        n_calls += 1
        usage = _merge_usage(usage, verify_usage)

        violations, parsed = parse_verification(verdict_text)
        if not parsed:
            parse_failures += 1
            log.warning(
                "B5 verification response did not parse for %s (round %d)",
                facts.case_id,
                round_index + 1,
            )
            break
        history.append(violations)
        if not violations:
            converged = True
            break

        rounds += 1
        repair_prompt = repair_template.render(
            {
                "fact_record": fact_record,
                "unavailable_block": unavailable,
                "typology_label": facts.typology.label,
                "salient_block": salient,
                "hedging_block": _hedging_block(vocab),
                "draft_narrative": narrative,
                "violation_block": _violation_block(violations),
                "min_words": str(min_words),
                "max_words": str(max_words),
            }
        )
        try:
            narrative, repair_usage = _complete(
                teacher,
                repair_prompt,
                case_id=facts.case_id,
                kind="baseline_repair",
                attempt=round_index,
            )
        except BaselineError:
            log.warning("B5 repair call failed for %s; keeping previous draft", facts.case_id)
            break
        narrative = narrative.strip()
        n_calls += 1
        usage = _merge_usage(usage, repair_usage)

    return BaselineOutput(
        case_id=facts.case_id,
        system=system,
        narrative=narrative,
        model=teacher.spec.model,
        prompt_name=initial.prompt_name,
        prompt_hash=initial.prompt_hash,
        rendered_hash=initial.rendered_hash,
        n_exemplars=initial.n_exemplars,
        exemplar_case_ids=initial.exemplar_case_ids,
        trace=AgenticTrace(
            rounds=rounds,
            n_calls=n_calls,
            violations_per_round=tuple(history),
            converged=converged,
            parse_failures=parse_failures,
        ),
        usage=usage,
    )


def _merge_usage(base: Mapping[str, Any], addition: Mapping[str, Any]) -> dict[str, Any]:
    """Accumulate token and cost counters across an agentic loop's calls.

    Args:
        base: Usage so far.
        addition: This call's usage.

    Returns:
        The merged mapping. Numeric fields sum; everything else takes the latest value,
        so ``model`` reflects what actually served the calls.
    """
    merged = dict(base)
    for key, value in addition.items():
        if isinstance(value, int | float) and isinstance(merged.get(key), int | float):
            merged[key] = merged[key] + value
        elif isinstance(value, int | float) and key not in merged:
            merged[key] = value
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------- the CPU arms ---


def render_template_baseline(
    case_ids: Sequence[str],
    bronze: Mapping[str, str],
    *,
    system: str = "B1",
) -> list[dict[str, Any]]:
    """Produce B1's outputs by reading the Bronze corpus.

    B1 is the deterministic template, which Phase 4 already rendered for all 15,707 cases.
    Re-rendering here would risk producing a *second* Bronze that differs from the
    committed one, so this reads the corpus rather than regenerating it -- and D-063's
    prohibition on regenerating Bronze applies for a separate reason anyway: Phase 7 has
    since populated ``model_signal``, so a regenerated corpus would push the encoder's own
    score into the serialisation baseline.

    Args:
        case_ids: Cases to emit, in order.
        bronze: Case id to its Bronze narrative.
        system: The system id to record.

    Returns:
        One mapping per case, in the shape
        :func:`g2t_aml.eval.types.load_system_outputs` reads.

    Raises:
        BaselineError: If a requested case has no Bronze narrative.
    """
    missing = [case_id for case_id in case_ids if case_id not in bronze]
    if missing:
        raise BaselineError(
            f"{len(missing)} case(s) have no Bronze narrative, e.g. {missing[:3]}; B1 is "
            "read from the committed corpus and never re-rendered (D-063)"
        )
    return [
        {
            "case_id": case_id,
            "system": system,
            "narrative": bronze[case_id],
            "model": None,
            "generator": "bronze-template",
        }
        for case_id in case_ids
    ]


def render_classifier_template_baseline(
    case_ids: Sequence[str],
    bronze: Mapping[str, str],
    predictions: Mapping[str, float],
    *,
    system: str = "B2",
    threshold: float = 0.5,
    render: Callable[[str, float, bool], str] | None = None,
) -> list[dict[str, Any]]:
    """Produce B2's outputs: the encoder's prediction rendered through the template.

    B2 is the plan's baseline (a) -- classify, then say the classification. Its narrative
    is the Bronze text plus an explicit statement of the model's risk score and its binary
    call, which is exactly what a deployed classifier-plus-template system emits and is
    the thing our contribution claims to improve on.

    Args:
        case_ids: Cases to emit.
        bronze: Case id to its Bronze narrative.
        predictions: Case id to the encoder's risk score.
        system: The system id to record.
        threshold: Score above which the case is called suspicious.
        render: Override for the sentence appended, taking ``(case_id, score, flagged)``.
            Present so the wording is testable and so a change to it is a change to a
            named function rather than to an f-string in the middle of a loop.

    Returns:
        One mapping per case.

    Raises:
        BaselineError: If a requested case has no Bronze narrative or no prediction.
    """
    missing_bronze = [c for c in case_ids if c not in bronze]
    missing_score = [c for c in case_ids if c not in predictions]
    if missing_bronze or missing_score:
        raise BaselineError(
            f"B2 needs both a Bronze narrative and a prediction per case; missing "
            f"{len(missing_bronze)} narrative(s) and {len(missing_score)} prediction(s)"
        )

    def _default(case_id: str, score: float, flagged: bool) -> str:  # noqa: ARG001
        call = "suspicious" if flagged else "not suspicious"
        return (
            f" The graph classifier assigned this case a risk score of {score:.2f} and "
            f"classified it as {call} at the configured threshold of {threshold:.2f}."
        )

    renderer = render or _default
    out: list[dict[str, Any]] = []
    for case_id in case_ids:
        score = float(predictions[case_id])
        flagged = score >= threshold
        out.append(
            {
                "case_id": case_id,
                "system": system,
                "narrative": bronze[case_id] + renderer(case_id, score, flagged),
                "model": "gatv2",
                "generator": "classifier-template",
                "risk_score": score,
                "flagged": flagged,
            }
        )
    return out


def prompts_dir_default() -> Path:
    """Return where the versioned baseline prompts live.

    Returns:
        The repository's ``prompts/`` directory, resolved from this module's position
        because a prompt is source rather than data -- committed, reviewed and hashed
        alongside the code that renders it.
    """
    return Path(__file__).resolve().parents[3] / "prompts"


def assert_prompts_loadable(prompts_dir: str | None = None) -> dict[str, str]:
    """Load every baseline prompt and return its content hash.

    Called by the runner before spending a single API dollar: a prompt file that does not
    parse should stop the run at second zero, not after four hundred cases.

    Args:
        prompts_dir: Override for the prompt directory. Tests only.

    Returns:
        Prompt name to content hash.

    Raises:
        FileNotFoundError: If a prompt file is missing.
        PromptRenderError: If a prompt's system section uses a per-case placeholder.
    """
    hashes: dict[str, str] = {}
    for name in (GENERATE_PROMPT_NAME, VERIFY_PROMPT_NAME, REPAIR_PROMPT_NAME):
        try:
            template = load_prompt(name, prompts_dir)
        except PromptRenderError:
            log.exception("baseline prompt %s failed to load", name)
            raise
        hashes[name] = template.content_hash
    return hashes
