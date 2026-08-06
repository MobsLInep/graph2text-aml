"""What the evaluation harness passes between its own layers.

Every module in :mod:`g2t_aml.eval` consumes and produces the types defined here, so that
Layer 1, Layer 2, the taxonomy scorer and the statistics module can be run independently
of one another and of whatever produced the narratives.

**A narrative and the record it is about arrive together, always.** There is no code path
in this package that scores a narrative against a fact record fetched by case id at the
point of use, because the pairing is the thing most easily got wrong at scale — an
off-by-one in a sort order silently scores every system against its neighbour's facts and
produces a plausible, entirely meaningless faithfulness number. :class:`ScoredCase`
carries the pair and :func:`pair_outputs_with_facts` is the one place the join happens.

**One system, one seed, one case, one narrative.** Multi-candidate decoding (Phase 9's
guard) is resolved *before* it reaches here, by the caller choosing the candidate it
intends to report; a harness that silently scored candidate zero would report the guard's
best-of-n arm as if it were greedy decoding.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from g2t_aml.corpus.record import SlotAnnotation
from g2t_aml.facts.schema import CaseFacts

__all__ = [
    "EvaluationInputError",
    "ScoredCase",
    "SystemOutput",
    "SystemOutputs",
    "load_system_outputs",
    "pair_outputs_with_facts",
]


class EvaluationInputError(ValueError):
    """Raised when evaluation inputs are missing, malformed or cannot be paired.

    Always fatal. A harness that skipped an unparseable row would report a metric over a
    silently smaller denominator than the one named in the paper, and nothing downstream
    could tell that from a genuinely smaller test set.
    """


@dataclass(frozen=True)
class SystemOutput:
    """One narrative from one system, on one case, under one seed.

    Attributes:
        system: The arm that produced it — ``"bronze"``, ``"S1"``, ``"B7"``, a Gold
            reference used as a system, anything. Free text, because Phase 12's ablation
            grid names arms the harness cannot know about.
        case_id: The case described.
        narrative: The narrative text, already canonicalised. Canonicalisation is the
            caller's job and happens once, at generation time — see
            :func:`~g2t_aml.corpus.silver.claim_extraction.canonicalise_narrative` for
            why doing it later breaks the character spans on the record.
        seed: The generation seed. None for a deterministic system such as Bronze.
            Reported variance is over this field, so a system that omits it is a system
            reported without variance, which the statistics module refuses to average.
        split: Which split the case came from, when known.
        stream: ``"balanced"`` or ``"realistic"``. The two test streams are reported
            separately and never pooled: the realistic stream's class imbalance makes a
            pooled mean a weighted average over two different populations.
        slots: Character-span alignment back to fact fields, when the producer emitted
            one. Bronze and Silver carry this; a model generation does not, and Method A
            recovers it by alignment instead.
        metadata: Anything the producer wants carried through to the report.
    """

    system: str
    case_id: str
    narrative: str
    seed: int | None = None
    split: str | None = None
    stream: str = "balanced"
    slots: tuple[SlotAnnotation, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, int | None]:
        """Return the identity a paired comparison joins on.

        Returns:
            ``(system, case_id, seed)``.
        """
        return (self.system, self.case_id, self.seed)


#: A collection of outputs. Aliased rather than wrapped: every consumer iterates it, none
#: of them needs behaviour on it, and a wrapper class would be a place for a filtering
#: convenience to grow that quietly changed a denominator.
SystemOutputs = Sequence[SystemOutput]


@dataclass(frozen=True)
class ScoredCase:
    """A narrative bound to the fact record it must be faithful to.

    Attributes:
        output: The narrative and its provenance.
        facts: The record it is scored against. Load-bearing: this is the record the
            narrative was *generated from*, not a record re-derived at scoring time.
        reference: The Gold narrative for this case, when one exists. Layer 1 needs it;
            Layer 2 never reads it, which is why faithfulness is measurable on the
            15,000 cases that will never have a human reference.
    """

    output: SystemOutput
    facts: CaseFacts
    reference: str | None = None

    @property
    def case_id(self) -> str:
        """Return the case id, asserting the pairing holds.

        Returns:
            The case id, which both halves agree on.

        Raises:
            EvaluationInputError: If the narrative and the record disagree about which
                case this is. Cheap to check and catastrophic to miss.
        """
        if self.output.case_id != self.facts.case_id:
            raise EvaluationInputError(
                f"narrative is for case {self.output.case_id!r} but the fact record is "
                f"for {self.facts.case_id!r}; the pairing is wrong"
            )
        return self.output.case_id

    @property
    def typology(self) -> str:
        """Return the case's typology label.

        Returns:
            The label, used for per-typology breakdowns.
        """
        return self.facts.typology.label

    @property
    def dataset(self) -> str:
        """Return the substrate key.

        Returns:
            The dataset the case came from, used for per-substrate breakdowns.
        """
        return self.facts.dataset


def _narrative_of(row: Mapping[str, Any], path: Path, index: int) -> str:
    """Read the narrative out of one JSONL row, whatever shape it is in.

    Three producers write narratives in this repository and they do not agree on a key:
    Phase 4/5/6 corpora use ``target_narrative``, Phase 9 generation uses ``texts`` (a
    list, because sampling returns several), and hand-written evaluation fixtures use
    ``narrative``. Reading all three here means the harness scores a corpus file and a
    generation file with the same call.

    Args:
        row: The parsed row.
        path: The file, for the error message.
        index: The row number, for the error message.

    Returns:
        The narrative text.

    Raises:
        EvaluationInputError: If the row carries no narrative under any known key, or
            carries an empty candidate list. Never returns the empty string as a
            fallback: an empty narrative scores as trivially non-hallucinating and would
            flatter a broken system.
    """
    for key in ("narrative", "target_narrative", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    texts = row.get("texts")
    if isinstance(texts, list) and texts:
        first = texts[0]
        if isinstance(first, str) and first.strip():
            return first
    raise EvaluationInputError(
        f"{path}:{index} carries no narrative; expected one of 'narrative', "
        "'target_narrative', 'text' or a non-empty 'texts' list"
    )


def load_system_outputs(
    path: str | Path,
    *,
    system: str | None = None,
    seed: int | None = None,
    stream: str = "balanced",
    limit: int | None = None,
) -> list[SystemOutput]:
    """Read a JSONL of narratives into system outputs.

    Args:
        path: A corpus file (``bronze.jsonl``), a Phase 9 generation file, or any JSONL
            with a ``case_id`` and a narrative under one of the recognised keys.
        system: The arm name to record. Taken from the row's ``system``/``tier`` field
            when omitted, and an error when neither supplies one — an unnamed system
            cannot be compared against anything.
        seed: Seed to record, overriding the row's own.
        stream: Which test stream these outputs belong to.
        limit: Stop after this many rows. For smoke runs only.

    Returns:
        The outputs, in file order.

    Raises:
        EvaluationInputError: If the file is missing, a row lacks a case id or a
            narrative, or no system name can be determined.
    """
    source = Path(path)
    if not source.is_file():
        raise EvaluationInputError(f"no system outputs at {source}")

    # Imported here rather than at module scope: utils.io pulls pandas and pyarrow in,
    # and eval.types is imported by every module in this package including the ones a
    # unit test exercises with no filesystem at all.
    from g2t_aml.utils.io import read_jsonl

    outputs: list[SystemOutput] = []
    for index, raw in enumerate(read_jsonl(source)):
        if limit is not None and len(outputs) >= limit:
            break
        if not isinstance(raw, dict):
            raise EvaluationInputError(f"{source}:{index} is not a JSON object")
        case_id = raw.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise EvaluationInputError(f"{source}:{index} carries no 'case_id'")
        name = system or raw.get("system") or raw.get("tier")
        if not isinstance(name, str) or not name:
            raise EvaluationInputError(
                f"{source}:{index} names no system; pass system= or add a 'system' or "
                "'tier' field"
            )
        row_seed = seed if seed is not None else raw.get("seed")
        slots = tuple(
            SlotAnnotation.from_dict(s)
            for s in raw.get("target_slots") or ()
            if isinstance(s, dict)
        )
        outputs.append(
            SystemOutput(
                system=name,
                case_id=case_id,
                narrative=_narrative_of(raw, source, index),
                seed=int(row_seed) if isinstance(row_seed, int) else None,
                split=raw.get("split") if isinstance(raw.get("split"), str) else None,
                stream=stream,
                slots=slots,
                metadata={k: v for k, v in raw.items() if k in ("generator", "family", "variant")},
            )
        )
    return outputs


def pair_outputs_with_facts(
    outputs: Iterable[SystemOutput],
    facts: Mapping[str, CaseFacts],
    *,
    references: Mapping[str, str] | None = None,
    on_missing: str = "raise",
) -> Iterator[ScoredCase]:
    """Join narratives to their fact records, and to their Gold reference when one exists.

    The one place the join happens, so there is one place to get it wrong and one place a
    test can pin. See the module docstring.

    Args:
        outputs: The narratives.
        facts: Fact records by case id.
        references: Gold narratives by case id. Cases without one get ``reference=None``
            and are excluded from Layer 1 rather than scored against nothing.
        on_missing: ``"raise"`` to fail when a narrative has no fact record, ``"skip"``
            to drop it. Defaults to raising: a missing record means the corpus and the
            generation disagree about which cases exist, and skipping hides that.

    Yields:
        One :class:`ScoredCase` per output that has a fact record.

    Raises:
        EvaluationInputError: If a narrative has no fact record and ``on_missing`` is
            ``"raise"``, or if ``on_missing`` is not one of the two permitted values.
    """
    if on_missing not in ("raise", "skip"):
        raise EvaluationInputError(f"on_missing must be 'raise' or 'skip', got {on_missing!r}")
    refs = references or {}
    for output in outputs:
        record = facts.get(output.case_id)
        if record is None:
            if on_missing == "skip":
                continue
            raise EvaluationInputError(
                f"no fact record for case {output.case_id!r} produced by system "
                f"{output.system!r}"
            )
        yield ScoredCase(output=output, facts=record, reference=refs.get(output.case_id))
