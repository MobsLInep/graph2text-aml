"""The training record: one (graph, facts, narrative) example, and its frozen schema.

``schemas/training_record_v1.json`` is the contract; this module is its Python face, the
same relationship :mod:`g2t_aml.facts.schema` has to ``case_facts_v1.json``.

**One schema for all three tiers.** Bronze, Silver and Gold produce records that differ
only in ``tier`` and in the ``generator`` block. Designing that now is what stops Phase 5
from needing a migration, and — more importantly — it is what lets the ten-point harness
gate all three with one implementation, so a claim that Silver is verified means exactly
what the same claim means for Bronze.

**The slot annotation is the load-bearing idea here.** A narrative that carries character
spans back to fact fields can be verified without an LLM extractor: the checker reads the
*rendered text* at each span, parses it, and compares against the record. Phase 10's
Layer-2 faithfulness evaluation becomes an alignment problem over these spans rather than
an extraction problem, and Silver's verifier can align a rewrite against the Bronze it was
rewritten from. Emitting them is cheap at generation time and impossible to reconstruct
afterwards, which is why every tier carries them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema
import referencing
import referencing.jsonschema

from g2t_aml.facts.schema import SCHEMA_PATH as CASE_FACTS_SCHEMA_PATH
from g2t_aml.facts.schema import CaseFacts, facts_to_dict

__all__ = [
    "TRAINING_RECORD_SCHEMA_PATH",
    "TRAINING_RECORD_SCHEMA_VERSION",
    "BronzeNarrative",
    "SlotAnnotation",
    "TrainingRecord",
    "load_training_record_schema",
    "training_record_validator",
    "validate_training_record",
]

#: FROZEN in Phase 4. Independent of ``case_facts``: the two version separately, and both
#: are recorded on every record (invariant 3).
TRAINING_RECORD_SCHEMA_VERSION = "1.0.0"

#: Location of the JSON Schema. Resolved from this module's position, because the schema
#: is source rather than data and must never disagree with the code beside it.
TRAINING_RECORD_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3] / "schemas" / "training_record_v1.json"
)


@dataclass(frozen=True)
class SlotAnnotation:
    """One filled slot: what was written, where, and which fact it came from.

    Attributes:
        field_path: Dotted path into the fact record, e.g. ``"motifs.fan_out.width"``.
            For a qualitative descriptor this is the field its binding resolves against,
            so even a hedge phrase is anchored to a number.
        span: ``(start, end)`` character offsets into the rendered narrative, half-open.
            ``narrative[start:end] == rendered_value`` always holds, and validation
            asserts it rather than assuming it.
        rendered_value: The text actually written. **This, not** :attr:`raw_value`, is
            what the checker parses back into a claim.
        raw_value: The value read from the record before formatting. Kept for diagnostics
            and for measuring rounding error, never for verification.
        claim_type: Which tolerance rule applies when the slot becomes a claim.
    """

    field_path: str
    span: tuple[int, int]
    rendered_value: str
    raw_value: Any
    claim_type: str

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised annotation.

        Returns:
            The mapping the training-record schema validates.
        """
        return {
            "field_path": self.field_path,
            "span": [self.span[0], self.span[1]],
            "rendered_value": self.rendered_value,
            "raw_value": self.raw_value,
            "claim_type": self.claim_type,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SlotAnnotation:
        """Rebuild an annotation from its serialised form.

        Args:
            payload: A ``target_slots`` entry.

        Returns:
            The annotation.
        """
        start, end = payload["span"]
        return cls(
            field_path=str(payload["field_path"]),
            span=(int(start), int(end)),
            rendered_value=str(payload["rendered_value"]),
            raw_value=payload["raw_value"],
            claim_type=str(payload["claim_type"]),
        )


@dataclass(frozen=True)
class BronzeNarrative:
    """A rendered narrative, its slot alignment, and which template produced it.

    Attributes:
        case_id: The case described.
        text: The narrative, plain text. This is the training target.
        annotated: The same narrative with every slot marked as
            ``{field.path|rendered value}``. Written alongside the plain text so a human
            reviewing the corpus, and Silver's rewrite prompt, can both see which spans
            are load-bearing.
        slots: Every filled slot, in the order they appear in the text.
        family: The template family used.
        variant: Which surface realisation of that family.
        sections: The four SAR sections, keyed ``subject``, ``activity``, ``pattern``,
            ``basis``. Kept separately so Phase 10 can report faithfulness per section —
            the pattern section is where a generator's hallucinations concentrate, and an
            aggregate over the whole narrative would hide that.
        renderer_version: The renderer that produced this text.
    """

    case_id: str
    text: str
    annotated: str
    slots: tuple[SlotAnnotation, ...]
    family: str
    variant: int
    sections: dict[str, str] = field(default_factory=dict)
    renderer_version: str = "0.1.0"

    def slot_paths(self) -> tuple[str, ...]:
        """Return the distinct fact fields this narrative mentions.

        Returns:
            Field paths, sorted and deduplicated. Used to score salience coverage.
        """
        return tuple(sorted({s.field_path for s in self.slots}))


@dataclass(frozen=True)
class TrainingRecord:
    """One example, exactly as it appears in a corpus JSONL file.

    Attributes:
        case_id: The case described.
        dataset: Substrate key.
        split: ``train``, ``val`` or ``test``, read from the frozen manifest.
        tier: ``bronze``, ``silver`` or ``gold``.
        facts: The complete fact record, embedded so the narrative can always be
            re-verified against the record it was actually written from.
        graph_ref: ``<case store>#<case_id>``.
        serialised_facts: Flat-text serialisation, the B7 baseline's input.
        target_narrative: The narrative.
        target_slots: Character-span alignment back to fact fields.
        generator: How the narrative was produced.
        verification: Checker output over this narrative's own claims.
        length: Token, word and character counts, with the counter that measured them.
        salience: Required, excused and mentioned salient fields, plus coverage.
    """

    case_id: str
    dataset: str
    split: str
    tier: str
    facts: CaseFacts
    graph_ref: str
    serialised_facts: str
    target_narrative: str
    target_slots: tuple[SlotAnnotation, ...]
    generator: dict[str, Any]
    verification: dict[str, Any]
    length: dict[str, Any] = field(default_factory=dict)
    salience: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the JSON the training-record schema validates.

        Returns:
            The record as a JSON-serialisable mapping.

        Raises:
            ValueError: If the embedded fact record carries no provenance.
        """
        payload: dict[str, Any] = {
            "schema_version": TRAINING_RECORD_SCHEMA_VERSION,
            "case_id": self.case_id,
            "dataset": self.dataset,
            "split": self.split,
            "tier": self.tier,
            "facts": facts_to_dict(self.facts),
            "graph_ref": self.graph_ref,
            "serialised_facts": self.serialised_facts,
            "target_narrative": self.target_narrative,
            "target_slots": [s.to_dict() for s in self.target_slots],
            "generator": dict(self.generator),
            "verification": dict(self.verification),
        }
        if self.length:
            payload["length"] = dict(self.length)
        if self.salience:
            payload["salience"] = dict(self.salience)
        return payload


@lru_cache(maxsize=1)
def load_training_record_schema() -> dict[str, Any]:
    """Load and cache the training-record JSON Schema.

    Returns:
        The parsed schema document.

    Raises:
        FileNotFoundError: If the schema is missing from the repository.
    """
    if not TRAINING_RECORD_SCHEMA_PATH.is_file():
        raise FileNotFoundError(
            f"training_record schema not found at {TRAINING_RECORD_SCHEMA_PATH}"
        )
    parsed: dict[str, Any] = json.loads(TRAINING_RECORD_SCHEMA_PATH.read_text(encoding="utf-8"))
    return parsed


@lru_cache(maxsize=1)
def training_record_validator() -> jsonschema.protocols.Validator:
    """Build the strict validator, with ``case_facts_v1.json`` resolvable locally.

    The training record ``$ref``s the frozen fact schema rather than restating it, so the
    two can never drift; that reference has to resolve against the committed file rather
    than the network, which is what the registry below is for. A validator that silently
    skipped an unresolvable ``$ref`` would validate the wrapper and none of the facts.

    Returns:
        A validator for the training-record schema.

    Raises:
        FileNotFoundError: If either schema is missing.
    """
    schema = load_training_record_schema()
    case_facts = json.loads(CASE_FACTS_SCHEMA_PATH.read_text(encoding="utf-8"))
    registry = referencing.Registry().with_resources(
        [
            (
                str(case_facts["$id"]),
                referencing.Resource.from_contents(
                    case_facts, default_specification=referencing.jsonschema.DRAFT202012
                ),
            ),
            (
                str(schema["$id"]),
                referencing.Resource.from_contents(
                    schema, default_specification=referencing.jsonschema.DRAFT202012
                ),
            ),
        ]
    )
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema, registry=registry)


def validate_training_record(payload: dict[str, Any]) -> None:
    """Validate a serialised training record against the frozen schema.

    Args:
        payload: The output of :meth:`TrainingRecord.to_dict`, or a record read from a
            corpus file.

    Raises:
        jsonschema.ValidationError: If the record violates the schema. Deliberately not
            wrapped in something friendlier: the validator's message names the exact
            failing path, which is what a debugging session needs.
        FileNotFoundError: If either schema is missing.
    """
    training_record_validator().validate(payload)
