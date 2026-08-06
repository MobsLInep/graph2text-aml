"""The controlled vocabulary, loaded and made executable.

``schemas/vocab_v1.yaml`` is the human-readable contract; this module turns it into
something the checker can run. The important piece is :func:`evaluate_condition`, which
resolves a qualitative phrase into a numeric verdict.

**Why the condition grammar is this small.** A binding is ``"< 24"`` or ``">= 8"`` — an
operator and a numeric literal, nothing else. There is no expression language, no boolean
combination, no field-to-field comparison, and there will not be one. This module decides
whether a published faithfulness number is correct; the grammar is deliberately too small
for a bug to hide in, and anything a richer grammar could express belongs in a named
detector in :mod:`g2t_aml.facts.motifs` where it can be unit-tested against a
hand-constructed fixture instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "VOCAB_PATH",
    "VOCAB_VERSION",
    "ControlledVocabulary",
    "RegulatoryReference",
    "RiskDescriptor",
    "evaluate_condition",
    "load_vocabulary",
    "parse_condition",
]

#: Location of the vocabulary, resolved from this module's position. It is source, not
#: data: committed next to the code, and the two must never disagree.
VOCAB_PATH = Path(__file__).resolve().parents[3] / "schemas" / "vocab_v1.yaml"

#: The vocabulary version this code expects. A mismatch raises rather than adapting.
VOCAB_VERSION = "1.0.0"

#: The complete condition grammar. An operator and a numeric literal. Nothing else.
_CONDITION_RE = re.compile(r"^\s*(?P<op><=|>=|==|!=|<|>)\s*(?P<value>-?\d+(?:\.\d+)?)\s*$")

_OPERATORS: dict[str, Any] = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


class VocabularyError(ValueError):
    """Raised when the vocabulary file is malformed or disagrees with the code."""


@dataclass(frozen=True)
class RiskDescriptor:
    """A qualitative intensifier and the numeric claim it makes.

    Attributes:
        name: The descriptor key, e.g. ``"rapid_dispersal"``.
        binds_to: The fact field path the phrase makes a claim about.
        condition: The condition on that field, e.g. ``"< 24"``.
        requires: Availability mask fields that must be True for the claim to be
            checkable at all. Absent flags yield UNVERIFIABLE rather than CONTRADICTED.
        phrase_variants: Every surface form of the descriptor, lower-cased.
    """

    name: str
    binds_to: str
    condition: str
    requires: tuple[str, ...]
    phrase_variants: tuple[str, ...]

    def holds_for(self, value: float | int) -> bool:
        """Report whether an observed value satisfies this descriptor's condition.

        Args:
            value: The value read from the fact record at :attr:`binds_to`.

        Returns:
            True when the condition holds.

        Raises:
            VocabularyError: If the condition is not parseable.
        """
        return evaluate_condition(self.condition, value)


@dataclass(frozen=True)
class RegulatoryReference:
    """One whitelisted regulatory citation.

    Anything outside the whitelist is H6 (Critical). Attributes mirror the YAML entry.

    Attributes:
        ident: Stable identifier, e.g. ``"us_ctr_10000"``.
        jurisdiction: Jurisdiction code.
        currency: The currency the threshold is denominated in.
        threshold: The numeric threshold.
        citation: The full citation text.
        phrase_variants: Surface forms, lower-cased.
    """

    ident: str
    jurisdiction: str
    currency: str
    threshold: float
    citation: str
    phrase_variants: tuple[str, ...]


@dataclass(frozen=True)
class ControlledVocabulary:
    """The whole vocabulary, parsed and indexed.

    Attributes:
        version: The vocabulary version.
        schema_version: The ``case_facts`` version it was written against.
        entity_roles: Role name to its definition mapping.
        typologies: Substrate key to its typology block.
        risk_descriptors: Descriptor name to :class:`RiskDescriptor`.
        hedging_allowed: Permitted hedge phrases, lower-cased.
        forbidden: Group name to ``(hallucination_class, phrases)``.
        required_for_inferred: Hedges that must wrap an inferred typology.
        regulatory: Whitelisted references, by identifier.
        salience: Typology to the field paths an adequate narrative must mention,
            already merged with the ``_common`` list.
    """

    version: str
    schema_version: str
    entity_roles: dict[str, dict[str, Any]]
    typologies: dict[str, dict[str, Any]]
    risk_descriptors: dict[str, RiskDescriptor]
    hedging_allowed: tuple[str, ...]
    forbidden: dict[str, tuple[str, tuple[str, ...]]]
    required_for_inferred: tuple[str, ...]
    regulatory: dict[str, RegulatoryReference]
    salience: dict[str, tuple[str, ...]]

    def role_names(self) -> tuple[str, ...]:
        """Return the closed entity-role vocabulary.

        Returns:
            Role names, sorted.
        """
        return tuple(sorted(self.entity_roles))

    def descriptor_for_phrase(self, phrase: str) -> RiskDescriptor | None:
        """Look up the descriptor a surface phrase belongs to.

        Args:
            phrase: A phrase as written in a narrative.

        Returns:
            The matching descriptor, or None when the phrase is not a controlled
            intensifier.
        """
        needle = phrase.strip().lower()
        for descriptor in self.risk_descriptors.values():
            if needle in descriptor.phrase_variants:
                return descriptor
        return None

    def forbidden_hit(self, text: str) -> tuple[str, str] | None:
        """Find the first forbidden phrase occurring in a text.

        Groups are searched in file order and phrases longest-first inside a group, so a
        specific phrase wins over a substring of itself.

        Args:
            text: The narrative text to scan.

        Returns:
            ``(hallucination_class, phrase)`` for the first hit, or None.
        """
        haystack = text.lower()
        for hallucination_class, phrases in self.forbidden.values():
            for phrase in sorted(phrases, key=len, reverse=True):
                if phrase in haystack:
                    return hallucination_class, phrase
        return None

    def salient_fields(self, typology: str) -> tuple[str, ...]:
        """Return the field paths an adequate narrative for a typology must mention.

        Args:
            typology: A typology label.

        Returns:
            The salience list, common fields first. Falls back to the ``unclassified``
            list for a typology with no entry of its own.
        """
        return self.salience.get(typology, self.salience["unclassified"])


def parse_condition(condition: str) -> tuple[str, float]:
    """Parse a binding condition into an operator and a threshold.

    Args:
        condition: A string such as ``"< 24"`` or ``">= 0.5"``.

    Returns:
        ``(operator, threshold)``.

    Raises:
        VocabularyError: If the condition is outside the grammar. Strict by design — a
            condition that silently failed to parse would make every claim bound to it
            quietly UNVERIFIABLE, which is invisible in an aggregate.
    """
    match = _CONDITION_RE.match(condition)
    if match is None:
        raise VocabularyError(
            f"condition {condition!r} is outside the grammar; expected an operator "
            f"({'|'.join(_OPERATORS)}) followed by a number"
        )
    return match.group("op"), float(match.group("value"))


def evaluate_condition(condition: str, value: float | int) -> bool:
    """Evaluate a binding condition against an observed value.

    Args:
        condition: A condition inside the grammar, e.g. ``">= 8"``.
        value: The observed value.

    Returns:
        True when the condition holds.

    Raises:
        VocabularyError: If the condition is outside the grammar.
    """
    operator, threshold = parse_condition(condition)
    result: bool = _OPERATORS[operator](float(value), threshold)
    return result


@lru_cache(maxsize=1)
def load_vocabulary(path: str | None = None) -> ControlledVocabulary:
    """Load, validate and cache the controlled vocabulary.

    Args:
        path: Override for the vocabulary location. Defaults to :data:`VOCAB_PATH`.
            Present for tests; production code passes nothing.

    Returns:
        The parsed vocabulary.

    Raises:
        FileNotFoundError: If the file is missing.
        VocabularyError: If the version disagrees with :data:`VOCAB_VERSION`, a required
            section is absent, or any binding condition is outside the grammar. Every
            condition is parsed at load time precisely so a typo fails once, loudly, at
            startup rather than turning one descriptor permanently UNVERIFIABLE.
    """
    source = Path(path) if path is not None else VOCAB_PATH
    if not source.is_file():
        raise FileNotFoundError(f"controlled vocabulary not found at {source}")
    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))

    version = str(raw.get("vocab_version", ""))
    if version != VOCAB_VERSION:
        raise VocabularyError(
            f"vocabulary version mismatch: file has {version!r}, code expects " f"{VOCAB_VERSION!r}"
        )
    for section in ("entity_roles", "typologies", "risk_descriptors", "hedging", "salience"):
        if section not in raw:
            raise VocabularyError(f"vocabulary is missing required section {section!r}")

    descriptors: dict[str, RiskDescriptor] = {}
    for name, body in raw["risk_descriptors"].items():
        condition = str(body["condition"])
        parse_condition(condition)  # fail loudly at load time, not silently at check time
        descriptors[name] = RiskDescriptor(
            name=name,
            binds_to=str(body["binds_to"]),
            condition=condition,
            requires=tuple(body.get("requires") or ()),
            phrase_variants=tuple(p.strip().lower() for p in body["phrase_variants"]),
        )

    hedging = raw["hedging"]
    forbidden: dict[str, tuple[str, tuple[str, ...]]] = {
        group: (
            str(body["hallucination_class"]),
            tuple(p.strip().lower() for p in body["phrases"]),
        )
        for group, body in hedging["forbidden"].items()
    }

    regulatory = {
        str(entry["id"]): RegulatoryReference(
            ident=str(entry["id"]),
            jurisdiction=str(entry["jurisdiction"]),
            currency=str(entry["currency"]),
            threshold=float(entry["threshold"]),
            citation=str(entry["citation"]),
            phrase_variants=tuple(p.strip().lower() for p in entry["phrase_variants"]),
        )
        for entry in raw.get("regulatory_references") or ()
    }

    raw_salience = dict(raw["salience"])
    common = tuple(raw_salience.pop("_common", ()))
    salience = {typology: common + tuple(fields) for typology, fields in raw_salience.items()}
    if "unclassified" not in salience:
        raise VocabularyError(
            "salience must define an 'unclassified' list; it is the fallback for any "
            "typology without an entry of its own"
        )

    return ControlledVocabulary(
        version=version,
        schema_version=str(raw.get("case_facts_schema_version", "")),
        entity_roles=dict(raw["entity_roles"]),
        typologies=dict(raw["typologies"]),
        risk_descriptors=descriptors,
        hedging_allowed=tuple(p.strip().lower() for p in hedging["allowed"]),
        forbidden=forbidden,
        required_for_inferred=tuple(
            p.strip().lower() for p in hedging["required_for_inferred"]["phrases"]
        ),
        regulatory=regulatory,
        salience=salience,
    )
