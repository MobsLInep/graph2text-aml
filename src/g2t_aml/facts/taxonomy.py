"""The hallucination taxonomy: nine classes, three of them Critical.

Every CONTRADICTED or UNVERIFIABLE verdict the checker returns carries one of these
classes. The classes are not decoration — they partition the error surface into groups
that mean different things to a compliance function, and three of them aggregate into a
**Critical Error Rate** reported independently of overall faithfulness.

Why report the critical three separately. A narrative that gets a count wrong is a
narrative that needs an edit. A narrative that calls an address a "mixer" (H4), cites a
regulation that does not exist (H6), or states that an account holder *is* laundering
money (H7) is a narrative that, filed as-is, would expose the institution. Averaging those
into a single faithfulness percentage lets a system with a 2% critical-error rate look
identical to one with 0%, and the difference between those two systems is the difference
between one that could be deployed and one that could not. A mean over unlike things hides
exactly the thing a reviewer needs to see.

H4, H6 and H7 also share a mechanism: each is an assertion the substrate cannot license at
all, rather than a value read off it incorrectly. That is why they are the classes the
controlled vocabulary attacks directly, by refusing to contain the words.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

__all__ = [
    "CRITICAL_CLASSES",
    "HallucinationClass",
    "Severity",
    "class_by_id",
    "taxonomy_table",
]


class Severity(str, Enum):
    """How badly a hallucination class damages the narrative's usability.

    Ordered ``LOW < MEDIUM < HIGH < CRITICAL`` by :meth:`rank`, which is what reporting
    sorts on. Inherits from ``str`` so a serialised record carries the readable name.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Return the severity's ordinal, ascending.

        Returns:
            0 for LOW through 3 for CRITICAL.
        """
        return _SEVERITY_RANK[self]


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


class HallucinationClass(Enum):
    """The nine hallucination classes, with severity and definition.

    Members are keyed ``H1``..``H9``. The value is a ``(id, title, severity,
    definition)`` tuple, unpacked onto properties so a call site reads as
    ``HallucinationClass.H4.severity`` rather than indexing a tuple.
    """

    H1 = (
        "H1",
        "Entity fabrication",
        Severity.HIGH,
        "Names a counterparty, account or institution that does not appear in the case "
        "subgraph. Checkable exactly against entity_inventory.node_ids.",
    )
    H2 = (
        "H2",
        "Numeric error",
        Severity.HIGH,
        "States a count, amount, degree, width or share that disagrees with the fact "
        "record beyond the tolerance declared for its claim type.",
    )
    H3 = (
        "H3",
        "Temporal error",
        Severity.MEDIUM,
        "Wrong ordering of events, an invented duration, or a timestamp outside the "
        "case window.",
    )
    H4 = (
        "H4",
        "Attribution fabrication",
        Severity.CRITICAL,
        "Assigns a business type or real-world identity to an entity — 'mixer', "
        "'exchange', 'shell company'. No substrate carries an entity-type column "
        "(availability.entity_types is false on both), so every such claim is "
        "unevidenced by construction.",
    )
    H5 = (
        "H5",
        "Typology error",
        Severity.MEDIUM,
        "Names a laundering typology the record does not carry, or asserts an inferred "
        "typology without the hedge the vocabulary requires.",
    )
    H6 = (
        "H6",
        "Regulatory fabrication",
        Severity.CRITICAL,
        "Cites a threshold, rule or reporting obligation outside the whitelist in "
        "vocab_v1.yaml. An invented rule is the failure mode that puts a real compliance "
        "team in breach.",
    )
    H7 = (
        "H7",
        "Guilt overclaim",
        Severity.CRITICAL,
        "Asserts guilt, criminality or proof rather than suspicion. A SAR is a referral "
        "for investigation, not a finding of fact, and the distinction is legal rather "
        "than stylistic.",
    )
    H8 = (
        "H8",
        "Unsupported inference",
        Severity.MEDIUM,
        "Asserts motive, intent, off-graph context, or the completeness of a scheme the "
        "case only partially contains (D-019).",
    )
    H9 = (
        "H9",
        "Omission of exculpatory fact",
        Severity.MEDIUM,
        "Leaves out a fact in the record that materially weakens the suspicion — a "
        "licit counterparty majority, an ordinary payment format, a benign phase "
        "ordering. The only class detected by absence rather than by assertion.",
    )

    def __init__(self, ident: str, title: str, severity: Severity, definition: str) -> None:
        """Unpack the member tuple onto named attributes.

        Args:
            ident: The ``H<n>`` identifier.
            title: Short human-readable name.
            severity: How badly this class damages usability.
            definition: What the class covers, in one paragraph.
        """
        self.ident = ident
        self.title = title
        self.severity = severity
        self.definition = definition

    @property
    def is_critical(self) -> bool:
        """Report whether this class counts toward the Critical Error Rate.

        Returns:
            True for H4, H6 and H7.
        """
        return self.severity is Severity.CRITICAL

    def to_dict(self) -> dict[str, Any]:
        """Return the class as a JSON-serialisable mapping.

        Returns:
            Identifier, title, severity, criticality and definition.
        """
        return {
            "id": self.ident,
            "title": self.title,
            "severity": self.severity.value,
            "is_critical": self.is_critical,
            "definition": self.definition,
        }


#: The three classes aggregated into the Critical Error Rate. Reported independently of
#: overall faithfulness — see the module docstring for why the mean is not enough.
CRITICAL_CLASSES: tuple[HallucinationClass, ...] = tuple(
    h for h in HallucinationClass if h.is_critical
)


def class_by_id(ident: str) -> HallucinationClass:
    """Look a hallucination class up by its ``H<n>`` identifier.

    Args:
        ident: One of ``"H1"``..``"H9"``, case-insensitive.

    Returns:
        The matching class.

    Raises:
        KeyError: If the identifier is not one of the nine. Deliberately strict: a
            checker that invents a class name would otherwise silently produce a
            report bucket nobody reads.
    """
    key = ident.strip().upper()
    try:
        return HallucinationClass[key]
    except KeyError as exc:
        known = ", ".join(h.ident for h in HallucinationClass)
        raise KeyError(f"unknown hallucination class {ident!r}; expected one of {known}") from exc


def taxonomy_table() -> list[dict[str, Any]]:
    """Return the whole taxonomy, in identifier order.

    Used by the reporting layer and by the annotation guidelines, so the table in the
    paper and the enum in the code cannot drift apart.

    Returns:
        One mapping per class, from :meth:`HallucinationClass.to_dict`.
    """
    return [h.to_dict() for h in HallucinationClass]
