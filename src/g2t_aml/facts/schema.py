"""The typed ``case_facts`` record, and the absence sentinel that makes it honest.

``schemas/case_facts_v1.json`` is the contract; this module is its Python face. Every
group is a frozen dataclass, every optional group is a union with :class:`Unavailable`,
and :func:`facts_to_dict` produces exactly the JSON the schema validates.

**Why a sentinel rather than ``None``.** Invariant 4 says nothing may assert a fact that
does not exist for its substrate, and the failure mode it guards against is quiet: a
missing amount defaulted to ``0.0`` reads as "nothing moved", and a missing timestamp
defaulted to ``None`` reads as "unknown, probably fine". Both are assertions the substrate
does not license. :class:`Unavailable` cannot be mistaken for either, because it is not a
number and not a null — a consumer that forgets to branch on it gets a type error at the
point of the mistake rather than a plausible wrong sentence three phases later.

There is one place a bare ``None`` survives, deliberately:
``labels.min_hops_to_known_illicit`` and the motif descriptors. There, ``None`` is a
*measured value* — "no known-illicit node is reachable", "there is no cycle" — not a
masked field. The distinction is documented per field in the JSON Schema and enforced by
the checkers, which return CONTRADICTED for a claim against a measured ``None`` and
UNVERIFIABLE for a claim against a sentinel.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeGuard, TypeVar

import jsonschema

from g2t_aml.data.canonical import AvailabilityMask

__all__ = [
    "CASE_FACTS_SCHEMA_VERSION",
    "EXTRACTOR_VERSION",
    "SCHEMA_PATH",
    "CaseFacts",
    "EntityInventory",
    "FlowFacts",
    "FocalEntity",
    "LabelFacts",
    "ModelSignal",
    "Money",
    "MotifFacts",
    "MotifResult",
    "PerCurrencyTotal",
    "Provenance",
    "StructureFacts",
    "TemporalFacts",
    "Typology",
    "Unavailable",
    "facts_to_dict",
    "is_available",
    "load_case_facts_schema",
    "unavailable_reason",
    "validate_facts",
]

#: FROZEN. Invariant 3: pinned here, echoed into every derived artifact, and bumping it
#: invalidates every generated corpus. Recorded as an invariant in CLAUDE.md.
CASE_FACTS_SCHEMA_VERSION = "1.0.0"

#: The extraction code's own version, independent of the schema. A bug fix that changes a
#: computed value bumps this and leaves the schema alone.
EXTRACTOR_VERSION = "0.1.0"

#: Location of the JSON Schema, relative to the repository root. Resolved from this
#: module's position rather than a config value, because the schema is source, not data:
#: it is committed next to the code and the two must never disagree.
SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "case_facts_v1.json"


class Unavailable:
    """The explicit absence sentinel: *this substrate cannot support this fact*.

    Not ``None``, not ``0``, not an empty string. A distinct type, so a consumer that
    forgets to check gets a ``TypeError`` where the mistake is rather than a plausible
    wrong number downstream.

    Instances are value-equal on :attr:`reason`, hashable, and serialise to
    ``{"available": false, "reason": ...}``.

    Attributes:
        reason: Machine-readable reason code, e.g.
            ``"substrate_has_no_monetary_amounts"``.
    """

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        """Create a sentinel.

        Args:
            reason: Non-empty machine-readable reason code.

        Raises:
            ValueError: If ``reason`` is empty. An unexplained absence is worse than no
                absence marker at all, because it cannot be reported.
        """
        if not reason:
            raise ValueError("an Unavailable sentinel must carry a non-empty reason")
        self.reason = reason

    @property
    def available(self) -> bool:
        """Report availability, always False.

        Present so duck-typed access mirrors the serialised form.

        Returns:
            False, always.
        """
        return False

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised sentinel.

        Returns:
            ``{"available": False, "reason": self.reason}``.
        """
        return {"available": False, "reason": self.reason}

    def __eq__(self, other: object) -> bool:
        """Compare on the reason code.

        Args:
            other: The object to compare against.

        Returns:
            True if ``other`` is an :class:`Unavailable` with the same reason.
        """
        return isinstance(other, Unavailable) and other.reason == self.reason

    def __hash__(self) -> int:
        """Hash on the reason code.

        Returns:
            The hash of ``(Unavailable, reason)``.
        """
        return hash(("Unavailable", self.reason))

    def __repr__(self) -> str:
        """Return an unambiguous representation.

        Returns:
            ``Unavailable('<reason>')``.
        """
        return f"Unavailable({self.reason!r})"

    def __bool__(self) -> bool:
        """Return False, so ``if facts.flow:`` reads correctly.

        Returns:
            False, always. Note that an *available* group is always truthy, because a
            populated dataclass has no ``__bool__`` and defaults to True.
        """
        return False


_T = TypeVar("_T")


def is_available(value: _T | Unavailable) -> TypeGuard[_T]:
    """Narrow a possibly-unavailable value to its available type.

    The idiom throughout the fact layer::

        if is_available(facts.flow):
            total = facts.flow.total_inflow   # mypy knows this is FlowFacts

    Args:
        value: A value that may be an :class:`Unavailable` sentinel.

    Returns:
        True when ``value`` is not a sentinel, narrowing its type for mypy.
    """
    return not isinstance(value, Unavailable)


def unavailable_reason(value: object) -> str | None:
    """Return the reason code when a value is a sentinel.

    Args:
        value: Any value from a fact record.

    Returns:
        The reason code, or None when the value is available.
    """
    return value.reason if isinstance(value, Unavailable) else None


# ------------------------------------------------------------------- leaves ---


@dataclass(frozen=True)
class Money:
    """A monetary quantity with its unit.

    There is no currency-free amount anywhere in this record. An amount without its unit
    is exactly the kind of fact a narrative can restate wrongly without contradicting
    anything, because there is nothing to contradict.

    Attributes:
        value: The amount.
        currency: The currency name as the substrate spells it, e.g. ``"US Dollar"``.
    """

    value: float
    currency: str

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised amount.

        Returns:
            ``{"value": ..., "currency": ...}``.
        """
        return {"value": self.value, "currency": self.currency}


@dataclass(frozen=True)
class PerCurrencyTotal:
    """One currency's share of a directional total.

    Always emitted alongside an aggregate, available or not: this is what makes the
    multi-currency sentinel safe, because the information is never lost, only the
    undefined sum is withheld.

    Attributes:
        currency: The currency name.
        value: Total in that currency.
        n_transfers: How many transfers contributed.
    """

    currency: str
    value: float
    n_transfers: int

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised total.

        Returns:
            Currency, value and transfer count.
        """
        return {
            "currency": self.currency,
            "value": self.value,
            "n_transfers": self.n_transfers,
        }


@dataclass(frozen=True)
class MotifResult:
    """One motif detector's verdict plus its quantitative descriptors.

    Attributes:
        present: Whether the motif fired at the configured threshold.
        descriptors: Named quantities, e.g. ``{"width": 12, "window_hours": 18.0}``. A
            descriptor is ``None`` exactly when the motif is absent, except for those
            documented in the JSON Schema as always reported (``chain.max_length``,
            ``bipartite.score``), which allow a narrative overclaim to be CONTRADICTED
            rather than merely UNVERIFIABLE.
        witness: The nodes evidencing the motif — the cycle's vertices, the fan's
            spokes. **Not serialised into the fact record**; carried so tests and the
            property-based suite can assert that a detected structure really is one.
    """

    present: bool
    descriptors: dict[str, Any] = field(default_factory=dict)
    witness: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised motif, witness excluded.

        Returns:
            ``{"present": ..., **descriptors}``.
        """
        return {"present": self.present, **self.descriptors}


# ------------------------------------------------------------------- groups ---


@dataclass(frozen=True)
class EntityInventory:
    """Every entity the case contains, so H1 is checkable from the record alone.

    Attributes:
        node_ids: Sorted, unique. An entity reference outside this list is H1.
        focal_id: The focal entity, always a member of ``node_ids``.
    """

    node_ids: tuple[str, ...]
    focal_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised inventory.

        Returns:
            Node ids as a list, plus the focal id.
        """
        return {"node_ids": list(self.node_ids), "focal_id": self.focal_id}


@dataclass(frozen=True)
class StructureFacts:
    """Pure topology. Available on every substrate — a graph always has a shape.

    Attributes:
        n_nodes: Node count.
        n_edges: Transaction count, self-loops included.
        n_components: Weakly-connected components.
        density: Directed simple density over distinct non-loop ordered pairs.
        diameter: Longest shortest path over the undirected projection, maximised within
            each component. ``None`` only when the case has fewer than two nodes.
        max_in_degree: Largest count of distinct in-neighbours, self excluded.
        max_out_degree: Largest count of distinct out-neighbours, self excluded.
        reciprocity: Share of distinct non-loop pairs whose reverse also exists.
        n_self_loops: Self-loop transactions, kept rather than cleaned (D-017).
    """

    n_nodes: int
    n_edges: int
    n_components: int
    density: float
    diameter: int | None
    max_in_degree: int
    max_out_degree: int
    reciprocity: float
    n_self_loops: int

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised structure block.

        Returns:
            Every field, in declaration order.
        """
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class FocalEntity:
    """The account the case is about.

    Attributes:
        id: Canonical node identifier.
        selection_rule: ``"extraction_seed"`` on a constructed case,
            ``"max_degree"`` on a provided one.
        in_degree: Distinct in-neighbours inside the case, self excluded.
        out_degree: Distinct out-neighbours inside the case, self excluded.
        n_transactions_in: Inbound transaction count, self-loops excluded.
        n_transactions_out: Outbound transaction count, self-loops excluded.
        role: A member of the controlled role vocabulary.
        first_seen: First activity, or a sentinel without absolute timestamps.
        last_seen: Last activity, or a sentinel without absolute timestamps.
    """

    id: str
    selection_rule: str
    in_degree: int
    out_degree: int
    n_transactions_in: int
    n_transactions_out: int
    role: str
    first_seen: datetime | Unavailable
    last_seen: datetime | Unavailable

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised focal-entity block.

        Returns:
            Every field, with timestamps as ISO-8601 strings or sentinels.
        """
        return {
            "id": self.id,
            "selection_rule": self.selection_rule,
            "in_degree": self.in_degree,
            "out_degree": self.out_degree,
            "n_transactions_in": self.n_transactions_in,
            "n_transactions_out": self.n_transactions_out,
            "role": self.role,
            "first_seen": _timestamp_to_json(self.first_seen),
            "last_seen": _timestamp_to_json(self.last_seen),
        }


@dataclass(frozen=True)
class TemporalFacts:
    """When things happened. Wholly unavailable without absolute timestamps.

    Attributes:
        window_start: First transaction in the case.
        window_end: Last transaction in the case.
        span_hours: Observed extent, ``window_end - window_start``. Note this is the
            *observed* extent, not the padded extraction window (D-019).
        burst_detected: Whether a qualifying burst exists.
        burst_window_hours: Observed span of the tightest qualifying burst. ``None``
            exactly when no burst was found — a value, not a mask.
        burst_txn_count: Transactions inside that burst, or ``None``.
        burst_start: First transaction of that burst, or ``None``.
        event_ordering: Phase sequence around the focal entity.
        n_transactions: Transactions the block was computed from.
    """

    window_start: datetime
    window_end: datetime
    span_hours: float
    burst_detected: bool
    burst_window_hours: float | None
    burst_txn_count: int | None
    burst_start: datetime | None
    event_ordering: tuple[str, ...]
    n_transactions: int

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised temporal block.

        Returns:
            Every field, with datetimes as ISO-8601 strings.
        """
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "span_hours": self.span_hours,
            "burst_detected": self.burst_detected,
            "burst_window_hours": self.burst_window_hours,
            "burst_txn_count": self.burst_txn_count,
            "burst_start": self.burst_start.isoformat() if self.burst_start else None,
            "event_ordering": list(self.event_ordering),
            "n_transactions": self.n_transactions,
        }


@dataclass(frozen=True)
class FlowFacts:
    """How much moved. Wholly unavailable without monetary amounts.

    Individual aggregates carry their own sentinels: summing across currencies without a
    conversion rate is undefined, and this record refuses to encode it as a number.

    Attributes:
        total_inflow: Value received by the focal entity from other accounts in the case.
        total_outflow: Value sent by the focal entity to other accounts in the case.
        retained: ``total_inflow - total_outflow``, only when both share one currency.
        max_single_transfer: Largest single transfer in the case.
        inflow_by_currency: Always populated, sorted by currency.
        outflow_by_currency: Always populated, sorted by currency.
        n_transfers_near_threshold: Transfers in the threshold currency inside the band.
        threshold_reference: The reporting threshold measured against.
        threshold_currency: The currency the threshold is denominated in.
        threshold_band_fraction: How far below the threshold counts as near.
        currencies_involved: Every currency appearing in the case, sorted.
        cross_border: Permanently a sentinel — no substrate carries jurisdiction.
        cross_institution: More than one distinct bank. What *is* derivable.
        n_distinct_banks: Distinct banks among the case's accounts.
        payment_formats: Distinct payment formats, sorted.
    """

    total_inflow: Money | Unavailable
    total_outflow: Money | Unavailable
    retained: Money | Unavailable
    max_single_transfer: Money | Unavailable
    inflow_by_currency: tuple[PerCurrencyTotal, ...]
    outflow_by_currency: tuple[PerCurrencyTotal, ...]
    n_transfers_near_threshold: int
    threshold_reference: float
    threshold_currency: str
    threshold_band_fraction: float
    currencies_involved: tuple[str, ...]
    cross_border: Unavailable
    cross_institution: bool | Unavailable
    n_distinct_banks: int | Unavailable
    payment_formats: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised flow block.

        Returns:
            Every field, with sentinels and money objects expanded.
        """
        return {
            "total_inflow": _money_to_json(self.total_inflow),
            "total_outflow": _money_to_json(self.total_outflow),
            "retained": _money_to_json(self.retained),
            "max_single_transfer": _money_to_json(self.max_single_transfer),
            "inflow_by_currency": [t.to_dict() for t in self.inflow_by_currency],
            "outflow_by_currency": [t.to_dict() for t in self.outflow_by_currency],
            "n_transfers_near_threshold": self.n_transfers_near_threshold,
            "threshold_reference": self.threshold_reference,
            "threshold_currency": self.threshold_currency,
            "threshold_band_fraction": self.threshold_band_fraction,
            "currencies_involved": list(self.currencies_involved),
            "cross_border": self.cross_border.to_dict(),
            "cross_institution": _scalar_to_json(self.cross_institution),
            "n_distinct_banks": _scalar_to_json(self.n_distinct_banks),
            "payment_formats": list(self.payment_formats),
        }


@dataclass(frozen=True)
class LabelFacts:
    """Counterparty ground truth around the focal entity.

    Unavailable when the substrate carries no per-transaction illicit label: Elliptic2
    labels whole subgraphs, which licenses no statement about an individual counterparty.

    Attributes:
        n_illicit_counterparties: Counterparties on at least one flagged transaction.
        n_licit_counterparties: Counterparties labelled throughout with no flag.
        n_unknown_counterparties: Counterparties with an unlabelled incident transaction.
        n_counterparties: Distinct counterparties of the focal entity.
        min_hops_to_known_illicit: Undirected hops to the nearest flagged node. ``0``
            when the focal entity is one; ``None`` when none is reachable — a value.
        illicit_inflow_share: Share of inflow *value* on flagged transactions.
        n_illicit_transactions: Flagged transactions in the case.
        focal_is_illicit: Whether the focal entity sits on a flagged transaction.
    """

    n_illicit_counterparties: int
    n_licit_counterparties: int
    n_unknown_counterparties: int
    n_counterparties: int
    min_hops_to_known_illicit: int | None
    illicit_inflow_share: float | Unavailable
    n_illicit_transactions: int
    focal_is_illicit: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised labels block.

        Returns:
            Every field, with the share expanded if it is a sentinel.
        """
        return {
            "n_illicit_counterparties": self.n_illicit_counterparties,
            "n_licit_counterparties": self.n_licit_counterparties,
            "n_unknown_counterparties": self.n_unknown_counterparties,
            "n_counterparties": self.n_counterparties,
            "min_hops_to_known_illicit": self.min_hops_to_known_illicit,
            "illicit_inflow_share": _scalar_to_json(self.illicit_inflow_share),
            "n_illicit_transactions": self.n_illicit_transactions,
            "focal_is_illicit": self.focal_is_illicit,
        }


@dataclass(frozen=True)
class MotifFacts:
    """The eight structural detectors.

    Attributes:
        fan_in: Many senders into one account.
        fan_out: One account to many recipients.
        chain: Longest simple directed path.
        cycle: Shortest directed cycle at or above the configured minimum.
        bipartite: Exact two-colourability with both sides above the minimum.
        stack: Consecutive layers of at least the minimum width.
        gather_scatter: One hub collecting then dispersing.
        scatter_gather: One origin splitting then recombining at one destination.
    """

    fan_in: MotifResult
    fan_out: MotifResult
    chain: MotifResult
    cycle: MotifResult
    bipartite: MotifResult
    stack: MotifResult
    gather_scatter: MotifResult
    scatter_gather: MotifResult

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised motifs block.

        Returns:
            Motif name to ``{"present": ..., **descriptors}``.
        """
        return {name: getattr(self, name).to_dict() for name in MOTIF_NAMES}

    def as_mapping(self) -> dict[str, MotifResult]:
        """Return the motifs as a name-keyed mapping.

        Returns:
            Motif name to :class:`MotifResult`, in schema order.
        """
        return {name: getattr(self, name) for name in MOTIF_NAMES}


#: Motif names in schema order. A single source of truth for iteration, so a new detector
#: cannot be added to the dataclass and forgotten by the serialiser.
MOTIF_NAMES: tuple[str, ...] = (
    "fan_in",
    "fan_out",
    "chain",
    "cycle",
    "bipartite",
    "stack",
    "gather_scatter",
    "scatter_gather",
)


@dataclass(frozen=True)
class Typology:
    """What kind of scheme this is, and on whose authority.

    Attributes:
        label: A member of the controlled typology vocabulary.
        source: ``"ground_truth"``, ``"inferred"`` or ``"none"``.
        confidence: 1.0 for ground truth; a documented function of motif evidence for
            an inferred label.
        scope: ``"stream_membership"`` means the case is *part of* a stream of this
            typology and may not exhibit it in full (D-019). ``"case_structure"`` means
            the label describes the case's own shape.
    """

    label: str
    source: str
    confidence: float
    scope: str

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised typology block.

        Returns:
            Label, source, confidence and scope.
        """
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ModelSignal:
    """The GAT's output, written back in Phase 7.

    Null-valued at extraction time and never inferred from the graph: this block is the
    model's opinion, not a fact about the subgraph, and conflating the two would let a
    model's own score be used to verify a narrative the model wrote.

    Attributes:
        gnn_risk_score: Risk score in [0, 1], or None before Phase 7.
        score_percentile: Percentile within the evaluation population, or None.
        top_contributing_nodes: ``(node_id, attribution)`` pairs, highest first.
        model_version: The checkpoint that produced the score, or None.
    """

    gnn_risk_score: float | None = None
    score_percentile: float | None = None
    top_contributing_nodes: tuple[tuple[str, float], ...] = ()
    model_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised model-signal block.

        Returns:
            Every field, with contributions as a list of objects.
        """
        return {
            "gnn_risk_score": self.gnn_risk_score,
            "score_percentile": self.score_percentile,
            "top_contributing_nodes": [
                {"node_id": n, "attribution": a} for n, a in self.top_contributing_nodes
            ],
            "model_version": self.model_version,
        }


@dataclass(frozen=True)
class Provenance:
    """Where every value came from.

    Attributes:
        case_extraction: The upstream Phase 2 provenance, verbatim.
        field_producers: Field path to the named graph computation that produced it. The
            checker uses this to attribute a disagreement to a specific computation
            rather than to the record as a whole.
        computed_at: When extraction ran.
        config: The resolved :class:`~g2t_aml.facts.config.FactConfig`, so a detector's
            verdict is reproducible from the record alone.
    """

    case_extraction: dict[str, Any]
    field_producers: dict[str, str]
    computed_at: datetime
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised provenance block.

        Returns:
            Every field, with ``computed_at`` as an ISO-8601 string.
        """
        return {
            "case_extraction": _jsonify(self.case_extraction),
            "field_producers": dict(sorted(self.field_producers.items())),
            "computed_at": self.computed_at.isoformat(),
            "config": _jsonify(self.config),
        }


@dataclass(frozen=True)
class CaseFacts:
    """The complete fact record for one case.

    Attributes:
        case_id: The case this describes.
        dataset: Substrate key.
        availability: The substrate's mask, carried verbatim so the record is
            self-describing.
        entity_inventory: Every entity in the case.
        structure: Topology.
        focal_entity: The account the case is about.
        temporal: When things happened, or a sentinel.
        flow: How much moved, or a sentinel.
        labels: Counterparty ground truth, or a sentinel.
        motifs: The eight structural detectors.
        typology: What kind of scheme, and on whose authority.
        model_signal: The GAT's output, null until Phase 7.
        provenance: Where every value came from.
        schema_version: Frozen at :data:`CASE_FACTS_SCHEMA_VERSION`.
        extractor_version: The code version that produced this record.
    """

    case_id: str
    dataset: str
    availability: AvailabilityMask
    entity_inventory: EntityInventory
    structure: StructureFacts
    focal_entity: FocalEntity
    temporal: TemporalFacts | Unavailable
    flow: FlowFacts | Unavailable
    labels: LabelFacts | Unavailable
    motifs: MotifFacts
    typology: Typology
    model_signal: ModelSignal = field(default_factory=ModelSignal)
    provenance: Provenance | None = None
    schema_version: str = CASE_FACTS_SCHEMA_VERSION
    extractor_version: str = EXTRACTOR_VERSION

    def with_model_signal(self, signal: ModelSignal) -> CaseFacts:
        """Return a copy carrying a Phase 7 model signal.

        The write-back path designed for now and used later: the record is frozen, so
        attaching a score produces a new record rather than mutating one another phase
        may already have hashed.

        Args:
            signal: The GAT's output for this case.

        Returns:
            A new record identical but for ``model_signal``.
        """
        return dataclasses.replace(self, model_signal=signal)


def facts_to_dict(facts: CaseFacts) -> dict[str, Any]:
    """Serialise a fact record to the JSON the schema validates.

    Args:
        facts: The record to serialise.

    Returns:
        A JSON-serialisable mapping matching ``schemas/case_facts_v1.json``.

    Raises:
        ValueError: If ``provenance`` is unset. A record without provenance is not a
            measurement, and writing one would defeat the point of the module.
    """
    if facts.provenance is None:
        raise ValueError(
            f"case {facts.case_id!r} has no provenance; a fact record without a record "
            "of how it was computed is not a measurement"
        )
    return {
        "schema_version": facts.schema_version,
        "case_id": facts.case_id,
        "dataset": facts.dataset,
        "extractor_version": facts.extractor_version,
        "availability": facts.availability.to_dict(),
        "entity_inventory": facts.entity_inventory.to_dict(),
        "structure": facts.structure.to_dict(),
        "focal_entity": facts.focal_entity.to_dict(),
        "temporal": _group_to_json(facts.temporal),
        "flow": _group_to_json(facts.flow),
        "labels": _group_to_json(facts.labels),
        "motifs": facts.motifs.to_dict(),
        "typology": facts.typology.to_dict(),
        "model_signal": facts.model_signal.to_dict(),
        "provenance": facts.provenance.to_dict(),
    }


@lru_cache(maxsize=1)
def load_case_facts_schema() -> dict[str, Any]:
    """Load and cache the JSON Schema.

    Returns:
        The parsed schema document.

    Raises:
        FileNotFoundError: If the schema is missing from the repository.
    """
    if not SCHEMA_PATH.is_file():
        raise FileNotFoundError(f"case_facts schema not found at {SCHEMA_PATH}")
    parsed: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return parsed


def validate_facts(payload: dict[str, Any]) -> None:
    """Validate a serialised fact record against the frozen schema.

    Args:
        payload: Output of :func:`facts_to_dict`, or a record read from disk.

    Raises:
        jsonschema.ValidationError: If the record violates the schema. Not caught and
            re-raised as something friendlier on purpose: the validator's message names
            the exact failing path, which is what a debugging session needs.
        FileNotFoundError: If the schema is missing.
    """
    jsonschema.validate(instance=payload, schema=load_case_facts_schema())


# -------------------------------------------------------------- serialising ---


def _group_to_json(group: Any) -> dict[str, Any]:
    """Serialise a possibly-unavailable fact group.

    Args:
        group: A group dataclass or an :class:`Unavailable` sentinel.

    Returns:
        The group's ``to_dict()`` output, or the sentinel's.
    """
    result: dict[str, Any] = group.to_dict()
    return result


def _money_to_json(value: Money | Unavailable) -> dict[str, Any]:
    """Serialise a monetary field.

    Args:
        value: A :class:`Money` or a sentinel.

    Returns:
        The serialised amount or the serialised sentinel.
    """
    return value.to_dict()


def _scalar_to_json(value: Any) -> Any:
    """Serialise a scalar that may be a sentinel.

    Args:
        value: A scalar or an :class:`Unavailable`.

    Returns:
        The scalar unchanged, or the sentinel's mapping.
    """
    return value.to_dict() if isinstance(value, Unavailable) else value


def _timestamp_to_json(value: datetime | Unavailable) -> Any:
    """Serialise a timestamp that may be a sentinel.

    Args:
        value: A datetime or an :class:`Unavailable`.

    Returns:
        An ISO-8601 string, or the sentinel's mapping.
    """
    return value.to_dict() if isinstance(value, Unavailable) else value.isoformat()


def _jsonify(value: Any) -> Any:
    """Coerce arbitrary provenance content into JSON-serialisable form.

    Phase 2 provenance carries datetimes and numpy scalars, and a fact record must
    round-trip through JSON without either. Unknown types fall back to ``str`` rather
    than raising: losing the exact type of a provenance annotation is acceptable, losing
    the whole record is not.

    Args:
        value: Any provenance value.

    Returns:
        A JSON-serialisable equivalent.
    """
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Unavailable):
        return value.to_dict()
    if value is None or isinstance(value, str | bool | int | float):
        return value
    return str(value)
