"""``extract_facts``: one case graph in, one checkable fact record out.

This is the forward direction of the fact layer. The reverse direction —
:mod:`g2t_aml.facts.checkers` — verifies narrative claims against exactly the record this
module produces, using the same field paths and the same semantics. That symmetry is what
makes the corpus generator and the faithfulness metric the same instrument, and it is why
a disagreement between the two is a bug in one of them rather than a tuning parameter.

Two choices in this file are worth reading before changing anything.

**The focal entity is the extraction seed when there is one.** A case was cut *around* an
account (D-018), and that account is what the case is about. Choosing a different focal
entity — the highest-degree node, say — would silently re-centre every narrative on
whichever counterparty happened to be busiest, which on HI-Small is frequently a
correspondent-bank-like hub that has nothing to do with why the case exists. Provided
subgraphs (Elliptic2) have no seed, so they fall back to maximum degree with a
lexicographic tie-break, and ``focal_entity.selection_rule`` records which rule fired.

**Roles are derived from in-case degree, never from anything outside the case.** An
account's global degree in HI-Small runs to 168,672; its degree inside a 48-hour case
window is what the narrative describes and what a reader can verify from the case. Reading
the node table's precomputed ``in_degree`` here would be the single easiest way to make
every narrative in the corpus unfaithful, since those columns are global aggregates.
"""

from __future__ import annotations

from datetime import UTC, datetime

from g2t_aml.data.canonical import TYPOLOGY_VOCABULARY, CanonicalGraph
from g2t_aml.facts import flow as flow_module
from g2t_aml.facts import labels as labels_module
from g2t_aml.facts import motifs as motifs_module
from g2t_aml.facts import structure as structure_module
from g2t_aml.facts import temporal as temporal_module
from g2t_aml.facts.caseview import CaseView, build_case_view
from g2t_aml.facts.config import FactConfig
from g2t_aml.facts.schema import (
    CaseFacts,
    EntityInventory,
    FocalEntity,
    ModelSignal,
    MotifFacts,
    Provenance,
    Typology,
    Unavailable,
)

__all__ = [
    "FIELD_PRODUCERS",
    "ROLE_VOCABULARY",
    "assign_role",
    "extract_facts",
    "extract_facts_from_view",
    "select_focal_entity",
]

#: The closed role vocabulary. Mirrors ``entity_roles`` in vocab_v1.yaml and
#: ``$defs.entity_role`` in the JSON Schema; a test asserts all three agree.
ROLE_VOCABULARY: tuple[str, ...] = (
    "originator",
    "intermediary",
    "beneficiary",
    "pass_through",
    "terminal",
    "hub",
)

#: Producers for the fields this module computes itself. The sub-extractors contribute
#: their own, and :func:`_field_producers` merges them all.
FIELD_PRODUCERS: dict[str, str] = {
    "entity_inventory.node_ids": "extractor.node_inventory",
    "entity_inventory.focal_id": "extractor.focal_selection",
    "focal_entity.id": "extractor.focal_selection",
    "focal_entity.selection_rule": "extractor.focal_selection",
    "focal_entity.in_degree": "extractor.in_case_distinct_degree",
    "focal_entity.out_degree": "extractor.in_case_distinct_degree",
    "focal_entity.n_transactions_in": "extractor.in_case_transaction_count",
    "focal_entity.n_transactions_out": "extractor.in_case_transaction_count",
    "focal_entity.role": "extractor.role_from_in_case_degree",
    "focal_entity.first_seen": "extractor.focal_activity_span",
    "focal_entity.last_seen": "extractor.focal_activity_span",
    "typology.label": "extractor.typology_resolution",
    "typology.source": "extractor.typology_resolution",
    "typology.confidence": "extractor.typology_resolution",
    "typology.scope": "extractor.typology_resolution",
}


class FactExtractionError(ValueError):
    """Raised when a case cannot yield a fact record."""


def select_focal_entity(view: CaseView) -> tuple[str, str]:
    """Choose the account the case is about.

    Args:
        view: The case view.

    Returns:
        ``(node_id, selection_rule)``. The rule is ``"extraction_seed"`` when Phase 2
        recorded a seed that is present in the case, and ``"max_degree"`` otherwise —
        maximum distinct counterparties, ties broken by the lexicographically smallest
        identifier so the choice never depends on row order.

    Raises:
        FactExtractionError: If the case has no accounts at all.
    """
    if not view.node_ids:
        raise FactExtractionError(f"case {view.case_id!r} has no nodes")

    seed = view.provenance.get("seed_node")
    if isinstance(seed, str) and seed in set(view.node_ids):
        return seed, "extraction_seed"

    best = max(view.node_ids, key=lambda n: (view.total_degree(n), [-ord(c) for c in n]))
    return best, "max_degree"


def assign_role(view: CaseView, node: str, config: FactConfig) -> str:
    """Assign a controlled role from the account's in-case degree.

    The rules, in the order they are tested — each is a binding published in
    ``vocab_v1.yaml``:

    - no counterparties either way  -> ``terminal`` (reachable only via self-loops)
    - sends only                    -> ``originator``
    - receives only                 -> ``beneficiary``
    - at least ``hub_min_degree`` distinct counterparties on both sides -> ``hub``
    - exactly one each way          -> ``pass_through``
    - otherwise                     -> ``intermediary``

    ``hub`` is tested before ``pass_through`` because the two cannot both hold: a hub
    needs five counterparties per side and a pass-through has one. The order is fixed
    anyway, so the rule set stays a total function no matter how the thresholds move.

    Args:
        view: The case view.
        node: The account to classify.
        config: Supplies ``hub_min_degree``.

    Returns:
        A member of :data:`ROLE_VOCABULARY`.
    """
    in_degree = view.in_degree(node)
    out_degree = view.out_degree(node)

    if in_degree == 0 and out_degree == 0:
        return "terminal"
    if in_degree == 0:
        return "originator"
    if out_degree == 0:
        return "beneficiary"
    if min(in_degree, out_degree) >= config.hub_min_degree:
        return "hub"
    if in_degree == 1 and out_degree == 1:
        return "pass_through"
    return "intermediary"


def _focal_activity(view: CaseView, focal: str) -> tuple[datetime | Unavailable, ...]:
    """Return the focal entity's first and last activity.

    Args:
        view: The case view.
        focal: The focal entity.

    Returns:
        ``(first_seen, last_seen)``, both sentinels when the substrate has no absolute
        timestamps or the account's transactions carry none.
    """
    if not view.availability.absolute_timestamps:
        sentinel = Unavailable("substrate_has_no_absolute_timestamps")
        return sentinel, sentinel
    stamps = sorted(
        e.timestamp for e in view.edges if focal in (e.src, e.dst) and e.timestamp is not None
    )
    if not stamps:
        sentinel = Unavailable("focal_entity_has_no_timestamped_transactions")
        return sentinel, sentinel
    return stamps[0], stamps[-1]


def _typology_in_case(view: CaseView) -> str | None:
    """Return the dominant typology carried by the case's OWN transactions.

    Args:
        view: The case view.

    Returns:
        The most frequent non-null typology among the case's transactions, ties broken by
        :data:`~g2t_aml.data.canonical.TYPOLOGY_VOCABULARY` order so the result never
        depends on row order. None when no transaction in the case carries one — which
        includes every licit case, and the 1.2% of stream-seeded cases whose window caught
        none of the stream's flagged transactions.
    """
    counts: dict[str, int] = {}
    for edge in view.edges:
        if edge.typology is not None:
            counts[edge.typology] = counts.get(edge.typology, 0) + 1
    if not counts:
        return None
    order = {name: i for i, name in enumerate(TYPOLOGY_VOCABULARY)}
    return min(counts, key=lambda name: (-counts[name], order.get(name, len(order))))


def _resolve_typology(view: CaseView, motifs: MotifFacts) -> Typology:
    """Decide the case's typology, its source and its scope.

    **Ground truth wins outright, including when it says "no scheme".** On a substrate with
    typology ground truth, a case carrying no typology is not an unlabelled case — the
    ground truth positively states it belongs to no laundering stream, and that is exactly
    what ``"unclassified"`` means in this vocabulary (D-013: ``None`` means the substrate
    has no typology ground truth at all, ``"unclassified"`` means it has some and reports
    no match). Falling through to motif inference there would be a serious error: 85% of
    the AMLworld corpus is licit, most licit cases are structurally *shaped* like something
    — an ordinary payroll run is a fan-out, supplier settlement is a chain — and inferring
    a laundering typology from that shape would attach one to 25,571 licit cases. Every
    Bronze narrative generated from those records would then assert a laundering typology
    for a case the data says is clean, and the corpus would teach the generator that
    structure alone implies a scheme. The structural finding is not lost: it is in the
    ``motifs`` block, where it belongs, as a shape rather than as a scheme.

    **The label is read from the case's own transactions, not from the seeding stream.**
    ``CaseCollection.materialise`` sets ``CanonicalGraph.typology`` from the *CaseRecord*,
    which carries the typology of the stream the case was seeded from. That is not the same
    thing as what the case contains. Measured on the built corpus: **346 of 30,000 cases
    (1.2%) carry a stream typology while holding no laundering transaction at all** — the
    48-hour window caught the seed account but none of its stream's flagged edges (D-019).
    Trusting the stream label there would emit ``{"label": "cycle", "source":
    "ground_truth"}`` for a subgraph containing no cycle, no flagged transaction and no
    evidence whatsoever, and a Bronze narrative built from it would assert a scheme the case
    cannot show. ``scope: stream_membership`` warns that a case may not exhibit its typology
    *in full*; it does not license claiming one the case exhibits *not at all*.

    So the typology is recomputed from the edge table's own ``typology`` column — which is
    what `case_extraction._dominant_typology` does at cut time, before ``materialise``
    discards it — with ties broken by :data:`TYPOLOGY_VOCABULARY` order so the result never
    depends on row order. The invariant this buys is checkable and is asserted in the test
    suite: **a record whose typology is not ``unclassified`` always has
    ``labels.n_illicit_transactions > 0``.**

    Three outcomes, in priority order:

    1. **Ground truth.** The substrate has typology ground truth. ``confidence`` is 1.0 and
       ``scope`` is ``stream_membership`` — a labelled case is *part of* a stream of that
       typology and, because of the 48-hour window cap, may not exhibit it in full (D-019);
       a case whose own transactions carry no typology is ``"unclassified"``, which is itself
       a ground-truth statement.
    2. **Inferred.** No ground truth at all (Elliptic2), but a motif detector fired. The
       label is the highest-priority firing motif, ``scope`` is ``case_structure``, and
       ``confidence`` is 0.5 for a single firing detector rising to 0.9 when several agree
       — a deliberately blunt scale, because a finer one would imply a calibration this
       evidence cannot support.
    3. **None.** No ground truth and no motif fired. Label ``unclassified``, source
       ``none``, confidence 0.0.

    Args:
        view: The case view.
        motifs: The detector results.

    Returns:
        The resolved typology block.
    """
    if view.availability.typology_ground_truth:
        return Typology(
            label=_typology_in_case(view) or "unclassified",
            source="ground_truth",
            confidence=1.0,
            scope="stream_membership",
        )

    # Priority order: the two-sided composites are more specific than the fans they are
    # built from, so a case exhibiting both is described by the more informative one.
    priority = (
        "gather_scatter",
        "scatter_gather",
        "cycle",
        "bipartite",
        "stack",
        "fan_out",
        "fan_in",
    )
    firing = [name for name in priority if motifs.as_mapping()[name].present]
    if not firing:
        return Typology(label="unclassified", source="none", confidence=0.0, scope="case_structure")
    confidence = 0.5 if len(firing) == 1 else min(0.9, 0.5 + 0.2 * (len(firing) - 1))
    return Typology(
        label=firing[0],
        source="inferred",
        confidence=round(confidence, 2),
        scope="case_structure",
    )


def _field_producers() -> dict[str, str]:
    """Merge every sub-extractor's provenance tags into one mapping.

    Returns:
        Field path to the named graph computation that produced it.
    """
    merged: dict[str, str] = {}
    for source in (
        FIELD_PRODUCERS,
        structure_module.FIELD_PRODUCERS,
        temporal_module.FIELD_PRODUCERS,
        flow_module.FIELD_PRODUCERS,
        labels_module.FIELD_PRODUCERS,
        motifs_module.FIELD_PRODUCERS,
    ):
        merged.update(source)
    return merged


def extract_facts_from_view(
    view: CaseView,
    config: FactConfig | None = None,
    *,
    computed_at: datetime | None = None,
) -> CaseFacts:
    """Extract a fact record from an already-built case view.

    Separated from :func:`extract_facts` so tests can drive the extractor from a
    hand-constructed view without materialising a Polars frame.

    Args:
        view: The case view.
        config: Every threshold. Defaults to :class:`~g2t_aml.facts.config.FactConfig`.
        computed_at: Extraction timestamp. Defaults to now (UTC). Injectable so golden
            files are byte-stable across runs.

    Returns:
        The complete fact record.

    Raises:
        FactExtractionError: If the case has no accounts, or holds more than
            ``config.max_inventory_nodes`` — the entity inventory the H1 checker depends
            on must be complete, and truncating it silently would make entity
            fabrication undetectable.
    """
    cfg = config if config is not None else FactConfig()
    if view.n_nodes > cfg.max_inventory_nodes:
        raise FactExtractionError(
            f"case {view.case_id!r} has {view.n_nodes} nodes, above max_inventory_nodes "
            f"({cfg.max_inventory_nodes}); the entity inventory must be complete for H1 "
            "to be checkable, so this raises rather than truncating it"
        )

    focal, selection_rule = select_focal_entity(view)
    first_seen, last_seen = _focal_activity(view, focal)
    motifs = motifs_module.extract_motifs(view, cfg)

    return CaseFacts(
        case_id=view.case_id,
        dataset=view.dataset,
        availability=view.availability,
        entity_inventory=EntityInventory(node_ids=view.node_ids, focal_id=focal),
        structure=structure_module.extract_structure(view),
        focal_entity=FocalEntity(
            id=focal,
            selection_rule=selection_rule,
            in_degree=view.in_degree(focal),
            out_degree=view.out_degree(focal),
            n_transactions_in=view.n_transactions_into(focal),
            n_transactions_out=view.n_transactions_out_of(focal),
            role=assign_role(view, focal, cfg),
            first_seen=first_seen,
            last_seen=last_seen,
        ),
        temporal=temporal_module.extract_temporal(view, focal, cfg),
        flow=flow_module.extract_flow(view, focal, cfg),
        labels=labels_module.extract_labels(view, focal),
        motifs=motifs,
        typology=_resolve_typology(view, motifs),
        model_signal=ModelSignal(),
        provenance=Provenance(
            case_extraction=dict(view.provenance),
            field_producers=_field_producers(),
            computed_at=computed_at if computed_at is not None else datetime.now(UTC),
            config=cfg.to_dict(),
        ),
    )


def extract_facts(
    case: CanonicalGraph,
    config: FactConfig | None = None,
    *,
    computed_at: datetime | None = None,
) -> CaseFacts:
    """Extract a checkable fact record from a case subgraph.

    The entrypoint the rest of the project uses. Deterministic: the same case and config
    produce the same record, ``computed_at`` aside.

    Args:
        case: The case, as materialised by Phase 2.
        config: Every threshold. Defaults to :class:`~g2t_aml.facts.config.FactConfig`.
        computed_at: Extraction timestamp. Defaults to now (UTC).

    Returns:
        The complete fact record, with availability sentinels wherever the substrate
        cannot support a fact family.

    Raises:
        FactExtractionError: If the case is empty or exceeds the inventory cap.
        ValueError: If an edge endpoint is absent from the case's node table.
    """
    return extract_facts_from_view(build_case_view(case), config, computed_at=computed_at)
