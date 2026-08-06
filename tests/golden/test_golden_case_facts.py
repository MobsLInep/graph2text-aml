"""Golden-file tests: twenty fixture cases with committed expected fact records.

Invariant 1 in practice. Any change to the extractor that alters one of these files must
be an explicit, reviewed decision — the diff shows up in review, and a reviewer can see
exactly which number moved and for which shape. Unit tests say "this value is right";
golden files say "this value has not changed without anyone noticing", and those are
different guarantees.

Regenerate deliberately, never reflexively:

    uv run python -m tests.golden.test_golden_case_facts --regenerate

and then read the diff before committing it.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

from g2t_aml.data.canonical import CanonicalGraph
from g2t_aml.facts.config import FactConfig
from g2t_aml.facts.extractor import extract_facts
from g2t_aml.facts.schema import facts_to_dict, validate_facts
from g2t_aml.facts.serialiser import serialise_facts
from tests.factories import (
    acct,
    as_laundering_stream,
    at,
    bipartite_case,
    chain_case,
    cycle_case,
    elliptic2_case,
    fan_in_case,
    fan_out_case,
    flat_case,
    gather_scatter_case,
    make_case,
    scatter_gather_case,
    stack_case,
)

GOLDEN_DIR = Path(__file__).resolve().parent / "case_facts"

#: Fixed so a golden record is byte-stable across runs. Extraction is otherwise
#: deterministic; only the wall clock is not.
FROZEN_CLOCK = datetime(2026, 8, 1, 12, 0, 0)

CONFIG = FactConfig()


def _labelled(case: CanonicalGraph, typology: str) -> CanonicalGraph:
    """Mark the case as a real AMLworld positive: typology ON THE TRANSACTIONS.

    Not merely `case.typology = ...`. The fact layer reads the typology from the edge
    table, so a case-level attribute alone would produce `unclassified` and the fixture
    would not resemble a real positive at all. See D-036.
    """
    return as_laundering_stream(case, typology)


def _multi_currency() -> CanonicalGraph:
    """A case whose inflows span two currencies, so aggregates must be withheld."""
    focal = acct(0)
    return make_case(
        [
            {
                "src": acct(1),
                "dst": focal,
                "amount": 4000.0,
                "currency": "US Dollar",
                "timestamp": at(0),
            },
            {
                "src": acct(2),
                "dst": focal,
                "amount": 3000.0,
                "currency": "Euro",
                "timestamp": at(2),
            },
            {
                "src": focal,
                "dst": acct(3),
                "amount": 6500.0,
                "currency": "US Dollar",
                "timestamp": at(6),
            },
        ],
        seed_node=focal,
        case_id="fixture-multi-currency",
    )


def _near_threshold() -> CanonicalGraph:
    """Four transfers sitting in the structuring band, plus one at the threshold."""
    focal = acct(0)
    amounts = [9200.0, 9500.0, 9800.0, 9950.0, 10_000.0]
    return make_case(
        [
            {"src": focal, "dst": acct(i + 1), "amount": a, "timestamp": at(i * 0.5)}
            for i, a in enumerate(amounts)
        ],
        seed_node=focal,
        case_id="fixture-near-threshold",
    )


def _burst() -> CanonicalGraph:
    """Eight transfers inside two hours: a burst, and rapid_dispersal holds."""
    focal = acct(0)
    return make_case(
        [
            {"src": focal, "dst": acct(i + 1), "amount": 1000.0, "timestamp": at(i * 0.25)}
            for i in range(8)
        ],
        seed_node=focal,
        case_id="fixture-burst",
    )


def _no_burst() -> CanonicalGraph:
    """Eight transfers, thirty hours apart: many transactions, deliberately no burst."""
    focal = acct(0)
    return make_case(
        [
            {"src": focal, "dst": acct(i + 1), "amount": 1000.0, "timestamp": at(i * 30.0)}
            for i in range(8)
        ],
        seed_node=focal,
        case_id="fixture-no-burst",
    )


def _consolidation() -> CanonicalGraph:
    """Inflow, a holding gap, then outflow: the phase ordering with consolidation."""
    focal = acct(0)
    edges = [
        {"src": acct(i + 1), "dst": focal, "amount": 2000.0, "timestamp": at(i)} for i in range(3)
    ]
    edges += [
        {"src": focal, "dst": acct(10 + i), "amount": 1900.0, "timestamp": at(20 + i)}
        for i in range(3)
    ]
    return make_case(edges, seed_node=focal, case_id="fixture-consolidation")


def _self_loops() -> CanonicalGraph:
    """Self-loops alongside real transfers. D-017 keeps them; structure ignores them."""
    focal = acct(0)
    return make_case(
        [
            {"src": focal, "dst": focal, "amount": 500.0, "timestamp": at(0)},
            {"src": focal, "dst": focal, "amount": 700.0, "timestamp": at(1)},
            {"src": focal, "dst": acct(1), "amount": 1000.0, "timestamp": at(2)},
            {"src": acct(2), "dst": focal, "amount": 800.0, "timestamp": at(3)},
        ],
        seed_node=focal,
        case_id="fixture-self-loops",
    )


def _illicit_mix() -> CanonicalGraph:
    """Flagged, unflagged and unlabelled counterparties: the three-way label split."""
    focal = acct(0)
    return make_case(
        [
            {
                "src": acct(1),
                "dst": focal,
                "amount": 6000.0,
                "timestamp": at(0),
                "is_laundering": True,
            },
            {
                "src": acct(2),
                "dst": focal,
                "amount": 2000.0,
                "timestamp": at(1),
                "is_laundering": False,
            },
            {
                "src": acct(3),
                "dst": focal,
                "amount": 2000.0,
                "timestamp": at(2),
                "is_laundering": None,
            },
            {
                "src": focal,
                "dst": acct(4),
                "amount": 9000.0,
                "timestamp": at(5),
                "is_laundering": False,
            },
        ],
        seed_node=focal,
        case_id="fixture-illicit-mix",
    )


def _disconnected() -> CanonicalGraph:
    """Two disjoint components plus an isolated account."""
    return make_case(
        [
            {"src": acct(1), "dst": acct(2), "amount": 100.0, "timestamp": at(0)},
            {"src": acct(3), "dst": acct(4), "amount": 200.0, "timestamp": at(5)},
        ],
        seed_node=acct(1),
        extra_nodes=[acct(9)],
        case_id="fixture-disconnected",
    )


def _hub() -> CanonicalGraph:
    """Five in and five out: the hub role at exactly hub_min_degree."""
    focal = acct(0)
    edges = [
        {"src": acct(i + 1), "dst": focal, "amount": 1000.0, "timestamp": at(i)} for i in range(5)
    ]
    edges += [
        {"src": focal, "dst": acct(20 + i), "amount": 900.0, "timestamp": at(10 + i)}
        for i in range(5)
    ]
    return make_case(edges, seed_node=focal, case_id="fixture-hub")


def _cross_bank() -> CanonicalGraph:
    """Accounts at three institutions, so cross_institution is true and banks count."""
    focal = acct(0, bank="001")
    return make_case(
        [
            {"src": focal, "dst": acct(1, bank="077"), "amount": 3000.0, "timestamp": at(0)},
            {"src": focal, "dst": acct(2, bank="120"), "amount": 4000.0, "timestamp": at(1)},
            {"src": acct(3, bank="077"), "dst": focal, "amount": 5000.0, "timestamp": at(2)},
        ],
        seed_node=focal,
        case_id="fixture-cross-bank",
    )


#: The twenty golden fixtures. Each name becomes a committed JSON file. The set spans
#: every motif, both substrates, every availability outcome and each phase ordering.
GOLDEN_CASES: dict[str, CanonicalGraph] = {
    "fan_out": _labelled(fan_out_case(width=6, hours_apart=1.5), "fan_out"),
    "fan_in": _labelled(fan_in_case(width=5), "fan_in"),
    "chain": _labelled(chain_case(length=4), "stack"),
    "cycle": _labelled(cycle_case(length=4), "cycle"),
    "bipartite": _labelled(bipartite_case(left=3, right=3), "bipartite"),
    "stack": _labelled(stack_case(depth=3, layer_width=2), "stack"),
    "gather_scatter": _labelled(gather_scatter_case(gather=4, scatter=3), "gather_scatter"),
    "scatter_gather": _labelled(scatter_gather_case(width=4), "scatter_gather"),
    "flat": flat_case(),
    "elliptic2": elliptic2_case(),
    "multi_currency": _multi_currency(),
    "near_threshold": _near_threshold(),
    "burst": _burst(),
    "no_burst": _no_burst(),
    "consolidation": _consolidation(),
    "self_loops": _self_loops(),
    "illicit_mix": _illicit_mix(),
    "disconnected": _disconnected(),
    "hub": _hub(),
    "cross_bank": _cross_bank(),
}


def _record(case: CanonicalGraph) -> dict:
    """Extract a fact record under the frozen clock."""
    return facts_to_dict(extract_facts(case, CONFIG, computed_at=FROZEN_CLOCK))


def regenerate() -> None:
    """Rewrite every golden file. Run deliberately, then read the diff."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name, case in GOLDEN_CASES.items():
        payload = _record(case)
        (GOLDEN_DIR / f"{name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"wrote {len(GOLDEN_CASES)} golden records to {GOLDEN_DIR}")


def test_there_are_twenty_golden_fixtures():
    assert len(GOLDEN_CASES) == 20


@pytest.mark.parametrize("name", sorted(GOLDEN_CASES))
def test_golden_record_is_unchanged(name, golden_dir):
    path = golden_dir / f"{name}.json"
    assert path.is_file(), (
        f"golden file {path.name} is missing; regenerate deliberately with "
        "`python -m tests.golden.test_golden_case_facts --regenerate`"
    )
    expected = json.loads(path.read_text(encoding="utf-8"))
    actual = _record(GOLDEN_CASES[name])
    assert actual == expected, (
        f"the extractor's output for {name!r} has changed. If that change is intended, "
        "regenerate the golden files and explain the diff in DECISIONS.md."
    )


@pytest.mark.parametrize("name", sorted(GOLDEN_CASES))
def test_golden_record_validates(name, golden_dir):
    validate_facts(json.loads((golden_dir / f"{name}.json").read_text(encoding="utf-8")))


@pytest.mark.parametrize("name", sorted(GOLDEN_CASES))
def test_golden_case_serialises_in_both_styles(name):
    facts = extract_facts(GOLDEN_CASES[name], CONFIG, computed_at=FROZEN_CLOCK)
    for style in ("verbose", "compact"):
        text = serialise_facts(facts, style)
        assert text and facts.case_id in text


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        print(__doc__)
