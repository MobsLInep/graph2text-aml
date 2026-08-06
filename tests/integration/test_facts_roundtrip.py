"""THE Phase 3 gate: 1,000 real cases, 100% SUPPORTED, zero CONTRADICTED.

This is the test the session exists for. It runs the fact layer forward over real cases,
renders a narrative that says only true things, runs the same code backwards over that
narrative, and asserts total agreement.

Any failure is one of three things, and all three are bugs:

1. the extractor computed something wrong,
2. the checker verified something wrong, or
3. the two disagree about what a field *means* — the dangerous one, because both sides
   pass their own unit tests while the corpus and the metric drift apart silently.

**The round trip alone catches (2) and (3) but NOT (1), and that had to be found the hard
way.** Mutation-testing this gate — injecting a seconds-for-hours span bug, an off-by-one
node count, and a degree-counts-transactions bug — left it at 100% SUPPORTED every time.
The reason is circularity: the probe renders its claims *from the fact record*, so a wrong
value is stated wrongly and then verified against itself. A gate that cannot fail is not a
gate.

:mod:`tests.oracle` closes it. It recomputes the same quantities directly from the raw
Polars tables, sharing no code with the fact layer, and
``test_extractor_agrees_with_an_independent_oracle`` compares the two. All three injected
bugs are caught there. The two tests are complementary and the gate is both of them: the
oracle calibrates the instrument, the round trip proves the two directions mean the same
thing by it.

Skips cleanly when the real corpus is absent, since the data is 475 MB and gitignored. It
is the real enforcement point regardless: `make facts` runs it against the built corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.oracle import all_quantities
from tests.probe import run_probe

from g2t_aml.data.canonical import CanonicalGraph
from g2t_aml.data.case_extraction import GraphIndex
from g2t_aml.data.case_sampling import CaseCollection
from g2t_aml.facts.checkers import Verdict, summarise
from g2t_aml.facts.extractor import extract_facts
from g2t_aml.facts.schema import Money, facts_to_dict, is_available, validate_facts
from g2t_aml.facts.vocab import load_vocabulary

REPO = Path(__file__).resolve().parents[2]
INTERIM = REPO / "data" / "interim" / "amlworld_hi_small"
CASES = REPO / "data" / "processed" / "amlworld_hi_small" / "cases"

#: The gate size the brief specifies.
N_CASES = 1000

pytestmark = pytest.mark.integration


def _available() -> bool:
    return (INTERIM / "canonical.json").is_file() and (CASES / "cases.jsonl").is_file()


requires_corpus = pytest.mark.skipif(
    not _available(), reason="built case corpus absent; run `make data && make cases`"
)


@pytest.fixture(scope="module")
def corpus():
    graph = CanonicalGraph.load(INTERIM)
    collection = CaseCollection.load(CASES)
    return collection, GraphIndex(graph)


@pytest.fixture(scope="module")
def vocabulary():
    return load_vocabulary()


@requires_corpus
@pytest.mark.slow
def test_round_trip_over_one_thousand_real_cases(corpus, vocabulary):
    collection, index = corpus
    case_ids = collection.case_ids[:N_CASES]
    assert len(case_ids) == N_CASES, f"corpus holds only {len(case_ids)} cases"

    results = []
    failures = []
    for case_id in case_ids:
        case = collection.materialise(case_id, index)
        facts = extract_facts(case)
        for result in run_probe(facts, vocabulary):
            results.append(result)
            if result.verdict is not Verdict.SUPPORTED:
                failures.append(
                    f"{case_id} :: {result.claim.field_path or result.claim.claim_type.value}"
                    f" :: {result.verdict.value} :: {result.reason}"
                )

    summary = summarise(results)
    detail = "\n".join(failures[:25])
    assert (
        summary["by_verdict"]["contradicted"] == 0
    ), f"{summary['by_verdict']['contradicted']} CONTRADICTED claims:\n{detail}"
    assert (
        summary["by_verdict"]["unverifiable"] == 0
    ), f"{summary['by_verdict']['unverifiable']} UNVERIFIABLE claims:\n{detail}"
    assert summary["supported_rate"] == 1.0
    assert summary["n_claims"] > N_CASES * 10, "probe made implausibly few claims per case"


@requires_corpus
@pytest.mark.slow
def test_every_record_validates_against_the_frozen_schema(corpus):
    collection, index = corpus
    for case_id in collection.case_ids[:N_CASES]:
        facts = extract_facts(collection.materialise(case_id, index))
        validate_facts(facts_to_dict(facts))


@requires_corpus
@pytest.mark.slow
def test_extractor_agrees_with_an_independent_oracle(corpus):
    # The calibration test. tests/oracle.py recomputes these quantities straight off the
    # Polars tables and shares no code with g2t_aml.facts, so a disagreement convicts one
    # of the two rather than being absorbed by the round trip's circularity.
    collection, index = corpus
    mismatches: list[str] = []

    for case_id in collection.case_ids[:N_CASES]:
        case = collection.materialise(case_id, index)
        facts = extract_facts(case)
        focal = facts.focal_entity.id
        expected = all_quantities(case, focal)

        actual = {
            "structure.n_nodes": facts.structure.n_nodes,
            "structure.n_edges": facts.structure.n_edges,
            "structure.n_self_loops": facts.structure.n_self_loops,
            "structure.n_components": facts.structure.n_components,
            "structure.max_in_degree": facts.structure.max_in_degree,
            "structure.max_out_degree": facts.structure.max_out_degree,
            "focal_entity.in_degree": facts.focal_entity.in_degree,
            "focal_entity.out_degree": facts.focal_entity.out_degree,
            "focal_entity.n_transactions_in": facts.focal_entity.n_transactions_in,
            "focal_entity.n_transactions_out": facts.focal_entity.n_transactions_out,
        }
        for path, want in actual.items():
            if want != expected[path]:
                mismatches.append(
                    f"{case_id} :: {path} :: extractor={want} oracle={expected[path]}"
                )

        # Durations must agree in HOURS. This is the comparison that catches a unit slip.
        if (
            is_available(facts.temporal)
            and expected["temporal.span_hours"] is not None
            and abs(facts.temporal.span_hours - expected["temporal.span_hours"]) > 1e-6
        ):
            mismatches.append(
                f"{case_id} :: temporal.span_hours :: extractor="
                f"{facts.temporal.span_hours} oracle={expected['temporal.span_hours']}"
            )

        if (
            is_available(facts.labels)
            and facts.labels.n_illicit_transactions != expected["labels.n_illicit_transactions"]
        ):
            mismatches.append(f"{case_id} :: labels.n_illicit_transactions")

        if is_available(facts.flow):
            if set(facts.flow.currencies_involved) != expected["flow.currencies_involved"]:
                mismatches.append(f"{case_id} :: flow.currencies_involved")
            for path, amount in (
                ("flow.total_inflow", facts.flow.total_inflow),
                ("flow.total_outflow", facts.flow.total_outflow),
            ):
                want_pair = expected[path]
                if isinstance(amount, Money):
                    assert want_pair is not None, f"{case_id} :: {path} oracle found nothing"
                    value, currency = want_pair
                    if abs(amount.value - value) > 0.01 or amount.currency != currency:
                        mismatches.append(
                            f"{case_id} :: {path} :: extractor={amount.value} {amount.currency} "
                            f"oracle={value} {currency}"
                        )
                elif want_pair is not None:
                    # The extractor withheld an aggregate the oracle could compute: only
                    # legitimate when the case really is multi-currency, which the oracle
                    # would then have reported as None. So this is a genuine mismatch.
                    mismatches.append(f"{case_id} :: {path} withheld but oracle computed it")

    assert not mismatches, "extractor disagrees with the independent oracle:\n" + "\n".join(
        mismatches[:25]
    )


@requires_corpus
@pytest.mark.slow
def test_no_real_case_claims_a_typology_it_holds_no_evidence_for(corpus):
    # The record-level invariant from D-036, asserted over the real corpus rather than a
    # fixture. 346 of 30,000 cases carry a seeding-stream typology while holding no flagged
    # transaction; none of them may surface a named typology.
    collection, index = corpus
    offenders = []
    for case_id in collection.case_ids[:N_CASES]:
        facts = extract_facts(collection.materialise(case_id, index))
        if facts.typology.label == "unclassified" or not is_available(facts.labels):
            continue
        if facts.labels.n_illicit_transactions == 0:
            offenders.append(f"{case_id} claims {facts.typology.label} with no flagged txn")
    assert not offenders, "\n".join(offenders[:20])


@requires_corpus
def test_extraction_is_deterministic_over_real_cases(corpus):
    collection, index = corpus
    for case_id in collection.case_ids[:50]:
        case = collection.materialise(case_id, index)
        first = facts_to_dict(extract_facts(case))
        second = facts_to_dict(extract_facts(case))
        first["provenance"].pop("computed_at")
        second["provenance"].pop("computed_at")
        assert first == second, case_id


@requires_corpus
def test_focal_entity_is_the_extraction_seed_on_constructed_cases(corpus):
    collection, index = corpus
    records = collection.by_id()
    for case_id in collection.case_ids[:100]:
        facts = extract_facts(collection.materialise(case_id, index))
        assert facts.focal_entity.selection_rule == "extraction_seed"
        assert facts.focal_entity.id == records[case_id].seed_node
