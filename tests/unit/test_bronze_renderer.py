"""Rendering: the substrate guard, span correctness, determinism, and faithfulness.

The four properties Phase 4 rests on, each tested against hand-built fixtures whose
expected answers can be worked out from the picture in the factory's docstring rather than
from another module's output.
"""

from __future__ import annotations

import pytest
from tests import factories

from g2t_aml.corpus.bronze.renderer import (
    MAX_TOKENS,
    MIN_TOKENS,
    SubstrateViolation,
    render_bronze,
    select_family,
    select_variant,
)
from g2t_aml.corpus.bronze.templates import FAMILIES, family_for
from g2t_aml.corpus.claims import claims_from_slots
from g2t_aml.corpus.tokenization import get_token_counter
from g2t_aml.facts.checkers import CheckContext, Verdict, check_claim, check_narrative_text
from g2t_aml.facts.extractor import extract_facts
from g2t_aml.facts.salience import required_fields
from g2t_aml.facts.vocab import load_vocabulary

TYPED_FIXTURES = {
    "fan_out": lambda: factories.fan_out_case(width=9),
    "fan_in": lambda: factories.fan_in_case(width=9),
    "gather_scatter": lambda: factories.gather_scatter_case(gather=4, scatter=4),
    "scatter_gather": lambda: factories.scatter_gather_case(width=4),
    "cycle": lambda: factories.cycle_case(length=4),
    "bipartite": lambda: factories.bipartite_case(left=3, right=3),
    "stack": lambda: factories.stack_case(depth=3, layer_width=2),
    "random": lambda: factories.fan_out_case(width=4),
}


@pytest.fixture(scope="module")
def vocabulary():
    return load_vocabulary()


def _facts_for(family: str):
    """Build a fact record whose ground-truth typology selects the named family."""
    case = TYPED_FIXTURES[family]()
    return extract_facts(factories.as_laundering_stream(case, family))


class TestFamilySelection:
    def test_every_typology_family_renders_its_own_fixture(self, vocabulary) -> None:
        """Acceptance criterion: every family renders for at least one fixture case."""
        for family in TYPED_FIXTURES:
            facts = _facts_for(family)
            narrative = render_bronze(facts, vocabulary=vocabulary)
            assert narrative.family == family, f"{family} fixture routed to {narrative.family}"
            assert narrative.text

    def test_a_licit_case_gets_the_no_finding_family(self, vocabulary) -> None:
        facts = extract_facts(factories.fan_out_case(width=5))
        assert select_family(facts) == "no_finding"
        assert (
            "no structural laundering pattern"
            in render_bronze(facts, vocabulary=vocabulary).text.lower()
            or "no laundering typology" in render_bronze(facts, vocabulary=vocabulary).text.lower()
        )

    def test_a_two_account_case_gets_the_minimal_family(self, vocabulary) -> None:
        facts = extract_facts(factories.flat_case())
        assert select_family(facts) == "minimal_activity"
        assert render_bronze(facts, vocabulary=vocabulary).family == "minimal_activity"

    def test_an_elliptic2_case_gets_the_topology_family(self, vocabulary) -> None:
        facts = extract_facts(factories.elliptic2_case())
        assert select_family(facts) == "topology_only"
        assert render_bronze(facts, vocabulary=vocabulary).family == "topology_only"


class TestSubstrateGuard:
    """Invariant 4, executable. A masked fact is a hard error, not a shorter narrative."""

    def test_an_amount_bearing_family_raises_on_an_elliptic2_record(self, vocabulary) -> None:
        facts = extract_facts(factories.elliptic2_case())
        with pytest.raises(SubstrateViolation, match="monetary_amounts"):
            render_bronze(facts, family="fan_out", vocabulary=vocabulary)

    @pytest.mark.parametrize("family", sorted(set(FAMILIES) - {"topology_only"}))
    def test_every_amount_bearing_family_raises_on_elliptic2(self, family, vocabulary) -> None:
        facts = extract_facts(factories.elliptic2_case())
        with pytest.raises(SubstrateViolation):
            render_bronze(facts, family=family, vocabulary=vocabulary)

    def test_the_topology_family_renders_without_raising(self, vocabulary) -> None:
        facts = extract_facts(factories.elliptic2_case())
        narrative = render_bronze(facts, family="topology_only", vocabulary=vocabulary)
        assert narrative.text
        assert "unavailable" not in narrative.text.lower()

    def test_a_case_level_absence_drops_a_sentence_rather_than_raising(self, vocabulary) -> None:
        """An originator has no inflow. That is a fact about the case, not the substrate."""
        facts = extract_facts(factories.fan_out_case(width=5))
        narrative = render_bronze(facts, vocabulary=vocabulary)
        assert "flow.total_inflow" not in narrative.slot_paths()
        assert narrative.text


class TestSlotAnnotations:
    """The alignment Phase 10's Layer-2 evaluation depends on."""

    @pytest.mark.parametrize("family", sorted(TYPED_FIXTURES))
    def test_every_span_holds_exactly_its_rendered_value(self, family, vocabulary) -> None:
        narrative = render_bronze(_facts_for(family), vocabulary=vocabulary)
        assert narrative.slots
        for slot in narrative.slots:
            start, end = slot.span
            assert narrative.text[start:end] == slot.rendered_value, slot

    def test_spans_are_ordered_and_do_not_overlap(self, vocabulary) -> None:
        narrative = render_bronze(_facts_for("gather_scatter"), vocabulary=vocabulary)
        previous = -1
        for slot in narrative.slots:
            assert slot.span[0] >= previous
            assert slot.span[0] < slot.span[1]
            previous = slot.span[1]

    def test_the_annotated_form_recovers_the_plain_text(self, vocabulary) -> None:
        narrative = render_bronze(_facts_for("fan_in"), vocabulary=vocabulary)
        rebuilt = narrative.annotated
        for slot in narrative.slots:
            rebuilt = rebuilt.replace(
                f"{{{slot.field_path}|{slot.rendered_value}}}", slot.rendered_value, 1
            )
        assert rebuilt == narrative.text

    def test_every_slot_names_a_field(self, vocabulary) -> None:
        for family in TYPED_FIXTURES:
            for slot in render_bronze(_facts_for(family), vocabulary=vocabulary).slots:
                assert slot.field_path


class TestDeterminism:
    def test_the_same_case_renders_identically_across_runs(self, vocabulary) -> None:
        facts = _facts_for("cycle")
        first = render_bronze(facts, vocabulary=vocabulary)
        second = render_bronze(facts, vocabulary=vocabulary)
        assert first.text == second.text
        assert first.variant == second.variant
        assert [s.to_dict() for s in first.slots] == [s.to_dict() for s in second.slots]

    def test_the_seed_does_not_change_the_output(self, vocabulary) -> None:
        """A corpus that moved with the global seed could not be regenerated."""
        facts = _facts_for("stack")
        assert (
            render_bronze(facts, seed=1, vocabulary=vocabulary).text
            == render_bronze(facts, seed=99, vocabulary=vocabulary).text
        )

    def test_variant_selection_is_a_function_of_the_case_id(self) -> None:
        family = family_for("fan_out")
        assert select_variant("case-a", family) == select_variant("case-a", family)
        assert 0 <= select_variant("case-a", family) < family.n_realisations

    def test_different_cases_reach_different_realisations(self) -> None:
        family = family_for("no_finding")
        chosen = {select_variant(f"case-{i}", family) for i in range(200)}
        assert len(chosen) > 50, "variant selection is collapsing onto a few realisations"


class TestFaithfulnessByConstruction:
    """The whole point: render, then run the same checker in reverse."""

    @pytest.mark.parametrize("family", sorted(TYPED_FIXTURES))
    def test_no_claim_is_contradicted(self, family, vocabulary) -> None:
        facts = _facts_for(family)
        narrative = render_bronze(facts, vocabulary=vocabulary)
        context = CheckContext(facts=facts, vocabulary=vocabulary)
        results = [
            check_claim(c, context) for c in claims_from_slots(narrative.slots, narrative.text)
        ]
        results += check_narrative_text(narrative.text, context)
        contradicted = [r for r in results if r.verdict is Verdict.CONTRADICTED]
        assert not contradicted, [r.reason for r in contradicted]

    @pytest.mark.parametrize("family", sorted(TYPED_FIXTURES))
    def test_the_unverifiable_budget_is_respected(self, family, vocabulary) -> None:
        facts = _facts_for(family)
        narrative = render_bronze(facts, vocabulary=vocabulary)
        context = CheckContext(facts=facts, vocabulary=vocabulary)
        results = [
            check_claim(c, context) for c in claims_from_slots(narrative.slots, narrative.text)
        ]
        unverifiable = [r for r in results if r.verdict is Verdict.UNVERIFIABLE]
        assert len(unverifiable) / len(results) <= 0.05, [r.reason for r in unverifiable]

    def test_rounded_numbers_pass_the_checker_at_the_declared_tolerance(self, vocabulary) -> None:
        """Rounding reconciliation, end to end rather than per formatter."""
        facts = _facts_for("fan_out")
        narrative = render_bronze(facts, vocabulary=vocabulary)
        context = CheckContext(facts=facts, vocabulary=vocabulary)
        monetary = [
            check_claim(c, context)
            for c in claims_from_slots(narrative.slots, narrative.text)
            if c.field_path and c.field_path.startswith("flow.total")
        ]
        assert monetary, "the fixture should render at least one aggregate amount"
        assert all(r.verdict is Verdict.SUPPORTED for r in monetary)


class TestLengthAndSalience:
    @pytest.mark.parametrize("family", sorted(TYPED_FIXTURES))
    def test_length_is_inside_the_bounds(self, family, vocabulary) -> None:
        narrative = render_bronze(_facts_for(family), vocabulary=vocabulary)
        assert MIN_TOKENS <= get_token_counter().count(narrative.text) <= MAX_TOKENS

    @pytest.mark.parametrize("family", sorted(TYPED_FIXTURES))
    def test_every_required_salient_field_is_mentioned(self, family, vocabulary) -> None:
        """Bronze reaches 100% salience coverage by construction. That is its ceiling."""
        facts = _facts_for(family)
        narrative = render_bronze(facts, vocabulary=vocabulary)
        required, _ = required_fields(facts, vocabulary)
        missing = [p for p in required if p not in narrative.slot_paths()]
        assert not missing, missing

    def test_all_four_sections_are_present(self, vocabulary) -> None:
        narrative = render_bronze(_facts_for("bipartite"), vocabulary=vocabulary)
        assert set(narrative.sections) == {"subject", "activity", "pattern", "basis"}
        assert narrative.text.count("\n\n") == 3
