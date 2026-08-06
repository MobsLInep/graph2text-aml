"""Slot alignment: what survives a paraphrase, what is an addition, what went missing.

The extractor decides what a rewrite is *asserting*, and every faithfulness number Silver
reports depends on it. The property tested hardest is the one that would be invisible if
it broke: an unaligned quantity must produce a claim. Silence there would make an invented
figure *raise* the supported rate.
"""

from __future__ import annotations

import pytest
from tests import factories

from g2t_aml.corpus.bronze.renderer import render_bronze
from g2t_aml.corpus.silver.claim_extraction import (
    SlotAlignmentExtractor,
    canonicalise_narrative,
    extract_report,
)
from g2t_aml.facts.checkers import CheckContext, ClaimType, Verdict, check_claim
from g2t_aml.facts.extractor import extract_facts
from g2t_aml.facts.vocab import load_vocabulary


@pytest.fixture(scope="module")
def vocab():
    return load_vocabulary()


@pytest.fixture(scope="module")
def fixture(vocab):
    facts = extract_facts(
        factories.as_laundering_stream(factories.fan_out_case(width=6), "fan_out")
    )
    return facts, render_bronze(facts, vocabulary=vocab)


class TestAlignment:
    def test_bronze_round_trips_to_all_supported(self, fixture, vocab):
        facts, bronze = fixture
        report = extract_report(
            canonicalise_narrative(bronze.text), facts, bronze, vocabulary=vocab
        )
        context = CheckContext(facts=facts, vocabulary=vocab)
        results = [check_claim(claim, context) for claim in report.claims]
        assert results
        assert all(r.verdict is Verdict.SUPPORTED for r in results)
        assert report.n_added == 0
        assert report.dropped_paths == ()
        assert report.unparseable == ()

    def test_claims_are_parsed_from_the_rewrite_not_read_from_the_record(self, fixture, vocab):
        """D-040 in the Silver direction. If a claim took its value from the fact record,
        moving a number in the text would still verify -- and every corpus ever built
        would report 100% SUPPORTED."""
        facts, bronze = fixture
        moved = bronze.text.replace(str(facts.structure.n_nodes), "9999", 1)
        report = extract_report(canonicalise_narrative(moved), facts, bronze, vocabulary=vocab)
        assert "9999" in {text for _, _, text in report.added_spans}

    def test_reordering_the_text_preserves_alignment(self, fixture, vocab):
        """A rewrite moves words. Alignment must follow them, which is the whole reason
        spans are re-derived rather than carried over."""
        facts, bronze = fixture
        paragraphs = bronze.text.split("\n\n")
        if len(paragraphs) < 2:
            pytest.skip("fixture rendered a single paragraph")
        shuffled = "\n\n".join(paragraphs[::-1])
        report = extract_report(canonicalise_narrative(shuffled), facts, bronze, vocabulary=vocab)
        assert report.dropped_paths == ()
        assert report.n_added == 0

    def test_every_returned_span_holds_its_value(self, fixture, vocab):
        facts, bronze = fixture
        text = canonicalise_narrative("An introductory clause. " + bronze.text)
        report = extract_report(text, facts, bronze, vocabulary=vocab)
        for claim in report.claims:
            start, end = claim.text_span
            assert text[start:end] == claim.raw_text


class TestTokenBoundaries:
    """The bug a scale run found and 300 unit tests did not.

    In Bronze's own document order a long value is always reached before the short values
    that could hide inside it. A rewrite reorders content, and then the slot rendering
    ``2`` aligns inside ``2022-09-02 15:01``: the timestamp is reported as a dropped fact
    *and* its leftover digits come back as invented quantities — one reordering charged
    twice. It failed 102 of 300 real paraphrased cases and would have inflated the real
    discard rate by about 34 points, all of it spurious.
    """

    def test_a_short_value_does_not_align_inside_a_longer_one(self, fixture, vocab):
        facts, bronze = fixture
        reordered = "\n\n".join(reversed(bronze.text.split("\n\n")))
        report = extract_report(canonicalise_narrative(reordered), facts, bronze, vocabulary=vocab)
        assert report.dropped_paths == ()
        assert report.n_added == 0

    def test_a_value_is_not_matched_inside_a_longer_number(self, fixture, vocab):
        """`12` must not align inside `126`."""
        facts, bronze = fixture
        short = next(
            (s for s in bronze.slots if s.rendered_value.isdigit() and len(s.rendered_value) <= 2),
            None,
        )
        if short is None:
            pytest.skip("this fixture renders no short numeric slot")
        text = canonicalise_narrative(
            f"An unrelated figure of {short.rendered_value}9999 appears first. " + bronze.text
        )
        report = extract_report(text, facts, bronze, vocabulary=vocab)
        aligned = [c for c in report.claims if c.field_path == short.field_path]
        assert aligned
        for claim in aligned:
            start, end = claim.text_span
            assert text[start:end] == short.rendered_value
            assert not text[end : end + 1].isdigit()

    def test_a_timestamp_survives_reordering_intact(self, fixture, vocab):
        facts, bronze = fixture
        timestamps = [
            s for s in bronze.slots if s.claim_type == "temporal" and "-" in s.rendered_value
        ]
        if not timestamps:
            pytest.skip("this fixture renders no timestamp")
        reordered = "\n\n".join(reversed(bronze.text.split("\n\n")))
        report = extract_report(canonicalise_narrative(reordered), facts, bronze, vocabulary=vocab)
        for slot in timestamps:
            assert slot.field_path in report.aligned_paths


class TestAdditions:
    def test_an_invented_amount_becomes_a_claim(self, fixture, vocab):
        facts, bronze = fixture
        text = canonicalise_narrative(bronze.text + " A further 88,412.00 US Dollar was moved.")
        report = extract_report(text, facts, bronze, vocabulary=vocab)
        assert "88,412.00" in {t for _, _, t in report.added_spans}

    def test_an_addition_is_unverifiable_and_never_supported(self, fixture, vocab):
        """The three-valued answer. An unaligned figure has not been shown to be wrong,
        it has been shown to be unbacked -- and UNVERIFIABLE is the bucket for exactly
        the compliance-dangerous claims the graph cannot support."""
        facts, bronze = fixture
        text = canonicalise_narrative(bronze.text + " Some 41% of the inflow was retained.")
        report = extract_report(text, facts, bronze, vocabulary=vocab)
        context = CheckContext(facts=facts, vocabulary=vocab)
        added = [c for c in report.claims if c.raw_text == "41%"]
        assert added
        assert all(check_claim(c, context).verdict is Verdict.UNVERIFIABLE for c in added)

    def test_a_fabricated_account_becomes_an_entity_claim(self, fixture, vocab):
        facts, bronze = fixture
        text = canonicalise_narrative(bronze.text + " Funds reached 999999|FAKE0001 directly.")
        report = extract_report(text, facts, bronze, vocabulary=vocab)
        context = CheckContext(facts=facts, vocabulary=vocab)
        entities = [c for c in report.claims if c.claim_type is ClaimType.ENTITY]
        fabricated = [c for c in entities if c.value == "999999|FAKE0001"]
        assert fabricated
        result = check_claim(fabricated[0], context)
        assert result.verdict is not Verdict.SUPPORTED
        assert result.hallucination_class == "H1"

    def test_a_risk_descriptor_the_rewrite_added_is_adjudicated(self, fixture, vocab):
        """A phrase with no digits that aligns to no slot, and still makes a quantitative
        assertion the binding table can decide."""
        facts, bronze = fixture
        phrase = next(iter(vocab.risk_descriptors.values())).phrase_variants[0]
        text = canonicalise_narrative(bronze.text + f" The activity shows {phrase}.")
        report = extract_report(text, facts, bronze, vocabulary=vocab)
        assert any(c.claim_type is ClaimType.QUALITATIVE for c in report.claims)

    def test_structural_numerals_are_not_charged_as_additions(self, fixture, vocab):
        facts, bronze = fixture
        text = canonicalise_narrative(bronze.text + " See the 4 sections above.")
        report = extract_report(text, facts, bronze, vocabulary=vocab)
        assert "4" not in {t for _, _, t in report.added_spans}


class TestDrops:
    def test_a_dropped_value_is_reported_as_a_drop_not_an_addition(self, fixture, vocab):
        facts, bronze = fixture
        # A value that appears exactly once. A repeated value (the focal account id, which
        # is also the fan-out hub) legitimately re-aligns to its other occurrence, and the
        # slot reported dropped is then the second one -- correct, but not what this test
        # is about.
        target = next(
            slot
            for slot in bronze.slots
            if bronze.text.count(slot.rendered_value) == 1 and len(slot.rendered_value) > 2
        )
        text = canonicalise_narrative(bronze.text.replace(target.rendered_value, "", 1))
        report = extract_report(text, facts, bronze, vocabulary=vocab)
        assert target.field_path in report.dropped_paths

    def test_a_repeated_value_aligns_occurrence_by_occurrence(self, fixture, vocab):
        """Two slots that render the same string must produce two claims, not one claim
        and one spurious candidate addition."""
        facts, bronze = fixture
        repeated = [
            slot.rendered_value
            for slot in bronze.slots
            if bronze.text.count(slot.rendered_value) > 1
        ]
        if not repeated:
            pytest.skip("this fixture renders no value twice")
        report = extract_report(
            canonicalise_narrative(bronze.text), facts, bronze, vocabulary=vocab
        )
        assert report.dropped_paths == ()
        assert report.n_added == 0

    def test_an_empty_rewrite_drops_everything_and_invents_nothing(self, fixture, vocab):
        facts, bronze = fixture
        report = extract_report(
            "Nothing of substance was written here.", facts, bronze, vocabulary=vocab
        )
        assert report.aligned_paths == ()
        assert report.n_added == 0


class TestCanonicalisation:
    def test_paragraph_structure_survives(self):
        text = "First   section  here.\n\n\n  Second section.  \n\nThird."
        assert canonicalise_narrative(text) == "First section here.\n\nSecond section.\n\nThird."

    def test_a_hard_wrapped_value_still_aligns(self, fixture, vocab):
        """Without canonicalisation a model that wraps a line mid-value produces a value
        that aligns to nothing, and the correct figure is scored as both a dropped fact
        and an invented one."""
        facts, bronze = fixture
        wrapped = bronze.text.replace(" ", "\n", 40)
        report = extract_report(canonicalise_narrative(wrapped), facts, bronze, vocabulary=vocab)
        assert report.dropped_paths == ()

    def test_canonicalisation_is_idempotent(self, fixture):
        _, bronze = fixture
        once = canonicalise_narrative(bronze.text)
        assert canonicalise_narrative(once) == once


class TestProtocol:
    def test_the_extractor_satisfies_the_phase_10_interface(self, fixture, vocab):
        facts, bronze = fixture
        extractor = SlotAlignmentExtractor(bronze, vocabulary=vocab)
        claims = extractor.extract(canonicalise_narrative(bronze.text), facts)
        assert isinstance(claims, list)
        assert claims
