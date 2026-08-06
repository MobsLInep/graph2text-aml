"""The controlled vocabulary's condition grammar, and the salience definition."""

from __future__ import annotations

import pytest
from tests.factories import (
    as_laundering_stream,
    elliptic2_case,
    fan_out_case,
    gather_scatter_case,
)

from g2t_aml.facts.extractor import extract_facts
from g2t_aml.facts.salience import field_value, is_field_available, salience_report
from g2t_aml.facts.schema import Unavailable
from g2t_aml.facts.vocab import (
    VocabularyError,
    evaluate_condition,
    load_vocabulary,
    parse_condition,
)

# ------------------------------------------------------- condition grammar ---


@pytest.mark.parametrize(
    ("condition", "value", "expected"),
    [
        ("< 24", 23.9, True),
        ("< 24", 24.0, False),
        ("<= 6", 6.0, True),
        ("<= 6", 6.1, False),
        (">= 8", 8, True),
        (">= 8", 7, False),
        ("> 0", 0.1, True),
        ("== 3", 3, True),
        ("!= 3", 3, False),
        (">= 0.5", 0.5, True),
    ],
)
def test_condition_evaluation(condition, value, expected):
    assert evaluate_condition(condition, value) is expected


@pytest.mark.parametrize(
    "bad",
    [
        "between 1 and 5",
        "< 24 and > 3",
        "burst < 24",
        "__import__('os')",
        "",
        "<",
        "< abc",
    ],
)
def test_grammar_rejects_anything_beyond_an_operator_and_a_number(bad):
    # The grammar is deliberately too small for a bug to hide in. Anything richer belongs
    # in a named, unit-tested detector instead.
    with pytest.raises(VocabularyError):
        parse_condition(bad)


def test_condition_parse_returns_operator_and_threshold():
    assert parse_condition(">= 8") == (">=", 8.0)


# ----------------------------------------------------------- vocab loading ---


def test_vocabulary_loads_and_indexes_descriptors():
    vocab = load_vocabulary()
    assert "rapid_dispersal" in vocab.risk_descriptors
    assert vocab.risk_descriptors["rapid_dispersal"].binds_to == "temporal.burst_window_hours"


def test_phrase_lookup_resolves_a_surface_form_to_its_descriptor():
    vocab = load_vocabulary()
    descriptor = vocab.descriptor_for_phrase("dispersed within a short window")
    assert descriptor is not None
    assert descriptor.name == "rapid_dispersal"


def test_phrase_lookup_returns_none_for_an_uncontrolled_phrase():
    assert load_vocabulary().descriptor_for_phrase("extremely dodgy") is None


def test_hedging_lists_are_disjoint():
    vocab = load_vocabulary()
    allowed = set(vocab.hedging_allowed)
    for _, phrases in vocab.forbidden.values():
        assert not (allowed & set(phrases))


def test_forbidden_hit_finds_the_longest_matching_phrase():
    vocab = load_vocabulary()
    hit = vocab.forbidden_hit("The subject is guilty of money laundering.")
    assert hit is not None
    hallucination_class, _ = hit
    assert hallucination_class == "H7"


def test_forbidden_hit_returns_none_on_clean_text():
    assert load_vocabulary().forbidden_hit("Activity warrants further review.") is None


def test_version_mismatch_raises(tmp_path):
    load_vocabulary.cache_clear()
    bad = tmp_path / "vocab.yaml"
    bad.write_text("vocab_version: '0.0.1'\n")
    with pytest.raises(VocabularyError, match="version mismatch"):
        load_vocabulary(str(bad))
    load_vocabulary.cache_clear()


def test_missing_file_raises(tmp_path):
    load_vocabulary.cache_clear()
    with pytest.raises(FileNotFoundError):
        load_vocabulary(str(tmp_path / "absent.yaml"))
    load_vocabulary.cache_clear()


# ---------------------------------------------------------------- salience ---


def test_field_value_resolves_a_nested_motif_descriptor():
    facts = extract_facts(fan_out_case(width=6))
    assert field_value(facts, "motifs.fan_out.width") == 6


def test_field_value_returns_the_sentinel_when_an_ancestor_is_masked():
    facts = extract_facts(elliptic2_case())
    value = field_value(facts, "flow.total_inflow")
    assert isinstance(value, Unavailable)


def test_field_value_returns_none_for_a_broken_path():
    # None (a bug) and a sentinel (a fact) are deliberately different returns.
    facts = extract_facts(fan_out_case())
    assert field_value(facts, "structure.not_a_field") is None


def test_availability_excuses_a_salient_field_on_a_masked_substrate():
    # An Elliptic2 narrative is not penalised for omitting an amount that cannot exist.
    facts = extract_facts(elliptic2_case())
    report = salience_report(facts)
    assert any(p.startswith("flow.") for p in report.excused)
    assert not any(p.startswith("flow.") for p in report.required)


def test_salient_fields_are_required_when_the_substrate_supports_them():
    case = as_laundering_stream(gather_scatter_case(gather=4, scatter=3), "gather_scatter")
    facts = extract_facts(case)
    report = salience_report(facts)
    assert report.typology == "gather_scatter"
    assert "motifs.gather_scatter.gather_width" in report.required
    assert "focal_entity.id" in report.required  # from the _common list


def test_required_and_excused_together_are_the_declared_list():
    facts = extract_facts(fan_out_case(width=5))
    vocab = load_vocabulary()
    report = salience_report(facts)
    declared = vocab.salient_fields(facts.typology.label)
    assert set(report.required) | set(report.excused) == set(declared)
    assert not (set(report.required) & set(report.excused))


def test_unknown_typology_falls_back_to_the_unclassified_list():
    vocab = load_vocabulary()
    assert vocab.salient_fields("not_a_typology") == vocab.salient_fields("unclassified")


def test_absent_motif_descriptor_is_not_required():
    # A narrative cannot be required to mention a fan width that does not exist.
    facts = extract_facts(fan_out_case(width=2))  # below the floor, so width is None
    assert is_field_available(facts, "motifs.fan_out.width") is False


def test_coverage_denominator_counts_only_required_fields():
    report = salience_report(extract_facts(elliptic2_case()))
    assert report.coverage_denominator == len(report.required)
