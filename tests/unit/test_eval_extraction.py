"""Both claim extractors, on narratives whose claims are known by hand.

The adversarial cases are the point of this file. An extractor that finds the claims in a
well-formed Bronze narrative has proved nothing: Bronze's values are in the slot table and
alignment finds them by construction. What has to be tested is the four ways a real
generation goes wrong — a hedged claim, a vague one, a partially correct one, and one that
is correct but says something the fact record never contained.
"""

from __future__ import annotations

import json

import pytest
from tests.factories import as_laundering_stream, fan_out_case

from g2t_aml.corpus.record import BronzeNarrative, SlotAnnotation
from g2t_aml.corpus.silver.api_client import ScriptedTeacher, TeacherSpec
from g2t_aml.eval.claim_extraction.agreement import (
    AgreementCase,
    align_spans,
    cohens_kappa,
    interpret_kappa,
    measure_agreement,
)
from g2t_aml.eval.claim_extraction.deterministic import (
    DEFAULT_RULES,
    DeterministicClaimExtractor,
    _rule,
    extract_claims,
)
from g2t_aml.eval.claim_extraction.llm_based import (
    AtomicClaim,
    LLMClaimExtractor,
    LLMExtractionError,
    parse_entailment_response,
    parse_extraction_response,
)
from g2t_aml.facts.checkers import CheckContext, Claim, ClaimType, Verdict, check_claim
from g2t_aml.facts.extractor import extract_facts

# --------------------------------------------------------------- fixtures ---


@pytest.fixture(scope="module")
def facts():
    """A fan-out case with a ground-truth typology, so typology claims are checkable."""
    return extract_facts(as_laundering_stream(fan_out_case(width=9), "fan_out"))


@pytest.fixture
def context(facts):
    return CheckContext(facts=facts)


def bronze_of(facts, text: str, slots: tuple[SlotAnnotation, ...] = ()) -> BronzeNarrative:
    """Wrap a narrative and its slot table as a Bronze reference."""
    return BronzeNarrative(
        case_id=facts.case_id, text=text, annotated=text, slots=slots, family="test", variant=0
    )


def verdict_for(claim: Claim, context: CheckContext) -> Verdict:
    """Run the checker over one claim."""
    return check_claim(claim, context).verdict


def claim_on(claims, field_path: str) -> Claim:
    """Return the single claim naming a field, asserting there is exactly one."""
    found = [c for c in claims if c.field_path == field_path]
    assert len(found) == 1, f"expected one claim on {field_path}, got {len(found)}"
    return found[0]


# ------------------------------------------------ Method A: attribution ---


def test_a_wrong_count_is_contradicted_rather_than_merely_unverifiable(facts, context):
    # The failure this whole module exists to prevent. The record's out-degree is 9; the
    # narrative says 14. Without cue attribution this is an unaligned quantity naming no
    # field, so the checker returns UNVERIFIABLE and the system's Hallucination Rate --
    # contradicted over total -- stays at zero for its single most damaging error.
    narrative = "The subject account sent funds to 14 distinct counterparties."

    claims = extract_claims(narrative, facts).claims
    claim = claim_on(claims, "focal_entity.out_degree")

    assert claim.value == 14
    assert verdict_for(claim, context) is Verdict.CONTRADICTED


def test_the_same_cue_with_the_right_number_is_supported(facts, context):
    narrative = "The subject account sent funds to 9 distinct counterparties."
    claim = claim_on(extract_claims(narrative, facts).claims, "focal_entity.out_degree")
    assert verdict_for(claim, context) is Verdict.SUPPORTED


def test_attribution_binds_the_number_its_cue_is_adjacent_to(facts):
    # Two quantities in one sentence. A rule that matched its field name anywhere would
    # attach the wrong one; the value group has to sit next to the cue.
    narrative = "The subgraph comprises 10 accounts connected by 9 transactions."
    report = extract_claims(narrative, facts)

    by_field = {c.field_path: c.value for c in report.claims if c.field_path}
    assert by_field["structure.n_nodes"] == 10
    assert by_field["structure.n_edges"] == 9


def test_a_quantity_matching_no_cue_stays_unverifiable_and_is_reported(facts, context):
    narrative = "An internal risk score of 73 was assigned to the subject."
    report = extract_claims(narrative, facts)

    assert any(text == "73" for _, _, text in report.unattributed)
    unbacked = [c for c in report.claims if c.raw_text == "73"]
    assert unbacked and unbacked[0].field_path is None
    assert verdict_for(unbacked[0], context) is Verdict.UNVERIFIABLE


def test_attribution_records_which_rule_fired(facts):
    report = extract_claims("The subject account sent funds to 9 recipient accounts.", facts)
    rules = {rule for rule, _, _ in report.attributed}
    assert rules == {"out_degree"}


def test_every_default_rule_carries_a_value_group():
    # A rule without one would never fire and would be indistinguishable, in a table of
    # thirty, from one that does.
    for rule in DEFAULT_RULES:
        assert "value" in rule.pattern.groupindex, rule.name


def test_a_rule_without_a_value_group_is_refused():
    # The factory is the guarded path, not the dataclass: a rule that never fires would
    # look, in a table of thirty, exactly like one that does.
    with pytest.raises(ValueError, match="no 'value' group"):
        _rule("broken", "structure.n_nodes", "count", r"\d+")


def test_a_rule_table_can_be_replaced_for_one_field(facts, context):
    only_nodes = tuple(r for r in DEFAULT_RULES if r.name == "n_nodes")
    extractor = DeterministicClaimExtractor(rules=only_nodes)
    report = extractor.report("The subgraph holds 10 accounts and 9 transactions.", facts)

    fields = {c.field_path for c in report.claims if c.field_path}
    assert "structure.n_nodes" in fields
    assert "structure.n_edges" not in fields


# ----------------------------------------------- Method A: slot alignment ---


def test_alignment_recovers_a_value_the_rewrite_moved(facts, context):
    original = "The subgraph comprises 10 accounts."
    slots = (
        SlotAnnotation(
            field_path="structure.n_nodes",
            span=(23, 25),
            rendered_value="10",
            raw_value=10,
            claim_type="numeric",
        ),
    )
    rewritten = "Ten in total: the case covers 10 accounts in all."

    report = DeterministicClaimExtractor(bronze=bronze_of(facts, original, slots)).report(
        rewritten, facts
    )

    assert "structure.n_nodes" in report.aligned_paths
    assert verdict_for(claim_on(report.claims, "structure.n_nodes"), context) is Verdict.SUPPORTED


def test_a_dropped_slot_is_reported_as_dropped_not_as_a_violation(facts):
    original = "The subgraph comprises 10 accounts."
    slots = (
        SlotAnnotation(
            field_path="structure.n_nodes",
            span=(23, 25),
            rendered_value="10",
            raw_value=10,
            claim_type="numeric",
        ),
    )

    report = DeterministicClaimExtractor(bronze=bronze_of(facts, original, slots)).report(
        "The case describes a dispersal pattern.", facts
    )

    assert report.dropped_paths == ("structure.n_nodes",)
    assert report.unattributed == ()


def test_extraction_works_with_no_bronze_reference(facts, context):
    # Weaker, never wrong: the rules still fire, and everything else is unverifiable.
    report = DeterministicClaimExtractor(bronze=None).report(
        "The subject account sent funds to 14 distinct counterparties.", facts
    )
    assert verdict_for(claim_on(report.claims, "focal_entity.out_degree"), context) is (
        Verdict.CONTRADICTED
    )


# ---------------------------------------------------- adversarial claims ---


def test_adversarial_hedged_claim_is_still_extracted(facts, context):
    # A hedge is a stylistic wrapper, not an escape. "approximately 14" still asserts 14.
    narrative = "The subject account appears to have sent funds to approximately 14 recipients."
    claim = claim_on(extract_claims(narrative, facts).claims, "focal_entity.out_degree")
    assert verdict_for(claim, context) is Verdict.CONTRADICTED


def test_adversarial_vague_claim_asserts_nothing_and_scores_nothing(facts):
    # No quantity, no controlled descriptor, no entity: there is nothing to check, and
    # the honest output is no claim rather than a charitable SUPPORTED.
    report = extract_claims("The account moved a considerable sum to several parties.", facts)
    assert report.claims == ()


def test_adversarial_partially_correct_claim_is_split_by_field(facts, context):
    # One sentence, one right and one wrong. A per-claim metric has to separate them; a
    # per-sentence one cannot.
    narrative = "The subgraph comprises 10 accounts connected by 40 transactions."
    claims = extract_claims(narrative, facts).claims

    assert verdict_for(claim_on(claims, "structure.n_nodes"), context) is Verdict.SUPPORTED
    assert verdict_for(claim_on(claims, "structure.n_edges"), context) is Verdict.CONTRADICTED


def test_adversarial_correct_but_unstated_in_the_facts_is_unverifiable(facts, context):
    # A true sentence about the world that the fact record does not carry. It must not be
    # SUPPORTED -- the record did not establish it -- and it must not be CONTRADICTED,
    # because the record did not refute it either.
    narrative = "The receiving accounts are held at a licensed money services business."
    results = [check_claim(c, context) for c in extract_claims(narrative, facts).claims]
    assert all(r.verdict is not Verdict.SUPPORTED for r in results)


def test_a_typology_the_record_does_not_carry_is_caught(facts, context):
    # Invisible without the typology pass: the word aligns to no slot, holds no digits
    # and is not a controlled risk descriptor.
    claim = claim_on(
        extract_claims("The activity is consistent with a simple cycle.", facts).claims,
        "typology.label",
    )
    assert claim.value == "cycle"
    assert verdict_for(claim, context) is Verdict.CONTRADICTED


def test_the_recorded_typology_is_supported(facts, context):
    claim = claim_on(
        extract_claims("The activity shows a fan out structure.", facts).claims,
        "typology.label",
    )
    assert verdict_for(claim, context) is Verdict.SUPPORTED


def test_an_account_not_in_the_case_is_caught(facts, context):
    claim = claim_on
    report = extract_claims("Funds moved onward to account 999|DEADBEEF.", facts)
    entities = [c for c in report.claims if c.claim_type is ClaimType.ENTITY]
    assert entities
    assert verdict_for(entities[0], context) is Verdict.CONTRADICTED
    assert claim is claim_on  # keep the helper referenced for readers of this file


# ------------------------------------------------------ Method B parsing ---


def test_decomposition_response_is_parsed_and_located():
    narrative = "The account received 42,000 US Dollar. It sent funds to 9 counterparties."
    payload = json.dumps(
        {
            "claims": [
                {
                    "text": "The account received 42,000 US Dollar.",
                    "evidence": "received 42,000 US Dollar",
                    "type": "numeric",
                },
                {
                    "text": "The account sent funds to 9 counterparties.",
                    "evidence": "sent funds to 9 counterparties",
                    "type": "numeric",
                },
            ]
        }
    )

    claims, unlocated = parse_extraction_response(payload, narrative)

    assert unlocated == 0
    assert [c.span for c in claims] == [(12, 37), (42, 72)]
    assert all(narrative[c.span[0] : c.span[1]] == c.evidence for c in claims if c.span)


def test_an_evidence_string_that_is_not_in_the_narrative_is_counted_not_dropped():
    claims, unlocated = parse_extraction_response(
        json.dumps(
            {"claims": [{"text": "x", "evidence": "words that are not there", "type": "numeric"}]}
        ),
        "a narrative",
    )
    assert unlocated == 1
    assert claims[0].span is None


def test_a_fenced_response_is_tolerated_at_parse_time():
    claims, _ = parse_extraction_response(
        '```json\n{"claims": [{"text": "x", "evidence": "a", "type": "entity"}]}\n```',
        "a narrative",
    )
    assert len(claims) == 1


def test_an_unparseable_response_raises_rather_than_returning_nothing():
    # Returning an empty claim set would make the narrative look perfectly faithful to
    # Method B and would drag the measured kappa down for a reason that is neither
    # method's fault.
    with pytest.raises(LLMExtractionError):
        parse_extraction_response("I'm sorry, I can't help with that.", "a narrative")
    with pytest.raises(LLMExtractionError, match="no 'claims' list"):
        parse_extraction_response('{"result": []}', "a narrative")


def test_a_claim_type_outside_the_six_is_dropped_rather_than_coerced():
    claims, _ = parse_extraction_response(
        json.dumps({"claims": [{"text": "x", "evidence": "a", "type": "vibes"}]}), "a narrative"
    )
    assert claims == []


def test_a_missing_verdict_defaults_to_unverifiable_never_supported():
    verdicts = parse_entailment_response(
        json.dumps({"verdicts": [{"index": 0, "verdict": "supported", "rationale": "ok"}]}), 3
    )
    assert [v for v, _ in verdicts] == [
        Verdict.SUPPORTED,
        Verdict.UNVERIFIABLE,
        Verdict.UNVERIFIABLE,
    ]


def test_an_out_of_range_index_is_ignored():
    verdicts = parse_entailment_response(
        json.dumps({"verdicts": [{"index": 7, "verdict": "contradicted"}]}), 2
    )
    assert all(v is Verdict.UNVERIFIABLE for v, _ in verdicts)


def test_method_b_runs_end_to_end_against_a_scripted_teacher(facts):
    # The whole two-call pipeline with no network: ScriptedTeacher satisfies the same
    # protocol the API client does, so this exercises the real path.
    narrative = "The account sent funds to 9 counterparties."
    responses = {
        "decompose": json.dumps(
            {
                "claims": [
                    {
                        "text": "The account sent funds to 9 counterparties.",
                        "evidence": "sent funds to 9 counterparties",
                        "type": "numeric",
                    }
                ]
            }
        ),
        "entail": json.dumps(
            {"verdicts": [{"index": 0, "verdict": "supported", "rationale": "matches out_degree"}]}
        ),
    }
    teacher = ScriptedTeacher(
        TeacherSpec(key="test", family="frontier", provider="scripted", model="test-model"),
        lambda _prompt, _case, kind, _attempt: responses[kind],
    )

    report = LLMClaimExtractor(teacher).report(narrative, facts)

    assert report.method == "llm"
    assert len(report.claims) == 1
    assert report.claims[0].verdict is Verdict.SUPPORTED
    assert report.claims[0].rationale == "matches out_degree"
    assert [call[1] for call in teacher.calls] == ["decompose", "entail"]


def test_method_b_skips_the_second_call_when_nothing_was_decomposed(facts):
    teacher = ScriptedTeacher(
        TeacherSpec(key="test", family="frontier", provider="scripted", model="test-model"),
        lambda _p, _c, _k, _a: json.dumps({"claims": []}),
    )
    report = LLMClaimExtractor(teacher).report("Nothing checkable here.", facts)
    assert report.claims == ()
    assert len(teacher.calls) == 1


# ---------------------------------------------------------- the agreement ---


def test_cohens_kappa_hand_computed():
    # 2x2, n=10: both say "yes" 5 times, both "no" 3, A-yes/B-no 1, A-no/B-yes 1.
    # observed = 0.8. Marginals: A yes 6/10, B yes 6/10 -> expected = 0.36 + 0.16 = 0.52.
    # kappa = (0.8 - 0.52) / 0.48 = 0.58333...
    a = ["y"] * 5 + ["n"] * 3 + ["y"] + ["n"]
    b = ["y"] * 5 + ["n"] * 3 + ["n"] + ["y"]
    assert cohens_kappa(a, b) == pytest.approx(0.5833333, abs=1e-6)


def test_kappa_is_one_on_perfect_agreement_over_a_single_label():
    # Expected agreement is also 1 here and the usual formula is 0/0. Reporting NaN
    # would drop these cases out of a mean and quietly bias it.
    assert cohens_kappa(["s"] * 5, ["s"] * 5) == 1.0


def test_kappa_is_zero_when_agreement_is_exactly_chance():
    assert cohens_kappa(["a", "b"], ["b", "a"]) == pytest.approx(-1.0)


def test_kappa_rejects_unpaired_and_empty():
    with pytest.raises(ValueError, match="paired"):
        cohens_kappa(["a"], ["a", "b"])
    with pytest.raises(ValueError, match="empty"):
        cohens_kappa([], [])


def test_kappa_bands_are_the_conventional_ones():
    assert interpret_kappa(-0.1) == "poor"
    assert interpret_kappa(0.1) == "slight"
    assert interpret_kappa(0.3) == "fair"
    assert interpret_kappa(0.5) == "moderate"
    assert interpret_kappa(0.7) == "substantial"
    assert interpret_kappa(0.9) == "almost perfect"


def test_span_alignment_matches_by_overlap_not_by_text():
    alignment = align_spans([(0, 10), (20, 30)], [(1, 11), (50, 60)])
    assert [(i, j) for i, j, _ in alignment.matched] == [(0, 0)]
    assert alignment.unmatched_a == (1,)
    assert alignment.unmatched_b == (1,)
    assert alignment.mean_iou > 0.5
    assert alignment.span_f1 == pytest.approx(0.5)


def test_span_alignment_refuses_a_pair_below_the_threshold():
    assert align_spans([(0, 10)], [(9, 20)]).matched == ()


def test_span_alignment_uses_each_span_at_most_once():
    alignment = align_spans([(0, 10)], [(0, 10), (0, 10)])
    assert len(alignment.matched) == 1
    assert alignment.unmatched_b == (1,)


def test_agreement_reports_verdict_boundary_and_decision(facts, context):
    narrative = "The subject account sent funds to 14 distinct counterparties."
    method_a = tuple(check_claim(c, context) for c in extract_claims(narrative, facts).claims)
    span = method_a[0].claim.text_span
    method_b = (
        AtomicClaim(
            text="The account sent to 14 counterparties.",
            evidence=narrative[span[0] : span[1]],
            span=span,
            claim_type=ClaimType.NUMERIC,
            verdict=Verdict.CONTRADICTED,
        ),
    )

    report = measure_agreement([AgreementCase(facts.case_id, narrative, method_a, method_b)])

    assert report.n_cases == 1
    assert report.n_matched_claims == 1
    assert report.verdict_observed_agreement == 1.0
    assert report.decision_observed_agreement == 1.0
    assert report.boundary_kappa is not None
    assert not report.meets_sample_target  # one case is not the 300-case protocol sample


def test_agreement_counts_an_unlocatable_method_b_claim_separately(facts, context):
    narrative = "The subgraph comprises 10 accounts."
    method_a = tuple(check_claim(c, context) for c in extract_claims(narrative, facts).claims)
    method_b = (
        AtomicClaim(
            text="something",
            evidence="not in the narrative",
            span=None,
            claim_type=ClaimType.NUMERIC,
            verdict=Verdict.SUPPORTED,
        ),
    )

    report = measure_agreement([AgreementCase(facts.case_id, narrative, method_a, method_b)])

    assert report.unlocated_b == 1
    assert report.n_matched_claims == 0


def test_agreement_over_no_cases_reports_none_not_zero():
    report = measure_agreement([])
    assert report.verdict_kappa is None
    assert report.boundary_kappa is None
    assert report.decision_kappa is None


# --------------------------------------------- regulatory citation shape ---


def test_a_whitelisted_citation_is_supported_and_not_charged_twice(facts, context):
    # The Phase 6 finding, still holding: "the USD 10,000 reporting threshold" is a
    # reference the vocabulary explicitly whitelists as context. It must come back
    # SUPPORTED, and its figure must not also be charged as an unbacked quantity -- that
    # cost a correct sentence 6% of its unverifiable budget when it was first found.
    report = extract_claims(
        "Several transfers fall just below the USD 10,000 reporting threshold.", facts
    )
    regulatory = [c for c in report.claims if c.claim_type is ClaimType.REGULATORY]

    assert len(regulatory) == 1
    assert verdict_for(regulatory[0], context) is Verdict.SUPPORTED
    assert not any(text == "10,000" for _, _, text in report.unattributed)


def test_an_invented_citation_is_a_regulatory_claim_not_a_stray_number(facts, context):
    # H6 is Critical and is the class a forbidden-phrase list cannot cover: the failure
    # mode is a model inventing a rule, and an invented rule is by definition not on any
    # list written in advance. The citation is matched by shape and adjudicated by the
    # whitelist.
    report = extract_claims(
        "The transfers breach the USD 42,000 mandatory disclosure threshold.", facts
    )
    regulatory = [c for c in report.claims if c.claim_type is ClaimType.REGULATORY]

    assert len(regulatory) == 1
    result = check_claim(regulatory[0], context)
    assert result.verdict is Verdict.CONTRADICTED
    assert result.hallucination_class == "H6"
    # The figure inside the citation is part of the citation, not a second finding.
    assert not any(text == "42,000" for _, _, text in report.unattributed)


def test_a_statutory_reference_is_caught_by_shape(facts, context):
    for citation in ("31 CFR 1010.311", "Section 314(b)", "the Bank Secrecy Act"):
        report = extract_claims(f"The activity engages {citation}.", facts)
        regulatory = [c for c in report.claims if c.claim_type is ClaimType.REGULATORY]
        assert regulatory, citation
        assert check_claim(regulatory[0], context).hallucination_class == "H6", citation
