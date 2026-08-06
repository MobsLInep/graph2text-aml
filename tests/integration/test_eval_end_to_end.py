"""The harness end to end, and the gate that makes it trustworthy.

**Bronze scored against its own fact records must come out perfect.** Bronze renders from
the record and every formatter ships with its inverse, so any claim it makes is a claim
the record entails. A Bronze run that scores below 100% is a bug in the extractor, the
checker or the renderer — never a property of Bronze — and that makes it the one gate the
whole harness can be regression-tested against without a trained model, a Gold reference
or an API key.

It is not a circular test in the D-034 sense. The claims here are parsed back out of the
*rendered text*, never read from the record, so a wrong value would be stated wrongly and
caught. What it cannot catch is an extractor that finds no claims at all — which is
exactly why the claim count is asserted alongside the rate.
"""

from __future__ import annotations

import dataclasses

import pytest
from tests.factories import (
    as_laundering_stream,
    bipartite_case,
    chain_case,
    cycle_case,
    fan_in_case,
    fan_out_case,
    gather_scatter_case,
    scatter_gather_case,
    stack_case,
)

from g2t_aml.corpus.bronze.renderer import render_bronze
from g2t_aml.corpus.record import BronzeNarrative
from g2t_aml.eval.claim_extraction.deterministic import extract_claims
from g2t_aml.eval.layer2_faithfulness import aggregate_faithfulness, score_case
from g2t_aml.eval.report import evaluate
from g2t_aml.eval.types import ScoredCase, SystemOutput
from g2t_aml.facts.extractor import extract_facts

pytestmark = pytest.mark.integration

#: One case per typology the substrate carries, so the gate covers every template family
#: rather than the one that happens to be easiest.
CASE_BUILDERS = {
    "fan_out": lambda: fan_out_case(width=6),
    "fan_in": lambda: fan_in_case(width=6),
    "chain": lambda: chain_case(length=4),
    "cycle": lambda: cycle_case(length=4),
    "bipartite": lambda: bipartite_case(left=3, right=3),
    "gather_scatter": lambda: gather_scatter_case(gather=4, scatter=3),
    "scatter_gather": lambda: scatter_gather_case(width=4),
    "stack": lambda: stack_case(depth=3, layer_width=2),
}


def bronze_corpus(seed: int = 42):
    """Render one Bronze narrative per typology, with its facts and its slot alignment."""
    rendered = []
    for index, (typology, builder) in enumerate(sorted(CASE_BUILDERS.items())):
        case = builder()
        if typology in {"fan_out", "fan_in", "cycle", "bipartite", "stack"}:
            case = as_laundering_stream(case, typology)
        facts = dataclasses.replace(extract_facts(case), case_id=f"gate-{index:02d}")
        narrative = render_bronze(facts, seed=seed)
        rendered.append((facts, narrative))
    return rendered


@pytest.fixture(scope="module")
def corpus():
    return bronze_corpus()


def test_the_corpus_fixture_covers_every_template_family(corpus):
    assert len(corpus) == len(CASE_BUILDERS)
    assert len({narrative.family for _, narrative in corpus}) > 1


def test_bronze_against_bronze_is_perfectly_faithful(corpus):
    # The gate. Every claim Bronze makes is a claim its record entails.
    results = []
    for facts, narrative in corpus:
        case = ScoredCase(
            output=SystemOutput(
                system="bronze",
                case_id=facts.case_id,
                narrative=narrative.text,
                slots=narrative.slots,
            ),
            facts=facts,
        )
        claims = extract_claims(narrative.text, facts, bronze=narrative).claims
        results.append(score_case(case, claims))

    aggregate = aggregate_faithfulness(results)

    assert aggregate.zero_hallucination_rate == 1.0
    assert aggregate.hallucination_rate == 0.0
    assert aggregate.fact_precision == 1.0
    assert aggregate.unverifiable_rate == 0.0
    assert aggregate.critical_error_rate == 0.0
    # An extractor that found nothing would also score perfectly. It has to have found
    # something, and something per narrative.
    assert aggregate.n_claims > 0
    assert aggregate.n_narratives_with_no_claims == 0
    assert all(r.n_claims > 0 for r in results)


def test_the_gate_fails_when_one_value_is_corrupted(corpus):
    # The gate has to be able to fail, or it is not a gate. Corrupting one rendered digit
    # must be caught -- if it is not, the extractor is reading its claims from the record
    # rather than from the text, which is the D-034 circularity in its most dangerous
    # place (D-040).
    facts, narrative = corpus[0]
    corrupted = None
    for slot in narrative.slots:
        if slot.claim_type == "numeric" and slot.rendered_value.isdigit():
            start, end = slot.span
            replacement = str(int(slot.rendered_value) + 7)
            corrupted = narrative.text[:start] + replacement + narrative.text[end:]
            break

    assert corrupted is not None, "the fixture carries no numeric slot to corrupt"

    case = ScoredCase(
        output=SystemOutput(system="corrupted", case_id=facts.case_id, narrative=corrupted),
        facts=facts,
    )
    # Aligned against the *original* Bronze, which is what a real evaluation does: the
    # reference alignment is Bronze's, and the narrative being scored is the system's.
    claims = extract_claims(corrupted, facts, bronze=narrative).claims
    result = score_case(case, claims)

    assert not result.zero_hallucination or result.n_unverifiable > 0


def test_the_full_report_runs_on_bronze_and_writes_every_artifact(tmp_path, corpus):
    outputs = [
        SystemOutput(
            system="bronze",
            case_id=facts.case_id,
            narrative=narrative.text,
            slots=narrative.slots,
        )
        for facts, narrative in corpus
    ]
    facts_index = {facts.case_id: facts for facts, _ in corpus}
    bronze_index: dict[str, BronzeNarrative] = {
        facts.case_id: narrative for facts, narrative in corpus
    }

    report = evaluate(
        outputs,
        {},
        facts_index,
        bronze=bronze_index,
        run_id="gate",
        bertscore_model=None,
        n_resamples=200,
    )
    written = report.write_all(tmp_path / "metrics")

    assert report.systems[("bronze", "balanced")].faithfulness.zero_hallucination_rate == 1.0
    assert set(written) >= {"json", "markdown", "latex_balanced", "errors"}
    assert all(path.is_file() for path in written.values())

    markdown = report.to_markdown()
    assert "Zero-Hallucination" in markdown
    # No Gold exists, so Layer 1 must report named absences rather than zeros.
    assert "no case has a Gold reference" in markdown


def test_a_single_system_produces_no_comparisons(corpus):
    outputs = [
        SystemOutput(system="bronze", case_id=facts.case_id, narrative=narrative.text)
        for facts, narrative in corpus
    ]
    report = evaluate(
        outputs,
        {},
        {facts.case_id: facts for facts, _ in corpus},
        bertscore_model=None,
        n_resamples=50,
    )
    assert report.comparisons == {}
    assert report.template_finding is None


def test_layer1_runs_when_a_reference_exists(corpus):
    pytest.importorskip("sacrebleu")
    # Bronze scored against itself as its own reference: the overlap metrics must be at
    # their ceiling, which is the check that the Layer 1 plumbing is wired to the right
    # texts rather than the check of any system.
    facts_index = {facts.case_id: facts for facts, _ in corpus}
    references = {facts.case_id: narrative.text for facts, narrative in corpus}
    outputs = [
        SystemOutput(system="bronze", case_id=facts.case_id, narrative=narrative.text)
        for facts, narrative in corpus
    ]

    report = evaluate(outputs, references, facts_index, bertscore_model=None, n_resamples=50)
    layer1 = report.systems[("bronze", "balanced")].layer1

    assert layer1 is not None
    assert layer1.n_pairs == len(corpus)
    assert layer1.bleu == pytest.approx(100.0)
    assert layer1.rouge_l == pytest.approx(1.0)
    assert layer1.bleu_signature
