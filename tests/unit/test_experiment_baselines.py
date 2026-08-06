"""Phase 11: the baselines are competitors, and these tests are what makes that checkable.

Deliberately weakening a baseline is research misconduct. The properties below are the
mechanical part of not doing it:

- the baseline prompt renders with the SAME guidance blocks our own systems get, and
  refuses to send if any of them is empty;
- B4's exemplars come from the train split, and a non-train pool raises;
- B5's agentic loop really iterates -- it verifies, it repairs, it stops when clean, it
  respects its round budget, and a malformed verification is recorded as a parse failure
  rather than retried into a better-looking result.

The whole agentic loop runs through ScriptedTeacher: no network, no credentials, and the
loop under test is the loop that runs in production.
"""

from __future__ import annotations

import pytest
from tests import factories

from g2t_aml.corpus.silver.api_client import ScriptedTeacher, TeacherSpec, TransientAPIError
from g2t_aml.experiments.baselines import (
    BaselineError,
    Exemplar,
    assert_baseline_not_starved,
    assert_prompts_loadable,
    generate_agentic,
    generate_few_shot,
    generate_zero_shot,
    parse_verification,
    render_classifier_template_baseline,
    render_template_baseline,
    select_exemplars,
)
from g2t_aml.facts.extractor import extract_facts
from g2t_aml.facts.vocab import load_vocabulary


@pytest.fixture(scope="module")
def vocab():
    return load_vocabulary()


@pytest.fixture(scope="module")
def facts():
    return extract_facts(factories.fan_out_case(width=6))


@pytest.fixture
def spec():
    return TeacherSpec(
        key="baseline",
        family="frontier",
        provider="anthropic",
        model="claude-opus-5",
        supports_sampling=False,
    )


def _teacher(spec, script):
    return ScriptedTeacher(spec, script)


# ------------------------------------------------------------------- the prompts ---


def test_all_three_baseline_prompts_load_and_hash():
    hashes = assert_prompts_loadable()
    assert set(hashes) == {
        "baseline_generate_v1",
        "baseline_verify_v1",
        "baseline_repair_v1",
    }
    assert all(len(h) == 64 for h in hashes.values())


def test_starved_prompt_is_refused():
    """An empty forbidden list is a strawman produced by accident rather than by intent."""
    with pytest.raises(BaselineError, match="empty"):
        assert_baseline_not_starved(
            {"fact_record": "something", "forbidden_block": "   ", "hedging_block": "x"}
        )


def test_zero_shot_prompt_carries_every_guidance_block(facts, vocab, spec):
    seen = {}

    def _script(prompt, case_id, kind, attempt):
        seen["system"] = prompt.system
        seen["user"] = prompt.user
        return "A narrative."

    generate_zero_shot(facts, _teacher(spec, _script), vocabulary=vocab)
    combined = seen["system"] + seen["user"]
    # The same instructions our own arms receive, asserted rather than trusted.
    assert facts.case_id in combined
    for phrase in vocab.hedging_allowed[:2]:
        assert phrase in combined.lower()
    assert "FACT FAMILIES UNAVAILABLE" in combined
    assert "FIELDS THAT MUST APPEAR" in combined
    assert "{" not in seen["user"].replace("{{", ""), "an unsubstituted placeholder leaked"


def test_the_system_message_is_case_invariant(facts, vocab, spec):
    """It is the prompt-cache prefix across thousands of calls; a per-case value in it
    would silently turn every request into a cache write."""
    systems = []

    def _script(prompt, case_id, kind, attempt):
        systems.append(prompt.system_hash)
        return "A narrative."

    teacher = _teacher(spec, _script)
    generate_zero_shot(facts, teacher, vocabulary=vocab)
    generate_zero_shot(extract_facts(factories.fan_out_case(width=9)), teacher, vocabulary=vocab)
    assert len(set(systems)) == 1


def test_zero_shot_records_its_provenance(facts, vocab, spec):
    output = generate_zero_shot(
        facts, _teacher(spec, lambda *a: "Narrative text."), vocabulary=vocab
    )
    assert output.system == "B3"
    assert output.model == "claude-opus-5"
    assert output.prompt_name == "baseline_generate_v1"
    assert output.prompt_hash and output.rendered_hash
    assert output.narrative == "Narrative text."
    assert output.n_exemplars == 0


def test_a_failed_call_raises_a_baseline_error(facts, vocab, spec):
    def _script(*_args):
        raise TransientAPIError("503")

    with pytest.raises(BaselineError, match="call failed"):
        generate_zero_shot(facts, _teacher(spec, _script), vocabulary=vocab)


# ------------------------------------------------------------------ the exemplars ---


def _pool(n=8, split="train"):
    return [
        Exemplar(
            case_id=f"case{i}",
            split=split,
            typology="fan_out" if i % 2 == 0 else "cycle",
            serialised_facts=f"record {i}",
            narrative=f"narrative {i}",
        )
        for i in range(n)
    ]


def test_a_non_train_exemplar_pool_raises():
    """A test-split exemplar would FLATTER this baseline, so the check is mechanical."""
    with pytest.raises(BaselineError, match="leak"):
        select_exemplars("target", _pool(split="test"))


def test_a_mixed_pool_names_the_leaking_cases():
    pool = [*_pool(), Exemplar("leaky", "test", "fan_out", "r", "n")]
    with pytest.raises(BaselineError, match="leaky"):
        select_exemplars("target", pool)


def test_exemplar_selection_is_deterministic():
    pool = _pool(12)
    first = select_exemplars("case-abc", pool, k=5)
    second = select_exemplars("case-abc", list(reversed(pool)), k=5)
    assert [e.case_id for e in first] == [e.case_id for e in second]
    assert len(first) == 5


def test_exemplar_selection_prefers_the_matching_typology():
    pool = _pool(12)
    pool.append(Exemplar("target", "train", "cycle", "r", "n"))
    chosen = select_exemplars("target", pool, k=3)
    assert all(e.typology == "cycle" for e in chosen)


def test_a_case_is_never_its_own_exemplar():
    pool = _pool(6)
    assert "case2" not in [e.case_id for e in select_exemplars("case2", pool, k=5)]


def test_few_shot_puts_the_exemplars_in_the_prompt(facts, vocab, spec):
    captured = {}

    def _script(prompt, *_args):
        captured["user"] = prompt.user
        return "A narrative."

    output = generate_few_shot(facts, _teacher(spec, _script), _pool(6), k=3, vocabulary=vocab)
    assert output.n_exemplars == 3
    assert len(output.exemplar_case_ids) == 3
    assert "WORKED EXAMPLES" in captured["user"]
    assert captured["user"].count("--- EXAMPLE") == 3
    # The exemplars precede the target case, so the model reads the pattern first. The
    # target's own record is the LAST "FACT RECORD" heading; the earlier ones are the
    # exemplars' own.
    assert captured["user"].index("END OF EXAMPLES") < captured["user"].rindex("FACT RECORD")


# ---------------------------------------------------------- the verification parser ---


def test_clean_verdict_parses():
    violations, parsed = parse_verification("VERDICT: CLEAN")
    assert parsed and violations == ()


def test_violations_parse_with_their_type_span_and_correction():
    violations, parsed = parse_verification(
        "VERDICT: VIOLATIONS\n"
        "- NUMBER | fourteen counterparties | the record says nine\n"
        "- REGULATION | the USD 42,000 threshold | not in the whitelist\n"
    )
    assert parsed
    assert [v.kind for v in violations] == ["NUMBER", "REGULATION"]
    assert violations[0].span == "fourteen counterparties"
    assert violations[1].correction == "not in the whitelist"


def test_an_unknown_violation_type_is_dropped():
    violations, parsed = parse_verification(
        "VERDICT: VIOLATIONS\n- VIBES | something | else\n- ENTITY | acct | absent\n"
    )
    assert parsed
    assert [v.kind for v in violations] == ["ENTITY"]


def test_a_malformed_response_does_not_parse():
    _violations, parsed = parse_verification("I think it looks fine to me!")
    assert not parsed


def test_a_violations_verdict_with_no_lines_does_not_parse():
    """Treating it as clean would convert every formatting failure into a pass and
    inflate the baseline's self-reported convergence."""
    violations, parsed = parse_verification("VERDICT: VIOLATIONS\n")
    assert not parsed
    assert violations == ()


# ------------------------------------------------------------------ the B5 loop ---


def test_agentic_loop_converges_and_records_its_calls(facts, vocab, spec):
    def _script(prompt, case_id, kind, attempt):
        if kind == "baseline_verify":
            return "VERDICT: CLEAN"
        return "The initial draft."

    output = generate_agentic(facts, _teacher(spec, _script), vocabulary=vocab)
    assert output.narrative == "The initial draft."
    assert output.trace is not None
    assert output.trace.converged is True
    assert output.trace.rounds == 0
    assert output.trace.n_calls == 2


def test_agentic_loop_repairs_then_converges(facts, vocab, spec):
    calls = []

    def _script(prompt, case_id, kind, attempt):
        calls.append(kind)
        if kind == "baseline_verify":
            if calls.count("baseline_verify") == 1:
                return "VERDICT: VIOLATIONS\n- NUMBER | fourteen | the record says nine\n"
            return "VERDICT: CLEAN"
        if kind == "baseline_repair":
            return "The repaired draft."
        return "The initial draft."

    output = generate_agentic(facts, _teacher(spec, _script), vocabulary=vocab)
    assert output.narrative == "The repaired draft."
    assert output.trace.rounds == 1
    assert output.trace.converged is True
    assert output.trace.n_calls == 4
    assert len(output.trace.violations_per_round) == 2


def test_the_repair_prompt_names_the_violations(facts, vocab, spec):
    seen = {}

    def _script(prompt, case_id, kind, attempt):
        if kind == "baseline_verify":
            return "VERDICT: VIOLATIONS\n- NUMBER | fourteen | the record says nine\n"
        if kind == "baseline_repair":
            seen["user"] = prompt.user
            return "Repaired."
        return "Draft."

    generate_agentic(facts, _teacher(spec, _script), max_rounds=1, vocabulary=vocab)
    assert "fourteen" in seen["user"]
    assert "the record says nine" in seen["user"]
    assert "NUMBER" in seen["user"]


def test_the_verifier_is_given_the_regulatory_whitelist(facts, vocab, spec):
    """A verifier not told the whitelist cannot catch an invented citation -- and H6 is
    the Critical class the paper leans on hardest."""
    seen = {}

    def _script(prompt, case_id, kind, attempt):
        if kind == "baseline_verify":
            seen["user"] = prompt.user
            return "VERDICT: CLEAN"
        return "Draft."

    generate_agentic(facts, _teacher(spec, _script), vocabulary=vocab)
    assert "PERMITTED REGULATORY WORDING" in seen["user"]
    some_citation = next(iter(vocab.regulatory.values())).citation
    assert some_citation in seen["user"]


def test_the_round_budget_is_respected(facts, vocab, spec):
    def _never_clean(prompt, case_id, kind, attempt):
        if kind == "baseline_verify":
            return "VERDICT: VIOLATIONS\n- NUMBER | x | y\n"
        return "Another draft."

    output = generate_agentic(facts, _teacher(spec, _never_clean), max_rounds=3, vocabulary=vocab)
    assert output.trace.rounds == 3
    assert output.trace.converged is False
    assert output.trace.n_calls == 7


def test_a_parse_failure_stops_the_loop_and_is_recorded(facts, vocab, spec):
    """Retrying until the format is right is a loop that selects for whichever answer
    happens to parse."""

    def _script(prompt, case_id, kind, attempt):
        if kind == "baseline_verify":
            return "Looks good to me."
        return "Draft."

    output = generate_agentic(facts, _teacher(spec, _script), vocabulary=vocab)
    assert output.trace.parse_failures == 1
    assert output.trace.converged is False
    assert output.trace.rounds == 0


def test_a_failed_verification_call_keeps_the_draft(facts, vocab, spec):
    """Discarding the case would silently shrink B5's test set relative to every other
    system's."""

    def _script(prompt, case_id, kind, attempt):
        if kind == "baseline_verify":
            raise TransientAPIError("503")
        return "The only draft."

    output = generate_agentic(facts, _teacher(spec, _script), vocabulary=vocab)
    assert output.narrative == "The only draft."
    assert output.trace.converged is False


def test_the_final_narrative_is_the_last_one_not_the_best_scoring(facts, vocab, spec):
    """Selecting the baseline's output with OUR checker is the advantage B5 is not given."""

    def _script(prompt, case_id, kind, attempt):
        if kind == "baseline_verify":
            return "VERDICT: VIOLATIONS\n- NUMBER | x | y\n"
        if kind == "baseline_repair":
            return "The worse repaired draft."
        return "The better initial draft."

    output = generate_agentic(facts, _teacher(spec, _script), max_rounds=1, vocabulary=vocab)
    assert output.narrative == "The worse repaired draft."


def test_b5_starts_from_the_few_shot_draft_when_a_pool_is_supplied(facts, vocab, spec):
    """A competitor is entitled to its best configuration."""
    seen = {}

    def _script(prompt, case_id, kind, attempt):
        if kind == "baseline":
            seen["user"] = prompt.user
            return "Draft."
        return "VERDICT: CLEAN"

    output = generate_agentic(facts, _teacher(spec, _script), pool=_pool(6), k=4, vocabulary=vocab)
    assert "WORKED EXAMPLES" in seen["user"]
    assert output.n_exemplars == 4


def test_the_trace_serialises_its_compute_advantage(facts, vocab, spec):
    def _script(prompt, case_id, kind, attempt):
        if kind == "baseline_verify":
            return "VERDICT: CLEAN"
        return "Draft."

    payload = generate_agentic(facts, _teacher(spec, _script), vocabulary=vocab).to_dict()
    assert payload["agentic_trace"]["n_calls"] == 2
    assert payload["agentic_trace"]["converged"] is True
    assert "n_self_reported_violations" in payload["agentic_trace"]


# ------------------------------------------------------------------- the CPU arms ---


def test_b1_reads_bronze_rather_than_re_rendering_it():
    rows = render_template_baseline(["a", "b"], {"a": "A text", "b": "B text"})
    assert [r["narrative"] for r in rows] == ["A text", "B text"]
    assert all(r["system"] == "B1" for r in rows)
    assert all(r["generator"] == "bronze-template" for r in rows)


def test_b1_refuses_a_case_with_no_bronze_narrative():
    with pytest.raises(BaselineError, match="no Bronze narrative"):
        render_template_baseline(["a", "missing"], {"a": "A text"})


def test_b2_states_the_classifier_score_and_its_call():
    rows = render_classifier_template_baseline(
        ["a", "b"], {"a": "A.", "b": "B."}, {"a": 0.9, "b": 0.1}
    )
    assert "0.90" in rows[0]["narrative"] and "suspicious" in rows[0]["narrative"]
    assert "not suspicious" in rows[1]["narrative"]
    assert rows[0]["flagged"] is True
    assert rows[1]["flagged"] is False


def test_b2_refuses_a_case_with_no_prediction():
    with pytest.raises(BaselineError, match="prediction"):
        render_classifier_template_baseline(["a"], {"a": "A."}, {})
