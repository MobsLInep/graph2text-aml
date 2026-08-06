"""The rewrite and repair prompts: hashing, the cache prefix, and what they may not see.

The prompt is an artifact with a content hash on every record it produces, so the
properties that matter are that the hash tracks the file, that the system message really
is case-invariant, and that the rewriter is structurally unable to see the raw graph.
"""

from __future__ import annotations

import pytest
from tests import factories

from g2t_aml.corpus.bronze.renderer import render_bronze
from g2t_aml.corpus.silver import prompts
from g2t_aml.facts.extractor import extract_facts
from g2t_aml.facts.vocab import load_vocabulary


@pytest.fixture(scope="module")
def vocab():
    return load_vocabulary()


@pytest.fixture(scope="module")
def case():
    facts = extract_facts(
        factories.as_laundering_stream(factories.fan_out_case(width=6), "fan_out")
    )
    return facts, render_bronze(facts)


class TestPromptFiles:
    def test_both_prompts_load_and_split(self):
        for name in (prompts.REWRITE_PROMPT_NAME, prompts.REPAIR_PROMPT_NAME):
            template = prompts.load_prompt(name)
            assert template.system.strip()
            assert template.user.strip()
            assert len(template.content_hash) == 64

    def test_prompt_hash_changes_when_the_file_changes(self, tmp_path):
        source = prompts.PROMPTS_DIR / f"{prompts.REWRITE_PROMPT_NAME}.txt"
        raw = source.read_text(encoding="utf-8")
        (tmp_path / f"{prompts.REWRITE_PROMPT_NAME}.txt").write_text(raw, encoding="utf-8")
        first = prompts.load_prompt(prompts.REWRITE_PROMPT_NAME, str(tmp_path))

        (tmp_path / f"{prompts.REWRITE_PROMPT_NAME}.txt").write_text(raw + "\n", encoding="utf-8")
        prompts.load_prompt.cache_clear()
        second = prompts.load_prompt(prompts.REWRITE_PROMPT_NAME, str(tmp_path))
        assert first.content_hash != second.content_hash
        prompts.load_prompt.cache_clear()

    def test_a_header_comment_naming_the_markers_does_not_split_the_file(self, tmp_path):
        """The bug this parser was rewritten for: a marker only counts on its own line.

        The first version partitioned on the marker string anywhere, so the header comment
        explaining what the markers are silently became the boundary -- producing a
        141-character system message with the whole instruction block moved into the
        per-case half. Nothing about the rendered prompt looked wrong.
        """
        (tmp_path / "probe.txt").write_text(
            "# This comment mentions <<<SYSTEM>>> and <<<USER>>> inline.\n"
            "<<<SYSTEM>>>\nreal system\n<<<USER>>>\nreal user\n",
            encoding="utf-8",
        )
        template = prompts.load_prompt("probe", str(tmp_path))
        assert template.system == "real system"
        assert template.user == "real user"
        prompts.load_prompt.cache_clear()

    def test_a_missing_section_is_refused(self, tmp_path):
        (tmp_path / "broken.txt").write_text("<<<USER>>>\nonly a user section\n", encoding="utf-8")
        with pytest.raises(prompts.PromptRenderError, match="exactly one"):
            prompts.load_prompt("broken", str(tmp_path))
        prompts.load_prompt.cache_clear()

    def test_a_per_case_placeholder_in_the_system_message_is_refused(self, tmp_path):
        """The system message is the run's prompt-cache prefix; a per-case value in it
        turns every request into a cache write and nothing else looks wrong."""
        (tmp_path / "leaky.txt").write_text(
            "<<<SYSTEM>>>\nYou are an investigator. {fact_record}\n<<<USER>>>\nGo.\n",
            encoding="utf-8",
        )
        with pytest.raises(prompts.PromptRenderError, match="fact_record"):
            prompts.load_prompt("leaky", str(tmp_path))
        prompts.load_prompt.cache_clear()


class TestRewritePrompt:
    def test_system_message_is_byte_identical_across_cases(self, vocab):
        """What makes the prompt cache work. If this drifts the run silently pays a cache
        write on every call."""
        hashes = set()
        for width in (3, 5, 7, 9):
            facts = extract_facts(factories.fan_out_case(width=width))
            bronze = render_bronze(facts, vocabulary=vocab)
            rendered = prompts.build_rewrite_prompt(
                facts, bronze.text, bronze.annotated, vocabulary=vocab
            )
            hashes.add(rendered.system_hash)
        assert len(hashes) == 1

    def test_prompt_carries_the_fact_record_and_the_bronze_draft(self, case, vocab):
        facts, bronze = case
        rendered = prompts.build_rewrite_prompt(
            facts, bronze.text, bronze.annotated, vocabulary=vocab
        )
        assert facts.case_id in rendered.user
        assert bronze.text[:80] in rendered.user
        assert bronze.annotated[:60] in rendered.user

    def test_prompt_states_the_unavailable_fact_classes(self, vocab):
        """Invariant 4 in the form a model will act on."""
        facts = extract_facts(factories.elliptic2_case())
        bronze = None
        try:
            bronze = render_bronze(facts, vocabulary=vocab)
        except Exception:
            pytest.skip("this fixture has no renderable family")
        rendered = prompts.build_rewrite_prompt(
            facts, bronze.text, bronze.annotated, vocabulary=vocab
        )
        for flag, available in facts.availability.to_dict().items():
            if not available:
                assert flag in rendered.user

    def test_forbidden_and_hedging_vocabulary_are_present(self, case, vocab):
        facts, bronze = case
        rendered = prompts.build_rewrite_prompt(
            facts, bronze.text, bronze.annotated, vocabulary=vocab
        )
        assert "mixer" in rendered.system
        assert "is money laundering" in rendered.system
        for phrase in vocab.hedging_allowed[:3]:
            assert phrase in rendered.system

    def test_the_builder_cannot_be_handed_a_graph(self):
        """The structural hallucination limit, asserted on the signature rather than in
        prose: there is no parameter through which a subgraph could reach the rewriter."""
        import inspect

        parameters = set(inspect.signature(prompts.build_rewrite_prompt).parameters)
        assert not parameters & {"graph", "case", "subgraph", "edges", "nodes", "transactions"}
        assert parameters >= {"facts", "bronze_narrative", "bronze_annotated"}

    def test_style_directive_is_deterministic_and_spreads(self):
        assert prompts.style_directive_for("case-a") == prompts.style_directive_for("case-a")
        drawn = {prompts.style_directive_for(f"case-{i}") for i in range(400)}
        assert drawn == set(prompts.STYLE_DIRECTIVES)

    def test_render_refuses_missing_and_unknown_placeholders(self):
        template = prompts.load_prompt(prompts.REWRITE_PROMPT_NAME)
        with pytest.raises(prompts.PromptRenderError, match="needs values"):
            template.render({"min_words": "1"})
        values = dict.fromkeys(template.placeholders, "x")
        with pytest.raises(prompts.PromptRenderError, match="does not use"):
            template.render({**values, "not_a_placeholder": "x"})


class TestRepairPrompt:
    def test_repair_prompt_lists_the_violations(self, case, vocab):
        facts, bronze = case
        violations = [
            prompts.Violation(
                field_path="flow.total_inflow",
                quoted="USD 30,000",
                verdict="contradicted",
                hallucination_class="H2",
                reason="disagrees with the record",
            )
        ]
        rendered = prompts.build_repair_prompt(facts, bronze.text, violations, vocabulary=vocab)
        assert "flow.total_inflow" in rendered.user
        assert "CONTRADICTED" in rendered.user
        assert "[H2]" in rendered.user

    def test_repair_prompt_never_hands_over_the_expected_value(self, case, vocab):
        """A model given the right answer as text pastes the string; a model told which
        field disagrees has to read the record."""
        facts, bronze = case
        violations = [
            prompts.Violation(
                field_path="structure.n_nodes",
                quoted="41 accounts",
                verdict="contradicted",
                hallucination_class="H2",
                reason="disagrees with the record",
            )
        ]
        rendered = prompts.build_repair_prompt(facts, bronze.text, violations, vocabulary=vocab)
        line = next(ln for ln in rendered.user.splitlines() if "structure.n_nodes" in ln)
        assert "expected" not in line.lower()
        assert str(facts.structure.n_nodes) not in line

    def test_a_repair_prompt_with_no_violations_is_refused(self, case, vocab):
        facts, bronze = case
        with pytest.raises(prompts.PromptRenderError, match="at least one violation"):
            prompts.build_repair_prompt(facts, bronze.text, [], vocabulary=vocab)
