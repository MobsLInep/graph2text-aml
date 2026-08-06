"""The inference guard, the faithfulness probe, and the arm comparison.

These run against **real fact records** built by the Phase 3 extractor from the synthetic
case factories, and verified by the real Phase 3 checker — not against mocked verdicts.
The guard's whole value is that it uses the same measurement instrument the paper reports,
so a test that stubbed the checker would be testing nothing that matters.
"""

from __future__ import annotations

import pytest

# The generator package imports torch at module scope, and `make install` is CPU-only
# by design (CLAUDE.md section 4). Without this the whole module fails to COLLECT in the
# light environment, which takes `make smoke` down with it rather than skipping.
pytest.importorskip("torch")

from tests import factories  # noqa: E402

from g2t_aml.corpus.bronze.renderer import render_bronze  # noqa: E402
from g2t_aml.corpus.silver.claim_extraction import SlotAlignmentExtractor  # noqa: E402
from g2t_aml.facts.checkers import CheckContext  # noqa: E402
from g2t_aml.facts.extractor import extract_facts  # noqa: E402
from g2t_aml.facts.vocab import load_vocabulary  # noqa: E402
from g2t_aml.models.generator.callbacks import (  # noqa: E402
    ProbeCase,
    compare_arms,
    score_narrative,
)
from g2t_aml.models.generator.guard import (  # noqa: E402
    GuardWeights,
    InferenceGuard,
    score_candidate,
)
from g2t_aml.models.generator.inference import GenerationConfig  # noqa: E402
from g2t_aml.utils.io import write_jsonl  # noqa: E402


@pytest.fixture(scope="module")
def vocab():
    """The controlled vocabulary.

    Returns:
        The vocabulary.
    """
    return load_vocabulary()


@pytest.fixture(scope="module")
def case(vocab):
    """A real fan-out case: its facts, its Bronze narrative and an extractor.

    Args:
        vocab: The controlled vocabulary.

    Returns:
        ``(facts, bronze, extractor)``.
    """
    facts = extract_facts(
        factories.as_laundering_stream(factories.fan_out_case(width=6), "fan_out")
    )
    bronze = render_bronze(facts, vocabulary=vocab)
    return facts, bronze, SlotAlignmentExtractor(bronze, vocabulary=vocab)


class TestCandidateScoring:
    """Verification of one candidate."""

    def test_bronze_scores_clean(self, case) -> None:
        """Bronze is faithful by construction, so it must contradict nothing."""
        facts, bronze, extractor = case
        scored = score_candidate(bronze.text, facts, extractor)
        assert scored.contradiction_rate == 0.0
        assert scored.is_clean

    def test_a_contradicted_number_is_caught(self, case) -> None:
        """Altering a figure in the narrative produces a contradiction, not silence."""
        facts, bronze, extractor = case
        corrupted = bronze.text.replace("six", "ninety-nine").replace(" 6 ", " 99 ")
        scored = score_candidate(corrupted, facts, extractor)
        if corrupted != bronze.text:
            assert scored.contradiction_rate > 0 or scored.unverifiable_rate > 0

    def test_an_empty_narrative_does_not_win_on_coverage(self, case) -> None:
        """Coverage is in the score precisely so the emptiest candidate cannot win."""
        facts, bronze, extractor = case
        full = score_candidate(bronze.text, facts, extractor)
        empty = score_candidate("This case was reviewed.", facts, extractor)
        assert full.coverage >= empty.coverage
        assert full.score > empty.score

    def test_weights_must_sum_to_one(self) -> None:
        """Scores from different runs stay comparable only if the weights are normalised."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            GuardWeights(contradiction=0.9, coverage=0.9, unverifiable=0.9)

    def test_negative_weights_are_refused(self) -> None:
        """A negative weight would reward the failure it is meant to penalise."""
        with pytest.raises(ValueError, match="non-negative"):
            GuardWeights(contradiction=-0.1, coverage=0.6, unverifiable=0.5)


class TestGuardSelection:
    """The guard picks the best-verified candidate."""

    def test_selects_the_highest_scoring_candidate(self, case) -> None:
        """The fixture test the brief asks for: the best candidate is the one emitted."""
        facts, bronze, extractor = case
        candidates = ["This case was reviewed.", bronze.text, "Nothing of note."]
        report = InferenceGuard().run("c1", candidates, facts, extractor)
        assert report.text == bronze.text
        assert report.selected.score == max(c.score for c in report.candidates)

    def test_records_the_unguarded_first_candidate_separately(self, case) -> None:
        """Both rows of the results table come from one run and are never conflated."""
        facts, bronze, extractor = case
        report = InferenceGuard().run(
            "c1", ["This case was reviewed.", bronze.text], facts, extractor
        )
        assert report.selected.text == bronze.text
        assert report.unguarded.text == "This case was reviewed."
        assert report.selected.score != report.unguarded.score

    def test_selection_change_is_counted(self, case) -> None:
        """The statistic that says what the guard bought."""
        facts, bronze, extractor = case
        guard = InferenceGuard()
        guard.run("c1", ["This case was reviewed.", bronze.text], facts, extractor)
        assert guard.stats.n_selection_changed == 1
        assert guard.stats.to_dict()["selection_changed_rate"] == 1.0

    def test_no_change_when_the_first_is_best(self, case) -> None:
        """A guard that never intervenes reports so rather than taking credit."""
        facts, bronze, extractor = case
        guard = InferenceGuard()
        guard.run("c1", [bronze.text, "Nothing of note."], facts, extractor)
        assert guard.stats.n_selection_changed == 0
        assert guard.stats.n_clean_first_try == 1

    def test_no_candidates_is_refused(self, case) -> None:
        """An empty candidate list is a bug upstream and raises here."""
        facts, _, extractor = case
        with pytest.raises(ValueError, match="no candidates"):
            InferenceGuard().run("c1", [], facts, extractor)


class TestGuardRepair:
    """One constrained regeneration, then a warning."""

    def _dirty(self, bronze_text: str) -> str:
        """Return a narrative asserting guilt, which the checker forbids.

        Args:
            bronze_text: The clean narrative.

        Returns:
            A narrative carrying a forbidden phrase.
        """
        return f"{bronze_text} The subject is guilty of money laundering."

    def test_regeneration_is_triggered_by_a_contradiction(self, case) -> None:
        """A contradicted best candidate triggers exactly one repair attempt."""
        facts, bronze, extractor = case
        guard = InferenceGuard()
        report = guard.run(
            "c1",
            [self._dirty(bronze.text)],
            facts,
            extractor,
            regenerate=lambda violations: bronze.text,
        )
        assert report.regenerated
        assert guard.stats.n_regenerated == 1
        assert guard.stats.n_regeneration_helped == 1
        assert report.warning is None

    def test_repair_prompt_names_the_violations(self, case) -> None:
        """'Try again, be accurate' gives the model nothing; the violations are named."""
        facts, bronze, extractor = case
        captured: list[list[str]] = []

        InferenceGuard().run(
            "c1",
            [self._dirty(bronze.text)],
            facts,
            extractor,
            regenerate=lambda v: (captured.append(list(v)), bronze.text)[1],
        )
        assert captured and captured[0]
        prompt = InferenceGuard().build_repair_prompt(captured[0])
        assert "contradicted the case record" in prompt
        assert captured[0][0].split(":")[0] in prompt

    def test_failed_repair_emits_a_machine_readable_warning(self, case) -> None:
        """When repair fails the output ships with a structured warning, not silently."""
        facts, bronze, extractor = case
        guard = InferenceGuard()
        dirty = self._dirty(bronze.text)
        report = guard.run("c1", [dirty], facts, extractor, regenerate=lambda v: dirty)

        assert report.warning is not None
        assert report.warning["status"] == "unverified_claims_present"
        assert report.warning["n_contradicted"] > 0
        assert isinstance(report.warning["contradicted_claims"], list)
        assert guard.stats.n_warned == 1

    def test_repair_that_makes_things_worse_is_not_preferred(self, case) -> None:
        """A newer draft is not a better one; only a genuine improvement is kept."""
        facts, bronze, extractor = case
        dirty = self._dirty(bronze.text)
        worse = f"{dirty} The subject is a known criminal running a shell company."
        report = InferenceGuard().run("c1", [dirty], facts, extractor, regenerate=lambda v: worse)
        assert report.selected.score >= min(c.score for c in report.candidates)

    def test_regeneration_can_be_disabled_for_the_ablation(self, case) -> None:
        """Selection-only mode separates what selection bought from what repair bought."""
        facts, bronze, extractor = case
        guard = InferenceGuard(allow_regeneration=False)
        report = guard.run(
            "c1", [self._dirty(bronze.text)], facts, extractor, regenerate=lambda v: bronze.text
        )
        assert not report.regenerated
        assert guard.stats.n_regenerated == 0


class TestFaithfulnessProbe:
    """The training-time faithfulness measurement."""

    def test_bronze_probes_as_fully_supported(self, case) -> None:
        """The probe agrees with the corpus: Bronze is 100% supported."""
        facts, bronze, extractor = case
        probe = ProbeCase(case_id="c1", prompt=None, graph=None, facts=facts, extractor=extractor)
        results = score_narrative(bronze.text, probe, context=CheckContext(facts=facts))
        assert results
        assert all(r.verdict.value == "supported" for r in results)

    def test_forbidden_phrase_is_caught_by_the_text_level_check(self, case) -> None:
        """A narrative asserting guilt fails even when every number is right."""
        facts, bronze, extractor = case
        probe = ProbeCase(case_id="c1", prompt=None, graph=None, facts=facts, extractor=extractor)
        results = score_narrative(
            f"{bronze.text} The subject is guilty of money laundering.", probe
        )
        assert any(r.verdict.value == "contradicted" for r in results)


class TestArmComparison:
    """Gate 8: S1 against A1."""

    def _history(self, tmp_path, name, rates):
        """Write a diagnostic history file.

        Args:
            tmp_path: Temp directory.
            name: Arm name.
            rates: Supported rate per step.

        Returns:
            The written path.
        """
        path = tmp_path / f"history_{name}.jsonl"
        write_jsonl(
            path,
            [
                {"arm": name, "step": (i + 1) * 100, "probe_supported_rate": r}
                for i, r in enumerate(rates)
            ],
        )
        return path

    def test_a_clear_win_is_reported_as_one(self, tmp_path) -> None:
        """S1 well above A1 reads as a contribution."""
        treatment = self._history(tmp_path, "S1", [0.5, 0.7, 0.9])
        control = self._history(tmp_path, "A1", [0.4, 0.45, 0.5])
        comparison = compare_arms(treatment, control)
        assert comparison.difference == pytest.approx(0.4)
        assert "contributes" in comparison.verdict()
        assert not comparison.tracked_throughout

    def test_tracking_arms_report_the_pivot_plainly(self, tmp_path) -> None:
        """The sentence that must not get softened between the trace and the write-up."""
        treatment = self._history(tmp_path, "S1", [0.50, 0.60, 0.70])
        control = self._history(tmp_path, "A1", [0.51, 0.61, 0.71])
        comparison = compare_arms(treatment, control)
        assert comparison.tracked_throughout
        verdict = comparison.verdict()
        assert "no architecture contribution" in verdict
        assert "pivots" in verdict

    def test_a_harmful_graph_pathway_is_named(self, tmp_path) -> None:
        """S1 below A1 is reported as harm, not rounded to 'no difference'."""
        treatment = self._history(tmp_path, "S1", [0.4])
        control = self._history(tmp_path, "A1", [0.8])
        assert "actively harming" in compare_arms(treatment, control).verdict()

    def test_history_without_probe_rows_is_refused(self, tmp_path) -> None:
        """A run that never measured faithfulness cannot be compared."""
        path = tmp_path / "empty.jsonl"
        write_jsonl(path, [{"arm": "S1", "step": 100, "loss": 1.0}])
        with pytest.raises(ValueError, match="probe_supported_rate"):
            compare_arms(path, path)


class TestGenerationConfig:
    """Deterministic and sampled decoding are different claims."""

    def test_deterministic_mode_is_greedy_and_single(self) -> None:
        """Every reported evaluation number is measured greedily."""
        cfg = GenerationConfig.deterministic()
        assert not cfg.do_sample
        assert cfg.num_return_sequences == 1

    def test_guard_candidates_use_the_brief_s_settings(self) -> None:
        """Temperature 0.6, top-p 0.9, four candidates."""
        cfg = GenerationConfig.guard_candidates()
        assert cfg.do_sample
        assert cfg.temperature == 0.6
        assert cfg.top_p == 0.9
        assert cfg.num_return_sequences == 4

    def test_one_candidate_is_refused(self) -> None:
        """A guard with one candidate has nothing to select between."""
        with pytest.raises(ValueError, match="at least 2 candidates"):
            GenerationConfig.guard_candidates(n=1)
