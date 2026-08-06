"""Bronze end to end, over the real corpus when it exists.

The unit suite proves each piece works on hand-built fixtures. This proves the pieces agree
on the real substrate, which is where a disagreement about what a field *means* actually
surfaces — Phase 3 found three of those, and all three were invisible to fixtures.

The tests that need the built corpus skip cleanly when it is absent, so a fresh checkout
runs green before `make bronze` has ever been run. They are not optional in CI, where the
pipeline has run.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from g2t_aml.corpus.bronze.renderer import MAX_TOKENS, MIN_TOKENS, render_bronze
from g2t_aml.corpus.claims import claims_from_slots
from g2t_aml.corpus.factsio import load_case_facts, load_case_facts_file
from g2t_aml.corpus.record import validate_training_record
from g2t_aml.corpus.tokenization import get_token_counter
from g2t_aml.corpus.validate import load_split_manifest, validate_corpus
from g2t_aml.facts.checkers import CheckContext, Verdict, check_claim, check_narrative_text
from g2t_aml.facts.vocab import load_vocabulary

pytestmark = pytest.mark.integration

DATASET = "amlworld_hi_small"
SAMPLE = 300


def _corpus_path(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / DATASET / "corpus" / "bronze.jsonl"


def _facts_dir(repo_root: Path) -> Path:
    return repo_root / "data" / "processed" / DATASET / "facts"


@pytest.fixture(scope="module")
def vocabulary():
    return load_vocabulary()


@pytest.fixture(scope="module")
def corpus_sample(repo_root: Path) -> list[dict]:
    path = _corpus_path(repo_root)
    if not path.is_file():
        pytest.skip(f"no built corpus at {path}; run `make bronze`")
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    rng = random.Random(0)
    return rng.sample(records, min(SAMPLE, len(records)))


@pytest.fixture(scope="module")
def fact_sample(repo_root: Path) -> list:
    directory = _facts_dir(repo_root)
    if not directory.is_dir():
        pytest.skip(f"no fact records at {directory}; run `make facts`")
    manifest = load_split_manifest(repo_root / "schemas" / "splits" / "amlworld")
    rng = random.Random(1)
    chosen = rng.sample(sorted(manifest), SAMPLE)
    records = []
    for case_id in chosen:
        path = directory / f"{case_id}.json"
        if path.is_file():
            records.append(load_case_facts_file(path))
    return records


class TestRenderingOnRealRecords:
    def test_every_sampled_case_renders(self, fact_sample, vocabulary) -> None:
        for facts in fact_sample:
            if facts.structure.n_nodes < 2:
                continue
            assert render_bronze(facts, vocabulary=vocabulary).text

    def test_no_claim_is_contradicted_over_the_sample(self, fact_sample, vocabulary) -> None:
        """The Phase 4 gate, measured rather than asserted."""
        contradicted: list[str] = []
        n_claims = 0
        for facts in fact_sample:
            if facts.structure.n_nodes < 2:
                continue
            narrative = render_bronze(facts, vocabulary=vocabulary)
            context = CheckContext(facts=facts, vocabulary=vocabulary)
            results = [
                check_claim(c, context) for c in claims_from_slots(narrative.slots, narrative.text)
            ]
            results += check_narrative_text(narrative.text, context)
            n_claims += len(results)
            contradicted += [
                f"{facts.case_id}: {r.reason}" for r in results if r.verdict is Verdict.CONTRADICTED
            ]
        assert n_claims > 0
        assert not contradicted, contradicted[:5]

    def test_rendering_is_reproducible_on_real_records(self, fact_sample, vocabulary) -> None:
        for facts in fact_sample[:50]:
            if facts.structure.n_nodes < 2:
                continue
            first = render_bronze(facts, vocabulary=vocabulary)
            second = render_bronze(facts, vocabulary=vocabulary)
            assert first.text == second.text

    def test_lengths_stay_inside_the_bounds(self, fact_sample, vocabulary) -> None:
        counter = get_token_counter()
        for facts in fact_sample:
            if facts.structure.n_nodes < 2:
                continue
            tokens = counter.count(render_bronze(facts, vocabulary=vocabulary).text)
            assert MIN_TOKENS <= tokens <= MAX_TOKENS, (facts.case_id, tokens)


class TestTheBuiltCorpus:
    def test_every_sampled_record_validates_against_the_schema(self, corpus_sample) -> None:
        for record in corpus_sample:
            validate_training_record(record)

    def test_the_sample_passes_the_ten_point_gate(
        self, corpus_sample, repo_root: Path, vocabulary
    ) -> None:
        manifest = load_split_manifest(repo_root / "schemas" / "splits" / "amlworld")
        report = validate_corpus(
            corpus_sample, repo_root=repo_root, split_manifest=manifest, vocabulary=vocabulary
        )
        assert report.gate_passed, report.summary()

    def test_slot_spans_index_the_narrative_correctly(self, corpus_sample) -> None:
        for record in corpus_sample:
            narrative = record["target_narrative"]
            for slot in record["target_slots"]:
                start, end = slot["span"]
                assert narrative[start:end] == slot["rendered_value"]

    def test_the_embedded_facts_are_the_record_the_narrative_was_written_from(
        self, corpus_sample, repo_root: Path
    ) -> None:
        """Self-containment: re-verification must not depend on the fact store.

        `model_signal` is excluded from the comparison, and the exclusion is the point
        rather than a concession. D-037 embeds the fact record so a narrative can be
        re-verified without the fact store, and the embedded copy is a **snapshot of what
        the narrative was written from**. Phase 7 populated `model_signal` on the on-disk
        records afterwards, so the two now differ in exactly that block — correctly. The
        narrative genuinely was written from a record carrying no model signal, because no
        Bronze template reads one and `model_signal` is the model's opinion rather than a
        fact about the subgraph (`facts/schema.py`, D-063).

        Every other block must still match byte for byte. A divergence anywhere else would
        mean the fact store and the corpus had drifted, which is the failure this test
        exists to catch.
        """
        for record in corpus_sample[:50]:
            facts = load_case_facts(record["facts"])
            assert facts.case_id == record["case_id"]
            on_disk = _facts_dir(repo_root) / f"{record['case_id']}.json"
            if not on_disk.is_file():
                continue
            live = json.loads(on_disk.read_text(encoding="utf-8"))
            embedded = dict(record["facts"])
            assert embedded.pop("model_signal", None) is not None
            live.pop("model_signal", None)
            assert embedded == live, (
                f"{record['case_id']}: the embedded fact record has drifted from the "
                "fact store outside the model_signal block"
            )

    def test_the_embedded_facts_carry_no_model_signal(self, corpus_sample) -> None:
        """The corpus predates the encoder, and the serialisation baseline depends on it.

        `serialised_facts` is rendered from the embedded record and reaches the
        serialisation baseline -- the "flatten the facts, no graph encoder" ablation arm.
        A Bronze regeneration after Phase 7's write-back would put the encoder's own risk
        score into the baseline it exists to be compared against. See DECISIONS.md D-063.
        """
        for record in corpus_sample:
            signal = record["facts"]["model_signal"]
            assert signal["gnn_risk_score"] is None, record["case_id"]
            assert signal["model_version"] is None, record["case_id"]
            assert signal["top_contributing_nodes"] == [], record["case_id"]

    def test_the_written_verification_block_matches_a_fresh_computation(
        self, corpus_sample, vocabulary
    ) -> None:
        """The build's self-report is checked against the harness, not trusted."""
        for record in corpus_sample[:50]:
            facts = load_case_facts(record["facts"])
            context = CheckContext(facts=facts, vocabulary=vocabulary)
            from g2t_aml.corpus.record import SlotAnnotation

            slots = [SlotAnnotation.from_dict(s) for s in record["target_slots"]]
            results = [
                check_claim(c, context)
                for c in claims_from_slots(slots, record["target_narrative"])
            ]
            supported = sum(1 for r in results if r.verdict is Verdict.SUPPORTED)
            assert supported == record["verification"]["supported"]
            assert record["verification"]["contradicted"] == 0

    def test_salience_coverage_is_complete(self, corpus_sample) -> None:
        """Bronze's adequacy ceiling, which Phase 10 scores learned systems against."""
        for record in corpus_sample:
            assert record["salience"]["coverage"] == 1.0, record["case_id"]

    def test_every_record_is_in_the_frozen_manifest(self, corpus_sample, repo_root: Path) -> None:
        manifest = load_split_manifest(repo_root / "schemas" / "splits" / "amlworld")
        for record in corpus_sample:
            assert manifest[record["case_id"]] == record["split"]


class TestReportsExist:
    @pytest.mark.parametrize(
        "name", ["bronze_validation.json", "bronze_diversity.json", "bronze_samples.md"]
    )
    def test_the_build_wrote_its_report(self, repo_root: Path, name: str) -> None:
        path = _corpus_path(repo_root).parent / name
        if not _corpus_path(repo_root).is_file():
            pytest.skip("no built corpus")
        assert path.is_file(), f"{name} was not written"

    def test_the_validation_report_records_a_passing_gate(self, repo_root: Path) -> None:
        path = _corpus_path(repo_root).parent / "bronze_validation.json"
        if not path.is_file():
            pytest.skip("no built corpus")
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["gate_passed"] is True
        assert report["failures_by_check"] == dict.fromkeys(report["failures_by_check"], 0)

    def test_the_diversity_report_is_not_pathological(self, repo_root: Path) -> None:
        path = _corpus_path(repo_root).parent / "bronze_diversity.json"
        if not path.is_file():
            pytest.skip("no built corpus")
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["self_bleu"] < 0.60, "the template pack is collapsing"
        assert report["skeleton_ratio"] > 0.5, "too many narratives share a surface form"
