"""The ten-point harness, with a deliberately broken record for every check.

**A gate nobody has seen fail is a gate nobody has tested.** Each test here takes a record
that passes all ten checks, breaks exactly one thing, and asserts that the harness catches
*that* check and — where the breakage is local — only that check. Without this, a check
that silently never fired would look identical to a check that always passes, and Bronze's
100% pass rate would be evidence of nothing.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from tests import factories

from g2t_aml.corpus.bronze.renderer import render_bronze
from g2t_aml.corpus.claims import claims_from_slots
from g2t_aml.corpus.graphref import build_graph_ref
from g2t_aml.corpus.record import TrainingRecord
from g2t_aml.corpus.validate import (
    CHECKS,
    MAX_UNVERIFIABLE_RATE,
    load_split_manifest,
    validate_corpus,
)
from g2t_aml.facts.checkers import CheckContext, check_claim, summarise
from g2t_aml.facts.extractor import extract_facts
from g2t_aml.facts.serialiser import serialise_facts
from g2t_aml.facts.vocab import load_vocabulary

CASE_STORE = Path("data/processed/fixture/cases")


@pytest.fixture(scope="module")
def vocabulary():
    return load_vocabulary()


@pytest.fixture
def record(vocabulary, tmp_path: Path) -> dict:
    """A record that passes all ten checks, over a case store built for this test."""
    case = factories.as_laundering_stream(factories.fan_out_case(width=9), "fan_out")
    facts = extract_facts(case)
    narrative = render_bronze(facts, vocabulary=vocabulary)
    context = CheckContext(facts=facts, vocabulary=vocabulary)
    verdicts = summarise(
        [check_claim(c, context) for c in claims_from_slots(narrative.slots, narrative.text)]
    )
    store = tmp_path / "data" / "processed" / "fixture" / "cases"
    _write_case_store(store, facts)
    return TrainingRecord(
        case_id=facts.case_id,
        dataset="amlworld_hi_small",
        split="train",
        tier="bronze",
        facts=facts,
        graph_ref=build_graph_ref(store, facts.case_id, tmp_path),
        serialised_facts=serialise_facts(facts, style="compact"),
        target_narrative=narrative.text,
        target_slots=narrative.slots,
        generator={
            "method": "template",
            "family": narrative.family,
            "variant": narrative.variant,
            "renderer_version": narrative.renderer_version,
        },
        verification={
            "supported": verdicts["by_verdict"]["supported"],
            "contradicted": verdicts["by_verdict"]["contradicted"],
            "unverifiable": verdicts["by_verdict"]["unverifiable"],
            "unverifiable_rate": verdicts["unverifiable_rate"],
            "n_claims": verdicts["n_claims"],
        },
    ).to_dict()


def _write_case_store(store: Path, facts) -> None:
    """Write membership tables matching a fact record's structure block."""
    import polars as pl

    store.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "case_id": [facts.case_id] * facts.structure.n_nodes,
            "node_index": list(range(facts.structure.n_nodes)),
            "node_id": list(facts.entity_inventory.node_ids),
        }
    ).write_parquet(store / "case_nodes.parquet")
    pl.DataFrame(
        {
            "case_id": [facts.case_id] * facts.structure.n_edges,
            "edge_index": list(range(facts.structure.n_edges)),
        }
    ).write_parquet(store / "case_edges.parquet")


def _run(records: list[dict], tmp_path: Path, vocabulary, manifest: dict[str, str] | None = None):
    if manifest is None:
        manifest = {str(r["case_id"]): str(r["split"]) for r in records}
    return validate_corpus(
        records, repo_root=tmp_path, split_manifest=manifest, vocabulary=vocabulary
    )


class TestTheHarnessPassesAGoodRecord:
    def test_a_clean_record_passes_all_ten(self, record, tmp_path, vocabulary) -> None:
        report = _run([record], tmp_path, vocabulary)
        assert report.gate_passed, report.summary()
        assert report.failures_by_check == dict.fromkeys(report.failures_by_check, 0)

    def test_the_report_serialises(self, record, tmp_path, vocabulary) -> None:
        payload = _run([record], tmp_path, vocabulary).to_dict()
        assert payload["gate_passed"] is True
        assert payload["total"] == 1


class TestEachCheckCatchesItsOwnBreakage:
    """One deliberately broken fixture per check. The order matches CHECKS."""

    def test_1_schema_violation(self, record, tmp_path, vocabulary) -> None:
        broken = copy.deepcopy(record)
        del broken["serialised_facts"]
        report = _run([broken], tmp_path, vocabulary)
        assert not report.gate_passed
        assert report.failures_by_check["schema_valid"] == 1

    def test_2_graph_ref_pointing_at_the_wrong_subgraph(self, record, tmp_path, vocabulary) -> None:
        """The failure this check exists for: it resolves, but to a different graph."""
        broken = copy.deepcopy(record)
        broken["facts"]["structure"]["n_nodes"] += 1
        report = _run([broken], tmp_path, vocabulary)
        assert report.failures_by_check["graph_ref_resolves"] == 1

    def test_2b_graph_ref_that_does_not_resolve_at_all(self, record, tmp_path, vocabulary) -> None:
        broken = copy.deepcopy(record)
        broken["graph_ref"] = "data/processed/fixture/cases#not-a-real-case"
        report = _run([broken], tmp_path, vocabulary)
        assert report.failures_by_check["graph_ref_resolves"] == 1

    def test_3_stale_fact_schema_version(self, record, tmp_path, vocabulary) -> None:
        broken = copy.deepcopy(record)
        broken["facts"]["schema_version"] = "0.9.0"
        report = _run([broken], tmp_path, vocabulary)
        assert not report.gate_passed
        # The training-record schema $refs case_facts, whose schema_version is a const,
        # so this is caught at the schema layer too. Either way it must not pass.
        assert (
            report.failures_by_check["facts_schema_version"]
            + report.failures_by_check["schema_valid"]
        ) >= 1

    def test_4_a_contradicted_claim(self, record, tmp_path, vocabulary) -> None:
        """Move a rendered number without moving the record. The classic corpus bug."""
        broken = copy.deepcopy(record)
        slot = next(s for s in broken["target_slots"] if s["field_path"] == "structure.n_nodes")
        start, end = slot["span"]
        wrong = str(int(slot["rendered_value"]) + 7)
        broken["target_narrative"] = (
            broken["target_narrative"][:start] + wrong + broken["target_narrative"][end:]
        )
        shift = len(wrong) - (end - start)
        slot["rendered_value"] = wrong
        slot["span"] = [start, start + len(wrong)]
        for other in broken["target_slots"]:
            if other is not slot and other["span"][0] >= end:
                other["span"] = [other["span"][0] + shift, other["span"][1] + shift]
        report = _run([broken], tmp_path, vocabulary)
        assert report.failures_by_check["zero_contradicted"] == 1

    def test_5_unverifiable_rate_over_budget(self, record, tmp_path, vocabulary) -> None:
        """A claim about a field the substrate cannot license is UNVERIFIABLE, not false."""
        broken = copy.deepcopy(record)
        addition = " Cross-border movement was observed."
        start = len(broken["target_narrative"]) + len(" Cross-border movement was ")
        broken["target_narrative"] += addition
        broken["target_slots"] = [
            {
                "field_path": "flow.cross_border",
                "span": [start, start + len("observed")],
                "rendered_value": "observed",
                "raw_value": True,
                "claim_type": "categorical",
            }
        ]
        report = _run([broken], tmp_path, vocabulary)
        assert report.failures_by_check["unverifiable_rate"] == 1
        assert report.unverifiable_rate_distribution["max"] > MAX_UNVERIFIABLE_RATE

    def test_6_narrative_too_short(self, record, tmp_path, vocabulary) -> None:
        broken = copy.deepcopy(record)
        broken["target_narrative"] = "The account moved money."
        broken["target_slots"] = []
        report = _run([broken], tmp_path, vocabulary)
        assert report.failures_by_check["length_in_bounds"] == 1

    def test_6b_narrative_too_long(self, record, tmp_path, vocabulary) -> None:
        broken = copy.deepcopy(record)
        broken["target_narrative"] += " The activity was reviewed carefully." * 200
        report = _run([broken], tmp_path, vocabulary)
        assert report.failures_by_check["length_in_bounds"] == 1

    def test_7_a_forbidden_guilt_assertion(self, record, tmp_path, vocabulary) -> None:
        broken = copy.deepcopy(record)
        broken["target_narrative"] += " The evidence proves the account holder is guilty."
        report = _run([broken], tmp_path, vocabulary)
        assert report.failures_by_check["vocabulary_clean"] == 1
        assert report.failures_by_check["zero_contradicted"] == 1

    def test_7b_an_out_of_substrate_entity_type(self, record, tmp_path, vocabulary) -> None:
        """H4, Critical. No substrate carries an entity-type column (D-029)."""
        broken = copy.deepcopy(record)
        broken["target_narrative"] += " The counterparty is a mixer."
        report = _run([broken], tmp_path, vocabulary)
        assert report.failures_by_check["vocabulary_clean"] == 1

    def test_7c_a_risk_descriptor_whose_binding_fails(self, record, tmp_path, vocabulary) -> None:
        """'rapid dispersal' is a claim that burst_window_hours <= 6, not a flourish."""
        broken = copy.deepcopy(record)
        phrase = "sustained over an extended period"
        start = len(broken["target_narrative"]) + len(" The activity was ")
        broken["target_narrative"] += f" The activity was {phrase}."
        broken["target_slots"].append(
            {
                "field_path": "temporal.span_hours",
                "span": [start, start + len(phrase)],
                "rendered_value": phrase,
                "raw_value": 8.0,
                "claim_type": "qualitative",
            }
        )
        report = _run([broken], tmp_path, vocabulary)
        assert report.failures_by_check["vocabulary_clean"] == 1
        assert report.failures_by_check["zero_contradicted"] == 1

    def test_8_a_real_world_identifier(self, record, tmp_path, vocabulary) -> None:
        broken = copy.deepcopy(record)
        broken["target_narrative"] += " Contact the holder at analyst@example.com."
        report = _run([broken], tmp_path, vocabulary)
        assert report.failures_by_check["no_pii_or_identifiers"] == 1

    def test_8b_an_iban(self, record, tmp_path, vocabulary) -> None:
        broken = copy.deepcopy(record)
        broken["target_narrative"] += " Settlement referenced GB29NWBK60161331926819."
        report = _run([broken], tmp_path, vocabulary)
        assert report.failures_by_check["no_pii_or_identifiers"] == 1

    def test_9_a_near_duplicate_pair(self, record, tmp_path, vocabulary) -> None:
        twin = copy.deepcopy(record)
        twin["case_id"] = record["case_id"] + "-twin"
        twin["facts"]["case_id"] = twin["case_id"]
        report = _run(
            [record, twin],
            tmp_path,
            vocabulary,
            manifest={record["case_id"]: "train", twin["case_id"]: "train"},
        )
        assert report.failures_by_check["deduplicated"] == 1
        assert report.duplicates is not None
        assert report.duplicates.n_dropped == 1

    def test_10_split_disagrees_with_the_manifest(self, record, tmp_path, vocabulary) -> None:
        report = _run([record], tmp_path, vocabulary, manifest={record["case_id"]: "test"})
        assert report.failures_by_check["split_consistent"] == 1

    def test_10b_case_absent_from_the_manifest(self, record, tmp_path, vocabulary) -> None:
        report = _run([record], tmp_path, vocabulary, manifest={})
        assert report.failures_by_check["split_consistent"] == 1


class TestTheHarnessDoesNotSkipChecksItCannotRun:
    """A harness that quietly drops a check reports a pass it never tested."""

    def test_a_missing_case_store_fails_check_2_rather_than_skipping_it(
        self, record, tmp_path, vocabulary
    ) -> None:
        report = validate_corpus(
            [record],
            repo_root=None,
            split_manifest={record["case_id"]: "train"},
            vocabulary=vocabulary,
        )
        assert report.failures_by_check["graph_ref_resolves"] == 1

    def test_a_missing_manifest_fails_check_10_rather_than_skipping_it(
        self, record, tmp_path, vocabulary
    ) -> None:
        report = validate_corpus(
            [record], repo_root=tmp_path, split_manifest=None, vocabulary=vocabulary
        )
        assert report.failures_by_check["split_consistent"] == 1


class TestSplitManifestLoading:
    def test_reads_the_committed_manifest(self, repo_root: Path) -> None:
        manifest = load_split_manifest(repo_root / "schemas" / "splits" / "amlworld")
        assert len(manifest) == 16156
        assert set(manifest.values()) == {"train", "val", "test"}

    def test_refuses_a_case_in_two_splits(self, tmp_path: Path) -> None:
        for split in ("train", "val", "test"):
            (tmp_path / f"{split}.txt").write_text("case-a\n" if split != "test" else "case-b\n")
        with pytest.raises(ValueError, match="appears in both"):
            load_split_manifest(tmp_path)


class TestTheHumanReadableSummary:
    """The summary is what a person actually reads when a build fails."""

    def test_it_names_every_check_and_the_verdict(self, record, tmp_path, vocabulary) -> None:
        summary = _run([record], tmp_path, vocabulary).summary()
        assert "GATE PASSED" in summary
        for check in CHECKS:
            assert check in summary
        assert "length tokens" in summary
        assert "unverifiable" in summary
        assert "duplicates" in summary

    def test_it_marks_a_failing_check_and_shows_an_example(
        self, record, tmp_path, vocabulary
    ) -> None:
        broken = copy.deepcopy(record)
        broken["target_narrative"] += " The evidence proves the holder is guilty."
        summary = _run([broken], tmp_path, vocabulary).summary()
        assert "GATE FAILED" in summary
        assert "!" in summary, "a failing check must be visually marked"
        assert broken["case_id"] in summary, "the summary must name a failing case"
