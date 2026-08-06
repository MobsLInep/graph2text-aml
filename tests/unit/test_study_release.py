"""The anonymised release: what leaves, what is replaced, and what never leaves at all."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from g2t_aml.human.study_design import build_design
from g2t_aml.human.study_release import prepare_release, pseudonymise
from g2t_aml.human.study_ui import RatingResponse

SYSTEMS = ["S1", "S2", "B7", "B3", "Bronze"]
NARRATIVE = "The subject account received funds from six counterparties within 22 hours."


@pytest.fixture(scope="module")
def study():
    design, key = build_design(
        [f"c{i:03d}" for i in range(100)],
        SYSTEMS,
        [f"rater-{i:02d}" for i in range(10)],
        dataset="amlworld_hi_small",
        items_per_rater=60,
    )
    responses = [
        RatingResponse(
            item_id=item.item_id,
            rater_id=item.rater_id,
            case_id=item.case_id,
            position=item.position,
            is_repeat=item.is_repeat,
            factual_correctness=5,
            completeness=5,
            actionability=5,
            readability=5,
            regulatory_tone=5,
            would_file=True,
            seconds_to_usable_draft=200.0,
            presented_narrative=NARRATIVE,
            corrected_narrative=NARRATIVE + " Verified against the record.",
            comment="I have seen this pattern before at my previous employer.",
            timing_source="browser",
        )
        for item in design.items
    ]
    return design, key, responses


@pytest.fixture
def released(tmp_path, study):
    design, key, responses = study
    report = prepare_release(responses, key, design, tmp_path / "release")
    rows = [
        json.loads(line)
        for line in (tmp_path / "release" / "responses.jsonl").read_text().splitlines()
    ]
    return report, rows, tmp_path


# ------------------------------------------------------------ what is kept ---


def test_every_row_carries_its_system(released):
    """The release is the point at which the study stops being blind."""
    _, rows, _ = released
    assert all(row["system"] in SYSTEMS for row in rows)
    assert {row["system"] for row in rows} == set(SYSTEMS)


def test_both_narratives_survive_so_edit_distance_is_recomputable(released):
    _, rows, _ = released
    assert all(row["presented_narrative"] and row["corrected_narrative"] for row in rows)


def test_timing_provenance_travels_with_the_times(released):
    _, rows, _ = released
    assert all(row["timing_source"] in {"browser", "server"} for row in rows)


def test_repeats_are_released_and_flagged(released):
    """Intra-rater reliability has to be recomputable from the release."""
    _, rows, _ = released
    assert any(row["is_repeat"] for row in rows)


# --------------------------------------------------------- what is removed ---


def test_free_text_comments_never_leave(released):
    """No regular expression finds the general case, so the field is dropped entirely."""
    _, rows, tmp_path = released
    assert all("comment" not in row for row in rows)
    assert "previous employer" not in (tmp_path / "release" / "responses.jsonl").read_text()


def test_internal_rater_ids_never_leave(released):
    _, rows, tmp_path = released
    assert all("rater_id" not in row for row in rows)
    assert "rater-00" not in (tmp_path / "release" / "responses.jsonl").read_text()


def test_submission_timestamps_never_leave(released):
    """A sequence of them records when a named professional was working."""
    _, rows, _ = released
    assert all("submitted_at" not in row for row in rows)


def test_the_rater_map_is_written_outside_the_release_directory(released):
    _, _, tmp_path = released
    assert not (tmp_path / "release" / "release_rater_map.PRIVATE.json").exists()
    private = tmp_path / "release_rater_map.PRIVATE.json"
    assert private.is_file()
    assert "WARNING" in json.loads(private.read_text())


# ------------------------------------------------------ the pseudonymisation ---


def test_released_labels_are_not_the_internal_ones_in_order():
    """Labelling in input order would make the re-pseudonymisation cosmetic."""
    ids = [f"rater-{i:02d}" for i in range(10)]
    mapping = pseudonymise(ids, salt="release-salt")
    assert set(mapping.values()) == {f"R{i:02d}" for i in range(1, 11)}
    assert [mapping[i] for i in ids] != sorted(mapping.values())


def test_pseudonymisation_is_stable_for_a_salt():
    ids = [f"rater-{i:02d}" for i in range(6)]
    assert pseudonymise(ids, "s") == pseudonymise(ids, "s")


def test_a_different_salt_gives_a_different_mapping():
    ids = [f"rater-{i:02d}" for i in range(6)]
    assert pseudonymise(ids, "a") != pseudonymise(ids, "b")


def test_sharing_the_design_salt_is_refused(tmp_path, study):
    design, key, responses = study
    with pytest.raises(ValueError, match="must differ"):
        prepare_release(responses, key, design, tmp_path / "r", salt=key.salt)


# ----------------------------------------------------------- the withholding ---


def test_a_correction_carrying_an_identifier_is_withheld_and_named(tmp_path, study):
    """Withholding must be visible in the release, not silent."""
    design, key, responses = study
    tainted = RatingResponse(
        **{
            **responses[0].__dict__,
            "corrected_narrative": "Escalate to compliance@realbank.example.com immediately.",
        }
    )
    report = prepare_release([tainted, *responses[1:]], key, design, tmp_path / "release")
    assert report.n_withheld == 1
    assert tainted.item_id in report.withheld_items
    assert "responses.jsonl" in report.files
    manifest = json.loads((tmp_path / "release" / "manifest.json").read_text())
    assert manifest["n_withheld"] == 1


def test_the_manifest_reconciles_against_the_design(released, study):
    report, rows, _ = released
    design, _, responses = study
    assert report.n_released + report.n_withheld == len(responses)
    assert report.n_released == len(rows)
    assert report.n_raters == 10


def test_every_released_file_is_hashed(released):
    report, _, tmp_path = released
    for name in ("responses.jsonl", "design.json", "scales.json", "README.md"):
        assert name in report.files
        assert len(report.files[name]) == 64


def test_the_release_carries_the_scale_anchors(released):
    """The scales are published with the data rather than described in prose in a paper."""
    _, _, tmp_path = released
    scales = json.loads((tmp_path / "release" / "scales.json").read_text())
    assert len(scales) == 5
    assert all(s["anchor_1"] and s["anchor_4"] and s["anchor_7"] for s in scales)


def test_the_released_design_still_carries_no_system_labels(released):
    """The design is published so the block structure is inspectable; it stays blind."""
    _, _, tmp_path = released
    items = json.loads((tmp_path / "release" / "design.json").read_text())["items"]
    blob = json.dumps(items)
    for system in SYSTEMS:
        assert f'"{system}"' not in blob


def test_the_readme_warns_about_server_timed_rows(released):
    _, _, tmp_path = released
    readme = (tmp_path / "release" / "README.md").read_text()
    assert "upper bound" in readme
    assert "agreement statistic" in readme
