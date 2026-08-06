"""The block design: balance, blinding, repeats, and the constraints that must not bend."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from g2t_aml.human.study_design import (
    MIN_REPEAT_SEPARATION,
    BlindKey,
    DesignError,
    StudyDesign,
    build_design,
    load_design,
    validate_design,
)

SYSTEMS = ["S1", "S2", "B7", "B3", "Bronze"]
CASES = [f"case-{i:03d}" for i in range(100)]
RATERS = [f"rater-{i:02d}" for i in range(10)]


@pytest.fixture(scope="module")
def built() -> tuple[StudyDesign, BlindKey]:
    return build_design(CASES, SYSTEMS, RATERS, dataset="amlworld_hi_small", items_per_rater=60)


# --------------------------------------------------------------- the design ---


def test_every_rater_gets_the_declared_workload(built):
    design, _ = built
    for rater_id in RATERS:
        assert len(design.for_rater(rater_id)) == 60


def test_no_rater_ever_sees_a_case_twice(built):
    """The anchoring constraint. A violation cannot be corrected in analysis."""
    design, _ = built
    for rater_id in RATERS:
        cases = [i.case_id for i in design.for_rater(rater_id) if not i.is_repeat]
        assert len(cases) == len(set(cases)), f"{rater_id} sees a case twice"


def test_no_rater_has_a_duplicate_case_system_pair(built):
    design, key = built
    for rater_id in RATERS:
        pairs = [
            (i.case_id, key.system_for(i.item_id))
            for i in design.for_rater(rater_id)
            if not i.is_repeat
        ]
        assert len(pairs) == len(set(pairs))


def test_per_rater_system_workload_is_balanced_within_one(built):
    design, key = built
    for rater_id in RATERS:
        counts = Counter(
            key.system_for(i.item_id) for i in design.for_rater(rater_id) if not i.is_repeat
        )
        assert set(counts) == set(SYSTEMS), f"{rater_id} misses an arm"
        assert max(counts.values()) - min(counts.values()) <= 1


def test_position_is_balanced_across_systems(built):
    """Order effects must cancel, not merely be randomised."""
    design, key = built
    positions: dict[str, list[int]] = {s: [] for s in SYSTEMS}
    for item in design.items:
        positions[key.system_for(item.item_id)].append(item.position)
    means = {s: sum(p) / len(p) for s, p in positions.items()}
    assert max(means.values()) - min(means.values()) < 2.0


def test_the_design_is_reproducible_from_its_seed():
    a, ka = build_design(CASES, SYSTEMS, RATERS, dataset="d", items_per_rater=60, seed=99)
    b, kb = build_design(CASES, SYSTEMS, RATERS, dataset="d", items_per_rater=60, seed=99)
    assert a.to_dict() == b.to_dict()
    assert ka.assignments == kb.assignments


def test_a_different_seed_gives_a_different_design():
    a, _ = build_design(CASES, SYSTEMS, RATERS, dataset="d", items_per_rater=60, seed=1)
    b, _ = build_design(CASES, SYSTEMS, RATERS, dataset="d", items_per_rater=60, seed=2)
    assert a.to_dict() != b.to_dict()


# -------------------------------------------------------------- the anchors ---


def test_anchor_cells_are_rated_by_every_rater(built):
    """Without these there is no inter-rater agreement statistic at all."""
    design, key = built
    coverage = Counter(
        (i.case_id, key.system_for(i.item_id)) for i in design.items if not i.is_repeat
    )
    anchors = [cell for cell, n in coverage.items() if n == len(RATERS)]
    assert len(anchors) >= len(SYSTEMS)
    assert design.report.n_anchor_cells == len(anchors)


def test_anchor_cells_span_every_system(built):
    design, key = built
    coverage = Counter(
        (i.case_id, key.system_for(i.item_id)) for i in design.items if not i.is_repeat
    )
    anchored_systems = {s for (_, s), n in coverage.items() if n == len(RATERS)}
    assert anchored_systems == set(SYSTEMS)


def test_a_design_with_no_anchors_reports_zero_anchor_cells():
    design, _ = build_design(
        CASES, SYSTEMS, RATERS, dataset="d", items_per_rater=60, n_anchor_cases=0
    )
    assert design.report.n_anchor_cells == 0


# -------------------------------------------------------------- the repeats ---


def test_repeats_match_their_original_and_clear_the_separation_bound(built):
    design, key = built
    by_id = {i.item_id: i for i in design.items}
    n_repeats = 0
    for item in design.items:
        if not item.is_repeat:
            continue
        n_repeats += 1
        original = by_id[item.repeat_of]
        assert original.case_id == item.case_id
        assert key.system_for(original.item_id) == key.system_for(item.item_id)
        assert item.position - original.position >= MIN_REPEAT_SEPARATION
    assert n_repeats == len(RATERS) * design.report.n_repeats_per_rater
    assert n_repeats > 0


def test_a_repeat_has_a_distinct_item_id_from_its_original(built):
    """Sharing an id would let the second rating overwrite the first in any item-keyed store."""
    design, _ = built
    ids = [i.item_id for i in design.items]
    assert len(ids) == len(set(ids))


# ------------------------------------------------------------- the blinding ---


def test_no_serialised_item_carries_a_system_identity(built):
    design, _ = built
    blob = json.dumps(design.to_dict()["items"])
    for system in SYSTEMS:
        assert f'"{system}"' not in blob


def test_item_ids_do_not_reveal_whether_two_items_share_a_system(built):
    """Ids are keyed digests, so sorting or comparing them carries no information."""
    design, key = built
    same = [i.item_id[:4] for i in design.items if key.system_for(i.item_id) == "S1"]
    other = [i.item_id[:4] for i in design.items if key.system_for(i.item_id) == "Bronze"]
    assert set(same) & set(other) or len(set(same) | set(other)) > len(SYSTEMS)


def test_writing_splits_design_and_key_into_two_files(tmp_path, built):
    design, key = built
    design_path, key_path = tmp_path / "design.json", tmp_path / "key.json"
    design.write(design_path, key, key_path)
    assert "assignments" not in design_path.read_text()
    assert "assignments" in key_path.read_text()


def test_loading_refuses_a_design_file_that_contains_a_key(tmp_path, built):
    design, key = built
    path = tmp_path / "design.json"
    payload = design.to_dict() | {"assignments": key.assignments}
    path.write_text(json.dumps(payload))
    with pytest.raises(DesignError, match="blind key"):
        load_design(path)


def test_round_trip_through_disk_preserves_the_design(tmp_path, built):
    design, key = built
    design.write(tmp_path / "d.json", key, tmp_path / "k.json")
    assert load_design(tmp_path / "d.json").to_dict() == design.to_dict()


# ------------------------------------------------------------- the refusals ---


def test_validate_rejects_a_rater_seeing_a_case_twice(built):
    design, key = built
    items = list(design.items)
    first = next(i for i in items if not i.is_repeat and i.rater_id == RATERS[0])
    second = next(
        i
        for i in items
        if not i.is_repeat and i.rater_id == RATERS[0] and i.case_id != first.case_id
    )
    items[items.index(second)] = type(second)(**{**second.__dict__, "case_id": first.case_id})
    broken = StudyDesign(items=tuple(items), report=design.report, systems=design.systems)
    with pytest.raises(DesignError, match="twice"):
        validate_design(broken, key)


def test_validate_rejects_a_key_that_does_not_match_the_design(built):
    design, key = built
    with pytest.raises(DesignError):
        validate_design(design, BlindKey(assignments={"nope": "S1"}, salt=key.salt))


def test_system_for_an_unknown_item_names_the_mismatch(built):
    _, key = built
    with pytest.raises(DesignError, match="different builds"):
        key.system_for("not-an-item")


def test_workload_cannot_exceed_the_case_pool():
    with pytest.raises(DesignError, match="never exceed"):
        build_design(CASES[:20], SYSTEMS, RATERS, dataset="d", items_per_rater=40)


def test_a_workload_too_small_to_cover_every_arm_is_refused():
    with pytest.raises(DesignError, match="cannot cover"):
        build_design(CASES, SYSTEMS, RATERS, dataset="d", items_per_rater=3)


def test_duplicate_cases_are_refused():
    with pytest.raises(DesignError, match="duplicates"):
        build_design([*CASES, CASES[0]], SYSTEMS, RATERS, dataset="d", items_per_rater=60)


def test_duplicate_raters_are_refused():
    with pytest.raises(DesignError, match="duplicates"):
        build_design(CASES, SYSTEMS, [*RATERS, RATERS[0]], dataset="d", items_per_rater=60)


# ------------------------------------------------------------------ capacity ---


@pytest.mark.parametrize(
    ("n_raters", "items", "floor"),
    [(10, 60, 80), (12, 56, 80)],
)
def test_the_planned_parameterisations_reach_the_gate(n_raters, items, floor):
    """The Phase 12 gate is stated in cases-per-system, not in pool size."""
    design, _ = build_design(
        CASES,
        SYSTEMS,
        [f"rater-{i:02d}" for i in range(n_raters)],
        dataset="d",
        items_per_rater=items,
    )
    assert min(design.report.cases_per_system.values()) >= floor
