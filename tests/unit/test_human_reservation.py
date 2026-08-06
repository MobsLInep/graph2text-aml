"""The Gold reservation, and the loader guard that makes it mean something."""

from __future__ import annotations

import json

import pytest

from g2t_aml.corpus.training_data import load_training_records
from g2t_aml.human.reservation import (
    GoldReservation,
    ReservationError,
    assert_not_reserved,
    load_reservation,
    write_reservation,
)

IDS = ("case-c", "case-a", "case-b")


def reservation(*case_ids: str) -> GoldReservation:
    return GoldReservation(
        dataset="amlworld_hi_small",
        case_ids=case_ids or IDS,
        created_at="2026-08-03T00:00:00+00:00",
    )


# --------------------------------------------------------------- the record ---


def test_ids_are_sorted_and_hashed_on_construction():
    held = reservation()
    assert held.case_ids == ("case-a", "case-b", "case-c")
    assert len(held.id_list_sha256) == 64


def test_an_empty_reservation_is_refused():
    """No file and an empty file must be distinguishable."""
    with pytest.raises(ReservationError, match="reserves nothing"):
        GoldReservation(dataset="d", case_ids=())


def test_duplicate_ids_are_refused():
    with pytest.raises(ReservationError, match="not unique"):
        GoldReservation(dataset="d", case_ids=("a", "a", "b"))


def test_round_trips_through_its_serialised_form():
    held = reservation()
    assert GoldReservation.from_dict(held.to_dict()) == held


def test_a_hand_edited_id_list_is_detected():
    payload = reservation().to_dict()
    payload["case_ids"].append("case-smuggled-in")
    with pytest.raises(ReservationError, match="does not match its recorded sha256"):
        GoldReservation.from_dict(payload)


# ------------------------------------------------------------------- on disk ---


def test_writes_both_a_reviewable_list_and_a_machine_readable_record(tmp_path):
    write_reservation(reservation(), tmp_path)
    assert (tmp_path / "gold_reserved.txt").read_text().split() == ["case-a", "case-b", "case-c"]
    assert json.loads((tmp_path / "gold_reservation.json").read_text())["n"] == 3


def test_absent_reservation_reads_as_none_rather_than_raising(tmp_path):
    """Phases 1-5 predate the reservation; no file is a legitimate state."""
    assert load_reservation(tmp_path) is None


def test_round_trips_through_disk(tmp_path):
    write_reservation(reservation(), tmp_path)
    assert load_reservation(tmp_path) == reservation()


def test_the_two_files_disagreeing_is_detected(tmp_path):
    write_reservation(reservation(), tmp_path)
    (tmp_path / "gold_reserved.txt").write_text("case-a\ncase-b\n")
    with pytest.raises(ReservationError, match="disagree"):
        load_reservation(tmp_path)


def test_a_reserved_case_outside_the_declared_split_is_refused(tmp_path):
    """Reserving a train case holds nothing out."""
    write_reservation(reservation(), tmp_path)
    with pytest.raises(ReservationError, match="not in the 'test' split"):
        load_reservation(
            tmp_path,
            split_assignment={"case-a": "test", "case-b": "train", "case-c": "test"},
        )


def test_a_reservation_wholly_inside_the_split_loads(tmp_path):
    write_reservation(reservation(), tmp_path)
    held = load_reservation(tmp_path, split_assignment=dict.fromkeys(IDS, "test"))
    assert held is not None and len(held) == 3


# ------------------------------------------------------------------- the guard ---


def test_no_reservation_permits_everything():
    assert_not_reserved(["anything"], None)


def test_a_reserved_case_in_training_raises():
    with pytest.raises(ReservationError, match="Gold test-only cases appeared"):
        assert_not_reserved(["case-a", "other"], reservation())


def test_an_unreserved_case_set_passes():
    assert_not_reserved(["other-1", "other-2"], reservation())


# --------------------------------------------------------- the training loader ---


def corpus(tmp_path, records):
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def record(case_id, split="train", tier="bronze"):
    return {"case_id": case_id, "split": split, "tier": tier}


def test_loader_reads_only_the_requested_split(tmp_path):
    path = corpus(tmp_path, [record("a"), record("b", "val"), record("c", "test")])
    slice_ = load_training_records(path, split="train")
    assert slice_.case_ids == ("a",)
    assert slice_.n_skipped == 2


def test_loader_refuses_a_reserved_case_in_a_training_split(tmp_path):
    path = corpus(tmp_path, [record("case-a"), record("other")])
    with pytest.raises(ReservationError, match="Gold test-only cases appeared"):
        load_training_records(path, split="train", reservation=reservation())


def test_loader_permits_a_reserved_case_on_a_test_load(tmp_path):
    """Reading a reserved case as evaluation is the entire point of reserving it."""
    path = corpus(tmp_path, [record("case-a", "test")])
    assert len(load_training_records(path, split="test", reservation=reservation())) == 1


def test_loader_refuses_a_gold_record_marked_as_training(tmp_path):
    path = corpus(tmp_path, [record("x", "train", "gold")])
    with pytest.raises(ReservationError, match="never training supervision"):
        load_training_records(path, split="train")


def test_loader_refuses_gold_without_an_explicit_opt_in(tmp_path):
    path = corpus(tmp_path, [record("x", "test", "gold")])
    with pytest.raises(ReservationError, match="without allow_gold"):
        load_training_records(path, split="test")


def test_loader_reads_gold_when_asked_explicitly(tmp_path):
    path = corpus(tmp_path, [record("x", "test", "gold")])
    slice_ = load_training_records(path, split="test", allow_gold=True)
    assert slice_.by_tier == {"gold": 1}


def test_allow_gold_on_a_training_split_is_a_contradiction(tmp_path):
    path = corpus(tmp_path, [record("x")])
    with pytest.raises(ValueError, match="no configuration in which"):
        load_training_records(path, split="train", allow_gold=True)
