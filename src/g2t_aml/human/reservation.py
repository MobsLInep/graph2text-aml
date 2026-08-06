"""The Gold test-only reservation: which cases training may never see.

Gold's primary role is a held-out human reference. A Gold narrative that also appears in
the training corpus is worse than no Gold narrative at all: the headline comparison
becomes "the model reproduces text it was fitted on", and nothing in the metrics would
show it — the faithfulness numbers would look *better*, because a memorised target is
trivially faithful to the record it was written from.

**So the reservation is data, not discipline.** The reserved case ids are written into the
frozen split manifest directory as a committed list with its own content hash, exactly the
way Phase 2 commits the three splits (D-006), and
:func:`assert_not_reserved` refuses any training batch that touches one. The check lives in
the loader rather than in a review checklist because a review cannot be run by CI.

**Why this does not violate invariant 2.** Nothing here regenerates or edits a split. The
reservation is a *subset of the existing test split*, recorded beside it: ``test.txt`` and
its sha256 are untouched, and :func:`load_reservation` asserts that every reserved id is in
fact a test-split member. A reservation naming a train or val case is a bug, and it raises.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from g2t_aml.utils.hashing import hash_id_list
from g2t_aml.utils.io import read_json, write_json

__all__ = [
    "RESERVATION_FILENAME",
    "RESERVATION_LIST_FILENAME",
    "RESERVATION_VERSION",
    "GoldReservation",
    "ReservationError",
    "assert_not_reserved",
    "load_reservation",
    "write_reservation",
]

#: Version of the reservation record. Bumped if the fields change shape.
RESERVATION_VERSION = "1.0.0"

#: The committed id list, one case per line, beside ``test.txt``.
RESERVATION_LIST_FILENAME = "gold_reserved.txt"

#: The machine-readable record, beside ``splits.json``.
RESERVATION_FILENAME = "gold_reservation.json"


class ReservationError(RuntimeError):
    """Raised when the reservation is malformed, or when training touches a reserved case.

    A :class:`RuntimeError` rather than a :class:`ValueError` on purpose: this is not a
    bad argument, it is a run that must not continue.
    """


@dataclass(frozen=True)
class GoldReservation:
    """The set of cases Gold holds out, and the evidence that it is what it says.

    Attributes:
        dataset: Substrate key the reservation belongs to.
        case_ids: Reserved case ids, sorted and unique.
        split: The split every reserved id belongs to. Always ``"test"``; carried
            explicitly so the record is self-describing rather than relying on a
            convention a reader has to know.
        id_list_sha256: Content hash over the sorted id set, same function Phase 2 uses.
        created_at: When the reservation was made, ISO-8601.
        provenance: How the sample was drawn — the stratification targets, the seed, the
            script version. Recorded so the sample is reproducible and reviewable.
        version: :data:`RESERVATION_VERSION`.
    """

    dataset: str
    case_ids: tuple[str, ...]
    split: str = "test"
    id_list_sha256: str = ""
    created_at: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    version: str = RESERVATION_VERSION

    def __post_init__(self) -> None:
        """Normalise the id list and compute its hash.

        Raises:
            ReservationError: If the list is empty or holds a duplicate. Both would make
                the count in the paper wrong, and the second silently.
        """
        ids = tuple(self.case_ids)
        if not ids:
            raise ReservationError(
                "a Gold reservation with no cases reserves nothing; write no file rather "
                "than an empty one, so a missing reservation is distinguishable from an "
                "empty one"
            )
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ReservationError(f"reserved case ids are not unique: {duplicates[:5]}")
        object.__setattr__(self, "case_ids", tuple(sorted(ids)))
        object.__setattr__(self, "id_list_sha256", hash_id_list(list(self.case_ids)))

    def __len__(self) -> int:
        """Return how many cases are reserved.

        Returns:
            The count.
        """
        return len(self.case_ids)

    @property
    def as_set(self) -> frozenset[str]:
        """Return the reserved ids as a set, for membership tests.

        Returns:
            The reserved case ids.
        """
        return frozenset(self.case_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised record.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "version": self.version,
            "dataset": self.dataset,
            "split": self.split,
            "n": len(self.case_ids),
            "case_ids": list(self.case_ids),
            "id_list_sha256": self.id_list_sha256,
            "created_at": self.created_at,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GoldReservation:
        """Rebuild a reservation from its serialised form.

        Args:
            payload: The mapping written by :meth:`to_dict`.

        Returns:
            The reservation.

        Raises:
            ReservationError: If the recorded hash does not match the id list, which
                means the file was edited by hand and the reservation can no longer be
                trusted to be the one the sample was drawn as.
        """
        reservation = cls(
            dataset=str(payload["dataset"]),
            case_ids=tuple(str(c) for c in payload["case_ids"]),
            split=str(payload.get("split", "test")),
            created_at=str(payload.get("created_at", "")),
            provenance=dict(payload.get("provenance") or {}),
            version=str(payload.get("version", RESERVATION_VERSION)),
        )
        recorded = str(payload.get("id_list_sha256", ""))
        if recorded and recorded != reservation.id_list_sha256:
            raise ReservationError(
                "the Gold reservation id list does not match its recorded sha256; the "
                "file has been edited. Re-run scripts/06_sample_gold_cases.py rather "
                "than repairing it by hand."
            )
        return reservation


def write_reservation(reservation: GoldReservation, manifest_dir: str | Path) -> Path:
    """Write the reservation into the frozen split manifest directory.

    Two artifacts, mirroring Phase 2's convention: a plain id list that is reviewable in a
    diff, and the machine-readable record this module reads back.

    Args:
        reservation: The reservation to record.
        manifest_dir: The substrate's manifest directory, normally
            ``schemas/splits/<substrate>``.

    Returns:
        The path of the written JSON record.

    Raises:
        OSError: If a write or rename fails.
    """
    out = Path(manifest_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / RESERVATION_LIST_FILENAME).write_text(
        "\n".join(reservation.case_ids) + "\n", encoding="utf-8"
    )
    return write_json(out / RESERVATION_FILENAME, reservation.to_dict(), canonical=True)


def load_reservation(
    manifest_dir: str | Path, *, split_assignment: dict[str, str] | None = None
) -> GoldReservation | None:
    """Read the reservation for a substrate, if one has been made.

    Args:
        manifest_dir: The substrate's manifest directory.
        split_assignment: Case id to split name, from
            :func:`g2t_aml.corpus.validate.load_split_manifest`. When given, every
            reserved id is asserted to be a member of the declared split — the check that
            makes the reservation meaningful rather than merely present.

    Returns:
        The reservation, or None when no Gold sample has been drawn for this substrate.
        None is a legitimate state, not an error: Phases 1-5 predate the reservation.

    Raises:
        ReservationError: If the record is malformed, its hash disagrees with its id
            list, or a reserved case is not in the declared split.
    """
    path = Path(manifest_dir) / RESERVATION_FILENAME
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except json.JSONDecodeError as exc:
        raise ReservationError(f"{path} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReservationError(f"{path} does not hold a reservation record")
    reservation = GoldReservation.from_dict(payload)

    listed = (Path(manifest_dir) / RESERVATION_LIST_FILENAME).read_text(encoding="utf-8").split()
    if tuple(sorted(listed)) != reservation.case_ids:
        raise ReservationError(
            f"{RESERVATION_LIST_FILENAME} and {RESERVATION_FILENAME} disagree about which "
            "cases are reserved; regenerate both rather than editing either"
        )

    if split_assignment is not None:
        misplaced = sorted(
            cid
            for cid in reservation.case_ids
            if split_assignment.get(cid, "<absent>") != reservation.split
        )
        if misplaced:
            raise ReservationError(
                f"{len(misplaced)} reserved cases are not in the {reservation.split!r} "
                f"split, so reserving them holds nothing out; first few: {misplaced[:5]}"
            )
    return reservation


def assert_not_reserved(
    case_ids: object,
    reservation: GoldReservation | None,
    *,
    context: str = "training",
) -> None:
    """Refuse a case set that touches the Gold test-only reservation.

    The enforcement point for "Gold test items are never trained on". Called by
    :func:`g2t_aml.corpus.training_data.load_training_records` on every load whose split
    is a training split, so the guarantee is a property of the loader rather than of the
    caller remembering.

    Args:
        case_ids: Any iterable of case ids about to be used.
        reservation: The reservation, or None when none has been made. None permits
            everything, because nothing has been held out yet.
        context: What the ids are about to be used for, quoted in the error.

    Raises:
        ReservationError: If any id is reserved. The message names the offenders, because
            the useful next question is always "which ones".
    """
    if reservation is None:
        return
    reserved = reservation.as_set
    offenders = sorted({str(cid) for cid in case_ids} & reserved)  # type: ignore[union-attr]
    if offenders:
        raise ReservationError(
            f"{len(offenders)} Gold test-only cases appeared in {context}. Gold is the "
            "held-out human reference and training on it makes every comparison against "
            f"it meaningless. First few: {offenders[:5]}"
        )
