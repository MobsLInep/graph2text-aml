"""Loading a corpus for training, with the Gold hold-out enforced in the loader.

Phases 7-9 read a corpus JSONL and hand it to an encoder or a finetuner. This module is
the one place that happens, and it exists so that two guarantees are properties of the
code rather than of whoever writes the training script:

1. **A Gold test-only case never reaches training.** The reservation
   (:mod:`g2t_aml.human.reservation`) is loaded from the frozen split manifest directory
   and asserted against every training load. A reserved case in a training batch raises
   before a single gradient step.
2. **The Gold tier never reaches training at all**, in any split. Gold is 250-400
   human-authored narratives; used as supervision they would be a rounding error on the
   loss and a catastrophe for the evaluation, because the held-out reference would then
   be text the model was fitted on.

Both failures are silent in every metric this project reports. A memorised target is
*more* faithful to its fact record, not less, so faithfulness rises; adequacy rises;
surface overlap against the reference rises most of all. There is no number in the paper
that would go the wrong way, which is exactly why the check has to be mechanical.

The loader is deliberately thin — it returns the serialised records, not tensors. Turning
a record into model input is Phase 7/9's business and depends on the encoder; refusing to
guess at that now keeps this module useful to both.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from g2t_aml.human.reservation import GoldReservation, ReservationError, assert_not_reserved
from g2t_aml.utils.io import read_jsonl

__all__ = [
    "GOLD_TIER",
    "TRAINING_SPLITS",
    "CorpusSlice",
    "load_training_records",
]

#: The splits a model is fitted on. ``test`` is not one of them, and a load for ``test``
#: skips the reservation check — reading a reserved case *as evaluation* is the entire
#: point of reserving it.
TRAINING_SPLITS: tuple[str, ...] = ("train", "val")

#: The tier that must never be trained on, in any split.
GOLD_TIER = "gold"


@dataclass(frozen=True)
class CorpusSlice:
    """The records for one split, with what was read and what was refused.

    Attributes:
        split: The split requested.
        records: The serialised training records, in file order.
        by_tier: How many records came from each tier.
        n_skipped: Records in the file belonging to another split.
        source: The corpus file read.
    """

    split: str
    records: list[dict[str, Any]]
    by_tier: dict[str, int]
    n_skipped: int
    source: Path

    def __len__(self) -> int:
        """Return how many records the slice holds.

        Returns:
            The record count.
        """
        return len(self.records)

    @property
    def case_ids(self) -> tuple[str, ...]:
        """Return the case ids in the slice, in file order.

        Returns:
            The case ids.
        """
        return tuple(str(r["case_id"]) for r in self.records)


def load_training_records(
    corpus_path: str | Path,
    *,
    split: str,
    reservation: GoldReservation | None = None,
    allow_gold: bool = False,
) -> CorpusSlice:
    """Read one split out of a corpus file, refusing anything Gold holds out.

    Args:
        corpus_path: A corpus JSONL — ``bronze.jsonl``, ``silver.jsonl``, ``gold.jsonl``.
        split: The split to load. When it is one of :data:`TRAINING_SPLITS`, both Gold
            guards apply.
        reservation: The Gold test-only reservation, normally from
            :func:`g2t_aml.human.reservation.load_reservation`. None means none has been
            made, and nothing is held out.
        allow_gold: Permit Gold-tier records in the result. Only ever true for an
            *evaluation* load; setting it on a training split raises rather than being
            honoured, because the combination has no legitimate meaning.

    Returns:
        The slice.

    Raises:
        FileNotFoundError: If the corpus file is absent.
        ReservationError: If a reserved case, or a Gold-tier record, would enter
            training.
        ValueError: If ``allow_gold`` is set on a training split.
    """
    path = Path(corpus_path)
    if not path.is_file():
        raise FileNotFoundError(f"corpus file {path} does not exist")

    is_training = split in TRAINING_SPLITS
    if is_training and allow_gold:
        raise ValueError(
            f"allow_gold=True was passed for split {split!r}, which is a training split. "
            "Gold is the held-out human reference; there is no configuration in which "
            "training on it is correct."
        )

    records: list[dict[str, Any]] = []
    tiers: Counter[str] = Counter()
    skipped = 0
    for payload in read_jsonl(path):
        if not isinstance(payload, dict):
            raise ValueError(f"{path} holds a line that is not a training record")
        if str(payload.get("split")) != split:
            skipped += 1
            continue
        records.append(payload)
        tiers[str(payload.get("tier", "<unknown>"))] += 1

    if is_training and tiers.get(GOLD_TIER):
        offenders = sorted(str(r["case_id"]) for r in records if str(r.get("tier")) == GOLD_TIER)
        raise ReservationError(
            f"{len(offenders)} Gold-tier records are marked split={split!r} in {path}. "
            "Gold is never training supervision, whatever split a record claims. First "
            f"few: {offenders[:5]}"
        )
    if not allow_gold and tiers.get(GOLD_TIER):
        raise ReservationError(
            f"{tiers[GOLD_TIER]} Gold-tier records were read from {path} without "
            "allow_gold=True. Pass it explicitly on an evaluation load, so that reading "
            "the human reference is always a deliberate act."
        )

    if is_training:
        assert_not_reserved(
            (str(r["case_id"]) for r in records), reservation, context=f"the {split!r} split"
        )

    return CorpusSlice(
        split=split,
        records=records,
        by_tier=dict(sorted(tiers.items())),
        n_skipped=skipped,
        source=path,
    )
