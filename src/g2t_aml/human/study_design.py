"""The decision-setting study's design: who rates what, in which order, under what label.

Phase 12 asks a different question from Phase 6. Phase 6 asked people to *write* a
narrative; this asks them to *judge* one, and the thing being judged is the system that
produced it. That inverts every design concern. An annotator working alone on a case they
have never seen has no way to bias the result. A rater who sees the same case four times,
once per system, has every way: by the third rendering they are no longer scoring a
narrative, they are scoring the differences between it and the two they remember.

Three mechanisms answer that, and each is a hard constraint the validator enforces rather
than a hope.

**No rater sees a case twice.** This is stronger than the usual "no rater sees all systems
for the same case" and it is stronger deliberately. A partial repeat still anchors: a rater
who saw case 41 under one system and meets it again under another is comparing, and the
second rating is a difference score wearing the clothes of an absolute one. Every rater's
assignment is therefore case-disjoint, which is what makes
:class:`~g2t_aml.human.study_design.StudyDesign` an *incomplete* block design — no rater
covers the whole case x system grid, and none is meant to.

**Position is balanced, not merely randomised.** Order effects in a rating task are large
and one-directional: raters speed up, and the tenth item of a session gets less attention
than the first. Randomising order per rater makes that noise; balancing system against
position makes it *cancel*. Items are dealt in rounds of one-per-system and the system
order within round *i* for rater *r* is row ``(r + i) mod k`` of a cyclic Latin square, so
across the panel every system meets every within-round position equally often. The residual
freedom — which case fills which slot — is what the per-rater seed randomises.

**The system is not in the item.** A :class:`StudyItem` carries an opaque
:attr:`~StudyItem.item_id` and the narrative; which system wrote it lives in
:class:`BlindKey`, a separate structure the interface never loads. This is not politeness
about naming. A rater who can infer the arm from a field name, a file order or an id
prefix is an unblinded rater, and the item id is therefore a keyed digest rather than
anything reconstructible — sorting items by id tells you nothing, and neither does
comparing two of them.

The fourth mechanism is the one that measures whether the other three worked: **5% of each
rater's items are repeats of their own earlier items**, same case and same system, planted
far enough downstream to be out of working memory. Their rating differences are intra-rater
reliability, and a panel that cannot reproduce its own judgements is a panel whose
between-system differences mean nothing. It is the only check here that reports on the
raters rather than on the systems.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from g2t_aml.utils.io import atomic_path

__all__ = [
    "DEFAULT_ANCHOR_CASES",
    "DEFAULT_REPEAT_RATE",
    "MIN_REPEAT_SEPARATION",
    "STUDY_DESIGN_SCHEMA_VERSION",
    "BlindKey",
    "DesignError",
    "DesignReport",
    "StudyDesign",
    "StudyItem",
    "build_design",
    "load_design",
    "validate_design",
]

#: Version of the serialised design. A design is an experimental record: the analysis reads
#: it to know which ratings are repeats and which system produced which item, so a change
#: in its shape invalidates the mapping between a response file and its conditions.
STUDY_DESIGN_SCHEMA_VERSION = "1.0.0"

#: Fraction of each rater's workload that is a repeat of one of their own earlier items.
#: 5% is the brief's figure and it is a compromise: repeats cost real rating budget and buy
#: no between-system information, so the number is the smallest that still gives a usable
#: per-rater reliability estimate at the panel sizes in :func:`build_design`.
DEFAULT_REPEAT_RATE = 0.05

#: Cells that **every** rater rates, in addition to their own case-disjoint block.
#:
#: Without these there is no inter-rater agreement statistic. The rest of the design is
#: deliberately spread as thinly as possible over the case x system grid — that is what
#: maximises the number of cells covered and the breadth of the comparison — and the
#: consequence is that almost no cell is rated twice. Krippendorff's alpha is computed over
#: units that two or more raters judged, so a design optimised purely for breadth yields
#: *zero* pairable units and the alpha the Phase 12 gate requires cannot be computed at all.
#: This was found by running the analysis against a simulated response set and reading the
#: warning, not by inspecting the design.
#:
#: Twelve cells at a panel of eight gives 12 units with 8 coders each, which is a usable
#: alpha with a bootstrap interval that is not embarrassing. They are spread across systems
#: so each arm carries anchor units of its own, and they cost every rater twelve items of
#: workload that contribute nothing to the between-system breadth — which is the trade, and
#: it is worth it, because a Likert mean without an agreement statistic must not be
#: reported at all.
DEFAULT_ANCHOR_CASES = 12

#: Minimum number of intervening items between an item and its repeat. Below roughly this
#: distance a rater is recalling their earlier answer rather than re-forming a judgement,
#: which measures memory and reports it as reliability.
MIN_REPEAT_SEPARATION = 15


class DesignError(RuntimeError):
    """Raised when a design cannot be built or does not satisfy its own constraints."""


@dataclass(frozen=True)
class StudyItem:
    """One (rater, case, system) cell, as the rater will meet it.

    The system is deliberately **not** an attribute. It is carried by :class:`BlindKey`,
    which the interface does not load and the response store does not read. Everything
    here can be handed to the renderer without unblinding anything.

    Attributes:
        item_id: Opaque, keyed digest of (rater, case, system, occurrence). The rater's
            handle on the item and the join key the analysis uses afterwards. Not
            reconstructible without the study salt, so two ids reveal nothing about
            whether their systems match.
        rater_id: Pseudonymous rater identifier.
        case_id: The case whose graph and fact record are shown.
        dataset: Substrate key, needed to render the right fact families (invariant 4).
        position: 0-based position in this rater's sequence. Recorded because order
            effects are real and the analysis models them.
        round_index: Which round of one-per-system this item was dealt in. The unit the
            Latin square balances over.
        is_repeat: True when this is the second showing of an earlier item for this rater.
        repeat_of: The :attr:`item_id` this repeats, or None.
    """

    item_id: str
    rater_id: str
    case_id: str
    dataset: str
    position: int
    round_index: int
    is_repeat: bool = False
    repeat_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised item.

        Returns:
            A JSON-serialisable mapping. Contains no system identity.
        """
        return {
            "item_id": self.item_id,
            "rater_id": self.rater_id,
            "case_id": self.case_id,
            "dataset": self.dataset,
            "position": self.position,
            "round_index": self.round_index,
            "is_repeat": self.is_repeat,
            "repeat_of": self.repeat_of,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StudyItem:
        """Rebuild an item from its serialised form.

        Args:
            payload: One entry of a serialised design's ``items`` list.

        Returns:
            The item.

        Raises:
            DesignError: If a required field is missing or of the wrong type.
        """
        try:
            return cls(
                item_id=str(payload["item_id"]),
                rater_id=str(payload["rater_id"]),
                case_id=str(payload["case_id"]),
                dataset=str(payload["dataset"]),
                position=int(payload["position"]),
                round_index=int(payload["round_index"]),
                is_repeat=bool(payload.get("is_repeat", False)),
                repeat_of=(
                    str(payload["repeat_of"]) if payload.get("repeat_of") is not None else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DesignError(f"malformed study item: {exc}") from exc


@dataclass(frozen=True)
class BlindKey:
    """The item-to-system mapping, kept apart from everything a rater can reach.

    Stored in its own file so that "did the interface have access to the system identity?"
    is answerable by looking at which files it opens, rather than by auditing every code
    path for a leak. :func:`~g2t_aml.human.study_ui.render_item` is given items and never
    the key, and a test asserts the rendered output contains no system id.

    Attributes:
        assignments: ``item_id`` to system id, for every item in the design.
        salt: The keyed-digest salt the item ids were built with. Recorded so a design can
            be verified as internally consistent later, and so a lost key can be rebuilt
            from the design plus the salt rather than from nothing.
    """

    assignments: dict[str, str]
    salt: str

    def system_for(self, item_id: str) -> str:
        """Return the system that produced an item's narrative.

        Args:
            item_id: The opaque item identifier.

        Returns:
            The system id, for example ``"S1"`` or ``"Bronze"``.

        Raises:
            DesignError: If the item is not in this key, which means the key and the
                design do not belong to each other.
        """
        try:
            return self.assignments[item_id]
        except KeyError as exc:
            raise DesignError(
                f"item {item_id!r} is not in this blind key. The key and the design it is "
                "being used with are from different builds; unblinding with a mismatched "
                "key silently relabels every row of the results table."
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised key.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "schema_version": STUDY_DESIGN_SCHEMA_VERSION,
            "salt": self.salt,
            "assignments": dict(sorted(self.assignments.items())),
        }


@dataclass(frozen=True)
class DesignReport:
    """What the built design actually achieved, as against what was asked for.

    Every field is something a reviewer can ask about a block design and that cannot be
    recovered from the item list at a glance.

    Attributes:
        n_raters: Panel size.
        n_cases: Distinct cases in the stimulus pool.
        n_systems: Arms compared.
        items_per_rater: Rating items each rater receives, repeats included.
        n_repeats_per_rater: How many of those are repeats.
        cell_coverage: How many raters saw each (case, system) cell, as a count of cells
            per coverage level. ``{1: 320, 2: 80}`` means 320 cells rated once and 80
            twice.
        min_cell_coverage: The least-covered cell's coverage. Zero means some (case,
            system) pair is never rated, which is legal in an incomplete design but has to
            be known.
        n_anchor_cells: Cells every rater rated in common. The units inter-rater agreement
            is computed over; zero means no agreement statistic is obtainable.
        max_cell_coverage: The most-covered cell's coverage, which for a design with an
            anchor block equals the panel size.
        cases_per_system: System id to how many distinct cases were rated under it. **This
            is the number the Phase 12 gate is stated in** — "at least 80 cases x at
            least 4 systems" is a claim about this mapping, not about the size of the
            stimulus pool. An
            incomplete design leaves cells empty by construction, so the pool can be 100
            cases while a system is only ever seen on 61 of them, and quoting the pool size
            would overstate the study.
        per_rater_system_counts: Rater id to a system-id-to-count mapping. The check that
            no rater's workload is skewed toward one arm.
        position_balance: System id to the mean 0-based position of its items across the
            whole panel. These should be close together; a spread means one system is
            systematically rated when raters are fresher.
        seed: The master seed the per-rater orderings were derived from.
    """

    n_raters: int
    n_cases: int
    n_systems: int
    items_per_rater: int
    n_repeats_per_rater: int
    cell_coverage: dict[int, int]
    min_cell_coverage: int
    n_anchor_cells: int
    max_cell_coverage: int
    cases_per_system: dict[str, int]
    per_rater_system_counts: dict[str, dict[str, int]]
    position_balance: dict[str, float]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised report.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "n_raters": self.n_raters,
            "n_cases": self.n_cases,
            "n_systems": self.n_systems,
            "items_per_rater": self.items_per_rater,
            "n_repeats_per_rater": self.n_repeats_per_rater,
            "cell_coverage": {str(k): v for k, v in sorted(self.cell_coverage.items())},
            "min_cell_coverage": self.min_cell_coverage,
            "n_anchor_cells": self.n_anchor_cells,
            "max_cell_coverage": self.max_cell_coverage,
            "cases_per_system": self.cases_per_system,
            "per_rater_system_counts": self.per_rater_system_counts,
            "position_balance": self.position_balance,
            "seed": self.seed,
        }

    def summary(self) -> str:
        """Return a short human-readable account of the design.

        Returns:
            Several lines, suitable for a log or a console.
        """
        spread = (
            max(self.position_balance.values()) - min(self.position_balance.values())
            if self.position_balance
            else 0.0
        )
        lines = [
            f"{self.n_raters} raters x {self.items_per_rater} items "
            f"({self.n_repeats_per_rater} repeats) = "
            f"{self.n_raters * self.items_per_rater} ratings",
            f"{self.n_cases} cases x {self.n_systems} systems = "
            f"{self.n_cases * self.n_systems} cells, "
            f"min coverage {self.min_cell_coverage}",
            "coverage: "
            + ", ".join(f"{v} cells x{k}" for k, v in sorted(self.cell_coverage.items())),
            f"anchor cells rated by all {self.n_raters} raters: {self.n_anchor_cells} "
            "(the units inter-rater alpha is computed over)",
            "cases per system (the gate's number): "
            + ", ".join(f"{s}={n}" for s, n in sorted(self.cases_per_system.items())),
            f"position balance: mean-position spread {spread:.2f} across systems",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class StudyDesign:
    """A complete, validated assignment of items to raters.

    Attributes:
        items: Every item, ordered by rater then position.
        report: What the design achieved.
        systems: The system ids compared, in registry order.
        schema_version: :data:`STUDY_DESIGN_SCHEMA_VERSION`.
    """

    items: tuple[StudyItem, ...]
    report: DesignReport
    systems: tuple[str, ...]
    schema_version: str = STUDY_DESIGN_SCHEMA_VERSION

    def for_rater(self, rater_id: str) -> tuple[StudyItem, ...]:
        """Return one rater's sequence in presentation order.

        Args:
            rater_id: The pseudonym.

        Returns:
            Their items, ordered by position. Empty when the rater is not in the design.
        """
        return tuple(
            sorted((i for i in self.items if i.rater_id == rater_id), key=lambda i: i.position)
        )

    def raters(self) -> tuple[str, ...]:
        """Return every rater in the design.

        Returns:
            Their pseudonyms, sorted.
        """
        return tuple(sorted({i.rater_id for i in self.items}))

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised design.

        Returns:
            A JSON-serialisable mapping. Contains no system identity per item — only the
            list of systems in play, which is public.
        """
        return {
            "schema_version": self.schema_version,
            "systems": list(self.systems),
            "report": self.report.to_dict(),
            "items": [i.to_dict() for i in self.items],
        }

    def write(self, design_path: Path, key: BlindKey, key_path: Path) -> None:
        """Write the design and its blind key to two separate files.

        Two files rather than one, and the caller passes both paths, because the whole
        blinding argument rests on the interface being able to load one without the other.
        A single file with a ``system`` field the UI promises not to read is a promise; two
        files is a fact about what was opened.

        Args:
            design_path: Where the item list goes. Safe to hand to the interface.
            key: The item-to-system mapping.
            key_path: Where the key goes. Must not be readable by the rating session.

        Raises:
            DesignError: If the key does not cover exactly this design's items.
            OSError: If either write fails.
        """
        expected = {i.item_id for i in self.items}
        if set(key.assignments) != expected:
            raise DesignError(
                "the blind key does not cover exactly this design's items: "
                f"{len(expected - set(key.assignments))} unkeyed, "
                f"{len(set(key.assignments) - expected)} unknown"
            )
        for raw, payload in ((design_path, self.to_dict()), (key_path, key.to_dict())):
            path = Path(raw)
            path.parent.mkdir(parents=True, exist_ok=True)
            with atomic_path(path) as tmp:
                tmp.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )


def _item_id(salt: str, rater_id: str, case_id: str, system: str, occurrence: int) -> str:
    """Return the opaque identifier for one cell.

    Keyed on the study salt so the id cannot be recomputed — and therefore cannot be
    inverted to a system — by anyone holding only the design. Includes ``occurrence`` so a
    repeat is a distinct item with its own ratings rather than a second row under the same
    id, which would silently overwrite the first in any store keyed by item.

    Args:
        salt: The study salt.
        rater_id: The rater.
        case_id: The case.
        system: The system id. Enters the digest but is not recoverable from it.
        occurrence: 0 for the first showing, 1 for the repeat.

    Returns:
        A 16-character hex digest.
    """
    payload = f"{salt}|{rater_id}|{case_id}|{system}|{occurrence}"
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def _allocate_cells(
    case_ids: list[str],
    systems: list[str],
    rater_ids: list[str],
    unique_items_per_rater: int,
    anchors: list[tuple[str, str]],
) -> dict[str, list[tuple[str, str]]]:
    """Assign (case, system) cells to raters under the block constraints.

    Deterministic greedy, filling each rater's slots by repeatedly taking the cell that is
    least-covered so far, breaking ties toward the system this rater has seen least and
    then by a stable digest. Greedy rather than a closed-form cyclic construction because
    the study's parameters are not guaranteed to divide: 8 raters, 100 cases, 5 systems and
    a 45-item workload has no exact Latin-square solution, and a construction that only
    works when ``k | r`` would silently be replaced by an unbalanced fallback on the day
    the panel loses a rater.

    Args:
        case_ids: The stimulus pool, excluding the anchor cases.
        systems: The arms.
        rater_ids: The panel.
        unique_items_per_rater: Non-repeat items each rater receives from this pool,
            excluding their anchor items.
        anchors: The anchor cells every rater also receives. Passed in so their system
            counts seed each rater's quota: a rater whose anchors are three Bronze items
            needs three fewer Bronze items from the greedy pool, and ignoring that would
            skew every rater's workload by exactly the anchor distribution.

    Returns:
        Rater id to their list of ``(case_id, system)`` cells, unordered, anchors excluded.

    Raises:
        DesignError: If a rater cannot be filled without seeing a case twice.
    """
    if unique_items_per_rater > len(case_ids):
        raise DesignError(
            f"cannot give each rater {unique_items_per_rater} case-disjoint items from a "
            f"pool of {len(case_ids)} cases: no rater may see a case twice, so the "
            "workload can never exceed the number of cases"
        )
    coverage: Counter[tuple[str, str]] = Counter()
    anchor_counts: Counter[str] = Counter(s for _, s in anchors)
    total = unique_items_per_rater + len(anchors)
    out: dict[str, list[tuple[str, str]]] = {}

    for rater_index, rater_id in enumerate(rater_ids):
        # Exact per-system targets rather than a shared ceiling. A ceiling of
        # ``ceil(total / k)`` lets the first systems considered fill up to it and pushes the
        # shortfall onto the last, which produced workloads like 13/13/12/12/11 -- a spread
        # of two, and a rater effect on a skewed workload is indistinguishable from a system
        # effect. Targets sum to exactly the workload, so the spread can never exceed one.
        #
        # Which systems receive the remainder rotates with the rater index, so that over the
        # panel the extra item is not always the same arm's.
        targets: dict[str, int] = {}
        for j, system in enumerate(systems):
            share = total // len(systems)
            if (j - rater_index) % len(systems) < total % len(systems):
                share += 1
            targets[system] = share - anchor_counts[system]
            if targets[system] < 0:
                raise DesignError(
                    f"the anchor block gives {anchor_counts[system]} {system} items, more "
                    f"than the {share} that rater {rater_id!r}'s balanced workload allows"
                )

        seen_cases: set[str] = set()
        per_system: Counter[str] = Counter()
        chosen: list[tuple[str, str]] = []
        for _ in range(unique_items_per_rater):
            best: tuple[Any, ...] | None = None
            best_cell: tuple[str, str] | None = None
            for case_id in case_ids:
                if case_id in seen_cases:
                    continue
                for system in systems:
                    if per_system[system] >= targets[system]:
                        continue
                    cell = (case_id, system)
                    tiebreak = hashlib.blake2b(
                        f"{rater_id}|{case_id}|{system}".encode(), digest_size=4
                    ).hexdigest()
                    rank = (coverage[cell], per_system[system], tiebreak)
                    if best is None or rank < best:
                        best, best_cell = rank, cell
            if best_cell is None:
                raise DesignError(
                    f"ran out of admissible cells for rater {rater_id!r} after "
                    f"{len(chosen)} items"
                )
            chosen.append(best_cell)
            seen_cases.add(best_cell[0])
            per_system[best_cell[1]] += 1
            coverage[best_cell] += 1
        out[rater_id] = chosen
    return out


def _order_with_latin_square(
    cells: list[tuple[str, str]],
    systems: list[str],
    rater_index: int,
    rng: random.Random,
) -> list[tuple[tuple[str, str], int]]:
    """Order one rater's cells into rounds, balancing system against position.

    Deals the cells into rounds of at most one per system, then within round *i* orders the
    systems by row ``(rater_index + i) mod k`` of a cyclic Latin square. Which case fills a
    given system's slot is randomised; where the *system* sits is not.

    Args:
        cells: The rater's ``(case_id, system)`` cells.
        systems: The arms, in a fixed order.
        rater_index: This rater's index in the panel, the square's row offset.
        rng: Seeded generator for the residual case-level randomisation.

    Returns:
        A list of ``(cell, round_index)`` in presentation order.
    """
    by_system: dict[str, list[tuple[str, str]]] = {s: [] for s in systems}
    for cell in cells:
        by_system[cell[1]].append(cell)
    for bucket in by_system.values():
        rng.shuffle(bucket)

    ordered: list[tuple[tuple[str, str], int]] = []
    round_index = 0
    while any(by_system.values()):
        offset = rater_index + round_index
        rotation = [systems[(offset + j) % len(systems)] for j in range(len(systems))]
        for system in rotation:
            if by_system[system]:
                ordered.append((by_system[system].pop(), round_index))
        round_index += 1
    return ordered


def _plant_repeats(
    ordered: list[tuple[tuple[str, str], int]],
    n_repeats: int,
    rng: random.Random,
) -> list[tuple[tuple[str, str], int, int]]:
    """Insert repeats of the rater's own earlier items into the tail of their sequence.

    Repeats are drawn from the first third of the sequence and placed in the last third, so
    the separation is never near :data:`MIN_REPEAT_SEPARATION` by accident. A repeat placed
    close to its original measures recall; the point is to measure whether the rater's
    judgement reproduces.

    Args:
        ordered: The rater's sequence as ``(cell, round_index)``.
        n_repeats: How many repeats to plant.
        rng: Seeded generator.

    Returns:
        The sequence as ``(cell, round_index, occurrence)``, repeats interleaved, where
        ``occurrence`` is 0 for a first showing and 1 for a repeat.

    Raises:
        DesignError: If the sequence is too short to separate a repeat from its original.
    """
    if n_repeats == 0:
        return [(cell, rnd, 0) for cell, rnd in ordered]
    if len(ordered) < MIN_REPEAT_SEPARATION + n_repeats:
        raise DesignError(
            f"a {len(ordered)}-item sequence cannot carry {n_repeats} repeats at least "
            f"{MIN_REPEAT_SEPARATION} items from their originals. Either lengthen the "
            "workload or set the repeat rate to zero and report intra-rater reliability "
            "as not measured."
        )
    # Sources come from the head of the sequence, and no later than the point past which
    # MIN_REPEAT_SEPARATION items cannot fit before the end. Bounding the pool by the first
    # third alone is not enough: on a 19-item sequence the first third reaches index 5,
    # and an insertion point in the final third lands at 15, which is a separation of 10.
    # The validator caught that, which is the argument for the validator.
    latest_source = len(ordered) - MIN_REPEAT_SEPARATION
    source_pool = list(range(max(1, min(len(ordered) // 3, latest_source))))
    if len(source_pool) < n_repeats:
        raise DesignError(
            f"a {len(ordered)}-item sequence has only {len(source_pool)} positions that can "
            f"carry a repeat {MIN_REPEAT_SEPARATION} items later; {n_repeats} were asked for"
        )
    rng.shuffle(source_pool)
    sources = sorted(source_pool[:n_repeats])

    out: list[tuple[tuple[str, str], int, int]] = [(cell, rnd, 0) for cell, rnd in ordered]
    # Insertion points spread through the final third, and never closer to their own source
    # than the separation bound. Inserting only ever shifts later items rightwards, so an
    # already-placed repeat keeps the separation it was given.
    tail_start = max(MIN_REPEAT_SEPARATION, (2 * len(ordered)) // 3)
    step = max(1, (len(ordered) - tail_start) // max(1, n_repeats))
    for n, source in enumerate(sources):
        cell, rnd = ordered[source]
        at = min(len(out), max(tail_start + n * step + n, source + MIN_REPEAT_SEPARATION))
        out.insert(at, (cell, rnd, 1))
    return out


def build_design(
    case_ids: list[str],
    systems: list[str],
    rater_ids: list[str],
    *,
    dataset: str,
    items_per_rater: int,
    seed: int = 12345,
    salt: str = "g2t-aml-phase12",
    repeat_rate: float = DEFAULT_REPEAT_RATE,
    n_anchor_cases: int = DEFAULT_ANCHOR_CASES,
) -> tuple[StudyDesign, BlindKey]:
    """Build and validate a balanced incomplete block design for the rating study.

    Args:
        case_ids: The stimulus pool. 80-120 in the planned study.
        systems: The arms compared, for example ``["S1", "S2", "B7", "B3", "Bronze"]``.
        rater_ids: The panel, as pseudonyms.
        dataset: Substrate key recorded on every item.
        items_per_rater: Total items per rater, repeats included.
        seed: Master seed. Each rater's residual randomisation is derived from it, so the
            whole design is reproducible from this integer and the inputs.
        salt: Study salt for the keyed item ids.
        repeat_rate: Fraction of the workload that is a repeat.
        n_anchor_cases: How many cells every rater rates in common. See
            :data:`DEFAULT_ANCHOR_CASES` — without these the study has no inter-rater
            agreement statistic at all.

    Returns:
        The design and its blind key.

    Raises:
        DesignError: If the parameters admit no valid design, or the built design fails
            :func:`validate_design`.
    """
    if not case_ids or not systems or not rater_ids:
        raise DesignError("a design needs at least one case, one system and one rater")
    if len(set(case_ids)) != len(case_ids):
        raise DesignError("case_ids contains duplicates")
    if len(set(rater_ids)) != len(rater_ids):
        raise DesignError("rater_ids contains duplicates")
    if items_per_rater < len(systems):
        raise DesignError(
            f"{items_per_rater} items per rater cannot cover {len(systems)} systems even "
            "once; a rater who never sees an arm contributes nothing to its comparison"
        )

    n_repeats = round(items_per_rater * repeat_rate)
    unique = items_per_rater - n_repeats
    # The anchor block: cells every rater rates, spread evenly across systems so that each
    # arm carries anchor units of its own. Taken from the head of the pool, which is
    # arbitrary but fixed -- the pool arrives already stratified from Phase 6's sampler.
    #
    # Rounded up to a whole multiple of the system count. An anchor block of 12 across 5
    # systems gives 3/3/2/2/2, and that unevenness becomes a permanent per-system offset in
    # every rater's workload which the balanced-workload check then rejects. Rounding costs
    # at most k-1 items of everyone's time and makes the arithmetic exact.
    n_anchors = math.ceil(n_anchor_cases / len(systems)) * len(systems) if n_anchor_cases else 0
    if n_anchors >= min(unique, len(case_ids)):
        raise DesignError(
            f"{n_anchors} anchor cells (rounded up from {n_anchor_cases} to a multiple of "
            f"{len(systems)}) does not fit a {unique}-item workload over {len(case_ids)} cases"
        )

    anchors = [(case_ids[i], systems[i % len(systems)]) for i in range(n_anchors)]
    rest = list(case_ids[n_anchors:])
    allocation = _allocate_cells(rest, list(systems), list(rater_ids), unique - n_anchors, anchors)

    items: list[StudyItem] = []
    assignments: dict[str, str] = {}
    for rater_index, rater_id in enumerate(rater_ids):
        rng = random.Random(f"{seed}|{rater_id}")
        cells = [*anchors, *allocation[rater_id]]
        ordered = _order_with_latin_square(cells, list(systems), rater_index, rng)
        with_repeats = _plant_repeats(ordered, n_repeats, rng)

        first_id: dict[tuple[str, str], str] = {}
        for position, (cell, round_index, occurrence) in enumerate(with_repeats):
            case_id, system = cell
            item_id = _item_id(salt, rater_id, case_id, system, occurrence)
            if occurrence == 0:
                first_id[cell] = item_id
            items.append(
                StudyItem(
                    item_id=item_id,
                    rater_id=rater_id,
                    case_id=case_id,
                    dataset=dataset,
                    position=position,
                    round_index=round_index,
                    is_repeat=occurrence == 1,
                    repeat_of=first_id.get(cell) if occurrence == 1 else None,
                )
            )
            assignments[item_id] = system

    report = _build_report(items, assignments, list(case_ids), list(systems), list(rater_ids), seed)
    design = StudyDesign(items=tuple(items), report=report, systems=tuple(systems))
    key = BlindKey(assignments=assignments, salt=salt)
    validate_design(design, key)
    return design, key


def _build_report(
    items: list[StudyItem],
    assignments: dict[str, str],
    case_ids: list[str],
    systems: list[str],
    rater_ids: list[str],
    seed: int,
) -> DesignReport:
    """Compute the design's achieved balance.

    Args:
        items: Every item.
        assignments: Item-to-system mapping.
        case_ids: The stimulus pool.
        systems: The arms.
        rater_ids: The panel.
        seed: The master seed, recorded for reproducibility.

    Returns:
        The report.
    """
    coverage: Counter[tuple[str, str]] = Counter()
    per_rater: dict[str, Counter[str]] = {r: Counter() for r in rater_ids}
    positions: dict[str, list[int]] = {s: [] for s in systems}
    for item in items:
        system = assignments[item.item_id]
        if not item.is_repeat:
            coverage[item.case_id, system] += 1
            per_rater[item.rater_id][system] += 1
        positions[system].append(item.position)

    levels: Counter[int] = Counter()
    cases_per_system: Counter[str] = Counter()
    for case_id in case_ids:
        for system in systems:
            n = coverage[case_id, system]
            levels[n] += 1
            if n > 0:
                cases_per_system[system] += 1

    return DesignReport(
        n_raters=len(rater_ids),
        n_cases=len(case_ids),
        n_systems=len(systems),
        items_per_rater=len(items) // len(rater_ids),
        n_repeats_per_rater=sum(1 for i in items if i.is_repeat) // len(rater_ids),
        cell_coverage=dict(levels),
        min_cell_coverage=min(levels) if levels else 0,
        n_anchor_cells=sum(1 for n in coverage.values() if n >= len(rater_ids)),
        max_cell_coverage=max(coverage.values()) if coverage else 0,
        cases_per_system={s: cases_per_system[s] for s in systems},
        per_rater_system_counts={r: dict(c) for r, c in per_rater.items()},
        position_balance={s: round(sum(p) / len(p), 3) if p else 0.0 for s, p in positions.items()},
        seed=seed,
    )


def validate_design(design: StudyDesign, key: BlindKey) -> None:
    """Assert every constraint the design is supposed to satisfy.

    Called by :func:`build_design`, and worth calling again on a design loaded from disk
    before a session starts: a design that has been hand-edited between build and use is
    exactly the situation these checks exist for.

    The checks, and what each one is protecting:

    1. **No rater sees a case twice.** Anchoring. The single most damaging violation, and
       the reason this is an incomplete design at all.
    2. **No duplicate (rater, case, system) outside a declared repeat.** A silent duplicate
       is a repeat that the analysis will treat as an independent second opinion, inflating
       every reliability statistic.
    3. **Per-rater system counts are within one of each other.** A rater whose workload is
       skewed toward one arm contributes a rater effect that looks like a system effect.
    4. **Every system is seen by every rater.** Otherwise the Friedman test's blocks are
       incomplete in a way it does not model.
    5. **Every repeat matches its original's case and system, and clears the separation
       bound.** A repeat that differs in either is not a reliability measurement.
    6. **The key covers exactly the design's items.**

    Args:
        design: The design to check.
        key: Its blind key.

    Raises:
        DesignError: On the first violation found, naming the rater and item involved.
    """
    if set(key.assignments) != {i.item_id for i in design.items}:
        raise DesignError("the blind key does not cover exactly this design's items")

    by_id = {i.item_id: i for i in design.items}
    for rater_id in design.raters():
        sequence = design.for_rater(rater_id)
        if [i.position for i in sequence] != list(range(len(sequence))):
            raise DesignError(f"rater {rater_id!r} has non-contiguous item positions")

        seen_cases: set[str] = set()
        seen_cells: set[tuple[str, str]] = set()
        counts: Counter[str] = Counter()
        for item in sequence:
            system = key.system_for(item.item_id)
            if item.is_repeat:
                original = by_id.get(item.repeat_of or "")
                if original is None:
                    raise DesignError(
                        f"repeat {item.item_id!r} for rater {rater_id!r} names an original "
                        "that is not in the design"
                    )
                if original.case_id != item.case_id or key.system_for(original.item_id) != system:
                    raise DesignError(
                        f"repeat {item.item_id!r} for rater {rater_id!r} does not match its "
                        "original's case and system, so it measures nothing"
                    )
                if item.position - original.position < MIN_REPEAT_SEPARATION:
                    raise DesignError(
                        f"repeat {item.item_id!r} for rater {rater_id!r} sits "
                        f"{item.position - original.position} items after its original, "
                        f"under the {MIN_REPEAT_SEPARATION}-item bound: at that distance it "
                        "measures recall rather than reliability"
                    )
                continue

            if item.case_id in seen_cases:
                raise DesignError(
                    f"rater {rater_id!r} would see case {item.case_id!r} twice under "
                    "different systems. The second rating would be a comparison against "
                    "the first, not an independent judgement, and no amount of analysis "
                    "recovers from it."
                )
            if (item.case_id, system) in seen_cells:
                raise DesignError(
                    f"rater {rater_id!r} has an undeclared duplicate of "
                    f"({item.case_id!r}, {system!r})"
                )
            seen_cases.add(item.case_id)
            seen_cells.add((item.case_id, system))
            counts[system] += 1

        missing = set(design.systems) - set(counts)
        if missing:
            raise DesignError(
                f"rater {rater_id!r} never sees {sorted(missing)}; the Friedman test's "
                "blocks would be incomplete in a way it does not model"
            )
        if max(counts.values()) - min(counts.values()) > 1:
            raise DesignError(
                f"rater {rater_id!r} has an unbalanced workload across systems: "
                f"{dict(counts)}. A rater effect on a skewed workload is indistinguishable "
                "from a system effect."
            )


def load_design(design_path: Path) -> StudyDesign:
    """Read a design from disk, without its key.

    The function the interface calls. There is deliberately no ``load_key`` beside it: the
    key is read by the analysis, from :mod:`g2t_aml.human.study_analysis`, and keeping the
    loader out of this module's public surface is one more reason a rating session cannot
    accidentally acquire one.

    Args:
        design_path: The design file.

    Returns:
        The design.

    Raises:
        DesignError: If the file is missing, malformed, or of an unknown schema version.
    """
    path = Path(design_path)
    if not path.is_file():
        raise DesignError(f"no design at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DesignError(f"{path}: malformed JSON: {exc}") from exc

    version = str(payload.get("schema_version", ""))
    if version != STUDY_DESIGN_SCHEMA_VERSION:
        raise DesignError(
            f"{path}: design schema version {version!r}, expected "
            f"{STUDY_DESIGN_SCHEMA_VERSION!r}"
        )
    if "assignments" in payload:
        raise DesignError(
            f"{path} carries an 'assignments' block, which is a blind key. This file is "
            "loaded by the rating interface; a key in it unblinds the study."
        )
    report = payload.get("report") or {}
    return StudyDesign(
        items=tuple(StudyItem.from_dict(i) for i in payload.get("items") or ()),
        report=DesignReport(
            n_raters=int(report.get("n_raters", 0)),
            n_cases=int(report.get("n_cases", 0)),
            n_systems=int(report.get("n_systems", 0)),
            items_per_rater=int(report.get("items_per_rater", 0)),
            n_repeats_per_rater=int(report.get("n_repeats_per_rater", 0)),
            cell_coverage={int(k): int(v) for k, v in (report.get("cell_coverage") or {}).items()},
            min_cell_coverage=int(report.get("min_cell_coverage", 0)),
            n_anchor_cells=int(report.get("n_anchor_cells", 0)),
            max_cell_coverage=int(report.get("max_cell_coverage", 0)),
            cases_per_system=dict(report.get("cases_per_system") or {}),
            per_rater_system_counts=dict(report.get("per_rater_system_counts") or {}),
            position_balance=dict(report.get("position_balance") or {}),
            seed=int(report.get("seed", 0)),
        ),
        systems=tuple(payload.get("systems") or ()),
        schema_version=version,
    )
