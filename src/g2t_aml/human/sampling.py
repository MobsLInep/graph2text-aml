"""Choosing which cases become Gold, and why those.

Gold costs roughly fifteen minutes of skilled attention per item. At 300-400 items that is
75-100 person-hours, so the sample is the single most expensive decision in the phase and
the one least able to be revised: a stratum discovered to be missing in week six cannot be
back-filled without another round of recruitment.

Four constraints, and they do not all fit at once on this data.

**Typology balance.** All eight AMLworld typologies plus ``unclassified``, so the human
reference can say something about each. The binding constraint is supply: the AMLworld test
split holds 19 ``stack`` cases and 88 ``gather_scatter`` ones, so "balanced" has to mean
*evenly allocated under capacity*, with a stratum that cannot fill its share giving the
remainder back. :func:`~g2t_aml.data.case_sampling.allocate_evenly` already implements
exactly that for Phase 2 and is reused rather than reimplemented.

**Hard negatives over-represented, at least 25%.** These are the cases where a system
fails and where a human reference is worth the most: legitimate activity whose *shape*
looks like laundering. A narrative that escalates a payroll run is the error that costs a
compliance team its credibility, and no template or model can be shown to avoid it without
human-written examples of the restraint.

**Those two constraints collide, and the collision is a fact about the data rather than a
modelling choice.** Every hard negative is licit, so it carries no laundering stream, so
its ``typology.label`` is ``unclassified`` — 839 of the 839 hard negatives in the AMLworld
test split. An even nine-way allocation over typologies therefore caps ``unclassified`` at
about a ninth of the budget and makes a 25% hard-negative floor unreachable. The floor
wins: the sample is allocated as three explicit blocks — a hard-negative block, a typed
block balanced across the eight, and an ordinary-unclassified block — with the shares named
in the parameters rather than emerging from an interaction nobody wrote down. See D-052.

**Case size spread.** A corpus of six-account cases teaches an annotator nothing about a
sixty-account one. Selection inside every block round-robins over size buckets before it
falls back to the deterministic order, so each block spans the range rather than
concentrating wherever the stratum happens to be dense.

Everything is drawn from the **test** split, because Gold's job is to be a held-out
reference, and the selection is written into the split manifest as a reservation
(:mod:`g2t_aml.human.reservation`) that the training loader refuses to cross.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from g2t_aml.data.case_sampling import allocate_evenly

__all__ = [
    "SIZE_BUCKETS",
    "TYPED_TYPOLOGIES",
    "GoldCandidate",
    "GoldSample",
    "GoldSamplingError",
    "GoldSamplingParams",
    "sample_gold_cases",
    "size_bucket_of",
]

#: The eight typologies that carry a structural story of their own. ``unclassified`` is
#: handled as its own two blocks and is deliberately not a member.
TYPED_TYPOLOGIES: tuple[str, ...] = (
    "fan_out",
    "fan_in",
    "gather_scatter",
    "scatter_gather",
    "cycle",
    "random",
    "bipartite",
    "stack",
)

#: Size buckets over ``structure.n_nodes``, as ``(name, lower, upper)`` with both bounds
#: inclusive and ``None`` meaning unbounded. The boundaries follow the test split's own
#: quartiles (median 4, p75 11) rather than round numbers, so each bucket is populated.
SIZE_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("small", 2, 5),
    ("medium", 6, 20),
    ("large", 21, None),
)


class GoldSamplingError(RuntimeError):
    """Raised when the requested sample cannot be drawn from the population offered."""


def size_bucket_of(n_nodes: int) -> str:
    """Return the size bucket a case falls in.

    Args:
        n_nodes: The case's account count.

    Returns:
        The bucket name.

    Raises:
        GoldSamplingError: If ``n_nodes`` is below the smallest bucket's floor. A
            single-account case has no counterparty in scope and nothing to narrate; it is
            excluded from Bronze for the same reason (D-038) and must not silently become
            a Gold item an annotator cannot write.
    """
    for name, lower, upper in SIZE_BUCKETS:
        if n_nodes >= lower and (upper is None or n_nodes <= upper):
            return name
    raise GoldSamplingError(
        f"a case with {n_nodes} accounts is below the smallest Gold size bucket; "
        "single-account cases carry no describable activity"
    )


@dataclass(frozen=True)
class GoldCandidate:
    """One case that could become a Gold item, with everything stratification reads.

    Attributes:
        case_id: The case.
        dataset: Substrate key.
        split: The split it belongs to, from the frozen manifest.
        typology: ``typology.label`` from the *fact record*, not the Phase 2 case index.
            The fact record is what the salience list keys on, so it is what the sample
            must balance (D-036).
        case_class: ``suspicious``, ``licit`` or ``hard_negative``, from Phase 2.
        n_nodes: Account count.
        n_edges: Transaction count.
    """

    case_id: str
    dataset: str
    split: str
    typology: str
    case_class: str
    n_nodes: int
    n_edges: int

    @property
    def size_bucket(self) -> str:
        """Return the case's size bucket.

        Returns:
            The bucket name.

        Raises:
            GoldSamplingError: If the case is too small to bucket.
        """
        return size_bucket_of(self.n_nodes)

    @property
    def block(self) -> str:
        """Return which of the three allocation blocks this candidate belongs to.

        Returns:
            ``"hard_negative"`` for a hard negative, ``"typed"`` for one of the eight
            typologies, ``"unclassified"`` otherwise.
        """
        if self.case_class == "hard_negative":
            return "hard_negative"
        if self.typology in TYPED_TYPOLOGIES:
            return "typed"
        return "unclassified"

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised candidate.

        Returns:
            A JSON-serialisable mapping, size bucket and block included.
        """
        return {
            "case_id": self.case_id,
            "dataset": self.dataset,
            "split": self.split,
            "typology": self.typology,
            "case_class": self.case_class,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "size_bucket": self.size_bucket,
            "block": self.block,
        }


@dataclass(frozen=True)
class GoldSamplingParams:
    """What the sample is asked to be.

    The three block shares are named rather than derived, because the derivation would
    otherwise encode the typology/hard-negative collision as an accident. They are
    normalised if they do not sum to one, and the hard-negative floor is asserted against
    the result.

    Attributes:
        n_cases: Total Gold items to select across all substrates.
        min_reserved: The reservation must hold at least this many cases, all test-only.
        hard_negative_share: Share of the budget drawn from hard negatives.
        typed_share: Share allocated evenly across :data:`TYPED_TYPOLOGIES`.
        unclassified_share: Share drawn from ordinary unclassified cases — licit ones and
            suspicious ones whose case carries no typology.
        hard_negative_floor: The minimum the brief sets. Asserted after allocation, so a
            substrate short of hard negatives fails loudly rather than quietly.
        substrate_shares: Substrate key to its share of the budget.
        reallocate_deficit: Whether a substrate that cannot fill its quota gives it to
            the substrates that can. True by default, and the deficit is reported either
            way: Elliptic2's 30% share is unobtainable until access is granted, and
            annotating 245 items instead of 350 because of it would spend the shortfall
            on nothing. The reallocation is recorded as its own entry so the sample is
            never mistaken for one that met the substrate split (D-052).
        split: The split every candidate must come from.
        seed: Selects the deterministic order inside a stratum.
    """

    n_cases: int = 350
    min_reserved: int = 200
    hard_negative_share: float = 0.28
    typed_share: float = 0.44
    unclassified_share: float = 0.28
    hard_negative_floor: float = 0.25
    substrate_shares: dict[str, float] = field(
        default_factory=lambda: {"amlworld_hi_small": 0.7, "elliptic2": 0.3}
    )
    reallocate_deficit: bool = True
    split: str = "test"
    seed: int = 1337

    def __post_init__(self) -> None:
        """Validate and normalise the shares.

        Raises:
            GoldSamplingError: If the budget is not positive, a share is negative, the
                block shares are all zero, or the hard-negative share is below its own
                floor — which would make the floor unsatisfiable before a single case had
                been looked at.
        """
        if self.n_cases <= 0:
            raise GoldSamplingError("n_cases must be positive")
        shares = (self.hard_negative_share, self.typed_share, self.unclassified_share)
        if any(s < 0 for s in shares):
            raise GoldSamplingError("block shares must be non-negative")
        total = sum(shares)
        if total <= 0:
            raise GoldSamplingError("at least one block share must be positive")
        if self.hard_negative_share / total < self.hard_negative_floor:
            raise GoldSamplingError(
                f"hard_negative_share normalises to "
                f"{self.hard_negative_share / total:.3f}, below the "
                f"{self.hard_negative_floor:.2f} floor the annotation brief sets"
            )
        object.__setattr__(self, "hard_negative_share", self.hard_negative_share / total)
        object.__setattr__(self, "typed_share", self.typed_share / total)
        object.__setattr__(self, "unclassified_share", self.unclassified_share / total)

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised parameters.

        Returns:
            A JSON-serialisable mapping, shares already normalised.
        """
        return {
            "n_cases": self.n_cases,
            "min_reserved": self.min_reserved,
            "hard_negative_share": round(self.hard_negative_share, 6),
            "typed_share": round(self.typed_share, 6),
            "unclassified_share": round(self.unclassified_share, 6),
            "hard_negative_floor": self.hard_negative_floor,
            "substrate_shares": dict(sorted(self.substrate_shares.items())),
            "reallocate_deficit": self.reallocate_deficit,
            "split": self.split,
            "seed": self.seed,
            "typed_typologies": list(TYPED_TYPOLOGIES),
            "size_buckets": [
                {"name": n, "min_nodes": lo, "max_nodes": hi} for n, lo, hi in SIZE_BUCKETS
            ],
        }


@dataclass(frozen=True)
class GoldSample:
    """The drawn sample and the evidence that it is what was asked for.

    Attributes:
        selected: The chosen candidates, sorted by case id.
        params: The parameters it was drawn under.
        by_typology: Selected count per typology.
        by_class: Selected count per case class.
        by_size_bucket: Selected count per size bucket.
        by_dataset: Selected count per substrate.
        by_block: Selected count per allocation block.
        deficits: Where the population could not supply what was asked, as
            ``stratum -> (requested, supplied)``. **Always reported, never silently
            absorbed**: an unmet Elliptic2 quota is the single most important thing a
            reader of this sample needs to know, and it is the one a quiet re-allocation
            would hide.
    """

    selected: tuple[GoldCandidate, ...]
    params: GoldSamplingParams
    by_typology: dict[str, int]
    by_class: dict[str, int]
    by_size_bucket: dict[str, int]
    by_dataset: dict[str, int]
    by_block: dict[str, int]
    deficits: dict[str, tuple[int, int]]

    def __len__(self) -> int:
        """Return how many cases were selected.

        Returns:
            The count.
        """
        return len(self.selected)

    @property
    def case_ids(self) -> tuple[str, ...]:
        """Return the selected case ids.

        Returns:
            The ids, sorted.
        """
        return tuple(c.case_id for c in self.selected)

    @property
    def hard_negative_rate(self) -> float:
        """Return the achieved hard-negative share.

        Returns:
            Hard negatives over total, or 0.0 over an empty sample.
        """
        return self.by_class.get("hard_negative", 0) / len(self.selected) if self.selected else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return the machine-readable sample report.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "n_selected": len(self.selected),
            "params": self.params.to_dict(),
            "by_typology": dict(sorted(self.by_typology.items())),
            "by_class": dict(sorted(self.by_class.items())),
            "by_size_bucket": dict(sorted(self.by_size_bucket.items())),
            "by_dataset": dict(sorted(self.by_dataset.items())),
            "by_block": dict(sorted(self.by_block.items())),
            "hard_negative_rate": round(self.hard_negative_rate, 6),
            "deficits": {
                name: {"requested": req, "supplied": got}
                for name, (req, got) in sorted(self.deficits.items())
            },
            "case_ids": list(self.case_ids),
        }

    def summary(self) -> str:
        """Return the human-readable summary.

        Returns:
            A short report suitable for a terminal or the phase log.
        """
        lines = [
            f"Gold sample: {len(self.selected)} of {self.params.n_cases} requested "
            f"from the {self.params.split!r} split",
            f"  hard negatives  {self.by_class.get('hard_negative', 0)} "
            f"({self.hard_negative_rate:.1%}, floor {self.params.hard_negative_floor:.0%})",
            "",
            f"  {'typology':<18} {'n':>5}",
            f"  {'-' * 18} {'-' * 5}",
        ]
        lines += [f"  {name:<18} {n:>5}" for name, n in sorted(self.by_typology.items())]
        lines += [
            "",
            "  size    " + "  ".join(f"{k} {v}" for k, v in sorted(self.by_size_bucket.items())),
            "  dataset " + "  ".join(f"{k} {v}" for k, v in sorted(self.by_dataset.items())),
        ]
        if self.deficits:
            lines += ["", "  DEFICITS (requested vs supplied):"]
            lines += [
                f"    {name}: {req} requested, {got} supplied"
                for name, (req, got) in sorted(self.deficits.items())
            ]
        return "\n".join(lines)


def _order_key(candidate: GoldCandidate, seed: int) -> str:
    """Return the deterministic shuffle key for a candidate.

    A digest rather than a seeded RNG draw so the order depends only on the seed and the
    case id: adding a case to the population does not renumber the draws for the others,
    which is what lets a sample be extended without invalidating the items already
    annotated under it.

    Args:
        candidate: The candidate.
        seed: The run seed.

    Returns:
        A hex digest used as a sort key.
    """
    return hashlib.sha256(f"{seed}:{candidate.case_id}".encode()).hexdigest()


def _take_spread(pool: list[GoldCandidate], n: int, seed: int) -> list[GoldCandidate]:
    """Take ``n`` candidates, round-robining over size buckets before falling back.

    Args:
        pool: The stratum's candidates.
        n: How many to take.
        seed: The run seed, selecting the order inside a bucket.

    Returns:
        The taken candidates, at most ``min(n, len(pool))`` of them.
    """
    if n <= 0 or not pool:
        return []
    buckets: dict[str, list[GoldCandidate]] = defaultdict(list)
    for candidate in sorted(pool, key=lambda c: _order_key(c, seed)):
        buckets[candidate.size_bucket].append(candidate)

    taken: list[GoldCandidate] = []
    order = [name for name, _, _ in SIZE_BUCKETS if buckets[name]]
    while len(taken) < n and order:
        for name in list(order):
            if len(taken) >= n:
                break
            if not buckets[name]:
                order.remove(name)
                continue
            taken.append(buckets[name].pop(0))
    return taken


def _sample_one_substrate(
    candidates: list[GoldCandidate],
    budget: int,
    params: GoldSamplingParams,
    deficits: dict[str, tuple[int, int]],
    dataset: str,
) -> list[GoldCandidate]:
    """Draw one substrate's share of the sample.

    Args:
        candidates: Every eligible candidate for this substrate.
        budget: This substrate's share of the total.
        params: The sampling parameters.
        deficits: Shortfall map, extended in place.
        dataset: The substrate key, used to name a deficit.

    Returns:
        The selected candidates.
    """
    if budget <= 0:
        return []
    if not candidates:
        deficits[f"dataset:{dataset}"] = (budget, 0)
        return []

    by_block: dict[str, list[GoldCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_block[candidate.block].append(candidate)

    hard_target = math.ceil(budget * params.hard_negative_share)
    typed_target = round(budget * params.typed_share)
    unclassified_target = max(0, budget - hard_target - typed_target)

    selected: list[GoldCandidate] = []

    # Hard negatives first: the floor is the constraint the other blocks give way to.
    hard = _take_spread(by_block["hard_negative"], hard_target, params.seed)
    if len(hard) < hard_target:
        deficits[f"{dataset}:hard_negative"] = (hard_target, len(hard))
    selected += hard

    # The typed block, allocated evenly across the eight under capacity.
    typed_pool: dict[str, list[GoldCandidate]] = defaultdict(list)
    for candidate in by_block["typed"]:
        typed_pool[candidate.typology].append(candidate)
    capacities = {name: len(typed_pool[name]) for name in TYPED_TYPOLOGIES}
    allocation = allocate_evenly(typed_target, capacities)
    for name in TYPED_TYPOLOGIES:
        want = allocation[name]
        got = _take_spread(typed_pool[name], want, params.seed)
        if capacities[name] < math.floor(typed_target / len(TYPED_TYPOLOGIES)):
            deficits[f"{dataset}:typology:{name}"] = (
                math.floor(typed_target / len(TYPED_TYPOLOGIES)),
                capacities[name],
            )
        selected += got

    # Whatever the typed block could not absorb is spent on ordinary unclassified cases
    # rather than left on the table: a smaller sample is worse than a differently
    # weighted one, and the weighting is reported.
    unclassified_target += typed_target - sum(allocation.values())
    ordinary = _take_spread(by_block["unclassified"], unclassified_target, params.seed)
    if len(ordinary) < unclassified_target:
        deficits[f"{dataset}:unclassified"] = (unclassified_target, len(ordinary))
    selected += ordinary
    return selected


def sample_gold_cases(
    candidates: list[GoldCandidate], params: GoldSamplingParams | None = None
) -> GoldSample:
    """Draw the Gold sample.

    Args:
        candidates: Every case eligible for Gold. Candidates outside ``params.split`` and
            candidates too small to bucket are dropped before allocation, and the drop is
            reported as a deficit rather than passing silently.
        params: The sampling parameters. Defaults are the ones the phase brief specifies.

    Returns:
        The sample, with its stratification counts and every shortfall.

    Raises:
        GoldSamplingError: If the achieved hard-negative share falls below
            ``params.hard_negative_floor``, or fewer than ``params.min_reserved`` cases
            could be drawn. Both are reasons to fix the population or the parameters, not
            to proceed: the first removes the stratum the phase brief calls out as the one
            that matters most, and the second makes the reservation too small to be a
            held-out reference.
    """
    params = params or GoldSamplingParams()
    deficits: dict[str, tuple[int, int]] = {}

    eligible: list[GoldCandidate] = []
    n_wrong_split = 0
    n_too_small = 0
    for candidate in candidates:
        if candidate.split != params.split:
            n_wrong_split += 1
            continue
        try:
            _ = candidate.size_bucket
        except GoldSamplingError:
            n_too_small += 1
            continue
        eligible.append(candidate)
    if n_too_small:
        deficits["excluded:below_min_nodes"] = (n_too_small, 0)

    by_dataset: dict[str, list[GoldCandidate]] = defaultdict(list)
    for candidate in eligible:
        by_dataset[candidate.dataset].append(candidate)

    selected: list[GoldCandidate] = []
    for dataset in sorted(params.substrate_shares):
        share = params.substrate_shares[dataset]
        budget = round(params.n_cases * share)
        selected += _sample_one_substrate(
            by_dataset.get(dataset, []), budget, params, deficits, dataset
        )

    # A substrate that could not fill its quota hands it to the ones that can, unless
    # told not to. The deficit stays in the report: the reallocation changes the sample's
    # size, never the account of what it does and does not cover.
    shortfall = params.n_cases - len(selected)
    if params.reallocate_deficit and shortfall > 0:
        taken = {c.case_id for c in selected}
        remaining = {
            dataset: [c for c in pool if c.case_id not in taken]
            for dataset, pool in by_dataset.items()
        }
        with_room = sorted(d for d, pool in remaining.items() if pool)
        if with_room:
            weights = {d: params.substrate_shares.get(d, 0.0) or 1.0 for d in with_room}
            total_weight = sum(weights.values())
            topped_up: dict[str, int] = {}
            for dataset in with_room:
                budget = round(shortfall * weights[dataset] / total_weight)
                extra = _sample_one_substrate(remaining[dataset], budget, params, {}, dataset)
                topped_up[dataset] = len(extra)
                selected += extra
            deficits["reallocated"] = (shortfall, sum(topped_up.values()))

    selected.sort(key=lambda c: c.case_id)
    sample = GoldSample(
        selected=tuple(selected),
        params=params,
        by_typology=dict(Counter(c.typology for c in selected)),
        by_class=dict(Counter(c.case_class for c in selected)),
        by_size_bucket=dict(Counter(c.size_bucket for c in selected)),
        by_dataset=dict(Counter(c.dataset for c in selected)),
        by_block=dict(Counter(c.block for c in selected)),
        deficits=deficits,
    )

    if sample.hard_negative_rate < params.hard_negative_floor:
        raise GoldSamplingError(
            f"the sample is {sample.hard_negative_rate:.1%} hard negatives, below the "
            f"{params.hard_negative_floor:.0%} floor. Hard negatives are where a system "
            "fails and where the human reference is worth the most; a Gold set without "
            "them cannot show restraint being exercised."
        )
    if len(sample) < params.min_reserved:
        raise GoldSamplingError(
            f"only {len(sample)} cases could be drawn, fewer than the "
            f"{params.min_reserved} the reservation requires. Deficits: "
            f"{sample.deficits}"
        )
    return sample
