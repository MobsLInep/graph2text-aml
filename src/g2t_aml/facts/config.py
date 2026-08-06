"""Every threshold the fact layer uses, in one frozen, serialisable place.

A motif detector's ``present`` flag is a *decision*, and a decision made against an
unrecorded threshold is not reproducible. :class:`FactConfig` is therefore written into
``provenance.config`` on every fact record, so a reviewer can always re-derive why a
detector fired without reading the code that fired it.

Defaults are taken from the substrate wherever the substrate offers a number, and the
docstring on each field says which. Where no substrate number exists the default is stated
as a judgement with its reasoning, so it can be argued with rather than guessed at.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

__all__ = ["FactConfig", "ToleranceConfig"]


@dataclass(frozen=True)
class ToleranceConfig:
    """The published tolerance policy, per claim type.

    This table is a commitment: it goes in the paper, and a reviewer must be able to check
    that the reported faithfulness number was measured under it. Each field is the
    tolerance for one row of that table.

    Attributes:
        monetary_relative: Relative tolerance on monetary amounts. **1% is chosen over
            exact** because a narrative that says "approximately USD 482,000" against a
            record of 482,300.00 is doing the right thing — an investigator reading a
            first draft wants a magnitude, and forcing exactness would mark good writing
            as unfaithful. Applied as ``|stated - actual| <= 0.01 * |actual|``. Counts do
            not get this latitude; see :attr:`counts_exact`.
        monetary_absolute_floor: Absolute floor under which the relative tolerance is
            replaced by this. Without it, 1% of a 0.01 BTC transfer is 0.0001, which no
            rounding of a written number can hit.
        counts_exact: Counts are exact. A narrative saying "nine accounts" when there are
            eight is wrong in a way no reader would forgive, and there is no rounding
            convention that makes it right.
        duration_granularity_units: Durations are correct within **one unit of the
            granularity the narrative itself states**. "About 3 days" against 76 hours is
            SUPPORTED, because 76 hours rounds to 3 days and the narrative claimed no
            more precision than that. "76 hours" against 80 hours is CONTRADICTED. The
            unit is read from the claim, not imposed by the checker — which is the only
            way to avoid punishing a narrative for being appropriately vague.
        share_absolute: Absolute tolerance on a share or ratio in [0, 1]. 0.01 lets
            "about 60%" match 0.61 without letting it match 0.55.
        score_absolute: Absolute tolerance on a bounded score such as bipartite.score or
            structure.density.
    """

    monetary_relative: float = 0.01
    monetary_absolute_floor: float = 0.01
    counts_exact: bool = True
    duration_granularity_units: float = 1.0
    share_absolute: float = 0.01
    score_absolute: float = 0.01

    def __post_init__(self) -> None:
        """Validate the tolerances.

        Raises:
            ValueError: If any tolerance is negative, or ``counts_exact`` is disabled.
                Counts being exact is a published commitment, not a knob: a run that
                relaxed it would report a number that does not mean what the paper says
                it means.
        """
        for name in (
            "monetary_relative",
            "monetary_absolute_floor",
            "duration_granularity_units",
            "share_absolute",
            "score_absolute",
        ):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if not self.counts_exact:
            raise ValueError(
                "counts_exact cannot be disabled: exact counts are a published tolerance "
                "commitment, not a tunable parameter"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the policy as a plain dict.

        Returns:
            Field name to value, in declaration order.
        """
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class FactConfig:
    """Thresholds and parameters for the whole fact layer.

    Attributes:
        burst_min_transactions: A burst is at least this many transactions inside a
            window of at most :attr:`burst_window_hours`. **5** because AMLworld's own
            fan patterns run to 16 counterparties and its streams to 32 transactions, so
            a threshold of 5 sits well inside a real pattern while excluding the median
            case, which holds 11 transactions across its whole 48-hour window.
        burst_window_hours: The maximum span a burst may occupy. **24** aligns the
            detector with the ``rapid_dispersal`` binding in vocab_v1.yaml, so the
            qualitative phrase and the structural detector cannot disagree about what
            "short" means.
        consolidation_min_gap_hours: The quiet interval between the last inflow and the
            first outflow that turns "received then sent" into a
            ``consolidation`` phase. **1.0** hour: AMLworld timestamps have minute
            resolution, and anything shorter is indistinguishable from same-batch
            processing rather than deliberate holding.
        hub_min_degree: Distinct counterparties on both sides before an entity is a
            ``hub`` rather than an ``intermediary``. **5**, matching the vocabulary's
            ``min_in_out_degree`` binding.
        fan_min_width: Distinct counterparties on one side before a fan is present.
            **3** — two counterparties is a payment, not a pattern. Matches FAN_FLOOR in
            data/motifs.py so the detector and the soft scorer agree on the floor even
            though they differ on everything above it.
        two_sided_min_width: Minimum width on *each* side of a gather-scatter or
            scatter-gather.
        chain_min_length: Edges in the longest simple directed path before ``chain`` is
            present. **3**: a two-edge path is a single forward, which every pass-through
            account produces.
        cycle_min_length: Shortest directed cycle that counts. **3** — a two-node
            back-and-forth is a refund, and HI-Small is full of them.
        cycle_max_length: Longest cycle searched for. Bounds an otherwise exponential
            search over a 150-node case.
        cycle_edge_budget: Above this edge count the cycle search is skipped and
            ``cycle.present`` is reported False. Recorded so the skip is never invisible.
        path_node_budget: Node count above which the longest-path search falls back to a
            DAG-exact bound rather than exhaustive DFS. Longest simple path is NP-hard;
            the fallback is documented in facts/motifs.py.
        stack_min_depth: Consecutive layers of width >= :attr:`stack_min_layer_width`
            before ``stack`` is present.
        stack_min_layer_width: Accounts in a layer before it counts as a layer.
        bipartite_min_side: Nodes on each side before a two-colouring counts as
            bipartite structure.
        threshold_reference: The reporting threshold near-threshold detection measures
            against. **10,000** in :attr:`threshold_currency`, matching the whitelisted
            ``us_ctr_10000`` regulatory reference.
        threshold_currency: The currency the threshold is denominated in. Transfers in
            any other currency are **not** counted, because converting them would require
            a rate the substrate does not carry.
        threshold_band_fraction: How far below the threshold counts as "near". **0.10**
            gives [9,000, 10,000), which is the band a structuring typology actually
            occupies.
        tolerance: The published tolerance policy.
        max_inventory_nodes: Guard on ``entity_inventory.node_ids``. Cases are capped at
            150 nodes (D-018); a case exceeding this is a bug upstream and raises rather
            than silently truncating the list the H1 checker depends on being complete.
    """

    burst_min_transactions: int = 5
    burst_window_hours: float = 24.0
    consolidation_min_gap_hours: float = 1.0

    hub_min_degree: int = 5
    fan_min_width: int = 3
    two_sided_min_width: int = 2
    chain_min_length: int = 3
    cycle_min_length: int = 3
    cycle_max_length: int = 8
    cycle_edge_budget: int = 2_000
    path_node_budget: int = 40
    stack_min_depth: int = 3
    stack_min_layer_width: int = 2
    bipartite_min_side: int = 2

    threshold_reference: float = 10_000.00
    threshold_currency: str = "US Dollar"
    threshold_band_fraction: float = 0.10

    tolerance: ToleranceConfig = dataclasses.field(default_factory=ToleranceConfig)

    max_inventory_nodes: int = 1_000

    def __post_init__(self) -> None:
        """Validate every threshold.

        Raises:
            ValueError: If a count threshold is below its logical minimum, a duration is
                not positive, the threshold band is outside (0, 1], or
                ``cycle_min_length`` exceeds ``cycle_max_length``.
        """
        positive_ints = {
            "burst_min_transactions": 1,
            "hub_min_degree": 1,
            "fan_min_width": 1,
            "two_sided_min_width": 1,
            "chain_min_length": 1,
            "cycle_min_length": 2,
            "cycle_max_length": 2,
            "cycle_edge_budget": 1,
            "path_node_budget": 1,
            "stack_min_depth": 1,
            "stack_min_layer_width": 1,
            "bipartite_min_side": 1,
            "max_inventory_nodes": 1,
        }
        for name, minimum in positive_ints.items():
            value = int(getattr(self, name))
            if value < minimum:
                raise ValueError(f"{name} must be >= {minimum}, got {value}")
        if self.cycle_min_length > self.cycle_max_length:
            raise ValueError(
                f"cycle_min_length ({self.cycle_min_length}) exceeds cycle_max_length "
                f"({self.cycle_max_length}); no cycle could ever be reported"
            )
        if self.burst_window_hours <= 0:
            raise ValueError(f"burst_window_hours must be > 0, got {self.burst_window_hours}")
        if self.consolidation_min_gap_hours < 0:
            raise ValueError(
                f"consolidation_min_gap_hours must be >= 0, got "
                f"{self.consolidation_min_gap_hours}"
            )
        if self.threshold_reference <= 0:
            raise ValueError(f"threshold_reference must be > 0, got {self.threshold_reference}")
        if not 0 < self.threshold_band_fraction <= 1:
            raise ValueError(
                f"threshold_band_fraction must be in (0, 1], got {self.threshold_band_fraction}"
            )
        if not self.threshold_currency:
            raise ValueError("threshold_currency must be a non-empty currency name")

    @property
    def threshold_floor(self) -> float:
        """Return the lower edge of the near-threshold band.

        Returns:
            ``threshold_reference * (1 - threshold_band_fraction)``.
        """
        return self.threshold_reference * (1.0 - self.threshold_band_fraction)

    def to_dict(self) -> dict[str, Any]:
        """Return the whole configuration as a JSON-serialisable mapping.

        Written verbatim into ``provenance.config`` on every fact record.

        Returns:
            Field name to value, with :attr:`tolerance` expanded to a nested dict.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_hydra(cls, cfg: Any) -> FactConfig:
        """Build a configuration from the ``facts`` Hydra config group.

        The YAML in ``configs/facts/`` is nested for readability; this dataclass is flat
        because a flat record serialises into ``provenance.config`` unambiguously. The
        projection between them lives here, and
        ``tests/integration/test_facts_config_contract.py`` asserts the defaults on both
        sides agree — two vocabularies that drift apart would let a run be configured one
        way and recorded another, which is the same failure D-014 guards against for the
        availability mask.

        Args:
            cfg: The ``cfg.facts`` node, or any mapping with the same shape.

        Returns:
            The configuration.

        Raises:
            ValueError: If any resulting threshold is out of range.
        """
        burst = cfg["burst"]
        motifs = cfg["motifs"]
        threshold = cfg["threshold"]
        tolerance = cfg["tolerance"]
        return cls(
            burst_min_transactions=int(burst["min_transactions"]),
            burst_window_hours=float(burst["window_hours"]),
            consolidation_min_gap_hours=float(cfg["consolidation_min_gap_hours"]),
            hub_min_degree=int(cfg["hub_min_degree"]),
            fan_min_width=int(motifs["fan_min_width"]),
            two_sided_min_width=int(motifs["two_sided_min_width"]),
            chain_min_length=int(motifs["chain_min_length"]),
            cycle_min_length=int(motifs["cycle_min_length"]),
            cycle_max_length=int(motifs["cycle_max_length"]),
            cycle_edge_budget=int(motifs["cycle_edge_budget"]),
            path_node_budget=int(motifs["path_node_budget"]),
            stack_min_depth=int(motifs["stack_min_depth"]),
            stack_min_layer_width=int(motifs["stack_min_layer_width"]),
            bipartite_min_side=int(motifs["bipartite_min_side"]),
            threshold_reference=float(threshold["reference"]),
            threshold_currency=str(threshold["currency"]),
            threshold_band_fraction=float(threshold["band_fraction"]),
            tolerance=ToleranceConfig(
                monetary_relative=float(tolerance["monetary_relative"]),
                monetary_absolute_floor=float(tolerance["monetary_absolute_floor"]),
                counts_exact=bool(tolerance["counts_exact"]),
                duration_granularity_units=float(tolerance["duration_granularity_units"]),
                share_absolute=float(tolerance["share_absolute"]),
                score_absolute=float(tolerance["score_absolute"]),
            ),
            max_inventory_nodes=int(cfg["max_inventory_nodes"]),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FactConfig:
        """Rebuild a configuration from :meth:`to_dict` output.

        Args:
            data: Mapping whose keys are a subset of the field names.

        Returns:
            The reconstructed configuration.

        Raises:
            ValueError: If ``data`` carries a key that is not a field, so a stale
                serialised config fails loudly rather than defaulting a threshold that
                would silently change a detector's verdict.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        if unknown := set(data) - known:
            raise ValueError(f"unknown fact config keys: {sorted(unknown)}")
        payload = dict(data)
        tolerance = payload.get("tolerance")
        if isinstance(tolerance, dict):
            payload["tolerance"] = ToleranceConfig(**tolerance)
        return cls(**payload)
