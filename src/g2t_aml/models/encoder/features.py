"""Turning a case subgraph into a tensor, without turning the label into one.

This module is the boundary between Phase 2's case store and the encoder. Three
commitments hold everywhere in it, and each has a test.

**Nothing derived from a label ever reaches a tensor.** The source columns this module is
permitted to read are enumerated in :data:`PERMITTED_EDGE_COLUMNS` and
:data:`PERMITTED_NODE_COLUMNS`, and :func:`assert_no_label_columns` checks that set
against ``leakage_audit.LABEL_PROXY_COLUMNS`` on every build. ``is_laundering``,
``typology`` and ``pattern_id`` are on the case's edge table and are never read from it.
The typology *target* is read separately, from the fact record, and lives on ``Data.y_typ``
where it is a supervision signal rather than an input.

**Every node feature is case-local.** The interim node table carries ``in_degree``,
``out_degree``, ``degree``, ``total_received`` and ``total_sent``, and all five are
*global* aggregates over the whole 515,088-account graph. CLAUDE.md note 8 records what
reading them does to the fact layer; the same trap applies here for a different reason.
A global degree is a popularity prior computed over both sides of the temporal boundary,
so it would let the encoder read a test-window account's future activity off its training-
window aggregate. Everything here is recomputed from the case's own edges.

**Amounts are standardised within a currency, never summed across them.** HI-Small has
fifteen currencies, 72,170 cross-currency transactions and no exchange rates (D-033), so
a raw sum of amounts is a number with no unit. Each amount becomes a z-score of
``log1p(amount)`` against that currency's own distribution, fitted on the **training
split only**. Standardised units are comparable and summable; the raw values are not.
This is a model feature and not a fact claim, so D-033's prohibition on *emitting*
cross-currency aggregates is respected rather than circumvented — nothing here is ever
asserted in a narrative.

Elliptic2 support follows invariant 4 without asserting anything: a fact family the
substrate lacks contributes zeros, and a companion mask channel records that the zeros
mean "unavailable" rather than "measured zero" — the same distinction D-025 draws in the
fact record, carried into the feature space.
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from g2t_aml.data.canonical import CanonicalGraph
from g2t_aml.data.leakage_audit import LABEL_PROXY_COLUMNS
from g2t_aml.models.encoder.positional import (
    laplacian_pe,
    random_walk_pe,
    undirected_adjacency,
)

if TYPE_CHECKING:  # pragma: no cover
    from torch_geometric.data import Data

#: Bumping this invalidates every cached feature tensor. It is recorded in the cache
#: manifest and in every run context, so a checkpoint can always be traced to the feature
#: space it was trained against.
FEATURE_SPEC_VERSION = "1.0.0"

#: The only edge columns this module may read. Enforced by
#: :func:`assert_no_label_columns` on every build.
PERMITTED_EDGE_COLUMNS: frozenset[str] = frozenset(
    {
        "src",
        "dst",
        "timestamp",
        "amount_paid",
        "payment_currency",
        "amount_received",
        "receiving_currency",
        "payment_format",
    }
)

#: The only node columns this module may read. Deliberately excludes every column of the
#: interim node table's global aggregates; see the module docstring.
PERMITTED_NODE_COLUMNS: frozenset[str] = frozenset({"node_id", "bank"})

#: Node feature names, in tensor column order. Names are load-bearing: the edge-feature
#: ablation and the interpretability figures index by them.
NODE_FEATURE_NAMES: tuple[str, ...] = (
    "log_in_degree",
    "log_out_degree",
    "log_total_degree",
    "log_n_txn_in",
    "log_n_txn_out",
    "log_n_self_loops",
    "degree_asymmetry",
    "reciprocity",
    "clustering",
    "is_seed",
    "amt_in_z_sum",
    "amt_in_z_max",
    "amt_in_z_mean",
    "amt_out_z_sum",
    "amt_out_z_max",
    "amt_out_z_mean",
    "amt_balance_z",
    "log_n_currencies",
    "log_n_counterparty_banks",
    "same_bank_share",
    "log_span_hours",
    "log_median_gap_hours",
    "first_touch_frac",
    "last_touch_frac",
    "has_amounts",
    "has_timestamps",
    "has_banks",
)

#: Continuous edge feature names, in tensor column order. The three categorical fields
#: (payment currency, receiving currency, payment format) are carried separately as
#: integer indices and embedded by the model, per the Phase 7 brief.
EDGE_FEATURE_NAMES: tuple[str, ...] = (
    "amt_paid_z",
    "amt_received_z",
    "log_amount_paid",
    "cross_currency",
    "fx_log_ratio",
    "log_dt_prev_src_hours",
    "log_dt_prev_dst_hours",
    "window_frac",
    "hour_sin",
    "hour_cos",
    "is_self_loop",
    "is_reciprocal",
    "has_amounts",
    "has_timestamps",
)

#: Index reserved for a category not seen in the training split. Vocabularies are fitted
#: on train only, so an unseen test-window currency maps here rather than silently
#: colliding with a fitted one.
OOV_INDEX = 0

#: Clip for the log FX ratio. A cross-currency pair can differ by four orders of
#: magnitude (JPY against BTC), and an unclipped log ratio dominates the layer norm.
_FX_CLIP = 10.0

#: Amount z-scores are clipped to this many standard deviations. HI-Small's amount
#: distribution has a long right tail even in log space, and a single 12-sigma transfer
#: otherwise saturates the batch statistics for every case it appears in.
_Z_CLIP = 8.0


class FeatureError(ValueError):
    """Raised when a case cannot be turned into a valid feature tensor."""


def assert_no_label_columns(columns: frozenset[str]) -> None:
    """Check that a permitted-column set contains no label or label proxy.

    Run on every feature build rather than once at import, because the cheap check is
    what makes the expensive guarantee credible. ``LABEL_PROXY_COLUMNS`` is Phase 2's
    enforced list and the leakage auditor treats a violation as fatal.

    Args:
        columns: The column names the caller intends to read.

    Raises:
        FeatureError: If any name is a label or a proxy for one.
    """
    if offending := sorted(columns & LABEL_PROXY_COLUMNS):
        raise FeatureError(
            f"columns {offending} are labels or label proxies and must never reach a "
            "feature tensor (Phase 2 leakage audit, fatal check 3)"
        )


@dataclass(frozen=True)
class FeatureSpace:
    """A fitted feature space: vocabularies, amount statistics and encoding widths.

    Fitted on the training split alone and then frozen, so that nothing about the
    validation or test windows can influence how a case is encoded. Serialised beside
    every checkpoint: a checkpoint without its feature space cannot be applied to a case.

    Attributes:
        version: :data:`FEATURE_SPEC_VERSION` at fit time.
        dataset: Substrate key the space was fitted on.
        currencies: Currency code to index. Index 0 is reserved for OOV.
        payment_formats: Payment-format string to index. Index 0 is reserved for OOV.
        amount_stats: Currency code to ``(mean, std)`` of ``log1p(amount)`` over the
            training split. Currencies absent here fall back to ``global_amount_stats``.
        global_amount_stats: ``(mean, std)`` over all training amounts, for an OOV
            currency.
        lap_pe_dim: Laplacian eigenvector components per node.
        rw_pe_dim: Random-walk steps per node.
        n_train_cases: How many cases the fit saw, recorded for provenance.
        availability: The substrate availability mask, as a plain dict.
    """

    version: str
    dataset: str
    currencies: dict[str, int]
    payment_formats: dict[str, int]
    amount_stats: dict[str, tuple[float, float]]
    global_amount_stats: tuple[float, float]
    lap_pe_dim: int
    rw_pe_dim: int
    n_train_cases: int
    availability: dict[str, bool] = field(default_factory=dict)

    @property
    def n_currencies(self) -> int:
        """Return the currency embedding table size, OOV slot included.

        Returns:
            One more than the number of fitted currencies.
        """
        return len(self.currencies) + 1

    @property
    def n_payment_formats(self) -> int:
        """Return the payment-format embedding table size, OOV slot included.

        Returns:
            One more than the number of fitted formats.
        """
        return len(self.payment_formats) + 1

    @property
    def node_dim(self) -> int:
        """Return the total node input width, positional encodings included.

        Returns:
            Number of columns in ``Data.x``.
        """
        return len(NODE_FEATURE_NAMES) + self.lap_pe_dim + self.rw_pe_dim

    @property
    def edge_continuous_dim(self) -> int:
        """Return the continuous edge feature width.

        Returns:
            Number of columns in ``Data.edge_attr``.
        """
        return len(EDGE_FEATURE_NAMES)

    @property
    def pe_slice(self) -> tuple[int, int]:
        """Return the half-open column range of ``Data.x`` holding positional encodings.

        Returns:
            ``(start, stop)`` such that ``x[:, start:stop]`` is the concatenated
            Laplacian and random-walk encoding. Used by the PE ablation, which zeroes
            exactly this block rather than rebuilding the cache.
        """
        start = len(NODE_FEATURE_NAMES)
        return start, start + self.lap_pe_dim + self.rw_pe_dim

    @property
    def lap_pe_slice(self) -> tuple[int, int]:
        """Return the column range holding the Laplacian encoding alone.

        Returns:
            ``(start, stop)`` into ``Data.x``.
        """
        start = len(NODE_FEATURE_NAMES)
        return start, start + self.lap_pe_dim

    def currency_index(self, code: str | None) -> int:
        """Map a currency code to its embedding index.

        Args:
            code: The currency code, or None where the substrate has no currencies.

        Returns:
            The fitted index, or :data:`OOV_INDEX` for an unseen or absent code.
        """
        return self.currencies.get(code or "", OOV_INDEX)

    def format_index(self, name: str | None) -> int:
        """Map a payment format to its embedding index.

        Args:
            name: The payment format, or None where the substrate has none.

        Returns:
            The fitted index, or :data:`OOV_INDEX` for an unseen or absent format.
        """
        return self.payment_formats.get(name or "", OOV_INDEX)

    def standardise(self, amount: float, currency: str | None) -> float:
        """Standardise one amount against its own currency's training distribution.

        Args:
            amount: The raw amount. Negative values are treated as zero, since
                ``log1p`` is undefined below -1 and no substrate emits a negative
                transfer.
            currency: The currency the amount is denominated in.

        Returns:
            A z-score of ``log1p(amount)``, clipped to +/- :data:`_Z_CLIP`.
        """
        mean, std = self.amount_stats.get(currency or "", self.global_amount_stats)
        value = math.log1p(max(amount, 0.0))
        z = (value - mean) / std if std > 0 else 0.0
        return float(min(max(z, -_Z_CLIP), _Z_CLIP))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable form.

        Returns:
            Every field, with tuple-valued statistics as two-element lists.
        """
        payload = dataclasses.asdict(self)
        payload["amount_stats"] = {k: list(v) for k, v in self.amount_stats.items()}
        payload["global_amount_stats"] = list(self.global_amount_stats)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureSpace:
        """Rebuild a space written by :meth:`to_dict`.

        Args:
            payload: The serialised space.

        Returns:
            The reconstructed space.

        Raises:
            FeatureError: If the payload was written by a different feature-spec
                version, which means every cached tensor and every checkpoint built
                against it is stale.
        """
        if payload.get("version") != FEATURE_SPEC_VERSION:
            raise FeatureError(
                f"feature space version mismatch: file has {payload.get('version')!r}, "
                f"code expects {FEATURE_SPEC_VERSION!r}; rebuild the feature cache"
            )
        return cls(
            version=str(payload["version"]),
            dataset=str(payload["dataset"]),
            currencies={str(k): int(v) for k, v in payload["currencies"].items()},
            payment_formats={str(k): int(v) for k, v in payload["payment_formats"].items()},
            amount_stats={
                str(k): (float(v[0]), float(v[1])) for k, v in payload["amount_stats"].items()
            },
            global_amount_stats=(
                float(payload["global_amount_stats"][0]),
                float(payload["global_amount_stats"][1]),
            ),
            lap_pe_dim=int(payload["lap_pe_dim"]),
            rw_pe_dim=int(payload["rw_pe_dim"]),
            n_train_cases=int(payload["n_train_cases"]),
            availability=dict(payload.get("availability") or {}),
        )


def fit_feature_space(
    edges: pl.DataFrame,
    *,
    dataset: str,
    availability: dict[str, bool],
    n_train_cases: int,
    lap_pe_dim: int = 8,
    rw_pe_dim: int = 16,
) -> FeatureSpace:
    """Fit vocabularies and per-currency amount statistics on training-split edges.

    Args:
        edges: The transactions belonging to training cases, and only those. Passing
            the whole graph would fit the encoding on the test window and is the reason
            this takes a frame rather than a collection.
        dataset: Substrate key.
        availability: The substrate's availability mask, as a dict.
        n_train_cases: How many cases contributed, recorded for provenance.
        lap_pe_dim: Laplacian eigenvector components per node.
        rw_pe_dim: Random-walk steps per node.

    Returns:
        The fitted, frozen feature space.

    Raises:
        FeatureError: If ``edges`` carries a label or label-proxy column that this
            module would be reading, which would mean the caller passed an unfiltered
            frame to a function documented not to look at one.
    """
    assert_no_label_columns(PERMITTED_EDGE_COLUMNS)

    currencies: set[str] = set()
    for column in ("payment_currency", "receiving_currency"):
        if column in edges.columns:
            currencies |= {c for c in edges[column].drop_nulls().unique().to_list() if c}
    formats: set[str] = set()
    if "payment_format" in edges.columns:
        formats = {f for f in edges["payment_format"].drop_nulls().unique().to_list() if f}

    # Both sides of a cross-currency transfer contribute to their own currency's
    # distribution: the paid leg is a JPY observation and the received leg a USD one, and
    # pooling them per currency is the whole point of standardising within one.
    by_currency: dict[str, list[float]] = defaultdict(list)
    all_logs: list[np.ndarray] = []
    for amount_col, currency_col in (
        ("amount_paid", "payment_currency"),
        ("amount_received", "receiving_currency"),
    ):
        if amount_col not in edges.columns or currency_col not in edges.columns:
            continue
        frame = edges.select(
            pl.col(amount_col).cast(pl.Float64).alias("amount"),
            pl.col(currency_col).alias("currency"),
        ).drop_nulls()
        if frame.height == 0:
            continue
        logs = np.log1p(np.clip(frame["amount"].to_numpy(), 0.0, None))
        all_logs.append(logs)
        for code, value in zip(frame["currency"].to_list(), logs, strict=True):
            by_currency[str(code)].append(float(value))

    stats: dict[str, tuple[float, float]] = {}
    for code, values in by_currency.items():
        array = np.asarray(values)
        stats[code] = (float(array.mean()), float(array.std() or 1.0))

    if all_logs:
        pooled = np.concatenate(all_logs)
        global_stats = (float(pooled.mean()), float(pooled.std() or 1.0))
    else:
        # Elliptic2 carries no amounts at all. The statistics are never consulted,
        # because every amount feature is masked out, but a well-formed space still
        # needs a defined fallback.
        global_stats = (0.0, 1.0)

    return FeatureSpace(
        version=FEATURE_SPEC_VERSION,
        dataset=dataset,
        currencies={code: i for i, code in enumerate(sorted(currencies), start=1)},
        payment_formats={name: i for i, name in enumerate(sorted(formats), start=1)},
        amount_stats=stats,
        global_amount_stats=global_stats,
        lap_pe_dim=lap_pe_dim,
        rw_pe_dim=rw_pe_dim,
        n_train_cases=n_train_cases,
        availability=dict(availability),
    )


def _column(frame: pl.DataFrame, name: str) -> list[Any] | None:
    """Return a column as a python list, or None when the substrate lacks it."""
    if name not in frame.columns:
        return None
    return frame[name].to_list()


def _safe_log1p(value: float) -> float:
    """Return ``log1p`` of a non-negative clamp of ``value``."""
    return float(math.log1p(max(value, 0.0)))


def build_case_data(  # noqa: PLR0912, PLR0915 -- one linear pass, one tensor per family;
    # splitting it would move locals into a parameter object and hide the ordering that
    # NODE_FEATURE_NAMES and EDGE_FEATURE_NAMES pin.
    graph: CanonicalGraph,
    space: FeatureSpace,
    *,
    seed_node: str | None = None,
    label: int | None = None,
    typology_index: int | None = None,
) -> Data:
    """Encode one case subgraph as a PyG ``Data`` object.

    Args:
        graph: The materialised case.
        space: The fitted feature space.
        seed_node: The account the case was built around, used for the ``is_seed``
            channel. None leaves the channel at zero.
        label: Binary target, 1 for suspicious. None on an unlabelled case.
        typology_index: Auxiliary target index into the typology vocabulary, or None.

    Returns:
        A ``Data`` carrying ``x``, ``edge_index``, ``edge_attr``, the three categorical
        edge index tensors ``edge_currency_paid`` / ``edge_currency_received`` /
        ``edge_format``, the targets ``y`` and ``y_typ``, and ``case_id`` /
        ``node_ids`` for traceability back to accounts.

    Raises:
        FeatureError: If the graph has no nodes, or an edge endpoint is absent from the
            node table.
        ImportError: If the ``graph`` extra is not installed.
    """
    try:
        import torch
        from torch_geometric.data import Data as PygData
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "torch and torch-geometric are required for the encoder feature layer; "
            "they live in the optional `graph` extra. Install with: make install-gpu"
        ) from exc

    assert_no_label_columns(PERMITTED_EDGE_COLUMNS | PERMITTED_NODE_COLUMNS)

    node_ids = graph.nodes["node_id"].to_list()
    n = len(node_ids)
    if n == 0:
        raise FeatureError(f"case {graph.graph_id!r} has no nodes")
    position = {nid: i for i, nid in enumerate(node_ids)}

    has_amounts = bool(space.availability.get("monetary_amounts", False))
    has_times = bool(space.availability.get("absolute_timestamps", False))
    has_banks = bool(space.availability.get("institution_identity", False))

    edge_frame = graph.edges
    m = edge_frame.height
    src_ids = _column(edge_frame, "src") or []
    dst_ids = _column(edge_frame, "dst") or []
    try:
        src = np.asarray([position[s] for s in src_ids], dtype=np.int64)
        dst = np.asarray([position[d] for d in dst_ids], dtype=np.int64)
    except KeyError as exc:
        raise FeatureError(
            f"case {graph.graph_id!r} has an edge endpoint {exc.args[0]!r} absent from "
            "its node table"
        ) from exc

    paid = np.asarray(
        [v if v is not None else 0.0 for v in (_column(edge_frame, "amount_paid") or [0.0] * m)],
        dtype=np.float64,
    )
    received = np.asarray(
        [
            v if v is not None else 0.0
            for v in (_column(edge_frame, "amount_received") or [0.0] * m)
        ],
        dtype=np.float64,
    )
    pay_ccy = _column(edge_frame, "payment_currency") or [None] * m
    recv_ccy = _column(edge_frame, "receiving_currency") or [None] * m
    formats = _column(edge_frame, "payment_format") or [None] * m
    stamps = _column(edge_frame, "timestamp") if has_times else None
    banks = _column(graph.nodes, "bank") if has_banks else None

    # -------------------------------------------------------------- temporal ---
    if stamps is not None and m and all(s is not None for s in stamps):
        epoch = min(s for s in stamps)
        offsets = np.asarray(
            [(s - epoch).total_seconds() / 3600.0 for s in stamps], dtype=np.float64
        )
        hour_of_day = np.asarray(
            [(s.hour + s.minute / 60.0) for s in stamps],
            dtype=np.float64,
        )
    else:
        offsets = np.zeros(m, dtype=np.float64)
        hour_of_day = np.zeros(m, dtype=np.float64)
        has_times = False

    span = float(offsets.max() - offsets.min()) if m else 0.0
    window_frac = (offsets - offsets.min()) / span if span > 0 else np.zeros(m, dtype=np.float64)

    # Time since the previous transaction touching each endpoint. The Phase 7 brief calls
    # this out specifically: a burst of transfers through one account is the temporal
    # signature of layering, and it is invisible in an amount-only edge encoding.
    order = np.argsort(offsets, kind="stable") if m else np.asarray([], dtype=np.int64)
    dt_src = np.zeros(m, dtype=np.float64)
    dt_dst = np.zeros(m, dtype=np.float64)
    last_touch: dict[int, float] = {}
    for e in order:
        s_node, d_node = int(src[e]), int(dst[e])
        now = float(offsets[e])
        dt_src[e] = now - last_touch.get(s_node, now)
        dt_dst[e] = now - last_touch.get(d_node, now)
        last_touch[s_node] = now
        last_touch[d_node] = now

    # ------------------------------------------------------------ node degree ---
    successors: dict[int, set[int]] = defaultdict(set)
    predecessors: dict[int, set[int]] = defaultdict(set)
    n_txn_in = np.zeros(n)
    n_txn_out = np.zeros(n)
    n_self = np.zeros(n)
    for e in range(m):
        s_node, d_node = int(src[e]), int(dst[e])
        if s_node == d_node:
            n_self[s_node] += 1
            continue
        successors[s_node].add(d_node)
        predecessors[d_node].add(s_node)
        n_txn_out[s_node] += 1
        n_txn_in[d_node] += 1

    adjacency = undirected_adjacency(n, src, dst)
    degree = adjacency.sum(axis=1)
    triangles = np.diag(adjacency @ adjacency @ adjacency)
    with np.errstate(divide="ignore", invalid="ignore"):
        clustering = np.where(degree > 1, triangles / (degree * (degree - 1)), 0.0)

    # --------------------------------------------------------- node amounts ----
    amt_in: dict[int, list[float]] = defaultdict(list)
    amt_out: dict[int, list[float]] = defaultdict(list)
    node_currencies: dict[int, set[str]] = defaultdict(set)
    node_times: dict[int, list[float]] = defaultdict(list)
    counterparty_banks: dict[int, set[str]] = defaultdict(set)
    same_bank_hits = np.zeros(n)
    incident = np.zeros(n)
    for e in range(m):
        s_node, d_node = int(src[e]), int(dst[e])
        if has_amounts:
            amt_out[s_node].append(space.standardise(float(paid[e]), pay_ccy[e]))
            amt_in[d_node].append(space.standardise(float(received[e]), recv_ccy[e]))
        for node, code in ((s_node, pay_ccy[e]), (d_node, recv_ccy[e])):
            if code:
                node_currencies[node].add(str(code))
        node_times[s_node].append(float(offsets[e]))
        node_times[d_node].append(float(offsets[e]))
        if banks is not None and s_node != d_node:
            s_bank, d_bank = banks[s_node], banks[d_node]
            counterparty_banks[s_node].add(str(d_bank))
            counterparty_banks[d_node].add(str(s_bank))
            incident[s_node] += 1
            incident[d_node] += 1
            if s_bank == d_bank:
                same_bank_hits[s_node] += 1
                same_bank_hits[d_node] += 1

    features = np.zeros((n, len(NODE_FEATURE_NAMES)), dtype=np.float32)
    seed_index = position.get(seed_node) if seed_node is not None else None
    for i in range(n):
        in_deg = len(predecessors.get(i, ()))
        out_deg = len(successors.get(i, ()))
        total_deg = len(successors.get(i, set()) | predecessors.get(i, set()))
        both = successors.get(i, set()) & predecessors.get(i, set())
        either = successors.get(i, set()) | predecessors.get(i, set())
        ins, outs = amt_in.get(i, []), amt_out.get(i, [])
        times = sorted(node_times.get(i, []))
        gaps = np.diff(times) if len(times) > 1 else np.asarray([0.0])

        features[i] = (
            _safe_log1p(in_deg),
            _safe_log1p(out_deg),
            _safe_log1p(total_deg),
            _safe_log1p(n_txn_in[i]),
            _safe_log1p(n_txn_out[i]),
            _safe_log1p(n_self[i]),
            (out_deg - in_deg) / (out_deg + in_deg + 1.0),
            len(both) / len(either) if either else 0.0,
            float(clustering[i]),
            1.0 if seed_index == i else 0.0,
            float(sum(ins)),
            float(max(ins)) if ins else 0.0,
            float(np.mean(ins)) if ins else 0.0,
            float(sum(outs)),
            float(max(outs)) if outs else 0.0,
            float(np.mean(outs)) if outs else 0.0,
            float(sum(ins) - sum(outs)),
            _safe_log1p(len(node_currencies.get(i, ()))),
            _safe_log1p(len(counterparty_banks.get(i, ()))),
            float(same_bank_hits[i] / incident[i]) if incident[i] else 0.0,
            _safe_log1p(times[-1] - times[0]) if len(times) > 1 else 0.0,
            _safe_log1p(float(np.median(gaps))),
            float((times[0] - offsets.min()) / span) if times and span > 0 else 0.0,
            float((times[-1] - offsets.min()) / span) if times and span > 0 else 0.0,
            1.0 if has_amounts else 0.0,
            1.0 if has_times else 0.0,
            1.0 if has_banks else 0.0,
        )

    lap = laplacian_pe(adjacency, space.lap_pe_dim)
    walk = random_walk_pe(adjacency, space.rw_pe_dim)
    x = np.concatenate([features, lap, walk], axis=1).astype(np.float32)

    # ----------------------------------------------------------- edge tensor ---
    reciprocal = {(int(a), int(b)) for a, b in zip(src, dst, strict=True)}
    edge_attr = np.zeros((m, len(EDGE_FEATURE_NAMES)), dtype=np.float32)
    for e in range(m):
        s_node, d_node = int(src[e]), int(dst[e])
        pay, recv = float(paid[e]), float(received[e])
        edge_attr[e] = (
            space.standardise(pay, pay_ccy[e]) if has_amounts else 0.0,
            space.standardise(recv, recv_ccy[e]) if has_amounts else 0.0,
            _safe_log1p(pay) if has_amounts else 0.0,
            1.0 if (pay_ccy[e] and recv_ccy[e] and pay_ccy[e] != recv_ccy[e]) else 0.0,
            (
                float(np.clip(math.log((recv + 1.0) / (pay + 1.0)), -_FX_CLIP, _FX_CLIP))
                if has_amounts
                else 0.0
            ),
            _safe_log1p(float(dt_src[e])) if has_times else 0.0,
            _safe_log1p(float(dt_dst[e])) if has_times else 0.0,
            float(window_frac[e]) if has_times else 0.0,
            math.sin(2 * math.pi * hour_of_day[e] / 24.0) if has_times else 0.0,
            math.cos(2 * math.pi * hour_of_day[e] / 24.0) if has_times else 0.0,
            1.0 if s_node == d_node else 0.0,
            1.0 if (d_node, s_node) in reciprocal and s_node != d_node else 0.0,
            1.0 if has_amounts else 0.0,
            1.0 if has_times else 0.0,
        )

    data = PygData(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(np.stack([src, dst])).long(),
        edge_attr=torch.from_numpy(edge_attr),
    )
    data.edge_currency_paid = torch.tensor(
        [space.currency_index(c) for c in pay_ccy], dtype=torch.long
    )
    data.edge_currency_received = torch.tensor(
        [space.currency_index(c) for c in recv_ccy], dtype=torch.long
    )
    data.edge_format = torch.tensor([space.format_index(f) for f in formats], dtype=torch.long)
    data.y = torch.tensor([label if label is not None else -1], dtype=torch.long)
    data.y_typ = torch.tensor(
        [typology_index if typology_index is not None else -1], dtype=torch.long
    )
    data.case_id = graph.graph_id
    data.node_ids = list(node_ids)
    data.num_nodes = n
    return data
