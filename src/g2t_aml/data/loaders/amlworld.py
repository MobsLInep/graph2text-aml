"""IBM AMLworld loader: transactions, laundering patterns, and the account graph.

Three things in this file are easy to get wrong, and each is guarded explicitly.

**The duplicated column name.** The CSV header is::

    Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,
    Amount Paid,Payment Currency,Payment Format,Is Laundering

``Account`` appears twice — once as the originating account, once as the beneficiary.
Polars silently renames the second to ``Account_duplicated_0``. That rename is a load-
bearing assumption, so :func:`load_transactions` asserts the raw header explicitly and
then renames positionally rather than by name.

**The node key.** Accounts are only unique *within* a bank. Keying nodes on the account
identifier alone yields 515,080 nodes; keying on ``bank|account`` yields 515,088, which is
the published figure. Eight account identifiers genuinely collide across banks, so the
composite key is the correct one and the naive one is off by exactly those eight.

**The patterns file.** It is the only source of typology ground truth. Streams are
delimited by ``BEGIN LAUNDERING ATTEMPT - <TYPOLOGY>`` / ``END LAUNDERING ATTEMPT -
<TYPOLOGY>`` lines with bare CSV rows between them, and it is parsed by a real state
machine that validates the delimiters rather than by a hopeful regex.

**The join key is textual, not numeric.** AMLworld rows carry no primary key, so a
transaction is identified by its natural key, and joining the patterns file to the CSV
means reconstructing that key on both sides. Amounts are *not* uniformly two-decimal:
fiat rows carry two decimals but the ~148,000 Bitcoin rows carry six, including some with
significant trailing zeros. Any reconstruction that routes the amount through a float and
re-formats it therefore cannot be made faithful — ``0.370000`` and ``0.37`` are the same
number and different text. The key is consequently built from the *source text* of each
field, before any cast. Getting this wrong costs exactly one transaction, which is how it
was found: ``unclassified`` came out at 1,969 against a published 1,968.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from g2t_aml.data.canonical import AMLWORLD_AVAILABILITY, CanonicalGraph

#: Variants shipped by the AMLworld release. HI/LI are higher/lower illicit ratio.
SIZES: tuple[str, ...] = (
    "HI-Small",
    "HI-Medium",
    "HI-Large",
    "LI-Small",
    "LI-Medium",
    "LI-Large",
)

#: The exact header line of every AMLworld transactions CSV, verified against
#: HI-Small_Trans.csv rather than recalled. Any deviation aborts the load.
EXPECTED_HEADER: tuple[str, ...] = (
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account",  # duplicated on purpose: this is the beneficiary account
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
)

#: Our column names, positionally aligned with :data:`EXPECTED_HEADER`. Renaming by
#: position sidesteps the duplicated ``Account`` entirely.
CANONICAL_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "src_bank",
    "src_account",
    "dst_bank",
    "dst_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
)

#: Timestamps are local wall-clock with minute resolution and no timezone.
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"

#: Typology names as they appear in the patterns file, mapped onto the controlled
#: vocabulary in ``data.canonical``. ``CYCLE`` becomes ``cycle``; the file never emits an
#: ``unclassified`` stream, since unclassified transactions are by definition the ones
#: that appear in no stream at all.
TYPOLOGY_MAP: dict[str, str] = {
    "FAN-OUT": "fan_out",
    "FAN-IN": "fan_in",
    "GATHER-SCATTER": "gather_scatter",
    "SCATTER-GATHER": "scatter_gather",
    "CYCLE": "cycle",
    "RANDOM": "random",
    "BIPARTITE": "bipartite",
    "STACK": "stack",
}

_BEGIN_RE = re.compile(
    r"^BEGIN LAUNDERING ATTEMPT\s*-\s*(?P<typology>[A-Z-]+?)\s*(?::(?P<detail>.*))?$"
)
_END_RE = re.compile(r"^END LAUNDERING ATTEMPT\s*-\s*(?P<typology>[A-Z-]+?)\s*$")

#: Published statistics for HI-Small, from Altman et al. (2023). The loader asserts
#: against these; see ``docs/data_cards/amlworld_hi_small.md`` for observed values.
PUBLISHED_STATISTICS: dict[str, dict[str, int]] = {
    "HI-Small": {"num_nodes": 515_088, "num_edges": 5_078_345},
}

#: Published per-typology transaction counts for HI-Small. These are counts of
#: *transactions* inside pattern streams, not counts of streams — a distinction that
#: matters, because HI-Small has 370 streams and 3,209 patterned transactions.
PUBLISHED_TYPOLOGY_COUNTS: dict[str, dict[str, int]] = {
    "HI-Small": {
        "fan_out": 342,
        "fan_in": 318,
        "gather_scatter": 716,
        "scatter_gather": 626,
        "cycle": 287,
        "random": 191,
        "bipartite": 263,
        "stack": 466,
        "unclassified": 1_968,
    },
}


class PatternsParseError(ValueError):
    """Raised when the patterns file violates its delimiter grammar."""


class HeaderMismatchError(ValueError):
    """Raised when a transactions CSV header is not the expected one."""


#: Fields making up a transaction's natural key, in order. All are taken as source text;
#: see the module docstring for why the amount must not be routed through a float.
KEY_FIELDS: tuple[str, ...] = (
    "timestamp",
    "src_bank",
    "src_account",
    "dst_bank",
    "dst_account",
    "amount_paid",
)

#: Separator for the joined key. Absent from every field's character set, so the join is
#: unambiguous.
KEY_SEPARATOR = "|"


@dataclass(frozen=True)
class TransactionKey:
    """The tuple identifying a transaction across the CSV and the patterns file.

    AMLworld rows carry no primary key, so a transaction is identified by its full natural
    key. Every field is the **source text**, never a re-formatted parse: the CSV and the
    patterns file are byte-identical for the same row, so text comparison is exact, while
    any float round-trip is not.
    """

    timestamp: str
    src_bank: str
    src_account: str
    dst_bank: str
    dst_account: str
    amount_paid: str

    def as_string(self) -> str:
        """Return the key as a single delimiter-joined string.

        Returns:
            Pipe-joined field values, suitable as a DataFrame join key.
        """
        return KEY_SEPARATOR.join(
            (
                self.timestamp,
                self.src_bank,
                self.src_account,
                self.dst_bank,
                self.dst_account,
                self.amount_paid,
            )
        )


def transactions_path(raw_dir: str | Path, size: str) -> Path:
    """Return the path to a variant's transactions CSV.

    Args:
        raw_dir: The ``paths.raw_dir`` root.
        size: One of :data:`SIZES`.

    Returns:
        ``raw_dir/amlworld/<size>_Trans.csv``.

    Raises:
        ValueError: If ``size`` is not a known variant.
    """
    _check_size(size)
    return Path(raw_dir) / "amlworld" / f"{size}_Trans.csv"


def patterns_path(raw_dir: str | Path, size: str) -> Path:
    """Return the path to a variant's patterns file.

    Args:
        raw_dir: The ``paths.raw_dir`` root.
        size: One of :data:`SIZES`.

    Returns:
        ``raw_dir/amlworld/<size>_Patterns.txt``.

    Raises:
        ValueError: If ``size`` is not a known variant.
    """
    _check_size(size)
    return Path(raw_dir) / "amlworld" / f"{size}_Patterns.txt"


def _check_size(size: str) -> None:
    """Validate a variant name.

    Args:
        size: Candidate variant.

    Raises:
        ValueError: If ``size`` is not in :data:`SIZES`.
    """
    if size not in SIZES:
        raise ValueError(f"unknown AMLworld size {size!r}; expected one of {SIZES}")


def read_header(path: str | Path) -> tuple[str, ...]:
    """Read the raw header line of a transactions CSV.

    Read as text rather than through a DataFrame library, because every such library
    de-duplicates the repeated ``Account`` column and so cannot show the real header.

    Args:
        path: The CSV.

    Returns:
        Header fields in file order, including the duplicate.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is empty.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        line = fh.readline()
    if not line.strip():
        raise ValueError(f"{path} is empty; expected a header line")
    return tuple(field.strip() for field in line.rstrip("\r\n").split(","))


def assert_header(path: str | Path) -> None:
    """Raise unless a transactions CSV has exactly the expected header.

    Args:
        path: The CSV.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        HeaderMismatchError: If the header differs in any field or in field order. A
            changed header means the positional rename below would silently mislabel
            columns, so this is fatal rather than a warning.
    """
    actual = read_header(path)
    if actual != EXPECTED_HEADER:
        raise HeaderMismatchError(
            f"unexpected AMLworld header in {path}\n"
            f"  expected: {EXPECTED_HEADER}\n"
            f"  actual:   {actual}\n"
            "Column positions are load-bearing (the two 'Account' columns are "
            "distinguished by position, not name). Refusing to guess."
        )


def load_transactions(
    size: str = "HI-Small",
    *,
    raw_dir: str | Path,
    n_rows: int | None = None,
) -> pl.DataFrame:
    """Load a variant's transactions into a typed frame.

    Args:
        size: One of :data:`SIZES`.
        raw_dir: The ``paths.raw_dir`` root.
        n_rows: Read only the first ``n_rows`` data rows. For fixtures and smoke runs
            only — any run that uses it must record the fact in its manifest, because a
            silently subsetted ingest produces statistics that disagree with the paper.

    Returns:
        A frame with :data:`CANONICAL_COLUMNS` plus ``transaction_key``: parsed
        ``timestamp`` (Datetime), ``amount_received``/``amount_paid`` as Float64,
        ``is_laundering`` as Boolean, and bank/account/currency/format fields as Utf8.
        Bank and account identifiers stay strings deliberately — they are zero-padded
        codes, and reading ``010`` as the integer 10 would collide it with bank ``10``.
        ``transaction_key`` is built from the source text before any cast, so it joins
        exactly against the patterns file.

    Raises:
        FileNotFoundError: If the CSV is absent.
        HeaderMismatchError: If the header is not the expected one.
        ValueError: If ``size`` is unknown.
    """
    path = transactions_path(raw_dir, size)
    assert_header(path)

    # Read every field as Utf8 first. Letting the CSV reader infer types would strip the
    # leading zeros from bank codes and could infer the wrong numeric width for amounts.
    frame = pl.read_csv(
        path,
        has_header=True,
        infer_schema_length=0,
        n_rows=n_rows,
        low_memory=False,
    )
    if frame.width != len(CANONICAL_COLUMNS):
        raise HeaderMismatchError(
            f"{path} has {frame.width} columns, expected {len(CANONICAL_COLUMNS)}"
        )
    # Positional rename: the two 'Account' columns differ only by position, and the
    # reader has already renamed the second to something like 'Account_duplicated_0'.
    frame = frame.rename(dict(zip(frame.columns, CANONICAL_COLUMNS, strict=True)))

    # The key must be taken from the source text, while the columns are still Utf8.
    # After the cast below, amount_paid is a float and the original text is unrecoverable.
    frame = frame.with_columns(transaction_key_expr().alias("transaction_key"))

    return frame.with_columns(
        pl.col("timestamp").str.to_datetime(TIMESTAMP_FORMAT, strict=True),
        pl.col("amount_received").cast(pl.Float64),
        pl.col("amount_paid").cast(pl.Float64),
        pl.col("is_laundering").cast(pl.Int8).cast(pl.Boolean),
    )


def transaction_key_expr() -> pl.Expr:
    """Return the expression building ``transaction_key`` from untyped source columns.

    Returns:
        An expression joining :data:`KEY_FIELDS` with :data:`KEY_SEPARATOR`. It must be
        applied while those columns are still Utf8 — see the module docstring.
    """
    return pl.concat_str([pl.col(f) for f in KEY_FIELDS], separator=KEY_SEPARATOR)


def _account_key(bank: pl.Expr, account: pl.Expr) -> pl.Expr:
    """Build the composite node key for an account.

    Args:
        bank: Expression yielding the bank code.
        account: Expression yielding the account identifier.

    Returns:
        An expression yielding ``"<bank>|<account>"``. Account identifiers are unique only
        within a bank; see the module docstring.
    """
    return pl.concat_str([bank, pl.lit("|"), account])


def load_patterns(size: str = "HI-Small", *, raw_dir: str | Path) -> pl.DataFrame:
    """Parse a variant's patterns file into a tidy table.

    Args:
        size: One of :data:`SIZES`.
        raw_dir: The ``paths.raw_dir`` root.

    Returns:
        One row per transaction *occurrence* within a stream, with columns:

        - ``pattern_id`` (Utf8) — stream identifier, ``"<typology>_<ordinal>"``
        - ``typology`` (Utf8) — controlled vocabulary member
        - ``typology_detail`` (Utf8) — the text after the colon on the BEGIN line, e.g.
          ``"Max 16-degree Fan-Out"``; empty when absent
        - ``transaction_key`` (Utf8) — joins to ``build_transaction_keys``
        - ``position_in_stream`` (UInt32) — 0-based, in file order
        - the eleven transaction fields, so the table stands alone

        A transaction may legitimately appear in more than one stream, so
        ``transaction_key`` is not unique.

    Raises:
        FileNotFoundError: If the patterns file is absent.
        ValueError: If ``size`` is unknown.
        PatternsParseError: On any grammar violation — a nested BEGIN, an END without a
            BEGIN, a mismatched END typology, an unterminated stream at EOF, a data row
            outside a stream, an unknown typology, or a row with the wrong field count.
    """
    path = patterns_path(raw_dir, size)
    return parse_patterns_text(Path(path).read_text(encoding="utf-8"), source=str(path))


def _parse_transaction_row(
    line: str,
    *,
    lineno: int,
    source: str,
    pattern_id: str,
    typology: str,
    detail: str,
    position: int,
) -> dict[str, object]:
    """Turn one CSV line from inside a stream into a tidy record.

    Args:
        line: The stripped source line.
        lineno: 1-based line number, for error messages.
        source: File name, for error messages.
        pattern_id: Identifier of the enclosing stream.
        typology: Controlled-vocabulary typology of the enclosing stream.
        detail: The enclosing stream's BEGIN-line detail text.
        position: 0-based position within the stream.

    Returns:
        A record carrying the eleven transaction fields as source text plus the stream
        metadata and the joinable ``transaction_key``.

    Raises:
        PatternsParseError: If the line does not have exactly the expected field count.
    """
    fields = [f.strip() for f in line.split(",")]
    if len(fields) != len(CANONICAL_COLUMNS):
        raise PatternsParseError(
            f"{source}:{lineno}: expected {len(CANONICAL_COLUMNS)} comma-separated "
            f"fields, got {len(fields)}: {line!r}"
        )
    record: dict[str, object] = dict(zip(CANONICAL_COLUMNS, fields, strict=True))
    record["pattern_id"] = pattern_id
    record["typology"] = typology
    record["typology_detail"] = detail
    record["position_in_stream"] = position
    record["transaction_key"] = TransactionKey(
        timestamp=str(record["timestamp"]),
        src_bank=str(record["src_bank"]),
        src_account=str(record["src_account"]),
        dst_bank=str(record["dst_bank"]),
        dst_account=str(record["dst_account"]),
        amount_paid=str(record["amount_paid"]),
    ).as_string()
    return record


class _StreamState:
    """State machine for the patterns-file grammar.

    Kept as a class so each grammar rule is one small method with its own error message,
    rather than one long loop body where a missing ``continue`` would silently change the
    meaning of the parse.
    """

    def __init__(self, source: str) -> None:
        """Initialise with no stream open.

        Args:
            source: File name, used in every error message.
        """
        self.source = source
        self.typology: str | None = None
        self.detail = ""
        self.pattern_id = ""
        self.position = 0
        self.begin_line = 0
        self._ordinals: dict[str, int] = {}

    def open_stream(self, match: re.Match[str], lineno: int) -> None:
        """Handle a BEGIN delimiter.

        Args:
            match: The matched BEGIN line.
            lineno: 1-based line number.

        Raises:
            PatternsParseError: If a stream is already open, or the typology is unknown.
        """
        if self.typology is not None:
            raise PatternsParseError(
                f"{self.source}:{lineno}: BEGIN inside the stream opened at line "
                f"{self.begin_line}; streams do not nest"
            )
        raw = match.group("typology")
        if raw not in TYPOLOGY_MAP:
            raise PatternsParseError(
                f"{self.source}:{lineno}: unknown typology {raw!r}; "
                f"expected one of {sorted(TYPOLOGY_MAP)}"
            )
        self.typology = TYPOLOGY_MAP[raw]
        self.detail = (match.group("detail") or "").strip()
        self._ordinals[self.typology] = self._ordinals.get(self.typology, 0) + 1
        self.pattern_id = f"{self.typology}_{self._ordinals[self.typology]:05d}"
        self.position = 0
        self.begin_line = lineno

    def close_stream(self, match: re.Match[str], lineno: int) -> None:
        """Handle an END delimiter.

        Args:
            match: The matched END line.
            lineno: 1-based line number.

        Raises:
            PatternsParseError: If no stream is open, the typology does not match the
                open stream, or the stream contained no transactions.
        """
        if self.typology is None:
            raise PatternsParseError(f"{self.source}:{lineno}: END without a matching BEGIN")
        if TYPOLOGY_MAP.get(match.group("typology")) != self.typology:
            raise PatternsParseError(
                f"{self.source}:{lineno}: END typology {match.group('typology')!r} does "
                f"not close the {self.typology!r} stream opened at line {self.begin_line}"
            )
        if self.position == 0:
            raise PatternsParseError(
                f"{self.source}:{lineno}: stream opened at line {self.begin_line} "
                "contains no transactions"
            )
        self.typology = None

    def parse_row(self, line: str, lineno: int) -> dict[str, object]:
        """Handle a line that is neither delimiter.

        Args:
            line: The stripped line.
            lineno: 1-based line number.

        Returns:
            The tidy record for this transaction occurrence.

        Raises:
            PatternsParseError: If the line is a malformed delimiter, falls outside any
                stream, or has the wrong field count.
        """
        # A typo'd delimiter must fail loudly rather than parse as a transaction.
        if "LAUNDERING ATTEMPT" in line:
            raise PatternsParseError(f"{self.source}:{lineno}: malformed delimiter line {line!r}")
        if self.typology is None:
            raise PatternsParseError(
                f"{self.source}:{lineno}: transaction row outside any stream: {line!r}"
            )
        record = _parse_transaction_row(
            line,
            lineno=lineno,
            source=self.source,
            pattern_id=self.pattern_id,
            typology=self.typology,
            detail=self.detail,
            position=self.position,
        )
        self.position += 1
        return record

    def finish(self) -> None:
        """Assert the file ended with no stream left open.

        Raises:
            PatternsParseError: If a stream is still open at EOF.
        """
        if self.typology is not None:
            raise PatternsParseError(
                f"{self.source}: stream opened at line {self.begin_line} is never closed"
            )


def parse_patterns_text(text: str, *, source: str = "<string>") -> pl.DataFrame:
    """Parse patterns-file content. The tested core of :func:`load_patterns`.

    The grammar is a flat sequence of streams::

        BEGIN LAUNDERING ATTEMPT - <TYPOLOGY>[:  <detail>]
        <csv row>
        ...
        END LAUNDERING ATTEMPT - <TYPOLOGY>
        <blank line>

    Args:
        text: Full file content.
        source: Name used in error messages.

    Returns:
        The tidy table described in :func:`load_patterns`.

    Raises:
        PatternsParseError: On any grammar violation. Line numbers are 1-based.
    """
    records: list[dict[str, object]] = []
    state = _StreamState(source=source)

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if begin_match := _BEGIN_RE.match(line):
            state.open_stream(begin_match, lineno)
        elif end_match := _END_RE.match(line):
            state.close_stream(end_match, lineno)
        else:
            records.append(state.parse_row(line, lineno))

    state.finish()

    schema = {
        "pattern_id": pl.Utf8,
        "typology": pl.Utf8,
        "typology_detail": pl.Utf8,
        "transaction_key": pl.Utf8,
        "position_in_stream": pl.UInt32,
        **{name: pl.Utf8 for name in CANONICAL_COLUMNS},
    }
    if not records:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(records).select(list(schema)).cast(schema)  # type: ignore[arg-type]


def require_transaction_keys(txns: pl.DataFrame) -> pl.DataFrame:
    """Return ``txns`` unchanged, having checked it carries ``transaction_key``.

    There is deliberately no fallback that reconstructs the key from typed columns. Doing
    so would mean re-formatting a float back to text, which cannot be made faithful for
    the Bitcoin rows (see the module docstring). A frame that lost the column must be
    re-loaded rather than patched up.

    Args:
        txns: Frame from :func:`load_transactions`.

    Returns:
        The same frame.

    Raises:
        ValueError: If ``transaction_key`` is absent.
    """
    if "transaction_key" not in txns.columns:
        raise ValueError(
            "transactions frame has no 'transaction_key' column. It is built from the "
            "source text during load_transactions() and cannot be reconstructed from the "
            "typed columns — re-load the frame rather than deriving it."
        )
    return txns


def attach_typologies(txns: pl.DataFrame, patterns: pl.DataFrame) -> pl.DataFrame:
    """Label each transaction with its typology.

    Every transaction flagged ``is_laundering`` that matches no pattern stream is labelled
    ``unclassified`` — a first-class member of the controlled vocabulary, not a null. In
    HI-Small there are 1,968 of them, and a narrative must be able to say "flagged, but
    matching no known structural pattern" rather than invent one.

    Args:
        txns: Frame from :func:`load_transactions`.
        patterns: Frame from :func:`load_patterns`.

    Returns:
        ``txns`` with ``transaction_key``, ``typology`` (null for non-laundering rows) and
        ``pattern_id`` (null when unclassified) appended. Where a transaction appears in
        several streams, the first by pattern_id is taken, so the result stays one row per
        input row.

    Raises:
        ValueError: If ``txns`` does not carry ``transaction_key``.
    """
    keyed = require_transaction_keys(txns)
    lookup = (
        patterns.select("transaction_key", "typology", "pattern_id")
        .sort("pattern_id")
        .unique(subset=["transaction_key"], keep="first", maintain_order=True)
    )
    joined = keyed.join(lookup, on="transaction_key", how="left")
    return joined.with_columns(
        pl.when(~pl.col("is_laundering"))
        .then(None)
        .otherwise(pl.col("typology").fill_null("unclassified"))
        .alias("typology")
    )


def build_account_graph(
    txns: pl.DataFrame,
    *,
    graph_id: str = "amlworld_hi_small",
    dataset: str = "amlworld_hi_small",
    provenance: dict[str, object] | None = None,
) -> CanonicalGraph:
    """Build the canonical account graph from a transactions frame.

    Nodes are accounts keyed by ``"<bank>|<account>"``; see the module docstring for why
    the bank code is part of the key. Node attributes are aggregates derived from the
    transactions themselves — AMLworld ships no separate account table.

    Args:
        txns: Frame from :func:`load_transactions`, optionally already passed through
            :func:`attach_typologies`.
        graph_id: Identifier for the resulting graph.
        dataset: Substrate key recorded on the graph.
        provenance: Source files, checksums and anything else the manifest should carry.

    Returns:
        A :class:`CanonicalGraph` whose edges are transactions with full attributes and
        whose availability mask is :data:`AMLWORLD_AVAILABILITY`.

    Raises:
        pl.exceptions.ColumnNotFoundError: If a required column is absent.
    """
    edges = txns.with_columns(
        _account_key(pl.col("src_bank"), pl.col("src_account")).alias("src"),
        _account_key(pl.col("dst_bank"), pl.col("dst_account")).alias("dst"),
    )

    # Node table: union of endpoints, with per-account activity aggregates. Built by
    # aggregating each side separately and outer-joining, which is far cheaper than
    # exploding a five-million-row frame into ten million endpoint rows.
    out_side = edges.group_by("src").agg(
        pl.len().alias("out_degree"),
        pl.col("amount_paid").sum().alias("total_sent"),
        pl.col("timestamp").min().alias("first_sent"),
        pl.col("timestamp").max().alias("last_sent"),
        pl.col("src_bank").first().alias("bank_out"),
    )
    in_side = edges.group_by("dst").agg(
        pl.len().alias("in_degree"),
        pl.col("amount_received").sum().alias("total_received"),
        pl.col("timestamp").min().alias("first_received"),
        pl.col("timestamp").max().alias("last_received"),
        pl.col("dst_bank").first().alias("bank_in"),
    )

    nodes = (
        out_side.join(in_side, left_on="src", right_on="dst", how="full", coalesce=True)
        .rename({"src": "node_id"})
        .with_columns(
            pl.lit("account").alias("node_type"),
            pl.coalesce("bank_out", "bank_in").alias("bank"),
            pl.col("in_degree").fill_null(0).cast(pl.UInt32),
            pl.col("out_degree").fill_null(0).cast(pl.UInt32),
            pl.col("total_sent").fill_null(0.0),
            pl.col("total_received").fill_null(0.0),
            pl.min_horizontal("first_sent", "first_received").alias("first_seen"),
            pl.max_horizontal("last_sent", "last_received").alias("last_seen"),
        )
        .with_columns((pl.col("in_degree") + pl.col("out_degree")).alias("degree"))
        .drop("bank_out", "bank_in", "first_sent", "last_sent", "first_received", "last_received")
        .select(
            "node_id",
            "node_type",
            "bank",
            "in_degree",
            "out_degree",
            "degree",
            "total_received",
            "total_sent",
            "first_seen",
            "last_seen",
        )
        .sort("node_id")
    )

    edge_columns = [
        "src",
        "dst",
        "timestamp",
        "amount_received",
        "receiving_currency",
        "amount_paid",
        "payment_currency",
        "payment_format",
        "is_laundering",
    ]
    for optional in ("typology", "pattern_id", "transaction_key"):
        if optional in edges.columns:
            edge_columns.append(optional)

    return CanonicalGraph(
        graph_id=graph_id,
        dataset=dataset,
        nodes=nodes,
        edges=edges.select(edge_columns),
        node_feature_names=[
            "in_degree",
            "out_degree",
            "degree",
            "total_received",
            "total_sent",
        ],
        edge_feature_names=["amount_received", "amount_paid"],
        availability=AMLWORLD_AVAILABILITY,
        label=None,
        typology=None,
        provenance=dict(provenance or {}),
    )


def typology_counts(patterns: pl.DataFrame, txns: pl.DataFrame | None = None) -> dict[str, int]:
    """Count transactions per typology, in the same terms as the published table.

    Args:
        patterns: Frame from :func:`load_patterns`.
        txns: Optional transactions frame. When given, ``unclassified`` is included: the
            laundering-flagged transactions that appear in no stream.

    Returns:
        Typology to transaction count. Counts occurrences within streams, so a transaction
        appearing in two streams contributes to both — which is how the published figures
        are constructed.

    Raises:
        ValueError: If ``txns`` is given but does not carry ``transaction_key``.
    """
    counts = {
        row["typology"]: int(row["len"])
        for row in patterns.group_by("typology").len().sort("typology").to_dicts()
    }
    if txns is not None:
        keyed = require_transaction_keys(txns).filter(pl.col("is_laundering"))
        patterned = patterns.select("transaction_key").unique()
        counts["unclassified"] = int(keyed.join(patterned, on="transaction_key", how="anti").height)
    return counts


def verify_published_statistics(
    graph: CanonicalGraph, size: str = "HI-Small"
) -> dict[str, dict[str, int | bool]]:
    """Compare a built graph against the published node and edge counts.

    Args:
        graph: Graph from :func:`build_account_graph`.
        size: Variant whose published figures to compare against.

    Returns:
        Per-quantity ``{"published": int, "observed": int, "matches": bool}``. Empty when
        no figures are published for ``size``.

    Raises:
        ValueError: If ``size`` is not a known variant.
    """
    _check_size(size)
    published = PUBLISHED_STATISTICS.get(size)
    if published is None:
        return {}
    observed = {"num_nodes": graph.num_nodes, "num_edges": graph.num_edges}
    return {
        key: {
            "published": value,
            "observed": observed[key],
            "matches": observed[key] == value,
        }
        for key, value in published.items()
    }
