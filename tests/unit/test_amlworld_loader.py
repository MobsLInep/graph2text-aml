"""AMLworld loader against a 500-row fixture slice.

The fixture is a genuine slice of HI-Small, not a hand-written imitation, so it carries
the schema quirks that matter: the duplicated ``Account`` header, zero-padded bank codes,
and — spliced in deliberately — six-decimal Bitcoin amounts, which are the case that broke
the transaction-key join during development.

The statistics regression against the real 5M-row file lives in
``tests/integration/test_amlworld_statistics.py`` and skips when the data is absent.
"""

from __future__ import annotations

import polars as pl
import pytest

from g2t_aml.data.canonical import AMLWORLD_AVAILABILITY, CanonicalGraph
from g2t_aml.data.loaders import amlworld as aml

FIXTURE_ROWS = 500


@pytest.fixture
def raw_dir(repo_root):
    """The fixture tree, laid out exactly like a real data/raw."""
    return repo_root / "tests" / "fixtures"


@pytest.fixture
def txns(raw_dir):
    return aml.load_transactions("HI-Small", raw_dir=raw_dir)


@pytest.fixture
def patterns(raw_dir):
    return aml.load_patterns("HI-Small", raw_dir=raw_dir)


# ------------------------------------------------------------------- header ---


def test_header_is_asserted_against_the_file_not_recalled(raw_dir):
    path = aml.transactions_path(raw_dir, "HI-Small")
    assert aml.read_header(path) == aml.EXPECTED_HEADER


def test_expected_header_contains_the_duplicated_account_column():
    """Both accounts share a base name; the loader must not rely on names being unique."""
    assert aml.EXPECTED_HEADER.count("Account") == 2


def test_header_mismatch_is_fatal(tmp_path):
    directory = tmp_path / "amlworld"
    directory.mkdir()
    (directory / "HI-Small_Trans.csv").write_text("Timestamp,From Bank,Account\n1,2,3\n")
    with pytest.raises(aml.HeaderMismatchError, match="Refusing to guess"):
        aml.load_transactions("HI-Small", raw_dir=tmp_path)


def test_unknown_size_is_rejected(raw_dir):
    with pytest.raises(ValueError, match="unknown AMLworld size"):
        aml.load_transactions("HI-Enormous", raw_dir=raw_dir)


# -------------------------------------------------------------- transactions ---


def test_loads_every_fixture_row(txns):
    assert txns.height == FIXTURE_ROWS


def test_columns_are_renamed_positionally(txns):
    for column in aml.CANONICAL_COLUMNS:
        assert column in txns.columns
    assert "Account_duplicated_0" not in txns.columns


def test_src_and_dst_accounts_are_distinct_columns(txns):
    """The whole point of the positional rename: the two Account columns stay separate."""
    non_loops = txns.filter(pl.col("src_account") != pl.col("dst_account"))
    assert non_loops.height > 0


def test_dtypes(txns):
    assert txns.schema["timestamp"] == pl.Datetime("us")
    assert txns.schema["amount_paid"] == pl.Float64
    assert txns.schema["is_laundering"] == pl.Boolean
    assert txns.schema["src_bank"] == pl.Utf8


def test_bank_codes_keep_their_leading_zeros(txns):
    """Reading '010' as the integer 10 would collide it with bank '10'."""
    assert any(b.startswith("0") for b in txns["src_bank"].to_list())


def test_n_rows_subsets(raw_dir):
    assert aml.load_transactions("HI-Small", raw_dir=raw_dir, n_rows=10).height == 10


def test_transaction_key_is_present_on_load(txns):
    assert "transaction_key" in txns.columns
    assert txns["transaction_key"].null_count() == 0


def test_transaction_key_preserves_six_decimal_bitcoin_amounts(txns):
    """The exact case that lost a transaction when the key was rebuilt from a float."""
    btc = txns.filter(pl.col("payment_currency") == "Bitcoin")
    assert btc.height > 0, "fixture must contain Bitcoin rows"
    decimals = [len(k.rsplit("|", 1)[1].split(".")[1]) for k in btc["transaction_key"]]
    assert all(d == 6 for d in decimals), f"six-decimal amounts were normalised: {decimals}"


def test_require_transaction_keys_refuses_to_reconstruct(txns):
    stripped = txns.drop("transaction_key")
    with pytest.raises(ValueError, match="cannot be reconstructed"):
        aml.require_transaction_keys(stripped)


# ------------------------------------------------------------------ patterns ---


def test_patterns_fixture_parses(patterns):
    assert patterns.height == 7
    assert patterns["pattern_id"].n_unique() == 3


def test_pattern_rows_join_to_the_transactions(txns, patterns):
    """Every pattern transaction must be findable in the CSV. This is the join that
    the six-decimal amounts broke."""
    keys = set(txns["transaction_key"].to_list())
    unmatched = [k for k in patterns["transaction_key"].to_list() if k not in keys]
    assert unmatched == []


def test_attach_typologies_labels_patterned_transactions(txns, patterns):
    labelled = aml.attach_typologies(txns, patterns)
    counts = aml.typology_counts(patterns, txns)
    assert counts["fan_out"] == 3
    assert counts["cycle"] == 2
    assert counts["stack"] == 2
    assert labelled.filter(pl.col("typology") == "fan_out").height == 3


def test_unclassified_covers_flagged_transactions_matching_no_pattern(txns, patterns):
    """A laundering row absent from every stream is `unclassified`, never null."""
    extra = txns.head(1).with_columns(
        pl.lit(True).alias("is_laundering"),
        pl.lit("SYNTHETIC-UNMATCHED-KEY").alias("transaction_key"),
    )
    counts = aml.typology_counts(patterns, pl.concat([txns, extra]))
    assert counts["unclassified"] >= 1


def test_non_laundering_rows_get_a_null_typology(txns, patterns):
    labelled = aml.attach_typologies(txns, patterns)
    clean = labelled.filter(~pl.col("is_laundering"))
    assert clean["typology"].null_count() == clean.height


def test_attach_typologies_preserves_row_count(txns, patterns):
    """A transaction in two streams must not duplicate the row."""
    assert aml.attach_typologies(txns, patterns).height == txns.height


# --------------------------------------------------------------------- graph ---


def test_builds_a_canonical_graph(txns, patterns):
    graph = aml.build_account_graph(aml.attach_typologies(txns, patterns))
    assert isinstance(graph, CanonicalGraph)
    assert graph.num_edges == FIXTURE_ROWS
    assert graph.availability == AMLWORLD_AVAILABILITY


def test_nodes_are_keyed_by_bank_and_account(txns):
    graph = aml.build_account_graph(txns)
    assert all("|" in n for n in graph.nodes["node_id"].to_list())


def test_node_key_prevents_cross_bank_collisions():
    """Account ids are unique only within a bank. Eight collide in the real HI-Small."""
    frame = pl.DataFrame(
        {
            "timestamp": ["2022/09/01 00:01", "2022/09/01 00:02"],
            "src_bank": ["001", "002"],
            "src_account": ["SHARED01", "SHARED01"],  # same account id, different banks
            "dst_bank": ["003", "003"],
            "dst_account": ["ACCT0001", "ACCT0001"],
            "amount_received": ["1.00", "2.00"],
            "receiving_currency": ["Euro", "Euro"],
            "amount_paid": ["1.00", "2.00"],
            "payment_currency": ["Euro", "Euro"],
            "payment_format": ["ACH", "ACH"],
            "is_laundering": ["0", "0"],
        }
    ).with_columns(
        pl.col("timestamp").str.to_datetime(aml.TIMESTAMP_FORMAT),
        pl.col("amount_received").cast(pl.Float64),
        pl.col("amount_paid").cast(pl.Float64),
        pl.col("is_laundering").cast(pl.Int8).cast(pl.Boolean),
    )
    graph = aml.build_account_graph(frame)
    assert graph.num_nodes == 3  # not 2: the two SHARED01 accounts are different nodes


def test_graph_has_referential_integrity(txns):
    aml.build_account_graph(txns).validate_referential_integrity()


def test_node_aggregates_are_consistent_with_the_edges(txns):
    graph = aml.build_account_graph(txns)
    assert int(graph.nodes["out_degree"].sum()) == graph.num_edges
    assert int(graph.nodes["in_degree"].sum()) == graph.num_edges


def test_declared_node_features_exist(txns):
    graph = aml.build_account_graph(txns)
    for name in graph.node_feature_names:
        assert name in graph.nodes.columns


def test_graph_round_trips_through_disk(tmp_path, txns, patterns):
    """Golden-file test: a real 500-row slice survives the canonical representation."""
    original = aml.build_account_graph(aml.attach_typologies(txns, patterns))
    original.save(tmp_path)
    restored = CanonicalGraph.load(tmp_path)
    assert restored.nodes.equals(original.nodes)
    assert restored.edges.equals(original.edges)
    assert restored.availability == original.availability
    assert restored.summary() == original.summary()


def test_verify_published_statistics_reports_a_mismatch(txns):
    """The fixture is a 500-row slice and must not be mistaken for the full dataset."""
    table = aml.verify_published_statistics(aml.build_account_graph(txns), "HI-Small")
    assert table["num_edges"]["matches"] is False
    assert table["num_edges"]["published"] == 5_078_345


def test_published_typology_counts_sum_to_the_laundering_total():
    """3,209 patterned + 1,968 unclassified = 5,177 laundering transactions."""
    published = aml.PUBLISHED_TYPOLOGY_COUNTS["HI-Small"]
    patterned = sum(v for k, v in published.items() if k != "unclassified")
    assert patterned == 3_209
    assert patterned + published["unclassified"] == 5_177
