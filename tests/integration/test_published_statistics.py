"""Statistics regression against the real substrates.

This is the Phase 1 gate expressed as a test: loading HI-Small must reproduce the counts
published by Altman et al., and parsing the patterns file must reproduce every typology
count in the published table. A loader that quietly disagrees with the paper it cites is
the specific failure this phase exists to prevent, so these are assertions rather than
warnings.

Both substrates skip when their raw data is absent, so a clean checkout with no data still
runs a green suite. That is a real trade-off — CI cannot enforce these numbers, since the
data is 475 MB and Elliptic2 is not redistributable at all — and the mitigation is that
``scripts/01_ingest.py`` performs the same checks and aborts the ingest on a mismatch.
"""

from __future__ import annotations

import pytest

from g2t_aml.data.download import is_available
from g2t_aml.data.loaders import amlworld as aml
from g2t_aml.data.loaders import elliptic2 as ell
from g2t_aml.data.stats import compute_dataset_statistics

pytestmark = [pytest.mark.integration, pytest.mark.slow]

RAW_DIR_NAME = "data/raw"

# Observed on 2026-08-01 from the checksum-verified HI-Small release. These are recorded
# so a change in the loader that shifts them is caught, not just a change that breaks the
# two headline counts.
OBSERVED_HI_SMALL = {
    "num_laundering_edges": 5_177,
    "num_self_loops": 591_212,
    "num_distinct_pairs": 1_015_736,
    "num_components": 114_139,
    "largest_component_size": 372_089,
    "num_pattern_streams": 370,
    "num_pattern_transactions": 3_209,
    "num_currencies": 15,
    "num_payment_formats": 7,
}


@pytest.fixture(scope="module")
def raw_dir(repo_root):
    return repo_root / RAW_DIR_NAME


@pytest.fixture(scope="module")
def amlworld_available(repo_root):
    return is_available("amlworld_hi_small", repo_root / RAW_DIR_NAME)


@pytest.fixture(scope="module")
def hi_small(raw_dir, amlworld_available):
    if not amlworld_available:
        pytest.skip("AMLworld HI-Small is not present under data/raw/amlworld")
    txns = aml.load_transactions("HI-Small", raw_dir=raw_dir)
    patterns = aml.load_patterns("HI-Small", raw_dir=raw_dir)
    graph = aml.build_account_graph(aml.attach_typologies(txns, patterns))
    return txns, patterns, graph


# --------------------------------------------------------------- AMLworld ---


def test_reproduces_the_published_node_and_edge_counts(hi_small):
    """515,088 accounts and 5,078,345 transactions (Altman et al., 2023)."""
    _, _, graph = hi_small
    assert graph.num_nodes == 515_088
    assert graph.num_edges == 5_078_345


def test_verify_published_statistics_reports_a_clean_match(hi_small):
    _, _, graph = hi_small
    table = aml.verify_published_statistics(graph, "HI-Small")
    assert all(entry["matches"] for entry in table.values()), table


def test_node_count_requires_the_bank_account_composite_key(hi_small):
    """Keying on the account id alone gives 515,080 -- eight ids collide across banks."""
    txns, _, graph = hi_small
    accounts_only = (
        txns.select("src_account")
        .rename({"src_account": "a"})
        .vstack(txns.select("dst_account").rename({"dst_account": "a"}))
        .n_unique()
    )
    assert accounts_only == 515_080
    assert graph.num_nodes == 515_088
    assert graph.num_nodes - accounts_only == 8


def test_reproduces_every_published_typology_count(hi_small):
    txns, patterns, _ = hi_small
    observed = aml.typology_counts(patterns, txns)
    published = aml.PUBLISHED_TYPOLOGY_COUNTS["HI-Small"]
    assert observed == published


def test_unclassified_is_the_laundering_remainder(hi_small):
    """1,968 flagged transactions match none of the eight structural patterns."""
    txns, patterns, _ = hi_small
    observed = aml.typology_counts(patterns, txns)
    total_laundering = txns.filter(txns["is_laundering"]).height
    patterned = sum(v for k, v in observed.items() if k != "unclassified")
    assert total_laundering == 5_177
    assert observed["unclassified"] == total_laundering - patterned == 1_968


def test_patterns_file_stream_and_row_counts(hi_small):
    _, patterns, _ = hi_small
    assert patterns["pattern_id"].n_unique() == OBSERVED_HI_SMALL["num_pattern_streams"]
    assert patterns.height == OBSERVED_HI_SMALL["num_pattern_transactions"]


def test_every_pattern_transaction_joins_to_the_csv(hi_small):
    """The join that a float-formatted key silently broke for the Bitcoin rows."""
    txns, patterns, _ = hi_small
    keys = set(txns["transaction_key"].to_list())
    unmatched = [k for k in patterns["transaction_key"].to_list() if k not in keys]
    assert unmatched == []


def test_laundering_rate_is_of_the_documented_order(hi_small):
    """Order 1 in 1,000 to 1 in 2,000 for the HI variants."""
    txns, _, _ = hi_small
    rate = txns.filter(txns["is_laundering"]).height / txns.height
    assert 1 / 2_000 <= rate <= 1 / 500


def test_structural_and_component_statistics_are_stable(hi_small):
    _, _, graph = hi_small
    stats = compute_dataset_statistics(graph)
    assert (
        stats["class_balance"]["num_laundering_edges"]
        == (OBSERVED_HI_SMALL["num_laundering_edges"])
    )
    assert stats["structural"]["num_self_loops"] == OBSERVED_HI_SMALL["num_self_loops"]
    assert stats["structural"]["num_distinct_pairs"] == (OBSERVED_HI_SMALL["num_distinct_pairs"])
    assert stats["components"]["num_components"] == OBSERVED_HI_SMALL["num_components"]
    assert (
        stats["components"]["largest_component_size"]
        == (OBSERVED_HI_SMALL["largest_component_size"])
    )
    assert len(stats["currency_distribution"]["payment"]) == (OBSERVED_HI_SMALL["num_currencies"])
    assert len(stats["payment_format_distribution"]) == (OBSERVED_HI_SMALL["num_payment_formats"])


def test_graph_is_referentially_intact(hi_small):
    _, _, graph = hi_small
    graph.validate_referential_integrity()


def test_temporal_span_matches_the_release(hi_small):
    _, _, graph = hi_small
    temporal = compute_dataset_statistics(graph, include_components=False)["temporal"]
    assert temporal["first"].startswith("2022-09-01")
    assert temporal["granularity"] == "minute"


# --------------------------------------------------------------- Elliptic2 ---
# Access-gated and not redistributable: these skip until access is granted.


@pytest.fixture(scope="module")
def elliptic2_memberships(raw_dir):
    if not ell.is_available(raw_dir):
        pytest.skip("Elliptic2 is access-gated and not present under data/raw/elliptic2")
    return ell.load_labelled_subgraphs(raw_dir)


def test_elliptic2_has_the_published_number_of_labelled_subgraphs(elliptic2_memberships):
    summary = ell.subgraph_labels(elliptic2_memberships)
    # The published figure is rounded to 122K, so this is a tolerance not an equality.
    assert abs(summary.height - ell.PUBLISHED_STATISTICS["num_labelled_subgraphs"]) < 2_000


def test_elliptic2_labels_are_the_documented_pair(elliptic2_memberships):
    assert set(elliptic2_memberships["label"].unique()) <= set(ell.LABELS)
