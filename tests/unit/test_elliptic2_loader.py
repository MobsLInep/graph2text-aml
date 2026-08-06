"""Elliptic2 loader.

Elliptic2 is access-gated, so tests touching the real files are marked ``skipif`` and the
rest run against a synthetic tree laid out to the documented schema. That split is
deliberate: the phase must not block on data we may not have been granted, but the loader
must still be exercised.

The most important assertions here are about what the loader refuses to do — guess a
column by position, and attach any meaning to an anonymised feature.
"""

from __future__ import annotations

import polars as pl
import pytest

from g2t_aml.data.canonical import ELLIPTIC2_AVAILABILITY
from g2t_aml.data.loaders import elliptic2 as ell

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def fake_root(tmp_path):
    """A synthetic Elliptic2 tree matching the documented file schema.

    Invariant 8: every identifier here is synthetic. Feature columns are deliberately
    named ``feat_*`` with meaningless values, because the real ones are anonymised and no
    test should imply otherwise.
    """
    root = tmp_path / "elliptic2"
    root.mkdir()
    (root / "background_nodes.csv").write_text(
        "clusterId,feat_0,feat_1\n" "C0001,0.1,0.2\nC0002,0.3,0.4\nC0003,0.5,0.6\nC0009,0.7,0.8\n"
    )
    (root / "background_edges.csv").write_text(
        "clusterId1,clusterId2\nC0001,C0002\nC0002,C0003\nC0003,C0009\n"
    )
    (root / "connected_components.csv").write_text(
        "ccId,clusterId,ccLabel\n"
        "SG1,C0001,suspicious\nSG1,C0002,suspicious\nSG1,C0003,suspicious\n"
        "SG2,C0009,licit\n"
    )
    (root / "nodes.csv").write_text("clusterId\nC0001\n")
    (root / "edges.csv").write_text("clusterId1,clusterId2\nC0001,C0002\n")
    return tmp_path


def test_is_available_detects_a_complete_tree(fake_root):
    assert ell.is_available(fake_root) is True


def test_is_available_is_false_when_a_file_is_missing(fake_root, tmp_path):
    (tmp_path / "elliptic2" / "edges.csv").unlink()
    assert ell.is_available(fake_root) is False


def test_is_available_does_not_raise_on_an_empty_tree(tmp_path):
    assert ell.is_available(tmp_path) is False


def test_missing_files_explain_how_to_request_access(tmp_path):
    with pytest.raises(ell.Elliptic2UnavailableError, match="elliptic.co/elliptic2"):
        ell.load_labelled_subgraphs(tmp_path)


def test_loads_labelled_subgraphs(fake_root):
    memberships = ell.load_labelled_subgraphs(fake_root)
    assert memberships.height == 4
    assert set(memberships.columns) == {"subgraph_id", "node_id", "label"}


def test_labels_are_normalised_to_the_controlled_pair(fake_root):
    labels = set(ell.load_labelled_subgraphs(fake_root)["label"].unique())
    assert labels <= set(ell.LABELS)


def test_integer_label_encoding_is_normalised(tmp_path):
    root = tmp_path / "elliptic2"
    root.mkdir()
    for name in ell.REQUIRED_FILES:
        (root / name).write_text("clusterId\nC1\n")
    (root / "connected_components.csv").write_text("ccId,clusterId,ccLabel\nS1,C1,1\nS2,C2,2\n")
    labels = ell.load_labelled_subgraphs(tmp_path)["label"].to_list()
    assert labels == [ell.LABEL_SUSPICIOUS, ell.LABEL_LICIT]


def test_an_unrecognised_label_is_rejected_not_defaulted(tmp_path):
    """Defaulting a label to licit is how a suspicious subgraph becomes invisible."""
    root = tmp_path / "elliptic2"
    root.mkdir()
    for name in ell.REQUIRED_FILES:
        (root / name).write_text("clusterId\nC1\n")
    (root / "connected_components.csv").write_text("ccId,clusterId,ccLabel\nS1,C1,maybe\n")
    with pytest.raises(ell.Elliptic2SchemaError, match="unrecognised subgraph labels"):
        ell.load_labelled_subgraphs(tmp_path)


def test_a_missing_column_is_reported_rather_than_guessed_by_position(tmp_path):
    root = tmp_path / "elliptic2"
    root.mkdir()
    for name in ell.REQUIRED_FILES:
        (root / name).write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ell.Elliptic2SchemaError, match="Refusing to guess by position"):
        ell.load_labelled_subgraphs(tmp_path)


def test_subgraph_labels_reduce_to_one_row_each(fake_root):
    summary = ell.subgraph_labels(ell.load_labelled_subgraphs(fake_root))
    assert summary.height == 2
    assert summary.filter(pl.col("subgraph_id") == "SG1")["num_nodes"][0] == 3


def test_conflicting_labels_within_a_subgraph_are_rejected():
    memberships = pl.DataFrame(
        {
            "subgraph_id": ["S1", "S1"],
            "node_id": ["C1", "C2"],
            "label": ["licit", "suspicious"],
        }
    )
    with pytest.raises(ell.Elliptic2SchemaError, match="conflicting labels"):
        ell.subgraph_labels(memberships)


def test_background_graph_is_lazy(fake_root):
    """49M nodes must never be materialised by opening the handle."""
    background = ell.load_background_graph(fake_root)
    assert isinstance(background.nodes, pl.LazyFrame)
    assert isinstance(background.edges, pl.LazyFrame)


def test_background_feature_columns_are_carried_without_interpretation(fake_root):
    background = ell.load_background_graph(fake_root)
    assert background.feature_columns == ("feat_0", "feat_1")


def test_builds_a_subgraph(fake_root):
    graph = ell.build_subgraph("SG1", raw_dir=fake_root)
    assert graph.num_nodes == 3
    assert graph.num_edges == 2  # C0003->C0009 leaves the subgraph and is excluded
    assert graph.label == "suspicious"


def test_subgraph_nodes_are_clusters_not_accounts(fake_root):
    """The unit is a set of addresses believed to share ownership."""
    graph = ell.build_subgraph("SG1", raw_dir=fake_root)
    assert set(graph.nodes["node_type"].unique()) == {"cluster"}


def test_subgraph_carries_the_masked_availability(fake_root):
    graph = ell.build_subgraph("SG1", raw_dir=fake_root)
    assert graph.availability == ELLIPTIC2_AVAILABILITY
    assert graph.availability.monetary_amounts is False
    assert graph.availability.semantic_node_features is False


def test_subgraph_typology_is_none_not_unclassified(fake_root):
    """None means no typology truth exists; unclassified would be a positive claim."""
    assert ell.build_subgraph("SG1", raw_dir=fake_root).typology is None


def test_subgraph_provenance_records_the_anonymisation(fake_root):
    provenance = ell.build_subgraph("SG1", raw_dir=fake_root).provenance
    assert "anonymised" in provenance["features"]
    assert provenance["unit"] == "bitcoin address cluster"


def test_subgraph_features_keep_their_opaque_names(fake_root):
    graph = ell.build_subgraph("SG1", raw_dir=fake_root)
    assert graph.node_feature_names == ["feat_0", "feat_1"]


def test_unknown_subgraph_id_raises(fake_root):
    with pytest.raises(KeyError, match="not in the membership table"):
        ell.build_subgraph("NOPE", raw_dir=fake_root)


def test_subgraph_has_referential_integrity(fake_root):
    ell.build_subgraph("SG1", raw_dir=fake_root).validate_referential_integrity()


def test_published_figures_are_recorded_for_the_data_card():
    assert ell.PUBLISHED_STATISTICS["num_labelled_subgraphs"] == 122_000
