"""End-to-end encoder tests against the real case store, when one is present.

These are the tests that would catch a mismatch between the cache, the frozen manifest and
the model — the class of failure that unit tests over synthetic fixtures cannot see,
because the fixture and the code agree by construction.

Every test here skips cleanly when the corpus is not on disk. `data/` is gitignored, so a
fresh checkout has no case store, and a skipped assertion is the honest outcome rather
than a failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from g2t_aml.models.encoder.dataset import (  # noqa: E402
    MANIFEST_SPLITS,
    REALISTIC_SPLIT,
    TYPOLOGY_CLASSES,
    load_case_ids,
    load_feature_space,
    load_split,
    verify_cache_against_manifest,
)
from g2t_aml.models.encoder.registry import ARMS, build_encoder  # noqa: E402
from g2t_aml.utils.io import read_json  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed" / "amlworld_hi_small"
CACHE = PROCESSED / "encoder" / "features"
SPLITS = REPO_ROOT / "schemas" / "splits" / "amlworld"

pytestmark = pytest.mark.skipif(
    not (CACHE / "cache_manifest.json").is_file(),
    reason="the encoder feature cache is not built in this checkout (`make encoder-features`)",
)


@pytest.fixture(scope="module")
def space():
    return load_feature_space(CACHE)


@pytest.fixture(scope="module")
def test_split():
    return load_split(CACHE, "test")


# ------------------------------------------------------------------- splits ---


def test_cache_agrees_with_the_frozen_manifest():
    """The check that stops a stale cache being trained on. Invariant 2."""
    verify_cache_against_manifest(CACHE, SPLITS)


def test_cached_counts_are_exactly_the_manifest_counts():
    manifest = read_json(CACHE / "cache_manifest.json")
    ids = load_case_ids(SPLITS)
    for name in MANIFEST_SPLITS:
        assert manifest["counts"][name] == len(ids[name]), name


def test_cached_case_ids_are_the_manifest_ids_in_order():
    """Not merely the same count: the same cases, in the same order."""
    ids = load_case_ids(SPLITS)
    for name in MANIFEST_SPLITS:
        cached = [g.case_id for g in load_split(CACHE, name)]
        assert cached == ids[name], f"{name} split diverged from the frozen manifest"


def test_no_case_appears_in_two_splits():
    seen: dict[str, str] = {}
    for name in MANIFEST_SPLITS:
        for graph in load_split(CACHE, name):
            assert (
                graph.case_id not in seen
            ), f"{graph.case_id} is in both {seen[graph.case_id]} and {name}"
            seen[graph.case_id] = name


def test_the_realistic_stream_is_evaluation_only_and_more_imbalanced():
    """D-023: the honest operating point, at a much lower prevalence than the training set."""
    manifest = read_json(CACHE / "cache_manifest.json")
    if REALISTIC_SPLIT not in manifest["counts"]:
        pytest.skip("the realistic-imbalance stream is not built")
    train_rate = manifest["positives"]["train"] / manifest["counts"]["train"]
    stream_rate = manifest["positives"][REALISTIC_SPLIT] / manifest["counts"][REALISTIC_SPLIT]
    assert stream_rate < train_rate / 1.5, (train_rate, stream_rate)


# ------------------------------------------------------------------ tensors ---


def test_real_cases_carry_no_label_proxy_tensor(test_split, space):
    """The standing leakage check, run against the corpus the model actually trains on."""
    from g2t_aml.data.leakage_audit import LABEL_PROXY_COLUMNS

    graph = test_split[0]
    tensor_names = {k for k, v in graph.items() if torch.is_tensor(v)}
    # y and y_typ are supervision targets, not features, and are named so.
    assert tensor_names & LABEL_PROXY_COLUMNS == set()
    assert graph.x.shape[1] == space.node_dim
    assert graph.edge_attr.shape[1] == space.edge_continuous_dim


def test_every_real_case_is_finite_and_well_formed(test_split, space):
    for graph in test_split[:500]:
        assert graph.x.shape[1] == space.node_dim
        assert torch.isfinite(graph.x).all(), graph.case_id
        assert torch.isfinite(graph.edge_attr).all(), graph.case_id
        assert graph.num_nodes >= 1
        if graph.edge_index.numel():
            assert int(graph.edge_index.max()) < graph.num_nodes, graph.case_id
        assert int(graph.y.item()) in (0, 1)
        assert -1 <= int(graph.y_typ.item()) < len(TYPOLOGY_CLASSES)


def test_currency_indices_stay_inside_the_fitted_vocabulary(test_split, space):
    """A test-window currency absent from the training split must land on the OOV slot."""
    for graph in test_split[:1000]:
        for name in ("edge_currency_paid", "edge_currency_received"):
            values = getattr(graph, name)
            if values.numel():
                assert int(values.max()) < space.n_currencies, graph.case_id
                assert int(values.min()) >= 0
        if graph.edge_format.numel():
            assert int(graph.edge_format.max()) < space.n_payment_formats


# -------------------------------------------------------------------- model ---


@pytest.mark.parametrize("arch", sorted(ARMS))
def test_every_arm_runs_on_real_batches(test_split, space, arch):
    from omegaconf import OmegaConf
    from torch_geometric.loader import DataLoader

    cfg = OmegaConf.load(REPO_ROOT / "configs" / "encoder" / f"{arch}.yaml")
    model = build_encoder(cfg, space).eval()
    batch = next(iter(DataLoader(test_split[:64], batch_size=64, shuffle=False)))

    with torch.no_grad():
        out = model(batch, want_attention=True)
    assert out.pooled_tokens.shape == (64, int(cfg.n_pooled_tokens), int(cfg.hidden_dim))
    assert out.risk_logits.shape == (64, 1)
    assert torch.isfinite(out.risk_logits).all()
    assert torch.isfinite(out.pooled_tokens).all()


def test_a_batch_of_degenerate_cases_still_encodes(test_split, space):
    """18.1% of the corpus is a two-account, one-transaction case. It must not crash."""
    from omegaconf import OmegaConf
    from torch_geometric.loader import DataLoader

    tiny = [g for g in test_split if g.num_nodes <= 2][:32]
    if not tiny:
        pytest.skip("no degenerate cases in this split")

    cfg = OmegaConf.load(REPO_ROOT / "configs" / "encoder" / "gatv2.yaml")
    model = build_encoder(cfg, space).eval()
    batch = next(iter(DataLoader(tiny, batch_size=len(tiny), shuffle=False)))
    with torch.no_grad():
        out = model(batch)
    assert torch.isfinite(out.risk_logits).all()
