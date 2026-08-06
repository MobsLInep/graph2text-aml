"""Arm tests: shapes, the overfit wiring check, edge-feature use, and determinism.

The overfit test is the one that catches wiring bugs. An arm that cannot drive the loss
near zero on twenty cases has something disconnected — an aggregation that discards the
batch dimension, a pooling head that averages away the signal, a head reading the wrong
tensor — and every one of those failures still produces plausible-looking training curves
on the real corpus.
"""

from __future__ import annotations

import pytest

from g2t_aml.models.encoder.dataset import TYPOLOGY_CLASSES
from g2t_aml.models.encoder.features import FEATURE_SPEC_VERSION, FeatureSpace

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch, Data  # noqa: E402

from g2t_aml.models.encoder.losses import (  # noqa: E402
    FocalLoss,
    WeightedBCELoss,
    build_binary_loss,
    inverse_frequency_weights,
)
from g2t_aml.models.encoder.registry import (  # noqa: E402
    ARMS,
    UnknownArmError,
    build_encoder,
    count_parameters,
)

HIDDEN = 64
TOKENS = 4


@pytest.fixture
def space() -> FeatureSpace:
    return FeatureSpace(
        version=FEATURE_SPEC_VERSION,
        dataset="amlworld_hi_small",
        currencies={"US Dollar": 1, "Euro": 2},
        payment_formats={"Wire": 1, "Cheque": 2},
        amount_stats={"US Dollar": (7.4, 3.4)},
        global_amount_stats=(7.4, 3.4),
        lap_pe_dim=8,
        rw_pe_dim=16,
        n_train_cases=100,
        availability={
            "monetary_amounts": True,
            "absolute_timestamps": True,
            "institution_identity": True,
        },
    )


def _config(arch: str, **overrides: object):
    from omegaconf import OmegaConf

    base = {
        "name": arch,
        "arch": arch,
        "hidden_dim": HIDDEN,
        "layers": 2,
        "heads": 4,
        "concat_heads": True,
        "dropout": 0.0,
        "edge_dim": 16,
        "residual": True,
        "n_pooled_tokens": TOKENS,
        "typology_head": True,
        "use_edge_features": arch != "mlp",
    }
    return OmegaConf.create(base | overrides)


def _case(space: FeatureSpace, n: int, m: int, *, label: int = 1, seed: int = 0) -> Data:
    generator = torch.Generator().manual_seed(seed)
    data = Data(
        x=torch.randn(n, space.node_dim, generator=generator),
        edge_index=torch.randint(0, n, (2, m), generator=generator),
        edge_attr=torch.randn(m, space.edge_continuous_dim, generator=generator),
    )
    data.edge_currency_paid = torch.randint(0, 3, (m,), generator=generator)
    data.edge_currency_received = torch.randint(0, 3, (m,), generator=generator)
    data.edge_format = torch.randint(0, 3, (m,), generator=generator)
    data.y = torch.tensor([label])
    data.y_typ = torch.tensor([label * 3])
    data.case_id = f"case-{seed}"
    data.node_ids = [f"N{i}" for i in range(n)]
    data.num_nodes = n
    return data


# ------------------------------------------------------------------- shapes ---


@pytest.mark.parametrize("arch", sorted(ARMS))
def test_every_arm_returns_the_declared_shapes(space, arch):
    """`pooled_tokens` must be [B, k, d] for every arm: Phase 8 consumes it directly."""
    model = build_encoder(_config(arch), space)
    graphs = [_case(space, 5, 9, seed=0), _case(space, 11, 25, seed=1), _case(space, 2, 1, seed=2)]
    out = model(Batch.from_data_list(graphs), want_attention=True)

    assert out.pooled_tokens.shape == (3, TOKENS, HIDDEN)
    assert out.graph_embedding.shape == (3, HIDDEN)
    assert out.risk_logits.shape == (3, 1)
    assert out.typology_logits.shape == (3, len(TYPOLOGY_CLASSES))
    assert out.node_embeddings.shape == (5 + 11 + 2, HIDDEN)
    assert "pooling" in out.attention_weights
    assert out.attention_weights["pooling"].shape == (5 + 11 + 2, TOKENS)


@pytest.mark.parametrize("arch", sorted(ARMS))
def test_typology_head_can_be_switched_off(space, arch):
    model = build_encoder(_config(arch, typology_head=False), space)
    out = model(Batch.from_data_list([_case(space, 4, 6)]))
    assert out.typology_logits is None
    assert out.risk_logits.shape == (1, 1)


@pytest.mark.parametrize("arch", sorted(ARMS))
def test_pooling_attention_sums_to_one_within_each_graph(space, arch):
    """Attention is softmaxed within a graph, never across the batch."""
    model = build_encoder(_config(arch), space).eval()
    graphs = [_case(space, 6, 10, seed=0), _case(space, 9, 14, seed=1)]
    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        weights = model(batch, want_attention=True).attention_weights["pooling"]
    for g in range(2):
        mass = weights[batch.batch == g].sum(dim=0)
        assert torch.allclose(mass, torch.ones(TOKENS), atol=1e-5)


def test_registry_rejects_an_unknown_arm(space):
    with pytest.raises(UnknownArmError, match="unknown encoder arm"):
        build_encoder(_config("transformer_xl"), space)


def test_arms_are_within_an_order_of_magnitude_on_capacity(space):
    """A capacity gap is an alternative explanation for a performance gap."""
    counts = {arch: count_parameters(build_encoder(_config(arch), space)) for arch in ARMS}
    assert min(counts.values()) * 10 > max(counts.values()), counts


# ------------------------------------------------------------------ overfit ---


@pytest.mark.parametrize("arch", sorted(ARMS))
def test_every_arm_overfits_twenty_cases(space, arch):
    """The wiring check. An arm that cannot memorise 20 cases has something disconnected.

    Deliberately not a performance claim: it asserts the gradient reaches every part of
    the model, nothing more.
    """
    torch.manual_seed(0)
    graphs = [_case(space, 4 + (i % 7), 6 + (i % 11), label=i % 2, seed=i) for i in range(20)]
    batch = Batch.from_data_list(graphs)
    targets = batch.y.reshape(-1).float()

    model = build_encoder(_config(arch, dropout=0.0), space)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss = torch.tensor(float("inf"))
    for _ in range(220):
        out = model(batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out.risk_logits.reshape(-1), targets
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    assert float(loss) < 0.05, f"{arch} could not overfit 20 cases: loss {float(loss):.4f}"


# ----------------------------------------------------------- edge features ---


@pytest.mark.parametrize("arch", ["gatv2", "gin", "graph_transformer"])
def test_zeroing_edge_features_changes_the_output(space, arch):
    """The brief's check: edge features must actually be used.

    Same weights, same input, edge encoder zeroed. If the output is unchanged the edge
    attributes were never reaching the convolution -- the failure mode the brief calls
    out, since `GATv2Conv` accepts `edge_attr` and most implementations forget to pass it.
    """
    torch.manual_seed(0)
    model = build_encoder(_config(arch), space).eval()
    batch = Batch.from_data_list([_case(space, 8, 16, seed=3)])

    with torch.no_grad():
        with_edges = model(batch).risk_logits.clone()
        model.use_edge_features = False
        without_edges = model(batch).risk_logits.clone()

    assert not torch.allclose(with_edges, without_edges, atol=1e-6)


@pytest.mark.parametrize("arch", ["sage", "gcn", "mlp"])
def test_arms_without_an_edge_channel_are_unaffected(space, arch):
    """SAGE, GCN and the MLP have no edge-attribute path, and that is reported honestly."""
    torch.manual_seed(0)
    model = build_encoder(_config(arch), space).eval()
    batch = Batch.from_data_list([_case(space, 8, 16, seed=3)])

    with torch.no_grad():
        before = model(batch).risk_logits.clone()
        model.use_edge_features = False
        after = model(batch).risk_logits.clone()

    assert torch.allclose(before, after)


def test_the_control_ignores_topology_entirely(space):
    """The MLP must be a genuine no-message-passing control: rewiring changes nothing."""
    torch.manual_seed(0)
    model = build_encoder(_config("mlp"), space).eval()
    case = _case(space, 9, 18, seed=4)
    batch = Batch.from_data_list([case])

    rewired = _case(space, 9, 18, seed=4)
    rewired.edge_index = torch.randint(0, 9, (2, 18), generator=torch.Generator().manual_seed(99))
    with torch.no_grad():
        assert torch.allclose(
            model(batch).risk_logits, model(Batch.from_data_list([rewired])).risk_logits
        )


@pytest.mark.parametrize("arch", ["gatv2", "gin", "sage", "gcn", "graph_transformer"])
def test_message_passing_arms_do_depend_on_topology(space, arch):
    """The mirror of the control test, so the control test is known to test something."""
    torch.manual_seed(0)
    model = build_encoder(_config(arch), space).eval()
    case = _case(space, 9, 18, seed=4)
    rewired = _case(space, 9, 18, seed=4)
    rewired.edge_index = torch.randint(0, 9, (2, 18), generator=torch.Generator().manual_seed(99))

    with torch.no_grad():
        a = model(Batch.from_data_list([case])).risk_logits
        b = model(Batch.from_data_list([rewired])).risk_logits
    assert not torch.allclose(a, b, atol=1e-6)


# -------------------------------------------------------------- determinism ---


@pytest.mark.parametrize("arch", sorted(ARMS))
def test_a_fixed_seed_reproduces_the_forward_pass(space, arch):
    batch = Batch.from_data_list([_case(space, 7, 12, seed=5), _case(space, 3, 4, seed=6)])

    torch.manual_seed(1234)
    first = build_encoder(_config(arch), space).eval()
    torch.manual_seed(1234)
    second = build_encoder(_config(arch), space).eval()

    with torch.no_grad():
        assert torch.equal(first(batch).risk_logits, second(batch).risk_logits)
        assert torch.equal(first(batch).pooled_tokens, second(batch).pooled_tokens)


# ------------------------------------------------------------------- losses ---


def test_focal_at_gamma_zero_is_weighted_bce():
    """The comparison isolates the focusing term, so the two must coincide at gamma 0."""
    logits = torch.randn(64)
    targets = (torch.rand(64) > 0.7).float()
    focal = FocalLoss(gamma=0.0, alpha=3.0)(logits, targets)
    bce = WeightedBCELoss(alpha=3.0)(logits, targets)
    assert torch.allclose(focal, bce, atol=1e-6)


def test_focal_downweights_easy_examples():
    """The whole point: a confident correct prediction should contribute almost nothing."""
    easy = torch.tensor([-8.0])
    hard = torch.tensor([0.1])
    target = torch.tensor([0.0])
    focal = FocalLoss(gamma=2.0)
    assert float(focal(easy, target)) < 0.01 * float(focal(hard, target))


def test_build_binary_loss_refuses_to_guess():
    with pytest.raises(ValueError, match="unknown binary loss"):
        build_binary_loss("hinge", gamma=2.0, alpha=1.0)


def test_inverse_frequency_weights_ignore_the_sentinel():
    labels = torch.tensor([-1, -1, 0, 0, 0, 0, 1])
    weights = inverse_frequency_weights(labels, 3)
    assert weights[1] > weights[0] > 0
    assert float(weights[2]) == 0.0  # absent class, not an infinite weight
