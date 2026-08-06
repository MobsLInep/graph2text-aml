"""The six encoder arms.

All six subclass :class:`~g2t_aml.models.encoder.base.BaseEncoder` and differ **only** in
:meth:`message_passing`. Input projection, edge encoding, attention pooling and the two
heads are shared, so a gap between two arms is attributable to message passing.

``gatv2``
    The primary. Dynamic attention, edge-conditioned.
``gin``
    The most WL-expressive of the six. If structure is the signal GIN should be
    competitive, and if it wins the primary arm should change.
``sage``
    The standard scalable comparator: mean aggregation, no attention.
``gcn``
    Topology-only sanity check: symmetric normalised averaging, no edge features,
    no attention.
``graph_transformer``
    Global attention through a virtual node, so every node reaches every other in
    one hop.
``mlp``
    **No message passing at all.** The control.

**GATv2 rather than GAT.** The original GAT computes ``a^T [Wh_i || Wh_j]`` and then
applies the nonlinearity, which makes the ranking of neighbours *static*: there is a
global ordering of keys that every query agrees on. GATv2 applies the nonlinearity before
the attention vector, which makes the ranking query-dependent. Brody et al. (2022) show
the first is strictly less expressive, and it costs nothing to use the second. There is no
argument for the original beyond inertia.

**The MLP is the control that matters.** It runs the identical trunk with the message
passing removed and a per-node feed-forward stack in its place, then pools with the same
attention head — so it is a DeepSets model over node features, not a crippled baseline.
It sees the same case-local degree, amount and burst features every other arm sees. If it
matches the GAT, topology carries no signal beyond what node-local summary statistics
already encode, and the premise of the project needs revisiting. It is built to be hard
to beat on purpose.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv, GCNConv, GINEConv, SAGEConv, TransformerConv

from g2t_aml.models.encoder.base import BaseEncoder, EncoderOutput


class _ResidualStack(BaseEncoder):
    """Shared plumbing for the arms that are a stack of residual, normalised layers.

    Residual connections and layer norm are applied identically across arms, so depth
    behaves the same way in all of them and a difference is not just one arm being easier
    to optimise at three layers.
    """

    def __init__(self, *, num_layers: int = 3, residual: bool = True, **kwargs: object) -> None:
        """Build the stack scaffolding.

        Args:
            num_layers: Number of message-passing layers.
            residual: Whether to add the input of each layer to its output.
            **kwargs: Forwarded to :class:`BaseEncoder`.
        """
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.num_layers = num_layers
        self.residual = residual
        self.norms = nn.ModuleList(nn.LayerNorm(self.hidden_dim) for _ in range(num_layers))
        self.activation = nn.GELU()
        self.layer_dropout = nn.Dropout(self.dropout_p)

    def _combine(self, previous: Tensor, updated: Tensor, layer: int) -> Tensor:
        """Apply residual, norm, activation and dropout after one layer."""
        out = updated + previous if self.residual else updated
        return self.layer_dropout(self.activation(self.norms[layer](out)))


class GATv2Encoder(_ResidualStack):
    """The primary arm: GATv2 with edge-conditioned dynamic attention."""

    def __init__(
        self,
        *,
        heads: int = 8,
        concat_heads: bool = True,
        num_layers: int = 3,
        residual: bool = True,
        **kwargs: object,
    ) -> None:
        """Build the GATv2 stack.

        Args:
            heads: Attention heads per layer.
            concat_heads: Concatenate heads and project back to ``hidden_dim`` when
                True, average them when False. Concatenation keeps per-head capacity;
                ``hidden_dim`` must divide by ``heads`` for it.
            num_layers: Number of layers.
            residual: Residual connections around each layer.
            **kwargs: Forwarded to :class:`BaseEncoder`.

        Raises:
            ValueError: If ``concat_heads`` is set and ``hidden_dim`` is not divisible by
                ``heads``, which would silently change the model width.
        """
        super().__init__(num_layers=num_layers, residual=residual, **kwargs)
        if concat_heads and self.hidden_dim % heads:
            raise ValueError(
                f"hidden_dim {self.hidden_dim} is not divisible by heads {heads}; with "
                "concat_heads the per-head width must be an integer"
            )
        per_head = self.hidden_dim // heads if concat_heads else self.hidden_dim
        self.convs = nn.ModuleList(
            GATv2Conv(
                self.hidden_dim,
                per_head,
                heads=heads,
                concat=concat_heads,
                edge_dim=self.edge_dim,
                add_self_loops=True,
            )
            for _ in range(num_layers)
        )

    def message_passing(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, *, want_attention: bool
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Run edge-conditioned GATv2 message passing.

        Args:
            x: ``[N, hidden_dim]`` projected node features.
            edge_index: ``[2, E]``.
            edge_attr: ``[E, edge_dim]``.
            want_attention: Retain per-layer attention coefficients.

        Returns:
            ``([N, hidden_dim], attention)``. Attention is keyed ``layer_{i}_alpha`` and
            ``layer_{i}_index``; the index is returned alongside because ``add_self_loops``
            appends edges that are not in the input ``edge_index``.
        """
        attention: dict[str, Tensor] = {}
        for i, conv in enumerate(self.convs):
            if want_attention:
                updated, (index, alpha) = conv(
                    x, edge_index, edge_attr, return_attention_weights=True
                )
                attention[f"layer_{i}_index"] = index.detach()
                attention[f"layer_{i}_alpha"] = alpha.detach()
            else:
                updated = conv(x, edge_index, edge_attr)
            x = self._combine(x, updated, i)
        return x, attention


class GINEncoder(_ResidualStack):
    """GIN with edge features (GINE): the most WL-expressive arm in the comparison."""

    def __init__(self, *, num_layers: int = 3, residual: bool = True, **kwargs: object) -> None:
        """Build the GINE stack.

        Args:
            num_layers: Number of layers.
            residual: Residual connections around each layer.
            **kwargs: Forwarded to :class:`BaseEncoder`.
        """
        super().__init__(num_layers=num_layers, residual=residual, **kwargs)
        self.convs = nn.ModuleList(
            GINEConv(
                nn.Sequential(
                    nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                ),
                train_eps=True,
                edge_dim=self.edge_dim,
            )
            for _ in range(num_layers)
        )

    def message_passing(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, *, want_attention: bool
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Run GINE message passing.

        Args:
            x: ``[N, hidden_dim]``.
            edge_index: ``[2, E]``.
            edge_attr: ``[E, edge_dim]``.
            want_attention: Ignored; GIN has no attention to retain. The pooling
                attention is still produced by the shared readout.

        Returns:
            ``([N, hidden_dim], {})``.
        """
        del want_attention
        for i, conv in enumerate(self.convs):
            x = self._combine(x, conv(x, edge_index, edge_attr), i)
        return x, {}


class GraphSAGEEncoder(_ResidualStack):
    """GraphSAGE: the standard scalable comparator, mean aggregation, no edge features.

    ``SAGEConv`` has no edge-attribute channel. Rather than fake one, the arm is left as
    the published architecture and the edge information it cannot see is reported as part
    of what the comparison measures.
    """

    def __init__(self, *, num_layers: int = 3, residual: bool = True, **kwargs: object) -> None:
        """Build the SAGE stack.

        Args:
            num_layers: Number of layers.
            residual: Residual connections around each layer.
            **kwargs: Forwarded to :class:`BaseEncoder`.
        """
        super().__init__(num_layers=num_layers, residual=residual, **kwargs)
        self.convs = nn.ModuleList(
            SAGEConv(self.hidden_dim, self.hidden_dim, aggr="mean") for _ in range(num_layers)
        )

    def message_passing(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, *, want_attention: bool
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Run GraphSAGE message passing.

        Args:
            x: ``[N, hidden_dim]``.
            edge_index: ``[2, E]``.
            edge_attr: Ignored; see the class docstring.
            want_attention: Ignored; SAGE has no attention.

        Returns:
            ``([N, hidden_dim], {})``.
        """
        del edge_attr, want_attention
        for i, conv in enumerate(self.convs):
            x = self._combine(x, conv(x, edge_index), i)
        return x, {}


class GCNEncoder(_ResidualStack):
    """GCN: the topology-only sanity check.

    Symmetric normalised averaging with no edge features and no attention. It is the
    weakest aggregator in the set on purpose — it brackets the MLP control from the other
    side, since the MLP has features and no topology and the GCN has topology and the
    least expressive use of it.
    """

    def __init__(self, *, num_layers: int = 3, residual: bool = True, **kwargs: object) -> None:
        """Build the GCN stack.

        Args:
            num_layers: Number of layers.
            residual: Residual connections around each layer.
            **kwargs: Forwarded to :class:`BaseEncoder`.
        """
        super().__init__(num_layers=num_layers, residual=residual, **kwargs)
        self.convs = nn.ModuleList(
            GCNConv(self.hidden_dim, self.hidden_dim, add_self_loops=True)
            for _ in range(num_layers)
        )

    def message_passing(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, *, want_attention: bool
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Run GCN message passing.

        Args:
            x: ``[N, hidden_dim]``.
            edge_index: ``[2, E]``.
            edge_attr: Ignored; GCN has no edge-attribute channel.
            want_attention: Ignored; GCN has no attention.

        Returns:
            ``([N, hidden_dim], {})``.
        """
        del edge_attr, want_attention
        for i, conv in enumerate(self.convs):
            x = self._combine(x, conv(x, edge_index), i)
        return x, {}


class GraphTransformerEncoder(_ResidualStack):
    """Graph transformer with a virtual node, so every node is one hop from every other.

    ``TransformerConv`` is local attention over the real edges; the virtual node adds a
    node connected bidirectionally to every node in its own graph, which gives the model a
    global read without the quadratic cost of dense attention. That matters here because a
    laundering path can run the length of the case: a three-layer local model cannot
    connect the two ends of a six-hop chain, and ``stack`` is exactly that shape.
    """

    def __init__(
        self,
        *,
        heads: int = 8,
        num_layers: int = 3,
        residual: bool = True,
        **kwargs: object,
    ) -> None:
        """Build the transformer stack.

        Args:
            heads: Attention heads per layer.
            num_layers: Number of layers.
            residual: Residual connections around each layer.
            **kwargs: Forwarded to :class:`BaseEncoder`.

        Raises:
            ValueError: If ``hidden_dim`` is not divisible by ``heads``.
        """
        super().__init__(num_layers=num_layers, residual=residual, **kwargs)
        if self.hidden_dim % heads:
            raise ValueError(f"hidden_dim {self.hidden_dim} is not divisible by heads {heads}")
        self.virtual_node = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.convs = nn.ModuleList(
            TransformerConv(
                self.hidden_dim,
                self.hidden_dim // heads,
                heads=heads,
                concat=True,
                edge_dim=self.edge_dim,
            )
            for _ in range(num_layers)
        )

    def forward(self, batch: object, *, want_attention: bool = False) -> EncoderOutput:
        """Encode a batch, adding one virtual node per graph before message passing.

        The virtual nodes are appended after the real ones, take part in every layer, and
        are dropped before pooling — so the pooled tokens read only real accounts and an
        attention figure never shows a weight on a node that does not exist.

        Args:
            batch: A PyG ``Batch``.
            want_attention: Retain attention weights.

        Returns:
            The encoder's :class:`~g2t_aml.models.encoder.base.EncoderOutput`.
        """
        self._n_real_nodes = int(batch.x.size(0))
        self._batch_index = batch.batch
        self._n_graphs = int(batch.num_graphs)
        return super().forward(batch, want_attention=want_attention)

    def message_passing(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, *, want_attention: bool
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Run transformer message passing over the graph plus its virtual node.

        Args:
            x: ``[N, hidden_dim]``.
            edge_index: ``[2, E]``.
            edge_attr: ``[E, edge_dim]``.
            want_attention: Retain per-layer attention coefficients.

        Returns:
            ``([N, hidden_dim], attention)`` over the **real** nodes only.
        """
        n_real, n_graphs = self._n_real_nodes, self._n_graphs
        device = x.device
        virtual = self.virtual_node.expand(n_graphs, -1)
        x_all = torch.cat([x, virtual], dim=0)

        node_ids = torch.arange(n_real, device=device)
        virtual_ids = n_real + self._batch_index
        to_virtual = torch.stack([node_ids, virtual_ids])
        from_virtual = torch.stack([virtual_ids, node_ids])
        edge_index_all = torch.cat([edge_index, to_virtual, from_virtual], dim=1)
        # Virtual edges get a zero edge vector: they carry topology, not a transaction,
        # and giving them a learned attribute would let the model route real amount
        # information through a channel that has no transaction behind it.
        pad = edge_attr.new_zeros(2 * n_real, edge_attr.size(1))
        edge_attr_all = torch.cat([edge_attr, pad], dim=0)

        attention: dict[str, Tensor] = {}
        for i, conv in enumerate(self.convs):
            if want_attention:
                updated, (index, alpha) = conv(
                    x_all, edge_index_all, edge_attr_all, return_attention_weights=True
                )
                attention[f"layer_{i}_index"] = index.detach()
                attention[f"layer_{i}_alpha"] = alpha.detach()
            else:
                updated = conv(x_all, edge_index_all, edge_attr_all)
            out = updated + x_all if self.residual else updated
            x_all = self.layer_dropout(self.activation(self.norms[i](out)))
        return x_all[:n_real], attention


class MLPEncoder(BaseEncoder):
    """The control: **no message passing**.

    A per-node feed-forward stack, then the same attention pooling every other arm uses.
    That makes it a DeepSets model over case-local node features rather than a hobbled
    GNN: it gets the identical feature engineering, the identical readout, the identical
    heads, the identical tuning budget and the identical seeds.

    Edge features are genuinely unavailable to it, because consuming an edge feature
    without message passing is not possible — an edge belongs to two nodes and reading it
    into either one is a one-hop aggregation. That is the honest statement of what "no
    message passing" costs, and it is why the node features carry case-local degree,
    amount and burst summaries: the control must see everything a node can know about
    itself.
    """

    def __init__(self, *, num_layers: int = 3, dropout: float = 0.2, **kwargs: object) -> None:
        """Build the per-node stack.

        Args:
            num_layers: Number of feed-forward blocks, matched to the other arms' depth.
            dropout: Dropout inside each block.
            **kwargs: Forwarded to :class:`BaseEncoder`.
        """
        super().__init__(dropout=dropout, **kwargs)  # type: ignore[arg-type]
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            )
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(self.hidden_dim) for _ in range(num_layers))

    def message_passing(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, *, want_attention: bool
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Transform each node independently. No information crosses an edge.

        Args:
            x: ``[N, hidden_dim]``.
            edge_index: Ignored. This is the point of the arm.
            edge_attr: Ignored.
            want_attention: Ignored.

        Returns:
            ``([N, hidden_dim], {})``.
        """
        del edge_index, edge_attr, want_attention
        for block, norm in zip(self.blocks, self.norms, strict=True):
            x = norm(block(x) + x)
        return x, {}
