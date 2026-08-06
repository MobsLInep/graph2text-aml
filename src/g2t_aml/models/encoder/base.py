"""The shared contract every encoder arm satisfies, and the parts they all share.

Six architectures are compared in Phase 11, so they are built now behind one interface.
Everything except the message-passing block itself lives here — the edge encoder, the
attention-pooling readout, the two heads — so that a difference between two arms is a
difference in message passing and not an accidental difference in how amounts were
embedded or how the graph was pooled. An ablation whose arms differ in three places at
once measures nothing.

The readout is **attention pooling to ``k`` tokens**, not mean pooling. Two reasons, and
the second is the operative one. Mean pooling a 150-node subgraph to a single 256-vector
is a hard information bottleneck exactly where the structure lives. And Phase 8's fusion
layer projects a *sequence* of graph tokens into the language model's embedding space, so
the pooling head has to produce ``[B, k, d]`` eventually; building it here means Phase 8
consumes this output directly rather than bolting a second pooling stage onto a trained
encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn
from torch_geometric.utils import softmax as scatter_softmax


@dataclass
class EncoderOutput:
    """Everything an encoder arm produces for one batch.

    Attributes:
        node_embeddings: ``[N, d]`` per-node representations, ``N`` summed over the
            batch. Consumed by the attention figures and by Phase 8's node-level fusion
            variant.
        pooled_tokens: ``[B, k, d]`` query-attended graph tokens. The fusion layer's
            input.
        graph_embedding: ``[B, d]`` single-vector case representation, the mean over the
            pooled tokens. Used by the linear probe and by the risk head.
        risk_logits: ``[B, 1]`` binary licit/suspicious logits.
        typology_logits: ``[B, 9]`` auxiliary typology logits, or None on an arm or
            substrate without typology ground truth.
        attention_weights: Interpretability payload — per-layer message-passing
            attention where the arm has any, and always the pooling attention. None when
            not requested, because retaining it costs memory on every training step.
    """

    node_embeddings: Tensor
    pooled_tokens: Tensor
    graph_embedding: Tensor
    risk_logits: Tensor
    typology_logits: Tensor | None = None
    attention_weights: dict[str, Tensor] | None = None


@runtime_checkable
class GraphEncoder(Protocol):
    """The interface every arm implements, so arms are drop-in interchangeable."""

    def forward(self, batch: object) -> EncoderOutput:
        """Encode a batch of case subgraphs.

        Args:
            batch: A PyG ``Batch`` carrying ``x``, ``edge_index``, ``edge_attr``, the
                three categorical edge index tensors and ``batch``.

        Returns:
            The encoder's output for the batch.
        """
        ...


class EdgeEncoder(nn.Module):
    """Projects continuous and categorical edge attributes to a single edge vector.

    Amount, currency, payment format and time-since-previous-transaction all carry real
    AML signal — structuring shows up in amounts, layering in inter-transaction gaps, and
    payment rail separates a wire from a cash deposit — and ``GATv2Conv`` accepts edge
    attributes but most implementations never pass them. This module is what makes
    ``edge_dim`` mean something.

    Currency and format are embedded rather than one-hot encoded: fifteen currencies with
    very unequal frequencies learn a better shared space from an embedding table, and
    index 0 is the OOV slot for a category absent from the training split.
    """

    def __init__(
        self,
        continuous_dim: int,
        n_currencies: int,
        n_formats: int,
        out_dim: int,
        *,
        categorical_dim: int = 8,
        dropout: float = 0.0,
    ) -> None:
        """Build the edge encoder.

        Args:
            continuous_dim: Width of ``edge_attr``.
            n_currencies: Currency embedding table size, OOV slot included.
            n_formats: Payment-format embedding table size, OOV slot included.
            out_dim: Output width, the model's ``edge_dim``.
            categorical_dim: Width of each categorical embedding.
            dropout: Dropout applied to the projected edge vector.
        """
        super().__init__()
        self.currency = nn.Embedding(n_currencies, categorical_dim)
        self.payment_format = nn.Embedding(n_formats, categorical_dim)
        self.project = nn.Sequential(
            nn.Linear(continuous_dim + 3 * categorical_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = out_dim

    def forward(
        self,
        edge_attr: Tensor,
        currency_paid: Tensor,
        currency_received: Tensor,
        payment_format: Tensor,
    ) -> Tensor:
        """Encode one batch of edges.

        Args:
            edge_attr: ``[E, continuous_dim]`` continuous features.
            currency_paid: ``[E]`` currency indices for the paid leg.
            currency_received: ``[E]`` currency indices for the received leg.
            payment_format: ``[E]`` payment-format indices.

        Returns:
            ``[E, out_dim]`` edge representations.
        """
        parts = [
            edge_attr,
            self.currency(currency_paid),
            self.currency(currency_received),
            self.payment_format(payment_format),
        ]
        return self.project(torch.cat(parts, dim=-1))


class AttentionPooling(nn.Module):
    """Pools node embeddings into ``k`` graph tokens with ``k`` learned queries.

    Each query attends over the nodes of its own graph — the scatter softmax is taken
    within a graph, never across the batch — so a token is a weighted read of one case.
    Different queries learn to read different things: in practice one converges onto the
    high-out-degree dispersal nodes and another onto the flagged counterparties, which is
    what makes the attention figures legible.
    """

    def __init__(self, dim: int, n_tokens: int, *, dropout: float = 0.0) -> None:
        """Build the pooling head.

        Args:
            dim: Node embedding width.
            n_tokens: Number of pooled tokens, ``k``.
            dropout: Dropout on the attention weights.
        """
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_tokens, dim) * dim**-0.5)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.n_tokens = n_tokens
        self.dim = dim

    def forward(self, x: Tensor, batch_index: Tensor, n_graphs: int) -> tuple[Tensor, Tensor]:
        """Pool node embeddings into per-graph tokens.

        Args:
            x: ``[N, dim]`` node embeddings.
            batch_index: ``[N]`` graph assignment per node.
            n_graphs: Number of graphs in the batch.

        Returns:
            ``([B, k, dim], [N, k])`` — the pooled tokens and the per-node attention
            weight under each query, the latter being what the interpretability figures
            plot.
        """
        keys = self.key(x)
        values = self.value(x)
        # [N, k]: score of every node under every query, scaled as in dot-product
        # attention so the softmax does not saturate at d = 256.
        scores = (keys @ self.queries.t()) * self.dim**-0.5
        weights = scatter_softmax(scores, batch_index, num_nodes=x.size(0), dim=0)
        weights = self.dropout(weights)

        # One token at a time rather than a [N, k, dim] intermediate: at k = 16 and
        # d = 256 the fused form allocates 157 MB on a full batch, which does not fit
        # alongside the backward graph on a 4 GB card. Sixteen [N, dim] scatters do.
        tokens = [
            x.new_zeros(n_graphs, self.dim).index_add(
                0, batch_index, weights[:, t : t + 1] * values
            )
            for t in range(self.n_tokens)
        ]
        return self.norm(torch.stack(tokens, dim=1)), weights


class PredictionHeads(nn.Module):
    """The binary risk head and the auxiliary typology head on a shared trunk.

    The typology head is close to free — one linear layer on a representation the risk
    head already needs — and it forces the trunk to keep typology-discriminative
    structure rather than collapsing everything onto one suspicious/licit axis. That
    structure is exactly what the narrative generator needs in order to name a typology,
    and the linear probe in ``analysis.py`` measures whether it survived.
    """

    def __init__(self, dim: int, n_typologies: int, *, dropout: float = 0.1) -> None:
        """Build the heads.

        Args:
            dim: Graph embedding width.
            n_typologies: Number of typology classes, or 0 to omit the head.
            dropout: Dropout before each head.
        """
        super().__init__()
        self.risk = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        self.typology = (
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, n_typologies),
            )
            if n_typologies
            else None
        )

    def forward(self, graph_embedding: Tensor) -> tuple[Tensor, Tensor | None]:
        """Score a batch of graph embeddings.

        Args:
            graph_embedding: ``[B, dim]``.

        Returns:
            ``([B, 1], [B, n_typologies] | None)``.
        """
        typology = self.typology(graph_embedding) if self.typology is not None else None
        return self.risk(graph_embedding), typology


class BaseEncoder(nn.Module):
    """Shared scaffolding: input projection, edge encoding, pooling and heads.

    An arm subclasses this and implements :meth:`message_passing`. Everything else — how
    a node's raw features enter the model, how edges are embedded, how the graph is
    pooled, how it is scored — is identical across arms by construction.
    """

    def __init__(
        self,
        *,
        node_dim: int,
        edge_continuous_dim: int,
        n_currencies: int,
        n_formats: int,
        hidden_dim: int = 256,
        edge_dim: int = 16,
        n_typologies: int = 9,
        n_pooled_tokens: int = 16,
        dropout: float = 0.2,
        use_edge_features: bool = True,
    ) -> None:
        """Build the shared scaffolding.

        Args:
            node_dim: Width of ``Data.x``, positional encodings included.
            edge_continuous_dim: Width of ``Data.edge_attr``.
            n_currencies: Currency embedding table size.
            n_formats: Payment-format embedding table size.
            hidden_dim: Model width, ``d``.
            edge_dim: Width the edge encoder projects to.
            n_typologies: Auxiliary head class count, 0 to omit.
            n_pooled_tokens: ``k``, the number of pooled graph tokens.
            dropout: Dropout rate used throughout.
            use_edge_features: When False the edge encoder is still built but its output
                is zeroed. This is the ablation switch: it keeps parameter count and
                initialisation identical so the comparison isolates the edge information
                rather than model capacity.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.n_pooled_tokens = n_pooled_tokens
        self.use_edge_features = use_edge_features
        self.dropout_p = dropout

        self.input_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.edge_encoder = EdgeEncoder(
            edge_continuous_dim,
            n_currencies,
            n_formats,
            edge_dim,
            dropout=dropout,
        )
        self.pooling = AttentionPooling(hidden_dim, n_pooled_tokens, dropout=dropout)
        self.heads = PredictionHeads(hidden_dim, n_typologies, dropout=dropout)

    def message_passing(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, *, want_attention: bool
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Run the arm's own message passing.

        Args:
            x: ``[N, hidden_dim]`` projected node features.
            edge_index: ``[2, E]``.
            edge_attr: ``[E, edge_dim]`` encoded edges, zeroed under the ablation.
            want_attention: Whether to retain attention weights.

        Returns:
            ``([N, hidden_dim], attention)`` — updated node embeddings and any retained
            attention tensors, keyed by layer.

        Raises:
            NotImplementedError: Always; an arm must implement this.
        """
        raise NotImplementedError

    def forward(self, batch: object, *, want_attention: bool = False) -> EncoderOutput:
        """Encode a batch of case subgraphs.

        Args:
            batch: A PyG ``Batch``.
            want_attention: Retain attention weights for the interpretability figures.
                Off during training, where retaining them costs memory on every step.

        Returns:
            The encoder's output.
        """
        x = self.input_projection(batch.x)
        edge_attr = self.edge_encoder(
            batch.edge_attr,
            batch.edge_currency_paid,
            batch.edge_currency_received,
            batch.edge_format,
        )
        if not self.use_edge_features:
            edge_attr = torch.zeros_like(edge_attr)

        node_embeddings, attention = self.message_passing(
            x, batch.edge_index, edge_attr, want_attention=want_attention
        )

        batch_index = batch.batch
        # From the Batch, not from `batch_index.max()`: a graph whose nodes all fall at
        # the end of the batch would still be counted, but a trailing empty graph would
        # not, and an off-by-one here silently drops a case from the metrics.
        n_graphs = int(batch.num_graphs)
        pooled, pool_weights = self.pooling(node_embeddings, batch_index, n_graphs)
        graph_embedding = pooled.mean(dim=1)
        risk_logits, typology_logits = self.heads(graph_embedding)

        weights: dict[str, Tensor] | None = None
        if want_attention:
            weights = dict(attention)
            weights["pooling"] = pool_weights

        return EncoderOutput(
            node_embeddings=node_embeddings,
            pooled_tokens=pooled,
            graph_embedding=graph_embedding,
            risk_logits=risk_logits,
            typology_logits=typology_logits,
            attention_weights=weights,
        )
