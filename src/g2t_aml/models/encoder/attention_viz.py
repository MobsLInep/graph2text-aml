"""Extracting and drawing the encoder's attention over a case subgraph.

Two purposes, and the second is the one that could change a conclusion.

The figures go in the paper: an investigator-facing system that produces a narrative
should be able to show which accounts drove the score, and the pooling attention is the
natural place to read that from.

The cross-check is the real work. AMLworld carries complete ground truth, so for any
suspicious case we know exactly which transactions are on the laundering path. That makes
it possible to ask whether the encoder's attention lands on those accounts or somewhere
else, and to answer with a number rather than by looking at a few pictures.
:func:`path_attention_alignment` computes it. **If the encoder scores well while attending
away from the laundering path, that is worth knowing and worth reporting** — it would mean
the score is driven by some case-level correlate rather than by the scheme, and the
attributions written into ``model_signal`` would be misleading to an investigator even
though the risk score is accurate.

The alignment measurement reads ``is_laundering`` from the case's edge table. That is
legitimate here and nowhere else in this package: this is post-hoc analysis of a trained
model, not a feature. The tensors the model saw never contained it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from g2t_aml.utils.io import write_json

if TYPE_CHECKING:  # pragma: no cover
    from torch_geometric.data import Data

    from g2t_aml.models.encoder.base import BaseEncoder


@dataclass
class CaseAttention:
    """The encoder's attention over one case.

    Attributes:
        case_id: The case.
        node_ids: Account identifiers, in tensor order.
        pooling_attention: ``[n_nodes]`` attention mass each account received, summed
            over the pooled tokens and normalised to sum to one. This is what
            ``model_signal.top_contributing_nodes`` is built from.
        per_token_attention: ``[n_nodes, k]`` the same before summing, so a figure can
            show that different queries read different parts of the case.
        risk_score: The model's risk probability for the case.
        label: Ground-truth binary label, or None.
        typology: Ground-truth typology, or None.
    """

    case_id: str
    node_ids: list[str]
    pooling_attention: np.ndarray
    per_token_attention: np.ndarray
    risk_score: float
    label: int | None = None
    typology: str | None = None

    def top_nodes(self, n: int = 5) -> list[tuple[str, float]]:
        """Return the highest-attention accounts.

        Args:
            n: How many to return.

        Returns:
            ``(node_id, attribution)`` pairs, highest first.
        """
        order = np.argsort(-self.pooling_attention)[:n]
        return [(self.node_ids[i], float(self.pooling_attention[i])) for i in order]


@dataclass
class AlignmentReport:
    """Whether attention lands on the laundering path, across a population.

    Attributes:
        n_cases: Suspicious cases measured.
        mean_path_attention: Mean share of attention mass falling on accounts that touch
            a flagged transaction.
        mean_path_share: Mean share of *accounts* that are on the path — the baseline a
            uniformly-attending model would achieve, and therefore what
            ``mean_path_attention`` must be compared against.
        lift: ``mean_path_attention / mean_path_share``. Above 1 means the encoder
            concentrates on the path; at 1 it is no better than uniform; below 1 it is
            actively attending elsewhere.
        top1_hit_rate: Fraction of cases whose single highest-attention account is on the
            laundering path.
        per_typology_lift: The same lift, broken down by typology.
    """

    n_cases: int
    mean_path_attention: float
    mean_path_share: float
    lift: float
    top1_hit_rate: float
    per_typology_lift: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable form.

        Returns:
            Every field.
        """
        return asdict(self)


@torch.no_grad()
def extract_attention(
    model: BaseEncoder,
    graphs: list[Data],
    device: torch.device,
    *,
    batch_size: int = 64,
) -> list[CaseAttention]:
    """Run a model with attention retained and split the result back into cases.

    Args:
        model: A trained arm.
        graphs: The cases to explain.
        device: Device to run on.
        batch_size: Batch size. Smaller than evaluation's, because retaining attention
            for every layer costs memory.

    Returns:
        One :class:`CaseAttention` per input case, in input order.
    """
    from torch_geometric.loader import DataLoader

    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    out: list[CaseAttention] = []

    for cpu_batch in loader:
        batch = cpu_batch.to(device)
        result = model(batch, want_attention=True)
        pooling = result.attention_weights["pooling"].cpu().numpy()  # [N, k]
        scores = torch.sigmoid(result.risk_logits.reshape(-1)).cpu().numpy()
        assignment = batch.batch.cpu().numpy()
        labels = batch.y.reshape(-1).cpu().numpy()

        for g in range(int(batch.num_graphs)):
            nodes = np.flatnonzero(assignment == g)
            per_token = pooling[nodes]
            summed = per_token.sum(axis=1)
            total = summed.sum()
            out.append(
                CaseAttention(
                    case_id=batch.case_id[g],
                    node_ids=list(batch.node_ids[g]),
                    pooling_attention=summed / total if total > 0 else summed,
                    per_token_attention=per_token,
                    risk_score=float(scores[g]),
                    label=int(labels[g]) if labels[g] >= 0 else None,
                )
            )
    return out


def laundering_path_nodes(edges: Any) -> set[str]:
    """Return the accounts that touch a flagged transaction.

    Reads ``is_laundering`` from the case's own edge table. This is post-hoc evaluation
    of a trained model — see the module docstring — and the column never enters a feature
    tensor.

    Args:
        edges: The case's Polars edge frame.

    Returns:
        Account identifiers on the laundering path, empty when the substrate carries no
        transaction-level labels.
    """
    if "is_laundering" not in edges.columns:
        return set()
    flagged = edges.filter(edges["is_laundering"].fill_null(False))
    return set(flagged["src"].to_list()) | set(flagged["dst"].to_list())


def path_attention_alignment(
    attentions: list[CaseAttention],
    path_nodes: dict[str, set[str]],
    typologies: dict[str, str] | None = None,
) -> AlignmentReport:
    """Measure whether attention concentrates on the laundering path.

    Args:
        attentions: Per-case attention, from :func:`extract_attention`.
        path_nodes: Case id to the accounts on its laundering path. Cases absent from
            this mapping, or mapping to an empty set, are skipped — a licit case has no
            path and including it would put a zero in an average of a quantity that is
            undefined for it.
        typologies: Case id to typology, for the per-typology breakdown.

    Returns:
        The alignment report, all-NaN when no case qualifies.
    """
    shares: list[float] = []
    baselines: list[float] = []
    hits: list[float] = []
    by_typology: dict[str, list[tuple[float, float]]] = {}

    for attention in attentions:
        path = path_nodes.get(attention.case_id) or set()
        if not path:
            continue
        on_path = np.asarray([nid in path for nid in attention.node_ids], dtype=bool)
        if not on_path.any() or on_path.all():
            # All-on-path gives a lift of exactly 1 by construction and tells us nothing
            # about where the model looked, so it is excluded rather than diluting the
            # average toward 1.
            continue
        mass = float(attention.pooling_attention[on_path].sum())
        baseline = float(on_path.mean())
        shares.append(mass)
        baselines.append(baseline)
        hits.append(1.0 if on_path[int(np.argmax(attention.pooling_attention))] else 0.0)
        if typologies:
            name = typologies.get(attention.case_id, "unclassified")
            by_typology.setdefault(name, []).append((mass, baseline))

    if not shares:
        return AlignmentReport(
            n_cases=0,
            mean_path_attention=float("nan"),
            mean_path_share=float("nan"),
            lift=float("nan"),
            top1_hit_rate=float("nan"),
        )

    mean_mass, mean_baseline = float(np.mean(shares)), float(np.mean(baselines))
    return AlignmentReport(
        n_cases=len(shares),
        mean_path_attention=mean_mass,
        mean_path_share=mean_baseline,
        lift=mean_mass / mean_baseline if mean_baseline else float("nan"),
        top1_hit_rate=float(np.mean(hits)),
        per_typology_lift={
            name: float(np.mean([m for m, _ in pairs]) / np.mean([b for _, b in pairs]))
            for name, pairs in sorted(by_typology.items())
            if np.mean([b for _, b in pairs]) > 0
        },
    )


def draw_case(
    attention: CaseAttention,
    edges: Any,
    path: str | Path,
    *,
    path_nodes: set[str] | None = None,
    title: str | None = None,
) -> Path | None:
    """Draw one case subgraph with node size and colour set by attention.

    Accounts on the laundering path are outlined, so the figure shows at a glance whether
    the encoder attended to them — which is the question the figure exists to answer.

    Args:
        attention: The case's attention.
        edges: The case's Polars edge frame, for the arrows.
        path: Destination image path.
        path_nodes: Accounts on the laundering path, outlined when given.
        title: Figure title. Defaults to the case id and risk score.

    Returns:
        The written path, or None when matplotlib or networkx is unavailable — a figure
        is never allowed to fail a run.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:  # pragma: no cover - optional visualisation dependency
        return None

    graph = nx.DiGraph()
    for i, node_id in enumerate(attention.node_ids):
        graph.add_node(node_id, weight=float(attention.pooling_attention[i]))
    for src, dst in zip(edges["src"].to_list(), edges["dst"].to_list(), strict=True):
        if src in graph and dst in graph:
            graph.add_edge(src, dst)

    layout = nx.spring_layout(graph, seed=0)
    weights = np.asarray([graph.nodes[n]["weight"] for n in graph])
    figure, axis = plt.subplots(figsize=(7, 6))
    nx.draw_networkx_edges(graph, layout, ax=axis, alpha=0.35, arrowsize=8, width=0.8)
    drawn = nx.draw_networkx_nodes(
        graph,
        layout,
        ax=axis,
        node_size=60 + 900 * weights,
        node_color=weights,
        cmap="viridis",
        edgecolors=["crimson" if (path_nodes and n in path_nodes) else "none" for n in graph],
        linewidths=1.6,
    )
    figure.colorbar(drawn, ax=axis, label="pooling attention", shrink=0.75)
    axis.set_axis_off()
    axis.set_title(
        title
        or f"{attention.case_id}  risk={attention.risk_score:.3f}"
        + (f"  typology={attention.typology}" if attention.typology else ""),
        fontsize=9,
    )
    figure.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return out


def save_alignment(report: AlignmentReport, path: str | Path) -> Path:
    """Write an alignment report atomically.

    Args:
        report: The report.
        path: Destination file.

    Returns:
        The path written.
    """
    return write_json(path, report.to_dict())
