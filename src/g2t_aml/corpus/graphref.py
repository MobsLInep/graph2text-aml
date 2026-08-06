"""Building and resolving ``graph_ref``: the pointer from a record to its subgraph.

A training record without a resolvable graph is untrainable, and — worse — a record whose
graph reference resolves to the *wrong* subgraph is trainable and silently wrong. The
encoder would be shown one graph while the narrative described another, and nothing
downstream would notice: the loss would fall, the faithfulness metric reads the embedded
facts rather than the graph, and the failure would surface only as a model that never
learned to condition on structure.

So resolution here is not a file-existence check. It reads the case store's membership
tables and compares the referenced subgraph's node and edge counts against what
``facts.structure`` says they are. Those two numbers come from completely different places
— the Phase 2 membership Parquet and the Phase 3 fact extractor — so agreement between
them is real evidence that the reference points where it claims to.

The reference format is ``<store path>#<case id>``, with the path repository-relative so a
corpus is portable between checkouts. It names the *case store* rather than a per-case
file because Phase 2 writes membership as two columnar tables rather than 30,000 files,
and inventing a file layout Phase 4 would have to materialise is a cost with no benefit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

__all__ = ["GraphRefError", "GraphRefResolver", "build_graph_ref", "parse_graph_ref"]


class GraphRefError(ValueError):
    """Raised when a graph reference is malformed, unresolvable or points elsewhere."""


def build_graph_ref(case_store: Path, case_id: str, repo_root: Path) -> str:
    """Build the reference for one case.

    Args:
        case_store: Directory holding ``case_nodes.parquet`` and ``case_edges.parquet``.
        case_id: The case.
        repo_root: Repository root, which the stored path is relative to.

    Returns:
        The reference, ``<processed dir>/<dataset>/cases#<case id>`` relative to the
        repository root. The path comes from ``cfg.paths.*`` by way of ``case_store``;
        nothing here hardcodes a root.

    Raises:
        GraphRefError: If the store is not inside the repository, which would make the
            reference unresolvable in another checkout.
    """
    try:
        relative = case_store.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise GraphRefError(
            f"case store {case_store} is outside the repository at {repo_root}; a "
            "graph_ref must be repository-relative to survive a different checkout"
        ) from exc
    return f"{relative.as_posix()}#{case_id}"


def parse_graph_ref(ref: str) -> tuple[str, str]:
    """Split a reference into its store path and case id.

    Args:
        ref: The reference.

    Returns:
        ``(store path, case id)``.

    Raises:
        GraphRefError: If the reference has no ``#`` separator or either half is empty.
    """
    store, sep, case_id = ref.partition("#")
    if not sep or not store or not case_id:
        raise GraphRefError(f"graph_ref {ref!r} is malformed; expected '<store path>#<case id>'")
    return store, case_id


@dataclass
class GraphRefResolver:
    """Resolves graph references against the case stores on disk.

    Membership counts are read once per store and cached, because the alternative is a
    Parquet scan per record and there are fifteen thousand of them.

    Attributes:
        repo_root: Repository root that references are relative to.
    """

    repo_root: Path
    _cache: dict[str, dict[str, tuple[int, int]]] = field(default_factory=dict, repr=False)

    def _counts(self, store: str) -> dict[str, tuple[int, int]]:
        """Return node and edge counts per case for one store, reading it at most once.

        Args:
            store: Repository-relative store path.

        Returns:
            Case id to ``(n_nodes, n_edges)``.

        Raises:
            GraphRefError: If the store or either membership table is missing.
        """
        if store in self._cache:
            return self._cache[store]
        directory = self.repo_root / store
        nodes_path = directory / "case_nodes.parquet"
        edges_path = directory / "case_edges.parquet"
        if not nodes_path.is_file() or not edges_path.is_file():
            raise GraphRefError(
                f"case store {directory} does not hold case_nodes.parquet and "
                "case_edges.parquet; run `make cases` for this substrate"
            )
        nodes = (
            pl.read_parquet(nodes_path, columns=["case_id"])
            .group_by("case_id")
            .len()
            .rename({"len": "n_nodes"})
        )
        edges = (
            pl.read_parquet(edges_path, columns=["case_id"])
            .group_by("case_id")
            .len()
            .rename({"len": "n_edges"})
        )
        joined = nodes.join(edges, on="case_id", how="full", coalesce=True).fill_null(0)
        counts = {
            row["case_id"]: (int(row["n_nodes"]), int(row["n_edges"]))
            for row in joined.iter_rows(named=True)
        }
        self._cache[store] = counts
        return counts

    def resolve(self, ref: str) -> tuple[int, int]:
        """Resolve a reference to the referenced subgraph's node and edge counts.

        Args:
            ref: The reference.

        Returns:
            ``(n_nodes, n_edges)``.

        Raises:
            GraphRefError: If the reference is malformed, the store is missing, or the
                case is absent from it.
        """
        store, case_id = parse_graph_ref(ref)
        counts = self._counts(store)
        if case_id not in counts:
            raise GraphRefError(f"case {case_id!r} is not present in case store {store!r}")
        return counts[case_id]

    def check(self, ref: str, n_nodes: int, n_edges: int) -> None:
        """Resolve a reference and assert it points at the graph the facts describe.

        Args:
            ref: The reference.
            n_nodes: ``facts.structure.n_nodes``.
            n_edges: ``facts.structure.n_edges``.

        Raises:
            GraphRefError: If the reference does not resolve, or resolves to a subgraph
                of a different size than the fact record reports. The second is the case
                worth having this function for: it is the only signal that a record's
                graph and its narrative have come apart.
        """
        actual_nodes, actual_edges = self.resolve(ref)
        if (actual_nodes, actual_edges) != (n_nodes, n_edges):
            raise GraphRefError(
                f"graph_ref {ref!r} resolves to a subgraph of {actual_nodes} nodes and "
                f"{actual_edges} edges, but the fact record describes one of {n_nodes} "
                f"and {n_edges}. The record's graph and its narrative describe different "
                "things; do not train on it."
            )
