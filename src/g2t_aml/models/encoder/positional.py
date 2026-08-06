"""Positional encodings for case subgraphs.

Message-passing GNNs are provably weak at the two things this project's typologies are
made of: counting (fan-out degree, transfer chains) and cycle detection. A 1-WL-bounded
network cannot distinguish a 6-cycle from two 3-cycles, and ``cycle`` is one of the eight
AMLworld typologies. Positional encodings do not remove that bound, but they inject
structural coordinates the message passing can then compare, which recovers a good deal of
the discrimination in practice.

Two encodings, both computed on the **undirected, simple, self-loop-free** projection of
the case:

- **Laplacian eigenvector PE** — the ``k`` non-trivial eigenvectors of the symmetric
  normalised Laplacian, smallest eigenvalue first. These are the graph's Fourier basis and
  separate communities and elongated chains. They carry a sign ambiguity — ``-v`` is as
  valid an eigenvector as ``v`` — so the training loader flips signs at random and the
  network is forced to learn a sign-invariant function rather than memorising an
  arbitrary convention (Dwivedi et al., 2022).
- **Random-walk PE** — the return probability ``diag((D^-1 A)^k)`` for ``k = 1..K``. It is
  sign-unambiguous, and its ``k``-th entry is the probability of returning to a node in
  ``k`` steps, which is a direct, local measurement of cycle structure at every length up
  to ``K``.

Both are deterministic functions of topology alone. Neither reads an amount, a timestamp
or a label, so neither can carry a label proxy into the feature tensor.
"""

from __future__ import annotations

import numpy as np

#: Below this many nodes a Laplacian eigenbasis has fewer components than requested and
#: the remainder is zero-padded. 41.5% of cases have fewer than five nodes (Phase 2), so
#: this is the common path rather than an edge case.
_MIN_EIGEN_NODES = 2


def undirected_adjacency(n_nodes: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Build the dense undirected simple adjacency of a case.

    Self-loops are dropped and parallel edges collapse to a single 1. This matches the
    fact layer's convention (``caseview``: adjacency ignores self-loops), so a motif the
    fact record reports and a motif the encoder can see are computed over the same graph.

    Args:
        n_nodes: Number of nodes.
        src: Source node indices, one per edge.
        dst: Destination node indices, one per edge.

    Returns:
        A symmetric ``(n_nodes, n_nodes)`` float64 array of 0/1, zero on the diagonal.
    """
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    if src.size:
        keep = src != dst
        a, b = src[keep], dst[keep]
        adjacency[a, b] = 1.0
        adjacency[b, a] = 1.0
    return adjacency


def laplacian_pe(adjacency: np.ndarray, k: int) -> np.ndarray:
    """Return the first ``k`` non-trivial Laplacian eigenvectors.

    The operator is the symmetric normalised Laplacian ``I - D^-1/2 A D^-1/2``, whose
    spectrum lies in ``[0, 2]`` and is therefore comparable across cases of very different
    size — an unnormalised Laplacian's eigenvalues scale with degree and would make a
    150-node case's encoding incomparable to a 3-node case's.

    Isolated nodes get degree zero; their rows are left at zero rather than dividing by
    zero, which is the correct encoding for a node with no structural position.

    Args:
        adjacency: Symmetric 0/1 adjacency from :func:`undirected_adjacency`.
        k: Number of components to return.

    Returns:
        An ``(n_nodes, k)`` float32 array, zero-padded when the graph has fewer than
        ``k + 1`` nodes. Eigenvector signs are arbitrary; see the module docstring.
    """
    n = adjacency.shape[0]
    out = np.zeros((n, k), dtype=np.float32)
    if n < _MIN_EIGEN_NODES:
        return out

    degree = adjacency.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    laplacian = np.eye(n) - (inv_sqrt[:, None] * adjacency * inv_sqrt[None, :])

    # Symmetric by construction, so eigh is both exact and cheap at n <= 150.
    values, vectors = np.linalg.eigh(laplacian)
    order = np.argsort(values)
    # Drop the first eigenvector: it is the constant vector of the trivial zero
    # eigenvalue and carries no positional information.
    selected = vectors[:, order[1 : k + 1]]
    out[:, : selected.shape[1]] = selected.astype(np.float32)
    return out


def random_walk_pe(adjacency: np.ndarray, k: int) -> np.ndarray:
    """Return the ``k``-step random-walk return probabilities per node.

    Entry ``(i, s)`` is the probability that a walk started at node ``i`` is back at
    ``i`` after ``s + 1`` steps under the row-stochastic transition matrix ``D^-1 A``.
    A node on a triangle shows mass at step 2; a node on a 4-cycle at step 3; and so on,
    which is exactly the signal a 1-WL-bounded network cannot compute for itself.

    Args:
        adjacency: Symmetric 0/1 adjacency from :func:`undirected_adjacency`.
        k: Number of walk lengths, starting at one step.

    Returns:
        An ``(n_nodes, k)`` float32 array in ``[0, 1]``. Isolated nodes are all-zero,
        since a walk from a node with no edges is undefined rather than certain to
        return.
    """
    n = adjacency.shape[0]
    out = np.zeros((n, k), dtype=np.float32)
    if n == 0:
        return out

    degree = adjacency.sum(axis=1)
    inv = np.zeros_like(degree)
    nonzero = degree > 0
    inv[nonzero] = 1.0 / degree[nonzero]
    transition = inv[:, None] * adjacency

    power = np.eye(n)
    for step in range(k):
        power = power @ transition
        out[:, step] = np.diag(power).astype(np.float32)
    # An isolated node's row of `transition` is all zero, so its return probability is
    # zero at every step; that is the intended encoding and needs no special-casing.
    return out
