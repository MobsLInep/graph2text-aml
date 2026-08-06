"""Near-duplicate detection over a corpus, by MinHash with LSH banding.

**Why deduplication is a gate and not a tidy-up.** Two nearly identical narratives in a
training corpus are worse than one: the pair teaches the generator that its output
distribution has a spike where the data happens to repeat, and if the two land on opposite
sides of a split they inflate every test-set surface metric by handing the model a training
example at evaluation time. Phase 2's leakage audit already found 148 near-duplicate
*cases* over 30,000; a template pack can manufacture near-duplicate *narratives* out of
cases that are not duplicates at all, which is the failure this module exists to catch.

**Why the threshold is Jaccard 0.85 on 5-grams.** Bronze narratives share their scaffolding
by design — the four-part structure, the hedging, the recommended action — so a 3-gram
similarity is high between any two records from the same family whatever their facts say,
and thresholding it would reject the corpus wholesale. A 5-gram shingle is long enough that
matching one usually means matching a rendered value, not merely a shared clause. 0.85 then
means "these two narratives agree on six sevenths of their five-word windows", which for
this template pack means they differ in almost no fact. Two records that pass at 0.85 can
still read similarly, and that is correct: they *are* similar reports about similar cases,
and pretending otherwise by tightening the threshold would delete real data.

**Why MinHash and not exact pairwise Jaccard.** 15,000 narratives is 112 million pairs.
128 permutations at 16 bands gives an S-curve whose transition sits close to 0.85, so the
candidate set is small and every candidate is then confirmed by *exact* Jaccard on the
underlying shingle sets. The approximation therefore only affects which pairs are examined,
never which are reported — a false negative in banding is possible and a false positive is
not, which is the right way round for a gate.

No new dependency: ``datasketch`` would bring one for a hundred lines of numpy.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "BANDS",
    "DEFAULT_THRESHOLD",
    "DuplicateReport",
    "MinHasher",
    "N_PERMUTATIONS",
    "SHINGLE_SIZE",
    "find_near_duplicates",
    "shingles",
]

#: Words per shingle. See the module docstring for why five and not three.
SHINGLE_SIZE = 5

#: MinHash signature length. 128 gives a standard error near 1/sqrt(128) = 8.8% on the
#: estimated Jaccard, which is ample given every candidate is confirmed exactly.
N_PERMUTATIONS = 128

#: LSH bands. 16 bands of 8 rows puts the S-curve's inflection at (1/16)^(1/8) = 0.84,
#: which is where the threshold is.
BANDS = 16

#: A pair at or above this exact Jaccard is a near-duplicate.
DEFAULT_THRESHOLD = 0.85

#: Members needed in an LSH bucket before it yields a candidate pair.
_PAIR = 2

#: 64-bit prime for the permutation family, and the 61-bit Mersenne modulus.
_MERSENNE = (1 << 61) - 1

_WORD_RE = re.compile(r"[a-z0-9|.,%-]+")


@dataclass(frozen=True)
class DuplicateReport:
    """Which records are near-duplicates of which.

    Attributes:
        threshold: The exact-Jaccard threshold applied.
        pairs: ``(kept_id, dropped_id, jaccard)`` for every confirmed near-duplicate.
        dropped: Identifiers removed from the corpus, in the order they were dropped.
        n_compared: Candidate pairs the banding surfaced, before exact confirmation.
    """

    threshold: float
    pairs: tuple[tuple[str, str, float], ...] = ()
    dropped: tuple[str, ...] = ()
    n_compared: int = 0

    @property
    def n_dropped(self) -> int:
        """Return how many records were removed.

        Returns:
            The count of dropped identifiers.
        """
        return len(self.dropped)

    def to_dict(self) -> dict[str, object]:
        """Return the report as a JSON-serialisable mapping.

        Returns:
            Threshold, counts, the dropped identifiers and up to twenty example pairs.
            Truncated because a pathological run would otherwise write a report larger
            than the corpus, and twenty examples is enough to diagnose one.
        """
        return {
            "threshold": self.threshold,
            "n_candidate_pairs": self.n_compared,
            "n_confirmed_pairs": len(self.pairs),
            "n_dropped": self.n_dropped,
            "dropped": list(self.dropped),
            "example_pairs": [
                {"kept": kept, "dropped": dropped, "jaccard": round(score, 4)}
                for kept, dropped, score in self.pairs[:20]
            ],
        }


def shingles(text: str, size: int = SHINGLE_SIZE) -> frozenset[str]:
    """Return the set of word n-grams in a narrative.

    Case is folded and punctuation that does not distinguish a value is dropped, so two
    narratives are not called different merely because one ends a sentence where the
    other does not. Digits, currency names and the ``bank|account`` separator (D-011) are
    kept: they are exactly what makes two reports about different cases different.

    Args:
        text: The narrative.
        size: Words per shingle.

    Returns:
        The shingle set. A text shorter than one shingle yields a single shingle holding
        all of it, so short narratives still compare rather than matching everything.
    """
    words = _WORD_RE.findall(text.lower())
    if len(words) <= size:
        return frozenset({" ".join(words)}) if words else frozenset()
    return frozenset(" ".join(words[i : i + size]) for i in range(len(words) - size + 1))


class MinHasher:
    """A fixed family of hash permutations, shared across a corpus.

    Deterministic across processes and machines: the coefficients come from a SHA-256
    stream over a fixed seed rather than from ``random``, so a corpus hashed today bands
    identically when it is re-checked.

    Attributes:
        n_permutations: Signature length.
    """

    def __init__(self, n_permutations: int = N_PERMUTATIONS, seed: int = 0) -> None:
        """Build the permutation family.

        Args:
            n_permutations: Signature length.
            seed: Seeds the coefficient stream.
        """
        self.n_permutations = n_permutations
        stream = np.frombuffer(
            b"".join(
                hashlib.sha256(f"{seed}:{i}".encode()).digest()
                for i in range((n_permutations * 16) // 32 + 1)
            ),
            dtype=np.uint64,
        )
        self._a = (stream[:n_permutations] % (_MERSENNE - 1)) + 1
        self._b = stream[n_permutations : 2 * n_permutations] % _MERSENNE

    def signature(self, members: frozenset[str]) -> np.ndarray:
        """Compute the MinHash signature of a shingle set.

        Args:
            members: The shingles.

        Returns:
            A ``(n_permutations,)`` array of uint64. An empty set yields the maximum
            signature, which collides with nothing.
        """
        if not members:
            return np.full(self.n_permutations, _MERSENNE, dtype=np.uint64)
        base = np.array(
            [int.from_bytes(hashlib.sha1(m.encode()).digest()[:8], "big") for m in sorted(members)],
            dtype=np.uint64,
        )
        # (a * h + b) mod (2^61 - 1), vectorised over shingles x permutations. Python
        # ints via object dtype would be exact but 400x slower; uint64 wraps, which is a
        # valid universal hash family here because the modulus is applied after.
        hashed = (base[:, None] * self._a[None, :] + self._b[None, :]) % np.uint64(_MERSENNE)
        result: np.ndarray = hashed.min(axis=0)
        return result


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Return the exact Jaccard similarity of two shingle sets.

    Args:
        left: One shingle set.
        right: The other.

    Returns:
        ``|intersection| / |union|``, and 1.0 when both are empty.
    """
    if not left and not right:
        return 1.0
    union = len(left | right)
    return len(left & right) / union if union else 1.0


@dataclass
class _Candidate:
    """Accumulates the banding buckets for one corpus."""

    buckets: dict[tuple[int, bytes], list[str]] = field(default_factory=lambda: defaultdict(list))


def find_near_duplicates(
    narratives: dict[str, str],
    threshold: float = DEFAULT_THRESHOLD,
    *,
    n_permutations: int = N_PERMUTATIONS,
    bands: int = BANDS,
) -> DuplicateReport:
    """Find near-duplicate narratives and decide which to drop.

    When a cluster of similar records is found, the one with the lexicographically
    smallest identifier is kept and the rest are dropped. Order-independence matters: a
    corpus built in a different order must dedupe to the same set, or the gate would not
    be reproducible.

    Args:
        narratives: Identifier to narrative text.
        threshold: Exact-Jaccard threshold at or above which a pair is a duplicate.
        n_permutations: Signature length.
        bands: LSH bands. Must divide ``n_permutations``.

    Returns:
        The report.

    Raises:
        ValueError: If ``bands`` does not divide ``n_permutations``, which would leave
            part of every signature unbanded and silently lower recall.
    """
    if n_permutations % bands:
        raise ValueError(
            f"bands ({bands}) must divide n_permutations ({n_permutations}); a remainder "
            "would leave part of every signature unbanded and quietly reduce recall"
        )
    rows = n_permutations // bands
    hasher = MinHasher(n_permutations=n_permutations)

    sets = {key: shingles(text) for key, text in narratives.items()}
    signatures = {key: hasher.signature(members) for key, members in sets.items()}

    candidate = _Candidate()
    for key in sorted(signatures):
        signature = signatures[key]
        for band in range(bands):
            chunk = signature[band * rows : (band + 1) * rows].tobytes()
            candidate.buckets[(band, chunk)].append(key)

    pairs: set[tuple[str, str]] = set()
    for members in candidate.buckets.values():
        if len(members) < _PAIR:
            continue
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                pairs.add((left, right) if left < right else (right, left))

    confirmed: list[tuple[str, str, float]] = []
    dropped: set[str] = set()
    for left, right in sorted(pairs):
        if left in dropped:
            continue
        score = jaccard(sets[left], sets[right])
        if score >= threshold and right not in dropped:
            dropped.add(right)
            confirmed.append((left, right, score))

    return DuplicateReport(
        threshold=threshold,
        pairs=tuple(confirmed),
        dropped=tuple(sorted(dropped)),
        n_compared=len(pairs),
    )
