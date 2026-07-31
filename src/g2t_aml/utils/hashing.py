"""Content hashing for manifests.

Every hash in this module is a hex-encoded SHA-256 over a *canonical byte encoding* of
the input. Canonicalisation is the whole point: two logically identical configs, split
manifests or dataframes must hash identically regardless of key order, row order in the
case of ID lists, or float formatting. Invariants 2 and 5 depend on that.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

_CHUNK_BYTES = 1 << 20
HASH_PREFIX_LEN = 12


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def short(digest: str, length: int = HASH_PREFIX_LEN) -> str:
    """Return a truncated digest for use in directory names.

    Args:
        digest: A full hex digest.
        length: Number of leading hex characters to keep.

    Returns:
        The first ``length`` characters of ``digest``.

    Raises:
        ValueError: If ``length`` exceeds the digest length.
    """
    if length > len(digest):
        raise ValueError(f"cannot take {length} chars from a {len(digest)}-char digest")
    return digest[:length]


def hash_file(path: str | Path) -> str:
    """Hash a file's bytes.

    Args:
        path: Path to an existing regular file. Read in chunks, so multi-GB raw
            transaction dumps are safe.

    Returns:
        Hex-encoded SHA-256 of the file contents.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        IsADirectoryError: If ``path`` is a directory.
    """
    p = Path(path)
    hasher = hashlib.sha256()
    with p.open("rb") as fh:
        while chunk := fh.read(_CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_dir(root: str | Path, *, pattern: str = "**/*") -> str:
    """Hash a directory tree as an ordered map of relative path to file digest.

    Args:
        root: Directory to walk.
        pattern: Glob restricting which files participate.

    Returns:
        Hex-encoded SHA-256 over the sorted ``relpath:filehash`` listing.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
    """
    r = Path(root)
    if not r.exists():
        raise FileNotFoundError(f"no such directory: {r}")
    entries = sorted(p for p in r.glob(pattern) if p.is_file())
    listing = "\n".join(f"{p.relative_to(r).as_posix()}:{hash_file(p)}" for p in entries)
    return _digest(listing.encode("utf-8"))


def canonical_json(obj: Any) -> str:
    """Render an object as canonical JSON: sorted keys, no whitespace padding.

    Args:
        obj: Any JSON-serialisable structure. Mappings are key-sorted recursively by
            ``json.dumps(sort_keys=True)``; sets and tuples are not supported because
            their ordering is not meaningful across processes.

    Returns:
        A deterministic JSON string.

    Raises:
        TypeError: If ``obj`` contains a non-serialisable value.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def hash_config(cfg: Mapping[str, Any] | Any) -> str:
    """Hash a resolved configuration.

    Args:
        cfg: A plain mapping, or an OmegaConf ``DictConfig``. OmegaConf objects are
            resolved to containers first, so interpolations are baked in and two configs
            that differ only in how a value was expressed hash the same.

    Returns:
        Hex-encoded SHA-256 of the canonical JSON encoding.

    Raises:
        TypeError: If ``cfg`` cannot be reduced to a JSON-serialisable structure.
    """
    payload: Any = cfg
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except ImportError:  # pragma: no cover - omegaconf is a core dependency
        pass
    else:
        if isinstance(cfg, DictConfig | ListConfig):
            payload = OmegaConf.to_container(cfg, resolve=True)
    return _digest(canonical_json(payload).encode("utf-8"))


def hash_id_list(ids: Iterable[str | int], *, sort: bool = True) -> str:
    """Hash a list of record IDs, as used by frozen split manifests.

    Args:
        ids: The IDs. Coerced to ``str`` so that ``7`` and ``"7"`` hash alike.
        sort: If True (the default) the IDs are sorted first, making the hash a hash of
            the *set*. Pass False only when the order of the list is itself meaningful.

    Returns:
        Hex-encoded SHA-256 over the newline-joined IDs.

    Raises:
        ValueError: If ``ids`` is empty, which is almost always a bug in split
            construction rather than an intentional empty split.
    """
    items = [str(i) for i in ids]
    if not items:
        raise ValueError("refusing to hash an empty ID list")
    if sort:
        items.sort()
    return _digest("\n".join(items).encode("utf-8"))


def hash_dataframe(df: pd.DataFrame, *, use_index: bool = False) -> str:
    """Hash a pandas DataFrame by content, independent of row order.

    Args:
        df: The frame to hash. Columns are sorted by name before hashing so that column
            order does not affect the result; row hashes are then sorted, so row order
            does not either.
        use_index: Whether the index participates in the per-row hash.

    Returns:
        Hex-encoded SHA-256 combining the sorted column names, dtypes and row hashes.

    Raises:
        TypeError: If ``df`` contains a column pandas cannot hash (e.g. nested objects).
    """
    import pandas as pd

    ordered = df[sorted(df.columns.astype(str))]
    row_hashes = pd.util.hash_pandas_object(ordered, index=use_index).to_numpy()
    row_hashes.sort()

    header = canonical_json(
        {
            "columns": [str(c) for c in ordered.columns],
            "dtypes": [str(t) for t in ordered.dtypes],
            "n_rows": int(len(ordered)),
        }
    )
    hasher = hashlib.sha256()
    hasher.update(header.encode("utf-8"))
    hasher.update(row_hashes.tobytes())
    return hasher.hexdigest()


def hash_manifest(entries: Sequence[Mapping[str, Any]]) -> str:
    """Hash an ordered sequence of manifest entries.

    Args:
        entries: Manifest rows, each a JSON-serialisable mapping.

    Returns:
        Hex-encoded SHA-256 of the canonical JSON encoding of the sequence.

    Raises:
        TypeError: If any entry is not JSON-serialisable.
    """
    return _digest(canonical_json(list(entries)).encode("utf-8"))
