"""Atomic file I/O.

Every writer here follows the same discipline: serialise to a temporary file in the
*same directory* as the destination, fsync it, then ``Path.replace`` it into place.
``Path.replace`` (an ``os.replace``) is atomic within a filesystem, so a killed job
leaves either the old file or the new one, never a truncated file that a later stage
mistakes for valid input.
Same-directory temporaries matter because a rename across filesystems is not atomic.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

from g2t_aml.utils.hashing import canonical_json

JSONValue = Any


@contextmanager
def atomic_path(path: str | Path, *, suffix: str = ".tmp") -> Iterator[Path]:
    """Yield a temporary path that is atomically moved onto ``path`` on clean exit.

    Args:
        path: Final destination. Parent directories are created if absent.
        suffix: Suffix for the temporary file.

    Yields:
        The temporary path to write to.

    Raises:
        OSError: If the destination directory cannot be created or the rename fails.
            On any exception the temporary file is removed and ``path`` is untouched.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=suffix)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        yield tmp
        with tmp.open("rb") as fh:
            os.fsync(fh.fileno())
        tmp.replace(dest)  # atomic within a filesystem
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_json(path: str | Path, obj: JSONValue, *, canonical: bool = False) -> Path:
    """Write a JSON document atomically.

    Args:
        path: Destination file.
        obj: JSON-serialisable object.
        canonical: If True, emit the same byte-stable encoding used for hashing (sorted
            keys, no padding). Use it for anything that will later be hashed; leave it
            False for human-facing files such as ``run_context.json``.

    Returns:
        The destination path.

    Raises:
        TypeError: If ``obj`` is not JSON-serialisable.
        OSError: If the write or rename fails.
    """
    text = canonical_json(obj) if canonical else json.dumps(obj, indent=2, ensure_ascii=False)
    with atomic_path(path) as tmp:
        tmp.write_text(text + "\n", encoding="utf-8")
    return Path(path)


def read_json(path: str | Path) -> JSONValue:
    """Read a JSON document.

    Args:
        path: Source file.

    Returns:
        The decoded object.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, records: Iterable[JSONValue], *, canonical: bool = True) -> Path:
    """Write an iterable of records as JSON Lines, atomically.

    Args:
        path: Destination file.
        records: Records to serialise, one per line. Consumed lazily, so this streams.
        canonical: If True, each line uses the canonical encoding.

    Returns:
        The destination path.

    Raises:
        TypeError: If any record is not JSON-serialisable.
        OSError: If the write or rename fails.
    """
    with atomic_path(path) as tmp, tmp.open("w", encoding="utf-8") as fh:
        for record in records:
            line = (
                canonical_json(record)
                if canonical
                else json.dumps(record, ensure_ascii=False, default=str)
            )
            fh.write(line + "\n")
    return Path(path)


def read_jsonl(path: str | Path) -> Iterator[JSONValue]:
    """Stream records from a JSON Lines file.

    Args:
        path: Source file. Blank lines are skipped.

    Yields:
        One decoded record per non-empty line.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If a line is not valid JSON.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if stripped := line.strip():
                yield json.loads(stripped)


def write_parquet(path: str | Path, df: pd.DataFrame, *, compression: str = "zstd") -> Path:
    """Write a DataFrame to Parquet atomically.

    Args:
        path: Destination file.
        df: Frame to write. The index is dropped; carry identifiers in columns.
        compression: Parquet codec.

    Returns:
        The destination path.

    Raises:
        ImportError: If pyarrow is unavailable.
        OSError: If the write or rename fails.
    """
    with atomic_path(path, suffix=".parquet.tmp") as tmp:
        df.to_parquet(tmp, engine="pyarrow", compression=compression, index=False)
    return Path(path)


def read_parquet(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame.

    Args:
        path: Source file.
        **kwargs: Forwarded to ``pandas.read_parquet`` (e.g. ``columns=``).

    Returns:
        The decoded DataFrame.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ImportError: If pyarrow is unavailable.
    """
    import pandas as pd

    return pd.read_parquet(path, engine="pyarrow", **kwargs)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist.

    Args:
        path: Directory to create.

    Returns:
        The directory path.

    Raises:
        FileExistsError: If ``path`` exists and is not a directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
