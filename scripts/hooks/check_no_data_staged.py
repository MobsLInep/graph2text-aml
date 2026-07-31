#!/usr/bin/env python3
"""Pre-commit hook: refuse commits that stage files under data/ or artifacts/.

.gitkeep markers are the only permitted exception. Datasets are large, often licensed
or access-gated, and results files are append-only run outputs (invariant 6) -- none of
them belong in git history, where a mistake is permanent.
"""

from __future__ import annotations

import sys
from pathlib import PurePosixPath

BLOCKED_ROOTS = ("data", "artifacts")
ALLOWED_NAMES = {".gitkeep"}


def offending(paths: list[str]) -> list[str]:
    """Return the staged paths that violate the rule.

    Args:
        paths: Staged file paths, repo-relative, as pre-commit passes them.

    Returns:
        The subset falling under a blocked root and not an allowed marker file.
    """
    bad = []
    for raw in paths:
        parts = PurePosixPath(raw).parts
        if parts and parts[0] in BLOCKED_ROOTS and PurePosixPath(raw).name not in ALLOWED_NAMES:
            bad.append(raw)
    return bad


def main(argv: list[str]) -> int:
    """Check the staged paths.

    Args:
        argv: Staged file paths.

    Returns:
        0 if clean, 1 if any blocked path was staged.
    """
    bad = offending(argv)
    if not bad:
        return 0
    print("Refusing to commit files under data/ or artifacts/:", file=sys.stderr)
    for path in bad:
        print(f"  {path}", file=sys.stderr)
    print(
        "\nThese directories are gitignored by design. If you genuinely need to commit\n"
        "a manifest or fixture, put it under schemas/ or tests/ instead.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
