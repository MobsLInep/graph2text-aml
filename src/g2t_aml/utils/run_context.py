"""Provenance record written into every run directory.

Invariant 5: every run records its git SHA, resolved config, data manifest hash, seeds
and library versions. `RunContext` is that record. It is captured once at the top of a
pipeline script and serialised to ``run_context.json`` beside the results, so a number
in the paper can always be traced back to the exact code and inputs that produced it.
"""

from __future__ import annotations

import getpass
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from g2t_aml import CASE_FACTS_SCHEMA_VERSION, __version__
from g2t_aml.utils.hashing import hash_config
from g2t_aml.utils.io import write_json

# Recorded for every run regardless of whether the run used them; absent packages are
# reported as None so the record distinguishes "not installed" from "not checked".
TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "pyarrow",
    "polars",
    "networkx",
    "pydantic",
    "hydra-core",
    "omegaconf",
    "torch",
    "torch-geometric",
    "transformers",
    "peft",
    "bitsandbytes",
    "accelerate",
    "trl",
    "datasets",
    "vllm",
    "sacrebleu",
    "bert-score",
)


def _git(*args: str) -> str | None:
    """Run a git command, returning None outside a repository or if git is absent."""
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip() or None


def collect_library_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Resolve installed versions for the tracked dependency set.

    Args:
        packages: Distribution names to look up.

    Returns:
        Mapping from distribution name to version string, or None where the
        distribution is not installed in the active environment.
    """
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@dataclass(frozen=True, slots=True)
class RunContext:
    """Immutable provenance snapshot for a single pipeline run."""

    run_id: str
    experiment_name: str
    timestamp_utc: str
    git_sha: str | None
    git_branch: str | None
    git_dirty: bool
    config_hash: str
    seeds: dict[str, Any]
    data_manifest_hash: str | None
    schema_versions: dict[str, str]
    package_version: str
    python_version: str
    platform: str
    hostname: str
    user: str
    library_versions: dict[str, str | None]
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        *,
        experiment_name: str,
        cfg: Any,
        seeds: dict[str, Any],
        run_id: str | None = None,
        data_manifest_hash: str | None = None,
        repo_root: str | Path | None = None,
        **extra: Any,
    ) -> RunContext:
        """Capture the current environment as a RunContext.

        Args:
            experiment_name: Name of the experiment, used in the run directory.
            cfg: Resolved configuration (mapping or OmegaConf node); only its hash is
                stored here, the config itself is saved separately by Hydra.
            seeds: The dict returned by ``seed_everything``.
            run_id: Explicit run identifier; defaults to a UTC timestamp.
            data_manifest_hash: Hash of the input data manifest, if the stage has one.
            repo_root: Directory to run git in; defaults to the current directory.
            **extra: Additional free-form provenance fields.

        Returns:
            A populated RunContext. Git fields are None when not run inside a
            repository, which is expected in containers built from a source tarball.
        """
        now = datetime.now(UTC)
        cwd = str(repo_root) if repo_root is not None else "."
        sha = _git("-C", cwd, "rev-parse", "HEAD")
        branch = _git("-C", cwd, "rev-parse", "--abbrev-ref", "HEAD")
        status = _git("-C", cwd, "status", "--porcelain")

        return cls(
            run_id=run_id or now.strftime("%Y%m%dT%H%M%SZ"),
            experiment_name=experiment_name,
            timestamp_utc=now.isoformat(),
            git_sha=sha,
            git_branch=branch,
            git_dirty=bool(status),
            config_hash=hash_config(cfg),
            seeds=seeds,
            data_manifest_hash=data_manifest_hash,
            schema_versions={"case_facts": CASE_FACTS_SCHEMA_VERSION},
            package_version=__version__,
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            hostname=socket.gethostname(),
            user=getpass.getuser(),
            library_versions=collect_library_versions(),
            extra=dict(extra),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the context as a plain dict.

        Returns:
            A JSON-serialisable dict of every field.
        """
        return asdict(self)

    def save(self, run_dir: str | Path, *, filename: str = "run_context.json") -> Path:
        """Write the context into a run directory.

        Args:
            run_dir: Directory to write into; created if absent.
            filename: File name to use.

        Returns:
            The path written.

        Raises:
            OSError: If the directory cannot be created or the file cannot be written.
        """
        return write_json(Path(run_dir) / filename, self.to_dict())
