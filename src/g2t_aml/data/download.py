"""Dataset acquisition and integrity verification.

Neither substrate can be fetched programmatically without credentials or a signed
agreement, so this module does not pretend to download anything. What it does is make the
manual step explicit and *verifiable*: a registry of expected files with their SHA-256
digests, and a checker that refuses to let a run proceed on data it cannot vouch for.

Silently proceeding past a checksum mismatch is the failure mode this exists to prevent.
A truncated or wrong-variant CSV that loads without error would produce dataset statistics
that quietly disagree with the published paper, which is exactly the class of bug Phase 1
is supposed to make impossible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from g2t_aml.utils.hashing import hash_file

#: Kaggle slug for the IBM AMLworld release. Downloading needs an API token, so the step
#: is documented rather than automated.
AMLWORLD_KAGGLE_SLUG = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"

#: Where Elliptic2 access is requested. The data is not redistributable.
ELLIPTIC2_ACCESS_URL = "https://www.elliptic.co/elliptic2"
ELLIPTIC2_TOOLING_URL = "https://github.com/MITIBMxGraph/Elliptic2"


class FileStatus(str, Enum):
    """Outcome of checking a single expected file."""

    OK = "ok"
    """Present, and its digest matches the registry."""
    MISSING = "missing"
    """Not on disk."""
    CHECKSUM_MISMATCH = "checksum_mismatch"
    """Present, but its content differs from the registered digest."""
    UNVERIFIED = "unverified"
    """Present, but the registry holds no digest to compare against yet."""


@dataclass(frozen=True)
class ExpectedFile:
    """One file a dataset must provide.

    Attributes:
        name: Filename relative to the dataset's raw directory.
        sha256: Expected digest, or None when not yet pinned (see
            :func:`register_observed_checksums`).
        size_bytes: Expected size, or None if unknown. Checked before the digest because
            it is free and catches truncation immediately.
        description: What the file holds, for error messages and the data card.
        required: If False, absence is reported but does not fail verification.
    """

    name: str
    sha256: str | None = None
    size_bytes: int | None = None
    description: str = ""
    required: bool = True


@dataclass(frozen=True)
class DatasetSource:
    """A registered dataset and how a human obtains it.

    Attributes:
        key: Registry key, e.g. ``"amlworld_hi_small"``.
        subdir: Directory under ``data/raw`` holding the files.
        files: Expected files.
        acquisition: Human-readable instructions, printed verbatim on failure.
        licence: Short licence identifier. The full terms live in the data card.
        redistributable: Whether we may ship the raw data with the paper artifact.
            Phase 14 depends on this being accurate.
        citation: Bibliographic reference.
    """

    key: str
    subdir: str
    files: tuple[ExpectedFile, ...]
    acquisition: str
    licence: str
    redistributable: bool
    citation: str


@dataclass
class FileReport:
    """Verification outcome for a single file.

    Attributes:
        name: Filename as registered.
        status: What was found.
        path: Where it was looked for.
        actual_size: Size on disk, or None if absent.
        actual_sha256: Digest computed, or None if absent or not computed.
        expected_sha256: Digest from the registry, or None if unpinned.
    """

    name: str
    status: FileStatus
    path: Path
    actual_size: int | None = None
    actual_sha256: str | None = None
    expected_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns:
            Plain-typed mapping suitable for a manifest.
        """
        return {
            "name": self.name,
            "status": self.status.value,
            "path": str(self.path),
            "actual_size": self.actual_size,
            "actual_sha256": self.actual_sha256,
            "expected_sha256": self.expected_sha256,
        }


@dataclass
class VerificationReport:
    """Verification outcome for a whole dataset.

    Attributes:
        dataset: Registry key.
        root: Directory the files were looked for in.
        files: Per-file outcomes.
    """

    dataset: str
    root: Path
    files: list[FileReport] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether every required file is present and verified.

        ``UNVERIFIED`` counts as acceptable: it means the file is present but the registry
        has no digest pinned yet, which is the state on a first ingest.

        Returns:
            True if no required file is missing or mismatched.
        """
        return not self.problems

    @property
    def problems(self) -> list[FileReport]:
        """Return the reports that should block a run.

        Returns:
            Reports whose status is MISSING (for a required file) or CHECKSUM_MISMATCH.
        """
        bad = {FileStatus.MISSING, FileStatus.CHECKSUM_MISMATCH}
        return [f for f in self.files if f.status in bad]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for the manifest.

        Returns:
            Dataset key, root, overall verdict and per-file outcomes.
        """
        return {
            "dataset": self.dataset,
            "root": str(self.root),
            "ok": self.ok,
            "files": [f.to_dict() for f in self.files],
        }

    def raise_for_status(self) -> None:
        """Raise unless verification passed.

        Returns:
            None, when everything required is present and matching.

        Raises:
            KeyError: Never; present so callers see the full contract in one place.
            DataUnavailableError: If a required file is missing.
            ChecksumMismatchError: If any present file's digest disagrees with the
                registry. Checked second, because a mismatch is the more alarming of the
                two and should be the message the user sees.
        """
        source = REGISTRY[self.dataset]
        mismatched = [f for f in self.files if f.status is FileStatus.CHECKSUM_MISMATCH]
        if mismatched:
            detail = "\n".join(
                f"  {f.name}\n    expected {f.expected_sha256}\n    actual   {f.actual_sha256}"
                for f in mismatched
            )
            raise ChecksumMismatchError(
                f"checksum mismatch for {self.dataset}:\n{detail}\n"
                "Refusing to proceed. The file on disk is not the file the registered "
                "statistics were computed from — re-download it, or if you deliberately "
                "changed the source, update the registry and record why in DECISIONS.md."
            )
        missing = [f for f in self.files if f.status is FileStatus.MISSING]
        if missing:
            names = ", ".join(f.name for f in missing)
            raise DataUnavailableError(
                f"{self.dataset}: missing required file(s) {names} under {self.root}\n\n"
                f"{source.acquisition}"
            )


class DataUnavailableError(FileNotFoundError):
    """Raised when a required raw file is not on disk."""


class ChecksumMismatchError(ValueError):
    """Raised when a raw file's digest disagrees with the registry."""


# --------------------------------------------------------------------- registry ---

_AMLWORLD_ACQUISITION = f"""\
IBM AMLworld is distributed through Kaggle and needs an API token, so the download is a
deliberate manual step.

  1. Create a Kaggle API token (Account -> Settings -> API -> Create New Token) and save
     it to ~/.kaggle/kaggle.json with mode 600.
  2. uv run pip install kaggle
  3. kaggle datasets download -d {AMLWORLD_KAGGLE_SLUG} -p data/raw/amlworld --unzip
  4. Re-run this command; the checksums are verified automatically.

The full release is roughly 20 GB across six variants. Only HI-Small_Trans.csv and
HI-Small_Patterns.txt are needed for development. Both belong in data/raw/amlworld/, which
is gitignored. If they already exist elsewhere they may be symlinked in rather than copied
-- symlinks are resolved before hashing -- but data/raw/amlworld/ is the canonical location
and the one every config and test assumes.
"""

_ELLIPTIC2_ACQUISITION = f"""\
Elliptic2 is access-gated and NOT redistributable. There is no automated download.

  1. Request access at {ELLIPTIC2_ACCESS_URL}
  2. Follow the official tooling at {ELLIPTIC2_TOOLING_URL}
  3. Unzip so that data/raw/elliptic2/ contains:
       background_edges.csv  background_nodes.csv  connected_components.csv
       edges.csv  nodes.csv
  4. Re-run this command.

Elliptic2 is a demonstration substrate. Phase 1 does not block on it: its tests skip when
the files are absent.
"""

REGISTRY: dict[str, DatasetSource] = {
    "amlworld_hi_small": DatasetSource(
        key="amlworld_hi_small",
        subdir="amlworld",
        files=(
            ExpectedFile(
                name="HI-Small_Trans.csv",
                sha256="b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040",
                size_bytes=475_664_283,
                description="5,078,345 transactions, one per row, with a header line.",
            ),
            ExpectedFile(
                name="HI-Small_Patterns.txt",
                sha256="7636c1d712168139ba0ff90f1b45aac9888d0ad46387084560886f204c03d6e6",
                size_bytes=323_843,
                description=(
                    "370 delimited laundering streams covering 3,209 transactions across "
                    "eight typologies. The only source of typology ground truth."
                ),
            ),
        ),
        acquisition=_AMLWORLD_ACQUISITION,
        # Verified 2026-08-01 against github.com/IBM/AML-Data: the repository code is
        # Apache-2.0 but the *data* is CDLA-Sharing-1.0. Share-alike applies to published
        # data and derived data; CDLA-Sharing-1.0 s3.5 explicitly exempts "Results",
        # which covers trained models and metrics. See the data card.
        licence="CDLA-Sharing-1.0 (see docs/data_cards/amlworld_hi_small.md)",
        redistributable=True,
        citation=(
            "Altman, E., Blanuša, J., von Niederhäusern, L., Egressy, B., Anghel, A., "
            "Atasu, K. Realistic Synthetic Financial Transactions for Anti-Money "
            "Laundering Models. NeurIPS Datasets & Benchmarks, 2023."
        ),
    ),
    # The 500-row fixture slice under tests/fixtures/amlworld. It is a real slice of
    # HI-Small, deliberately registered without pinned checksums so that regenerating it
    # does not require a registry edit. It exists so the ingest entrypoint can be tested
    # end-to-end without the 475 MB release, and must never be used for a real ingest --
    # `expected_rows` records why its statistics cannot match the published figures.
    "amlworld_fixture": DatasetSource(
        key="amlworld_fixture",
        subdir="amlworld",
        files=(
            ExpectedFile(name="HI-Small_Trans.csv", description="500-row slice of HI-Small."),
            ExpectedFile(
                name="HI-Small_Patterns.txt",
                description="Three hand-assembled streams over rows of that slice.",
            ),
        ),
        acquisition="Regenerate with the snippet in docs/data_cards/amlworld_hi_small.md.",
        # Committing this slice is a redistribution of AMLworld data, so CDLA-Sharing-1.0
        # applies to it and its share-alike terms are honoured by tests/fixtures/NOTICE.
        licence="CDLA-Sharing-1.0, inherited from AMLworld",
        redistributable=True,
        citation="Slice of Altman et al. (2023); see the amlworld_hi_small entry.",
    ),
    "elliptic2": DatasetSource(
        key="elliptic2",
        subdir="elliptic2",
        files=(
            ExpectedFile(name="nodes.csv", description="Nodes of the labelled subgraphs."),
            ExpectedFile(name="edges.csv", description="Edges of the labelled subgraphs."),
            ExpectedFile(
                name="connected_components.csv",
                description="Subgraph membership and licit/suspicious labels.",
            ),
            ExpectedFile(
                name="background_nodes.csv",
                description="~49M background cluster nodes with anonymised features.",
            ),
            ExpectedFile(
                name="background_edges.csv",
                description="~196M background edges.",
            ),
        ),
        acquisition=_ELLIPTIC2_ACQUISITION,
        # As of 2026-08-01 no licence text could be located for the *data*: the official
        # tooling repository is Apache-2.0 but says nothing about the dataset, and the
        # download is behind a request form. Treated as non-redistributable until written
        # confirmation is obtained. See docs/data_cards/elliptic2.md.
        licence="Undetermined; access-gated. Treated as NOT redistributable",
        redistributable=False,
        citation=(
            "Bellei, C., Fenton, R., et al. The Shape of Money Laundering: Subgraph "
            "Representation Learning on the Blockchain with the Elliptic2 Dataset. "
            "KDD Workshop on Machine Learning in Finance, 2024."
        ),
    ),
}


def dataset_root(raw_dir: str | Path, dataset: str) -> Path:
    """Return the directory a dataset's raw files live in.

    Args:
        raw_dir: The ``paths.raw_dir`` root.
        dataset: Registry key.

    Returns:
        ``raw_dir / <subdir>``.

    Raises:
        KeyError: If ``dataset`` is not registered.
    """
    return Path(raw_dir) / REGISTRY[dataset].subdir


def verify(
    dataset: str, raw_dir: str | Path, *, compute_checksums: bool = True
) -> VerificationReport:
    """Check that a dataset's raw files are present and unmodified.

    Symlinks are resolved, so raw data may live outside the repository and be linked into
    ``data/raw/`` without copying gigabytes.

    Args:
        dataset: Registry key, e.g. ``"amlworld_hi_small"``.
        raw_dir: The ``paths.raw_dir`` root.
        compute_checksums: If False, presence and size are checked but digests are not
            computed. Hashing a 475 MB CSV takes a couple of seconds, which is worth
            skipping in unit tests but never in an ingest run.

    Returns:
        A report. It does not raise on failure — call
        :meth:`VerificationReport.raise_for_status` for that, so a caller may inspect and
        record the outcome before deciding.

    Raises:
        KeyError: If ``dataset`` is not registered.
        OSError: If a file exists but cannot be read.
    """
    source = REGISTRY[dataset]
    root = dataset_root(raw_dir, dataset)
    report = VerificationReport(dataset=dataset, root=root)

    for expected in source.files:
        path = root / expected.name
        if not path.exists():
            status = FileStatus.MISSING if expected.required else FileStatus.UNVERIFIED
            report.files.append(
                FileReport(
                    name=expected.name,
                    status=status,
                    path=path,
                    expected_sha256=expected.sha256,
                )
            )
            continue

        actual_size = path.stat().st_size
        # A size mismatch is a content mismatch, and reporting it without paying for a
        # 475 MB hash makes truncation fail fast.
        if expected.size_bytes is not None and actual_size != expected.size_bytes:
            report.files.append(
                FileReport(
                    name=expected.name,
                    status=FileStatus.CHECKSUM_MISMATCH,
                    path=path,
                    actual_size=actual_size,
                    actual_sha256=f"<not computed: size is {actual_size}, "
                    f"expected {expected.size_bytes}>",
                    expected_sha256=expected.sha256,
                )
            )
            continue

        if expected.sha256 is None:
            report.files.append(
                FileReport(
                    name=expected.name,
                    status=FileStatus.UNVERIFIED,
                    path=path,
                    actual_size=actual_size,
                    actual_sha256=hash_file(path) if compute_checksums else None,
                )
            )
            continue

        if not compute_checksums:
            report.files.append(
                FileReport(
                    name=expected.name,
                    status=FileStatus.UNVERIFIED,
                    path=path,
                    actual_size=actual_size,
                    expected_sha256=expected.sha256,
                )
            )
            continue

        actual = hash_file(path)
        report.files.append(
            FileReport(
                name=expected.name,
                status=(
                    FileStatus.OK if actual == expected.sha256 else FileStatus.CHECKSUM_MISMATCH
                ),
                path=path,
                actual_size=actual_size,
                actual_sha256=actual,
                expected_sha256=expected.sha256,
            )
        )

    return report


def is_available(dataset: str, raw_dir: str | Path) -> bool:
    """Report whether a dataset's required files are on disk, without hashing them.

    Used by ``pytest.mark.skipif`` guards, which must be cheap and must not fail.

    Args:
        dataset: Registry key.
        raw_dir: The ``paths.raw_dir`` root.

    Returns:
        True if every required file exists. False if any is absent or the dataset is not
        registered.
    """
    if dataset not in REGISTRY:
        return False
    root = dataset_root(raw_dir, dataset)
    return all((root / f.name).exists() for f in REGISTRY[dataset].files if f.required)


def register_observed_checksums(dataset: str, raw_dir: str | Path) -> dict[str, str]:
    """Compute digests for a dataset's files, for pinning into :data:`REGISTRY`.

    This is a developer helper for the first ingest of a new variant. It deliberately does
    not write to the registry: pinning a checksum is a decision that belongs in a reviewed
    diff, not in a side effect of running a script.

    Args:
        dataset: Registry key.
        raw_dir: The ``paths.raw_dir`` root.

    Returns:
        Filename to SHA-256, for files that exist.

    Raises:
        KeyError: If ``dataset`` is not registered.
    """
    root = dataset_root(raw_dir, dataset)
    return {
        f.name: hash_file(root / f.name)
        for f in REGISTRY[dataset].files
        if (root / f.name).exists()
    }
