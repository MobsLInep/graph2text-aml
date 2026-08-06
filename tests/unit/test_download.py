"""Dataset registry and checksum verification.

The one behaviour that matters most here is negative: verification must never let a run
proceed on data whose content does not match what the registered statistics were computed
from.
"""

from __future__ import annotations

import pytest

from g2t_aml.data.download import (
    REGISTRY,
    ChecksumMismatchError,
    DatasetSource,
    DataUnavailableError,
    ExpectedFile,
    FileStatus,
    dataset_root,
    is_available,
    register_observed_checksums,
    verify,
)

CONTENT = b"synthetic fixture content\n"


@pytest.fixture
def fake_registry(tmp_path, monkeypatch):
    """Register a one-file dataset backed by a real temp file with a real digest."""
    from g2t_aml.utils.hashing import hash_file

    root = tmp_path / "toydata"
    root.mkdir()
    (root / "toy.csv").write_bytes(CONTENT)
    digest = hash_file(root / "toy.csv")

    source = DatasetSource(
        key="toy",
        subdir="toydata",
        files=(ExpectedFile(name="toy.csv", sha256=digest, size_bytes=len(CONTENT)),),
        acquisition="Obtain toy.csv from the fixture factory.",
        licence="CC0",
        redistributable=True,
        citation="nobody, 2026",
    )
    monkeypatch.setitem(REGISTRY, "toy", source)
    return tmp_path


def test_verify_passes_on_matching_content(fake_registry):
    report = verify("toy", fake_registry)
    assert report.ok
    assert report.files[0].status is FileStatus.OK
    report.raise_for_status()


def test_verify_flags_modified_content(fake_registry):
    path = dataset_root(fake_registry, "toy") / "toy.csv"
    path.write_bytes(CONTENT.replace(b"synthetic", b"tampered"))
    report = verify("toy", fake_registry)
    assert not report.ok
    assert report.files[0].status is FileStatus.CHECKSUM_MISMATCH


def test_checksum_mismatch_refuses_to_proceed(fake_registry):
    """Never silently proceed on a checksum mismatch."""
    path = dataset_root(fake_registry, "toy") / "toy.csv"
    path.write_bytes(CONTENT.replace(b"synthetic", b"tampered"))
    with pytest.raises(ChecksumMismatchError, match="Refusing to proceed"):
        verify("toy", fake_registry).raise_for_status()


def test_truncation_is_caught_by_size_without_hashing(fake_registry):
    path = dataset_root(fake_registry, "toy") / "toy.csv"
    path.write_bytes(CONTENT[:5])
    report = verify("toy", fake_registry, compute_checksums=False)
    assert report.files[0].status is FileStatus.CHECKSUM_MISMATCH


def test_missing_file_reports_and_raises_with_instructions(fake_registry):
    (dataset_root(fake_registry, "toy") / "toy.csv").unlink()
    report = verify("toy", fake_registry)
    assert report.files[0].status is FileStatus.MISSING
    with pytest.raises(DataUnavailableError, match="fixture factory"):
        report.raise_for_status()


def test_mismatch_is_reported_ahead_of_absence(fake_registry, monkeypatch):
    """A tampered file is the more alarming finding and must be the message shown."""
    source = REGISTRY["toy"]
    monkeypatch.setitem(
        REGISTRY,
        "toy",
        DatasetSource(
            key="toy",
            subdir="toydata",
            files=(*source.files, ExpectedFile(name="absent.csv")),
            acquisition=source.acquisition,
            licence=source.licence,
            redistributable=source.redistributable,
            citation=source.citation,
        ),
    )
    path = dataset_root(fake_registry, "toy") / "toy.csv"
    path.write_bytes(b"tampered")
    with pytest.raises(ChecksumMismatchError):
        verify("toy", fake_registry).raise_for_status()


def test_unpinned_checksum_is_unverified_not_ok(tmp_path, monkeypatch):
    root = tmp_path / "toydata"
    root.mkdir()
    (root / "toy.csv").write_bytes(CONTENT)
    monkeypatch.setitem(
        REGISTRY,
        "toy",
        DatasetSource(
            key="toy",
            subdir="toydata",
            files=(ExpectedFile(name="toy.csv"),),
            acquisition="",
            licence="CC0",
            redistributable=True,
            citation="",
        ),
    )
    report = verify("toy", tmp_path)
    assert report.files[0].status is FileStatus.UNVERIFIED
    assert report.ok  # present but unpinned does not block a first ingest
    assert report.files[0].actual_sha256 is not None


def test_is_available_is_cheap_and_non_raising(fake_registry):
    assert is_available("toy", fake_registry) is True
    (dataset_root(fake_registry, "toy") / "toy.csv").unlink()
    assert is_available("toy", fake_registry) is False
    assert is_available("no_such_dataset", fake_registry) is False


def test_register_observed_checksums_does_not_mutate_the_registry(fake_registry):
    before = REGISTRY["toy"].files[0].sha256
    observed = register_observed_checksums("toy", fake_registry)
    assert observed["toy.csv"] == before
    assert REGISTRY["toy"].files[0].sha256 == before


def test_report_is_json_serialisable(fake_registry):
    import json

    json.dumps(verify("toy", fake_registry).to_dict())


def test_unknown_dataset_raises_keyerror(tmp_path):
    with pytest.raises(KeyError):
        verify("not_registered", tmp_path)


# --------------------------------------------------- the real registry ---


def test_both_substrates_are_registered():
    assert {"amlworld_hi_small", "elliptic2"} <= set(REGISTRY)


def test_elliptic2_is_marked_not_redistributable():
    """Phase 14 depends on this being accurate."""
    assert REGISTRY["elliptic2"].redistributable is False


def test_elliptic2_acquisition_points_at_the_access_request():
    assert "elliptic.co/elliptic2" in REGISTRY["elliptic2"].acquisition


def test_amlworld_pins_checksums_for_both_required_files():
    for expected in REGISTRY["amlworld_hi_small"].files:
        assert expected.sha256 is not None, expected.name
        assert len(expected.sha256) == 64


def test_amlworld_acquisition_documents_the_manual_kaggle_step():
    acquisition = REGISTRY["amlworld_hi_small"].acquisition
    assert "kaggle" in acquisition.lower()
    assert "kaggle.json" in acquisition
