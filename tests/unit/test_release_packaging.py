"""Phase 14: the release bundles must stay licence-homogeneous.

The two bundles carry different licences, and merging them is a licence breach rather than
an untidiness. Nothing in the packaging script fails loudly if a source is added to the
wrong bundle -- the archive just comes out wrong -- so these tests are the enforcement.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    """Import a numbered script, which is not an importable module name.

    Args:
        name: The script filename stem, e.g. ``14_package_release``.

    Returns:
        The loaded module.
    """
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def packaging():
    """The packaging script, loaded as a module."""
    return _load("14_package_release")


def test_exactly_two_bundles_with_distinct_licences(packaging) -> None:
    """Two bundles, and they must not share a licence -- that is the whole design."""
    bundles = list(packaging.BUNDLES.values())
    assert len(bundles) == 2
    licences = {b.licence for b in bundles}
    assert licences == {"Apache-2.0", "CDLA-Sharing-1.0"}


def test_corpus_bundle_carries_no_apache_licensed_source(packaging) -> None:
    """Schemas, split manifests and the vocabulary are Apache-2.0 and ship in the code bundle.

    Copying one into the CDLA bundle would make it licence-heterogeneous, which is the one
    property it must not have.
    """
    corpus_sources = [src for src, _ in packaging.CORPUS_AND_FACTS.sources]
    for src in corpus_sources:
        assert not src.startswith(
            "schemas/"
        ), f"{src} is Apache-2.0 and must not enter the CDLA-Sharing-1.0 bundle"


def test_code_bundle_carries_no_corpus_or_case_data(packaging) -> None:
    """The Apache bundle must embed no source Data.

    A narrative or a fact record here would make an Apache-2.0 archive carry Enhanced Data,
    which is the breach in the other direction.
    """
    forbidden = ("data/processed", "data/interim", "data/raw", "corpus")
    for src, _ in packaging.CODE_AND_RESULTS.sources:
        assert not any(
            src.startswith(f) for f in forbidden
        ), f"{src} is Enhanced Data and must not enter the Apache-2.0 bundle"


def test_no_bundle_ships_anything_elliptic2(packaging) -> None:
    """Elliptic2 is access-gated with an unlocated data licence. Nothing derived ships."""
    for bundle in packaging.BUNDLES.values():
        for src, dest in bundle.sources:
            assert "elliptic" not in src.lower()
            assert "elliptic" not in dest.lower()


def test_corpus_bundle_carries_attribution_and_change_notice(packaging) -> None:
    """CDLA-Sharing-1.0 s3.2 requires attribution and a record of changes. Both are owed."""
    notice = packaging.CORPUS_AND_FACTS.notice
    assert "Altman" in notice, "the Data Provider attribution is a licence obligation"
    assert "s3.2" in notice or "3.2" in notice
    assert "Changes made" in notice
    assert "Elliptic2" in notice, "the notice must say what is deliberately absent"


def test_apache_bundle_is_not_given_a_cdla_notice(packaging) -> None:
    """An attribution notice on the Apache bundle would imply an obligation it does not carry."""
    assert packaging.CODE_AND_RESULTS.notice == ""


def test_bundle_names_are_distinct_and_descriptive(packaging) -> None:
    """The archive names must say which licence a downloader is getting."""
    names = [b.name for b in packaging.BUNDLES.values()]
    assert len(set(names)) == len(names)
    for name in names:
        assert name.startswith("graph2text-aml-")


def test_dockerignore_excludes_everything_gitignore_does() -> None:
    """`.gitignore` does not apply to `docker build`, and forgetting that costs 8 GB.

    Without `.dockerignore`, `COPY . .` sweeps in data/ (1.5 GB), .venv/ (6.3 GB),
    artifacts/ and wandb/. Nothing fails -- the image is simply enormous and its layer
    cache is invalidated by every run output. This is the check that would have caught it.
    """
    ignore = REPO_ROOT / ".dockerignore"
    assert ignore.is_file(), ".dockerignore is missing; the build context would be ~8 GB"
    text = ignore.read_text(encoding="utf-8")
    for required in ("data/", "artifacts/", ".venv/", "wandb/", ".git/", ".env"):
        assert required in text, f".dockerignore does not exclude {required}"


def test_dockerignore_keeps_what_the_build_time_quickstart_needs() -> None:
    """Both images run the quickstart at build time, so its inputs must reach the context.

    A broad `tests/` exclusion would leave the image building and then failing its own
    `RUN python scripts/14_quickstart.py` -- which is the check, defeating itself.
    """
    lines = [
        line.strip()
        for line in (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for fatal in ("tests", "tests/", "scripts", "scripts/", "*"):
        assert fatal not in lines, f".dockerignore excludes {fatal!r}, which the build needs"


def test_both_images_pin_their_base_by_digest() -> None:
    """A tag is not a pin: upstream rebuilds `python:3.11-slim-bookworm` regularly."""
    for name in ("Dockerfile.cpu", "Dockerfile.gpu"):
        text = (REPO_ROOT / "docker" / name).read_text(encoding="utf-8")
        froms = [ln for ln in text.splitlines() if ln.startswith("FROM ")]
        assert froms, f"{name} has no FROM"
        for line in froms:
            assert "@sha256:" in line, f"{name} pins a base by tag, not digest: {line}"


def test_reconstruction_script_names_no_pinned_elliptic2_checksums() -> None:
    """We have never seen the Elliptic2 files, so pinning a digest would be a fabrication."""
    module = _load("14_reconstruct_elliptic2")
    assert len(module.EXPECTED_FILES) == 5
    source = (REPO_ROOT / "scripts" / "14_reconstruct_elliptic2.py").read_text("utf-8")
    assert (
        "sha256" not in source.lower()
    ), "no Elliptic2 checksum may be pinned: the files have never been obtained"
