"""Structural guarantees the Phase 0 gate promises to every later phase."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

INVARIANT_MARKERS = (
    "The fact layer is a measurement instrument.",
    "Splits are temporal and frozen.",
    "Schema versions are pinned and recorded in every derived artifact.",
    "Nothing may assert a fact that does not exist for its substrate.",
    "Every run records:",
    "Never delete or overwrite a results file.",
    "Negative and null results are kept and reported.",
    "No real-world PII or identifiers ever enter the repo",
)

REQUIRED_PACKAGES = ("data", "facts", "corpus", "models", "eval", "human", "utils")

GITIGNORE_MUST_CONTAIN = ("/data/**", "/artifacts/**", "*.pt", "*.ckpt", ".env")


def test_claude_md_contains_all_eight_invariants(repo_root):
    text = (repo_root / "CLAUDE.md").read_text(encoding="utf-8")
    missing = [m for m in INVARIANT_MARKERS if m not in text]
    assert not missing, f"CLAUDE.md is missing invariants: {missing}"


REQUIRED_TOP_LEVEL = ["DECISIONS.md", "PHASE_LOG.md", "README.md", "CITATION.cff", "LICENSE"]


@pytest.mark.parametrize("name", REQUIRED_TOP_LEVEL)
def test_required_top_level_files_exist_and_are_not_empty(repo_root, name):
    path = repo_root / name
    assert path.is_file(), f"{name} missing"
    assert len(path.read_text(encoding="utf-8").strip()) > 100, f"{name} is a stub"


@pytest.mark.parametrize("pkg", REQUIRED_PACKAGES)
def test_source_packages_are_importable(pkg):
    __import__(f"g2t_aml.{pkg}")


@pytest.mark.parametrize("entry", GITIGNORE_MUST_CONTAIN)
def test_gitignore_excludes_the_required_paths(repo_root, entry):
    assert entry in (repo_root / ".gitignore").read_text(encoding="utf-8")


def test_no_stub_modules_in_models(repo_root):
    """`Do not create placeholder Python files with pass bodies` -- stubs rot."""
    models = repo_root / "src" / "g2t_aml" / "models"
    assert [p.name for p in models.glob("*.py")] == ["__init__.py"]
    assert (models / "__init__.py").read_text().strip() == ""


def test_src_never_imports_from_notebooks(repo_root):
    for path in (repo_root / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import notebooks" not in text
        assert "from notebooks" not in text


def test_no_hardcoded_data_or_artifact_paths_in_src(repo_root):
    """All directory roots go through Hydra `paths`; nothing in src/ may hardcode one."""
    forbidden = ('"data/', "'data/", '"artifacts/', "'artifacts/", '"/home/', '"/mnt/')
    offenders = []
    for path in (repo_root / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        offenders.extend(f"{path.name}:{tok}" for tok in forbidden if tok in text)
    assert not offenders, f"hardcoded paths found: {offenders}"


def test_every_public_util_function_has_a_docstring():
    import inspect

    from g2t_aml.utils import hashing, io, logging, run_context, seeding

    missing = []
    for module in (hashing, io, logging, run_context, seeding):
        for name, obj in vars(module).items():
            if name.startswith("_") or not inspect.isfunction(obj):
                continue
            if obj.__module__ != module.__name__:
                continue
            if not (obj.__doc__ or "").strip():
                missing.append(f"{module.__name__}.{name}")
    assert not missing, f"public functions without docstrings: {missing}"
