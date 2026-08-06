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
    """`Do not create placeholder Python files with pass bodies` -- stubs rot.

    Written in Phase 0, when `models/` held nothing and the check was that its
    `__init__.py` stayed empty. It now holds three real subpackages, and Phase 14 gave the
    `__init__.py` a docstring -- so the assertion is restated as what it always meant: no
    module here may consist of nothing but placeholder bodies. A docstring is documentation,
    not a stub, and CLAUDE.md section 6 says as much.
    """
    import ast

    models = repo_root / "src" / "g2t_aml" / "models"
    assert [p.name for p in models.glob("*.py")] == ["__init__.py"]

    for path in models.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if not functions:
            continue

        def is_stub(node: ast.FunctionDef) -> bool:
            body = [
                stmt
                for stmt in node.body
                if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
            ]
            return len(body) == 1 and isinstance(body[0], ast.Pass)

        assert not all(is_stub(fn) for fn in functions), f"{path} is all placeholder bodies"


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


def test_every_hydra_entrypoint_propagates_its_exit_code(repo_root):
    """`@hydra.main` discards its wrapped function's return value.

    `sys.exit(main())` on a Hydra-decorated `main` therefore always exits 0, which
    silently disabled the non-zero exit every pipeline script documents: a failing gate
    looked exactly like a passing one to CI, to `make`, and to anything checking `$?`.
    It was live in Phases 1-4 and was found in Phase 5.

    Each script keeps its Hydra entrypoint as `_run` and records the code in `_EXIT_CODE`,
    with a thin `main()` that returns it. This asserts the shape rather than the behaviour
    because running eight pipeline stages in a unit test is not a unit test.
    """
    offenders = []
    for path in sorted((repo_root / "scripts").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "@hydra.main" not in text:
            continue
        if "def main(cfg" in text or "def main(cfg: DictConfig)" in text:
            offenders.append(f"{path.name}: @hydra.main decorates main() directly")
        if "_EXIT_CODE" not in text:
            offenders.append(f"{path.name}: no _EXIT_CODE capture")
        if "sys.exit(main())" not in text:
            offenders.append(f"{path.name}: does not sys.exit(main())")
    assert not offenders, (
        "Hydra entrypoints must capture their exit code; @hydra.main returns None: " f"{offenders}"
    )


# ---------------------------------------------------------- Phase 6: Gold ---

#: Symbols that produce, carry or read a generated narrative. Named exactly rather than
#: matched by module, because the annotator-facing modules legitimately import Bronze's
#: *formatters* and *display maps*: the panel has to render `9,435 Canadian Dollar` the
#: way the alignment reads it back, or an annotator who copies it correctly is scored as
#: having invented it. What is forbidden is reaching the text.
GENERATED_NARRATIVE_SOURCES = (
    "render_bronze",
    "BronzeNarrative",
    "bronze_narrative_from_record",
    "target_narrative",
    "load_training_records",
    "SlotAlignmentExtractor",
    "extract_report",
)

#: Modules the annotation interface loads. None of them may reach a generator: an
#: annotator shown a template rendering is editing it, and a Gold set of edits to Bronze
#: cannot be used to evaluate anything that was trained on Bronze.
ANNOTATOR_FACING_MODULES = (
    "human/caseloader.py",
    "human/factpanel.py",
    "human/graphview.py",
    "human/validation.py",
    "human/annotation_ui.py",
)


@pytest.mark.parametrize("relative", ANNOTATOR_FACING_MODULES)
def test_no_annotator_facing_module_can_reach_a_generated_narrative(repo_root, relative):
    """Gold's independence, asserted structurally rather than trusted.

    `factpanel` is permitted to import Bronze's *formatters and display maps* — the panel
    must render values in the spelling the alignment reads back, or a correctly-copied
    amount scores as invented. What it may not do is touch a narrative, a renderer or a
    template body.
    """
    text = (repo_root / "src" / "g2t_aml" / relative).read_text(encoding="utf-8")
    code = "\n".join(line for line in text.splitlines() if not line.strip().startswith(("#", "*")))
    offenders = [
        token
        for token in GENERATED_NARRATIVE_SOURCES
        if f"import {token}" in code or f"{token}(" in code or f".{token}" in code
    ]
    assert not offenders, (
        f"{relative} can reach generated narrative text ({offenders}); an annotator shown "
        "a draft is editing it"
    )


def test_the_annotation_case_carries_no_narrative_field():
    """The load path has no field a generated narrative could travel in."""
    import dataclasses

    from g2t_aml.human.caseloader import AnnotationCase

    names = {f.name for f in dataclasses.fields(AnnotationCase)}
    assert not {n for n in names if "narrative" in n or "text" in n}, names


def test_the_gold_reservation_is_committed(repo_root):
    """It is data, not discipline: the ids are in the repo next to the split."""
    manifest = repo_root / "schemas" / "splits" / "amlworld"
    assert (manifest / "gold_reserved.txt").is_file()
    assert (manifest / "gold_reservation.json").is_file()


def test_the_annotation_protocol_documents_exist_and_are_substantial(repo_root):
    docs = repo_root / "docs" / "annotation"
    for name in ("annotation_guidelines.md", "recruitment.md"):
        text = (docs / name).read_text(encoding="utf-8")
        assert len(text) > 5000, f"{name} is a stub"
