from __future__ import annotations

import json

from g2t_aml import CASE_FACTS_SCHEMA_VERSION
from g2t_aml.utils.run_context import RunContext, collect_library_versions
from g2t_aml.utils.seeding import seed_everything


def _ctx(**kw):
    return RunContext.capture(
        experiment_name="unit",
        cfg={"seed": 42, "experiment": {"name": "unit"}},
        seeds=seed_everything(42),
        **kw,
    )


def test_capture_populates_required_provenance_fields():
    ctx = _ctx(data_manifest_hash="deadbeef")
    assert ctx.experiment_name == "unit"
    assert ctx.config_hash
    assert ctx.seeds["seed"] == 42
    assert ctx.data_manifest_hash == "deadbeef"
    assert ctx.schema_versions["case_facts"] == CASE_FACTS_SCHEMA_VERSION
    assert ctx.python_version.startswith("3.")
    assert "numpy" in ctx.library_versions


def test_extra_fields_are_carried():
    assert _ctx(phase="0").extra == {"phase": "0"}


def test_save_writes_json_run_context(tmp_path):
    path = _ctx().save(tmp_path)
    assert path.name == "run_context.json"
    loaded = json.loads(path.read_text())
    assert loaded["schema_versions"]["case_facts"] == CASE_FACTS_SCHEMA_VERSION
    assert loaded["config_hash"]


def test_git_fields_are_none_outside_a_repo(tmp_path):
    ctx = _ctx(repo_root=tmp_path)
    assert ctx.git_sha is None
    assert ctx.git_branch is None


def test_collect_library_versions_reports_missing_as_none():
    versions = collect_library_versions(("numpy", "definitely-not-installed-xyz"))
    assert versions["numpy"] is not None
    assert versions["definitely-not-installed-xyz"] is None
