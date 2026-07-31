"""Hydra composition contract.

These guard the Phase 0 gate: the config tree composes, `paths` is the only source of
directory roots, and the substrate availability masks (invariant 4) are present and
correctly shaped before any code depends on them.
"""

from __future__ import annotations

import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from g2t_aml.utils.hashing import hash_config

pytestmark = pytest.mark.integration

# Fact families that every substrate config must declare a verdict on. A missing key is
# an error, not an implicit False -- silence is how invariant 4 gets violated.
REQUIRED_AVAILABILITY_KEYS = {
    "amounts",
    "currencies",
    "real_timestamps",
    "bank_ids",
    "entity_types",
    "account_ids",
    "typology_labels",
    "node_features",
}
AMLWORLD_TYPOLOGY_COUNT = 8


@pytest.fixture
def cfg_factory(configs_dir):
    """Compose the config the way a real entrypoint does.

    `paths.root` resolves through `${hydra:runtime.cwd}`, which reads the HydraConfig
    singleton. Outside @hydra.main that singleton is unset, so we compose with
    return_hydra_config=True and install it, then drop the `hydra` node so tests see the
    same shape a job body sees.
    """

    def _make(*overrides: str):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(configs_dir), version_base="1.3"):
            cfg = compose(
                config_name="config",
                overrides=list(overrides),
                return_hydra_config=True,
            )
            HydraConfig.instance().set_config(cfg)
            with open_dict(cfg):
                del cfg["hydra"]
            return cfg

    yield _make
    GlobalHydra.instance().clear()


def test_default_config_composes_and_resolves(cfg_factory):
    cfg = cfg_factory()
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["seed"] == 42
    assert resolved["data"]["name"] == "amlworld"
    assert resolved["encoder"]["name"] == "gat"
    assert resolved["fusion"]["name"] == "prefix"
    assert resolved["corpus"]["tier"] == "bronze"
    assert resolved["experiment"]["name"] == "debug"


def test_paths_group_resolves_every_root(cfg_factory):
    paths = OmegaConf.to_container(cfg_factory().paths, resolve=True)
    for key, value in paths.items():
        assert isinstance(value, str) and value, f"paths.{key} did not resolve"
        assert "${" not in value, f"paths.{key} left an unresolved interpolation"
    assert paths["raw_dir"].endswith("/data/raw")
    assert paths["runs_dir"].endswith("/artifacts/runs")


def test_hydra_run_dir_template_points_at_artifacts_runs(configs_dir):
    """Invariant 6: results land in a fresh timestamped dir, never overwrite one."""
    # Read as plain YAML: the template is asserted verbatim, unresolved.
    root = yaml.safe_load((configs_dir / "config.yaml").read_text())
    run_dir = root["hydra"]["run"]["dir"]
    assert "${paths.runs_dir}" in run_dir
    assert "${now:%Y-%m-%d}" in run_dir
    assert "${now:%H-%M-%S}" in run_dir
    assert "${experiment.name}" in run_dir


def test_schema_version_is_pinned_and_propagates(cfg_factory):
    cfg = cfg_factory()
    assert cfg.schema_version.case_facts == "0.1.0"
    assert cfg.corpus.schema_version == cfg.schema_version.case_facts


@pytest.mark.parametrize("substrate", ["amlworld", "elliptic2"])
def test_substrate_declares_full_availability_mask(cfg_factory, substrate):
    data = cfg_factory(f"data={substrate}").data
    mask = OmegaConf.to_container(data.availability, resolve=True)
    assert set(mask) == REQUIRED_AVAILABILITY_KEYS
    assert all(isinstance(v, bool) for v in mask.values())


def test_elliptic2_masks_out_facts_it_cannot_support(cfg_factory):
    """Invariant 4: Elliptic2 has no amounts, currencies, timestamps or entity types."""
    mask = cfg_factory("data=elliptic2").data.availability
    assert mask.amounts is False
    assert mask.currencies is False
    assert mask.real_timestamps is False
    assert mask.entity_types is False
    assert mask.bank_ids is False


def test_amlworld_carries_all_eight_typologies(cfg_factory):
    assert len(cfg_factory("data=amlworld").data.typologies) == AMLWORLD_TYPOLOGY_COUNT


@pytest.mark.parametrize("tier", ["bronze", "silver", "gold"])
def test_all_corpus_tiers_compose(cfg_factory, tier):
    assert cfg_factory(f"corpus={tier}").corpus.tier == tier


def test_splits_are_referenced_as_committed_manifests(cfg_factory):
    """Invariant 2: splits come from manifest files, never from a runtime seed."""
    split = cfg_factory().data.split
    assert split.strategy == "temporal"
    assert "seed" not in split
    assert str(split.manifest_dir).endswith("schemas/splits/amlworld")


def test_config_hash_is_stable_and_override_sensitive(cfg_factory):
    base = hash_config(cfg_factory())
    assert base == hash_config(cfg_factory())
    assert base != hash_config(cfg_factory("seed=43"))
    assert base != hash_config(cfg_factory("data=elliptic2"))
