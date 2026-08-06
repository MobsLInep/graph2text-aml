"""The Hydra `facts` group and FactConfig must not drift apart.

Two vocabularies for the same thresholds is tolerable; two that *disagree* is not, because
a run could then be configured one way and recorded another — and `provenance.config` is
what a reviewer would use to reproduce a detector's verdict. Same reasoning as D-014 for
the availability mask.
"""

from __future__ import annotations

import dataclasses

import pytest
from hydra import compose, initialize_config_dir

from g2t_aml.facts.config import FactConfig
from g2t_aml.facts.schema import CASE_FACTS_SCHEMA_VERSION

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def cfg(configs_dir):
    with initialize_config_dir(version_base="1.3", config_dir=str(configs_dir)):
        return compose(config_name="config")


def test_facts_group_composes(cfg):
    assert cfg.facts.name == "default"


def test_hydra_defaults_reproduce_the_dataclass_defaults(cfg):
    # If these ever disagree, a `make facts` run silently uses different thresholds from
    # every unit test and every golden file.
    assert FactConfig.from_hydra(cfg.facts) == FactConfig()


def test_every_dataclass_field_is_reachable_from_the_config(cfg):
    from_hydra = FactConfig.from_hydra(cfg.facts)
    for field in dataclasses.fields(FactConfig):
        assert hasattr(from_hydra, field.name)


def test_config_overrides_reach_the_dataclass(configs_dir):
    with initialize_config_dir(version_base="1.3", config_dir=str(configs_dir)):
        overridden = compose(
            config_name="config",
            overrides=["facts.motifs.fan_min_width=7", "facts.burst.min_transactions=9"],
        )
    config = FactConfig.from_hydra(overridden.facts)
    assert config.fan_min_width == 7
    assert config.burst_min_transactions == 9


def test_the_config_is_what_lands_in_provenance(cfg):
    from tests.factories import fan_out_case

    from g2t_aml.facts.extractor import extract_facts

    config = FactConfig.from_hydra(cfg.facts)
    facts = extract_facts(fan_out_case(width=5), config)
    assert facts.provenance is not None
    assert facts.provenance.config == config.to_dict()


def test_schema_version_declared_by_hydra_matches_the_frozen_code_version(cfg):
    assert str(cfg.schema_version.case_facts) == CASE_FACTS_SCHEMA_VERSION


def test_counts_exact_cannot_be_turned_off():
    # A published tolerance commitment, not a knob.
    from g2t_aml.facts.config import ToleranceConfig

    with pytest.raises(ValueError, match="counts_exact cannot be disabled"):
        ToleranceConfig(counts_exact=False)


def test_burst_binding_is_tighter_than_the_configured_detection_window(cfg):
    # The vacuous-descriptor guard, asserted against the CONFIGURED window rather than
    # the dataclass default, so an override cannot make rapid_dispersal always true.
    from g2t_aml.facts.vocab import load_vocabulary, parse_condition

    config = FactConfig.from_hydra(cfg.facts)
    for descriptor in load_vocabulary().risk_descriptors.values():
        if descriptor.binds_to == "temporal.burst_window_hours":
            _, threshold = parse_condition(descriptor.condition)
            assert threshold < config.burst_window_hours
