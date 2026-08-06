"""Availability masks (invariant 4).

Nothing may assert a fact that does not exist for its substrate. These tests pin the two
masks, check the mask cannot be constructed sloppily, and — most importantly — assert that
the Hydra config masks and the code masks agree. Two vocabularies for the same concept is
already a risk; two vocabularies that disagree would let a fact be licensed by one and
forbidden by the other.
"""

from __future__ import annotations

import dataclasses

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from g2t_aml.data.canonical import (
    AMLWORLD_AVAILABILITY,
    ELLIPTIC2_AVAILABILITY,
    AvailabilityMask,
)

# The semantic group: everything Elliptic2's anonymisation takes away.
ELLIPTIC2_MUST_BE_FALSE = (
    "absolute_timestamps",
    "fine_temporal_resolution",
    "monetary_amounts",
    "currencies",
    "institution_identity",
    "entity_types",
    "typology_ground_truth",
    "semantic_node_features",
)


def test_elliptic2_masks_out_every_semantic_field():
    for name in ELLIPTIC2_MUST_BE_FALSE:
        assert getattr(ELLIPTIC2_AVAILABILITY, name) is False, name


def test_elliptic2_keeps_only_its_subgraph_labels():
    """The label is the one thing Elliptic2 does supply."""
    assert ELLIPTIC2_AVAILABILITY.node_labels is True


def test_amlworld_supports_everything_except_entity_types_and_supplied_features():
    expected_false = {"entity_types", "semantic_node_features"}
    for field in dataclasses.fields(AvailabilityMask):
        value = getattr(AMLWORLD_AVAILABILITY, field.name)
        assert value is (field.name not in expected_false), field.name


def test_amlworld_has_no_entity_types():
    """The CSV carries a bank code and nothing that says 'mixer' or 'exchange'."""
    assert AMLWORLD_AVAILABILITY.entity_types is False


def test_mask_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        AMLWORLD_AVAILABILITY.monetary_amounts = False  # type: ignore[misc]


def test_round_trips_through_dict():
    assert AvailabilityMask.from_dict(AMLWORLD_AVAILABILITY.to_dict()) == AMLWORLD_AVAILABILITY


def test_missing_field_is_rejected_rather_than_defaulted():
    """Silence must not become False; that is how invariant 4 gets violated quietly."""
    partial = AMLWORLD_AVAILABILITY.to_dict()
    del partial["currencies"]
    with pytest.raises(ValueError, match="missing fields"):
        AvailabilityMask.from_dict(partial)


def test_unknown_field_is_rejected():
    extra = AMLWORLD_AVAILABILITY.to_dict() | {"vibes": True}
    with pytest.raises(ValueError, match="unknown fields"):
        AvailabilityMask.from_dict(extra)


def test_assert_available_passes_for_supported_facts():
    AMLWORLD_AVAILABILITY.assert_available("monetary_amounts", "currencies")


def test_assert_available_raises_permission_error_for_masked_facts():
    """PermissionError, not ValueError: an invariant-4 breach is not an ordinary bug."""
    with pytest.raises(PermissionError, match="invariant 4"):
        ELLIPTIC2_AVAILABILITY.assert_available("monetary_amounts")


def test_assert_available_reports_every_denied_field():
    with pytest.raises(PermissionError) as exc:
        ELLIPTIC2_AVAILABILITY.assert_available("monetary_amounts", "currencies")
    assert "currencies" in str(exc.value)
    assert "monetary_amounts" in str(exc.value)


def test_assert_available_rejects_a_typo_in_a_field_name():
    with pytest.raises(ValueError, match="not availability mask fields"):
        AMLWORLD_AVAILABILITY.assert_available("moentary_amounts")


# ------------------------------------------------- config / code agreement ---


@pytest.fixture
def config_mask(configs_dir):
    def _load(substrate: str) -> dict[str, bool]:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(configs_dir), version_base="1.3"):
            cfg = compose(
                config_name="config",
                overrides=[f"data={substrate}"],
                return_hydra_config=True,
            )
            HydraConfig.instance().set_config(cfg)
            with open_dict(cfg):
                del cfg["hydra"]
            return OmegaConf.to_container(cfg.data.availability, resolve=True)

    yield _load
    GlobalHydra.instance().clear()


@pytest.mark.parametrize(
    ("substrate", "mask"),
    [("amlworld", AMLWORLD_AVAILABILITY), ("elliptic2", ELLIPTIC2_AVAILABILITY)],
)
def test_config_mask_matches_the_code_mask(config_mask, substrate, mask):
    """configs/data/*.yaml and data/canonical.py must not drift apart."""
    assert config_mask(substrate) == mask.to_config_mask()
