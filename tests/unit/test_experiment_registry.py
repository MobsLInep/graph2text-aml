"""Phase 11: the registry declares the matrix, and these tests pin what it declares.

The registry is the single source of truth for what the paper's results table contains. A
silent edit here changes an experiment, so the properties that carry the paper's claims are
asserted rather than reviewed: the seed policy, the A1 control's single degree of freedom,
the baseline currency floor, and the fact that every declared system can actually be
dispatched to an executor.
"""

from __future__ import annotations

import pytest

from g2t_aml.experiments.executors import coverage_of_registry
from g2t_aml.experiments.registry import (
    CENTRAL_CLAIM_SYSTEMS,
    SEEDS_CENTRAL,
    SEEDS_SINGLE,
    Executor,
    FusionVariant,
    Resource,
    SystemSpec,
    TextMode,
    UnknownSystemError,
    all_systems,
    central_claim_family,
    expand_runs,
    get_system,
    matrix_summary,
    resolution_order,
    system_ids,
    validate_registry,
    with_seeds,
)


def test_registry_is_self_consistent():
    assert validate_registry() == []


def test_every_declared_system_has_an_executor():
    """A system with no executor silently disappears from the results table."""
    assert coverage_of_registry(all_systems()) == []


def test_the_sixteen_systems_of_the_brief_are_all_present():
    ids = set(system_ids())
    for expected in ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "S1", "S2"):
        assert expected in ids
    for expected in ("A1", "A2", "A4", "A5", "A6"):
        assert expected in ids
    # A3 is a fusion sweep, so it expands into one arm per variant. Its F1 point is B8.
    assert {"A3_F3", "A3_F4"} <= ids


def test_seed_asymmetry_is_exactly_the_stated_policy():
    """Three seeds on the four systems carrying the central claim, one elsewhere."""
    assert {"S1", "S2", "A1", "B7"} == CENTRAL_CLAIM_SYSTEMS
    for spec in all_systems():
        expected = SEEDS_CENTRAL if spec.system_id in CENTRAL_CLAIM_SYSTEMS else SEEDS_SINGLE
        assert spec.seeds == expected, f"{spec.system_id} runs at {spec.seeds}"


def test_run_count_matches_the_seed_policy():
    runs = expand_runs()
    n_central = len(CENTRAL_CLAIM_SYSTEMS) * len(SEEDS_CENTRAL)
    n_other = (len(all_systems()) - len(CENTRAL_CLAIM_SYSTEMS)) * len(SEEDS_SINGLE)
    assert len(runs) == n_central + n_other


def test_a1_differs_from_s1_only_in_the_shuffle_flag():
    """The control's single degree of freedom, asserted at the registry level.

    tests/unit/test_generator_configs.py makes the same assertion about the Hydra
    composition. Both exist because Phase 9 found that an inert `overrides:` block had
    composed A1 with shuffle=false -- the control would have trained as a second copy of
    the treatment, nothing would have failed, and "S1 does not beat A1" would have been
    two runs of S1 compared against each other.
    """
    s1, a1 = get_system("S1"), get_system("A1")
    for field in (
        "encoder_arm",
        "fusion",
        "text_mode",
        "base_model",
        "training_config",
        "guard",
        "seeds",
        "resource",
        "executor",
    ):
        assert getattr(s1, field) == getattr(a1, field), f"A1 differs from S1 on {field}"
    assert s1.experiment_config != a1.experiment_config


def test_s2_is_graph_only_and_b7_is_text_only():
    assert get_system("S2").text_mode is TextMode.NONE
    assert get_system("S2").fusion is FusionVariant.F2
    assert get_system("B7").text_mode is TextMode.SERIALISED
    assert get_system("B7").fusion is FusionVariant.F0
    assert get_system("B7").encoder_arm is None


def test_b8_is_s1_with_the_gate_off():
    s1, b8 = get_system("S1"), get_system("B8")
    assert s1.fusion is FusionVariant.F2
    assert b8.fusion is FusionVariant.F1
    assert s1.fusion.gated is True
    assert b8.fusion.gated is False
    assert s1.fusion.projector == b8.fusion.projector
    assert s1.text_mode == b8.text_mode
    assert s1.encoder_arm == b8.encoder_arm


def test_fusion_variants_map_onto_the_built_machinery():
    """F1-F4 are gate-flag plus projector-kind, and the mapping is declared not inferred."""
    assert (FusionVariant.F1.gated, FusionVariant.F1.projector) == (False, "mlp")
    assert (FusionVariant.F2.gated, FusionVariant.F2.projector) == (True, "mlp")
    assert (FusionVariant.F3.gated, FusionVariant.F3.projector) == (True, "linear")
    assert (FusionVariant.F4.gated, FusionVariant.F4.projector) == (True, "perceiver")
    assert FusionVariant.F0.gated is None
    assert FusionVariant.F0.projector is None


def test_a3_holds_the_gate_fixed_and_moves_only_the_projector():
    s1 = get_system("S1")
    for arm in ("A3_F3", "A3_F4"):
        spec = get_system(arm)
        assert spec.fusion.gated is True, f"{arm} must keep the gate at F2's setting"
        assert spec.fusion.projector != s1.fusion.projector
        assert spec.text_mode == s1.text_mode
        assert spec.encoder_arm == s1.encoder_arm


def test_a2_uses_the_no_message_passing_control_encoder():
    assert get_system("A2").encoder_arm == "mlp"
    assert get_system("S1").encoder_arm == "gatv2"


def test_a5_trains_nothing_and_depends_on_s1():
    a5 = get_system("A5")
    assert a5.trained is False
    assert a5.guard is False
    assert "S1" in a5.depends_on
    assert get_system("S1").guard is True


def test_a6_uses_a_different_base_model():
    a6, s1 = get_system("A6"), get_system("S1")
    assert a6.base_model != s1.base_model
    assert a6.fusion == s1.fusion
    assert a6.text_mode == s1.text_mode


def test_every_baseline_records_a_dated_model_version():
    """A comparison table that cannot be dated is a desk-reject trigger at this venue."""
    for spec in all_systems():
        if spec.base_model is None:
            continue
        assert spec.base_model_version, f"{spec.system_id} has no pinned version"
        assert spec.base_model_release_date, f"{spec.system_id} has no release date"
        assert (
            spec.base_model_release_date >= "2024-01-01"
        ), f"{spec.system_id} uses a pre-2024 baseline"


def test_frontier_baselines_are_2025_or_later():
    """B3-B5 stand in for the current state of the art and must not go stale quietly."""
    for arm in ("B3", "B4", "B5"):
        assert get_system(arm).base_model_release_date >= "2025-01-01"


def test_b5_is_configured_as_a_real_agentic_competitor():
    b5 = get_system("B5")
    assert b5.executor is Executor.API_AGENTIC
    assert b5.extra["self_verify"] is True
    assert b5.extra["max_repair_rounds"] >= 2
    # B5 gets B4's exemplars, i.e. its best configuration rather than a bare zero-shot
    # draft. Starting the agentic loop from a deliberately weaker draft would be the
    # easiest way to weaken this baseline without it being visible in the results.
    assert b5.base_model == get_system("B4").base_model


def test_resolution_order_places_dependencies_first():
    order = [spec.system_id for spec in resolution_order()]
    assert order.index("S1") < order.index("A5")


def test_resolution_order_rejects_a_selection_missing_a_dependency():
    with pytest.raises(ValueError, match="depends on"):
        resolution_order([get_system("A5")])


def test_resolution_order_detects_a_cycle():
    a = SystemSpec(
        system_id="X",
        role="r",
        description="d",
        executor=Executor.TEMPLATE,
        resource=Resource.CPU,
        depends_on=("Y",),
    )
    b = SystemSpec(
        system_id="Y",
        role="r",
        description="d",
        executor=Executor.TEMPLATE,
        resource=Resource.CPU,
        depends_on=("X",),
    )
    with pytest.raises(ValueError, match="cycle"):
        resolution_order([a, b])


def test_external_dependencies_are_not_treated_as_matrix_members():
    """`encoder:gatv2` is a Phase 7 precondition, not a system to schedule."""
    order = resolution_order([get_system("S1")])
    assert [s.system_id for s in order] == ["S1"]


def test_unknown_system_raises():
    with pytest.raises(UnknownSystemError):
        get_system("Z9")


def test_validate_reports_every_problem_not_just_the_first():
    broken = [
        SystemSpec(
            system_id="P",
            role="r",
            description="d",
            executor=Executor.TRAINED_GENERATOR,
            resource=Resource.GPU,
            trained=True,
            base_model="old-model",
            base_model_release_date="2021-01-01",
        ),
    ]
    problems = validate_registry(broken)
    assert len(problems) >= 2
    assert any("2021-01-01" in p for p in problems)
    assert any("training config" in p for p in problems)


def test_validate_rejects_a_central_system_run_at_one_seed():
    bad = with_seeds("S1", [42])
    problems = validate_registry([bad])
    assert any("central claim" in p for p in problems)


def test_with_seeds_is_the_documented_extension_path():
    extended = with_seeds("A2", [42, 1337, 2024])
    assert extended.seeds == (42, 1337, 2024)
    assert get_system("A2").seeds == SEEDS_SINGLE  # the registry is unchanged


def test_with_seeds_refuses_an_empty_seed_list():
    with pytest.raises(ValueError, match="at least one seed"):
        with_seeds("S1", [])


def test_central_claim_family_leads_with_gate_8():
    family = central_claim_family()
    assert family[0] == ("S1", "A1")
    assert ("S1", "B7") in family


def test_matrix_summary_reports_the_asymmetry_explicitly():
    summary = matrix_summary()
    assert sorted(summary["multi_seed_systems"]) == sorted(CENTRAL_CLAIM_SYSTEMS)
    assert summary["seeds_central"] == list(SEEDS_CENTRAL)
    assert summary["seeds_single"] == list(SEEDS_SINGLE)
    assert summary["n_runs"] == len(expand_runs())


def test_spec_serialises_without_importing_the_registry():
    payload = get_system("S1").to_dict()
    assert payload["fusion"] == "F2"
    assert payload["fusion_gated"] is True
    assert payload["fusion_projector"] == "mlp"
    assert payload["is_central_claim"] is True
    assert payload["n_runs"] == len(SEEDS_CENTRAL)
