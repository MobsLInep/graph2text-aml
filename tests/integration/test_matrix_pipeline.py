"""Phase 11 end to end: plan, run, resume, aggregate, tabulate, on a synthetic matrix.

The unit tests check each stage. This checks that the stages agree with each other -- that
what the runner writes is what the aggregator reads, that a system's id survives from the
registry through the run directory into a LaTeX row, and that an interrupted matrix
resumed and then aggregated produces the same numbers as one that ran straight through.

It also pins the two properties the whole phase exists to protect: the arm configs compose
to the values the registry claims they do, and every declared system can be dispatched.
"""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from g2t_aml.experiments.aggregate import (
    HEADLINE_METRIC,
    aggregate_matrix,
    main_table_latex,
    missing_report,
    write_outputs,
)
from g2t_aml.experiments.executors import coverage_of_registry
from g2t_aml.experiments.registry import (
    FusionVariant,
    TextMode,
    all_systems,
    get_system,
    system_ids,
)
from g2t_aml.experiments.runner import RunStatus, plan_matrix, run_matrix
from g2t_aml.utils.io import write_json, write_jsonl


@pytest.fixture
def compose_experiment(repo_root):
    """Compose an arm config the way a real entrypoint does."""

    def _make(experiment: str):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(repo_root / "configs"), version_base="1.3"):
            cfg = compose(
                config_name="config",
                overrides=[f"experiment={experiment}"],
                return_hydra_config=True,
            )
            HydraConfig.instance().set_config(cfg)
            with open_dict(cfg):
                del cfg["hydra"]
            return cfg

    yield _make
    GlobalHydra.instance().clear()


def _scoring_executor(value_by_system):
    """An executor that writes what Phase 10 would write, at a scripted score."""

    def _executor(spec, seed, directory):
        value = value_by_system[spec.system_id]
        write_json(
            directory / "metrics.json",
            {
                "systems": {
                    f"{spec.system_id}/balanced": {
                        "system": spec.system_id,
                        "stream": "balanced",
                        "faithfulness": {
                            "system": spec.system_id,
                            "n_cases": 20,
                            HEADLINE_METRIC: value,
                            "fact_coverage": value * 0.9,
                        },
                        "taxonomy": {"rate_by_class": {"H9": 1.0 - value}},
                        "by_typology": {},
                        "by_dataset": {},
                        "layer1": None,
                        "worst_cases": [],
                    }
                }
            },
        )
        write_jsonl(
            directory / "per_case.jsonl",
            [{"case_id": f"c{i}", "stream": "balanced", HEADLINE_METRIC: value} for i in range(20)],
        )
        return {"n_cases": 20}

    return _executor


# ----------------------------------------------------------------- config contract ---


def test_every_arm_config_composes(compose_experiment):
    for spec in all_systems():
        if spec.experiment_config is None:
            continue
        cfg = compose_experiment(spec.experiment_config)
        assert cfg.experiment.arm == spec.system_id, spec.system_id


def test_composed_configs_agree_with_what_the_registry_claims(compose_experiment):
    """A registry that says F2 and a config that composes ungated is a results table
    whose rows are labelled wrong, and nothing would fail."""
    for spec in all_systems():
        if spec.experiment_config is None or spec.fusion in (
            FusionVariant.F0,
            FusionVariant.NA,
        ):
            continue
        cfg = compose_experiment(spec.experiment_config)
        assert bool(cfg.fusion.gated) is spec.fusion.gated, spec.system_id
        assert str(cfg.fusion.projector) == spec.fusion.projector, spec.system_id


def test_text_mode_matches_the_registry(compose_experiment):
    """Scoped to the arms that actually run the generator.

    B3-B6 declare a text mode descriptively -- they consume the serialised record through
    the baseline prompt, not through `generator.text_mode` -- so asserting against the
    generator block for them would be asserting against a value nothing reads.
    """
    for spec in all_systems():
        if spec.experiment_config is None or spec.text_mode is TextMode.NA:
            continue
        if str(spec.executor) != "trained_generator":
            continue
        cfg = compose_experiment(spec.experiment_config)
        assert str(cfg.generator.text_mode) == str(spec.text_mode), spec.system_id


def test_a1_composes_with_the_shuffle_on_and_s1_with_it_off(compose_experiment):
    """Phase 9 found an inert `overrides:` block that composed A1 with shuffle=false --
    the control would have trained as a second copy of the treatment."""
    s1 = compose_experiment("generator_s1")
    a1 = compose_experiment("generator_a1")
    assert s1.fusion.shuffle is False
    assert a1.fusion.shuffle is True
    assert a1.fusion.shuffle_mode == "across_batch"
    # And nothing else differs in the fusion block.
    s1_fusion = OmegaConf.to_container(s1.fusion, resolve=True)
    a1_fusion = OmegaConf.to_container(a1.fusion, resolve=True)
    differing = {k for k in s1_fusion if s1_fusion[k] != a1_fusion[k]}
    assert differing == {"shuffle"}


def test_a5_composes_with_the_guard_disabled(compose_experiment):
    assert compose_experiment("matrix_a5").training.guard.enabled is False
    assert compose_experiment("generator_s1").training.guard.enabled is True


def test_a4_composes_with_the_encoder_unfrozen(compose_experiment):
    """S1 is the FROZEN configuration; A4 is the arm that unfreezes."""
    assert compose_experiment("generator_s1").generator.freeze_encoder is True
    assert compose_experiment("matrix_a4").generator.freeze_encoder is False


def test_a2_composes_with_the_no_message_passing_encoder(compose_experiment):
    assert compose_experiment("matrix_a2").encoder.name == "mlp"


def test_b5_composes_as_an_agentic_arm_with_a_current_model(compose_experiment):
    cfg = compose_experiment("matrix_b5")
    assert cfg.baseline.agentic is True
    assert cfg.baseline.max_repair_rounds >= 2
    assert str(cfg.baseline.model_release_date) >= "2025-01-01"
    # B5 inherits B4's exemplars: its best configuration, not a bare zero-shot draft.
    assert cfg.baseline.k_shot == compose_experiment("matrix_b4").baseline.k_shot


def test_every_declared_system_can_be_dispatched():
    assert coverage_of_registry(all_systems()) == []


# ------------------------------------------------------------------ end to end ---


def test_plan_run_aggregate_tabulate(tmp_path):
    scores = {"S1": 0.92, "A1": 0.55, "B7": 0.81}
    specs = [get_system(s) for s in scores]
    plan = plan_matrix(specs, root=tmp_path)
    result = run_matrix(
        plan,
        {"trained_generator": _scoring_executor(scores)},
        external_artifacts=("encoder:gatv2",),
    )
    assert result.ok
    assert len(result.records) == 9

    aggregated = aggregate_matrix(tmp_path, specs=specs, n_resamples=200)
    assert set(aggregated.systems_present) == set(scores)
    for system, value in scores.items():
        summary = aggregated.summaries[("balanced", HEADLINE_METRIC, system)]
        assert summary.mean == pytest.approx(value)
        assert summary.n_seeds == 3
        assert summary.std == pytest.approx(0.0)

    latex = main_table_latex(aggregated, metrics=(HEADLINE_METRIC,))
    assert "0.9200" in latex and "0.5500" in latex
    # The systems that did not run are still rows, and are named.
    assert "Systems with no numbers" in latex

    written = write_outputs(aggregated, tmp_path / "out")
    assert written["metrics_long_csv"].is_file()
    frame = aggregated.frame()
    assert set(frame["system"]) == set(scores)
    assert set(frame["seed"]) == {42, 1337, 2024}


def test_gate_8_comparison_survives_the_whole_pipeline(tmp_path):
    """S1 against A1, from run directories to a corrected p-value in a table cell."""
    specs = [get_system("S1"), get_system("A1")]
    plan = plan_matrix(specs, root=tmp_path)
    run_matrix(
        plan,
        {"trained_generator": _scoring_executor({"S1": 1.0, "A1": 0.0})},
        external_artifacts=("encoder:gatv2",),
    )
    aggregated = aggregate_matrix(
        tmp_path, specs=specs, metrics=(HEADLINE_METRIC,), n_resamples=500
    )
    family = aggregated.comparisons[f"balanced/{HEADLINE_METRIC}"]
    comparison = next(c for c in family if {c.system_a, c.system_b} == {"S1", "A1"})
    assert comparison.p_adjusted is not None
    assert comparison.p_adjusted < 0.05
    latex = main_table_latex(aggregated, metrics=(HEADLINE_METRIC,), reference="A1")
    assert "***" in latex or "**" in latex or "*" in latex


def test_an_interrupted_matrix_resumed_gives_the_same_numbers(tmp_path):
    scores = {"S1": 0.9, "A1": 0.4}
    specs = [get_system(s) for s in scores]
    plan = plan_matrix(specs, root=tmp_path)
    good = _scoring_executor(scores)

    def _die_on_a1_seed1337(spec, seed, directory):
        if spec.system_id == "A1" and seed == 1337:
            raise RuntimeError("interrupted")
        return good(spec, seed, directory)

    first = run_matrix(
        plan,
        {"trained_generator": _die_on_a1_seed1337},
        external_artifacts=("encoder:gatv2",),
    )
    assert not first.ok
    partial = aggregate_matrix(tmp_path, specs=specs, n_resamples=200)
    assert len(partial.missing) == 1
    assert partial.missing[0].system == "A1"
    assert "A1" in missing_report(partial)

    second = run_matrix(plan, {"trained_generator": good}, external_artifacts=("encoder:gatv2",))
    assert second.ok
    assert len(second.by_status(RunStatus.SKIPPED)) == 5
    assert len(second.by_status(RunStatus.COMPLETED)) == 1

    complete = aggregate_matrix(tmp_path, specs=specs, n_resamples=200)
    assert complete.missing == ()
    for system, value in scores.items():
        assert complete.summaries[("balanced", HEADLINE_METRIC, system)].mean == pytest.approx(
            value
        )


def test_the_full_registry_aggregates_from_an_empty_root(tmp_path):
    """The reporting path must work before any arm has trained -- which is where the
    project is, and RESULTS.md is written from exactly this call."""
    aggregated = aggregate_matrix(tmp_path, n_resamples=100)
    assert aggregated.systems_present == ()
    assert {m.system for m in aggregated.missing} == set(system_ids())
    written = write_outputs(aggregated, tmp_path / "out")
    assert all(path.is_file() for path in written.values())
