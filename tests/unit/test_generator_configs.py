"""Every Phase 9 arm composes to the settings its name claims.

**This file exists because of a real bug.** The first version of the arm configs expressed
their settings as a nested ``overrides:`` block of dotted keys under ``experiment``. Hydra
never applies such a block — it is inert data — so ``experiment=generator_a1`` composed
with ``fusion.shuffle=false`` and the *control arm would have trained as a second copy of
the treatment*. Nothing would have failed. The run would have completed, the curves would
have been near-identical, and the honest conclusion from that comparison — "S1 does not
beat A1, there is no architecture contribution" — would have been drawn from two runs of
S1.

Every assertion below is one that bug would have broken.
"""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir

ARMS = ("generator_s1", "generator_a1", "generator_s2", "generator_b7", "generator_b8")


@pytest.fixture
def composed(configs_dir):
    """Return a function that composes one experiment config.

    Args:
        configs_dir: The repository's configs directory.

    Returns:
        A callable taking an experiment name and returning the resolved config.
    """

    def _compose(experiment: str):
        with initialize_config_dir(config_dir=str(configs_dir), version_base="1.3"):
            return compose(config_name="config", overrides=[f"experiment={experiment}"])

    return _compose


class TestArmComposition:
    """Each arm's identity survives composition."""

    @pytest.mark.parametrize("experiment", ARMS)
    def test_arm_name_is_set(self, composed, experiment) -> None:
        """Every arm declares which system it is, for the checkpoint and the history."""
        assert composed(experiment).experiment.arm

    def test_s1_is_gated_fusion_with_full_text(self, composed) -> None:
        """S1: GAT + F2 + full text, and the graph is NOT shuffled."""
        cfg = composed("generator_s1")
        assert cfg.experiment.arm == "S1"
        assert cfg.generator.text_mode == "full"
        assert cfg.generator.use_fusion is True
        assert cfg.fusion.gated is True
        assert cfg.fusion.shuffle is False

    def test_a1_actually_shuffles(self, composed) -> None:
        """THE assertion. A1 with shuffle=false is not a control, it is a second S1."""
        cfg = composed("generator_a1")
        assert cfg.experiment.arm == "A1"
        assert cfg.fusion.shuffle is True, (
            "A1 composed without shuffling. It would train as a copy of S1 and the "
            "S1-vs-A1 comparison would be a comparison of one arm against itself."
        )
        assert cfg.fusion.shuffle_mode == "across_batch"

    def test_a1_differs_from_s1_only_in_the_shuffle(self, composed) -> None:
        """The control must be identical to the treatment in every other respect."""
        s1 = composed("generator_s1")
        a1 = composed("generator_a1")

        assert s1.generator.text_mode == a1.generator.text_mode
        assert s1.generator.use_fusion == a1.generator.use_fusion
        assert s1.generator.max_seq_len == a1.generator.max_seq_len
        assert s1.generator.lora.r == a1.generator.lora.r
        assert s1.fusion.gated == a1.fusion.gated
        assert s1.fusion.projector == a1.fusion.projector
        assert s1.fusion.num_prefix_tokens == a1.fusion.num_prefix_tokens
        assert s1.encoder.arch == a1.encoder.arch
        assert s1.seed == a1.seed
        for key in (
            "epochs",
            "lr",
            "fusion_lr",
            "encoder_lr",
            "per_device_batch_size",
            "gradient_accumulation",
            "warmup_ratio",
            "max_grad_norm",
        ):
            assert s1.training[key] == a1.training[key], f"training.{key} differs between arms"

    def test_s2_removes_the_text(self, composed) -> None:
        """S2 is the headline arm: the graph is the only source of case information."""
        cfg = composed("generator_s2")
        assert cfg.generator.text_mode == "none"
        assert cfg.generator.use_fusion is True

    def test_b7_has_no_graph_at_all(self, composed) -> None:
        """B7 is the text-only baseline: no encoder, no fusion."""
        cfg = composed("generator_b7")
        assert cfg.generator.use_fusion is False
        assert cfg.generator.text_mode == "serialised"

    def test_b8_is_ungated(self, composed) -> None:
        """B8 is F1: the G-Retriever-style prefix, differing from S1 only in the gate."""
        s1 = composed("generator_s1")
        b8 = composed("generator_b8")
        assert b8.fusion.gated is False
        assert b8.generator.text_mode == s1.generator.text_mode
        assert b8.fusion.projector == s1.fusion.projector


class TestTrainingRegime:
    """The three learning rates and the guard weights survive composition."""

    def test_three_distinct_learning_rates(self, composed) -> None:
        """A single rate across all three is the documented way to get S1 == A1."""
        training = composed("generator_s1").training
        assert training.lr == pytest.approx(2e-4)
        assert training.fusion_lr == pytest.approx(1e-3)
        assert training.encoder_lr == pytest.approx(1e-5)
        assert training.fusion_lr > training.lr > training.encoder_lr

    def test_completion_only_loss_is_on(self, composed) -> None:
        """Off is not a supported training regime; it exists to be measured once."""
        assert composed("generator_s1").training.loss_on_completion_only is True

    def test_overfit_check_is_required(self, composed) -> None:
        """A full run cannot start until the wiring check has passed."""
        assert composed("generator_s1").training.require_overfit_check is True

    def test_fusion_projector_is_in_modules_to_save(self, composed) -> None:
        """The projector is trained in full and saved with the adapter, not adapted."""
        assert "fusion_projector" in composed("generator_s1").generator.lora.modules_to_save

    def test_guard_weights_sum_to_one(self, composed) -> None:
        """Scores from different runs are only comparable if the weights are normalised."""
        weights = composed("generator_s1").training.guard.weights
        total = weights.contradiction + weights.coverage + weights.unverifiable
        assert total == pytest.approx(1.0)

    def test_fusion_widths_track_the_encoder(self, composed) -> None:
        """The fusion layer cannot silently disagree with the encoder about widths."""
        cfg = composed("generator_s1")
        assert cfg.fusion.graph_dim == cfg.encoder.hidden_dim
        assert cfg.fusion.num_prefix_tokens == cfg.encoder.n_pooled_tokens

    def test_curriculum_is_bronze_silver_then_silver(self, composed) -> None:
        """The brief's default curriculum, expressed as config so the ablation is cheap."""
        stages = composed("generator_s1").training.curriculum
        assert [s.name for s in stages] == ["mixed", "silver_only"]
        assert list(stages[0].tiers) == ["bronze", "silver"]
        assert list(stages[1].tiers) == ["silver"]

    def test_no_stage_trains_on_gold_by_default(self, composed) -> None:
        """The Gold tail is opt-in; Gold TEST items are refused by the loader regardless."""
        stages = composed("generator_s1").training.curriculum
        assert all("gold" not in list(s.tiers) for s in stages)
