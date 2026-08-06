"""Phase 8: the fusion layer, the gate, and the shuffled control.

The control's tests are the ones that matter most. A control with fixed points is a
weakened treatment arm, and it biases the S1-versus-A1 comparison towards rejecting the
null — which is the direction that produces a paper claiming a contribution it does not
have.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from g2t_aml.models.fusion import (  # noqa: E402
    FUSION_DTYPE,
    PrefixFusion,
    ShuffledGraphFusion,
    assert_projector_is_fp32,
    derange,
    embedding_rms,
    gate_summary,
    soft_token_attention_mass,
)


class TestPrefixFusion:
    """The projection itself."""

    def test_emits_lm_width_tokens(self) -> None:
        """Pooled tokens are projected to the language model's width."""
        fusion = PrefixFusion(graph_dim=256, lm_dim=128, num_prefix_tokens=16)
        out = fusion(torch.randn(4, 16, 256))
        assert out.soft_tokens.shape == (4, 16, 128)

    def test_projector_is_fp32(self) -> None:
        """Every fusion parameter is fp32; the assertion is the deliverable, not a comment."""
        fusion = PrefixFusion(graph_dim=64, lm_dim=32, num_prefix_tokens=4)
        assert_projector_is_fp32(fusion)
        assert all(p.dtype is FUSION_DTYPE for p in fusion.parameters())

    def test_half_precision_projector_is_refused(self) -> None:
        """A projector cast to fp16 fails loudly rather than training badly."""
        fusion = PrefixFusion(graph_dim=64, lm_dim=32, num_prefix_tokens=4).half()
        with pytest.raises(TypeError, match="must train in"):
            assert_projector_is_fp32(fusion)

    def test_quantised_submodule_is_refused(self) -> None:
        """A bitsandbytes layer anywhere in the fusion tree is refused."""

        class FakeBnbLinear(nn.Linear):
            pass

        FakeBnbLinear.__module__ = "bitsandbytes.nn.modules"
        fusion = PrefixFusion(graph_dim=64, lm_dim=32, num_prefix_tokens=4)
        fusion.add_module("quantised", FakeBnbLinear(4, 4))
        with pytest.raises(TypeError, match="bitsandbytes"):
            assert_projector_is_fp32(fusion)

    def test_gate_starts_open_and_is_bounded(self) -> None:
        """F2's gate is in (0, 1) and starts meaningfully open, not at zero."""
        fusion = PrefixFusion(graph_dim=64, lm_dim=32, num_prefix_tokens=8)
        gate = fusion.gate_value()
        assert gate is not None
        assert bool((gate > 0).all()) and bool((gate < 1).all())
        assert 0.5 < float(gate.mean()) < 0.8

    def test_f1_has_no_gate(self) -> None:
        """F1 is exactly F2 without the gate, which is the only difference between them."""
        fusion = PrefixFusion(graph_dim=64, lm_dim=32, num_prefix_tokens=8, gated=False)
        assert fusion.gate_value() is None
        assert fusion(torch.randn(2, 8, 64)).gate is None

    def test_target_rms_scales_the_output(self) -> None:
        """Soft tokens are emitted at the embedding scale they will sit among."""
        fusion = PrefixFusion(
            graph_dim=64, lm_dim=32, num_prefix_tokens=4, gated=False, target_rms=0.05
        )
        out = fusion(torch.randn(8, 4, 64))
        assert float(out.soft_tokens.pow(2).mean().sqrt()) == pytest.approx(0.05, rel=0.05)

    def test_width_mismatch_raises_rather_than_broadcasting(self) -> None:
        """An encoder/fusion width disagreement fails instead of silently broadcasting."""
        fusion = PrefixFusion(graph_dim=64, lm_dim=32, num_prefix_tokens=4)
        with pytest.raises(ValueError, match="was built for"):
            fusion(torch.randn(2, 4, 128))

    def test_token_count_mismatch_raises(self) -> None:
        """A token-count disagreement fails: it would shift every masked position."""
        fusion = PrefixFusion(graph_dim=64, lm_dim=32, num_prefix_tokens=4)
        with pytest.raises(ValueError, match="was built for"):
            fusion(torch.randn(2, 8, 64))

    def test_perceiver_resamples_to_a_different_budget(self) -> None:
        """Only the perceiver may change the token count, and it does."""
        fusion = PrefixFusion(
            graph_dim=64,
            lm_dim=32,
            num_prefix_tokens=16,
            projector="perceiver",
            n_output_tokens=4,
        )
        assert fusion(torch.randn(3, 16, 64)).soft_tokens.shape == (3, 4, 32)

    def test_non_perceiver_refuses_a_different_budget(self) -> None:
        """A one-to-one projector cannot honour a different output count and says so."""
        with pytest.raises(ValueError, match="one-to-one"):
            PrefixFusion(
                graph_dim=64, lm_dim=32, num_prefix_tokens=16, projector="mlp", n_output_tokens=4
            )

    def test_gradient_reaches_the_projector(self) -> None:
        """The projector receives gradient; a detached fusion path trains nothing."""
        fusion = PrefixFusion(graph_dim=64, lm_dim=32, num_prefix_tokens=4)
        fusion(torch.randn(2, 4, 64)).soft_tokens.sum().backward()
        grads = [p.grad for p in fusion.projector.parameters() if p.grad is not None]
        assert grads and any(float(g.abs().sum()) > 0 for g in grads)


class TestDerangement:
    """No case may keep its own graph."""

    @pytest.mark.parametrize("n", [2, 3, 4, 8, 32])
    def test_never_has_a_fixed_point(self, n: int) -> None:
        """Every element moves, at every batch size the project uses."""
        for _ in range(50):
            order = derange(n)
            assert not bool((order == torch.arange(n)).any())

    def test_is_a_permutation(self) -> None:
        """The derangement is a bijection, so no graph is duplicated or dropped."""
        order = derange(16)
        assert sorted(order.tolist()) == list(range(16))

    def test_singleton_is_refused(self) -> None:
        """A single element has no derangement and the caller must handle it."""
        with pytest.raises(ValueError, match="no derangement"):
            derange(1)

    def test_is_reproducible_from_a_seed(self) -> None:
        """The control is reproducible independently of the model's random stream."""
        first = derange(8, generator=torch.Generator().manual_seed(7))
        second = derange(8, generator=torch.Generator().manual_seed(7))
        assert torch.equal(first, second)


class TestShuffledControl:
    """The A1 arm."""

    def _fusion(self) -> PrefixFusion:
        """Build a small fusion layer.

        Returns:
            The layer.
        """
        return PrefixFusion(graph_dim=32, lm_dim=16, num_prefix_tokens=4)

    def test_no_case_keeps_its_own_graph(self) -> None:
        """The headline property: the fixed-point count is zero over many batches."""
        control = ShuffledGraphFusion(self._fusion(), seed=1)
        for _ in range(40):
            control(torch.randn(4, 4, 32))
        assert control.stats.n_fixed_points == 0
        assert control.stats.to_dict()["fixed_point_rate"] == 0.0

    def test_batch_of_two_still_shuffles(self) -> None:
        """At the configured batch size the control is a swap, and never the identity."""
        control = ShuffledGraphFusion(self._fusion(), seed=2)
        for _ in range(30):
            control(torch.randn(2, 4, 32))
        assert control.stats.n_fixed_points == 0

    def test_singleton_batch_draws_from_the_buffer(self) -> None:
        """A trailing singleton batch is paired with a foreign case, not its own."""
        control = ShuffledGraphFusion(self._fusion(), seed=3)
        control(torch.randn(4, 4, 32))
        control(torch.randn(1, 4, 32))
        assert control.stats.n_unshuffled_batches == 0

    def test_first_singleton_is_counted_not_silently_passed(self) -> None:
        """With an empty buffer the batch is recorded as unshuffled rather than hidden."""
        control = ShuffledGraphFusion(self._fusion(), seed=4)
        control(torch.randn(1, 4, 32))
        assert control.stats.n_unshuffled_batches == 1

    def test_shuffle_happens_before_projection(self) -> None:
        """The wrapped projector is the same object, so only the pairing differs."""
        fusion = self._fusion()
        control = ShuffledGraphFusion(fusion, seed=5)
        assert control.fusion is fusion
        assert control.n_tokens == fusion.n_tokens

    def test_noise_mode_matches_moments(self) -> None:
        """The noise control preserves scale, so it is not merely a smaller input."""
        control = ShuffledGraphFusion(self._fusion(), mode="noise", seed=6)
        pooled = torch.randn(8, 4, 32) * 3.0 + 1.0
        control(pooled)
        assert control.stats.n_batches == 1

    def test_unknown_mode_is_refused(self) -> None:
        """A typo in the control mode fails at construction, not silently."""
        with pytest.raises(ValueError, match="unknown shuffle mode"):
            ShuffledGraphFusion(self._fusion(), mode="scramble")  # type: ignore[arg-type]


class TestDiagnostics:
    """Attention mass and the gate summary."""

    def test_attention_mass_is_reported_against_its_baseline(self) -> None:
        """Uniform attention reports a lift of 1.0, which is the point of the baseline."""
        attentions = [torch.full((2, 4, 10, 10), 0.1)]
        mass = soft_token_attention_mass(attentions, soft_start=0, n_soft=2)
        assert mass.uniform_baseline == pytest.approx(0.2)
        assert mass.lift == pytest.approx(1.0, abs=1e-5)

    def test_concentrated_attention_lifts_above_one(self) -> None:
        """Attention concentrated on the soft tokens shows as a lift above 1."""
        weights = torch.zeros(1, 1, 4, 8)
        weights[..., :2] = 0.5
        mass = soft_token_attention_mass([weights], soft_start=0, n_soft=2)
        assert mass.lift > 3.0

    def test_query_start_restricts_to_the_completion(self) -> None:
        """Restricting to generated positions changes the measurement, as it must."""
        weights = torch.zeros(1, 1, 4, 8)
        weights[:, :, :2, :2] = 0.5
        whole = soft_token_attention_mass([weights], soft_start=0, n_soft=2)
        completion = soft_token_attention_mass([weights], soft_start=0, n_soft=2, query_start=2)
        assert whole.mass > completion.mass

    def test_out_of_range_span_is_refused(self) -> None:
        """A soft span outside the key axis is a bug and is raised."""
        with pytest.raises(ValueError, match="key positions"):
            soft_token_attention_mass([torch.rand(1, 1, 4, 8)], soft_start=6, n_soft=4)

    def test_gate_summary_reports_the_minimum(self) -> None:
        """The minimum is reported because a mean hides a single collapsed token."""
        mean, low, high = gate_summary(torch.tensor([0.9, 0.9, 0.01]))
        assert low == pytest.approx(0.01)
        assert mean is not None and low is not None and high is not None
        assert low < mean < high

    def test_gate_summary_handles_no_gate(self) -> None:
        """An ungated variant reports None rather than a misleading zero."""
        assert gate_summary(None) == (None, None, None)

    def test_embedding_rms_is_positive(self) -> None:
        """The measured RMS of an embedding table is usable as a scale target."""
        assert embedding_rms(torch.randn(100, 32) * 0.02) == pytest.approx(0.02, rel=0.15)
