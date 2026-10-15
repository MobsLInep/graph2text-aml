"""Phase 9: the wiring tests that have to pass before an eighteen-hour run starts.

Every test here targets a failure that is **silent in the loss curve**. A misaligned loss
mask, a detached fusion path, a checkpoint that omits the projector, a Gold test item in
the training set — each produces a run that looks entirely normal and a result that is
wrong in a direction that favours the paper. That is why they are unit tests against a
stub backbone rather than something to check by eye on the real model: they have to be
cheap enough to run every time.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

torch = pytest.importorskip("torch")

from tests.stubs import StubCausalLM, StubTokenizer  # noqa: E402

from g2t_aml.human.reservation import GoldReservation, ReservationError  # noqa: E402
from g2t_aml.models.fusion import PrefixFusion  # noqa: E402
from g2t_aml.models.generator import (  # noqa: E402
    GeneratorConfig,
    GeneratorDataset,
    Graph2TextGenerator,
    GraphCollator,
    PromptBuilder,
    SegmentRole,
    TrainingConfig,
    assert_no_gold_test,
    build_curriculum,
    build_optimizer,
    loss_mask_report,
    overfit_check,
)
from g2t_aml.models.generator.train import (  # noqa: E402
    GeneratorTrainer,
    cosine_schedule_with_warmup,
)


@pytest.fixture
def tokenizer() -> StubTokenizer:
    """A deterministic stub tokeniser.

    Returns:
        The tokeniser.
    """
    return StubTokenizer()


@pytest.fixture
def records() -> list[dict[str, object]]:
    """Twenty-four synthetic training records with distinct narratives.

    Returns:
        The records. Synthetic ids only (invariant 8).
    """
    return [
        {
            "case_id": f"synthetic-{i:04d}",
            "serialised_facts": f"n_nodes={i + 3} n_edges={i + 5} typology=fan_out",
            "target_narrative": f"account {i} dispersed funds to {i + 2} counterparties",
        }
        for i in range(24)
    ]


@pytest.fixture
def builder(tokenizer: StubTokenizer) -> PromptBuilder:
    """A prompt builder reserving four soft tokens.

    Args:
        tokenizer: The stub tokeniser.

    Returns:
        The builder.
    """
    return PromptBuilder(
        tokenizer,
        n_soft_tokens=4,
        soft_token_id=tokenizer.soft_token_id,
        text_mode="full",
        max_seq_len=128,
    )


def _graph(n_nodes: int = 5, graph_dim: int = 8):
    """Build a small PyG Data object.

    Args:
        n_nodes: Node count, varied across cases so batching is exercised.
        graph_dim: Node feature width.

    Returns:
        The graph.
    """
    from torch_geometric.data import Data

    return Data(
        x=torch.randn(n_nodes, graph_dim),
        edge_index=torch.stack([torch.arange(n_nodes - 1), torch.arange(1, n_nodes)]),
    )


class _PooledEncoder(torch.nn.Module):
    """An encoder stub returning pooled tokens derived from the batch's node features."""

    def __init__(self, *, graph_dim: int = 8, hidden: int = 16, k: int = 4) -> None:
        """Build the stub encoder.

        Args:
            graph_dim: Node feature width.
            hidden: Pooled token width.
            k: Pooled token count.
        """
        super().__init__()
        self.project = torch.nn.Linear(graph_dim, hidden)
        self.k = k
        self.hidden_dim = hidden

    def forward(self, batch):
        """Pool node features into k tokens per graph.

        Args:
            batch: A PyG Batch.

        Returns:
            An object carrying ``pooled_tokens``.
        """
        from types import SimpleNamespace

        n_graphs = int(batch.num_graphs)
        pooled = torch.zeros(n_graphs, self.hidden_dim)
        pooled.index_add_(0, batch.batch, self.project(batch.x))
        return SimpleNamespace(
            pooled_tokens=pooled.unsqueeze(1).expand(-1, self.k, -1).contiguous()
        )


def _generator(*, gated: bool = True, freeze_encoder: bool = False) -> Graph2TextGenerator:
    """Assemble a stub generator with a real fusion layer.

    Args:
        gated: Build F2 when True, F1 when False.
        freeze_encoder: Whether to freeze the encoder.

    Returns:
        The generator.
    """
    lm = StubCausalLM(hidden_size=64, vocab_size=512)
    fusion = PrefixFusion(graph_dim=16, lm_dim=64, num_prefix_tokens=4, gated=gated)
    return Graph2TextGenerator(
        language_model=lm,
        fusion=fusion,
        encoder=_PooledEncoder(),
        config=GeneratorConfig(max_seq_len=128, freeze_encoder=freeze_encoder),
    )


class TestLossMasking:
    """The loss must see the completion and nothing else."""

    def test_soft_and_prompt_positions_are_masked(self, builder, records) -> None:
        """The brief's requirement, asserted directly: zero loss on prompt and soft tokens."""
        prompt = builder.build(records[0])
        for label, is_soft in zip(prompt.labels, prompt.soft_mask, strict=True):
            if is_soft:
                assert label == -100

        for segment, start in _segment_spans(prompt):
            if segment.role is not SegmentRole.COMPLETION:
                assert all(label == -100 for label in prompt.labels[start : start + len(segment)])

    def test_supervised_positions_are_exactly_the_completion(self, builder, records) -> None:
        """What the loss sees equals the target narrative, token for token."""
        prompt = builder.build(records[0])
        supervised = [label for label in prompt.labels if label != -100]
        assert supervised == prompt.input_ids[prompt.completion_start :]

    def test_collated_batch_supervises_no_padding(self, builder, records) -> None:
        """Padding contributes nothing, so the model is never taught to predict pad."""
        dataset = GeneratorDataset(records[:4], builder=builder)
        batch = GraphCollator(pad_token_id=0)([dataset[i] for i in range(4)])
        report = loss_mask_report(batch)
        assert report["n_supervised_soft"] == 0
        assert report["n_supervised_pad"] == 0
        assert report["n_rows_unsupervised"] == 0
        assert report["n_supervised"] > 0

    def test_loss_is_zero_when_every_label_is_masked(self, builder, records) -> None:
        """A fully masked batch yields no gradient rather than a spurious number."""
        dataset = GeneratorDataset(records[:2], builder=builder, for_training=False)
        batch = GraphCollator(pad_token_id=0)([dataset[i] for i in range(2)])
        assert batch.n_supervised_tokens == 0

    def test_inference_prompt_carries_no_answer(self, builder, records) -> None:
        """An inference prompt must not contain the target, or generation is cheating."""
        prompt = builder.build(records[0], for_training=False)
        assert all(label == -100 for label in prompt.labels)
        assert not any(s.role is SegmentRole.COMPLETION for s in prompt.segments)


def _segment_spans(prompt):
    """Yield each segment with its start offset.

    Args:
        prompt: A built prompt.

    Returns:
        Pairs of segment and start index.
    """
    offset = 0
    spans = []
    for segment in prompt.segments:
        spans.append((segment, offset))
        offset += len(segment)
    return spans


class TestForwardPass:
    """Shapes and the splice."""

    def test_logit_shape_with_soft_tokens(self, builder, records) -> None:
        """Splicing soft tokens leaves the sequence length and vocab axis intact."""
        generator = _generator()
        dataset = GeneratorDataset(
            records[:2],
            builder=builder,
            graphs={r["case_id"]: _graph(i + 3) for i, r in enumerate(records[:2])},
        )
        batch = GraphCollator(pad_token_id=0)([dataset[0], dataset[1]])
        out = generator(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            soft_mask=batch.soft_mask,
            graph_batch=batch.graph_batch,
        )
        assert out.logits.shape == (2, batch.input_ids.size(1), 512)
        assert out.loss is not None

    def test_splice_replaces_exactly_the_reserved_positions(self) -> None:
        """The soft tokens land where the mask says and nowhere else."""
        generator = _generator()
        embeds = torch.zeros(2, 6, 64)
        soft = torch.ones(2, 2, 64)
        mask = torch.zeros(2, 6, dtype=torch.bool)
        mask[:, 1:3] = True
        out = generator.splice_soft_tokens(embeds, soft, mask)
        assert bool((out[:, 1:3] == 1).all())
        assert bool((out[:, [0, 3, 4, 5]] == 0).all())

    def test_splice_refuses_a_count_mismatch(self) -> None:
        """A mismatched reservation raises instead of shifting every later position."""
        generator = _generator()
        mask = torch.zeros(2, 6, dtype=torch.bool)
        mask[0, :2] = True
        mask[1, :3] = True
        with pytest.raises(ValueError, match="exactly 2 soft-token positions"):
            generator.splice_soft_tokens(torch.zeros(2, 6, 64), torch.ones(2, 2, 64), mask)

    def test_fusion_arm_without_soft_mask_raises(self, builder, records) -> None:
        """Forgetting the mask fails loudly rather than generating a graph-free narrative."""
        generator = _generator()
        dataset = GeneratorDataset(records[:1], builder=builder)
        batch = GraphCollator(pad_token_id=0)([dataset[0]])
        with pytest.raises(ValueError, match="needs soft_mask"):
            generator(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask, soft_mask=None
            )


class TestGradientFlow:
    """Gradient must reach the parts the paper claims are trained."""

    def test_gradient_reaches_fusion_and_encoder(self, builder, records) -> None:
        """An unfrozen encoder and the projector both receive gradient."""
        generator = _generator(freeze_encoder=False)
        graphs = {r["case_id"]: _graph(i + 3) for i, r in enumerate(records[:2])}
        dataset = GeneratorDataset(records[:2], builder=builder, graphs=graphs)
        batch = GraphCollator(pad_token_id=0)([dataset[0], dataset[1]])
        out = generator(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            soft_mask=batch.soft_mask,
            graph_batch=batch.graph_batch,
        )
        out.loss.backward()

        fusion_grad = sum(
            float(p.grad.abs().sum())
            for p in generator.fusion_projector.parameters()
            if p.grad is not None
        )
        encoder_grad = sum(
            float(p.grad.abs().sum()) for p in generator.encoder.parameters() if p.grad is not None
        )
        assert fusion_grad > 0
        assert encoder_grad > 0

    def test_frozen_encoder_receives_none(self, builder, records) -> None:
        """A frozen encoder is genuinely frozen, so Phase 7's selection is not undone."""
        generator = _generator(freeze_encoder=True)
        assert all(not p.requires_grad for p in generator.encoder.parameters())

    def test_three_parameter_groups_carry_their_own_rates(self) -> None:
        """The three learning rates survive into the optimiser, which is the whole design."""
        generator = _generator(freeze_encoder=False)
        groups = generator.trainable_parameter_groups(lora_lr=2e-4, fusion_lr=1e-3, encoder_lr=1e-5)
        by_name = {g["name"]: g["lr"] for g in groups}
        assert by_name == {"lora": 2e-4, "fusion": 1e-3, "encoder": 1e-5}

    def test_groups_do_not_share_parameters(self) -> None:
        """No parameter appears twice, which would double its effective learning rate."""
        generator = _generator(freeze_encoder=False)
        groups = generator.trainable_parameter_groups(lora_lr=1e-4, fusion_lr=1e-3, encoder_lr=1e-5)
        seen: set[int] = set()
        for group in groups:
            for param in group["params"]:
                assert id(param) not in seen
                seen.add(id(param))

    def test_scheduler_preserves_the_rate_ratio(self) -> None:
        """LambdaLR scales each group, so the 2e-4 / 1e-3 / 1e-5 ratio holds all run."""
        generator = _generator(freeze_encoder=False)
        groups = generator.trainable_parameter_groups(lora_lr=2e-4, fusion_lr=1e-3, encoder_lr=1e-5)
        optimizer = torch.optim.AdamW(groups)
        scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps=2, total_steps=10)
        for _ in range(5):
            scheduler.step()
        rates = [g["lr"] for g in optimizer.param_groups]
        assert rates[1] / rates[0] == pytest.approx(1e-3 / 2e-4)
        assert rates[2] / rates[0] == pytest.approx(1e-5 / 2e-4)


class TestOverfit:
    """The cheapest test in the project."""

    def test_twenty_examples_reach_near_zero_loss(self, builder, records) -> None:
        """20 examples, 100 steps, loss near zero — the run-gating wiring check."""
        generator = _generator(freeze_encoder=False)
        graphs = {r["case_id"]: _graph(3 + (i % 5)) for i, r in enumerate(records)}
        dataset = GeneratorDataset(records, builder=builder, graphs=graphs)
        result = overfit_check(
            generator,
            dataset,
            GraphCollator(pad_token_id=0),
            n_examples=20,
            n_steps=100,
            lr=3e-3,
            fusion_lr=3e-3,
            threshold=0.15,
        )
        assert result["passed"] == 1.0, f"overfit check failed: {result}"
        assert result["final_loss"] < result["initial_loss"]

    def test_micro_batched_check_memorises_the_same_examples(self, builder, records) -> None:
        """Accumulating over passes of two is the same check on a card that cannot hold twenty."""
        generator = _generator(freeze_encoder=False)
        graphs = {r["case_id"]: _graph(3 + (i % 5)) for i, r in enumerate(records)}
        dataset = GeneratorDataset(records, builder=builder, graphs=graphs)
        result = overfit_check(
            generator,
            dataset,
            GraphCollator(pad_token_id=0),
            n_examples=20,
            n_steps=100,
            lr=3e-3,
            fusion_lr=3e-3,
            threshold=0.15,
            micro_batch_size=2,
        )
        assert result["passed"] == 1.0, f"micro-batched overfit check failed: {result}"
        assert result["final_loss"] < result["initial_loss"]

    def test_the_check_leaves_the_generator_as_it_found_it(self, builder, records) -> None:
        """The run it gates must start from the initialisation, not from twenty memorised cases."""
        generator = _generator(freeze_encoder=False)
        graphs = {r["case_id"]: _graph(3 + (i % 5)) for i, r in enumerate(records)}
        dataset = GeneratorDataset(records, builder=builder, graphs=graphs)
        before = {n: p.detach().clone() for n, p in generator.named_parameters()}
        result = overfit_check(
            generator, dataset, GraphCollator(pad_token_id=0), n_steps=30, lr=3e-3, fusion_lr=3e-3
        )
        assert result["final_loss"] < result["initial_loss"], "the check did not train at all"
        for name, param in generator.named_parameters():
            assert torch.equal(param, before[name]), f"{name} was left modified by the check"

    def test_micro_batch_size_below_one_is_refused(self, builder, records) -> None:
        """A size of zero would loop forever building empty passes."""
        dataset = GeneratorDataset(records, builder=builder)
        with pytest.raises(ValueError, match="micro_batch_size"):
            overfit_check(_generator(), dataset, GraphCollator(pad_token_id=0), micro_batch_size=0)

    def test_trainer_refuses_to_start_without_it(self, builder, records) -> None:
        """A full run cannot begin until the overfit check has passed."""
        generator = _generator()
        dataset = GeneratorDataset(records[:4], builder=builder)
        trainer = GeneratorTrainer(
            generator,
            config=TrainingConfig(require_overfit_check=True),
            collator=GraphCollator(pad_token_id=0),
            run_dir="/tmp/g2t-test-run",
        )
        with pytest.raises(RuntimeError, match="overfit check has not passed"):
            trainer.train([("stage", dataset)])


class TestGoldHoldout:
    """A Gold test item must never reach training."""

    def test_reserved_case_in_the_dataset_raises(self, builder) -> None:
        """The assertion the brief asks for, in the loader, not in anyone's memory."""
        reservation = GoldReservation(dataset="synthetic", case_ids=("synthetic-0002",))
        records = [
            {"case_id": "synthetic-0001", "serialised_facts": "a", "target_narrative": "x"},
            {"case_id": "synthetic-0002", "serialised_facts": "b", "target_narrative": "y"},
        ]
        with pytest.raises(ReservationError, match="reserved Gold test case"):
            GeneratorDataset(records, builder=builder, reservation=reservation)

    def test_clean_dataset_passes(self, builder, records) -> None:
        """A training set disjoint from the reservation is accepted."""
        reservation = GoldReservation(dataset="synthetic", case_ids=("held-out-9999",))
        dataset = GeneratorDataset(records[:4], builder=builder, reservation=reservation)
        assert len(dataset) == 4

    def test_assertion_names_the_offending_cases(self) -> None:
        """The error names what leaked, so the fix does not need a bisect."""
        reservation = GoldReservation(dataset="synthetic", case_ids=("a", "b"))
        with pytest.raises(ReservationError, match="'a'"):
            assert_no_gold_test(["a", "c"], reservation, where="a test")

    def test_evaluation_load_is_not_blocked(self, builder, records) -> None:
        """Reading a reserved case *as evaluation* is the point of reserving it."""
        reservation = GoldReservation(dataset="synthetic", case_ids=("synthetic-0000",))
        dataset = GeneratorDataset(
            records[:2], builder=builder, reservation=reservation, for_training=False
        )
        assert len(dataset) == 2


class TestCheckpointRoundTrip:
    """A checkpoint must restore the graph pathway, not only the adapters."""

    def test_fusion_and_encoder_survive_a_round_trip(self, tmp_path, builder, records) -> None:
        """Saving and reloading reproduces the fusion and encoder weights exactly."""
        generator = _generator(freeze_encoder=False)
        trainer = GeneratorTrainer(
            generator,
            config=TrainingConfig(require_overfit_check=False),
            collator=GraphCollator(pad_token_id=0),
            run_dir=tmp_path,
            arm="S1",
        )
        with torch.no_grad():
            for param in generator.fusion_projector.parameters():
                param.add_(0.5)
        before = {k: v.clone() for k, v in generator.fusion_projector.state_dict().items()}
        path = trainer.save_checkpoint(tag="test")

        restored = _generator(freeze_encoder=False)
        restored_trainer = GeneratorTrainer(
            restored,
            config=TrainingConfig(require_overfit_check=False),
            collator=GraphCollator(pad_token_id=0),
            run_dir=tmp_path,
            arm="S1",
        )
        restored_trainer.load_checkpoint(path)
        after = restored.fusion_projector.state_dict()
        for key, value in before.items():
            assert torch.allclose(value, after[key]), f"{key} did not round-trip"

    def test_checkpoint_carries_its_training_regime(self, tmp_path) -> None:
        """D-067: a checkpoint records how it was trained so a resume can refuse it."""
        generator = _generator()
        trainer = GeneratorTrainer(
            generator,
            config=TrainingConfig(epochs=3, require_overfit_check=False),
            collator=GraphCollator(pad_token_id=0),
            run_dir=tmp_path,
            arm="S1",
        )
        payload = torch.load(trainer.save_checkpoint(tag="t"), weights_only=False)
        assert payload["training_config"]["epochs"] == 3
        assert payload["arm"] == "S1"

    def test_resume_refuses_a_different_regime(self, tmp_path) -> None:
        """A smoke-run checkpoint cannot silently resume into a converged run's results."""
        generator = _generator()
        smoke = GeneratorTrainer(
            generator,
            config=TrainingConfig(epochs=1, require_overfit_check=False),
            collator=GraphCollator(pad_token_id=0),
            run_dir=tmp_path,
            arm="S1",
        )
        path = smoke.save_checkpoint(tag="smoke")

        full = GeneratorTrainer(
            _generator(),
            config=TrainingConfig(epochs=3, require_overfit_check=False),
            collator=GraphCollator(pad_token_id=0),
            run_dir=tmp_path,
            arm="S1",
        )
        with pytest.raises(ValueError, match="different regime"):
            full.load_checkpoint(path)

    def test_missing_fusion_state_is_refused(self, tmp_path) -> None:
        """Loading a text-only checkpoint into a fusion arm fails rather than half-loading."""
        generator = _generator()
        with pytest.raises(ValueError, match="disagree about whether"):
            generator.load_state_for_checkpoint({"fusion_state": None, "encoder_state": None})


class TestControlMatchesTreatment:
    """A1 may differ from S1 in which graph a narrative is paired with, and in nothing else."""

    @staticmethod
    def _pair(embedding_scale: float):
        """Build an S1 and an A1 generator over a backbone with a given embedding scale."""
        from g2t_aml.models.fusion import ShuffledGraphFusion

        built = []
        for shuffled in (False, True):
            torch.manual_seed(7)
            lm = StubCausalLM(hidden_size=64, vocab_size=512)
            with torch.no_grad():
                lm.get_input_embeddings().weight.mul_(embedding_scale)
            fusion = PrefixFusion(graph_dim=16, lm_dim=64, num_prefix_tokens=4, gated=False)
            built.append(
                (
                    Graph2TextGenerator(
                        language_model=lm,
                        fusion=ShuffledGraphFusion(fusion, seed=1) if shuffled else fusion,
                        encoder=_PooledEncoder(),
                        config=GeneratorConfig(max_seq_len=128),
                    ),
                    fusion,
                )
            )
        return built

    def test_the_control_is_given_the_treatment_arms_soft_token_scale(self) -> None:
        """Llama's embedding RMS is ~0.02, not ~1: an unscaled control is a different arm.

        The stub's embeddings are near unit scale, where a projector that was never scaled
        is indistinguishable from one that was. That is how A1 reached the real model with
        soft tokens tens of times larger than S1's (D-111).
        """
        (_, treatment), (_, control) = self._pair(embedding_scale=0.02)
        assert treatment.target_rms is not None and treatment.target_rms < 0.05
        assert control.target_rms == pytest.approx(treatment.target_rms)

    def test_the_two_arms_emit_soft_tokens_of_the_same_size(self) -> None:
        """Measured on the output, not on the attribute."""
        (s1, _), (a1, _) = self._pair(embedding_scale=0.02)
        pooled = torch.randn(4, 4, 16)
        for generator in (s1, a1):
            generator.eval()  # dropout off: the mean over a batch is then order-independent
        norms = [
            float(g.fusion_projector(pooled).token_norms.mean())  # type: ignore[misc]
            for g in (s1, a1)
        ]
        assert norms[1] == pytest.approx(norms[0], rel=1e-4)

    def test_a_control_over_an_unscalable_variant_says_so(self) -> None:
        """Skipping the scale silently is the failure; refusing is the fix."""
        from g2t_aml.models.fusion import ShuffledGraphFusion

        class _NoScale(torch.nn.Module):
            n_tokens = 4

        with pytest.raises(TypeError, match="set_target_rms"):
            ShuffledGraphFusion(_NoScale(), seed=1).set_target_rms(0.02)


class TestResume:
    """An interrupted run that is continued must be the run that was interrupted."""

    _CONFIG = TrainingConfig(
        epochs=1,
        per_device_batch_size=1,
        gradient_accumulation=2,
        save_every=None,
        resume_every=3,
        require_overfit_check=True,
    )
    _PASSED: ClassVar[dict[str, float]] = {"passed": 1.0, "initial_loss": 2.0, "final_loss": 0.01}

    @staticmethod
    def _dataset(builder, records) -> GeneratorDataset:
        chosen = records[:16]
        graphs = {r["case_id"]: _graph(3 + (i % 5)) for i, r in enumerate(chosen)}
        return GeneratorDataset(chosen, builder=builder, graphs=graphs)

    def _trainer(self, run_dir, *, arm: str = "S1", config: TrainingConfig | None = None):
        torch.manual_seed(1234)
        return GeneratorTrainer(
            _generator(freeze_encoder=False),
            config=config or self._CONFIG,
            collator=GraphCollator(pad_token_id=0),
            run_dir=run_dir,
            arm=arm,
        )

    def _interrupted_at_step_3(self, tmp_path, builder, records):
        """Train to completion, keeping the resume file as it stood after step 3."""
        trainer = self._trainer(tmp_path / "whole")
        kept = tmp_path / "S1_resume_step3.pt"
        write = trainer.save_resume_state

        def _keep_step_3():
            path = write()
            if trainer.state.step == 3:
                kept.write_bytes(path.read_bytes())
            return path

        trainer.save_resume_state = _keep_step_3  # type: ignore[method-assign]
        state = trainer.train(
            [("only", self._dataset(builder, records))], overfit_result=self._PASSED
        )
        return trainer, state, kept

    def test_a_resumed_run_ends_where_the_uninterrupted_one_does(
        self, tmp_path, builder, records
    ) -> None:
        """Same weights, bit for bit, as the run that was never stopped."""
        whole, whole_state, kept = self._interrupted_at_step_3(tmp_path, builder, records)
        assert whole_state.step == 8 and kept.is_file()

        resumed = self._trainer(tmp_path / "resumed")
        state = resumed.train([("only", self._dataset(builder, records))], resume_from=kept)

        assert state.step == whole_state.step
        assert state.resumed_at_steps == [3]
        expected = dict(whole.generator.named_parameters())
        for name, param in resumed.generator.named_parameters():
            if param.requires_grad:
                assert torch.equal(param, expected[name]), f"{name} diverged after the resume"

    def test_restarting_from_a_checkpoint_instead_is_refused(
        self, tmp_path, builder, records
    ) -> None:
        """A checkpoint has no optimiser state; training on from it is a different run."""
        trainer = self._trainer(tmp_path / "a")
        path = trainer.save_checkpoint(tag="step_3")
        fresh = self._trainer(tmp_path / "b")
        with pytest.raises(ValueError, match="not a resume file"):
            fresh.train(
                [("only", self._dataset(builder, records))],
                overfit_result=self._PASSED,
                resume_from=path,
            )

    def test_another_arms_resume_file_is_refused(self, tmp_path, builder, records) -> None:
        """S1's half-trained weights must never be finished off under A1's name."""
        _, _, kept = self._interrupted_at_step_3(tmp_path, builder, records)
        control = self._trainer(tmp_path / "a1", arm="A1")
        with pytest.raises(ValueError, match="belongs to arm"):
            control.train([("only", self._dataset(builder, records))], resume_from=kept)

    def test_a_changed_regime_is_refused(self, tmp_path, builder, records) -> None:
        """Everything but the write cadence has to match the run being continued."""
        from dataclasses import replace

        _, _, kept = self._interrupted_at_step_3(tmp_path, builder, records)
        for changed in (
            replace(self._CONFIG, gradient_accumulation=4),
            replace(self._CONFIG, seed=7),
            replace(self._CONFIG, fusion_lr=2e-4),
        ):
            other = self._trainer(tmp_path / "other", config=changed)
            with pytest.raises(ValueError, match="different regime"):
                other.train([("only", self._dataset(builder, records))], resume_from=kept)
        cadence = self._trainer(tmp_path / "cadence", config=replace(self._CONFIG, resume_every=1))
        assert (
            cadence.train([("only", self._dataset(builder, records))], resume_from=kept).step == 8
        )

    def test_a_resume_cannot_bypass_a_failed_overfit_check(
        self, tmp_path, builder, records
    ) -> None:
        """The file carries the result the run was admitted on, and only that result."""
        _, _, kept = self._interrupted_at_step_3(tmp_path, builder, records)
        payload = torch.load(kept, weights_only=False)
        assert payload["overfit_result"]["passed"] == 1.0
        payload["overfit_result"] = None
        torch.save(payload, kept)
        with pytest.raises(RuntimeError, match="overfit check has not passed"):
            self._trainer(tmp_path / "x").train(
                [("only", self._dataset(builder, records))], resume_from=kept
            )


class TestCurriculum:
    """The curriculum is data, so the ablation is a config change."""

    def test_default_is_bronze_silver_then_silver(self) -> None:
        """The brief's default: mixed for epoch 1, Silver for epochs 2-3."""
        stages = build_curriculum(None)
        assert [s.name for s in stages] == ["mixed", "silver_only"]
        assert stages[0].tiers == ("bronze", "silver")
        assert stages[1].tiers == ("silver",)

    def test_a_stage_with_no_tiers_is_refused(self) -> None:
        """A stage naming no tiers would train on nothing and report a healthy loss."""
        with pytest.raises(ValueError, match="names no tiers"):
            build_curriculum([{"name": "empty", "tiers": []}])

    def test_gold_tail_can_be_step_capped(self) -> None:
        """The optional Gold-train tail is expressed as a step cap, as the brief asks."""
        stages = build_curriculum(
            [{"name": "gold_tail", "tiers": ["gold"], "epochs": 1, "max_steps": 200}]
        )
        assert stages[0].max_steps == 200


class TestTextModes:
    """The arms differ in what text accompanies the graph."""

    def test_graph_only_mode_omits_the_facts(self, tokenizer, records) -> None:
        """S2's text_mode=none gives the model no serialised facts at all."""
        builder = PromptBuilder(
            tokenizer, n_soft_tokens=4, soft_token_id=3, text_mode="none", max_seq_len=128
        )
        prompt = builder.build(records[0])
        assert "Case" not in tokenizer.decode(prompt.input_ids)
        assert sum(prompt.soft_mask) == 4

    def test_text_only_mode_reserves_no_soft_tokens(self, tokenizer, records) -> None:
        """B7 is text-only: no graph positions are reserved."""
        builder = PromptBuilder(
            tokenizer, n_soft_tokens=0, soft_token_id=3, text_mode="serialised", max_seq_len=128
        )
        prompt = builder.build(records[0])
        assert sum(prompt.soft_mask) == 0

    def test_graph_only_without_soft_tokens_is_refused(self, tokenizer) -> None:
        """No facts and no graph would train the model to invent a whole narrative."""
        with pytest.raises(ValueError, match="no case information"):
            PromptBuilder(tokenizer, n_soft_tokens=0, soft_token_id=3, text_mode="none")

    def test_truncation_removes_facts_not_the_narrative(self, tokenizer) -> None:
        """Truncation eats the facts; the completion survives intact."""
        builder = PromptBuilder(
            tokenizer, n_soft_tokens=2, soft_token_id=3, text_mode="full", max_seq_len=80
        )
        record = {
            "case_id": "synthetic-0001",
            "serialised_facts": " ".join(f"field{i}={i}" for i in range(100)),
            "target_narrative": "the account dispersed funds rapidly to nine counterparties",
        }
        prompt = builder.build(record)
        assert prompt.n_facts_truncated > 0
        assert len(prompt.input_ids) <= 80
        supervised = [label for label in prompt.labels if label != -100]
        assert len(supervised) == len(tokenizer.encode(record["target_narrative"])) + 1


class TestOptimizerSelection:
    """Phase 14: the paged optimiser must not be selected over CPU tensors.

    `PagedAdamW8bit` constructs happily over CPU parameters and only raises
    `ValueError: Expected a cuda device, but got: cpu` at the first `.step()`, so the
    failure lands mid-training rather than at setup. Guarding on `bitsandbytes` being
    importable is not enough, and neither is `torch.cuda.is_available()` -- a host can have
    a working CUDA runtime while the model sits on CPU, which is every CPU test in this
    repository on a laptop that does have a card.
    """

    @staticmethod
    def _cpu_generator() -> Any:
        """A minimal object exposing the one method `build_optimizer` calls."""

        class _Stub:
            def __init__(self) -> None:
                self.weight = torch.nn.Parameter(torch.zeros(2, 2))

            def trainable_parameter_groups(
                self, *, lora_lr: float, fusion_lr: float, encoder_lr: float
            ) -> list[dict[str, Any]]:
                return [{"params": [self.weight], "lr": lora_lr}]

        return _Stub()

    def test_cpu_parameters_fall_back_to_adamw(self) -> None:
        """The whole point: no CUDA tensors, no paged optimiser."""
        cfg = TrainingConfig(optim="paged_adamw_8bit")
        optimizer = build_optimizer(self._cpu_generator(), cfg)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert type(optimizer).__name__ == "AdamW"

    def test_the_fallback_is_logged_and_names_its_reason(self) -> None:
        """A silent substitution changes the memory profile Phase 13 reports."""
        warnings: list[tuple[Any, ...]] = []

        class _Log:
            def warning(self, *args: Any) -> None:
                warnings.append(args)

        build_optimizer(self._cpu_generator(), TrainingConfig(optim="paged_adamw_8bit"), log=_Log())
        assert warnings, "the substitution must be logged, never silent"
        rendered = " ".join(str(a) for a in warnings[0])
        assert "CPU" in rendered or "cpu" in rendered
        assert "AdamW" in rendered

    def test_an_explicitly_configured_adamw_is_left_alone(self) -> None:
        """The guard applies to the paged request only; it does not override a choice."""
        optimizer = build_optimizer(self._cpu_generator(), TrainingConfig(optim="adamw"))
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_no_trainable_parameters_is_an_error_not_a_silent_no_op(self) -> None:
        """Every component frozen means the run produces nothing; say so loudly."""

        class _Frozen:
            def trainable_parameter_groups(self, **_: float) -> list[dict[str, Any]]:
                return []

        with pytest.raises(ValueError, match="no trainable parameters"):
            build_optimizer(_Frozen(), TrainingConfig())
