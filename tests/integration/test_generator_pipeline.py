"""End to end on a stub: train a few steps, generate, resume, and guard the output.

The unit tests check each component in isolation. This checks that they compose — that a
model trained through the trainer can be generated from through the inference path, that an
interrupted test-set run resumes without gaps or duplicates, and that the diagnostics the
whole phase turns on actually populate.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tests.stubs import StubCausalLM, StubTokenizer  # noqa: E402

from g2t_aml.models.fusion import PrefixFusion  # noqa: E402
from g2t_aml.models.generator import (  # noqa: E402
    GenerationConfig,
    GeneratorConfig,
    GeneratorDataset,
    Graph2TextGenerator,
    GraphCollator,
    PromptBuilder,
    TrainingConfig,
    generate_batch,
    run_test_set,
)
from g2t_aml.models.generator.profiling import RunProfile, device_info, profile_phase  # noqa: E402
from g2t_aml.models.generator.train import GeneratorTrainer  # noqa: E402
from g2t_aml.utils.io import read_jsonl  # noqa: E402


@pytest.fixture
def setup():
    """Assemble a stub generator, dataset and collator.

    Returns:
        ``(generator, dataset, collator, tokenizer)``.
    """
    from torch_geometric.data import Data

    torch.manual_seed(0)
    tokenizer = StubTokenizer()
    builder = PromptBuilder(
        tokenizer, n_soft_tokens=4, soft_token_id=3, text_mode="full", max_seq_len=96
    )
    records = [
        {
            "case_id": f"synthetic-{i:04d}",
            "serialised_facts": f"n_nodes={i + 3} typology=fan_out",
            "target_narrative": f"account {i} dispersed funds to {i + 2} counterparties",
        }
        for i in range(8)
    ]
    graphs = {
        r["case_id"]: Data(
            x=torch.randn(3 + i % 4, 8),
            edge_index=torch.stack([torch.arange(2 + i % 4), torch.arange(1, 3 + i % 4)]),
        )
        for i, r in enumerate(records)
    }
    generator = Graph2TextGenerator(
        language_model=StubCausalLM(hidden_size=32, vocab_size=512),
        fusion=PrefixFusion(graph_dim=16, lm_dim=32, num_prefix_tokens=4),
        encoder=_encoder(),
        config=GeneratorConfig(max_seq_len=96, freeze_encoder=False),
    )
    dataset = GeneratorDataset(records, builder=builder, graphs=graphs)
    return generator, dataset, GraphCollator(pad_token_id=0), tokenizer, builder, records, graphs


def _encoder():
    """Build a small encoder stub.

    Returns:
        The encoder.
    """
    from types import SimpleNamespace

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.project = torch.nn.Linear(8, 16)
            self.hidden_dim = 16

        def forward(self, batch):
            pooled = torch.zeros(int(batch.num_graphs), 16)
            pooled.index_add_(0, batch.batch, self.project(batch.x))
            return SimpleNamespace(pooled_tokens=pooled.unsqueeze(1).expand(-1, 4, -1).contiguous())

    return Encoder()


class TestTrainingRun:
    """The trainer drives a curriculum and produces a history."""

    def test_two_stage_curriculum_runs_and_checkpoints(self, setup, tmp_path) -> None:
        """A curriculum of two stages trains, logs and writes checkpoints."""
        generator, dataset, collator, *_ = setup
        trainer = GeneratorTrainer(
            generator,
            config=TrainingConfig(
                epochs=1,
                per_device_batch_size=2,
                gradient_accumulation=1,
                eval_every=2,
                save_every=None,
                require_overfit_check=False,
            ),
            collator=collator,
            run_dir=tmp_path,
            arm="S1",
        )
        state = trainer.train([("mixed", dataset), ("silver_only", dataset)])

        assert state.step > 0
        assert (tmp_path / "S1_final.pt").is_file()
        assert (tmp_path / "S1_stage_mixed.pt").is_file()

        history = list(read_jsonl(tmp_path / "history_S1.jsonl"))
        assert history
        assert all(row["arm"] == "S1" for row in history)

    def test_history_carries_the_gate_and_per_group_norms(self, setup, tmp_path) -> None:
        """The diagnostics the phase turns on are actually populated, not merely defined."""
        generator, dataset, collator, *_ = setup
        trainer = GeneratorTrainer(
            generator,
            config=TrainingConfig(
                epochs=1,
                per_device_batch_size=2,
                gradient_accumulation=1,
                eval_every=1,
                save_every=None,
                require_overfit_check=False,
            ),
            collator=collator,
            run_dir=tmp_path,
            arm="S1",
        )
        trainer.train([("mixed", dataset)])
        row = next(iter(read_jsonl(tmp_path / "history_S1.jsonl")))

        assert 0.0 < row["gate_mean"] < 1.0
        assert row["grad_norm/fusion"] > 0
        assert row["grad_norm/encoder"] > 0
        assert row["pre_gate_rms"] > 0
        assert row["n_supervised_tokens"] > 0

    def test_gate_does_not_collapse_over_a_short_run(self, setup, tmp_path) -> None:
        """The Phase 9 acceptance criterion, checked as a property of the trace."""
        generator, dataset, collator, *_ = setup
        trainer = GeneratorTrainer(
            generator,
            config=TrainingConfig(
                epochs=2,
                per_device_batch_size=2,
                gradient_accumulation=1,
                eval_every=1,
                save_every=None,
                require_overfit_check=False,
            ),
            collator=collator,
            run_dir=tmp_path,
            arm="S1",
        )
        trainer.train([("mixed", dataset)])
        gates = [r["gate_mean"] for r in read_jsonl(tmp_path / "history_S1.jsonl")]
        assert min(gates) > 0.01, f"the gate collapsed: {gates}"


class TestGeneration:
    """The inference path."""

    def test_generates_one_narrative_per_case(self, setup) -> None:
        """Greedy decoding returns exactly one text per case, in batch order."""
        generator, _, collator, tokenizer, builder, records, graphs = setup
        eval_set = GeneratorDataset(records[:3], builder=builder, graphs=graphs, for_training=False)
        batch = collator([eval_set[i] for i in range(3)])
        results = generate_batch(
            generator,
            batch,
            GenerationConfig.deterministic(max_new_tokens=6),
            tokenizer=tokenizer,
        )
        assert [r.case_id for r in results] == list(batch.case_ids)
        assert all(len(r.texts) == 1 for r in results)

    def test_sampling_returns_the_requested_candidate_count(self, setup) -> None:
        """The guard's four candidates arrive as four texts per case."""
        generator, _, collator, tokenizer, builder, records, graphs = setup
        eval_set = GeneratorDataset(records[:2], builder=builder, graphs=graphs, for_training=False)
        batch = collator([eval_set[i] for i in range(2)])
        results = generate_batch(
            generator,
            batch,
            GenerationConfig(max_new_tokens=5, do_sample=True, num_return_sequences=4),
            tokenizer=tokenizer,
        )
        assert all(len(r.texts) == 4 for r in results)

    def test_generating_from_a_training_batch_is_refused(self, setup) -> None:
        """A batch carrying labels would condition generation on the answer."""
        generator, dataset, collator, tokenizer, *_ = setup
        batch = collator([dataset[0], dataset[1]])
        with pytest.raises(ValueError, match="carries training labels"):
            generate_batch(generator, batch, GenerationConfig.deterministic(), tokenizer=tokenizer)

    def test_greedy_decoding_is_reproducible(self, setup) -> None:
        """Two greedy runs over the same input agree exactly, as a measurement must."""
        generator, _, collator, tokenizer, builder, records, graphs = setup
        eval_set = GeneratorDataset(records[:2], builder=builder, graphs=graphs, for_training=False)
        batch = collator([eval_set[0], eval_set[1]])
        cfg = GenerationConfig.deterministic(max_new_tokens=6)
        first = generate_batch(generator, batch, cfg, tokenizer=tokenizer)
        second = generate_batch(generator, batch, cfg, tokenizer=tokenizer)
        assert [r.texts for r in first] == [r.texts for r in second]


class TestResumability:
    """An interrupted test-set run resumes without gaps or duplicates."""

    def test_resume_skips_completed_cases(self, setup, tmp_path) -> None:
        """Every case appears exactly once across an interrupted and resumed run."""
        generator, _, collator, tokenizer, builder, records, graphs = setup
        eval_set = GeneratorDataset(records, builder=builder, graphs=graphs, for_training=False)
        items = [eval_set[i] for i in range(len(eval_set))]
        path = tmp_path / "generations.jsonl"
        cfg = GenerationConfig.deterministic(max_new_tokens=4)

        run_test_set(
            generator,
            items[:4],
            collator=collator,
            tokenizer=tokenizer,
            cfg=cfg,
            output_path=path,
            batch_size=2,
        )
        assert len(list(read_jsonl(path))) == 4

        run_test_set(
            generator,
            items,
            collator=collator,
            tokenizer=tokenizer,
            cfg=cfg,
            output_path=path,
            batch_size=2,
        )
        written = list(read_jsonl(path))
        ids = [r["case_id"] for r in written]
        assert len(ids) == len(set(ids)) == len(records)


class TestProfiling:
    """VRAM and throughput, captured now because Phase 13 needs it."""

    def test_phase_profile_records_time_and_throughput(self) -> None:
        """A measured phase reports seconds and a tokens-per-second rate."""
        with profile_phase("train", arm="S1", device="cpu") as profile:
            profile.n_tokens = 1000
            profile.n_examples = 10
        assert profile.seconds > 0
        assert profile.tokens_per_second > 0
        assert profile.examples_per_second > 0

    def test_profile_survives_an_exception(self) -> None:
        """A run that died still yields its measurement; Phase 13 wants the OOM point."""
        profile = None
        with (
            pytest.raises(RuntimeError),
            profile_phase("train", arm="S1", device="cpu") as p,
        ):
            profile = p
            raise RuntimeError("boom")
        assert profile is not None and profile.seconds > 0

    def test_run_profile_records_deviations(self, tmp_path) -> None:
        """A memory-forced hyperparameter change is recorded where it will be read."""
        run = RunProfile(arm="S1", device=device_info("cpu"))
        run.note_deviation("max_seq_len 2048 -> 1536, OOM at batch size 2")
        with profile_phase("train", arm="S1", device="cpu") as profile:
            profile.n_tokens = 10
        run.add(profile)
        written = run.write(tmp_path / "profile.json")
        assert written.is_file()
        assert run.to_dict()["deviations"]
        assert run.to_dict()["phases"][0]["phase"] == "train"
