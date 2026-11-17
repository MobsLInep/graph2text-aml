"""Guarded decoding over a split (D-116): candidates, repairs, selection.

The guard's selection logic is tested in ``test_generator_guard.py``. These tests cover the
orchestration around it: that every case comes out once, in order, with the guard's
verdict beside it; that the repair pass is run for exactly the cases the guard would have
asked to repair; and that an interrupted decode resumes from its files. Candidate and
repair texts are written into the pass files beforehand where the test needs to control
them — ``run_test_set`` skips cases already present, which is the resume mechanism itself.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tests import factories  # noqa: E402
from tests.stubs import StubCausalLM, StubTokenizer  # noqa: E402

from g2t_aml.corpus.bronze.renderer import render_bronze  # noqa: E402
from g2t_aml.corpus.silver.claim_extraction import SlotAlignmentExtractor  # noqa: E402
from g2t_aml.facts.extractor import extract_facts  # noqa: E402
from g2t_aml.facts.vocab import load_vocabulary  # noqa: E402
from g2t_aml.models.generator import Graph2TextGenerator  # noqa: E402
from g2t_aml.models.generator.dataset import GeneratorDataset, GraphCollator  # noqa: E402
from g2t_aml.models.generator.guarded_decode import (  # noqa: E402
    CANDIDATES_FILE,
    REPAIRS_FILE,
    GuardDecodeConfig,
    guarded_test_set,
    repair_prompt,
)
from g2t_aml.models.generator.model import GeneratorConfig  # noqa: E402
from g2t_aml.models.generator.prompts import PromptBuilder  # noqa: E402
from g2t_aml.utils.io import read_jsonl, write_jsonl  # noqa: E402

GUILT = " The subject is guilty of money laundering."


@pytest.fixture(scope="module")
def cases():
    """Three real fact records, their Bronze texts and extractors."""
    vocabulary = load_vocabulary()
    built = []
    for index, width in enumerate((4, 5, 6)):
        facts = extract_facts(
            factories.as_laundering_stream(factories.fan_out_case(width=width), "fan_out")
        )
        bronze = render_bronze(facts, vocabulary=vocabulary)
        record = {
            "case_id": f"synthetic-{index:04d}",
            "serialised_facts": f"fan-out case with {width} counterparties",
        }
        built.append((record, facts, bronze, SlotAlignmentExtractor(bronze, vocabulary=vocabulary)))
    return built


@pytest.fixture
def setup(cases):
    """A text-only stub generator and everything the decode needs."""
    tokenizer = StubTokenizer()
    builder = PromptBuilder(
        tokenizer, n_soft_tokens=0, soft_token_id=0, text_mode="serialised", max_seq_len=512
    )
    generator = Graph2TextGenerator(
        language_model=StubCausalLM(hidden_size=32, vocab_size=512),
        config=GeneratorConfig(text_mode="serialised", use_fusion=False, max_seq_len=512),
    )
    records = [c[0] for c in cases]
    items = [
        GeneratorDataset(records, builder=builder, graphs=None, for_training=False)[i]
        for i in range(len(records))
    ]
    inputs = {c[0]["case_id"]: (c[1], c[3]) for c in cases}
    return generator, tokenizer, builder, records, items, inputs


def _run(setup, run_dir, **overrides):
    generator, tokenizer, builder, records, items, inputs = setup
    config = GuardDecodeConfig(max_new_tokens=6, batch_size=2, **overrides)
    return guarded_test_set(
        generator,
        records,
        items,
        builder=builder,
        collator=GraphCollator(pad_token_id=0),
        tokenizer=tokenizer,
        config=config,
        case_inputs=inputs.__getitem__,
        run_dir=run_dir,
    )


def test_every_case_comes_out_once_in_order_with_its_verdict(setup, tmp_path) -> None:
    rows, stats = _run(setup, tmp_path)
    assert [r["case_id"] for r in rows] == [r["case_id"] for r in setup[3]]
    assert all(r["guard"] is True and len(r["texts"]) == 1 for r in rows)
    assert all(r["n_candidates"] == 4 for r in rows)
    assert stats.n_cases == len(rows)
    candidates = list(read_jsonl(tmp_path / CANDIDATES_FILE))
    assert all(len(c["texts"]) == 4 for c in candidates)


def test_the_best_verified_candidate_is_emitted(setup, cases, tmp_path) -> None:
    """A dirty first candidate loses to a clean later one; the first is kept beside it."""
    write_jsonl(
        tmp_path / CANDIDATES_FILE,
        [
            {"case_id": rec["case_id"], "texts": [bronze.text + GUILT, bronze.text, "x", "y"]}
            for rec, _facts, bronze, _ex in cases
        ],
    )
    rows, stats = _run(setup, tmp_path)
    for row, (_rec, _facts, bronze, _ex) in zip(rows, cases, strict=True):
        assert row["texts"] == [bronze.text]
        assert row["unguarded_first_candidate"] == bronze.text + GUILT
        assert row["guard_warning"] is None
    assert stats.n_selection_changed == len(cases)
    assert not (tmp_path / REPAIRS_FILE).exists(), "nothing needed repairing"


def test_only_the_cases_the_guard_cannot_clean_are_repaired(setup, cases, tmp_path) -> None:
    rec0, _f0, bronze0, _e0 = cases[0]
    write_jsonl(
        tmp_path / CANDIDATES_FILE,
        [{"case_id": rec0["case_id"], "texts": [bronze0.text + GUILT] * 4}]
        + [
            {"case_id": rec["case_id"], "texts": [bronze.text] * 4}
            for rec, _facts, bronze, _ex in cases[1:]
        ],
    )
    # The repair for the dirty case, as the repair pass would have written it.
    write_jsonl(tmp_path / REPAIRS_FILE, [{"case_id": rec0["case_id"], "texts": [bronze0.text]}])
    rows, stats = _run(setup, tmp_path)
    assert rows[0]["guard_regenerated"] is True
    assert rows[0]["texts"] == [bronze0.text]
    assert rows[0]["guard_warning"] is None
    assert [r["guard_regenerated"] for r in rows[1:]] == [False, False]
    assert stats.n_regenerated == 1 and stats.n_regeneration_helped == 1
    assert [r["case_id"] for r in read_jsonl(tmp_path / REPAIRS_FILE)] == [rec0["case_id"]]


def test_with_regeneration_off_a_dirty_case_is_emitted_with_a_warning(setup, cases, tmp_path):
    write_jsonl(
        tmp_path / CANDIDATES_FILE,
        [{"case_id": rec["case_id"], "texts": [b.text + GUILT] * 4} for rec, _f, b, _e in cases],
    )
    rows, stats = _run(setup, tmp_path, allow_regeneration=False)
    assert all(r["guard_warning"] is not None for r in rows)
    assert stats.n_warned == len(cases) and stats.n_regenerated == 0
    assert not (tmp_path / REPAIRS_FILE).exists()


def test_the_repair_prompt_extends_the_original_and_supervises_nothing(setup, cases) -> None:
    _generator, _tokenizer, builder, records, _items, _inputs = setup
    base = builder.build(records[0], for_training=False)
    repaired = repair_prompt(builder, records[0], "draft text", "fix the count")
    assert repaired.input_ids[: len(base.input_ids)] == base.input_ids
    assert len(repaired.input_ids) > len(base.input_ids)
    assert len(repaired.soft_mask) == len(repaired.input_ids) == len(repaired.labels)
    assert set(repaired.labels) == {-100}
    assert repaired.case_id == base.case_id
