"""Write-back tests: `model_signal` is additive, and the no-encoder baseline stays clean.

The second of those is the one that matters. `facts.serialiser._compact` emits
`gnn_risk_score`, and that string is stored as `serialised_facts` on every training
record. Once Phase 7 populates the block, regenerating Bronze would push the encoder's own
risk score into the **serialisation baseline** — the "flatten the facts, no graph encoder"
ablation arm — and it would stop being a control without anything failing. This module
pins the invariant so a future regeneration cannot do that silently. See DECISIONS.md
D-063.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from g2t_aml.facts.schema import CaseFacts, ModelSignal, facts_to_dict, validate_facts
from g2t_aml.facts.serialiser import serialise_facts

REPO_ROOT = Path(__file__).resolve().parents[2]
BRONZE = REPO_ROOT / "data" / "processed" / "amlworld_hi_small" / "corpus" / "bronze.jsonl"
GOLDEN = REPO_ROOT / "tests" / "golden" / "case_facts"


def _golden_facts() -> CaseFacts:
    from g2t_aml.corpus.factsio import load_case_facts_file

    candidates = sorted(GOLDEN.glob("*.json"))
    if not candidates:
        pytest.skip("no golden fact records in this checkout")
    return load_case_facts_file(candidates[0])


def _signal() -> ModelSignal:
    return ModelSignal(
        gnn_risk_score=0.8734,
        score_percentile=97.2,
        top_contributing_nodes=(("BANK|ACCT001", 0.41), ("BANK|ACCT002", 0.22)),
        model_version="gatv2-seed42-epoch37",
    )


# ------------------------------------------------------------------ additive ---


def test_write_back_produces_a_new_record_and_leaves_the_original_alone():
    """The frozen record is replaced, never mutated: another phase may already have hashed it."""
    facts = _golden_facts()
    assert facts.model_signal.gnn_risk_score is None

    scored = facts.with_model_signal(_signal())
    assert scored.model_signal.gnn_risk_score == pytest.approx(0.8734)
    assert facts.model_signal.gnn_risk_score is None
    assert scored is not facts


def test_a_scored_record_still_validates_against_the_frozen_schema():
    """Populating the block must not require a schema bump (invariant 9)."""
    scored = _golden_facts().with_model_signal(_signal())
    payload = facts_to_dict(scored)
    validate_facts(payload)
    assert payload["schema_version"] == scored.schema_version
    assert payload["model_signal"]["model_version"] == "gatv2-seed42-epoch37"
    assert len(payload["model_signal"]["top_contributing_nodes"]) == 2


def test_write_back_changes_nothing_outside_the_model_signal_block():
    """Additive means additive: every other field is byte-identical."""
    facts = _golden_facts()
    before = facts_to_dict(facts)
    after = facts_to_dict(facts.with_model_signal(_signal()))
    del before["model_signal"], after["model_signal"]
    assert json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True)


# --------------------------------------------------------- baseline hygiene ---


def test_the_compact_serialisation_does_carry_the_score_once_populated():
    """Establishes the premise of the next test: the contamination route is real."""
    facts = _golden_facts()
    assert "gnn_risk_score=none" in serialise_facts(facts, style="compact")
    scored = serialise_facts(facts.with_model_signal(_signal()), style="compact")
    assert "gnn_risk_score=0.8734" in scored


@pytest.mark.skipif(not BRONZE.is_file(), reason="the Bronze corpus is not built here")
def test_bronze_serialised_facts_carry_no_model_signal():
    """The no-encoder ablation arm must stay free of the encoder's output.

    `serialised_facts` is the input to the serialisation baseline. If a Bronze
    regeneration after Phase 7's write-back put a real `gnn_risk_score` in there, the
    baseline would be reading the encoder it exists to be compared against, and every
    "graph fusion beats flattened facts" number would be measuring the wrong thing.
    """
    checked = 0
    with BRONZE.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            assert "gnn_risk_score=none" in record["serialised_facts"], (
                f"{record['case_id']}: the serialisation baseline carries a model score. "
                "Bronze was regenerated after the Phase 7 write-back; that contaminates "
                "the no-encoder ablation arm. See DECISIONS.md D-063."
            )
            checked += 1
            if checked >= 500:
                break
    assert checked > 0


@pytest.mark.skipif(not BRONZE.is_file(), reason="the Bronze corpus is not built here")
def test_bronze_narratives_never_mention_a_model_score():
    """No template reads `model_signal`, which is why the corpus needs no regeneration."""
    with BRONZE.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            record = json.loads(line)
            text = record["target_narrative"].lower()
            for phrase in ("risk score", "model risk", "percentile", "gnn"):
                assert phrase not in text, f"{record['case_id']} mentions {phrase!r}"
            if i >= 500:
                break
