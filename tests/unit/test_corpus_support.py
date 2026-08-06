"""The supporting machinery: fact I/O, claim parsing, dedup, diversity, tokenisation, PII.

Small modules, but three of them carry a load-bearing property that is easy to lose and
invisible when lost: the fact deserialiser must be exactly lossless, claims must be parsed
from the *text* rather than read from the record, and the dedup gate must be
order-independent.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from tests import factories

from g2t_aml.corpus import dedupe, diversity
from g2t_aml.corpus.bronze.renderer import render_bronze
from g2t_aml.corpus.claims import ClaimParseError, claim_from_slot, claims_from_slots
from g2t_aml.corpus.factsio import FactsIOError, facts_from_dict, load_case_facts
from g2t_aml.corpus.pii import scan_for_identifiers
from g2t_aml.corpus.record import SlotAnnotation, validate_training_record
from g2t_aml.corpus.tokenization import (
    DEFAULT_TOKENIZER,
    TokenizerUnavailableError,
    get_token_counter,
    word_count,
)
from g2t_aml.facts.extractor import extract_facts
from g2t_aml.facts.schema import Unavailable, facts_to_dict
from g2t_aml.facts.vocab import load_vocabulary

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "case_facts"


@pytest.fixture(scope="module")
def vocabulary():
    return load_vocabulary()


class TestFactsIO:
    """The deserialiser Bronze reads its input through."""

    @pytest.mark.parametrize("path", sorted(GOLDEN.glob("*.json")), ids=lambda p: p.stem)
    def test_every_golden_record_round_trips_exactly(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert facts_to_dict(load_case_facts(payload)) == payload

    def test_a_lossy_read_is_refused_rather_than_returned(self) -> None:
        """The assertion that stops a corpus verifying against a record nobody extracted.

        A field the deserialiser does not read is a field the narrative would be rendered
        without and then verified without, so it must fail loudly rather than vanish.
        """
        payload = json.loads((GOLDEN / "fan_out.json").read_text(encoding="utf-8"))
        smuggled = copy.deepcopy(payload)
        smuggled["flow"]["a_field_the_reader_does_not_know_about"] = 42
        assert facts_to_dict(facts_from_dict(smuggled)) != smuggled
        with pytest.raises(FactsIOError, match="lossy"):
            load_case_facts(smuggled)

    def test_a_stale_schema_version_is_refused(self) -> None:
        payload = json.loads((GOLDEN / "fan_out.json").read_text(encoding="utf-8"))
        with pytest.raises(FactsIOError, match="frozen at"):
            load_case_facts({**payload, "schema_version": "0.9.0"})

    def test_a_measured_null_stays_a_measured_null(self) -> None:
        """D-025: `None` and a sentinel mean different things to the checker."""
        payload = json.loads((GOLDEN / "flat.json").read_text(encoding="utf-8"))
        facts = load_case_facts(payload)
        assert facts.motifs.cycle.descriptors["length"] is None
        assert not isinstance(facts.motifs.cycle.descriptors["length"], Unavailable)

    def test_a_sentinel_stays_a_sentinel(self) -> None:
        payload = json.loads((GOLDEN / "elliptic2.json").read_text(encoding="utf-8"))
        facts = load_case_facts(payload)
        assert isinstance(facts.flow, Unavailable)
        assert facts.flow.reason == "substrate_has_no_monetary_amounts"

    def test_cross_border_must_be_a_sentinel(self) -> None:
        """D-030: permanently unavailable on every substrate."""
        payload = json.loads((GOLDEN / "fan_out.json").read_text(encoding="utf-8"))
        payload["flow"]["cross_border"] = True
        with pytest.raises(FactsIOError, match="cross_border"):
            load_case_facts(payload)


class TestClaimsAreParsedFromTheText:
    """The anti-circularity property. If this breaks, every corpus scores 100%."""

    def test_a_claim_reads_the_narrative_not_the_raw_value(self) -> None:
        slot = SlotAnnotation(
            field_path="structure.n_nodes",
            span=(0, 2),
            rendered_value="14",
            raw_value=999,  # deliberately disagrees with the text
            claim_type="numeric",
        )
        claim = claim_from_slot(slot, "14 accounts")
        assert claim.value == 14, "the claim must come from the text, not from raw_value"

    def test_a_drifted_span_is_an_error_rather_than_a_silent_realignment(self) -> None:
        slot = SlotAnnotation(
            field_path="structure.n_nodes",
            span=(0, 2),
            rendered_value="14",
            raw_value=14,
            claim_type="numeric",
        )
        with pytest.raises(ClaimParseError, match="span"):
            claim_from_slot(slot, "99 accounts")

    def test_unparseable_text_raises_rather_than_becoming_unverifiable(self) -> None:
        slot = SlotAnnotation(
            field_path="flow.total_inflow",
            span=(0, 6),
            rendered_value="lots!!",
            raw_value=1.0,
            claim_type="numeric",
        )
        with pytest.raises(ClaimParseError):
            claim_from_slot(slot, "lots!! of money")

    def test_display_maps_invert(self, vocabulary) -> None:
        facts = extract_facts(factories.fan_out_case(width=5))
        narrative = render_bronze(facts, vocabulary=vocabulary)
        claims = {c.field_path: c.value for c in claims_from_slots(narrative.slots, narrative.text)}
        assert claims["focal_entity.role"] == facts.focal_entity.role


class TestDedupe:
    def test_identical_narratives_are_caught(self) -> None:
        text = "The account received funds from nine counterparties over twelve hours."
        report = dedupe.find_near_duplicates({"a": text, "b": text})
        assert report.n_dropped == 1
        assert report.dropped == ("b",)

    def test_unrelated_narratives_are_kept(self) -> None:
        report = dedupe.find_near_duplicates(
            {
                "a": "The account received funds from nine counterparties over twelve hours.",
                "b": "Bipartite structure separates four accounts from six with no internal edges.",
            }
        )
        assert report.n_dropped == 0

    def test_the_kept_record_is_order_independent(self) -> None:
        text = "The account received funds from nine counterparties over twelve hours."
        forward = dedupe.find_near_duplicates({"a": text, "b": text})
        reverse = dedupe.find_near_duplicates({"b": text, "a": text})
        assert forward.dropped == reverse.dropped == ("b",)

    def test_the_threshold_is_applied_on_exact_jaccard(self) -> None:
        left = frozenset({"a b c d e", "b c d e f"})
        right = frozenset({"a b c d e"})
        assert dedupe.jaccard(left, right) == pytest.approx(0.5)

    def test_bands_must_divide_the_signature(self) -> None:
        with pytest.raises(ValueError, match="must divide"):
            dedupe.find_near_duplicates({"a": "x"}, n_permutations=128, bands=7)

    def test_shingles_keep_the_numbers_that_distinguish_cases(self) -> None:
        members = dedupe.shingles("the account received 9 counterparties over 12 hours here")
        assert any("9" in shingle for shingle in members)


class TestDiversity:
    def test_a_collapsed_corpus_scores_high(self) -> None:
        corpus = ["the account received funds from nine counterparties over twelve hours"] * 40
        assert diversity.self_bleu(corpus) > 0.9
        assert diversity.distinct_n(corpus, 3) < 0.05

    def test_a_varied_corpus_scores_low(self) -> None:
        words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet".split()
        corpus = [
            " ".join(words[(i + j) % len(words)] + str(i * j) for j in range(12)) for i in range(40)
        ]
        assert diversity.self_bleu(corpus) < 0.2

    def test_self_bleu_rises_with_the_reference_count(self) -> None:
        """The saturation that made the raw number misleading (D-043)."""
        corpus = [
            f"the subject account moved value to {i} recipients within a short window"
            for i in range(60)
        ]
        curve = diversity.self_bleu_curve(corpus)
        assert curve[1] <= curve[10] <= curve[50]

    def test_skeletons_blank_the_slot_values(self) -> None:
        blanked = diversity.skeletons(["there were 14 accounts"], [[(11, 13)]])
        assert blanked == ["there were \N{BULLET} accounts"]

    def test_two_narratives_differing_only_in_values_share_a_skeleton(self) -> None:
        blanked = diversity.skeletons(
            ["there were 14 accounts", "there were 9 accounts"], [[(11, 13)], [(11, 12)]]
        )
        assert blanked[0] == blanked[1]

    def test_the_report_serialises_and_summarises(self) -> None:
        report = diversity.measure_diversity(
            ["one narrative here about accounts", "another narrative about transfers"],
            families=["a", "b"],
            typologies=["fan_out", "fan_in"],
            variants=[0, 1],
            slot_spans=[[], []],
        )
        assert report.to_dict()["n_records"] == 2
        assert "diversity over" in report.summary()


class TestTokenization:
    def test_the_default_counter_is_deterministic(self) -> None:
        counter = get_token_counter()
        assert counter.name == DEFAULT_TOKENIZER
        assert counter.count("the account received funds") == counter.count(
            "the account received funds"
        )

    def test_it_over_approximates_rather_than_under(self) -> None:
        """A narrative passing the gate under the heuristic passes under the real one."""
        counter = get_token_counter()
        text = "The account received 482,300 US Dollar from 9 counterparties."
        assert counter.count(text) >= word_count(text)

    def test_asking_for_an_unavailable_tokenizer_raises_rather_than_falling_back(self) -> None:
        with pytest.raises(TokenizerUnavailableError):
            get_token_counter("meta-llama/Llama-3.1-8B")


class TestPiiScanner:
    @pytest.mark.parametrize(
        "text",
        [
            "reach them at analyst@example.com",
            "settlement via GB29NWBK60161331926819",
            "card 4111 1111 1111 1111",
            "see https://example.com/case",
            "wallet 0x52908400098527886E0F7030069857D2E4169EE7",
        ],
    )
    def test_it_catches_a_real_world_identifier(self, text: str) -> None:
        assert scan_for_identifiers(text)

    def test_it_does_not_flag_the_substrates_own_account_key(self) -> None:
        """D-011 keys are synthetic and every narrative must be able to name one."""
        assert scan_for_identifiers("account 0137897|812AD4070 sent 482,300 US Dollar") == []

    def test_it_does_not_flag_a_rendered_timestamp_or_amount(self) -> None:
        assert (
            scan_for_identifiers("from 2022-09-05 16:07 to 2022-09-07 09:40, 1,234,567.89 Euro")
            == []
        )


class TestTrainingRecordSchema:
    def test_it_rejects_a_slot_without_a_field_path(self) -> None:
        payload = {
            "schema_version": "1.0.0",
            "case_id": "c",
            "dataset": "amlworld_hi_small",
            "split": "train",
            "tier": "bronze",
            "facts": {},
            "graph_ref": "a/b#c",
            "serialised_facts": "x",
            "target_narrative": "y",
            "target_slots": [
                {
                    "field_path": "",
                    "span": [0, 1],
                    "rendered_value": "y",
                    "raw_value": 1,
                    "claim_type": "numeric",
                }
            ],
            "generator": {"method": "template", "renderer_version": "0.1.0"},
            "verification": {
                "supported": 1,
                "contradicted": 0,
                "unverifiable": 0,
                "unverifiable_rate": 0.0,
                "n_claims": 1,
            },
        }
        with pytest.raises(Exception, match="field_path|minLength|facts"):
            validate_training_record(payload)
