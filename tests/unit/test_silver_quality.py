"""Degeneracy, deduplication, and the Bronze-copy check the checker cannot see.

A verified record is faithful, not automatically useful. Every check here catches a
failure that passes verification untouched.
"""

from __future__ import annotations

import pytest

from g2t_aml.corpus.silver.quality import (
    QualityConfig,
    degeneracy_reasons,
    filter_records,
)

WELL_FORMED = (
    "The subject account 001|80000000 acted as originator across the subgraph examined "
    "for this referral, which covers sixteen accounts in total.\n\n"
    "Activity totalled a substantial sum across nine counterparties during a window of "
    "under one day, with every transfer settling inside that period.\n\n"
    "The dispersal pattern across those counterparties appears consistent with layering "
    "as recorded in the source data for this stream.\n\n"
    "This warrants further review by an investigator with access to customer records "
    "that are not represented in the transaction subgraph."
)


def record(case_id, text, teacher="frontier"):
    return {"case_id": case_id, "target_narrative": text, "generator": {"teacher": teacher}}


class TestDegeneracy:
    def test_a_well_formed_narrative_passes(self):
        assert degeneracy_reasons(WELL_FORMED) == ()

    def test_truncation_is_caught(self):
        assert "truncated" in degeneracy_reasons(WELL_FORMED[:-40])

    def test_an_ngram_loop_is_caught(self):
        looped = ("the account received funds from the counterparty and " * 8) + "."
        assert "repetitive" in degeneracy_reasons(looped)

    def test_a_missing_section_is_caught(self):
        assert "too_few_sections" in degeneracy_reasons(
            WELL_FORMED.split("\n\n")[0] + "\n\n" + WELL_FORMED.split("\n\n")[1]
        )

    def test_a_stub_section_is_caught(self):
        assert "stub_section" in degeneracy_reasons(WELL_FORMED + "\n\nAlso this.")

    @pytest.mark.parametrize(
        "prefix",
        ["Here is the narrative:\n\n", "Sure! ", "Below is the rewritten SAR narrative.\n\n"],
    )
    def test_a_chat_preamble_is_caught(self, prefix):
        assert "preamble_or_markdown" in degeneracy_reasons(prefix + WELL_FORMED)

    @pytest.mark.parametrize("markup", ["## Subject\n\n", "- bullet\n\n", "```\n\n"])
    def test_markdown_is_caught(self, markup):
        assert "preamble_or_markdown" in degeneracy_reasons(markup + WELL_FORMED)

    def test_low_lexical_variety_is_caught(self):
        assert "low_lexical_variety" in degeneracy_reasons(("funds moved " * 60) + ".")


class TestFiltering:
    def test_a_near_verbatim_copy_of_its_own_bronze_is_dropped(self):
        """The failure a faithfulness metric will never flag: a rewrite that reproduces
        the template is perfectly faithful and contributes nothing beyond Bronze."""
        kept, report = filter_records([record("c1", WELL_FORMED)], {"c1": WELL_FORMED})
        assert kept == []
        assert report.by_reason["bronze_verbatim"] == 1

    def test_a_genuine_rewrite_survives(self):
        different = (
            "Account 001|80000000 originated the movements under review here, across a "
            "subgraph of sixteen accounts examined for this referral in full.\n\n"
            "Over a window shorter than a day, value reached nine separate counterparties, "
            "with each transfer settling before the window closed entirely.\n\n"
            "That dispersal is consistent with layering, per the typology recorded against "
            "this stream in the underlying source data for the case.\n\n"
            "Further review is merited by an investigator holding customer records beyond "
            "those represented in this transaction subgraph alone."
        )
        kept, report = filter_records([record("c1", different)], {"c1": WELL_FORMED})
        assert len(kept) == 1
        assert report.n_dropped == 0
        assert report.bronze_similarity["max"] < 0.9

    def test_a_silver_duplicate_of_another_silver_record_is_dropped(self):
        kept, report = filter_records(
            [record("c1", WELL_FORMED), record("c2", WELL_FORMED)],
            {"c1": "unrelated bronze text one", "c2": "unrelated bronze text two"},
        )
        assert len(kept) == 1
        assert "near_duplicate" in report.by_reason

    def test_bronze_wins_a_cross_tier_duplicate(self):
        """Bronze is the reference tier; a Silver build must never remove it."""
        kept, report = filter_records(
            [record("c1", WELL_FORMED)], {"c9": WELL_FORMED, "c1": "unrelated bronze"}
        )
        assert kept == []
        assert "near_duplicate" in report.by_reason

    def test_per_teacher_drop_asymmetry_is_reported(self):
        """If one teacher's work is dropped far more often, the surviving corpus is not
        the corpus that was assigned -- and that is itself a finding."""
        # Genuinely distinct texts, not one text with a word swapped: five near-identical
        # narratives would be dropped as duplicates of each other, which is correct
        # behaviour and not what this test is measuring.
        subjects = ["alpha", "bravo", "charlie", "delta", "echo"]
        records = [
            record(
                f"good{i}",
                f"Account 00{i}|8000000{i} originated the {word} movements under review "
                f"across a subgraph of {i + 11} accounts examined here.\n\n"
                f"Value reached {i + 3} counterparties inside a window of {i + 2} hours, "
                f"each transfer settling before that period closed entirely.\n\n"
                f"The {word} dispersal is consistent with layering per the typology "
                f"recorded against this particular stream in the source data.\n\n"
                f"Review is merited by an investigator holding {word} customer records "
                f"beyond those represented in the transaction subgraph.",
                "open_weights",
            )
            for i, word in enumerate(subjects)
        ]
        broken = [record(f"bad{i}", "Truncated and short", "frontier") for i in range(5)]
        _, report = filter_records(records + broken, {})
        rates = report.drop_rate_by_teacher()
        assert rates["frontier"] == 1.0
        assert rates["open_weights"] == 0.0
        assert report.to_dict()["teacher_drop_rate_spread"] == 1.0

    def test_the_report_counts_every_reason(self):
        _, report = filter_records([record("c1", "Short.")], {})
        assert report.n_input == 1
        assert report.n_kept == 0
        assert sum(report.by_reason.values()) >= 1
        assert report.to_dict()["drop_rate"] == 1.0

    def test_thresholds_are_configurable_but_default_to_the_standard(self):
        config = QualityConfig()
        assert config.dedup_jaccard == 0.85
        assert config.bronze_verbatim_jaccard == 0.90
