"""Patterns-file parser.

The patterns file is the only source of typology ground truth, so its parser is held to
the same standard as the fact layer: a hand-written fixture covering every typology, a
stream long enough to catch off-by-one position bugs, and an explicit case for every way
the grammar can be violated. A parser that silently drops a malformed stream would
under-count a typology, and the under-count would look exactly like a real dataset
property.
"""

from __future__ import annotations

import pytest

from g2t_aml.data.loaders.amlworld import (
    CANONICAL_COLUMNS,
    TYPOLOGY_MAP,
    PatternsParseError,
    parse_patterns_text,
)

FIXTURE = "tests/fixtures/patterns_valid.txt"

ROW = "2022/09/01 00:06,000001,ACCT000001,000002,ACCT000002,100.00,Euro,100.00,Euro,ACH,1"


@pytest.fixture
def valid_text(repo_root) -> str:
    return (repo_root / FIXTURE).read_text(encoding="utf-8")


def _wrap(typology: str, *rows: str) -> str:
    body = "\n".join(rows)
    return f"BEGIN LAUNDERING ATTEMPT - {typology}\n{body}\nEND LAUNDERING ATTEMPT - {typology}\n"


# ------------------------------------------------------------------ happy path ---


def test_parses_every_typology_in_the_fixture(valid_text):
    frame = parse_patterns_text(valid_text)
    assert set(frame["typology"].unique()) == set(TYPOLOGY_MAP.values())


def test_row_and_stream_counts(valid_text):
    frame = parse_patterns_text(valid_text)
    assert frame.height == 23
    assert frame["pattern_id"].n_unique() == 8


def test_carries_all_transaction_columns(valid_text):
    frame = parse_patterns_text(valid_text)
    for column in CANONICAL_COLUMNS:
        assert column in frame.columns


def test_position_in_stream_is_zero_based_and_contiguous(valid_text):
    """A long stream is the case where an off-by-one would hide."""
    frame = parse_patterns_text(valid_text)
    longest = frame.filter(frame["typology"] == "gather_scatter")
    assert longest.height == 12
    assert longest["position_in_stream"].to_list() == list(range(12))


def test_pattern_ids_are_unique_per_stream_and_ordinal(valid_text):
    frame = parse_patterns_text(valid_text)
    fan_out = frame.filter(frame["typology"] == "fan_out")
    assert fan_out["pattern_id"].unique().to_list() == ["fan_out_00001"]


def test_detail_is_captured_and_optional(valid_text):
    frame = parse_patterns_text(valid_text)
    detailed = frame.filter(frame["typology"] == "fan_out")["typology_detail"][0]
    assert detailed == "Max 3-degree Fan-Out"
    # BIPARTITE has no colon-detail on its BEGIN line.
    bare = frame.filter(frame["typology"] == "bipartite")["typology_detail"][0]
    assert bare == ""


def test_transaction_key_uses_source_text_not_a_reparsed_float(valid_text):
    """Bitcoin rows carry six decimals; a float round-trip would corrupt the key."""
    frame = parse_patterns_text(valid_text)
    keys = frame["transaction_key"].to_list()
    assert any(k.endswith("|0.500000") for k in keys), "six-decimal amount was normalised away"
    assert all("|" in k for k in keys)


def test_self_loop_row_is_preserved(valid_text):
    """Reinvestment rows are self-loops and are legitimate members of a stream."""
    frame = parse_patterns_text(valid_text)
    loops = frame.filter(frame["src_account"] == frame["dst_account"])
    assert loops.height == 1
    assert loops["payment_format"][0] == "Reinvestment"


def test_blank_lines_between_streams_are_ignored():
    text = _wrap("CYCLE", ROW) + "\n\n\n" + _wrap("RANDOM", ROW)
    assert parse_patterns_text(text).height == 2


def test_empty_input_yields_empty_frame_with_full_schema():
    frame = parse_patterns_text("")
    assert frame.height == 0
    assert "transaction_key" in frame.columns
    assert "position_in_stream" in frame.columns


def test_repeated_typology_increments_the_ordinal():
    text = _wrap("CYCLE", ROW) + _wrap("CYCLE", ROW)
    ids = parse_patterns_text(text)["pattern_id"].to_list()
    assert ids == ["cycle_00001", "cycle_00002"]


def test_same_transaction_may_appear_in_two_streams():
    """Legitimate: a transaction can participate in more than one laundering pattern."""
    frame = parse_patterns_text(_wrap("CYCLE", ROW) + _wrap("STACK", ROW))
    assert frame.height == 2
    assert frame["transaction_key"].n_unique() == 1


# --------------------------------------------------------------- malformed input ---


def test_rejects_nested_begin():
    text = "BEGIN LAUNDERING ATTEMPT - CYCLE\n" + ROW + "\nBEGIN LAUNDERING ATTEMPT - STACK\n"
    with pytest.raises(PatternsParseError, match="do not nest"):
        parse_patterns_text(text)


def test_rejects_end_without_begin():
    with pytest.raises(PatternsParseError, match="END without a matching BEGIN"):
        parse_patterns_text("END LAUNDERING ATTEMPT - CYCLE\n")


def test_rejects_mismatched_end_typology():
    text = f"BEGIN LAUNDERING ATTEMPT - CYCLE\n{ROW}\nEND LAUNDERING ATTEMPT - STACK\n"
    with pytest.raises(PatternsParseError, match="does not close"):
        parse_patterns_text(text)


def test_rejects_unterminated_stream_at_eof():
    with pytest.raises(PatternsParseError, match="never closed"):
        parse_patterns_text(f"BEGIN LAUNDERING ATTEMPT - CYCLE\n{ROW}\n")


def test_rejects_row_outside_any_stream():
    with pytest.raises(PatternsParseError, match="outside any stream"):
        parse_patterns_text(ROW + "\n")


def test_rejects_unknown_typology():
    with pytest.raises(PatternsParseError, match="unknown typology"):
        parse_patterns_text(
            f"BEGIN LAUNDERING ATTEMPT - SPIRAL\n{ROW}\nEND LAUNDERING ATTEMPT - SPIRAL\n"
        )


def test_rejects_wrong_field_count():
    short_row = "2022/09/01 00:06,000001,ACCT000001,000002"
    with pytest.raises(PatternsParseError, match="comma-separated"):
        parse_patterns_text(_wrap("CYCLE", short_row))


def test_rejects_empty_stream():
    with pytest.raises(PatternsParseError, match="contains no transactions"):
        parse_patterns_text("BEGIN LAUNDERING ATTEMPT - CYCLE\nEND LAUNDERING ATTEMPT - CYCLE\n")


def test_rejects_malformed_delimiter_rather_than_reading_it_as_data():
    """A typo'd delimiter must fail loudly, not be parsed as a transaction row."""
    text = f"BEGIN LAUNDERING ATTEMPT - CYCLE\n{ROW}\nEND LAUNDERING ATTEMPTS - CYCLE\n"
    with pytest.raises(PatternsParseError, match="malformed delimiter"):
        parse_patterns_text(text)


def test_error_messages_carry_line_numbers():
    text = f"BEGIN LAUNDERING ATTEMPT - CYCLE\n{ROW}\n{ROW[:20]}\nEND LAUNDERING ATTEMPT - CYCLE\n"
    with pytest.raises(PatternsParseError, match=r":3:"):
        parse_patterns_text(text)


def test_source_name_appears_in_errors():
    with pytest.raises(PatternsParseError, match="mypatterns.txt"):
        parse_patterns_text(ROW + "\n", source="mypatterns.txt")
