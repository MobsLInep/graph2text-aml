from __future__ import annotations

import json

import pandas as pd
import pytest

from g2t_aml.utils.io import (
    atomic_path,
    ensure_dir,
    read_json,
    read_jsonl,
    read_parquet,
    write_json,
    write_jsonl,
    write_parquet,
)


def test_json_roundtrip(tmp_path):
    obj = {"b": 1, "a": [1, 2, {"c": None}]}
    path = write_json(tmp_path / "x" / "o.json", obj)
    assert read_json(path) == obj


def test_json_canonical_is_byte_stable(tmp_path):
    p1 = write_json(tmp_path / "a.json", {"b": 1, "a": 2}, canonical=True)
    p2 = write_json(tmp_path / "b.json", {"a": 2, "b": 1}, canonical=True)
    assert p1.read_bytes() == p2.read_bytes()


def test_jsonl_roundtrip(tmp_path):
    records = [{"i": i, "id": f"ACCT-{i:04d}"} for i in range(5)]
    path = write_jsonl(tmp_path / "r.jsonl", records)
    assert list(read_jsonl(path)) == records


def test_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text('{"a":1}\n\n{"a":2}\n')
    assert list(read_jsonl(path)) == [{"a": 1}, {"a": 2}]


def test_atomic_write_leaves_no_partial_file_on_failure(tmp_path):
    target = tmp_path / "out.json"
    write_json(target, {"generation": 1})

    with pytest.raises(RuntimeError), atomic_path(target) as tmp:
        tmp.write_text('{"generation": 2')  # truncated on purpose
        raise RuntimeError("interrupted")

    # Original survives untouched and no temp file is left behind.
    assert json.loads(target.read_text()) == {"generation": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]


def test_atomic_write_replaces_on_success(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("old")
    with atomic_path(target) as tmp:
        tmp.write_text("new")
    assert target.read_text() == "new"
    assert list(tmp_path.iterdir()) == [target]


def test_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({"id": ["ACCT-0001", "ACCT-0002"], "amount": [10.5, 20.0]})
    path = write_parquet(tmp_path / "t.parquet", df)
    pd.testing.assert_frame_equal(read_parquet(path), df)


def test_ensure_dir_is_idempotent(tmp_path):
    d = ensure_dir(tmp_path / "a" / "b")
    assert d.is_dir()
    assert ensure_dir(d) == d
