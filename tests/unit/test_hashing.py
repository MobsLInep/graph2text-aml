from __future__ import annotations

import pandas as pd
import pytest
from omegaconf import OmegaConf

from g2t_aml.utils.hashing import (
    canonical_json,
    hash_config,
    hash_dataframe,
    hash_dir,
    hash_file,
    hash_id_list,
    hash_manifest,
    short,
)

SHA256_HEX_LEN = 64


def test_hash_file_matches_known_digest(tmp_path):
    # sha256("") is a fixed, well-known constant.
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    assert hash_file(p) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_hash_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "nope")


def test_hash_dir_is_order_independent(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")
    first = hash_dir(tmp_path)
    (tmp_path / "b.txt").write_text("beta")  # rewrite, same content
    assert hash_dir(tmp_path) == first


def test_hash_dir_detects_change(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    before = hash_dir(tmp_path)
    (tmp_path / "a.txt").write_text("alphb")
    assert hash_dir(tmp_path) != before


def test_hash_dir_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        hash_dir(tmp_path / "nope")


def test_canonical_json_is_key_order_stable():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_hash_config_ignores_key_order():
    a = {"z": 1, "a": {"y": 2, "x": 3}}
    b = {"a": {"x": 3, "y": 2}, "z": 1}
    assert hash_config(a) == hash_config(b)


def test_hash_config_resolves_omegaconf_interpolations():
    direct = OmegaConf.create({"root": "/tmp", "sub": "/tmp/x"})
    interpolated = OmegaConf.create({"root": "/tmp", "sub": "${root}/x"})
    assert hash_config(direct) == hash_config(interpolated)


def test_hash_config_detects_value_change():
    assert hash_config({"lr": 1e-4}) != hash_config({"lr": 2e-4})


def test_hash_id_list_is_set_like_by_default():
    assert hash_id_list(["b", "a", "c"]) == hash_id_list(["c", "b", "a"])


def test_hash_id_list_respects_order_when_asked():
    assert hash_id_list(["b", "a"], sort=False) != hash_id_list(["a", "b"], sort=False)


def test_hash_id_list_coerces_ints():
    assert hash_id_list([1, 2]) == hash_id_list(["1", "2"])


def test_hash_id_list_rejects_empty():
    with pytest.raises(ValueError, match="empty ID list"):
        hash_id_list([])


def test_hash_dataframe_is_row_and_column_order_independent():
    a = pd.DataFrame({"x": [1, 2, 3], "y": ["p", "q", "r"]})
    b = a.iloc[::-1][["y", "x"]].reset_index(drop=True)
    assert hash_dataframe(a) == hash_dataframe(b)


def test_hash_dataframe_detects_value_change():
    a = pd.DataFrame({"x": [1, 2, 3]})
    b = pd.DataFrame({"x": [1, 2, 4]})
    assert hash_dataframe(a) != hash_dataframe(b)


def test_hash_manifest_is_order_sensitive():
    one = [{"id": "A"}, {"id": "B"}]
    assert hash_manifest(one) != hash_manifest(list(reversed(one)))


def test_short_and_length():
    digest = hash_id_list(["A"])
    assert len(digest) == SHA256_HEX_LEN
    assert short(digest, 8) == digest[:8]
    with pytest.raises(ValueError):
        short(digest, SHA256_HEX_LEN + 1)
