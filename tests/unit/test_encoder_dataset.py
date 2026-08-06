"""Dataset tests: splits come from the frozen manifest, and only from there.

Invariant 2 says splits are temporal and frozen and are never regenerated from a seed at
runtime. The encoder stack has exactly one door to a split, and these tests assert that
door is locked in the two ways it could be picked: an absent manifest must stop the run
rather than fall back, and a cache built against a different manifest must be detected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from g2t_aml.data.canonical import TYPOLOGY_VOCABULARY
from g2t_aml.data.splits import SplitError
from g2t_aml.models.encoder.dataset import (
    ALL_SPLITS,
    MANIFEST_SPLITS,
    REALISTIC_SPLIT,
    TYPOLOGY_CLASSES,
    DatasetError,
    load_case_ids,
    load_typologies,
    typology_index,
    verify_cache_against_manifest,
)
from g2t_aml.utils.hashing import hash_id_list

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_MANIFEST = REPO_ROOT / "schemas" / "splits" / "amlworld"


def _manifest(tmp_path: Path, ids: dict[str, list[str]]) -> Path:
    out = tmp_path / "splits"
    out.mkdir()
    (out / "splits.json").write_text(
        json.dumps(
            {
                "manifest_version": "1.0.0",
                "dataset": "amlworld_hi_small",
                "splits": {
                    name: {
                        "case_ids": values,
                        "n": len(values),
                        "id_list_sha256": hash_id_list(values),
                    }
                    for name, values in ids.items()
                },
            }
        )
    )
    return out


# --------------------------------------------------------------- typologies ---


def test_typology_classes_match_the_canonical_vocabulary():
    """The head's class indices and the fact record's labels cannot be allowed to drift."""
    assert TYPOLOGY_CLASSES == TYPOLOGY_VOCABULARY
    assert len(TYPOLOGY_CLASSES) == 9
    assert "unclassified" in TYPOLOGY_CLASSES


def test_absent_typology_maps_to_the_ignore_sentinel():
    """-1 keeps the case in the batch for the binary head and out of the typology loss."""
    assert typology_index(None) == -1
    assert typology_index("not_a_typology") == -1
    assert typology_index("fan_out") == TYPOLOGY_CLASSES.index("fan_out")


def test_missing_fact_aggregate_leaves_the_head_unsupervised(tmp_path):
    assert load_typologies(tmp_path / "nope.parquet") == {}


# -------------------------------------------------------------------- splits ---


def test_absent_manifest_stops_the_run_rather_than_falling_back(tmp_path):
    """Invariant 2: there is deliberately no path that regenerates a split at runtime."""
    with pytest.raises(DatasetError, match="frozen split manifest"):
        load_case_ids(tmp_path)


def test_split_names_are_the_three_frozen_ones_plus_the_evaluation_stream():
    assert MANIFEST_SPLITS == ("train", "val", "test")
    assert REALISTIC_SPLIT not in MANIFEST_SPLITS
    assert (*MANIFEST_SPLITS, REALISTIC_SPLIT) == ALL_SPLITS


def test_hand_edited_manifest_is_rejected(tmp_path):
    """A content hash that no longer matches means every derived result is suspect."""
    manifest_dir = _manifest(
        tmp_path, {name: [f"{name}-{i}" for i in range(3)] for name in MANIFEST_SPLITS}
    )
    payload = json.loads((manifest_dir / "splits.json").read_text())
    payload["splits"]["test"]["case_ids"].append("smuggled-case")
    (manifest_dir / "splits.json").write_text(json.dumps(payload))

    with pytest.raises(SplitError, match="does not match its recorded sha256"):
        load_case_ids(manifest_dir)


def test_load_case_ids_returns_the_manifest_order(tmp_path):
    ids = {name: [f"{name}-{i}" for i in range(5)] for name in MANIFEST_SPLITS}
    assert load_case_ids(_manifest(tmp_path, ids)) == ids


@pytest.mark.skipif(
    not (COMMITTED_MANIFEST / "splits.json").is_file(),
    reason="the committed split manifest is not present in this checkout",
)
def test_the_committed_manifest_loads_and_is_disjoint():
    """The real thing: the encoder trains on exactly these ids and no others."""
    ids = load_case_ids(COMMITTED_MANIFEST)
    assert set(ids) == set(MANIFEST_SPLITS)
    train, val, test = (set(ids[name]) for name in MANIFEST_SPLITS)
    assert not train & val
    assert not train & test
    assert not val & test
    assert len(train) > len(test) > len(val)


# --------------------------------------------------------------------- cache ---


def test_cache_verification_needs_a_cache(tmp_path):
    with pytest.raises(DatasetError, match="no cache manifest"):
        verify_cache_against_manifest(tmp_path, tmp_path)


def test_cache_built_from_a_different_manifest_is_rejected(tmp_path):
    """The failure this exists to catch: manifest rebuilt, cache not."""
    from g2t_aml.models.encoder.features import FEATURE_SPEC_VERSION

    ids = {name: [f"{name}-{i}" for i in range(4)] for name in MANIFEST_SPLITS}
    manifest_dir = _manifest(tmp_path, ids)

    cache = tmp_path / "cache"
    cache.mkdir()
    stale = {name: hash_id_list(values) for name, values in ids.items()}
    stale["test"] = hash_id_list(["a-different-population"])
    (cache / "cache_manifest.json").write_text(
        json.dumps({"feature_spec_version": FEATURE_SPEC_VERSION, "split_id_hashes": stale})
    )

    with pytest.raises(DatasetError, match="different id list"):
        verify_cache_against_manifest(cache, manifest_dir)


def test_cache_built_by_a_stale_feature_spec_is_rejected(tmp_path):
    from g2t_aml.models.encoder.features import FeatureError

    ids = {name: [f"{name}-{i}" for i in range(4)] for name in MANIFEST_SPLITS}
    manifest_dir = _manifest(tmp_path, ids)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "cache_manifest.json").write_text(
        json.dumps({"feature_spec_version": "0.0.1", "split_id_hashes": {}})
    )

    with pytest.raises(FeatureError, match="feature spec"):
        verify_cache_against_manifest(cache, manifest_dir)
