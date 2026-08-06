"""The Phase 7 graph encoder: six arms, one interface, one honest comparison.

The encoder has two jobs. It produces a **case-level risk score**, which populates
``model_signal`` in the fact record through ``scripts/07b_score_cases.py``. And it
produces **node and pooled-token embeddings**, which Phase 8's fusion layer projects into
the language model's embedding space.

It is trained on a real supervised objective first, deliberately, rather than end-to-end
with the generator. An encoder trained only through the language-model loss underfits
badly on a corpus this size, and then the fusion ablation in Phase 9 cannot be read: a
null result would be indistinguishable between "fusion does not help" and "the encoder was
never any good". Training it against ground truth first makes the Phase 9 comparison
interpretable.

Nothing here imports at package-import time that would pull torch into a CPU-only
environment: ``g2t_aml.models`` stays importable without the ``graph`` extra, and the
submodules that need torch raise a clear message when it is absent (D-004).
"""

from __future__ import annotations

from g2t_aml.models.encoder.dataset import (
    ALL_SPLITS,
    MANIFEST_SPLITS,
    REALISTIC_SPLIT,
    TYPOLOGY_CLASSES,
    CacheManifest,
    DatasetError,
    build_feature_cache,
    load_case_ids,
    load_feature_space,
    load_split,
    typology_index,
    verify_cache_against_manifest,
)
from g2t_aml.models.encoder.features import (
    EDGE_FEATURE_NAMES,
    FEATURE_SPEC_VERSION,
    NODE_FEATURE_NAMES,
    FeatureError,
    FeatureSpace,
    assert_no_label_columns,
    build_case_data,
    fit_feature_space,
)

__all__ = [
    "ALL_SPLITS",
    "EDGE_FEATURE_NAMES",
    "FEATURE_SPEC_VERSION",
    "MANIFEST_SPLITS",
    "NODE_FEATURE_NAMES",
    "REALISTIC_SPLIT",
    "TYPOLOGY_CLASSES",
    "CacheManifest",
    "DatasetError",
    "FeatureError",
    "FeatureSpace",
    "assert_no_label_columns",
    "build_case_data",
    "build_feature_cache",
    "fit_feature_space",
    "load_case_ids",
    "load_feature_space",
    "load_split",
    "typology_index",
    "verify_cache_against_manifest",
]
