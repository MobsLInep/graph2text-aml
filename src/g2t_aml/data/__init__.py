"""Substrate ingestion, case construction, and the frozen temporal splits.

Importing this package pulls in Polars but never torch: the PyG conversion lives in
``g2t_aml.data.pyg_adapter`` and is imported explicitly by the phases that need it.

Re-exported here are the types that cross a phase boundary. Everything else — the loaders,
the motif scorer, the samplers, the auditor's internals — is reached through its own
module, because a name that appears here is a promise later phases may build on.
"""

from g2t_aml.data.canonical import (
    AMLWORLD_AVAILABILITY,
    CANONICAL_SCHEMA_VERSION,
    ELLIPTIC2_AVAILABILITY,
    TYPOLOGY_VOCABULARY,
    AvailabilityMask,
    CanonicalGraph,
)
from g2t_aml.data.case_extraction import (
    EXTRACTION_PROTOCOL_VERSION,
    ExtractionParams,
    GraphIndex,
    TimeWindow,
    extract_case,
    passthrough_case,
)
from g2t_aml.data.case_sampling import (
    SAMPLING_SCHEMA_VERSION,
    CaseCollection,
    CaseRecord,
    SamplingParams,
)
from g2t_aml.data.splits import (
    MANIFEST_VERSION,
    SPLIT_NAMES,
    SplitParams,
    load_split_manifest,
    split_of,
)

__all__ = [
    "AMLWORLD_AVAILABILITY",
    "CANONICAL_SCHEMA_VERSION",
    "ELLIPTIC2_AVAILABILITY",
    "EXTRACTION_PROTOCOL_VERSION",
    "MANIFEST_VERSION",
    "SAMPLING_SCHEMA_VERSION",
    "SPLIT_NAMES",
    "TYPOLOGY_VOCABULARY",
    "AvailabilityMask",
    "CanonicalGraph",
    "CaseCollection",
    "CaseRecord",
    "ExtractionParams",
    "GraphIndex",
    "SamplingParams",
    "SplitParams",
    "TimeWindow",
    "extract_case",
    "load_split_manifest",
    "passthrough_case",
    "split_of",
]
