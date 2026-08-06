"""Graph2Text AML: graph-conditioned generation of SAR narratives."""

__version__ = "0.1.0"

# Schema version for the `case_facts` record. Invariant 3: this is pinned and recorded
# in every derived artifact. Bumping it invalidates every generated corpus.
#
# FROZEN at 1.0.0 in Phase 3, when schemas/case_facts_v1.json was written. The single
# source of truth is g2t_aml.facts.schema.CASE_FACTS_SCHEMA_VERSION; this re-export
# exists so a consumer that has not imported the fact layer still sees the same number,
# and tests/unit/test_facts_coverage.py asserts the two, the JSON Schema's `const`, the
# vocabulary and configs/config.yaml all agree.
CASE_FACTS_SCHEMA_VERSION = "1.0.0"

__all__ = ["CASE_FACTS_SCHEMA_VERSION", "__version__"]
