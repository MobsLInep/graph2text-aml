"""Graph2Text AML: graph-conditioned generation of SAR narratives."""

__version__ = "0.1.0"

# Schema version for the `case_facts` record. Invariant 3: this is pinned and recorded
# in every derived artifact. Bumping it invalidates every generated corpus.
CASE_FACTS_SCHEMA_VERSION = "0.1.0"

__all__ = ["CASE_FACTS_SCHEMA_VERSION", "__version__"]
