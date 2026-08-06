"""Two independent claim extractors, and the agreement between them.

Layer 2 rests entirely on turning a narrative back into checkable claims. That step is
where a faithfulness metric is most easily wrong and least easily seen to be wrong: an
extractor that quietly fails to find a claim reports a narrative as more faithful than it
is, and nothing about the resulting number looks unusual.

The defence is two implementations that share no machinery:

**Method A** (:mod:`~g2t_aml.eval.claim_extraction.deterministic`) aligns against the
Bronze slot annotation and applies rules over the controlled vocabulary. Deterministic,
free and fast enough to run at every training checkpoint.

**Method B** (:mod:`~g2t_aml.eval.claim_extraction.llm_based`) decomposes the narrative
into atomic claims with a language model and verifies each against the serialised fact
record as premises. Slow, costly, and run on a sample.

**The agreement between them is what makes the metric credible**, and it is measured
rather than assumed — :mod:`~g2t_aml.eval.claim_extraction.agreement` computes Cohen's κ
on both verdict assignment and claim-boundary alignment over a 300-case sample, and the κ
is reported in the paper. An automatic faithfulness metric validated against nothing is a
number with a method section.
"""

from g2t_aml.eval.claim_extraction.agreement import (
    AgreementReport,
    BoundaryAlignment,
    cohens_kappa,
    measure_agreement,
)
from g2t_aml.eval.claim_extraction.deterministic import (
    DEFAULT_RULES,
    AttributionRule,
    DeterministicClaimExtractor,
    DeterministicReport,
    extract_claims,
)
from g2t_aml.eval.claim_extraction.llm_based import (
    AtomicClaim,
    LLMClaimExtractor,
    LLMExtractionError,
    parse_extraction_response,
)

__all__ = [
    "DEFAULT_RULES",
    "AgreementReport",
    "AtomicClaim",
    "AttributionRule",
    "BoundaryAlignment",
    "DeterministicClaimExtractor",
    "DeterministicReport",
    "LLMClaimExtractor",
    "LLMExtractionError",
    "cohens_kappa",
    "extract_claims",
    "measure_agreement",
    "parse_extraction_response",
]
