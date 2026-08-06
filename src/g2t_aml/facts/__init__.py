"""The fact layer: extract, verify, bound.

One module with three jobs, and the reason invariant 1 exists.

**Forward** — :func:`~g2t_aml.facts.extractor.extract_facts` turns a case subgraph into a
structured, checkable fact record.

**Reverse** — :func:`~g2t_aml.facts.checkers.check_claim` verifies whether a generated
narrative's claims hold against that record, using the same field paths and the same
semantics. The faithfulness metric and the corpus generator are this code run in opposite
directions, which is why a disagreement between them is a bug rather than a parameter.

**Bounding** — ``schemas/vocab_v1.yaml``, loaded by
:func:`~g2t_aml.facts.vocab.load_vocabulary`, defines what any component is permitted to
assert at all.

Re-exported here are the types that cross a phase boundary. Everything else is reached
through its own module, because a name that appears here is a promise later phases build
on.
"""

from g2t_aml.facts.checkers import (
    CHECKER_REGISTRY,
    CheckContext,
    CheckResult,
    Claim,
    ClaimType,
    DurationClaim,
    Verdict,
    check_claim,
    check_narrative_text,
    checkable_field_paths,
    summarise,
)
from g2t_aml.facts.config import FactConfig, ToleranceConfig
from g2t_aml.facts.extractor import ROLE_VOCABULARY, extract_facts, extract_facts_from_view
from g2t_aml.facts.salience import SalienceReport, salience_report
from g2t_aml.facts.schema import (
    CASE_FACTS_SCHEMA_VERSION,
    EXTRACTOR_VERSION,
    CaseFacts,
    ModelSignal,
    Money,
    Unavailable,
    facts_to_dict,
    is_available,
    validate_facts,
)
from g2t_aml.facts.serialiser import serialise_facts
from g2t_aml.facts.taxonomy import CRITICAL_CLASSES, HallucinationClass, Severity
from g2t_aml.facts.vocab import ControlledVocabulary, load_vocabulary

__all__ = [
    "CASE_FACTS_SCHEMA_VERSION",
    "CHECKER_REGISTRY",
    "CRITICAL_CLASSES",
    "EXTRACTOR_VERSION",
    "ROLE_VOCABULARY",
    "CaseFacts",
    "CheckContext",
    "CheckResult",
    "Claim",
    "ClaimType",
    "ControlledVocabulary",
    "DurationClaim",
    "FactConfig",
    "HallucinationClass",
    "ModelSignal",
    "Money",
    "SalienceReport",
    "Severity",
    "ToleranceConfig",
    "Unavailable",
    "Verdict",
    "check_claim",
    "check_narrative_text",
    "checkable_field_paths",
    "extract_facts",
    "extract_facts_from_view",
    "facts_to_dict",
    "is_available",
    "load_vocabulary",
    "salience_report",
    "serialise_facts",
    "summarise",
    "validate_facts",
]
