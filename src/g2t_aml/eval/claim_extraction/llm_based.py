"""Method B: LLM atomic claim decomposition, then NLI-style verification.

The validation extractor. Slow, costly, and run on a sample — its job is not to score the
corpus but to say whether Method A can be believed.

**It shares nothing with Method A on purpose.** No slot alignment, no cue table, no
controlled-vocabulary lookup, and — importantly — it does not call
:func:`~g2t_aml.facts.checkers.check_claim`. It reaches its own verdict from the
serialised fact record used as premises. Two extractors that agreed because they both
consulted the same checker would agree by construction, and the κ between them would
measure nothing. The only thing the two methods share is the three-valued verdict
vocabulary, and that is what makes them comparable rather than what makes them agree.

**Two calls, not one.** Decomposition runs without the fact record; verification runs with
it. A single call that did both lets the model settle on a verdict and then choose a
decomposition that supports it, which biases the claim boundaries — the thing the
boundary half of the κ is measuring. The prompt headers record the reasoning.

**No provider dependency is added.** Both stages go through the Phase 5 :class:`Teacher`
protocol, so the whole pipeline runs under
:class:`~g2t_aml.corpus.silver.api_client.ScriptedTeacher` with no network, and the tests
exercise the real path rather than a mock beside it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from g2t_aml.corpus.silver.api_client import Teacher
from g2t_aml.corpus.silver.prompts import PromptRenderError, load_prompt
from g2t_aml.facts.checkers import Claim, ClaimType, Verdict
from g2t_aml.facts.schema import CaseFacts
from g2t_aml.facts.serialiser import SerialisationStyle, serialise_facts

__all__ = [
    "DECOMPOSITION_PROMPT_NAME",
    "ENTAILMENT_PROMPT_NAME",
    "EXTRACTOR_METHOD",
    "AtomicClaim",
    "LLMClaimExtractor",
    "LLMExtractionError",
    "LLMExtractionReport",
    # Re-exported so a caller catching prompt failures need not reach into the Silver
    # package for the exception type Method B raises through it.
    "PromptRenderError",
    "parse_entailment_response",
    "parse_extraction_response",
]

#: Recorded on every report, so a stored claim set says which method produced it.
EXTRACTOR_METHOD = "llm"

DECOMPOSITION_PROMPT_NAME = "eval_claim_decomposition_v1"
ENTAILMENT_PROMPT_NAME = "eval_claim_entailment_v1"

#: Claim types the decomposer is allowed to emit, mapped onto the checker's enum. A type
#: outside this set is a model that ignored its instructions, and the claim is dropped
#: rather than coerced: coercing it would put a claim of unknown kind into the agreement
#: sample under a type it does not have.
_CLAIM_TYPES: dict[str, ClaimType] = {
    "numeric": ClaimType.NUMERIC,
    "temporal": ClaimType.TEMPORAL,
    "entity": ClaimType.ENTITY,
    "categorical": ClaimType.CATEGORICAL,
    "qualitative": ClaimType.QUALITATIVE,
    "regulatory": ClaimType.REGULATORY,
}

#: A fenced JSON block, in case a model wraps its object despite being told not to.
#: Tolerated at parse time and nowhere else: the instruction stays in the prompt, because
#: a model that has started fencing has probably started prefacing too.
_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>\{.*\})\s*```", re.DOTALL)


class LLMExtractionError(ValueError):
    """Raised when a teacher's response cannot be read as a claim set or a verdict set.

    Never softened into an empty result. An unparseable response that returned no claims
    would make the narrative look perfectly faithful to Method B and would drag the
    measured κ toward zero for a reason that has nothing to do with either method.
    """


@dataclass(frozen=True)
class AtomicClaim:
    """One atomic claim, with Method B's own verdict on it.

    Attributes:
        text: The self-contained restatement the decomposer produced.
        evidence: The exact narrative substring the claim was drawn from.
        span: ``(start, end)`` offsets of ``evidence`` in the narrative, or None when the
            evidence could not be located. An unlocatable claim takes no part in boundary
            agreement — it is not evidence of disagreement, it is a missing measurement.
        claim_type: The kind of claim.
        verdict: Method B's verdict, assigned by the entailment stage. None before that
            stage has run.
        rationale: The one-sentence justification the model gave.
    """

    text: str
    evidence: str
    span: tuple[int, int] | None
    claim_type: ClaimType
    verdict: Verdict | None = None
    rationale: str = ""

    def to_claim(self) -> Claim:
        """Return this as a checker :class:`~g2t_aml.facts.checkers.Claim`.

        Field-free by construction: Method B never resolves a claim to a fact field, and
        inventing one here would make its output an input to Method A's machinery.

        Returns:
            The claim, spanning ``(0, 0)`` when the evidence could not be located.
        """
        return Claim(
            text_span=self.span if self.span is not None else (0, 0),
            field_path=None,
            claim_type=self.claim_type,
            value=self.text,
            raw_text=self.evidence,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the claim as a JSON-serialisable mapping.

        Returns:
            Every field, with the verdict as its string value.
        """
        return {
            "text": self.text,
            "evidence": self.evidence,
            "span": list(self.span) if self.span is not None else None,
            "claim_type": self.claim_type.value,
            "verdict": self.verdict.value if self.verdict is not None else None,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class LLMExtractionReport:
    """What Method B found for one narrative.

    Attributes:
        case_id: The case.
        claims: The atomic claims, in the order the decomposer emitted them.
        unlocated: Claims whose evidence string does not appear in the narrative. Counted
            and reported rather than dropped silently: a high rate means the decomposer is
            paraphrasing its evidence, which invalidates the boundary half of the κ and is
            invisible in the verdict half.
        cost_usd: What the two calls cost.
        method: :data:`EXTRACTOR_METHOD`.
    """

    case_id: str
    claims: tuple[AtomicClaim, ...]
    unlocated: int = 0
    cost_usd: float = 0.0
    method: str = EXTRACTOR_METHOD

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable mapping.

        Returns:
            The claims and the diagnostics.
        """
        return {
            "case_id": self.case_id,
            "method": self.method,
            "n_claims": len(self.claims),
            "unlocated": self.unlocated,
            "cost_usd": self.cost_usd,
            "claims": [claim.to_dict() for claim in self.claims],
        }


def _payload(text: str) -> dict[str, Any]:
    """Parse a teacher response into a JSON object.

    Args:
        text: The raw completion.

    Returns:
        The parsed object.

    Raises:
        LLMExtractionError: If the response is not a JSON object, fenced or otherwise.
    """
    candidate = text.strip()
    fenced = _FENCE_RE.search(candidate)
    if fenced is not None:
        candidate = fenced.group("body")
    elif not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise LLMExtractionError(f"response carries no JSON object: {text[:200]!r}")
        candidate = candidate[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMExtractionError(f"response is not valid JSON ({exc}): {text[:200]!r}") from exc
    if not isinstance(parsed, dict):
        raise LLMExtractionError(f"response is a {type(parsed).__name__}, expected an object")
    return parsed


def parse_extraction_response(text: str, narrative: str) -> tuple[list[AtomicClaim], int]:
    """Parse a decomposition response and locate each claim in the narrative.

    Args:
        text: The raw completion from the decomposition call.
        narrative: The narrative it decomposed, for locating the evidence spans.

    Returns:
        ``(claims, n_unlocated)``. A claim whose evidence does not appear verbatim gets
        ``span=None`` and is counted; it still carries a verdict, so it participates in
        verdict agreement and not in boundary agreement.

    Raises:
        LLMExtractionError: If the response is not JSON, or carries no ``claims`` list.
    """
    payload = _payload(text)
    raw = payload.get("claims")
    if not isinstance(raw, list):
        raise LLMExtractionError("decomposition response has no 'claims' list")

    claims: list[AtomicClaim] = []
    unlocated = 0
    cursor = 0
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        claim_text = str(entry.get("text", "")).strip()
        evidence = str(entry.get("evidence", "")).strip()
        kind = _CLAIM_TYPES.get(str(entry.get("type", "")).strip().lower())
        if not claim_text or kind is None:
            continue
        # Searched from a cursor rather than from zero so a phrase occurring twice is
        # located at successive occurrences, which keeps the claims in document order and
        # stops two claims collapsing onto one span in the boundary comparison.
        span: tuple[int, int] | None = None
        if evidence:
            found = narrative.find(evidence, cursor)
            if found < 0:
                found = narrative.find(evidence)
            if found >= 0:
                span = (found, found + len(evidence))
                cursor = found + len(evidence)
        if span is None:
            unlocated += 1
        claims.append(AtomicClaim(text=claim_text, evidence=evidence, span=span, claim_type=kind))
    return claims, unlocated


def parse_entailment_response(text: str, n_claims: int) -> list[tuple[Verdict, str]]:
    """Parse a verification response into one verdict per claim.

    Args:
        text: The raw completion from the entailment call.
        n_claims: How many claims were sent.

    Returns:
        ``(verdict, rationale)`` per claim, index-aligned with what was sent. Claims the
        model did not return a verdict for come back UNVERIFIABLE with a rationale saying
        so — never SUPPORTED, because a missing judgement is not a favourable one.

    Raises:
        LLMExtractionError: If the response is not JSON or carries no ``verdicts`` list.
    """
    payload = _payload(text)
    raw = payload.get("verdicts")
    if not isinstance(raw, list):
        raise LLMExtractionError("entailment response has no 'verdicts' list")

    out: list[tuple[Verdict, str]] = [
        (Verdict.UNVERIFIABLE, "the judge returned no verdict for this claim")
    ] * n_claims
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        index = entry.get("index")
        if not isinstance(index, int) or not 0 <= index < n_claims:
            continue
        try:
            verdict = Verdict(str(entry.get("verdict", "")).strip().lower())
        except ValueError:
            continue
        out[index] = (verdict, str(entry.get("rationale", "")).strip())
    return out


class LLMClaimExtractor:
    """Method B: decompose with one call, verify with a second."""

    def __init__(
        self,
        teacher: Teacher,
        *,
        prompts_dir: str | None = None,
        serialisation_style: SerialisationStyle = "verbose",
    ) -> None:
        """Bind the extractor to a teacher.

        Args:
            teacher: Anything satisfying the Phase 5 :class:`Teacher` protocol.
                :class:`~g2t_aml.corpus.silver.api_client.ScriptedTeacher` in tests, an
                :class:`~g2t_aml.corpus.silver.api_client.APITeacher` in a real run.
            prompts_dir: Override for the prompt directory. Tests only.
            serialisation_style: Which fact serialisation to use as premises. Verbose by
                default: the compact form drops the availability annotations, and a judge
                that cannot see which fields are masked will contradict claims about them.

        Raises:
            PromptRenderError: If either prompt file is malformed.
            FileNotFoundError: If either prompt file is missing.
        """
        self.teacher = teacher
        self._decompose_prompt = load_prompt(DECOMPOSITION_PROMPT_NAME, prompts_dir)
        self._entail_prompt = load_prompt(ENTAILMENT_PROMPT_NAME, prompts_dir)
        self.serialisation_style = serialisation_style

    def extract(self, narrative: str, facts: CaseFacts) -> list[Claim]:
        """Extract every claim the narrative makes, satisfying the shared protocol.

        Args:
            narrative: The narrative text.
            facts: The record it is about.

        Returns:
            The claims. Verdicts are discarded by this call — use :meth:`report` when the
            verdicts are wanted, which is every use inside this package.
        """
        return [claim.to_claim() for claim in self.report(narrative, facts).claims]

    def report(self, narrative: str, facts: CaseFacts) -> LLMExtractionReport:
        """Decompose and verify one narrative.

        Args:
            narrative: The narrative text.
            facts: The record used as premises.

        Returns:
            The claims with their verdicts, and the cost of the two calls.

        Raises:
            LLMExtractionError: If either response cannot be parsed.
            PromptRenderError: If a prompt and this method disagree about placeholders.
        """
        decomposition = self.teacher.complete(
            self._decompose_prompt.render({"narrative": narrative}),
            case_id=facts.case_id,
            kind="decompose",
        )
        claims, unlocated = parse_extraction_response(decomposition.text, narrative)
        cost = float(decomposition.cost_usd)

        if not claims:
            return LLMExtractionReport(
                case_id=facts.case_id, claims=(), unlocated=unlocated, cost_usd=cost
            )

        entailment = self.teacher.complete(
            self._entail_prompt.render(
                {
                    "fact_record": serialise_facts(facts, style=self.serialisation_style),
                    "availability_block": _availability_block(facts),
                    "claim_block": _claim_block(claims),
                }
            ),
            case_id=facts.case_id,
            kind="entail",
        )
        cost += float(entailment.cost_usd)
        verdicts = parse_entailment_response(entailment.text, len(claims))

        judged = tuple(
            AtomicClaim(
                text=claim.text,
                evidence=claim.evidence,
                span=claim.span,
                claim_type=claim.claim_type,
                verdict=verdict,
                rationale=rationale,
            )
            for claim, (verdict, rationale) in zip(claims, verdicts, strict=True)
        )
        return LLMExtractionReport(
            case_id=facts.case_id, claims=judged, unlocated=unlocated, cost_usd=cost
        )


def _claim_block(claims: Sequence[AtomicClaim]) -> str:
    """Render the claims as a numbered list for the entailment prompt.

    Args:
        claims: The decomposed claims.

    Returns:
        One numbered line per claim, indices matching the ``index`` field the judge is
        asked to return.
    """
    return "\n".join(f"{i}. [{c.claim_type.value}] {c.text}" for i, c in enumerate(claims))


def _availability_block(facts: CaseFacts) -> str:
    """Render the substrate's availability mask for the judge.

    Invariant 4 in the form the judge needs it: a claim about a fact family the substrate
    does not carry is UNVERIFIABLE whatever it says, and a judge that cannot see the mask
    will contradict it instead. Rendered here rather than imported from the Silver prompt
    builder because that one renders it as an instruction to a *writer* — "omit these" —
    and this one is a statement to a *judge*.

    Args:
        facts: The record whose mask is described.

    Returns:
        A bullet list of unavailable fact families, or a line saying there are none.
    """
    mask = facts.availability.to_dict()
    missing = [
        f"- `{flag}` — this substrate carries no such data" for flag, ok in mask.items() if not ok
    ]
    if not missing:
        return "AVAILABILITY\n------------\nEvery fact family is available for this substrate."
    return "AVAILABILITY\n------------\n" + "\n".join(missing)
