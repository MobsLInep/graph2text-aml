"""Method A: slot alignment plus rule-based attribution over the controlled vocabulary.

The primary extractor. Deterministic, free, and fast enough to run inside a training
callback at every checkpoint, which is the property that decides the design — a
faithfulness signal that can only be computed after a run is a report, not a signal.

**It is Phase 5's extractor plus one pass.** The alignment itself —
:class:`~g2t_aml.corpus.silver.claim_extraction.SlotAlignmentExtractor` — is imported and
composed rather than reimplemented. Reimplementing it would mean a second copy of
longest-value-first ordering and the token-boundary guard, and getting either subtly wrong
produces a corpus that scores itself as perfect (D-048). One implementation, one place to
be wrong, one place a test pins.

**The pass this module adds is attribution, and it exists because the alternative
understates hallucination.** Phase 5 treats every quantity that aligns to no Bronze slot
as a claim naming no field, which the checker resolves to UNVERIFIABLE. For Silver's
verification loop that is right and sufficient: an unbacked figure exhausts the budget and
the record is repaired or discarded either way. For Phase 10 it is not, because
Hallucination Rate is *contradicted* over total and Unverifiable Rate is a separate,
much softer number. A system that writes "the subject received from 14 distinct
counterparties" where the record says 9 would score zero hallucinations and one
unverifiable claim — the single most damaging thing a SAR narrative can do, filed under
the bucket for things the graph merely cannot speak to.

So a residual quantity is matched against a table of cue patterns
(:data:`DEFAULT_RULES`), each binding a surface phrasing to a fact field. When a cue
fires, the claim names that field and the checker adjudicates it properly: SUPPORTED,
or CONTRADICTED as H2. When no cue fires the claim still names no field and is still
UNVERIFIABLE — attribution can only ever move a claim from the soft bucket to a real
verdict, never the other way, which is why a missing rule costs sensitivity rather than
correctness.

**The rules are cues, not templates.** They are written to catch the phrasings a
paraphrasing model reaches for, not Bronze's exact wording, and they are deliberately
required to sit adjacent to the number. A rule that matched a field name anywhere in the
sentence would attribute the wrong number to it as soon as a sentence carried two.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from g2t_aml.corpus.bronze.format import (
    FormatError,
    parse_count,
    parse_duration,
    parse_percent,
    parse_timestamp,
)
from g2t_aml.corpus.record import BronzeNarrative
from g2t_aml.corpus.silver.claim_extraction import ExtractionReport, SlotAlignmentExtractor
from g2t_aml.facts.checkers import Claim, ClaimType, DurationClaim
from g2t_aml.facts.schema import CaseFacts
from g2t_aml.facts.vocab import ControlledVocabulary, load_vocabulary

__all__ = [
    "DEFAULT_RULES",
    "EXTRACTOR_METHOD",
    "AttributionRule",
    "DeterministicClaimExtractor",
    "DeterministicReport",
    "extract_claims",
]

#: Recorded on every report so a stored result says which method produced it. Method A
#: and Method B disagreeing is the finding; a stored claim set that does not say which
#: made it cannot participate in that comparison.
EXTRACTOR_METHOD = "deterministic"

#: A number as any of the formatters render it, for embedding in a cue pattern. Kept as a
#: fragment rather than a compiled pattern because every rule splices it into a named
#: group of its own.
_NUM = r"\d[\d,]*(?:\.\d+)?"

#: A currency name as ``format_money`` renders it: alphabetic words, which is exactly the
#: constraint ``format_money`` enforces so that value and unit stay separable.
_CUR = r"[A-Za-z][A-Za-z ]*[A-Za-z]"

#: How far either side of a quantity a cue may sit. Two words of slack, enough for
#: "9 distinct counterparties" and "9 further recipient accounts", not enough to reach
#: the next quantity in a sentence.
_CUE_SLACK = r"(?:\s+\w+){0,2}\s+"

#: The *shape* of a regulatory citation, matched regardless of content. Anything this
#: finds that the whitelist pass did not already consume is an invented rule, which is
#: H6 and Critical. Matching shape rather than content is the whole point: a list of
#: forbidden citations cannot contain the one a model has not invented yet.
#:
#: Two families. A threshold or obligation phrase — "the USD 42,000 mandatory disclosure
#: threshold", "a EUR 3,000 filing requirement" — and a statutory reference — "31 CFR
#: 1010.311", "Section 314(b)", "the Bank Secrecy Act". Both are things a compliance
#: reader would act on and neither is establishable from a transaction graph.
_REGULATION_RE = re.compile(
    r"(?:(?:the|a|an)\s+)?"
    r"(?:[A-Z]{3}\s*[\d,]+(?:\.\d+)?\s+|\d[\d,]*(?:\.\d+)?\s+[A-Z][a-z]+\s+)?"
    r"(?:\w+\s+){0,2}"
    r"(?:reporting|filing|disclosure|declaration|notification)\s+"
    r"(?:threshold|requirement|obligation|limit|rule)"
    r"|(?:\d+\s+CFR\s+[\d.]+)"
    r"|(?:Section\s+\d+(?:\([a-z0-9]+\))?)"
    r"|(?:Bank\s+Secrecy\s+Act|Money\s+Laundering\s+Regulations|"
    r"(?:Fourth|Fifth|Sixth)\s+(?:EU\s+)?(?:Anti-Money\s+Laundering|AML)\s+Directive)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AttributionRule:
    """One surface cue binding a quantity to a fact field.

    Attributes:
        name: Stable identifier, reported when the rule fires so a systematic
            mis-attribution can be traced to one rule rather than to "the extractor".
        field_path: The fact field the quantity is a claim about.
        claim_type: Which tolerance rule applies.
        kind: How to parse the matched text — ``"count"``, ``"ratio"``, ``"percent"``,
            ``"money"``, ``"duration"`` or ``"timestamp"``.
        pattern: A regex over the narrative carrying a named ``value`` group, and for
            money a named ``currency`` group and for durations a named ``unit`` group.
            The ``value`` group's span is what the rule is keyed on, so a rule fires only
            for the exact quantity its cue is adjacent to.
    """

    name: str
    field_path: str
    claim_type: ClaimType
    kind: str
    pattern: re.Pattern[str]


def _rule(
    name: str,
    field_path: str,
    kind: str,
    pattern: str,
    *,
    claim_type: ClaimType = ClaimType.NUMERIC,
) -> AttributionRule:
    """Compile one attribution rule.

    Args:
        name: The rule identifier.
        field_path: The fact field.
        kind: The parse kind.
        pattern: The regex source. Must carry a ``value`` group.
        claim_type: Which tolerance rule applies.

    Returns:
        The compiled rule.

    Raises:
        ValueError: If the pattern carries no ``value`` group. A rule without one would
            never fire and would look, in a table of forty, exactly like one that does.
    """
    compiled = re.compile(pattern, re.IGNORECASE)
    if "value" not in compiled.groupindex:
        raise ValueError(f"attribution rule {name!r} has no 'value' group")
    return AttributionRule(
        name=name, field_path=field_path, claim_type=claim_type, kind=kind, pattern=compiled
    )


#: The cue table. Ordered: the first rule whose ``value`` group lands on a quantity wins,
#: so the specific cues come before the general ones. "9 inbound transactions" must reach
#: ``focal_entity.n_transactions_in`` before the bare "9 transactions" rule claims it for
#: ``structure.n_edges``.
DEFAULT_RULES: tuple[AttributionRule, ...] = (
    # --- focal entity degree and transaction counts -------------------------------
    _rule(
        "in_degree",
        "focal_entity.in_degree",
        "count",
        rf"received\s+(?:funds\s+)?from{_CUE_SLACK}?(?P<value>{_NUM})"
        rf"(?:\s+distinct)?\s+(?:counterpart|sender|source|origin)",
    ),
    _rule(
        "in_degree_senders",
        "focal_entity.in_degree",
        "count",
        rf"(?P<value>{_NUM})\s+(?:distinct\s+)?(?:inbound|sending|originating)\s+"
        rf"(?:counterpart|account|part)",
    ),
    _rule(
        "out_degree",
        "focal_entity.out_degree",
        "count",
        rf"(?:sent|dispersed|forwarded|paid|transferred)\s+(?:funds\s+)?"
        rf"(?:on(?:ward)?\s+)?to{_CUE_SLACK}?(?P<value>{_NUM})"
        rf"(?:\s+distinct)?\s+(?:counterpart|recipient|destination|beneficiar|account)",
    ),
    _rule(
        "out_degree_recipients",
        "focal_entity.out_degree",
        "count",
        rf"(?P<value>{_NUM})\s+(?:distinct\s+)?(?:outbound|receiving|downstream|recipient)\s+"
        rf"(?:counterpart|account|part)",
    ),
    _rule(
        "n_transactions_in",
        "focal_entity.n_transactions_in",
        "count",
        rf"(?P<value>{_NUM})\s+inbound\s+(?:transaction|transfer|payment)",
    ),
    _rule(
        "n_transactions_out",
        "focal_entity.n_transactions_out",
        "count",
        rf"(?P<value>{_NUM})\s+outbound\s+(?:transaction|transfer|payment)",
    ),
    # --- labels --------------------------------------------------------------------
    _rule(
        "n_illicit_counterparties",
        "labels.n_illicit_counterparties",
        "count",
        rf"(?P<value>{_NUM})\s+(?:of\s+(?:them|which)\s+)?"
        rf"(?:are|were|have\s+been)?\s*(?:previously\s+)?"
        rf"(?:associated|linked|connected)\s+with\s+(?:transactions\s+)?"
        rf"(?:previously\s+)?(?:flagged|identified|associated)",
    ),
    _rule(
        "n_illicit_transactions",
        "labels.n_illicit_transactions",
        "count",
        rf"(?P<value>{_NUM})\s+transactions?\s+(?:in\s+the\s+subgraph\s+)?"
        rf"(?:carry|carries|bear|bears|hold|holds)\s+a",
    ),
    _rule(
        "n_counterparties",
        "labels.n_counterparties",
        "count",
        rf"of\s+(?P<value>{_NUM})\s+counterpart",
    ),
    _rule(
        "illicit_inflow_share",
        "labels.illicit_inflow_share",
        "percent",
        rf"(?P<value>{_NUM}%)\s+of\s+(?:the\s+)?(?:value\s+)?(?:received|inflow|incoming)",
    ),
    # --- flow ----------------------------------------------------------------------
    _rule(
        "total_inflow",
        "flow.total_inflow",
        "money",
        rf"(?:received|inflow|took\s+in|incoming|credited)"
        rf"(?:[^.\d]{{0,40}})(?P<value>{_NUM})\s+(?P<currency>{_CUR})",
    ),
    _rule(
        "total_outflow",
        "flow.total_outflow",
        "money",
        rf"(?:sent|dispersed|outflow|paid\s+out|onward|outgoing|debited)"
        rf"(?:[^.\d]{{0,40}})(?P<value>{_NUM})\s+(?P<currency>{_CUR})",
    ),
    _rule(
        "max_single_transfer",
        "flow.max_single_transfer",
        "money",
        rf"(?:largest|biggest|single\s+largest)(?:[^.\d]{{0,40}})"
        rf"(?P<value>{_NUM})\s+(?P<currency>{_CUR})",
    ),
    _rule(
        "n_distinct_banks",
        "flow.n_distinct_banks",
        "count",
        rf"(?P<value>{_NUM})\s+(?:distinct\s+|separate\s+)?(?:institution|bank)",
    ),
    _rule(
        "n_transfers_near_threshold",
        "flow.n_transfers_near_threshold",
        "count",
        rf"(?P<value>{_NUM})\s+transfers?\s+(?:fall|fell|sit|sits|lie|lies|were|are)"
        rf"[^.]{{0,30}}\bbelow\b",
    ),
    # --- temporal ------------------------------------------------------------------
    _rule(
        "burst_txn_count",
        "temporal.burst_txn_count",
        "count",
        rf"(?P<value>{_NUM})\s+transactions?\s+within\s+{_NUM}",
    ),
    _rule(
        "burst_window_hours",
        "temporal.burst_window_hours",
        "duration",
        rf"within\s+(?P<value>{_NUM})\s+(?P<unit>minutes?|hours?|days?)",
    ),
    _rule(
        "span_hours",
        "temporal.span_hours",
        "duration",
        rf"(?:span(?:ning|s|ned)?|period|over|covering|extends?\s+over|runs?)"
        rf"(?:\s+of)?\s+(?P<value>{_NUM})\s+(?P<unit>minutes?|hours?|days?)",
    ),
    _rule(
        "window_start",
        "temporal.window_start",
        "timestamp",
        r"(?:from|beginning|opening|starting|between)\s+"
        r"(?P<value>\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?)",
        claim_type=ClaimType.TEMPORAL,
    ),
    _rule(
        "window_end",
        "temporal.window_end",
        "timestamp",
        r"(?:to|until|through|ending|closing)\s+"
        r"(?P<value>\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?)",
        claim_type=ClaimType.TEMPORAL,
    ),
    _rule(
        "n_transactions",
        "temporal.n_transactions",
        "count",
        rf"(?P<value>{_NUM})\s+transactions?\s+(?:in|within|across)\s+the\s+"
        rf"(?:review\s+)?window",
    ),
    # --- motifs --------------------------------------------------------------------
    _rule(
        "fan_out_width",
        "motifs.fan_out.width",
        "count",
        rf"(?:fan(?:ned|s)?\s+out|dispersed|distributed|scattered|split)"
        rf"(?:[^.\d]{{0,30}})(?P<value>{_NUM})",
    ),
    _rule(
        "fan_in_width",
        "motifs.fan_in.width",
        "count",
        rf"(?:fan(?:ned|s)?\s+in|gathered|aggregated|consolidated|collected)"
        rf"(?:[^.\d]{{0,30}})(?P<value>{_NUM})",
    ),
    _rule(
        "chain_max_length",
        "motifs.chain.max_length",
        "count",
        rf"chain\s+(?:of|with|spanning|running)\s+(?P<value>{_NUM})",
    ),
    _rule(
        "cycle_length",
        "motifs.cycle.length",
        "count",
        rf"cycle\s+(?:of|with|spanning|length|involving)\s*(?P<value>{_NUM})",
    ),
    _rule(
        "stack_depth",
        "motifs.stack.depth",
        "count",
        rf"(?:stack|layer(?:ing|ed)?)\s+(?:of|with|to\s+a\s+depth\s+of|depth)\s*(?P<value>{_NUM})",
    ),
    _rule(
        "bipartite_left",
        "motifs.bipartite.left_size",
        "count",
        rf"(?P<value>{_NUM})\s+(?:source|sending|left)\s+accounts?",
    ),
    _rule(
        "bipartite_right",
        "motifs.bipartite.right_size",
        "count",
        rf"(?P<value>{_NUM})\s+(?:destination|receiving|right)\s+accounts?",
    ),
    # --- structure. Last: the most general cues, so a specific rule wins first. -----
    _rule(
        "n_nodes",
        "structure.n_nodes",
        "count",
        rf"(?P<value>{_NUM})\s+(?:distinct\s+|separate\s+|further\s+)?accounts?\b",
    ),
    _rule(
        "n_edges",
        "structure.n_edges",
        "count",
        rf"(?P<value>{_NUM})\s+(?:recorded\s+)?(?:transactions?|transfers?)\b",
    ),
    _rule(
        "density",
        "structure.density",
        "ratio",
        rf"densit(?:y|ies)\s+(?:of\s+)?(?P<value>{_NUM})",
    ),
)


@dataclass(frozen=True)
class DeterministicReport:
    """What Method A found for one narrative.

    Attributes:
        case_id: The case.
        claims: Every claim, in document order.
        aligned_paths: Fact fields the narrative demonstrably preserved from Bronze.
        dropped_paths: Fields Bronze asserted that this narrative does not carry. Feeds
            Fact Coverage, not the hallucination count — dropping a fact is an editorial
            choice unless the field is salient.
        attributed: ``(rule name, field path, text)`` for each quantity a cue rule bound
            to a field. Reported so a mis-attribution is traceable to one rule.
        unattributed: ``(start, end, text)`` for each quantity that matched no cue and no
            slot. These stay UNVERIFIABLE.
        unparseable: Slot values found in the text but not readable back as claims. A
            formatter/parser inconsistency, never a model error.
        method: :data:`EXTRACTOR_METHOD`.
    """

    case_id: str
    claims: tuple[Claim, ...]
    aligned_paths: tuple[str, ...] = ()
    dropped_paths: tuple[str, ...] = ()
    attributed: tuple[tuple[str, str, str], ...] = ()
    unattributed: tuple[tuple[int, int, str], ...] = ()
    unparseable: tuple[str, ...] = ()
    method: str = EXTRACTOR_METHOD

    def to_dict(self) -> dict[str, Any]:
        """Return the report without the claims, which serialise separately.

        Returns:
            The diagnostics, JSON-serialisable.
        """
        return {
            "case_id": self.case_id,
            "method": self.method,
            "n_claims": len(self.claims),
            "aligned_paths": list(self.aligned_paths),
            "dropped_paths": list(self.dropped_paths),
            "attributed": [
                {"rule": rule, "field_path": path, "text": text}
                for rule, path, text in self.attributed
            ],
            "unattributed": [
                {"start": start, "end": end, "text": text} for start, end, text in self.unattributed
            ],
            "unparseable": list(self.unparseable),
        }


def _within(span: tuple[int, int], spans: Sequence[tuple[int, int]]) -> bool:
    """Report whether a span sits wholly inside any of a set of spans.

    Args:
        span: The candidate.
        spans: The containers.

    Returns:
        True when some container encloses the candidate entirely. Containment rather
        than overlap: a quantity that merely brushes a citation is still a quantity.
    """
    return any(start <= span[0] and span[1] <= end for start, end in spans)


def _empty_bronze(case_id: str) -> BronzeNarrative:
    """Build a slot-free Bronze reference, for a case whose Bronze narrative is absent.

    Alignment against it finds nothing, which is the honest behaviour: with no reference
    spans, every quantity in the narrative is a residual and goes through attribution or
    stays unverifiable. The vocabulary, entity and regulatory passes still run, so a
    reference-free extraction is weaker but never wrong.

    Args:
        case_id: The case.

    Returns:
        A Bronze narrative with no text and no slots.
    """
    return BronzeNarrative(
        case_id=case_id, text="", annotated="", slots=(), family="none", variant=0
    )


def _parse_value(  # noqa: PLR0911 -- one return per parse kind; a dispatch table would
    # separate each kind from the parser that owns it for no gain.
    kind: str,
    text: str,
    match: re.Match[str],
) -> tuple[float, str] | DurationClaim | datetime | float | int | None:
    """Turn matched text into the value shape the claim's checker expects.

    Args:
        kind: The rule's parse kind.
        text: The matched ``value`` text.
        match: The whole rule match, for the ``currency`` and ``unit`` groups.

    Returns:
        The parsed value, or None when the text does not parse as its kind — which is a
        rule matching something it should not, and produces no claim rather than a claim
        about a number nobody wrote.
    """
    try:
        if kind == "count":
            return parse_count(text)
        if kind == "percent":
            return parse_percent(text)
        if kind == "ratio":
            return float(text.replace(",", ""))
        if kind == "money":
            currency = match.groupdict().get("currency")
            if not currency:
                return None
            return (float(text.replace(",", "")), currency.strip())
        if kind == "duration":
            value, unit = parse_duration(f"{text} {match.groupdict().get('unit', 'hours')}")
            return DurationClaim(value=value, unit=unit)
        if kind == "timestamp":
            return parse_timestamp(text.replace("T", " "))
    except (FormatError, ValueError):
        return None
    return None


class DeterministicClaimExtractor:
    """Method A: Bronze slot alignment, vocabulary rules, and cue-based attribution."""

    def __init__(
        self,
        *,
        bronze: BronzeNarrative | None = None,
        vocabulary: ControlledVocabulary | None = None,
        rules: Sequence[AttributionRule] = DEFAULT_RULES,
    ) -> None:
        """Bind the extractor to a reference narrative and a rule table.

        Args:
            bronze: The Bronze narrative for this case, whose slot annotation is the
                alignment reference. Omitted for a case with no Bronze rendering, which
                weakens the extractor to rules alone rather than failing.
            vocabulary: The controlled vocabulary. Loaded from disk when omitted.
            rules: The attribution table. Overridable so a test can exercise one rule in
                isolation, and so a substrate with different phrasings can supply its own.
        """
        self.bronze = bronze
        self.vocabulary = vocabulary if vocabulary is not None else load_vocabulary()
        self.rules = tuple(rules)

    def extract(self, narrative: str, facts: CaseFacts) -> list[Claim]:
        """Extract every checkable claim the narrative makes.

        Args:
            narrative: The narrative text.
            facts: The record it is about. Read for the case id and for typology surface
                forms — **never** as a source of claim values (D-040).

        Returns:
            The claims, in document order.
        """
        return list(self.report(narrative, facts).claims)

    def report(self, narrative: str, facts: CaseFacts) -> DeterministicReport:
        """Extract claims and the diagnostics around them.

        Args:
            narrative: The narrative text.
            facts: The record it is about.

        Returns:
            The full report.
        """
        reference = self.bronze if self.bronze is not None else _empty_bronze(facts.case_id)
        base: ExtractionReport = SlotAlignmentExtractor(
            reference, vocabulary=self.vocabulary
        ).report(narrative, facts)

        index = self._rule_index(narrative)
        claims: list[Claim] = []
        attributed: list[tuple[str, str, str]] = []
        unattributed: list[tuple[int, int, str]] = []
        residual = {(start, end) for start, end, _ in base.added_spans}

        # Citation spans are computed first because a figure *inside* a citation is part
        # of the citation, not a separate unbacked quantity. Without this, "the USD
        # 42,000 mandatory disclosure threshold" leaves 42,000 as a residual number, the
        # citation span overlaps it and is skipped, and the invented rule -- H6, Critical
        # -- disappears into a generic unsupported-inference finding.
        regulation_spans = [m.span() for m in _REGULATION_RE.finditer(narrative)]

        for claim in base.claims:
            if claim.claim_type is not ClaimType.REGULATORY and _within(
                claim.text_span, regulation_spans
            ):
                continue
            if claim.text_span not in residual:
                claims.append(claim)
                continue
            hit = index.get(claim.text_span)
            if hit is None:
                claims.append(claim)
                unattributed.append((claim.text_span[0], claim.text_span[1], claim.raw_text))
                continue
            rule, match = hit
            value = _parse_value(
                rule.kind, narrative[claim.text_span[0] : claim.text_span[1]], match
            )
            if value is None:
                claims.append(claim)
                unattributed.append((claim.text_span[0], claim.text_span[1], claim.raw_text))
                continue
            claims.append(
                Claim(
                    text_span=claim.text_span,
                    field_path=rule.field_path,
                    claim_type=rule.claim_type,
                    value=value,
                    raw_text=narrative[match.start() : match.end()],
                )
            )
            attributed.append((rule.name, rule.field_path, claim.raw_text))

        claims.extend(self._typology_claims(narrative, facts, claims))
        claims.extend(self._invented_regulation_claims(regulation_spans, narrative, claims))
        claims.sort(key=lambda c: c.text_span)

        return DeterministicReport(
            case_id=facts.case_id,
            claims=tuple(claims),
            aligned_paths=base.aligned_paths,
            dropped_paths=base.dropped_paths,
            attributed=tuple(attributed),
            unattributed=tuple(unattributed),
            unparseable=base.unparseable,
        )

    def _invented_regulation_claims(
        self,
        spans: Sequence[tuple[int, int]],
        narrative: str,
        existing: Sequence[Claim],
    ) -> list[Claim]:
        """Emit a regulatory claim for any citation the whitelist pass did not consume.

        H6 is Critical and it is the class the paper leans on hardest, so it cannot
        depend on a phrase appearing in a forbidden list — the whole failure mode is a
        model **inventing** a rule, and an invented rule is by definition not on any list
        written in advance.

        The Phase 5 pass matches only *whitelisted* citations, deliberately, so that it
        can never launder one. That leaves the complement uncovered: "the USD 42,000
        mandatory disclosure threshold" produced one unbacked quantity and no regulatory
        claim at all, so the single most damaging class in the taxonomy scored as a
        generic unsupported inference. This pass closes it by matching the *shape* of a
        citation rather than its content, and letting
        :func:`~g2t_aml.facts.checkers.check_regulatory` decide: whitelisted is
        SUPPORTED, anything else is CONTRADICTED as H6.

        Args:
            spans: Citation-shaped spans found in the narrative.
            narrative: The narrative.
            existing: Claims already extracted. A whitelisted citation was consumed by
                the earlier pass and already carries a REGULATORY claim, so it is not
                re-emitted and cannot be double-counted.

        Returns:
            One REGULATORY claim per citation-shaped phrase with no claim on it yet.
        """
        taken = [c.text_span for c in existing]
        claims: list[Claim] = []
        for span in spans:
            if any(span[0] < end and start < span[1] for start, end in taken):
                continue
            taken.append(span)
            text = narrative[span[0] : span[1]]
            claims.append(
                Claim(
                    text_span=span,
                    field_path=None,
                    claim_type=ClaimType.REGULATORY,
                    value=text,
                    raw_text=text,
                )
            )
        return claims

    def _rule_index(
        self, narrative: str
    ) -> dict[tuple[int, int], tuple[AttributionRule, re.Match[str]]]:
        """Index every rule hit by the span of the quantity it attributes.

        Rules are applied in table order and the first hit on a span wins, so a specific
        cue beats a general one. Building an index rather than scanning per residual
        keeps the pass linear in the rule count instead of quadratic.

        Args:
            narrative: The narrative to scan.

        Returns:
            Quantity span to the rule that claims it and the match that found it.
        """
        index: dict[tuple[int, int], tuple[AttributionRule, re.Match[str]]] = {}
        for rule in self.rules:
            for match in rule.pattern.finditer(narrative):
                span = match.span("value")
                if span not in index:
                    index[span] = (rule, match)
        return index

    def _typology_claims(
        self, narrative: str, facts: CaseFacts, existing: Sequence[Claim]
    ) -> list[Claim]:
        """Emit a categorical claim for every laundering typology the narrative names.

        Without this pass a narrative that calls a fan-out case a cycle makes no claim at
        all: the word aligns to no slot, contains no digits and is not a controlled risk
        descriptor. It would be invisible, and naming the wrong scheme is H5 — one of the
        errors an investigator would notice first.

        Args:
            narrative: The narrative.
            facts: The record, read for the substrate whose typology vocabulary applies.
            existing: Claims already extracted, so a typology Bronze rendered into a slot
                is not counted twice.

        Returns:
            One CATEGORICAL claim per distinct typology named, excluding spans already
            claimed.
        """
        taken = [c.text_span for c in existing]
        haystack = narrative.lower()
        claims: list[Claim] = []
        seen: set[str] = set()
        for label in self._typology_labels(facts):
            surface = label.replace("_", " ")
            for needle in (surface, label):
                start = haystack.find(needle)
                if start < 0:
                    continue
                span = (start, start + len(needle))
                if any(span[0] < end and s < span[1] for s, end in taken) or label in seen:
                    continue
                seen.add(label)
                taken.append(span)
                claims.append(
                    Claim(
                        text_span=span,
                        field_path="typology.label",
                        claim_type=ClaimType.CATEGORICAL,
                        value=label,
                        raw_text=narrative[span[0] : span[1]],
                    )
                )
                break
        return claims

    def _typology_labels(self, facts: CaseFacts) -> tuple[str, ...]:
        """Return the typology labels this substrate can carry.

        Args:
            facts: The record, whose ``dataset`` selects the vocabulary block.

        Returns:
            The labels, longest first so ``gather_scatter`` is matched before ``scatter``.
        """
        block = self.vocabulary.typologies.get(facts.dataset) or {}
        labels: list[str] = [str(label) for label in block.get("members") or ()]
        if not labels:
            # An unknown substrate key: fall back to the union, because scoring a claim
            # against the wrong substrate's vocabulary is better than not scoring it. The
            # substrate itself is checked by `check_typology`, which owns that verdict.
            labels = sorted(
                {
                    str(label)
                    for entry in self.vocabulary.typologies.values()
                    for label in entry.get("members") or ()
                }
            )
        # "unclassified" is the absence of a typology, not a claim to name one.
        return tuple(
            sorted((label for label in labels if label != "unclassified"), key=len, reverse=True)
        )


def extract_claims(
    narrative: str,
    facts: CaseFacts,
    *,
    bronze: BronzeNarrative | None = None,
    vocabulary: ControlledVocabulary | None = None,
) -> DeterministicReport:
    """Run Method A over one narrative.

    Args:
        narrative: The narrative text.
        facts: The record it is about.
        bronze: The Bronze narrative for this case, when one exists.
        vocabulary: The controlled vocabulary. Loaded from disk when omitted.

    Returns:
        The extraction report.
    """
    return DeterministicClaimExtractor(bronze=bronze, vocabulary=vocabulary).report(
        narrative, facts
    )
