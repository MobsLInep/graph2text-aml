"""The ten-point validation harness. It gates Bronze, Silver and Gold, identically.

Built once, properly, because it is the only thing standing between a corpus and the
paper. A record passes only if **all ten** hold:

===  ==========================================================================
 1   Valid against ``schemas/training_record_v1.json`` (jsonschema, strict)
 2   ``graph_ref`` resolves, and to the graph the facts describe
 3   ``facts.schema_version`` matches the current pinned version
 4   **Zero CONTRADICTED claims** against its own fact record
 5   ``unverifiable_rate <= 0.05``
 6   Narrative length within [80, 400] tokens
 7   No vocabulary violations: no out-of-substrate entity type, no forbidden-list
     guilt assertion, no risk descriptor whose quantitative binding fails
 8   No PII, no real-world identifiers, no leaked source-data strings
 9   Deduplication passed (MinHash LSH, exact Jaccard <= 0.85 corpus-wide)
10   Split assignment consistent with the frozen Phase 2 manifest
===  ==========================================================================

**Check 4 is recomputed, never read.** The record carries a ``verification`` block written
at build time, and this harness ignores it and re-derives the verdicts from the narrative's
own slots. Trusting a self-reported number would make the gate a formality — the generator
that wrote the narrative also wrote the number.

**Checks 9 and 10 are corpus-level, and are computed once for the whole corpus rather than
per record.** Deduplication is meaningless against a single record, and split consistency
needs the manifest. They are folded into the same report so a caller has one gate rather
than three.

**On not weakening the harness.** Bronze failing a check is a bug in the renderer or in the
fact layer, and the fix belongs there. The most tempting relaxation is check 5 — the
unverifiable budget — because an unverifiable claim feels harmless. It is not: UNVERIFIABLE
is the bucket that collects compliance-dangerous assertions the graph cannot back, and a
corpus that trains a model to produce them has taught the exact failure this project exists
to detect.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

from g2t_aml.corpus.claims import ClaimParseError, claims_from_slots
from g2t_aml.corpus.dedupe import DEFAULT_THRESHOLD, DuplicateReport, find_near_duplicates
from g2t_aml.corpus.factsio import FactsIOError, load_case_facts
from g2t_aml.corpus.graphref import GraphRefError, GraphRefResolver
from g2t_aml.corpus.pii import scan_for_identifiers
from g2t_aml.corpus.record import SlotAnnotation, validate_training_record
from g2t_aml.corpus.tokenization import TokenCounter, get_token_counter, word_count
from g2t_aml.facts.checkers import CheckContext, Verdict, check_claim, check_narrative_text
from g2t_aml.facts.schema import CASE_FACTS_SCHEMA_VERSION
from g2t_aml.facts.vocab import ControlledVocabulary, load_vocabulary

__all__ = [
    "CHECKS",
    "MAX_UNVERIFIABLE_RATE",
    "ValidationReport",
    "validate_corpus",
]

#: The ten checks, in the order the report presents them. Names are stable: they key the
#: machine-readable report and are quoted in the phase log.
CHECKS: tuple[str, ...] = (
    "schema_valid",
    "graph_ref_resolves",
    "facts_schema_version",
    "zero_contradicted",
    "unverifiable_rate",
    "length_in_bounds",
    "vocabulary_clean",
    "no_pii_or_identifiers",
    "deduplicated",
    "split_consistent",
)

#: The unverifiable budget. 5% of claims, per record.
MAX_UNVERIFIABLE_RATE = 0.05

#: Default token bounds. Imported here rather than from the renderer so the harness does
#: not depend on the Bronze package: it gates Silver and Gold too, and they have no
#: renderer.
MIN_TOKENS = 80
MAX_TOKENS = 400

_EXAMPLES_PER_CHECK = 8


@dataclass
class ValidationReport:
    """The result of gating a corpus.

    Attributes:
        total: Records examined.
        passed: Records passing all ten checks.
        failures_by_check: Check name to how many records failed it. A record failing two
            checks is counted under both, so the values sum to more than ``total -
            passed``.
        failure_examples: Check name to a sample of failing case ids, with the reason.
        unverifiable_rate_distribution: Summary statistics over per-record unverifiable
            rates.
        length_distribution: Summary statistics over per-record token counts.
        duplicates: The corpus-level near-duplicate report.
        by_tier: Records seen per tier, so a mixed corpus reports honestly.
        by_family: Records seen per template family.
        verification_totals: Claim verdict counts summed over the corpus.
    """

    total: int = 0
    passed: int = 0
    failures_by_check: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CHECKS, 0))
    failure_examples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    unverifiable_rate_distribution: dict[str, float] = field(default_factory=dict)
    length_distribution: dict[str, float] = field(default_factory=dict)
    duplicates: DuplicateReport | None = None
    by_tier: dict[str, int] = field(default_factory=dict)
    by_family: dict[str, int] = field(default_factory=dict)
    verification_totals: dict[str, int] = field(default_factory=dict)

    @property
    def failed(self) -> int:
        """Return how many records failed at least one check.

        Returns:
            The failure count.
        """
        return self.total - self.passed

    @property
    def gate_passed(self) -> bool:
        """Report whether the corpus may proceed.

        Returns:
            True when every record passed every check. There is no partial credit: a
            corpus with nine bad records in fifteen thousand is a corpus with a bug.
        """
        return self.total > 0 and self.passed == self.total

    def to_dict(self) -> dict[str, Any]:
        """Return the machine-readable report.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "gate_passed": self.gate_passed,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.passed / self.total, 6) if self.total else 0.0,
            "failures_by_check": dict(self.failures_by_check),
            "failure_examples": {k: v for k, v in self.failure_examples.items() if v},
            "unverifiable_rate_distribution": self.unverifiable_rate_distribution,
            "length_distribution": self.length_distribution,
            "duplicates": self.duplicates.to_dict() if self.duplicates else None,
            "by_tier": dict(sorted(self.by_tier.items())),
            "by_family": dict(sorted(self.by_family.items())),
            "verification_totals": dict(sorted(self.verification_totals.items())),
            "thresholds": {
                "max_unverifiable_rate": MAX_UNVERIFIABLE_RATE,
                "min_tokens": MIN_TOKENS,
                "max_tokens": MAX_TOKENS,
                "dedup_jaccard": DEFAULT_THRESHOLD,
                "case_facts_schema_version": CASE_FACTS_SCHEMA_VERSION,
            },
        }

    def summary(self) -> str:
        """Return the human-readable summary.

        Returns:
            A short report suitable for a terminal or a phase log.
        """
        lines = [
            f"ten-point validation: {self.passed}/{self.total} passed "
            f"({'GATE PASSED' if self.gate_passed else 'GATE FAILED'})",
            "",
            f"  {'check':<24} {'failed':>8}",
            f"  {'-' * 24} {'-' * 8}",
        ]
        for i, check in enumerate(CHECKS, start=1):
            count = self.failures_by_check.get(check, 0)
            marker = "  " if count == 0 else " !"
            lines.append(f" {marker}{i:>2}. {check:<20} {count:>8}")
        if self.length_distribution:
            distribution = self.length_distribution
            lines += [
                "",
                f"  length tokens   min {distribution['min']:.0f}  p05 "
                f"{distribution['p05']:.0f}  median {distribution['median']:.0f}  p95 "
                f"{distribution['p95']:.0f}  max {distribution['max']:.0f}",
            ]
        if self.unverifiable_rate_distribution:
            distribution = self.unverifiable_rate_distribution
            lines.append(
                f"  unverifiable    mean {distribution['mean']:.4f}  max "
                f"{distribution['max']:.4f}  budget {MAX_UNVERIFIABLE_RATE}"
            )
        if self.duplicates is not None:
            lines.append(
                f"  duplicates      {self.duplicates.n_dropped} dropped at Jaccard "
                f">= {self.duplicates.threshold}"
            )
        if self.verification_totals:
            totals = self.verification_totals
            claims = sum(totals.values()) or 1
            lines.append(
                f"  claims          {claims:,}  supported "
                f"{totals.get('supported', 0) / claims:.4f}  contradicted "
                f"{totals.get('contradicted', 0)}  unverifiable "
                f"{totals.get('unverifiable', 0)}"
            )
        for check, examples in self.failure_examples.items():
            if examples:
                lines += ["", f"  {check}:"] + [f"    - {e}" for e in examples]
        return "\n".join(lines)


def _record_failure(report: ValidationReport, check: str, case_id: str, reason: str) -> None:
    """Record one check failure.

    Args:
        report: The report to update.
        check: The check that failed.
        case_id: The record.
        reason: Why.
    """
    report.failures_by_check[check] = report.failures_by_check.get(check, 0) + 1
    if len(report.failure_examples[check]) < _EXAMPLES_PER_CHECK:
        report.failure_examples[check].append(f"{case_id}: {reason}")


def _distribution(values: list[float]) -> dict[str, float]:
    """Summarise a distribution.

    Args:
        values: The observations.

    Returns:
        Min, five percentiles, mean and max. Empty input yields an empty mapping rather
        than zeros, so an absent distribution is distinguishable from a zero one.
    """
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)

    def percentile(p: float) -> float:
        return ordered[min(n - 1, int(p * n))]

    return {
        "n": float(n),
        "min": ordered[0],
        "p05": percentile(0.05),
        "median": percentile(0.5),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / n,
    }


def _check_record(  # noqa: PLR0912, PLR0915 -- one linear block per numbered check, in
    # the order the report presents them. Splitting it would separate a check from the
    # failure it records, and the ten are meant to be read against the module docstring.
    payload: dict[str, Any],
    report: ValidationReport,
    *,
    resolver: GraphRefResolver | None,
    split_manifest: dict[str, str] | None,
    vocabulary: ControlledVocabulary,
    counter: TokenCounter,
    duplicate_ids: frozenset[str],
    lengths: list[float],
    rates: list[float],
    verdict_totals: Counter[str],
) -> bool:
    """Run checks 1-8 and 10 on one record, plus the corpus-level verdict from check 9.

    Args:
        payload: The serialised training record.
        report: The report to update.
        resolver: Resolves ``graph_ref``. When None, check 2 is failed rather than
            skipped: a harness that quietly drops a check reports a pass it did not test.
        split_manifest: Case id to split, from the frozen Phase 2 manifest. When None,
            check 10 is failed for the same reason.
        vocabulary: The controlled vocabulary.
        counter: The token counter.
        duplicate_ids: Records the corpus-level dedup pass flagged.
        lengths: Accumulates token counts.
        rates: Accumulates per-record unverifiable rates.
        verdict_totals: Accumulates claim verdicts.

    Returns:
        True when all ten checks pass for this record.
    """
    case_id = str(payload.get("case_id", "<unknown>"))
    ok = True

    # 1. schema
    try:
        validate_training_record(payload)
    except jsonschema.ValidationError as exc:
        _record_failure(
            report, "schema_valid", case_id, f"{'.'.join(str(p) for p in exc.path)}: {exc.message}"
        )
        ok = False
        return ok  # nothing below can be trusted on a record that is not the right shape

    report.by_tier[payload["tier"]] = report.by_tier.get(payload["tier"], 0) + 1
    family = str(payload.get("generator", {}).get("family", "-"))
    report.by_family[family] = report.by_family.get(family, 0) + 1

    # 3. fact schema version, before anything reads the facts
    declared = str(payload["facts"].get("schema_version", ""))
    if declared != CASE_FACTS_SCHEMA_VERSION:
        _record_failure(
            report,
            "facts_schema_version",
            case_id,
            f"facts declare {declared!r}, code is frozen at {CASE_FACTS_SCHEMA_VERSION!r}",
        )
        ok = False

    # 2. graph reference
    if resolver is None:
        _record_failure(report, "graph_ref_resolves", case_id, "no case store was provided")
        ok = False
    else:
        try:
            resolver.check(
                str(payload["graph_ref"]),
                int(payload["facts"]["structure"]["n_nodes"]),
                int(payload["facts"]["structure"]["n_edges"]),
            )
        except GraphRefError as exc:
            _record_failure(report, "graph_ref_resolves", case_id, str(exc))
            ok = False

    narrative = str(payload["target_narrative"])

    # 4, 5, 7: everything that needs the fact record and the checker.
    try:
        facts = load_case_facts(payload["facts"])
    except FactsIOError as exc:
        _record_failure(report, "facts_schema_version", case_id, str(exc))
        return False

    context = CheckContext(facts=facts, vocabulary=vocabulary)
    slots = [SlotAnnotation.from_dict(s) for s in payload["target_slots"]]
    try:
        claims = claims_from_slots(slots, narrative)
    except ClaimParseError as exc:
        _record_failure(report, "zero_contradicted", case_id, str(exc))
        return False

    results = [check_claim(claim, context) for claim in claims]
    text_results = check_narrative_text(narrative, context)

    contradicted = [r for r in results + text_results if r.verdict is Verdict.CONTRADICTED]
    unverifiable = [r for r in results if r.verdict is Verdict.UNVERIFIABLE]
    for result in results:
        verdict_totals[result.verdict.value] += 1

    if contradicted:
        _record_failure(
            report,
            "zero_contradicted",
            case_id,
            f"{len(contradicted)} contradicted, first: {contradicted[0].reason}",
        )
        ok = False

    rate = len(unverifiable) / len(results) if results else 0.0
    rates.append(rate)
    if rate > MAX_UNVERIFIABLE_RATE:
        _record_failure(
            report,
            "unverifiable_rate",
            case_id,
            f"{rate:.3f} exceeds {MAX_UNVERIFIABLE_RATE} "
            f"({len(unverifiable)}/{len(results)} claims)",
        )
        ok = False

    # 7. vocabulary. A forbidden phrase or a failed descriptor binding is CONTRADICTED
    # above; this reports it under its own check so the two are separable in the report.
    vocabulary_hits = [r for r in text_results if r.verdict is Verdict.CONTRADICTED] + [
        r
        for r in results
        if r.verdict is Verdict.CONTRADICTED and r.claim.claim_type.value == "qualitative"
    ]
    if vocabulary_hits:
        _record_failure(report, "vocabulary_clean", case_id, vocabulary_hits[0].reason)
        ok = False

    # 6. length
    tokens = counter.count(narrative)
    lengths.append(float(tokens))
    if not MIN_TOKENS <= tokens <= MAX_TOKENS:
        _record_failure(
            report,
            "length_in_bounds",
            case_id,
            f"{tokens} tokens outside [{MIN_TOKENS}, {MAX_TOKENS}] "
            f"({word_count(narrative)} words, counter {counter.name})",
        )
        ok = False

    # 8. PII and real-world identifiers
    if hits := scan_for_identifiers(narrative):
        _record_failure(
            report,
            "no_pii_or_identifiers",
            case_id,
            f"{hits[0][0]}: {hits[0][1]!r}",
        )
        ok = False

    # 9. deduplication, decided corpus-wide
    if case_id in duplicate_ids:
        _record_failure(report, "deduplicated", case_id, "near-duplicate of another corpus record")
        ok = False

    # 10. split assignment
    if split_manifest is None:
        _record_failure(report, "split_consistent", case_id, "no split manifest was provided")
        ok = False
    else:
        expected = split_manifest.get(case_id)
        if expected is None:
            _record_failure(
                report,
                "split_consistent",
                case_id,
                "absent from the frozen split manifest, so its split is undefined",
            )
            ok = False
        elif expected != payload["split"]:
            _record_failure(
                report,
                "split_consistent",
                case_id,
                f"record says {payload['split']!r}, manifest says {expected!r}",
            )
            ok = False

    return ok


def validate_corpus(
    records: list[dict[str, Any]],
    *,
    repo_root: Path | None = None,
    split_manifest: dict[str, str] | None = None,
    vocabulary: ControlledVocabulary | None = None,
    token_counter: TokenCounter | None = None,
    dedup_threshold: float = DEFAULT_THRESHOLD,
) -> ValidationReport:
    """Gate a whole corpus against the ten checks.

    Args:
        records: Serialised training records.
        repo_root: Repository root, for resolving ``graph_ref``. Check 2 fails for every
            record when omitted.
        split_manifest: Case id to split from the frozen Phase 2 manifest. Check 10 fails
            for every record when omitted.
        vocabulary: The controlled vocabulary. Loaded from disk when omitted.
        token_counter: Counts tokens for check 6. Defaults to the heuristic counter.
        dedup_threshold: Exact-Jaccard threshold for check 9.

    Returns:
        The report. ``gate_passed`` is the single question a caller needs answered.
    """
    vocab = vocabulary if vocabulary is not None else load_vocabulary()
    counter = token_counter if token_counter is not None else get_token_counter()
    resolver = GraphRefResolver(repo_root=repo_root) if repo_root is not None else None

    report = ValidationReport(total=len(records))
    duplicates = find_near_duplicates(
        {
            str(r.get("case_id", i)): str(r.get("target_narrative", ""))
            for i, r in enumerate(records)
        },
        threshold=dedup_threshold,
    )
    report.duplicates = duplicates
    duplicate_ids = frozenset(duplicates.dropped)

    lengths: list[float] = []
    rates: list[float] = []
    verdict_totals: Counter[str] = Counter()

    for payload in records:
        if _check_record(
            payload,
            report,
            resolver=resolver,
            split_manifest=split_manifest,
            vocabulary=vocab,
            counter=counter,
            duplicate_ids=duplicate_ids,
            lengths=lengths,
            rates=rates,
            verdict_totals=verdict_totals,
        ):
            report.passed += 1

    report.length_distribution = _distribution(lengths)
    report.unverifiable_rate_distribution = _distribution(rates)
    report.verification_totals = dict(verdict_totals)
    return report


def load_split_manifest(manifest_dir: Path) -> dict[str, str]:
    """Read the frozen Phase 2 split manifest into a case-to-split mapping.

    Reads the committed id lists rather than ``splits.json`` so check 10 is answered by
    the same files invariant 2 protects.

    Args:
        manifest_dir: Directory holding ``train.txt``, ``val.txt`` and ``test.txt``.

    Returns:
        Case id to split name.

    Raises:
        FileNotFoundError: If a split file is missing.
        ValueError: If a case id appears in two splits, which would make the assignment
            ambiguous and the leakage audit meaningless.
    """
    assignment: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = manifest_dir / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"split manifest {path} is missing")
        for case_id in path.read_text(encoding="utf-8").split():
            if case_id in assignment:
                raise ValueError(
                    f"case {case_id!r} appears in both {assignment[case_id]!r} and "
                    f"{split!r}; the frozen manifest is inconsistent"
                )
            assignment[case_id] = split
    return assignment


def write_report(report: ValidationReport, path: Path) -> None:
    """Write the machine-readable report.

    Args:
        report: The report.
        path: Destination.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
