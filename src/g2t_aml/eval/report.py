"""One call in, three artifacts out: JSON for machines, markdown for people, LaTeX for the paper.

**Layer 2 leads, everywhere.** Zero-Hallucination Rate is the first number in the JSON, the
first table in the markdown and the first table in the LaTeX. Layer 1 appears below it,
under a heading that says what it is for. This is a reporting decision with a reason: the
order a results section is read in is the order its numbers are believed in, and an
evaluation whose first table is BLEU has told the reader which number matters before it
has argued for one.

**The two test streams never pool.** The balanced set and the realistic-imbalance stream
are different populations — the second is dominated by cases whose typology is
``unclassified`` — so a mean over both is a weighted average whose weights are a property
of the sampler. They are reported side by side and the report refuses to produce a
combined figure.

**The error analysis is a deliverable, not a debug aid.** The qualitative section of the
paper needs concrete narratives with their concrete violations, and picking them by hand
after reading the numbers is how a qualitative section becomes an illustration of the
conclusion. :func:`worst_cases` selects them mechanically, by a rule fixed here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from g2t_aml.corpus.record import BronzeNarrative
from g2t_aml.eval.claim_extraction.agreement import AgreementReport
from g2t_aml.eval.claim_extraction.deterministic import DeterministicClaimExtractor
from g2t_aml.eval.layer1_automatic import (
    Layer1Metrics,
    LearnedMetric,
    PerplexityModel,
    TemplateBaselineFinding,
    compute_layer1,
    template_baseline_finding,
)
from g2t_aml.eval.layer2_faithfulness import (
    CaseFaithfulness,
    FaithfulnessMetrics,
    aggregate_faithfulness,
    score_case,
)
from g2t_aml.eval.metric_validation import MetricValidationReport
from g2t_aml.eval.statistics import (
    BOOTSTRAP_RESAMPLES,
    Interval,
    PairedComparison,
    SeedSummary,
    bootstrap_ci,
    compare_systems,
    publication_table,
    seed_summary,
)
from g2t_aml.eval.taxonomy_scorer import TaxonomyReport, score_taxonomy
from g2t_aml.eval.types import ScoredCase, SystemOutput, pair_outputs_with_facts
from g2t_aml.facts.config import FactConfig
from g2t_aml.facts.schema import CaseFacts
from g2t_aml.facts.taxonomy import HallucinationClass
from g2t_aml.facts.vocab import ControlledVocabulary, load_vocabulary

__all__ = [
    "HEADLINE_METRIC",
    "PRIMARY_METRICS",
    "EvaluationReport",
    "SystemReport",
    "evaluate",
    "worst_cases",
]

#: The number the paper leads with. Named once, here, and read by the report, the LaTeX
#: emitter and the statistics call, so "the headline" cannot mean different things in
#: three places.
HEADLINE_METRIC = "zero_hallucination"

#: The metrics that get bootstrap intervals and pairwise significance tests. Layer 2
#: only: a corrected significance test on BLEU would give an overlap metric the same
#: statistical dress as the faithfulness metrics and invite the reader to weigh them
#: equally, which is the opposite of this project's argument.
PRIMARY_METRICS: tuple[str, ...] = (
    "zero_hallucination",
    "fact_precision",
    "hallucination_rate",
    "fact_coverage",
    "fact_f1",
    "unverifiable_rate",
)

#: How many worst cases to keep per system for the qualitative section.
_WORST_CASES_PER_SYSTEM = 5


def _metric_value(case: CaseFaithfulness, metric: str) -> float | None:
    """Read one per-case metric by name.

    Args:
        case: The per-case faithfulness.
        metric: One of :data:`PRIMARY_METRICS`, or any float/bool property on the case.

    Returns:
        The value as a float, or None when the metric is not defined for this case.
    """
    if metric == "zero_hallucination":
        return float(case.zero_hallucination)
    value = getattr(case, metric, None)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return None


@dataclass(frozen=True)
class SystemReport:
    """Everything measured about one system on one stream.

    Attributes:
        system: The arm.
        stream: ``"balanced"`` or ``"realistic"``.
        faithfulness: Layer 2, aggregated.
        taxonomy: The H1—H9 breakdown.
        layer1: Surface-overlap metrics, or None when no Gold reference exists.
        intervals: Metric name to its bootstrap interval.
        by_typology: Typology to that typology's Layer 2 aggregate.
        by_dataset: Substrate to its Layer 2 aggregate.
        seeds: Metric name to the across-seed summary.
        worst: The worst-scoring cases, for the qualitative section.
    """

    system: str
    stream: str
    faithfulness: FaithfulnessMetrics
    taxonomy: TaxonomyReport
    layer1: Layer1Metrics | None = None
    intervals: Mapping[str, Interval] = field(default_factory=dict)
    by_typology: Mapping[str, FaithfulnessMetrics] = field(default_factory=dict)
    by_dataset: Mapping[str, FaithfulnessMetrics] = field(default_factory=dict)
    seeds: Mapping[str, SeedSummary] = field(default_factory=dict)
    worst: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the system report as a JSON-serialisable mapping.

        Returns:
            Layer 2 first, then the taxonomy, then Layer 1 — the reading order the
            module docstring argues for, preserved in the file as well as in the prose.
        """
        return {
            "system": self.system,
            "stream": self.stream,
            "faithfulness": self.faithfulness.to_dict(),
            "taxonomy": self.taxonomy.to_dict(),
            "intervals": {k: v.to_dict() for k, v in sorted(self.intervals.items())},
            "seeds": {k: v.to_dict() for k, v in sorted(self.seeds.items())},
            "by_typology": {k: v.to_dict() for k, v in sorted(self.by_typology.items())},
            "by_dataset": {k: v.to_dict() for k, v in sorted(self.by_dataset.items())},
            "layer1": self.layer1.to_dict() if self.layer1 is not None else None,
            "worst_cases": list(self.worst),
        }


def worst_cases(
    cases: Sequence[CaseFaithfulness], *, n: int = _WORST_CASES_PER_SYSTEM
) -> tuple[dict[str, Any], ...]:
    """Select the worst-scoring narratives, with their violations.

    Ordered by contradicted claims descending, then critical findings descending, then
    Fact F1 ascending. Fixed here rather than chosen per system, because a selection rule
    picked after reading the results produces a qualitative section that illustrates the
    conclusion instead of testing it.

    Args:
        cases: The per-case results for one system.
        n: How many to keep.

    Returns:
        One mapping per case, carrying the metrics and every adverse finding with its
        reason and span, ready for the report.
    """
    ranked = sorted(
        cases,
        key=lambda c: (-c.n_contradicted, -c.n_critical, c.fact_f1, c.case_id),
    )
    out: list[dict[str, Any]] = []
    for case in ranked[:n]:
        violations = [
            {
                "verdict": result.verdict.value,
                "hallucination_class": result.hallucination_class,
                "field_path": result.claim.field_path,
                "text": result.claim.raw_text,
                "reason": result.reason,
                "span": list(result.claim.text_span),
                "producer": result.producer,
            }
            for result in case.results
            if result.verdict.value != "supported"
        ]
        entry = case.to_dict()
        entry["violations"] = violations
        out.append(entry)
    return tuple(out)


@dataclass(frozen=True)
class EvaluationReport:
    """The complete evaluation, for one run.

    Attributes:
        run_id: Identifies the run directory this belongs to.
        systems: ``(system, stream)`` to its report.
        comparisons: Metric to the corrected pairwise comparisons on that metric, per
            stream. Keyed ``"<stream>/<metric>"`` so the correction family is visible in
            the key.
        template_finding: Whether overlap metrics separate the Bronze template from the
            best model arm.
        agreement: Method A against Method B, when Method B has run.
        metric_validation: Correlation with human ratings, when Phase 12 has landed.
        metadata: Run context — git SHA, config hash, seeds, library versions.
    """

    run_id: str
    systems: Mapping[tuple[str, str], SystemReport]
    comparisons: Mapping[str, tuple[PairedComparison, ...]] = field(default_factory=dict)
    template_finding: TemplateBaselineFinding | None = None
    agreement: AgreementReport | None = None
    metric_validation: MetricValidationReport | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def streams(self) -> tuple[str, ...]:
        """Return the streams this report covers.

        Returns:
            The stream names, sorted, balanced first when present.
        """
        found = sorted({stream for _, stream in self.systems})
        return tuple(sorted(found, key=lambda s: (s != "balanced", s)))

    def systems_in(self, stream: str) -> list[SystemReport]:
        """Return the system reports for one stream.

        Args:
            stream: The stream name.

        Returns:
            The reports, ordered by the headline metric descending.
        """
        return sorted(
            (report for (_, s), report in self.systems.items() if s == stream),
            key=lambda r: -r.faithfulness.zero_hallucination_rate,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the whole report as a JSON-serialisable mapping.

        Returns:
            The report, with the headline named explicitly so a consumer does not have
            to know which key matters.
        """
        return {
            "run_id": self.run_id,
            "headline_metric": HEADLINE_METRIC,
            "streams": list(self.streams),
            "systems": {
                f"{system}/{stream}": report.to_dict()
                for (system, stream), report in sorted(self.systems.items())
            },
            "comparisons": {
                key: [c.to_dict() for c in group] for key, group in sorted(self.comparisons.items())
            },
            "template_baseline_finding": (
                self.template_finding.to_dict() if self.template_finding is not None else None
            ),
            "extractor_agreement": (
                self.agreement.to_dict() if self.agreement is not None else None
            ),
            "metric_validation": (
                self.metric_validation.to_dict() if self.metric_validation is not None else None
            ),
            "metadata": dict(self.metadata),
        }

    def to_json(self, path: str | Path) -> Path:
        """Write the machine-readable report.

        Args:
            path: Destination.

        Returns:
            The path written.
        """
        from g2t_aml.utils.io import write_json

        return write_json(path, self.to_dict())

    # ------------------------------------------------------------- markdown ---

    def to_markdown(self) -> str:
        """Render the human-readable summary.

        Returns:
            Markdown. Layer 2 first, the taxonomy table second, Layer 1 last under a
            heading that says what it is worth.
        """
        parts: list[str] = [f"# Evaluation report — `{self.run_id}`", ""]
        if self.metadata:
            parts.append(
                "Run: "
                + ", ".join(f"`{k}`={v}" for k, v in sorted(self.metadata.items()) if v is not None)
            )
            parts.append("")

        for stream in self.streams:
            parts.extend(self._stream_markdown(stream))

        if self.agreement is not None:
            parts.extend(self._agreement_markdown())
        if self.template_finding is not None:
            parts.extend(
                [
                    "## The template baseline",
                    "",
                    self.template_finding.headline,
                    "",
                ]
            )
        if self.metric_validation is not None:
            parts.extend(
                ["## Correlation with human judgement", "", self.metric_validation.markdown(), ""]
            )
        return "\n".join(parts)

    def _stream_markdown(self, stream: str) -> list[str]:
        """Render one stream's section.

        Args:
            stream: The stream name.

        Returns:
            Markdown lines.
        """
        reports = self.systems_in(stream)
        if not reports:
            return []
        lines = [f"## Stream: `{stream}`", ""]

        lines.extend(
            [
                "### Layer 2 — faithfulness",
                "",
                "**Zero-Hallucination Rate is the headline.** A narrative containing one "
                "fabricated fact is unusable regardless of the rest, so the per-narrative "
                "binary is what a compliance function acts on; the averaged rates below it "
                "are diagnostic.",
                "",
                "| System | Zero-Halluc. | Fact Precision | Halluc. Rate | Unverif. Rate | "
                "Coverage | Fact F1 | Numeric Acc. | Typology Acc. | Critical Err. | n |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for report in reports:
            f = report.faithfulness
            lines.append(
                f"| `{f.system}` | **{f.zero_hallucination_rate:.3f}** | "
                f"{f.fact_precision:.3f} | {f.hallucination_rate:.3f} | "
                f"{f.unverifiable_rate:.3f} | {f.fact_coverage:.3f} | {f.fact_f1:.3f} | "
                f"{_opt(f.numeric_accuracy)} | {_opt(f.typology_accuracy)} | "
                f"{f.critical_error_rate:.3f} | {f.n_cases} |"
            )
        lines.append("")

        key = f"{stream}/{HEADLINE_METRIC}"
        if key in self.comparisons:
            summaries = [
                report.seeds.get(
                    HEADLINE_METRIC,
                    seed_summary(
                        report.system,
                        HEADLINE_METRIC,
                        {-1: report.faithfulness.zero_hallucination_rate},
                    ),
                )
                for report in reports
            ]
            intervals = {
                report.system: report.intervals[HEADLINE_METRIC]
                for report in reports
                if HEADLINE_METRIC in report.intervals
            }
            lines.extend(
                [
                    f"#### {HEADLINE_METRIC} with intervals and corrected significance",
                    "",
                    publication_table(HEADLINE_METRIC, summaries, intervals, self.comparisons[key]),
                    "",
                ]
            )

        lines.extend(self._taxonomy_markdown(reports))

        lines.extend(
            [
                "### Layer 1 — surface overlap",
                "",
                "Reported for comparability with prior work. These metrics measure "
                "resemblance to a reference, not truth of the case, and the template "
                "baseline below is the evidence for how little they separate systems.",
                "",
                "| System | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | BERTScore F1 | "
                "Distinct-2 | Self-BLEU@5 | Len ratio | n pairs |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for report in reports:
            m = report.layer1
            if m is None:
                lines.append(f"| `{report.system}` | " + " | ".join(["—"] * 10) + " |")
                continue
            lines.append(
                f"| `{m.system}` | {_opt(m.bleu, 2)} | {_opt(m.rouge1)} | {_opt(m.rouge2)} | "
                f"{_opt(m.rouge_l)} | {_opt(m.meteor)} | {_opt(m.bertscore_f1)} | "
                f"{_opt(m.distinct_2)} | {_opt(m.self_bleu)} | {_opt(m.length_ratio)} | "
                f"{m.n_pairs} |"
            )
        signatures = {
            r.layer1.bleu_signature
            for r in reports
            if r.layer1 is not None and r.layer1.bleu_signature
        }
        if signatures:
            lines.extend(["", "BLEU signature: " + "; ".join(f"`{s}`" for s in sorted(signatures))])
        unavailable = {
            name: reason
            for r in reports
            if r.layer1 is not None
            for name, reason in r.layer1.unavailable.items()
        }
        if unavailable:
            lines.append("")
            lines.append(
                "Not computed: " + "; ".join(f"`{k}` — {v}" for k, v in sorted(unavailable.items()))
            )
        lines.append("")
        return lines

    def _taxonomy_markdown(self, reports: Sequence[SystemReport]) -> list[str]:
        """Render the H1—H9 table across systems.

        Args:
            reports: The system reports for one stream.

        Returns:
            Markdown lines.
        """
        header = " | ".join(h.ident for h in HallucinationClass)
        lines = [
            "### Hallucination taxonomy — per-narrative rate by class",
            "",
            "Each cell is the fraction of narratives carrying at least one finding of "
            "that class. **Critical Error Rate is H4 + H6 + H7 and is reported "
            "separately**, because those three are assertions the substrate cannot "
            "license at all rather than values read off it wrongly.",
            "",
            f"| System | {header} | Critical |",
            "|---|" + "---:|" * (len(HallucinationClass) + 1),
        ]
        for report in reports:
            rates = report.taxonomy.rate_by_class
            cells = " | ".join(f"{rates.get(h.ident, 0.0):.3f}" for h in HallucinationClass)
            lines.append(
                f"| `{report.system}` | {cells} | **{report.taxonomy.critical_error_rate:.3f}** |"
            )
        lines.append("")
        lines.append(
            "Classes: " + "; ".join(f"**{h.ident}** {h.title}" for h in HallucinationClass) + "."
        )
        lines.append("")
        return lines

    def _agreement_markdown(self) -> list[str]:
        """Render the extractor-agreement section.

        Returns:
            Markdown lines.
        """
        assert self.agreement is not None
        a = self.agreement
        note = (
            ""
            if a.meets_sample_target
            else f" **Below the {a.sample_size_target}-case protocol sample.**"
        )
        return [
            "## Extractor agreement — Method A against Method B",
            "",
            f"Measured on {a.n_cases} cases.{note} The two extractors share no machinery: "
            "Method A aligns against Bronze slots and applies vocabulary rules, Method B "
            "decomposes with a language model and judges entailment against the serialised "
            "record. They share only the three-valued verdict vocabulary.",
            "",
            "| Agreement | κ | band | raw agreement |",
            "|---|---:|---|---:|",
            f"| Verdict (matched claims, n={a.n_matched_claims}) | {_opt(a.verdict_kappa, 3)} | "
            f"{_band(a.verdict_kappa)} | {_opt(a.verdict_observed_agreement, 3)} |",
            f"| Claim boundary (token-level) | {_opt(a.boundary_kappa, 3)} | "
            f"{_band(a.boundary_kappa)} | — |",
            f"| Zero-hallucination decision (per narrative) | {_opt(a.decision_kappa, 3)} | "
            f"{_band(a.decision_kappa)} | {_opt(a.decision_observed_agreement, 3)} |",
            "",
            f"Claims found only by Method A: {a.n_only_a}. Only by Method B: {a.n_only_b}. "
            f"Method B claims whose evidence could not be located: {a.unlocated_b}.",
            "",
        ]

    # ---------------------------------------------------------------- LaTeX ---

    def to_latex(self, stream: str = "balanced") -> str:
        """Emit the paper's two main tables for one stream.

        Args:
            stream: Which stream to tabulate.

        Returns:
            LaTeX source for the faithfulness table and the taxonomy table. Numbers only:
            no package lines and no document preamble, because the paper owns those and a
            generated preamble is a generated conflict.
        """
        reports = self.systems_in(stream)
        if not reports:
            return f"% no systems scored on the {stream} stream\n"

        lines = [
            "% Generated by g2t_aml.eval.report -- do not edit by hand.",
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{Layer 2 faithfulness on the \\texttt{{{stream}}} test stream. "
            "Zero-Hallucination Rate is the fraction of narratives containing no "
            "contradicted claim.}",
            f"\\label{{tab:faithfulness-{stream}}}",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "System & Zero-Hall. & Fact Prec. & Hall. Rate & Coverage & Fact F1 & "
            "Crit. Err. \\\\",
            "\\midrule",
        ]
        for report in reports:
            f = report.faithfulness
            lines.append(
                f"\\texttt{{{_tex(f.system)}}} & \\textbf{{{f.zero_hallucination_rate:.3f}}} & "
                f"{f.fact_precision:.3f} & {f.hallucination_rate:.3f} & "
                f"{f.fact_coverage:.3f} & {f.fact_f1:.3f} & {f.critical_error_rate:.3f} \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

        lines.extend(
            [
                "\\begin{table}[t]",
                "\\centering",
                f"\\caption{{Hallucination classes, per-narrative rate, "
                f"\\texttt{{{stream}}} stream. H4, H6 and H7 aggregate into the "
                "Critical Error Rate.}",
                f"\\label{{tab:taxonomy-{stream}}}",
                "\\begin{tabular}{l" + "r" * (len(HallucinationClass) + 1) + "}",
                "\\toprule",
                "System & " + " & ".join(h.ident for h in HallucinationClass) + " & Crit. \\\\",
                "\\midrule",
            ]
        )
        for report in reports:
            rates = report.taxonomy.rate_by_class
            cells = " & ".join(f"{rates.get(h.ident, 0.0):.3f}" for h in HallucinationClass)
            lines.append(
                f"\\texttt{{{_tex(report.system)}}} & {cells} & "
                f"\\textbf{{{report.taxonomy.critical_error_rate:.3f}}} \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
        return "\n".join(lines)

    def write_all(self, directory: str | Path) -> dict[str, Path]:
        """Write the JSON, the markdown and the LaTeX into one directory.

        Args:
            directory: Where to write. Created if absent.

        Returns:
            Artifact name to the path written.
        """
        from g2t_aml.utils.io import ensure_dir

        root = ensure_dir(directory)
        paths = {"json": self.to_json(root / "evaluation.json")}
        (root / "evaluation.md").write_text(self.to_markdown(), encoding="utf-8")
        paths["markdown"] = root / "evaluation.md"
        for stream in self.streams:
            target = root / f"tables_{stream}.tex"
            target.write_text(self.to_latex(stream), encoding="utf-8")
            paths[f"latex_{stream}"] = target
        errors = [
            error.to_dict() for report in self.systems.values() for error in report.taxonomy.errors
        ]
        (root / "errors.jsonl").write_text(
            "\n".join(json.dumps(e, sort_keys=True) for e in errors) + ("\n" if errors else ""),
            encoding="utf-8",
        )
        paths["errors"] = root / "errors.jsonl"
        return paths


def _opt(value: float | None, places: int = 3) -> str:
    """Format an optional number, or an em dash.

    Args:
        value: The number, or None.
        places: Decimal places.

    Returns:
        The formatted number, or ``"—"``.
    """
    return "—" if value is None else f"{value:.{places}f}"


def _band(kappa: float | None) -> str:
    """Return a κ's verbal band, or an em dash.

    Args:
        kappa: The coefficient, or None.

    Returns:
        The band name, or ``"—"``.
    """
    if kappa is None:
        return "—"
    from g2t_aml.eval.claim_extraction.agreement import interpret_kappa

    return interpret_kappa(kappa)


def _tex(text: str) -> str:
    """Escape the characters LaTeX will not take literally.

    Args:
        text: A system name or label.

    Returns:
        The escaped text.
    """
    for char in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(char, "\\" + char)
    return text


def evaluate(  # noqa: PLR0912, PLR0915 -- one linear pipeline; splitting it separates
    # each stage from the reason it runs where it does.
    system_outputs: Sequence[SystemOutput],
    references: Mapping[str, str],
    facts: Mapping[str, CaseFacts],
    *,
    bronze: Mapping[str, BronzeNarrative] | None = None,
    run_id: str = "adhoc",
    config: FactConfig | None = None,
    vocabulary: ControlledVocabulary | None = None,
    bertscore_model: str | None = "microsoft/deberta-xlarge-mnli",
    learned: LearnedMetric | None = None,
    perplexity_model: PerplexityModel | None = None,
    agreement: AgreementReport | None = None,
    metric_validation: MetricValidationReport | None = None,
    metadata: Mapping[str, Any] | None = None,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 42,
) -> EvaluationReport:
    """Score every system on every stream and assemble the report.

    Args:
        system_outputs: Narratives from every system being compared.
        references: Gold narratives by case id. Layer 1 runs only where one exists;
            Layer 2 runs everywhere, which is why faithfulness is measurable on the
            fifteen thousand cases that will never have a human reference.
        facts: Fact records by case id.
        bronze: Bronze narratives by case id, whose slot annotation is Method A's
            alignment reference. Omitted weakens extraction to the rule table alone; the
            report does not fail, and the effect shows up as a higher unverifiable rate.
        run_id: Identifies the run directory.
        config: Thresholds and the tolerance policy.
        vocabulary: The controlled vocabulary, loaded once and shared.
        bertscore_model: BERTScore encoder, or None to skip it.
        learned: A BLEURT/COMET-class metric, or None.
        perplexity_model: A held-out LM, or None.
        agreement: A previously computed Method A/B agreement report, to embed.
        metric_validation: A previously computed human correlation, to embed.
        metadata: Run context to record.
        n_resamples: Bootstrap resamples.
        seed: Seeds the bootstrap and self-BLEU sampling.

    Returns:
        The complete report.

    Raises:
        EvaluationInputError: If a narrative has no fact record.
    """
    vocab = vocabulary if vocabulary is not None else load_vocabulary()
    cfg = config if config is not None else FactConfig()
    bronze_index = bronze or {}

    scored = list(pair_outputs_with_facts(system_outputs, facts, references=references))

    per_slice: dict[tuple[str, str], list[tuple[ScoredCase, CaseFaithfulness]]] = {}
    for case in scored:
        extractor = DeterministicClaimExtractor(
            bronze=bronze_index.get(case.case_id), vocabulary=vocab
        )
        claims = extractor.extract(case.output.narrative, case.facts)
        faithfulness = score_case(case, claims, config=cfg, vocabulary=vocab)
        per_slice.setdefault((case.output.system, case.output.stream), []).append(
            (case, faithfulness)
        )

    systems: dict[tuple[str, str], SystemReport] = {}
    for (system, stream), entries in per_slice.items():
        cases = [f for _, f in entries]
        aggregate = aggregate_faithfulness(cases, system=system)
        taxonomy = score_taxonomy(entries, system=system)

        intervals: dict[str, Interval] = {}
        for metric in PRIMARY_METRICS:
            values = [v for v in (_metric_value(c, metric) for c in cases) if v is not None]
            if values:
                intervals[metric] = bootstrap_ci(values, n_resamples=n_resamples, seed=seed)

        by_typology: dict[str, FaithfulnessMetrics] = {}
        for typology in sorted({c.typology for c in cases}):
            subset = [c for c in cases if c.typology == typology]
            by_typology[typology] = aggregate_faithfulness(subset, system=system)

        by_dataset: dict[str, FaithfulnessMetrics] = {}
        for dataset in sorted({c.dataset for c in cases}):
            subset = [c for c in cases if c.dataset == dataset]
            by_dataset[dataset] = aggregate_faithfulness(subset, system=system)

        seeds: dict[str, SeedSummary] = {}
        seed_values = sorted({c.seed for c in cases if c.seed is not None})
        if seed_values:
            for metric in PRIMARY_METRICS:
                per_seed: dict[int, float] = {}
                for s in seed_values:
                    subset = [c for c in cases if c.seed == s]
                    values = [
                        v for v in (_metric_value(c, metric) for c in subset) if v is not None
                    ]
                    if values:
                        per_seed[s] = sum(values) / len(values)
                if per_seed:
                    seeds[metric] = seed_summary(system, metric, per_seed)

        hypotheses: list[str] = []
        gold: list[str] = []
        for case, _ in entries:
            if case.reference:
                hypotheses.append(case.output.narrative)
                gold.append(case.reference)
        layer1 = compute_layer1(
            system,
            hypotheses,
            gold,
            all_narratives=[case.output.narrative for case, _ in entries],
            bertscore_model=bertscore_model,
            learned=learned,
            perplexity_model=perplexity_model,
            seed=seed,
        )

        systems[(system, stream)] = SystemReport(
            system=system,
            stream=stream,
            faithfulness=aggregate,
            taxonomy=taxonomy,
            layer1=layer1,
            intervals=intervals,
            by_typology=by_typology,
            by_dataset=by_dataset,
            seeds=seeds,
            worst=worst_cases(cases),
        )

    comparisons: dict[str, tuple[PairedComparison, ...]] = {}
    for stream in sorted({stream for _, stream in systems}):
        for metric in PRIMARY_METRICS:
            values_by_system: dict[str, dict[str, float]] = {}
            for (system, slice_stream), entries in per_slice.items():
                if slice_stream != stream:
                    continue
                row = {
                    f.case_id: value
                    for _, f in entries
                    if (value := _metric_value(f, metric)) is not None
                }
                if row:
                    values_by_system[system] = row
            found = compare_systems(
                metric,
                values_by_system,
                n_resamples=n_resamples,
                seed=seed,
                family=f"{metric} on the {stream} stream",
            )
            if found:
                comparisons[f"{stream}/{metric}"] = tuple(found)

    balanced = [
        report.layer1
        for (_, stream), report in systems.items()
        if stream == "balanced" and report.layer1 is not None
    ]
    finding = template_baseline_finding(balanced) if balanced else None

    return EvaluationReport(
        run_id=run_id,
        systems=systems,
        comparisons=comparisons,
        template_finding=finding,
        agreement=agreement,
        metric_validation=metric_validation,
        metadata=dict(metadata or {}),
    )
