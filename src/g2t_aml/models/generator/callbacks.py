"""Training-time diagnostics — and the S1-versus-A1 comparison, plotted from step 0.

**Loss is not the metric that matters here.** A generator conditioned on serialised facts
in its prompt will drive the loss down whether or not it reads the graph, so the loss curve
of a working fusion layer and the loss curve of a broken one are the same curve.
Faithfulness measured by the Phase 3 checker is the metric that matters, and the shuffled
control is what makes it interpretable.

**Two different controls, and conflating them would be a reviewer's first catch.**

``within-run shuffle``
    The model currently being trained, evaluated on the same held-out cases with its graph
    tokens deranged at *inference* time. Same weights, same prompt, wrong graph. Free to
    compute, available every diagnostic step from step 0, and the intended early-warning
    signal: if the model's faithfulness is unchanged when the graph is scrambled, the graph
    is contributing nothing and the run should be stopped and diagnosed rather than left to
    burn fourteen hours. It is **not** the paper's control, because a model trained on
    correct pairings can be robust to inference-time scrambling for reasons that have
    nothing to do with whether it needs the graph.

``the A1 arm``
    A separate model trained end to end on deranged pairings, under identical settings.
    This is the paper's control and the one Gate 8 is decided on.
    :func:`compare_arms` reads two runs' histories and does that comparison.

:class:`FaithfulnessCallback` computes the first. It records both curves on the same axes
from step 0 so the divergence — or the absence of one — is visible while there is still
time to act on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from g2t_aml.facts.checkers import CheckContext, check_claim, check_narrative_text, summarise
from g2t_aml.facts.schema import CaseFacts
from g2t_aml.models.fusion.control import ShuffledGraphFusion
from g2t_aml.models.fusion.diagnostics import soft_token_attention_mass
from g2t_aml.models.generator.dataset import GraphCollator
from g2t_aml.models.generator.inference import GenerationConfig, generate_batch
from g2t_aml.models.generator.model import Graph2TextGenerator
from g2t_aml.models.generator.prompts import BuiltPrompt
from g2t_aml.utils.io import read_jsonl

__all__ = [
    "ArmComparison",
    "AttentionMassCallback",
    "FaithfulnessCallback",
    "FaithfulnessScore",
    "ProbeCase",
    "compare_arms",
    "score_narrative",
]


@dataclass(frozen=True)
class ProbeCase:
    """One held-out case the diagnostics generate from, every time, unchanged.

    **The same ten cases at every checkpoint** is the point. A fresh random sample each
    time mixes drift in the model with drift in the sample, and the resulting curve cannot
    distinguish a model that got worse from a draw that was harder.

    Attributes:
        case_id: The case.
        prompt: Its inference prompt, built once.
        graph: Its PyG ``Data``, or None on a text-only arm.
        facts: The fact record every claim is checked against.
        extractor: The claim extractor bound to this case's Bronze alignment.
    """

    case_id: str
    prompt: BuiltPrompt
    graph: Any
    facts: CaseFacts
    extractor: Any


@dataclass(frozen=True)
class FaithfulnessScore:
    """The Phase 3 checker's verdict over one set of generations.

    Attributes:
        supported_rate: Share of claims the fact record supports. The headline.
        contradicted_rate: Share the record contradicts. The number that must not rise.
        unverifiable_rate: Share that could not be checked either way.
        critical_error_rate: Share falling in the H4/H6/H7 critical classes, reported
            separately from faithfulness as Phase 3's taxonomy requires.
        n_claims: Total claims extracted. Reported because a model that says less makes
            fewer claims and scores better on every rate — a rate without its denominator
            is not a measurement.
        n_cases: Cases scored.
    """

    supported_rate: float
    contradicted_rate: float
    unverifiable_rate: float
    critical_error_rate: float
    n_claims: int
    n_cases: int

    def to_dict(self, prefix: str = "") -> dict[str, float | int]:
        """Return the score as a loggable mapping.

        Args:
            prefix: Prepended to every key, so the treatment and control curves land in
                separate series.

        Returns:
            The fields, prefixed.
        """
        return {
            f"{prefix}supported_rate": self.supported_rate,
            f"{prefix}contradicted_rate": self.contradicted_rate,
            f"{prefix}unverifiable_rate": self.unverifiable_rate,
            f"{prefix}critical_error_rate": self.critical_error_rate,
            f"{prefix}n_claims": self.n_claims,
            f"{prefix}n_cases": self.n_cases,
        }


def score_narrative(
    text: str, case: ProbeCase, *, context: CheckContext | None = None
) -> list[Any]:
    """Extract a narrative's claims and check every one against the fact record.

    This is the Phase 3 measurement instrument run in reverse, exactly as Silver's verifier
    runs it — the same code, so a disagreement between corpus verification and evaluation
    is a bug rather than a parameter (invariant 1).

    Args:
        text: The generated narrative.
        case: The probe case, carrying its facts and its extractor.
        context: A prepared checking context, or None to build one from the case's facts.

    Returns:
        Every check result: one per extracted claim, plus the text-level forbidden-phrase
        and hedging findings.
    """
    ctx = context if context is not None else CheckContext(facts=case.facts)
    claims = case.extractor.extract(text, case.facts)
    results = [check_claim(claim, ctx) for claim in claims]
    results.extend(check_narrative_text(text, ctx))
    return results


def _aggregate(results_by_case: Sequence[Sequence[Any]]) -> FaithfulnessScore:
    """Summarise check results across cases.

    Args:
        results_by_case: One list of check results per case.

    Returns:
        The aggregate score.
    """
    flat = [r for case_results in results_by_case for r in case_results]
    summary = summarise(list(flat))
    return FaithfulnessScore(
        supported_rate=float(summary["supported_rate"]),
        contradicted_rate=float(summary["contradicted_rate"]),
        unverifiable_rate=float(summary["unverifiable_rate"]),
        critical_error_rate=float(summary["critical_error_rate"]),
        n_claims=int(summary["n_claims"]),
        n_cases=len(results_by_case),
    )


class FaithfulnessCallback:
    """Generates on fixed held-out cases and scores them, with and without the graph.

    Attributes:
        tracking_tolerance: How close the shuffled control's supported rate has to be to
            the model's before the two count as tracking each other.
        tracking_patience: How many consecutive diagnostic steps they must track before
            :attr:`tracking_alarm` is raised.
        tracking_alarm: Set once the two curves have tracked each other for
            ``tracking_patience`` consecutive evaluations. The trainer logs it loudly; a
            human decides whether to stop. It is deliberately not an automatic halt — an
            early run has both curves near zero for ordinary reasons, and a callback that
            killed the job for that would be worse than the problem.
    """

    def __init__(
        self,
        cases: Sequence[ProbeCase],
        *,
        collator: GraphCollator,
        tokenizer: Any,
        device: str = "cpu",
        max_new_tokens: int = 384,
        run_shuffled_control: bool = True,
        shuffle_seed: int = 42,
        tracking_tolerance: float = 0.02,
        tracking_patience: int = 3,
        samples_path: str | Path | None = None,
        log: Any = None,
    ) -> None:
        """Build the callback.

        Args:
            cases: The fixed probe cases. Ten is the brief's number.
            collator: The collator.
            tokenizer: The tokeniser.
            device: Device to generate on.
            max_new_tokens: Generation cap for the probe, below the full inference cap
                because this runs every ``eval_every`` steps.
            run_shuffled_control: Also generate with the graph tokens deranged. Off only
                for a text-only arm, where there is no graph to shuffle.
            shuffle_seed: Seed for the control's derangement.
            tracking_tolerance: See :attr:`tracking_tolerance`.
            tracking_patience: See :attr:`tracking_patience`.
            samples_path: JSONL to append the generated text to, so drift is readable and
                not only plottable. Ten narratives per checkpoint is a small file and the
                single most useful artifact when a curve does something surprising.
            log: Optional logger.
        """
        self.cases = list(cases)
        self.collator = collator
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.run_shuffled_control = run_shuffled_control
        self.shuffle_seed = shuffle_seed
        self.tracking_tolerance = tracking_tolerance
        self.tracking_patience = tracking_patience
        self.samples_path = Path(samples_path) if samples_path else None
        self.log = log
        self.tracking_alarm = False
        self._consecutive_tracking = 0
        self._contexts = {c.case_id: CheckContext(facts=c.facts) for c in self.cases}

    def _generate(self, generator: Graph2TextGenerator) -> list[str]:
        """Generate greedily on the probe cases.

        Args:
            generator: The model under training.

        Returns:
            One narrative per probe case, in case order.
        """
        batch = self.collator([(c.prompt, c.graph) for c in self.cases])
        results = generate_batch(
            generator,
            batch,
            GenerationConfig.deterministic(max_new_tokens=self.max_new_tokens),
            tokenizer=self.tokenizer,
            device=self.device,
        )
        return [r.texts[0] for r in results]

    def _generate_shuffled(self, generator: Graph2TextGenerator) -> list[str]:
        """Generate with the graph tokens deranged, on the same cases and same weights.

        The fusion layer is temporarily wrapped rather than rebuilt, so the control runs
        the identical projector — the only difference is which case's graph reaches it.

        Args:
            generator: The model under training.

        Returns:
            One narrative per probe case, in case order.
        """
        original = generator.fusion_projector
        generator.fusion_projector = ShuffledGraphFusion(
            original, mode="across_batch", seed=self.shuffle_seed
        )
        try:
            return self._generate(generator)
        finally:
            generator.fusion_projector = original

    def on_step(self, generator: Graph2TextGenerator, state: Any, batch: Any) -> dict[str, Any]:
        """Run the diagnostics for one training step.

        Args:
            generator: The model under training.
            state: The trainer's state, read for the step number.
            batch: The training batch that produced the step. Unused here; the probe
                deliberately runs on fixed held-out cases instead.

        Returns:
            A mapping of metrics to merge into the trainer's history row: the model's
            faithfulness under ``probe_``, the shuffled control's under ``shuffled_``,
            their difference, and the tracking alarm.
        """
        del batch
        was_training = generator.training
        row: dict[str, Any] = {}

        try:
            texts = self._generate(generator)
            scored = _aggregate(
                [
                    score_narrative(text, case, context=self._contexts[case.case_id])
                    for text, case in zip(texts, self.cases, strict=True)
                ]
            )
            row.update(scored.to_dict("probe_"))

            if self.run_shuffled_control and generator.fusion_projector is not None:
                shuffled_texts = self._generate_shuffled(generator)
                shuffled = _aggregate(
                    [
                        score_narrative(text, case, context=self._contexts[case.case_id])
                        for text, case in zip(shuffled_texts, self.cases, strict=True)
                    ]
                )
                row.update(shuffled.to_dict("shuffled_"))
                gap = scored.supported_rate - shuffled.supported_rate
                row["faithfulness_gap"] = gap

                if abs(gap) <= self.tracking_tolerance:
                    self._consecutive_tracking += 1
                else:
                    self._consecutive_tracking = 0
                if self._consecutive_tracking >= self.tracking_patience:
                    self.tracking_alarm = True
                    row["tracking_alarm"] = True
                    if self.log is not None:
                        self.log.warning(
                            "the shuffled control has tracked the model within %.3f for %d "
                            "consecutive evaluations (gap %.4f). The graph may be "
                            "contributing nothing. Diagnose before continuing.",
                            self.tracking_tolerance,
                            self._consecutive_tracking,
                            gap,
                        )
                self._write_samples(state, texts, shuffled_texts)
            else:
                self._write_samples(state, texts, None)
        finally:
            generator.train(was_training)

        return row

    def _write_samples(
        self, state: Any, texts: Sequence[str], shuffled: Sequence[str] | None
    ) -> None:
        """Append this checkpoint's generations to the samples file.

        Args:
            state: The trainer state, read for the step number.
            texts: The model's narratives.
            shuffled: The control's narratives, or None.
        """
        if self.samples_path is None:
            return
        from g2t_aml.utils.io import write_jsonl

        existing = list(read_jsonl(self.samples_path)) if self.samples_path.is_file() else []
        for index, case in enumerate(self.cases):
            existing.append(
                {
                    "step": int(getattr(state, "step", 0)),
                    "case_id": case.case_id,
                    "text": texts[index],
                    "shuffled_text": shuffled[index] if shuffled is not None else None,
                }
            )
        write_jsonl(self.samples_path, existing)


class AttentionMassCallback:
    """Measures how much attention the completion pays to the soft tokens.

    Reported against its uniform baseline, never alone: with 16 soft tokens in a
    130-position sequence, uniform attention *is* 12%, and a bare "12% of attention went to
    the graph" says only that the tokens were present.
    """

    def __init__(self, *, layers: tuple[int, ...] | None = None, log: Any = None) -> None:
        """Build the callback.

        Args:
            layers: Which layers to average over, or None for all of them.
            log: Optional logger.
        """
        self.layers = layers
        self.log = log

    def on_step(self, generator: Graph2TextGenerator, state: Any, batch: Any) -> dict[str, Any]:
        """Measure attention mass on one batch.

        Args:
            generator: The model under training.
            state: The trainer state. Unused.
            batch: The training batch, re-run with ``output_attentions=True``.

        Returns:
            The attention-mass measurement, or an empty mapping on a text-only arm or when
            the backbone does not return attentions.
        """
        del state
        if generator.fusion_projector is None:
            return {}

        was_training = generator.training
        generator.eval()
        try:
            with torch.no_grad():
                out = generator(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    labels=None,
                    soft_mask=batch.soft_mask,
                    graph_batch=batch.graph_batch,
                    output_attentions=True,
                )
            if not out.attentions:
                return {}
            soft_positions = batch.soft_mask[0].nonzero().flatten()
            if soft_positions.numel() == 0:
                return {}
            mass = soft_token_attention_mass(
                out.attentions,
                soft_start=int(soft_positions[0]),
                n_soft=int(soft_positions.numel()),
                query_start=min(batch.completion_starts) if batch.completion_starts else None,
                layers=self.layers,
            )
        finally:
            generator.train(was_training)

        return {f"attention_{k}": v for k, v in mass.to_dict().items()}


@dataclass
class ArmComparison:
    """The S1-versus-A1 comparison, stated as numbers rather than as an impression.

    Attributes:
        treatment_arm: Name of the treatment run.
        control_arm: Name of the control run.
        treatment_supported: Final supported rate on the probe cases.
        control_supported: The control's.
        difference: ``treatment - control``. **This is the project's decision variable.**
        n_steps_compared: How many aligned diagnostic steps were available.
        tracked_throughout: Whether the two curves stayed within tolerance at every
            compared step. True is the halt condition: it means the graph never mattered
            at any point in training, not that it stopped mattering.
    """

    treatment_arm: str
    control_arm: str
    treatment_supported: float
    control_supported: float
    difference: float
    n_steps_compared: int
    tracked_throughout: bool

    def verdict(self, *, tolerance: float = 0.02) -> str:
        """Return a plain-language reading of the comparison.

        Args:
            tolerance: Difference below which the two arms count as indistinguishable.

        Returns:
            A sentence stating what the numbers mean for the project's contribution.
            Blunt on purpose: this is the sentence that goes into ``PHASE_LOG.md``, and
            the failure mode it guards against is a comparison that gets softened between
            the trace and the write-up.
        """
        if self.difference > tolerance:
            return (
                f"{self.treatment_arm} beats {self.control_arm} by "
                f"{self.difference:+.4f} supported rate: the graph fusion contributes."
            )
        if self.difference < -tolerance:
            return (
                f"{self.treatment_arm} is *worse* than {self.control_arm} by "
                f"{self.difference:+.4f}: the graph pathway is actively harming the model."
            )
        return (
            f"{self.treatment_arm} and {self.control_arm} are indistinguishable "
            f"({self.difference:+.4f}, tolerance {tolerance}). There is no architecture "
            "contribution. The paper pivots to the dataset and evaluation framework."
        )


def compare_arms(
    treatment_history: str | Path,
    control_history: str | Path,
    *,
    tolerance: float = 0.02,
) -> ArmComparison:
    """Compare a treatment run against its control from their diagnostic histories.

    This is the Gate 8 comparison — two separately trained arms, not the within-run
    shuffle.

    Args:
        treatment_history: The treatment run's ``history_*.jsonl``.
        control_history: The control run's.
        tolerance: Difference below which the arms count as tracking each other.

    Returns:
        The comparison.

    Raises:
        ValueError: If either history holds no probe measurements, which means the run
            never evaluated faithfulness and there is nothing to compare.
    """
    treatment = [r for r in read_jsonl(treatment_history) if "probe_supported_rate" in r]
    control = [r for r in read_jsonl(control_history) if "probe_supported_rate" in r]
    if not treatment or not control:
        raise ValueError(
            "both histories must carry probe_supported_rate rows; a run with no "
            "faithfulness measurements cannot be compared"
        )

    by_step = {int(r["step"]): float(r["probe_supported_rate"]) for r in control}
    aligned = [
        (float(r["probe_supported_rate"]), by_step[int(r["step"])])
        for r in treatment
        if int(r["step"]) in by_step
    ]
    tracked = bool(aligned) and all(abs(t - c) <= tolerance for t, c in aligned)

    final_treatment = float(treatment[-1]["probe_supported_rate"])
    final_control = float(control[-1]["probe_supported_rate"])
    return ArmComparison(
        treatment_arm=str(treatment[-1].get("arm", "treatment")),
        control_arm=str(control[-1].get("arm", "control")),
        treatment_supported=final_treatment,
        control_supported=final_control,
        difference=final_treatment - final_control,
        n_steps_compared=len(aligned),
        tracked_throughout=tracked,
    )
