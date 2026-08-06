"""Phase 11: the experiment matrix, declared once and read by everything downstream.

**A system is reproducible from its config alone.** That is the property this module
exists to hold. Every row of the paper's results table is a :class:`SystemSpec` here, and
a spec fully determines the encoder arm, the fusion variant, the text mode, the base
model, the training regime, whether the inference guard runs, and which seeds. The runner
resolves nothing that is not written down here, and the aggregator reads the same table,
so a number in the results file can be traced back to one row without consulting anybody's
memory.

Three things in this module are decisions rather than descriptions, each documented at its
definition:

- **The seed asymmetry** (:data:`SEEDS_CENTRAL`, :data:`SEEDS_SINGLE`). Three seeds on the
  four systems carrying the central claim, one on everything else, because the full matrix
  at three seeds is 3-5 GPU-weeks. It is stated in the paper, not hidden.
- **F3 and F4 are named here** (:class:`FusionVariant`), because Phase 8 implemented a gate
  flag and three projector kinds rather than four numbered variants, and the mapping from
  the brief's names onto the built machinery has to live somewhere a reader can check.
- **A3's F1 point is B8** (:data:`_A3_REUSES_B8`). Running a third arm identical to an
  existing one to give a column a different heading would spend GPU-days to produce a
  duplicate row.

Nothing here imports torch, transformers or any GPU dependency: the registry is read by
the aggregator and the figure code on machines that will never train anything.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

__all__ = [
    "CENTRAL_CLAIM_SYSTEMS",
    "SEEDS_CENTRAL",
    "SEEDS_SINGLE",
    "Executor",
    "FusionVariant",
    "Resource",
    "SystemSpec",
    "TextMode",
    "UnknownSystemError",
    "all_systems",
    "central_claim_family",
    "comparison_family",
    "expand_runs",
    "get_system",
    "matrix_summary",
    "resolution_order",
    "system_ids",
    "validate_registry",
]


class UnknownSystemError(KeyError):
    """Raised when a system id is not in the registry."""


class TextMode(StrEnum):
    """What textual case information reaches the language model.

    Attributes:
        FULL: Instruction, serialised fact record, and graph soft tokens.
        NONE: Instruction and graph soft tokens only. The S2 headline arm.
        SERIALISED: Instruction and serialised fact record, no graph. The B7 baseline.
        NA: The system has no language model, so the axis does not apply.
    """

    FULL = "full"
    NONE = "none"
    SERIALISED = "serialised"
    NA = "n/a"


class FusionVariant(StrEnum):
    """How graph structure enters the embedding space.

    Phase 8 implemented **one gate flag and three projector kinds**, not four numbered
    variants. The brief's F1-F4 names map onto that machinery as follows, and the mapping
    is declared here rather than inferred at a call site because a fusion ablation whose
    arms nobody can identify is not an ablation:

    ===== ====== ========== =========================================================
    Name  Gated  Projector  What it isolates
    ===== ====== ========== =========================================================
    F0    --     --         No fusion at all. B7's text-only control.
    F1    no     mlp        Project and splice: the G-Retriever-style prefix.
    F2    yes    mlp        F1 plus the learned per-token gate. **The contribution.**
    F3    yes    linear     Whether the projector needs a hidden layer.
    F4    yes    perceiver  Whether resampling beats one-to-one projection.
    ===== ====== ========== =========================================================

    F1 and F2 differ in exactly one flag, which is what makes the gate's contribution
    measurable rather than confounded with two separately-tuned models. F3 and F4 hold the
    gate fixed and move the projector, so the two axes are never varied together.

    Attributes:
        F0: No fusion.
        F1: Ungated MLP prefix.
        F2: Gated MLP prefix.
        F3: Gated linear prefix.
        F4: Gated perceiver-resampler prefix.
        NA: No language model, so no fusion.
    """

    F0 = "F0"
    F1 = "F1"
    F2 = "F2"
    F3 = "F3"
    F4 = "F4"
    NA = "n/a"

    @property
    def gated(self) -> bool | None:
        """Report the gate flag this variant sets.

        Returns:
            True for the gated variants, False for F1, None where fusion does not run.
        """
        if self in (FusionVariant.F0, FusionVariant.NA):
            return None
        return self is not FusionVariant.F1

    @property
    def projector(self) -> str | None:
        """Report the projector kind this variant selects.

        Returns:
            One of ``linear``, ``mlp``, ``perceiver``, or None where fusion does not run.
        """
        return {
            FusionVariant.F1: "mlp",
            FusionVariant.F2: "mlp",
            FusionVariant.F3: "linear",
            FusionVariant.F4: "perceiver",
        }.get(self)


class Executor(StrEnum):
    """Which code path produces this system's narratives.

    The runner dispatches on this and on nothing else, so adding a system means adding a
    row here and a config, never an ``if system_id ==`` anywhere downstream.

    Attributes:
        TEMPLATE: Deterministic Bronze rendering from the fact record. No model.
        CLASSIFIER_TEMPLATE: Phase 7 encoder's prediction fed into the template.
        API_ZERO_SHOT: One frontier-API call on the serialised record.
        API_FEW_SHOT: One frontier-API call with k in-context exemplars.
        API_AGENTIC: Generate, self-verify, repair. The published-competitor comparator.
        LOCAL_ZERO_SHOT: Untuned local base model, one call.
        TRAINED_GENERATOR: A Phase 9 QLoRA arm read from a checkpoint.
    """

    TEMPLATE = "template"
    CLASSIFIER_TEMPLATE = "classifier_template"
    API_ZERO_SHOT = "api_zero_shot"
    API_FEW_SHOT = "api_few_shot"
    API_AGENTIC = "api_agentic"
    LOCAL_ZERO_SHOT = "local_zero_shot"
    TRAINED_GENERATOR = "trained_generator"


class Resource(StrEnum):
    """What a run needs, which is how the scheduler decides what may run beside what.

    Attributes:
        CPU: No accelerator, no network. Parallelisable without limit.
        GPU: Needs the accelerator. Serialised against every other GPU job.
        API: Needs network and spend authorisation. Parallelisable, rate-limited.
        GPU_INFERENCE: Needs the accelerator but not for training; still serialised.
    """

    CPU = "cpu"
    GPU = "gpu"
    GPU_INFERENCE = "gpu_inference"
    API = "api"


# --------------------------------------------------------------------------- seeds ---

# THE SEED ASYMMETRY. Three seeds on the four systems the central claim rests on, one on
# everything else.
#
# The full sixteen-system matrix at three seeds is roughly 3-5 GPU-weeks, which does not
# fit any schedule this project has. The alternative to an asymmetry is not symmetry, it is
# a smaller matrix -- dropping ablations to afford variance estimates on arms nobody
# disputes. Between "every system at one seed" and "the systems carrying the claim at
# three, the rest at one", the second puts the variance estimate where a reviewer will ask
# for it: on S1 vs A1, and on S1 vs B7.
#
# This is stated in the paper rather than inferred from a table. `SeedSummary.std` is None
# at one seed by construction, so a single-seed row prints an em dash and cannot be read as
# a zero-variance result (eval/statistics.py). See DECISIONS.md D-081.
SEEDS_CENTRAL: tuple[int, ...] = (42, 1337, 2024)
SEEDS_SINGLE: tuple[int, ...] = (42,)

#: The systems carrying the paper's central claim, and the only ones run at three seeds.
#: S1 vs A1 is Gate 8; S1 vs B7 is whether the graph contributes over serialised facts;
#: S2 is the headline. Everything else is an ablation around those three comparisons.
CENTRAL_CLAIM_SYSTEMS: frozenset[str] = frozenset({"S1", "S2", "A1", "B7"})

#: If compute frees up, these get seeds next, in this order. A2 answers "is topology
#: needed" and B8 is the published-baseline arm; both currently carry a claim on one seed.
SEED_EXTENSION_ORDER: tuple[str, ...] = ("A2", "B8")

# A3's F1 point IS B8: gated=False, projector=mlp, text_mode=full, same base model, same
# training regime. Running a third arm with those settings under a different name would
# spend GPU-days to produce a duplicate row. The fusion ablation therefore reports
# {B8 (F1), S1 (F2), A3_F3, A3_F4} and says so in the table caption.
_A3_REUSES_B8 = True


@dataclass(frozen=True)
class SystemSpec:
    """One row of the experiment matrix, fully specified.

    Attributes:
        system_id: The identifier used everywhere -- run directories, metrics files,
            table rows, figure legends.
        role: What this system is in the paper's argument. Printed in the results table
            so a reader never has to guess why an arm is present.
        description: One line, for logs and captions.
        executor: Which code path generates the narratives.
        resource: What the run needs; the scheduler reads this and nothing else.
        encoder_arm: The Phase 7 encoder checkpoint arm, or None where no encoder runs.
        fusion: The fusion variant.
        text_mode: What textual case information reaches the model.
        base_model: Model identifier, or None for the template systems.
        base_model_version: The exact pinned version. **Recorded, not defaulted**: a
            comparison table whose baseline says "GPT-4" and not which one is a table a
            reviewer cannot reproduce, and stale baselines are a documented desk-reject
            trigger at this venue.
        base_model_release_date: ISO date, so the "is this baseline current" question is
            answerable from the table itself.
        experiment_config: The Hydra experiment config that reproduces this system.
        training_config: The Hydra training config, or None where nothing is trained.
        trained: Whether this system requires a training run.
        guard: Whether the Phase 9 inference guard is enabled.
        seeds: Every seed this system runs at. Length 3 for the central-claim systems,
            1 otherwise.
        depends_on: System ids or artifact keys that must complete first.
        notes: Anything a reader of the results table needs in order to read the row
            correctly.
    """

    system_id: str
    role: str
    description: str
    executor: Executor
    resource: Resource
    encoder_arm: str | None = None
    fusion: FusionVariant = FusionVariant.NA
    text_mode: TextMode = TextMode.NA
    base_model: str | None = None
    base_model_version: str | None = None
    base_model_release_date: str | None = None
    experiment_config: str | None = None
    training_config: str | None = None
    trained: bool = False
    guard: bool = True
    seeds: tuple[int, ...] = SEEDS_SINGLE
    depends_on: tuple[str, ...] = ()
    notes: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def uses_encoder(self) -> bool:
        """Report whether this system runs a graph encoder.

        Returns:
            True when an encoder arm is configured.
        """
        return self.encoder_arm is not None

    @property
    def n_runs(self) -> int:
        """Report how many runs this system expands into.

        Returns:
            One per seed.
        """
        return len(self.seeds)

    @property
    def is_central(self) -> bool:
        """Report whether this system carries the central claim.

        Returns:
            True for the multi-seed systems.
        """
        return self.system_id in CENTRAL_CLAIM_SYSTEMS

    def to_dict(self) -> dict[str, Any]:
        """Return the spec as a JSON-serialisable mapping.

        Returns:
            Every field, with the enums as their string values, so a resolved config
            written beside a run is readable without importing this module.
        """
        return {
            "system_id": self.system_id,
            "role": self.role,
            "description": self.description,
            "executor": str(self.executor),
            "resource": str(self.resource),
            "encoder_arm": self.encoder_arm,
            "fusion": str(self.fusion),
            "fusion_gated": self.fusion.gated,
            "fusion_projector": self.fusion.projector,
            "text_mode": str(self.text_mode),
            "base_model": self.base_model,
            "base_model_version": self.base_model_version,
            "base_model_release_date": self.base_model_release_date,
            "experiment_config": self.experiment_config,
            "training_config": self.training_config,
            "trained": self.trained,
            "guard": self.guard,
            "seeds": list(self.seeds),
            "n_runs": self.n_runs,
            "depends_on": list(self.depends_on),
            "is_central_claim": self.is_central,
            "notes": self.notes,
            "extra": dict(self.extra),
        }


# ------------------------------------------------------------------ the base models ---

# BASELINE CURRENCY. Every model identifier and its release date, in one block, because
# the acceptance criterion is "baselines use current 2025-2026 models with versions
# recorded" and a version recorded in five places is a version that will disagree with
# itself. `validate_registry` refuses a baseline whose release date is before
# `_BASELINE_FLOOR_DATE`.
_BASELINE_FLOOR_DATE = "2024-01-01"

FRONTIER_MODEL = "claude-opus-5"
FRONTIER_MODEL_RELEASED = "2026-02-05"
LOCAL_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LOCAL_BASE_MODEL_RELEASED = "2024-07-23"
SECOND_BASE_MODEL = "Qwen/Qwen3-8B"
SECOND_BASE_MODEL_RELEASED = "2025-04-29"

_SYSTEMS: tuple[SystemSpec, ...] = (
    # ---------------------------------------------------------------- baselines ---
    SystemSpec(
        system_id="B1",
        role="Faithfulness ceiling",
        description="Deterministic template rendered from the fact record (Bronze).",
        executor=Executor.TEMPLATE,
        resource=Resource.CPU,
        fusion=FusionVariant.NA,
        text_mode=TextMode.NA,
        experiment_config="matrix_b1",
        guard=False,
        notes=(
            "Faithful by construction: every formatter ships with its inverse, so this "
            "scores 1.0000 Zero-Hallucination and that is a regression test on the "
            "harness rather than a result about Bronze. Its H9 rate of 0.9179 is the "
            "dimension on which a trained system can beat it."
        ),
    ),
    SystemSpec(
        system_id="B2",
        role="Plan's baseline (a): classifier + template",
        description="Phase 7 GATv2 classifier's prediction rendered through the template.",
        executor=Executor.CLASSIFIER_TEMPLATE,
        resource=Resource.CPU,
        encoder_arm="gatv2",
        fusion=FusionVariant.NA,
        text_mode=TextMode.NA,
        experiment_config="matrix_b2",
        trained=False,
        guard=False,
        depends_on=("encoder:gatv2",),
        notes=(
            "The encoder is already trained (Phase 7); this system runs its scoring pass "
            "and renders, so it is CPU-only at matrix time."
        ),
    ),
    SystemSpec(
        system_id="B3",
        role="Plan's baseline (b): frontier zero-shot",
        description="Frontier LLM, zero-shot, on the serialised fact record.",
        executor=Executor.API_ZERO_SHOT,
        resource=Resource.API,
        fusion=FusionVariant.F0,
        text_mode=TextMode.SERIALISED,
        base_model=FRONTIER_MODEL,
        base_model_version=FRONTIER_MODEL,
        base_model_release_date=FRONTIER_MODEL_RELEASED,
        experiment_config="matrix_b3",
        guard=False,
        notes="No exemplars, no verification loop. The frontier floor.",
    ),
    SystemSpec(
        system_id="B4",
        role="Stronger baseline (b): frontier few-shot",
        description="Frontier LLM with k=5 in-context exemplars drawn from the train split.",
        executor=Executor.API_FEW_SHOT,
        resource=Resource.API,
        fusion=FusionVariant.F0,
        text_mode=TextMode.SERIALISED,
        base_model=FRONTIER_MODEL,
        base_model_version=FRONTIER_MODEL,
        base_model_release_date=FRONTIER_MODEL_RELEASED,
        experiment_config="matrix_b4",
        guard=False,
        extra={"k_shot": 5},
        notes=(
            "Exemplars are selected by typology match then by nearest case size, from the "
            "TRAIN split only, and the selection is deterministic given the case id."
        ),
    ),
    SystemSpec(
        system_id="B5",
        role="Co-Investigator-style agentic comparator",
        description="Frontier LLM with a generate -> self-verify -> repair loop.",
        executor=Executor.API_AGENTIC,
        resource=Resource.API,
        fusion=FusionVariant.F0,
        text_mode=TextMode.SERIALISED,
        base_model=FRONTIER_MODEL,
        base_model_version=FRONTIER_MODEL,
        base_model_release_date=FRONTIER_MODEL_RELEASED,
        experiment_config="matrix_b5",
        guard=False,
        extra={"max_repair_rounds": 3, "self_verify": True},
        notes=(
            "THE CLOSEST EXISTING COMPETITOR, and it gets the same effort the primary arm "
            "gets: real self-verification against the serialised record, up to three "
            "repair rounds, and the same few-shot exemplars B4 uses. It is allowed MORE "
            "inference compute than any of our arms and that asymmetry is reported rather "
            "than corrected for. Deliberately weakening it would be misconduct."
        ),
    ),
    SystemSpec(
        system_id="B6",
        role="Untuned floor",
        description="Llama-3.1-8B-Instruct, zero-shot, on the serialised fact record.",
        executor=Executor.LOCAL_ZERO_SHOT,
        resource=Resource.GPU_INFERENCE,
        fusion=FusionVariant.F0,
        text_mode=TextMode.SERIALISED,
        base_model=LOCAL_BASE_MODEL,
        base_model_version=LOCAL_BASE_MODEL,
        base_model_release_date=LOCAL_BASE_MODEL_RELEASED,
        experiment_config="matrix_b6",
        guard=False,
        notes=(
            "B7 minus the finetuning: the same base model and the same prompt, so the "
            "B6 -> B7 delta is what QLoRA bought and nothing else."
        ),
    ),
    SystemSpec(
        system_id="B7",
        role="THE THREATENING BASELINE",
        description="Llama-3.1-8B QLoRA on serialised facts only. No encoder, no fusion.",
        executor=Executor.TRAINED_GENERATOR,
        resource=Resource.GPU,
        fusion=FusionVariant.F0,
        text_mode=TextMode.SERIALISED,
        base_model=LOCAL_BASE_MODEL,
        base_model_version=LOCAL_BASE_MODEL,
        base_model_release_date=LOCAL_BASE_MODEL_RELEASED,
        experiment_config="generator_b7",
        training_config="generator",
        trained=True,
        seeds=SEEDS_CENTRAL,
        notes=(
            "Read against Phase 7's finding that the MLP control reaches 0.80 AUC-PR with "
            "no message passing. If B7 matches S1, the graph encoder contributes nothing "
            "the serialised facts do not already carry, and the paper's contribution is "
            "the corpus and the evaluation framework."
        ),
    ),
    SystemSpec(
        system_id="B8",
        role="Standard graph-LLM recipe",
        description="G-Retriever-style: F1 ungated prefix fusion + full text.",
        executor=Executor.TRAINED_GENERATOR,
        resource=Resource.GPU,
        encoder_arm="gatv2",
        fusion=FusionVariant.F1,
        text_mode=TextMode.FULL,
        base_model=LOCAL_BASE_MODEL,
        base_model_version=LOCAL_BASE_MODEL,
        base_model_release_date=LOCAL_BASE_MODEL_RELEASED,
        experiment_config="generator_b8",
        training_config="generator",
        trained=True,
        depends_on=("encoder:gatv2",),
        notes=(
            "Also serves as the F1 point of the A3 fusion ablation: it is S1 with the "
            "gate off and nothing else changed, which is exactly what A3's F1 arm would "
            "have been. Next in line for extra seeds after the central four."
        ),
    ),
    # ------------------------------------------------------------------- ours ---
    SystemSpec(
        system_id="S1",
        role="Full system",
        description="GATv2 encoder + F2 gated fusion + serialised facts.",
        executor=Executor.TRAINED_GENERATOR,
        resource=Resource.GPU,
        encoder_arm="gatv2",
        fusion=FusionVariant.F2,
        text_mode=TextMode.FULL,
        base_model=LOCAL_BASE_MODEL,
        base_model_version=LOCAL_BASE_MODEL,
        base_model_release_date=LOCAL_BASE_MODEL_RELEASED,
        experiment_config="generator_s1",
        training_config="generator",
        trained=True,
        seeds=SEEDS_CENTRAL,
        depends_on=("encoder:gatv2",),
        notes="The primary arm. Every other row is read against this one.",
    ),
    SystemSpec(
        system_id="S2",
        role="HEADLINE",
        description="GATv2 encoder + F2 gated fusion, graph only: no serialised facts.",
        executor=Executor.TRAINED_GENERATOR,
        resource=Resource.GPU,
        encoder_arm="gatv2",
        fusion=FusionVariant.F2,
        text_mode=TextMode.NONE,
        base_model=LOCAL_BASE_MODEL,
        base_model_version=LOCAL_BASE_MODEL,
        base_model_release_date=LOCAL_BASE_MODEL_RELEASED,
        experiment_config="generator_s2",
        training_config="generator",
        trained=True,
        seeds=SEEDS_CENTRAL,
        depends_on=("encoder:gatv2",),
        notes=(
            "Everything the narrative states must arrive through the soft tokens. "
            "Phase 7's linear probe reached 0.33 structural macro-F1 on exactly those "
            "tokens, so fan_out, gather_scatter and cycle are recoverable here and stack "
            "and random are not -- read the per-typology breakdown before the mean."
        ),
    ),
    # -------------------------------------------------------------- ablations ---
    SystemSpec(
        system_id="A1",
        role="SANITY CONTROL",
        description="S1 with graph tokens deranged across the batch.",
        executor=Executor.TRAINED_GENERATOR,
        resource=Resource.GPU,
        encoder_arm="gatv2",
        fusion=FusionVariant.F2,
        text_mode=TextMode.FULL,
        base_model=LOCAL_BASE_MODEL,
        base_model_version=LOCAL_BASE_MODEL,
        base_model_release_date=LOCAL_BASE_MODEL_RELEASED,
        experiment_config="generator_a1",
        training_config="generator",
        trained=True,
        seeds=SEEDS_CENTRAL,
        depends_on=("encoder:gatv2",),
        notes=(
            "Gate 8. Same encoder, same projector, same parameter and token count, same "
            "optimiser, same seed, same curriculum, same number of steps -- every "
            "narrative paired with a DIFFERENT case's graph. The derangement's "
            "fixed-point count must be zero (D-071). If S1 does not significantly beat "
            "this, the fusion layer is decoration."
        ),
    ),
    SystemSpec(
        system_id="A2",
        role="Is topology needed?",
        description="S1 with the Phase 7 MLP control in place of the GAT encoder.",
        executor=Executor.TRAINED_GENERATOR,
        resource=Resource.GPU,
        encoder_arm="mlp",
        fusion=FusionVariant.F2,
        text_mode=TextMode.FULL,
        base_model=LOCAL_BASE_MODEL,
        base_model_version=LOCAL_BASE_MODEL,
        base_model_release_date=LOCAL_BASE_MODEL_RELEASED,
        experiment_config="matrix_a2",
        training_config="generator",
        trained=True,
        depends_on=("encoder:mlp",),
        notes=(
            "The MLP control is a DeepSets model over case-local node features with no "
            "message passing, and it reaches 0.80 AUC-PR against GATv2's 0.872. Every "
            "claim about what graph topology contributes is a claim about that 0.07. "
            "Next in line for extra seeds after the central four."
        ),
    ),
    SystemSpec(
        system_id="A3_F3",
        role="Fusion ablation: projector capacity",
        description="S1 with a linear projector instead of the MLP.",
        executor=Executor.TRAINED_GENERATOR,
        resource=Resource.GPU,
        encoder_arm="gatv2",
        fusion=FusionVariant.F3,
        text_mode=TextMode.FULL,
        base_model=LOCAL_BASE_MODEL,
        base_model_version=LOCAL_BASE_MODEL,
        base_model_release_date=LOCAL_BASE_MODEL_RELEASED,
        experiment_config="matrix_a3_f3",
        training_config="generator",
        trained=True,
        depends_on=("encoder:gatv2",),
        notes="Gate held at F2's setting; only the projector moves.",
    ),
    SystemSpec(
        system_id="A3_F4",
        role="Fusion ablation: resampling",
        description="S1 with a perceiver-resampler projector instead of the MLP.",
        executor=Executor.TRAINED_GENERATOR,
        resource=Resource.GPU,
        encoder_arm="gatv2",
        fusion=FusionVariant.F4,
        text_mode=TextMode.FULL,
        base_model=LOCAL_BASE_MODEL,
        base_model_version=LOCAL_BASE_MODEL,
        base_model_release_date=LOCAL_BASE_MODEL_RELEASED,
        experiment_config="matrix_a3_f4",
        training_config="generator",
        trained=True,
        depends_on=("encoder:gatv2",),
        notes=(
            "The only variant permitted to change the soft-token count; the one-to-one "
            "projectors refuse a mismatch at construction rather than reshaping."
        ),
    ),
    SystemSpec(
        system_id="A4",
        role="Joint-training ablation",
        description="S1 with the Phase 7 encoder frozen throughout.",
        executor=Executor.TRAINED_GENERATOR,
        resource=Resource.GPU,
        encoder_arm="gatv2",
        fusion=FusionVariant.F2,
        text_mode=TextMode.FULL,
        base_model=LOCAL_BASE_MODEL,
        base_model_version=LOCAL_BASE_MODEL,
        base_model_release_date=LOCAL_BASE_MODEL_RELEASED,
        experiment_config="matrix_a4",
        training_config="generator",
        trained=True,
        depends_on=("encoder:gatv2",),
        notes=(
            "NOTE THE DIRECTION. `generator.freeze_encoder` defaults to TRUE, because "
            "unfreezing lets the narrative loss undo the val-AUC-PR selection Phase 7 "
            "made. So S1 is the frozen configuration and A4 is the arm that UNFREEZES, "
            "at encoder_lr 1e-5. The row is labelled by what it varies, not by its name."
        ),
        extra={"freeze_encoder": False},
    ),
    SystemSpec(
        system_id="A5",
        role="Guard contribution",
        description="S1 evaluated with the inference guard disabled.",
        executor=Executor.TRAINED_GENERATOR,
        resource=Resource.GPU_INFERENCE,
        encoder_arm="gatv2",
        fusion=FusionVariant.F2,
        text_mode=TextMode.FULL,
        base_model=LOCAL_BASE_MODEL,
        base_model_version=LOCAL_BASE_MODEL,
        base_model_release_date=LOCAL_BASE_MODEL_RELEASED,
        experiment_config="matrix_a5",
        training_config="generator",
        trained=False,
        guard=False,
        depends_on=("S1",),
        notes=(
            "NO TRAINING RUN. A5 is S1's checkpoint decoded with the guard off, which is "
            "why it depends on S1 and costs an inference pass rather than a training one. "
            "Guarded and unguarded are two separate table rows and neither is the "
            "headline on its own (D-073)."
        ),
    ),
    SystemSpec(
        system_id="A6",
        role="Generality",
        description="S1's recipe on a second base model (Qwen3-8B).",
        executor=Executor.TRAINED_GENERATOR,
        resource=Resource.GPU,
        encoder_arm="gatv2",
        fusion=FusionVariant.F2,
        text_mode=TextMode.FULL,
        base_model=SECOND_BASE_MODEL,
        base_model_version=SECOND_BASE_MODEL,
        base_model_release_date=SECOND_BASE_MODEL_RELEASED,
        experiment_config="matrix_a6",
        training_config="generator",
        trained=True,
        depends_on=("encoder:gatv2",),
        notes=(
            "MATTERS MORE THAN IT LOOKS. A single-base-model result generalises weakly, "
            "and a fusion layer that works only on Llama's embedding geometry is a "
            "property of Llama. The soft tokens are scaled to whichever base model's own "
            "embedding RMS (fusion.target_rms is null = measure at build time), so this "
            "arm needs no constant changed -- which is itself the claim being tested."
        ),
    ),
)

_BY_ID: dict[str, SystemSpec] = {spec.system_id: spec for spec in _SYSTEMS}


def all_systems() -> tuple[SystemSpec, ...]:
    """Return every system in the matrix.

    Returns:
        The specs in registry order: baselines, then ours, then ablations. That order is
        the reading order of the paper's tables, so nothing downstream has to re-sort to
        produce a table a reader can follow.
    """
    return _SYSTEMS


def system_ids() -> tuple[str, ...]:
    """Return every system id, in registry order.

    Returns:
        The identifiers.
    """
    return tuple(spec.system_id for spec in _SYSTEMS)


def get_system(system_id: str) -> SystemSpec:
    """Look up one system.

    Args:
        system_id: The identifier.

    Returns:
        Its spec.

    Raises:
        UnknownSystemError: If no such system is registered.
    """
    try:
        return _BY_ID[system_id]
    except KeyError as exc:
        raise UnknownSystemError(
            f"{system_id!r} is not a registered system; known: {', '.join(system_ids())}"
        ) from exc


def expand_runs(
    specs: Sequence[SystemSpec] | None = None,
) -> tuple[tuple[SystemSpec, int], ...]:
    """Expand systems into (spec, seed) pairs, which is what the runner schedules.

    Args:
        specs: Systems to expand; the whole registry when omitted.

    Returns:
        One pair per run, in registry order and then ascending seed. The length of this
        is the matrix's true size, which is larger than the system count because of the
        seed asymmetry and is the number that should be quoted when estimating compute.
    """
    chosen = specs if specs is not None else _SYSTEMS
    return tuple((spec, seed) for spec in chosen for seed in sorted(spec.seeds))


def resolution_order(specs: Sequence[SystemSpec] | None = None) -> tuple[SystemSpec, ...]:
    """Order systems so that every dependency precedes its dependents.

    Dependencies naming an artifact outside the matrix -- ``encoder:gatv2`` and the like
    -- are checked for existence by the runner and ignored here, because a Phase 7
    checkpoint is a precondition of the matrix rather than a member of it.

    Args:
        specs: Systems to order; the whole registry when omitted.

    Returns:
        The specs in a valid execution order, stable within a dependency level so two
        runs of the planner produce the same order.

    Raises:
        ValueError: If the dependency graph has a cycle, or names a system that is not in
            the set being ordered.
    """
    chosen = list(specs if specs is not None else _SYSTEMS)
    present = {spec.system_id for spec in chosen}
    pending = {spec.system_id: {d for d in spec.depends_on if ":" not in d} for spec in chosen}
    for system_id, deps in pending.items():
        missing = deps - present
        if missing:
            raise ValueError(
                f"{system_id} depends on {sorted(missing)}, which is not in the selection; "
                "select those systems too or run them first"
            )

    ordered: list[SystemSpec] = []
    remaining = list(chosen)
    satisfied: set[str] = set()
    while remaining:
        ready = [spec for spec in remaining if pending[spec.system_id] <= satisfied]
        if not ready:
            stuck = sorted(spec.system_id for spec in remaining)
            raise ValueError(f"dependency cycle among {stuck}")
        ordered.extend(ready)
        satisfied.update(spec.system_id for spec in ready)
        remaining = [spec for spec in remaining if spec.system_id not in satisfied]
    return tuple(ordered)


def comparison_family(metric: str) -> str:
    """Name the correction family a metric's comparisons belong to.

    The family is every pairwise comparison of one metric on one slice, which is what one
    call to :func:`g2t_aml.eval.statistics.compare_systems` produces. Naming it here keeps
    the label consistent between the aggregator and the tables (D-079).

    Args:
        metric: The metric name.

    Returns:
        The family label.
    """
    return f"phase11-matrix/{metric}"


def central_claim_family() -> tuple[tuple[str, str], ...]:
    """Return the comparisons the paper's claims actually rest on.

    Stated as a fixed list rather than derived, because "which comparisons is the paper
    making" is a question that must be answerable before the numbers exist. Everything
    else in the matrix is context.

    Returns:
        Ordered ``(treatment, control)`` pairs: Gate 8 first, then the graph-over-text
        question, then the headline arm against the threatening baseline.
    """
    return (
        ("S1", "A1"),
        ("S1", "B7"),
        ("S2", "B7"),
        ("S1", "B8"),
        ("S1", "A2"),
        ("S1", "B5"),
    )


def validate_registry(  # noqa: PLR0912 -- one check per branch, and each branch is a
    # distinct way the matrix can be wrong; collapsing them would lose the messages.
    specs: Sequence[SystemSpec] | None = None,
) -> list[str]:
    """Check the registry's internal consistency and return every problem found.

    Returns problems rather than raising on the first, so a config edit that breaks three
    things reports three things. The runner calls this before scheduling anything and
    refuses to start if it returns a non-empty list.

    Args:
        specs: Systems to validate; the whole registry when omitted.

    Returns:
        One human-readable line per problem. Empty when the registry is consistent.
    """
    chosen = list(specs if specs is not None else _SYSTEMS)
    problems: list[str] = []

    seen: set[str] = set()
    for spec in chosen:
        if spec.system_id in seen:
            problems.append(f"duplicate system id {spec.system_id}")
        seen.add(spec.system_id)

    for spec in chosen:
        if spec.is_central and spec.seeds != SEEDS_CENTRAL:
            problems.append(
                f"{spec.system_id} carries the central claim but runs {len(spec.seeds)} "
                f"seed(s); the policy is {len(SEEDS_CENTRAL)}"
            )
        if not spec.is_central and len(spec.seeds) != len(SEEDS_SINGLE):
            problems.append(
                f"{spec.system_id} is not a central-claim system but runs "
                f"{len(spec.seeds)} seeds; extending seeds is a decision that belongs in "
                "DECISIONS.md and in CENTRAL_CLAIM_SYSTEMS, not in one spec"
            )
        if spec.base_model is not None and spec.base_model_release_date is None:
            problems.append(
                f"{spec.system_id} names a base model with no release date; a comparison "
                "table that cannot be dated is a desk-reject trigger at this venue"
            )
        if (
            spec.base_model_release_date is not None
            and spec.base_model_release_date < _BASELINE_FLOOR_DATE
        ):
            problems.append(
                f"{spec.system_id} uses {spec.base_model} released "
                f"{spec.base_model_release_date}, before the {_BASELINE_FLOOR_DATE} floor"
            )
        if spec.trained and spec.training_config is None:
            problems.append(f"{spec.system_id} is trained but names no training config")
        if (
            spec.uses_encoder
            and spec.fusion in (FusionVariant.F0, FusionVariant.NA)
            and spec.executor is not Executor.CLASSIFIER_TEMPLATE
        ):
            problems.append(
                f"{spec.system_id} configures an encoder but no fusion variant, so "
                "the encoder's output would reach nothing"
            )
        if spec.fusion not in (FusionVariant.F0, FusionVariant.NA) and not spec.uses_encoder:
            problems.append(f"{spec.system_id} configures fusion {spec.fusion} with no encoder arm")
        if spec.executor is Executor.TRAINED_GENERATOR and spec.experiment_config is None:
            problems.append(f"{spec.system_id} is a trained arm with no experiment config")

    # S1 and A1 must differ in the shuffle flag and NOTHING else. This is the same
    # assertion tests/unit/test_generator_configs.py makes about the Hydra composition,
    # restated at the registry level: an A1 that differs from S1 in a second axis is not a
    # control, and the Phase 9 log records how close that came to happening silently.
    if {"S1", "A1"} <= seen:
        s1, a1 = _BY_ID["S1"], _BY_ID["A1"]
        differing = [
            name
            for name in (
                "encoder_arm",
                "fusion",
                "text_mode",
                "base_model",
                "training_config",
                "guard",
                "seeds",
            )
            if getattr(s1, name) != getattr(a1, name)
        ]
        if differing:
            problems.append(
                f"A1 differs from S1 on {differing}; the control must differ ONLY in the "
                "fusion shuffle flag (D-071)"
            )

    return problems


def matrix_summary(specs: Sequence[SystemSpec] | None = None) -> dict[str, Any]:
    """Summarise the matrix's shape, for the run log and the paper's methods section.

    Args:
        specs: Systems to summarise; the whole registry when omitted.

    Returns:
        Counts by resource and by seed policy, the total run count, and the base models
        with their dates -- everything the methods section has to state.
    """
    chosen = list(specs if specs is not None else _SYSTEMS)
    by_resource: dict[str, int] = {}
    for spec in chosen:
        by_resource[str(spec.resource)] = by_resource.get(str(spec.resource), 0) + 1
    models = {
        spec.base_model: spec.base_model_release_date
        for spec in chosen
        if spec.base_model is not None
    }
    return {
        "n_systems": len(chosen),
        "n_runs": len(expand_runs(chosen)),
        "n_trained": sum(1 for s in chosen if s.trained),
        "runs_by_resource": dict(sorted(by_resource.items())),
        "multi_seed_systems": sorted(s.system_id for s in chosen if s.is_central),
        "seeds_central": list(SEEDS_CENTRAL),
        "seeds_single": list(SEEDS_SINGLE),
        "seed_extension_order": list(SEED_EXTENSION_ORDER),
        "base_models": dict(sorted(models.items())),
        "a3_f1_point_is_b8": _A3_REUSES_B8,
        "central_claim_comparisons": [list(pair) for pair in central_claim_family()],
    }


def with_seeds(system_id: str, seeds: Sequence[int]) -> SystemSpec:
    """Return a copy of a spec running at different seeds.

    Exists for the documented extension path -- if compute frees up, A2 and B8 get seeds
    next -- so extending is a call with a recorded argument rather than an edit to the
    table.

    Args:
        system_id: The system to copy.
        seeds: The seeds to run at.

    Returns:
        A new spec.

    Raises:
        UnknownSystemError: If no such system is registered.
        ValueError: If ``seeds`` is empty.
    """
    if not seeds:
        raise ValueError("a system must run at at least one seed")
    return replace(get_system(system_id), seeds=tuple(sorted(seeds)))


def iter_runs() -> Iterator[tuple[str, int]]:
    """Iterate every ``(system_id, seed)`` the matrix contains.

    Yields:
        The pairs, in scheduling order.
    """
    for spec, seed in expand_runs(resolution_order()):
        yield spec.system_id, seed
