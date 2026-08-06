"""Phase 9: the QLoRA generator, its training harness, and the inference guard.

The pipeline this package implements, end to end::

    case subgraph -> Phase 7 encoder -> Phase 8 fusion -> soft tokens
                                                              |
    serialised facts + instruction ---------------------------+--> Llama-3.1-8B (4-bit,
                                                                   LoRA) -> narrative
                                                                              |
                                          Phase 3 checker <-- guard <----------+

Five things in here are load-bearing and are documented where they live rather than here:

- The fusion projector trains in **fp32** and is never quantised
  (:func:`~g2t_aml.models.fusion.base.assert_projector_is_fp32`).
- The loss is computed on the **completion only** — never the system message, the prompt,
  the soft-token positions or the padding
  (:mod:`~g2t_aml.models.generator.prompts`).
- **Three learning rates**, because the randomly-initialised projector and the pretrained
  adapters do not converge on the same schedule
  (:meth:`~g2t_aml.models.generator.model.Graph2TextGenerator.trainable_parameter_groups`).
- **Gold test items never reach training**, asserted in the loader rather than remembered
  (:func:`~g2t_aml.models.generator.dataset.assert_no_gold_test`).
- The **shuffled control** runs at every checkpoint, and the guarded and unguarded results
  are reported as two separate rows
  (:mod:`~g2t_aml.models.generator.callbacks`, :mod:`~g2t_aml.models.generator.guard`).

``transformers``, ``peft`` and ``bitsandbytes`` are the ``llm`` extra and are imported
lazily. Every code path here — the collator, the loss masking, the overfit check, the
guard's selection, the checkpoint round-trip — runs against a stub backbone on CPU, in the
same way :class:`~g2t_aml.corpus.silver.api_client.ScriptedTeacher` exercises Phase 5
without a provider SDK.
"""

from g2t_aml.models.generator.callbacks import (
    ArmComparison,
    AttentionMassCallback,
    FaithfulnessCallback,
    FaithfulnessScore,
    ProbeCase,
    compare_arms,
    score_narrative,
)
from g2t_aml.models.generator.dataset import (
    CurriculumStage,
    GeneratorBatch,
    GeneratorDataset,
    GraphCollator,
    assert_no_gold_test,
    build_curriculum,
    loss_mask_report,
)
from g2t_aml.models.generator.guard import (
    DEFAULT_WEIGHTS,
    CandidateScore,
    GuardReport,
    GuardStatistics,
    GuardWeights,
    InferenceGuard,
    score_candidate,
)
from g2t_aml.models.generator.inference import (
    GenerationConfig,
    GenerationResult,
    generate_batch,
    run_test_set,
)
from g2t_aml.models.generator.model import (
    CausalLM,
    GeneratorConfig,
    GeneratorOutput,
    Graph2TextGenerator,
    LoraConfigSpec,
    QuantizationSpec,
    build_generator,
    load_base_model,
)
from g2t_aml.models.generator.profiling import (
    DeviceInfo,
    PhaseProfile,
    RunProfile,
    device_info,
    profile_phase,
)
from g2t_aml.models.generator.prompts import (
    DEFAULT_SYSTEM_MESSAGE,
    TEXT_MODES,
    BuiltPrompt,
    PromptBuilder,
    PromptSegment,
    SegmentRole,
    Tokenizer,
)
from g2t_aml.models.generator.train import (
    GeneratorTrainer,
    TrainingConfig,
    TrainingState,
    build_optimizer,
    cosine_schedule_with_warmup,
    overfit_check,
)

__all__ = [
    "DEFAULT_SYSTEM_MESSAGE",
    "DEFAULT_WEIGHTS",
    "TEXT_MODES",
    "ArmComparison",
    "AttentionMassCallback",
    "BuiltPrompt",
    "CandidateScore",
    "CausalLM",
    "CurriculumStage",
    "DeviceInfo",
    "FaithfulnessCallback",
    "FaithfulnessScore",
    "GenerationConfig",
    "GenerationResult",
    "GeneratorBatch",
    "GeneratorConfig",
    "GeneratorDataset",
    "GeneratorOutput",
    "GeneratorTrainer",
    "Graph2TextGenerator",
    "GraphCollator",
    "GuardReport",
    "GuardStatistics",
    "GuardWeights",
    "InferenceGuard",
    "LoraConfigSpec",
    "PhaseProfile",
    "ProbeCase",
    "PromptBuilder",
    "PromptSegment",
    "QuantizationSpec",
    "RunProfile",
    "SegmentRole",
    "Tokenizer",
    "TrainingConfig",
    "TrainingState",
    "assert_no_gold_test",
    "build_curriculum",
    "build_generator",
    "build_optimizer",
    "compare_arms",
    "cosine_schedule_with_warmup",
    "device_info",
    "generate_batch",
    "load_base_model",
    "loss_mask_report",
    "overfit_check",
    "profile_phase",
    "run_test_set",
    "score_candidate",
    "score_narrative",
]
