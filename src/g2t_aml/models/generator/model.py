"""The generator: a 4-bit Llama-3.1-8B with LoRA adapters and an fp32 graph projector.

Three components and one splice. The Phase 7 encoder turns a case subgraph into ``k``
pooled tokens; the Phase 8 fusion layer projects them into the language model's embedding
space; and this module splices the result into the input sequence at reserved positions,
so the language model reads the graph as if it were text it had already been given.

**The splice is done on embeddings, not on token ids**, which is why every path here runs
through ``inputs_embeds`` rather than ``input_ids``. The collator writes a placeholder
token at each reserved position purely so that lengths, padding and the attention mask are
computed by the tokeniser in the ordinary way; those placeholder embeddings are then
overwritten and never contribute a gradient.

**What is quantised and what is not.**

===================  ==============  ==================================================
Component            Precision       Why
===================  ==============  ==================================================
Llama base weights   nf4, frozen     8B parameters do not otherwise fit, and they are
                                     not being trained.
LoRA adapters        bf16            Trained. Small enough that precision costs nothing.
Fusion projector     **fp32**        Randomly initialised, learning to land inside an
                                     already-trained embedding distribution. Quantising
                                     it does not crash; it produces a fourteen-hour run
                                     whose soft tokens the LM reads as noise.
GAT encoder          fp32            Pretrained in Phase 7 at fp32; frozen by default.
===================  ==============  ==================================================

:func:`~g2t_aml.models.fusion.base.assert_projector_is_fp32` runs after construction, not
as a comment. PEFT walks the module tree replacing and casting layers, so a projector that
was fp32 when it was built is not necessarily fp32 by the time the first batch arrives —
and ``modules_to_save`` is precisely the mechanism that walks it.

**transformers and peft are imported lazily**, inside the builders that need them. They
are the ``llm`` extra and are GPU-only; the same discipline that keeps
:class:`~g2t_aml.corpus.silver.api_client.Teacher` behind a protocol so ``ScriptedTeacher``
can exercise Phase 5 without an SDK keeps :class:`CausalLM` behind a protocol here, so the
whole training harness is testable on a CPU box with neither library installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from g2t_aml.models.fusion.base import FusionOutput, assert_projector_is_fp32, embedding_rms

__all__ = [
    "DEFAULT_TARGET_MODULES",
    "CausalLM",
    "GeneratorConfig",
    "GeneratorOutput",
    "Graph2TextGenerator",
    "LoraConfigSpec",
    "QuantizationSpec",
    "build_generator",
    "load_base_model",
]

#: The attention and MLP projections LoRA adapts. All seven, not just the attention four:
#: on an instruction-tuned base the MLP projections carry a large share of the adaptation,
#: and the memory difference at r=32 is small next to the frozen 4-bit trunk.
DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@runtime_checkable
class CausalLM(Protocol):
    """The slice of a causal language model this project uses.

    Narrow on purpose. Everything the trainer, the callbacks and the guard need is here,
    so a stub implementing these four members exercises every code path in Phase 9 without
    ``transformers`` being installed — which is what makes the overfit test, the loss-mask
    assertions and the checkpoint round-trip runnable in CI on CPU.
    """

    config: Any

    def get_input_embeddings(self) -> nn.Module:
        """Return the input embedding module.

        Returns:
            The embedding layer, whose weight supplies both the embedding lookup and the
            RMS the soft tokens are scaled to.
        """
        ...

    def __call__(self, **kwargs: Any) -> Any:
        """Run a forward pass.

        Args:
            **kwargs: ``inputs_embeds``, ``attention_mask``, ``labels``,
                ``output_attentions``.

        Returns:
            An object carrying ``loss``, ``logits`` and optionally ``attentions``.
        """
        ...


@dataclass(frozen=True)
class QuantizationSpec:
    """4-bit loading parameters for the frozen base model.

    Attributes:
        load_in_4bit: Whether to quantise at all. False loads in
            :attr:`GeneratorConfig.dtype`, which needs ~16 GB and is only for a machine
            that has it.
        bnb_4bit_quant_type: ``nf4`` or ``fp4``.
        bnb_4bit_use_double_quant: Quantise the quantisation constants too, saving a
            further ~0.4 GB at 8B.
        bnb_4bit_compute_dtype: Dtype the dequantised matmuls run in.
    """

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass(frozen=True)
class LoraConfigSpec:
    """LoRA hyperparameters.

    Attributes:
        r: Rank. 32 is the configured default; 16 is the documented OOM fallback and must
            be applied to every arm or none, because rank is capacity and an arm with less
            of it is not a like-for-like comparison.
        alpha: Scaling numerator; the update is scaled by ``alpha / r``.
        dropout: Dropout on the LoRA path.
        target_modules: Which projections to adapt.
        modules_to_save: Modules trained in full and saved with the adapter. The fusion
            projector lives here — it is new parameters, not an adaptation of existing
            ones, so there is nothing for a low-rank update to be low-rank *against*.
        bias: PEFT's bias policy.
    """

    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    modules_to_save: tuple[str, ...] = ("fusion_projector",)
    bias: str = "none"


@dataclass(frozen=True)
class GeneratorConfig:
    """Everything needed to build the generator.

    Attributes:
        base_model: HuggingFace model id or a local path.
        dtype: Compute dtype for the unquantised parts.
        attn_implementation: ``flash_attention_2`` where the card supports it, else
            ``sdpa``. Recorded because it changes throughput but not results, and Phase 13
            reports throughput.
        quantization: 4-bit parameters.
        lora: Adapter parameters.
        max_seq_len: Truncation length. A deviation here must be applied to every arm
            uniformly or not at all — it changes how much of the serialised facts survives
            truncation, which is a change to the *input*, not to the optimiser.
        gradient_checkpointing: Trade compute for activation memory.
        freeze_encoder: Keep the Phase 7 encoder frozen. True by default: the encoder was
            selected on val AUC-PR and unfreezing it lets the narrative loss undo that
            selection, so an unfrozen run is a separate arm rather than a better default.
        text_mode: What text accompanies the soft tokens. ``full`` is the serialised
            facts plus the instruction (S1, B7); ``none`` is the instruction only, so the
            graph is the sole source of case information (S2, the headline); ``serialised``
            is the facts with no graph at all (B7's ablation partner).
        use_fusion: False builds a text-only model with no encoder and no projector — the
            B7 baseline.
        soft_token_id: Placeholder token id reserved for soft-token positions. Resolved
            from the tokeniser at build time; the default is a value the collator can use
            in tests.
    """

    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    quantization: QuantizationSpec = field(default_factory=QuantizationSpec)
    lora: LoraConfigSpec = field(default_factory=LoraConfigSpec)
    max_seq_len: int = 2048
    gradient_checkpointing: bool = True
    freeze_encoder: bool = True
    text_mode: str = "full"
    use_fusion: bool = True
    soft_token_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a plain mapping for the checkpoint and run record.

        Returns:
            A JSON-serialisable mapping.
        """
        return asdict(self)


@dataclass
class GeneratorOutput:
    """One forward pass.

    Attributes:
        loss: The completion-only cross-entropy, or None when no labels were given.
        logits: ``[B, T, vocab]``.
        fusion: The fusion layer's output and diagnostics, or None on a text-only arm.
        attentions: Per-layer attention, present only when requested. Retained on
            diagnostic steps rather than on every step, because it is ``[B, heads, T, T]``
            per layer and does not fit otherwise.
    """

    loss: Tensor | None
    logits: Tensor
    fusion: FusionOutput | None = None
    attentions: tuple[Tensor, ...] | None = None


def _torch_dtype(name: str) -> torch.dtype:
    """Resolve a dtype name to a torch dtype.

    Args:
        name: ``bfloat16``, ``float16`` or ``float32``.

    Returns:
        The dtype.

    Raises:
        ValueError: If the name is not one of the three.
    """
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in dtypes:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(dtypes)}")
    return dtypes[name]


def load_base_model(cfg: GeneratorConfig) -> tuple[Any, Any]:
    """Load the quantised base model and its tokeniser.

    Imports ``transformers``, ``peft`` and ``bitsandbytes`` lazily, so importing this
    module on a CPU-only box costs nothing and fails nothing.

    Args:
        cfg: The generator configuration.

    Returns:
        ``(model, tokenizer)``. The model is the PEFT-wrapped base with LoRA adapters
        attached and the trunk frozen; the tokeniser has a pad token guaranteed.

    Raises:
        ImportError: If the ``llm`` extra is not installed, with the command to install
            it.
    """
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the generator needs the `llm` extra: `uv sync --extra llm`. Every other "
            "Phase 9 code path runs without it against a stub backbone."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        # Llama-3.1 ships no pad token. Reusing EOS is standard and safe here because the
        # collator masks every pad position out of the loss, so the model is never taught
        # to emit EOS at a padded position.
        tokenizer.pad_token = tokenizer.eos_token

    quant = None
    if cfg.quantization.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=_torch_dtype(cfg.quantization.bnb_4bit_compute_dtype),
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quant,
        torch_dtype=_torch_dtype(cfg.dtype),
        attn_implementation=cfg.attn_implementation,
        device_map="auto",
    )
    if cfg.quantization.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )

    lora = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        bias=cfg.lora.bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()
    return model, tokenizer


class Graph2TextGenerator(nn.Module):
    """Encoder, fusion and language model, with the soft tokens spliced in.

    The fusion projector is registered as ``fusion_projector`` so that PEFT's
    ``modules_to_save`` finds it under the name
    :attr:`LoraConfigSpec.modules_to_save` declares.
    """

    def __init__(
        self,
        *,
        language_model: CausalLM,
        fusion: nn.Module | None = None,
        encoder: nn.Module | None = None,
        config: GeneratorConfig | None = None,
    ) -> None:
        """Assemble the generator.

        Args:
            language_model: The PEFT-wrapped quantised base, or a stub satisfying
                :class:`CausalLM`.
            fusion: The Phase 8 fusion layer, or None for a text-only arm.
            encoder: The Phase 7 encoder, or None when pooled tokens are supplied
                precomputed. Frozen when ``config.freeze_encoder``.
            config: The configuration. Defaults are used when omitted.

        Raises:
            TypeError: If the fusion projector is not fp32. Checked here rather than
                trusted, because this constructor runs after PEFT has walked the tree.
        """
        super().__init__()
        self.language_model = language_model
        self.fusion_projector = fusion
        self.encoder = encoder
        self.config = config if config is not None else GeneratorConfig()

        if self.encoder is not None and self.config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad_(False)

        if self.fusion_projector is not None:
            assert_projector_is_fp32(self.fusion_projector, name="fusion_projector")
            self._align_soft_token_scale()

    @property
    def lm_dim(self) -> int:
        """Return the language model's hidden width.

        Returns:
            The hidden size read from the model config.
        """
        return int(self.language_model.config.hidden_size)

    @property
    def n_soft_tokens(self) -> int:
        """Return how many soft tokens the fusion layer emits.

        Returns:
            The count, or 0 on a text-only arm.
        """
        return 0 if self.fusion_projector is None else int(self.fusion_projector.n_tokens)

    def _align_soft_token_scale(self) -> None:
        """Scale soft tokens to the base model's own input-embedding RMS.

        A projector whose output lands an order of magnitude off the scale of the token
        embeddings it sits among spends its early training budget fixing magnitude rather
        than content. Measuring the target from the loaded model beats hardcoding it,
        because the RMS differs between base models and between quantisation settings.
        """
        setter = getattr(self.fusion_projector, "set_target_rms", None)
        if not callable(setter):
            return
        weight = self.language_model.get_input_embeddings().weight
        setter(embedding_rms(weight))

    def encode_graph(self, graph_batch: object) -> Tensor:
        """Run the encoder and return its pooled tokens.

        Args:
            graph_batch: A PyG ``Batch`` of case subgraphs.

        Returns:
            ``[B, k, graph_dim]`` pooled tokens.

        Raises:
            RuntimeError: If this generator was built without an encoder.
        """
        if self.encoder is None:
            raise RuntimeError("this generator has no encoder; supply pooled_tokens instead")
        if self.config.freeze_encoder:
            with torch.no_grad():
                return self.encoder(graph_batch).pooled_tokens
        return self.encoder(graph_batch).pooled_tokens

    def splice_soft_tokens(
        self, inputs_embeds: Tensor, soft_tokens: Tensor, soft_mask: Tensor
    ) -> Tensor:
        """Overwrite the reserved positions with the projected graph tokens.

        Args:
            inputs_embeds: ``[B, T, lm_dim]`` embedded placeholder sequence.
            soft_tokens: ``[B, n_soft, lm_dim]`` from the fusion layer, fp32.
            soft_mask: ``[B, T]`` boolean, True at exactly the reserved positions.

        Returns:
            ``[B, T, lm_dim]`` with the soft tokens in place, in the embedding dtype.

        Raises:
            ValueError: If any row's reserved-position count differs from the number of
                soft tokens. This is the failure that must never pass silently: a
                mismatch shifts every subsequent position by one, so the loss mask no
                longer lines up with the completion and the model trains on a target that
                is off by a token.
        """
        per_row = soft_mask.sum(dim=1)
        expected = soft_tokens.size(1)
        if not bool((per_row == expected).all()):
            counts = sorted({int(v) for v in per_row})
            raise ValueError(
                f"every row must reserve exactly {expected} soft-token positions, found "
                f"{counts}; the collator and the fusion layer disagree about token count"
            )
        source = soft_tokens.reshape(-1, soft_tokens.size(-1)).to(inputs_embeds.dtype)
        return inputs_embeds.masked_scatter(
            soft_mask.unsqueeze(-1).expand_as(inputs_embeds), source
        )

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
        soft_mask: Tensor | None = None,
        graph_batch: object | None = None,
        pooled_tokens: Tensor | None = None,
        output_attentions: bool = False,
    ) -> GeneratorOutput:
        """Run one forward pass with the graph spliced into the sequence.

        Args:
            input_ids: ``[B, T]`` with placeholders at the reserved positions.
            attention_mask: ``[B, T]``.
            labels: ``[B, T]`` with ``-100`` everywhere the loss is masked — the system
                message, the prompt, the soft-token positions and the padding. Built by
                :func:`~g2t_aml.models.generator.dataset.build_labels`.
            soft_mask: ``[B, T]`` True at the reserved positions. Required whenever this
                generator has a fusion layer.
            graph_batch: A PyG ``Batch``, encoded here. Ignored when ``pooled_tokens`` is
                given.
            pooled_tokens: Precomputed ``[B, k, graph_dim]``, which is how the frozen
                encoder is run once per epoch rather than once per step.
            output_attentions: Retain attention weights for the soft-token attention-mass
                diagnostic.

        Returns:
            The loss, logits and fusion diagnostics.

        Raises:
            ValueError: If a fusion arm was given no ``soft_mask``, or no graph input.
        """
        embed = self.language_model.get_input_embeddings()
        inputs_embeds = embed(input_ids)
        fusion_out: FusionOutput | None = None

        if self.fusion_projector is not None:
            if soft_mask is None:
                raise ValueError("a fusion arm needs soft_mask marking the reserved positions")
            if pooled_tokens is None:
                if graph_batch is None:
                    raise ValueError("supply either graph_batch or pooled_tokens")
                pooled_tokens = self.encode_graph(graph_batch)
            fusion_out = self.fusion_projector(pooled_tokens)
            inputs_embeds = self.splice_soft_tokens(
                inputs_embeds, fusion_out.soft_tokens, soft_mask
            )

        result = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=output_attentions,
        )
        return GeneratorOutput(
            loss=getattr(result, "loss", None),
            logits=result.logits,
            fusion=fusion_out,
            attentions=getattr(result, "attentions", None) if output_attentions else None,
        )

    def trainable_parameter_groups(
        self, *, lora_lr: float, fusion_lr: float, encoder_lr: float
    ) -> list[dict[str, Any]]:
        """Split trainable parameters into the three groups with their own learning rates.

        **A single learning rate across all three does not work**, which is why this is a
        method rather than a line in the trainer. The LoRA adapters modulate a trained
        8B model and want 2e-4. The projector is randomly initialised and has to travel a
        long way to land inside the embedding distribution; at 2e-4 it is still wandering
        when the LoRA path has converged, and the model learns to solve the task from the
        text alone — which is indistinguishable, in every metric, from the graph being
        useless. The encoder is already trained and wants 1e-5 if it is unfrozen at all.

        Args:
            lora_lr: Learning rate for the adapters.
            fusion_lr: Learning rate for the fusion projector.
            encoder_lr: Learning rate for the encoder, used only when it is unfrozen.

        Returns:
            One group per component that has trainable parameters, each carrying its
            ``lr`` and a ``name`` used by the per-group gradient-norm logging.
        """
        groups: list[dict[str, Any]] = []

        fusion_params = (
            [p for p in self.fusion_projector.parameters() if p.requires_grad]
            if self.fusion_projector is not None
            else []
        )
        encoder_params = (
            [p for p in self.encoder.parameters() if p.requires_grad]
            if self.encoder is not None
            else []
        )
        owned = {id(p) for p in fusion_params} | {id(p) for p in encoder_params}
        lora_params = [
            p for p in self.language_model.parameters() if p.requires_grad and id(p) not in owned
        ]

        if lora_params:
            groups.append({"name": "lora", "params": lora_params, "lr": lora_lr})
        if fusion_params:
            groups.append({"name": "fusion", "params": fusion_params, "lr": fusion_lr})
        if encoder_params:
            groups.append({"name": "encoder", "params": encoder_params, "lr": encoder_lr})
        return groups

    def state_for_checkpoint(self) -> dict[str, Any]:
        """Return the fusion and encoder state a checkpoint must carry.

        The LoRA adapters are saved by PEFT's own ``save_pretrained``; these two are not,
        and a checkpoint missing them restores a model whose graph pathway is randomly
        initialised while every adapter is trained — which loads without error and
        generates fluent, graph-free text.

        Returns:
            A mapping with the fusion and encoder state dicts, on CPU, and the
            configuration they were built from.
        """
        return {
            "fusion_state": (
                {k: v.detach().cpu() for k, v in self.fusion_projector.state_dict().items()}
                if self.fusion_projector is not None
                else None
            ),
            "encoder_state": (
                {k: v.detach().cpu() for k, v in self.encoder.state_dict().items()}
                if self.encoder is not None
                else None
            ),
            "generator_config": self.config.to_dict(),
        }

    def load_state_for_checkpoint(self, payload: dict[str, Any], *, strict: bool = True) -> None:
        """Restore the fusion and encoder state written by :meth:`state_for_checkpoint`.

        Args:
            payload: The mapping that method produced.
            strict: Require every key to match.

        Raises:
            ValueError: If the checkpoint carries fusion state and this generator has no
                fusion layer, or the reverse. Loading half a graph pathway is worse than
                failing, because it produces a model that runs.
        """
        fusion_state = payload.get("fusion_state")
        if (fusion_state is None) != (self.fusion_projector is None):
            raise ValueError(
                "checkpoint and model disagree about whether there is a fusion layer: "
                f"checkpoint has {'one' if fusion_state else 'none'}, model has "
                f"{'one' if self.fusion_projector else 'none'}"
            )
        if fusion_state is not None and self.fusion_projector is not None:
            self.fusion_projector.load_state_dict(fusion_state, strict=strict)

        encoder_state = payload.get("encoder_state")
        if encoder_state is not None and self.encoder is not None:
            self.encoder.load_state_dict(encoder_state, strict=strict)


def build_generator(
    cfg: GeneratorConfig,
    *,
    fusion: nn.Module | None = None,
    encoder: nn.Module | None = None,
    language_model: CausalLM | None = None,
) -> tuple[Graph2TextGenerator, Any]:
    """Build the generator, loading the base model unless one is supplied.

    Args:
        cfg: The generator configuration.
        fusion: The Phase 8 fusion layer, or None for a text-only arm.
        encoder: The Phase 7 encoder.
        language_model: A pre-built backbone. Supplying one skips the ``transformers``
            load entirely, which is how the tests run against a stub.

    Returns:
        ``(generator, tokenizer)``. The tokeniser is None when a backbone was supplied.

    Raises:
        TypeError: If the fusion projector is not fp32 after construction.
    """
    tokenizer = None
    if language_model is None:
        language_model, tokenizer = load_base_model(cfg)

    generator = Graph2TextGenerator(
        language_model=language_model,
        fusion=fusion if cfg.use_fusion else None,
        encoder=encoder if cfg.use_fusion else None,
        config=cfg,
    )
    if generator.fusion_projector is not None:
        # Re-checked after assembly: this is the assertion the brief asks for, and the
        # window it guards is exactly the one between construction and the first step.
        assert_projector_is_fp32(generator.fusion_projector, name="fusion_projector")
    return generator, tokenizer
