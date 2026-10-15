"""Generation: greedy for measurement, sampled for the guard, resumable for the test set.

**vLLM is not used, and the reason is a finding rather than an omission.** vLLM's fast
paths are built around token ids: the scheduler, the paged KV cache and the prefix cache
all key on them. This model's graph conditioning enters as *embeddings* at reserved
positions, which vLLM's ``LLM.generate`` has no supported entry point for — ``prompt_embeds``
covers a whole prompt, not a splice at chosen positions inside one, and the soft tokens
differ per case so there is no shared prefix to cache. Working around it means either
reconstructing the embedding splice inside a custom vLLM worker or giving up the graph
pathway. So the batch evaluator uses HuggingFace ``generate`` with ``inputs_embeds`` and
batches manually. **The throughput cost is real and is recorded** by
:mod:`~g2t_aml.models.generator.profiling`, because a deployed-system paper that quietly
omits its serving throughput has skipped the question a practitioner asks first.

**Two modes, and they are not interchangeable.** Greedy decoding is what every reported
evaluation number is measured under, because a sampled number is a sample and reporting it
without its variance overstates precision. Sampling at temperature 0.6 exists to give the
guard four candidates to choose between. Reporting a guarded, sampled result as the
model's faithfulness conflates two different claims — see
:mod:`~g2t_aml.models.generator.guard`.

**Resumability is not a convenience here.** The test set is 3,192 cases at roughly a
second each without vLLM; that is a run long enough to be interrupted, and restarting from
zero wastes an hour. Results are appended as they complete and the runner skips case ids
already present.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from g2t_aml.models.generator.dataset import GeneratorBatch, GraphCollator
from g2t_aml.models.generator.model import Graph2TextGenerator
from g2t_aml.models.generator.prompts import IGNORE_INDEX, BuiltPrompt
from g2t_aml.utils.io import read_jsonl, write_jsonl

#: Fewer than this leaves the guard nothing to select between.
_MIN_GUARD_CANDIDATES = 2

__all__ = [
    "GenerationConfig",
    "GenerationResult",
    "generate_batch",
    "run_test_set",
]


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding parameters.

    Attributes:
        max_new_tokens: Cap on generated length.
        do_sample: False is greedy — the mode every reported number uses.
        temperature: Sampling temperature. 0.6 for the guard's candidates.
        top_p: Nucleus threshold.
        num_return_sequences: Candidates per case. 4 for the guard, 1 for evaluation.
        seed: Sampling seed, so a sampled run is reproducible.
    """

    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.6
    top_p: float = 0.9
    num_return_sequences: int = 1
    seed: int = 42

    @classmethod
    def deterministic(cls, *, max_new_tokens: int = 512) -> GenerationConfig:
        """Return the greedy configuration used for every reported evaluation.

        Args:
            max_new_tokens: Cap on generated length.

        Returns:
            A greedy configuration.
        """
        return cls(max_new_tokens=max_new_tokens, do_sample=False, num_return_sequences=1)

    @classmethod
    def guard_candidates(cls, *, n: int = 4, seed: int = 42) -> GenerationConfig:
        """Return the sampling configuration the guard draws candidates with.

        Args:
            n: How many candidates.
            seed: Sampling seed.

        Returns:
            A sampling configuration at temperature 0.6, top-p 0.9.

        Raises:
            ValueError: If fewer than two candidates are requested, which leaves the guard
                nothing to select between.
        """
        if n < _MIN_GUARD_CANDIDATES:
            raise ValueError(f"the guard needs at least 2 candidates to select between, got {n}")
        return cls(do_sample=True, temperature=0.6, top_p=0.9, num_return_sequences=n, seed=seed)

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a plain mapping.

        Returns:
            A JSON-serialisable mapping, recorded beside every generation.
        """
        return asdict(self)


@dataclass
class GenerationResult:
    """What one case produced.

    Attributes:
        case_id: The case.
        texts: The generated narratives — one under greedy decoding, several under
            sampling.
        prompt_length: Tokens in the prompt, for throughput accounting.
        n_generated: Tokens generated, summed over candidates.
        config: The decoding parameters used.
    """

    case_id: str
    texts: list[str]
    prompt_length: int
    n_generated: int
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a JSON-serialisable mapping.

        Returns:
            The fields.
        """
        return asdict(self)


def _sample_next(
    logits: Tensor, cfg: GenerationConfig, generator: torch.Generator | None
) -> Tensor:
    """Pick the next token from a logit row.

    Args:
        logits: ``[B, vocab]`` final-position logits.
        cfg: Decoding parameters.
        generator: Torch generator for reproducible sampling.

    Returns:
        ``[B]`` next token ids.
    """
    if not cfg.do_sample:
        return logits.argmax(dim=-1)

    scaled = logits / max(1e-6, cfg.temperature)
    probs = torch.softmax(scaled, dim=-1)

    ordered, index = probs.sort(dim=-1, descending=True)
    cumulative = ordered.cumsum(dim=-1)
    # Keep the smallest prefix whose mass reaches top_p. The shift keeps the first token
    # always eligible, so a single token above top_p does not empty the nucleus.
    drop = cumulative - ordered > cfg.top_p
    ordered = ordered.masked_fill(drop, 0.0)
    ordered = ordered / ordered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    picked = torch.multinomial(ordered, num_samples=1, generator=generator)
    return index.gather(-1, picked).squeeze(-1)


def _left_pad(
    input_ids: Tensor, attention_mask: Tensor, soft_mask: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Move each row's padding from the right to the left, and drop all-padding columns.

    Args:
        input_ids: ``[B, T]`` right-padded token ids.
        attention_mask: ``[B, T]``, 1 on real positions.
        soft_mask: ``[B, T]`` boolean, True at the reserved soft-token positions.

    Returns:
        The three tensors with every row's real tokens ending at the final column. Each
        row keeps its own tokens in their own order, so the soft-token positions still
        line up with the row they belong to.
    """
    width = input_ids.size(1)
    shifts = [width - int(n) for n in attention_mask.sum(dim=1)]
    keep = min(shifts)

    def shift(tensor: Tensor) -> Tensor:
        rows = [torch.roll(tensor[row], shifts=by, dims=0) for row, by in enumerate(shifts)]
        return torch.stack(rows)[:, keep:]

    return shift(input_ids), shift(attention_mask), shift(soft_mask)


def _decoder_and_head(language_model: Any) -> tuple[Any, Any] | None:
    """Split a Hugging Face causal LM into its decoder trunk and its output head.

    Only the final position's logits choose the next token, but a causal LM's own forward
    pass projects every position onto the vocabulary — on the prompt pass that is
    ``rows x prompt_length x vocab`` floats, several gigabytes for a batch of ten at an 8B
    base's 128k vocabulary. Running the trunk and applying the head to one position gives
    the same logits for that position without the rest.

    Args:
        language_model: The PEFT-wrapped base, a bare Hugging Face model, or a stub.

    Returns:
        ``(decoder, head)``, or None when the model does not expose them — the CPU stub —
        in which case the caller uses the model's ordinary forward pass. LoRA adapters are
        injected into the trunk's own layers, so calling the trunk directly still runs them.
    """
    unwrap = getattr(language_model, "get_base_model", None)
    base = unwrap() if callable(unwrap) else language_model
    prefix = getattr(base, "base_model_prefix", None)
    decoder = getattr(base, prefix, None) if isinstance(prefix, str) and prefix else None
    get_head = getattr(base, "get_output_embeddings", None)
    head = get_head() if callable(get_head) else None
    if decoder is None or head is None or not hasattr(head, "weight"):
        return None
    return decoder, head


def _decode_step(
    language_model: Any,
    split: tuple[Any, Any] | None,
    step_inputs: Tensor,
    attention_mask: Tensor,
    past: Any,
) -> tuple[Tensor, Any]:
    """Run one decoding step.

    Args:
        language_model: The language model.
        split: The result of :func:`_decoder_and_head`.
        step_inputs: The embeddings to feed — the whole prompt on the first step, one new
            position afterwards when a cache is in use.
        attention_mask: The mask over everything fed so far.
        past: The key-value cache from the previous step, or None.

    Returns:
        ``[B, vocab]`` logits for the final position, and the cache to pass to the next
        step — None when the backbone keeps none.
    """
    if split is None:
        out = language_model(
            inputs_embeds=step_inputs,
            attention_mask=attention_mask,
            labels=None,
            past_key_values=past,
            use_cache=True,
        )
        return out.logits[:, -1, :], getattr(out, "past_key_values", None)

    decoder, head = split
    out = decoder(
        inputs_embeds=step_inputs,
        attention_mask=attention_mask,
        past_key_values=past,
        use_cache=True,
    )
    final_hidden = out.last_hidden_state[:, -1, :]
    logits = head(final_hidden.to(head.weight.dtype))
    return logits, getattr(out, "past_key_values", None)


@torch.no_grad()
def generate_batch(
    generator: Graph2TextGenerator,
    batch: GeneratorBatch,
    cfg: GenerationConfig,
    *,
    tokenizer: Any,
    device: str = "cpu",
) -> list[GenerationResult]:
    """Generate narratives for one collated batch.

    Decoding runs as an explicit loop over ``inputs_embeds`` rather than through
    ``model.generate``. The splice has to happen on embeddings *before* the first forward
    pass and must not be recomputed on continuation steps, and expressing that through
    ``generate``'s hooks is more fragile than writing the loop — which also keeps the whole
    path runnable against a stub backbone, so the guard's selection logic is testable
    without a GPU.

    Args:
        generator: The trained generator.
        batch: A collated batch of *inference* prompts, whose labels are all ``-100``.
        cfg: Decoding parameters.
        tokenizer: Used to decode ids and to supply ``eos_token_id``.
        device: Device to run on.

    Returns:
        One result per case, in batch order.

    Raises:
        ValueError: If the batch carries supervised labels, which means it was built for
            training and its prompt already contains the answer.
    """
    if int((batch.labels != IGNORE_INDEX).sum()) > 0:
        raise ValueError(
            "this batch carries training labels; generating from it would condition on "
            "the target narrative. Build the prompt with for_training=False."
        )

    generator.eval()
    batch = batch.to(device)
    n_return = cfg.num_return_sequences

    # The collator pads on the right, which is correct for training and wrong here: the
    # next token is read from the final position, and on a right-padded row that position
    # is padding — for any prompt whose length is not a multiple of the collator's
    # `pad_to_multiple_of`, at batch size one included. Decoding needs the prompt to END at
    # the final position, so the padding moves to the left.
    row_ids, row_attention, row_soft = _left_pad(
        batch.input_ids, batch.attention_mask, batch.soft_mask
    )

    # Candidates are produced by repeating each row, so one batched forward pass covers
    # every candidate of every case rather than looping the model n times.
    input_ids = row_ids.repeat_interleave(n_return, dim=0)
    attention_mask = row_attention.repeat_interleave(n_return, dim=0)
    soft_mask = row_soft.repeat_interleave(n_return, dim=0)

    embed = generator.language_model.get_input_embeddings()
    inputs_embeds = embed(input_ids)

    if generator.fusion_projector is not None:
        pooled = generator.encode_graph(batch.graph_batch)
        soft = generator.fusion_projector(pooled).soft_tokens
        soft = soft.repeat_interleave(n_return, dim=0)
        inputs_embeds = generator.splice_soft_tokens(inputs_embeds, soft, soft_mask)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(cfg.seed)

    eos = getattr(tokenizer, "eos_token_id", None)
    n_rows = inputs_embeds.size(0)
    produced: list[list[int]] = [[] for _ in range(n_rows)]
    finished = torch.zeros(n_rows, dtype=torch.bool)

    # With a key-value cache each step feeds one new position; without one it re-runs the
    # whole sequence, which on an 8B base is minutes per narrative instead of seconds. A
    # backbone that returns no cache (the CPU stub) keeps the full-sequence path.
    step_inputs = inputs_embeds
    past: Any = None
    split = _decoder_and_head(generator.language_model)

    for _ in range(cfg.max_new_tokens):
        final_logits, cache = _decode_step(
            generator.language_model, split, step_inputs, attention_mask, past
        )
        next_ids = _sample_next(final_logits.float().cpu(), cfg, rng)

        for row in range(n_rows):
            if not bool(finished[row]):
                produced[row].append(int(next_ids[row]))
        if eos is not None:
            finished |= next_ids == eos
        if bool(finished.all()):
            break

        step_embeds = embed(next_ids.to(device)).unsqueeze(1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((n_rows, 1), dtype=attention_mask.dtype, device=device)],
            dim=1,
        )
        past = cache
        if past is None:
            inputs_embeds = torch.cat([inputs_embeds, step_embeds], dim=1)
            step_inputs = inputs_embeds
        else:
            step_inputs = step_embeds

    results: list[GenerationResult] = []
    for case_index, case_id in enumerate(batch.case_ids):
        rows = produced[case_index * n_return : (case_index + 1) * n_return]
        texts = [
            tokenizer.decode([t for t in row if eos is None or t != eos], skip_special_tokens=True)
            for row in rows
        ]
        results.append(
            GenerationResult(
                case_id=case_id,
                texts=texts,
                prompt_length=int(batch.attention_mask[case_index].sum()),
                n_generated=sum(len(r) for r in rows),
                config=cfg.to_dict(),
            )
        )
    return results


def run_test_set(
    generator: Graph2TextGenerator,
    items: Sequence[tuple[BuiltPrompt, Any]],
    *,
    collator: GraphCollator,
    tokenizer: Any,
    cfg: GenerationConfig,
    output_path: str | Path,
    batch_size: int = 4,
    device: str = "cpu",
    log: Any = None,
) -> Path:
    """Generate over a whole split, appending as it goes and skipping completed cases.

    Args:
        generator: The trained generator.
        items: ``(prompt, graph)`` pairs, built with ``for_training=False``.
        collator: The collator.
        tokenizer: The tokeniser.
        cfg: Decoding parameters.
        output_path: JSONL to append results to. Existing case ids are skipped, which is
            what makes an interrupted run resumable.
        batch_size: Cases per batch. Distinct from the training batch size: generation
            holds no backward graph, so it fits more rows.
        device: Device to run on.
        log: Optional logger.

    Returns:
        The output path.
    """
    path = Path(output_path)
    done: set[str] = set()
    if path.is_file():
        done = {str(row["case_id"]) for row in read_jsonl(path)}
        if log is not None:
            log.info("resuming: %d cases already generated", len(done))

    pending = [item for item in items if item[0].case_id not in done]
    written = [dict(row) for row in read_jsonl(path)] if path.is_file() else []

    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        results = generate_batch(
            generator, collator(chunk), cfg, tokenizer=tokenizer, device=device
        )
        written.extend(r.to_dict() for r in results)
        # Rewritten atomically after every batch: an interrupted run leaves a valid file
        # holding every completed case, never a half-written record a resume would trust.
        write_jsonl(path, written)
        if log is not None:
            log.info("generated %d / %d", min(start + batch_size, len(pending)), len(pending))

    return path
