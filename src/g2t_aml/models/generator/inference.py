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

    # Candidates are produced by repeating each row, so one batched forward pass covers
    # every candidate of every case rather than looping the model n times.
    input_ids = batch.input_ids.repeat_interleave(n_return, dim=0)
    attention_mask = batch.attention_mask.repeat_interleave(n_return, dim=0)
    soft_mask = batch.soft_mask.repeat_interleave(n_return, dim=0)

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

    for _ in range(cfg.max_new_tokens):
        out = generator.language_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=None
        )
        next_ids = _sample_next(out.logits[:, -1, :].float().cpu(), cfg, rng)

        for row in range(n_rows):
            if not bool(finished[row]):
                produced[row].append(int(next_ids[row]))
        if eos is not None:
            finished |= next_ids == eos
        if bool(finished.all()):
            break

        step_embeds = embed(next_ids.to(device)).unsqueeze(1)
        inputs_embeds = torch.cat([inputs_embeds, step_embeds], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((n_rows, 1), dtype=attention_mask.dtype, device=device)],
            dim=1,
        )

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
