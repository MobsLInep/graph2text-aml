#!/usr/bin/env python
"""B6: the untuned local base model, zero-shot, on the baselines' own prompt.

B6 is "B7 minus the finetuning" and "B3 minus the frontier model": the same base model B7
fine-tunes, given the **same prompt file** B3, B4 and B5 use. So the prompt is not built
here. Each case's prompt is rendered by :func:`g2t_aml.experiments.baselines.generate_zero_shot`
itself, through a teacher that records the rendered prompt instead of calling anything --
the prompt B6 sees is byte-for-byte the prompt B3 would send (D-117).

Generation is then batched on the GPU: the base model in 4-bit, the model's own chat
template, greedy, up to ``--max-new-tokens``. ``generations.jsonl`` is appended to after
every batch and existing cases are skipped, so an interrupted run resumes.

Usage:
    uv run python scripts/09c_local_zero_shot.py --out-dir <run dir> --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from g2t_aml.corpus.factsio import facts_from_dict
from g2t_aml.corpus.silver.api_client import ScriptedTeacher, TeacherSpec
from g2t_aml.corpus.silver.prompts import RenderedPrompt
from g2t_aml.experiments.baselines import GENERATE_PROMPT_NAME, generate_zero_shot
from g2t_aml.facts.vocab import load_vocabulary
from g2t_aml.utils.io import read_jsonl, write_json, write_jsonl
from g2t_aml.utils.logging import configure_logging, get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def render_prompts(records: list[dict[str, Any]]) -> dict[str, RenderedPrompt]:
    """Render every case's zero-shot prompt through the baselines' own code path.

    Args:
        records: Test records carrying their fact record under ``facts``.

    Returns:
        Case id to the exact prompt :func:`generate_zero_shot` would have sent.
    """
    vocabulary = load_vocabulary()
    captured: dict[str, RenderedPrompt] = {}

    def capture(prompt: RenderedPrompt, case_id: str, _kind: str, _attempt: int) -> str:
        captured[case_id] = prompt
        return ""

    spec = TeacherSpec(key="b6-local", family="open_weights", provider="local", model=MODEL)
    teacher = ScriptedTeacher(spec, capture)
    for record in records:
        facts = facts_from_dict(dict(record["facts"]))
        generate_zero_shot(facts, teacher, system="B6", vocabulary=vocabulary)
    return captured


def main() -> int:
    """Generate B6's narratives for the test split.

    Returns:
        0 on success, 1 when the corpus is missing.
    """
    parser = argparse.ArgumentParser(description="B6: untuned local zero-shot baseline.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "amlworld_hi_small" / "corpus" / "bronze.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="first N test cases; 0 for all")
    args = parser.parse_args()

    configure_logging()
    log = get_logger("b6")
    if not args.corpus.is_file():
        log.error("no corpus at %s", args.corpus)
        return 1

    records = sorted(
        (r for r in read_jsonl(args.corpus) if isinstance(r, dict) and r.get("split") == "test"),
        key=lambda r: str(r["case_id"]),
    )
    if args.limit:
        records = records[: args.limit]
    prompts = render_prompts(records)
    log.info("rendered %d zero-shot prompts from %s", len(prompts), GENERATE_PROMPT_NAME)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    ).eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "generations.jsonl"
    written = [dict(r) for r in read_jsonl(out)] if out.is_file() else []
    done = {str(r["case_id"]) for r in written}
    pending = [r for r in records if str(r["case_id"]) not in done]
    if done:
        log.info("resuming: %d cases already generated", len(done))

    for start in range(0, len(pending), args.batch_size):
        chunk = pending[start : start + args.batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": prompts[str(r["case_id"])].system},
                    {"role": "user", "content": prompts[str(r["case_id"])].user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in chunk
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for record, row in zip(chunk, generated[:, enc["input_ids"].shape[1] :], strict=True):
            case_id = str(record["case_id"])
            written.append(
                {
                    "case_id": case_id,
                    "texts": [tokenizer.decode(row, skip_special_tokens=True).strip()],
                    "system": "B6",
                    "seed": args.seed,
                    "stream": "balanced",
                    "split": "test",
                    "guard": False,
                    "model": args.model,
                    "prompt_name": prompts[case_id].template_name,
                    "prompt_hash": prompts[case_id].prompt_hash,
                    "rendered_hash": prompts[case_id].rendered_hash,
                    "n_generated": int((row != tokenizer.pad_token_id).sum()),
                    "prompt_length": int(enc["attention_mask"][0].sum()),
                }
            )
        write_jsonl(out, written)
        log.info("generated %d / %d", min(start + args.batch_size, len(pending)), len(pending))

    write_json(
        args.out_dir / "b6_run.json",
        {
            "model": args.model,
            "seed": args.seed,
            "n_cases": len(written),
            "max_new_tokens": args.max_new_tokens,
            "prompt_name": GENERATE_PROMPT_NAME,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
