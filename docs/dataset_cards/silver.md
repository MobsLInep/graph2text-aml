# Dataset card — Silver corpus

## **This corpus does not exist. Zero records have been generated.**

| | |
|---|---|
| Card written | 2026-08-06 (Phase 14), for release `v0.1.0` |
| Records | **0** |
| Blocker | **No teacher-API credentials.** Zero teacher calls have ever been made. |
| Machinery | Complete, tested, and exercised end to end with a scripted teacher |
| Code | `src/g2t_aml/corpus/silver/` · gate: `make silver-gate` |

This card is published rather than omitted for two reasons: a reviewer checking whether the
paper's corpus claims are honest needs it, and anyone who obtains credentials and wants to
build the tier needs to know what they will get and what it will cost.

**Nothing below is a measurement. Everything is a description of machinery.**

---

## 1. What it would be

LLM rewrites of Bronze narratives, each gated by the same Phase 3 checker that measures
faithfulness at evaluation time — run in reverse. Silver is the tier that gives the corpus
stylistic variety without giving up checkability.

A rewrite that asserts an unsupported or masked fact gets **at most two targeted repair
attempts, then is discarded and logged** (D-046).

> **The discard log is a deliverable, not a debug artifact.** It is the record of how often
> a frontier model, explicitly instructed to paraphrase without adding facts, added facts
> anyway. That rate is a finding in its own right and it would be reported.

Records would be `training_record_v1.json` at 1.0.0, **differing from Bronze only in
`tier` and `generator`** (D-037), and gated by the same ten-point harness.

---

## 2. Construction design

### Two teachers, from different families, and the code refuses one

Assigned **deterministically and balanced per stratum**. One teacher alone is distillation
from a single model, and `src/g2t_aml/corpus/silver/` will not run in that configuration.

### The loop

```
Bronze narrative ─► teacher rewrite ─► checker
                                        ├─ all SUPPORTED ──────────────► accept
                                        └─ any CONTRADICTED/UNVERIFIABLE
                                             └─► targeted repair (max 2) ─► accept
                                                                   └─────► discard + log
```

Driven entirely through the `Teacher` protocol, so `ScriptedTeacher` exercises the whole
pipeline with **no network**, and the base environment needs no provider SDK — `anthropic`
is the optional `api` extra.

### Two things that would silently corrupt this tier, and are fixed

- **Slot alignment runs longest-value-first, on token boundaries** (D-048). In Bronze's
  document order a long value is always reached before the short values hiding inside it.
  A rewrite reorders content, and then `2` aligns inside `2022-09-02 15:01` — the timestamp
  is scored as dropped **and** its digits come back as invented. Document-order alignment
  **failed 102 of 300 real paraphrased cases** and would have put roughly 34 spurious
  points into the discard rate. This was Phase 5's headline finding. **Do not "simplify"
  the ordering back.**
- **Every current frontier Anthropic model rejects `temperature` and `top_p` with a 400**
  (D-045) — Opus 5, Sonnet 5, Opus 4.8, 4.7. They are not ignored; a teacher spec setting
  `supports_sampling: true` on one of them fails *every call in the run*. Depth comes from
  `effort` and surface variety from per-case style directives.

---

## 3. What is verified, and what that does and does not establish

`make silver-gate` runs the Phase 5 tests over prompts, extraction, the
generate→verify→repair→discard loop, the response cache, the budget guard and resume.
`tests/integration/test_silver_pipeline.py` drives the whole pipeline with `ScriptedTeacher`.

**This establishes that the pipeline is correct given a teacher. It establishes nothing
about what a real teacher produces** — not the discard rate, not the repair success rate,
not the quality of the surviving text, not the cost.

---

## 4. Blockers

| Blocker | Detail |
|---|---|
| **Teacher-API credentials** | Spend authorisation. This is the whole blocker; there is no engineering left. |

Run **`make silver-dry-run` before `make silver`.** It processes 20 records and reports
projected cost and quality without writing anything or spending materially.

### What this blocker also blocks

- **Epochs 2–3 of every trained arm's curriculum.** The generator's training schedule
  assumes a Silver tier.
- **The Method A / Method B extractor agreement κ** — the number that makes the
  faithfulness metric defensible rather than merely defined. Method B decomposes with an
  LLM and needs the same credentials.
- Systems **B3, B4 and B5** in the experiment matrix.

---

## 5. Limitations this tier would have

Stated in advance so they are not discovered at review time.

- **Its faithfulness is bounded by the checker, not by truth.** A rewrite survives if the
  checker cannot contradict it. Cue attribution can only sharpen a verdict, never soften
  one (D-076), so an unaligned quantity matching no cue rule stays UNVERIFIABLE rather than
  being promoted to a hallucination — which **understates** the true rate.
- **The ten-point harness cannot see an unaligned quantity here.** It rebuilds claims from
  `target_slots`, which are exactly the values that *did* align, so an invented figure
  produces no slot and is invisible to check 5. Bronze is immune by construction; Silver
  enforces an unverifiable budget at ingestion against the extractor's own rate instead
  (D-057).
- **Two teachers is two teachers.** Balanced assignment removes the worst of single-model
  distillation; it does not make the tier model-agnostic.
- **It inherits Bronze's content**, including the 92% H9 exculpatory-omission rate — a
  rewrite that faithfully paraphrases a one-sided narrative produces a one-sided narrative.
  Silver was never designed to fix H9.
- Everything in [`README.md`](README.md) § Common limitations.

---

## 6. Licence

**CDLA-Sharing-1.0**, on the same reasoning as [`bronze.md`](bronze.md) §7: Silver rewrites
Bronze and carries the same AMLworld identifiers, timestamps and amounts. Should it ever be
built, it ships in the data bundle, never in the Apache-2.0 code bundle.

The **discard log** carries the same content and the same licence.

---

## 7. Intended use and misuse

If built: training data for the generator arms, and the source of the discard-rate finding.

`docs/ETHICS.md` §2 applies in full. One addition specific to this tier: **the discard log
must not be suppressed or summarised away.** It is the honest record of how often a
frontier model instructed not to invent facts invented them, and it is more useful to the
field than the corpus it is a byproduct of.

---

## 8. Reproduction

```bash
make silver-dry-run   # 20 records, projected cost and quality, nothing written
make silver           # the full build. COSTS MONEY.
make silver-resume    # continue an interrupted build from its checkpoint
make silver-gate      # the tests (no network, no credentials needed)
```

Only `make silver-gate` and `make silver-dry-run` are runnable without credentials, and the
dry run still needs them to reach a teacher.
