# Model cards

One card per released model, plus an explicit account of what is **not** released and why.

Written Phase 14, 2026-08-06, for release `v0.1.0`.

---

## What is released

| Model | Card | Artifact | Status |
|---|---|---|---|
| **GATv2 case encoder** (primary) | [`gatv2-case-encoder.md`](gatv2-case-encoder.md) | 3 checkpoints, 2.53 MB each | **released** |
| Encoder comparison arms (GIN, GraphSAGE, GCN, graph transformer, MLP control) | [`encoder-arms.md`](encoder-arms.md) | 15 checkpoints | **released** |
| Encoder ablation arms (no-PE, no-edge-features, weighted-BCE) | [`encoder-arms.md`](encoder-arms.md) | 9 checkpoints | **released** |

All 27 checkpoints total ~78 MB. Each records its own `training_config`, and resume
refuses a mismatch (D-067) — a checkpoint carries evidence of how long it trained, because
a two-epoch wiring check otherwise resumes exactly as happily as a converged one and its
number lands in a results table looking like every other row.

---

## What is NOT released, and why

**These are the artifacts the project was designed around. None of them exists.**

| Artifact | Status | Reason |
|---|---|---|
| **LoRA adapters** (S1, S2, A1, B7, B8, A2, A3, A4, A6) | **DOES NOT EXIST** | No generator arm has been trained. Llama-3.1-8B at nf4 is 4.5–5.6 GB of weights before a single activation; the development machine has a 4 GB RTX 2050 and 7 GB of system RAM, which also closes CPU offload. D-068. |
| **Fusion projector** (the technical contribution) | **DOES NOT EXIST as a trained artifact** | Same blocker. The `PrefixFusion` module, its three projectors and the `ShuffledGraphFusion` control are built, tested on CPU against a stub backbone, and **never trained**. |
| **Gate 8 outcome** — does the fusion layer beat its own shuffled control? | **OPEN, not answered** | Neither S1 nor A1 has run. |

There is no adapter to publish to the HuggingFace Hub and no fusion checkpoint to write a
card for. Publishing a card for an untrained module would describe a file that does not
exist.

**Do not read the encoder cards as evidence about the method.** The encoder is a component
the architecture consumes, and its numbers say nothing about whether a language model can
read a soft graph token. See `RESULTS.md` §2 and `docs/ETHICS.md` §3.1.

---

## Common to every card here

**Licence.** Model weights are **Results** under CDLA-Sharing-1.0 §3.5 — the share-alike
obligation on AMLworld data does not reach them — and are released under **Apache-2.0**,
the same licence as the code. This is a different bundle from the corpus, which is
CDLA-Sharing-1.0. See `README.md` § Licensing.

**Provenance.** Every checkpoint was produced by a run that wrote `run_context.json`: git
SHA, resolved Hydra config, data manifest hash, all seeds, library versions (invariant 5).

**Misuse.** The prohibitions in `docs/ETHICS.md` §2 apply to every model in this
directory without exception, and each card restates the ones specific to it.
