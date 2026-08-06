# Container images

Two images. **Use the CPU one unless you are training something.**

| | `Dockerfile.cpu` | `Dockerfile.gpu` |
|---|---|---|
| Base | `python:3.11.14-slim-bookworm` (digest-pinned) | `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04` (digest-pinned) |
| Size | ~1 GB | ~8 GB |
| Covers | Phases 1–6, 10 — ingestion, splits, fact layer, corpus, evaluation | Everything, plus phases 7–9, 11–13 |
| torch | **none** | 2.4.0+cu121 |
| Needs a GPU to build | no | **no** |
| Needs a GPU to run | no | for the training targets |
| Verified in CI | **yes, on a schedule** | build only |

---

## CPU — the default

```bash
docker build -f docker/Dockerfile.cpu -t g2t-aml:cpu .
docker run --rm g2t-aml:cpu                      # the quickstart, ~2 s
docker run --rm g2t-aml:cpu make smoke           # lint + typecheck + tests + smoke run
```

To run the real pipeline, mount the data and artifact trees — they are gitignored and so
absent from the build context:

```bash
docker run --rm \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/artifacts:/workspace/artifacts" \
  g2t-aml:cpu make bronze
```

**The CPU image runs its own quickstart at build time**, so a build that succeeds has
already reproduced one published result. An image that builds and then fails its own
quickstart is worse than a build failure, because it ships.

## GPU

```bash
docker build -f docker/Dockerfile.gpu -t g2t-aml:gpu .
docker run --rm --gpus all g2t-aml:gpu \
    python -c "import torch; print(torch.cuda.is_available())"

docker run --rm --gpus all \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/artifacts:/workspace/artifacts" \
  g2t-aml:gpu make train-encoder
```

Requires the NVIDIA Container Toolkit on the host.

**Hardware.** The encoder (phase 7) needs under 4 GB and is comfortable anywhere. **QLoRA
finetuning of Llama-3.1-8B (phase 9) needs 24 GB VRAM minimum** and is comfortable at
48 GB. **No generator arm has ever been trained** — read D-068 and `RESULTS.md` §2 before
scheduling compute against this image.

---

## Reproducibility

Both images pin every layer that can be pinned:

- **Base image by digest**, not by tag. Upstream rebuilds `python:3.11-slim-bookworm`
  regularly; an image that silently changes its libc under a locked dependency set is
  exactly what a pinned build is for.
- **uv 0.10.2**, copied from `ghcr.io/astral-sh/uv` at that exact version.
- **Every Python dependency from `uv.lock`, installed `--frozen`.** The build fails rather
  than re-resolving.
- **The three PyG companion wheels by exact version** from the `torch-2.4.0+cu121` index.
  They are deliberately *not* in `uv.lock` — their sdists import torch at build time, so
  they cannot be resolved into a lockfile, and a source build would compile against
  whatever CUDA the build host happens to have (D-007).

**`.dockerignore` is load-bearing, not tidiness.** `.gitignore` does not apply to
`docker build`, so without it `COPY . .` sweeps in `data/` (1.5 GB), `.venv/` (6.3 GB),
`artifacts/` and `wandb/` — an ~8 GB build context that bloats the image and invalidates
the layer cache on every run output. With it the context is **6.6 MB**. Keep it in step
with `.gitignore`, and note that `tests/fixtures/` and `tests/golden/` must stay in the
context because the build-time quickstart reads them.

What is still not pinned, and cannot be: Debian and Ubuntu `apt` package versions, and the
contents of the PyG wheel index. Both are recorded in `run_context.json` at run time
rather than pretended away.

**A container does not make GPU results bit-reproducible.** Kernel selection, atomics
ordering and TF32 all depend on the physical card. See `docs/REPRODUCTION.md` §6 for the
tolerance policy.

---

## Verification

```bash
uv run python scripts/14_verify_release.py --in-docker
```

Builds the CPU image and runs the nine release checks inside it against a clean `git
archive` export — no untracked files, no populated `data/`, no `.venv`. This is what runs
on a schedule in `.github/workflows/verify-release.yml`.
