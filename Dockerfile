# Graph2Text AML — CPU image.
#
# Covers phases 1-6 and 10 (ingestion, splits, fact layer, Bronze/Silver corpus,
# evaluation), which are CPU-only by design. For the GPU phases build FROM
# nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 instead and run the install-gpu step
# below; the PyG companion wheels are compiled against torch 2.4.0 + cu121 exactly.

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    G2T_AML_ROOT=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /usr/local/bin/uv

WORKDIR /workspace

# Dependency layer first, so source edits do not invalidate the install.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --group dev --no-install-project

COPY . .
RUN uv sync --frozen --group dev

# data/ and artifacts/ are gitignored and absent from the build context; recreate them
# so a container run has somewhere to write. Mount real volumes over them in practice.
RUN mkdir -p data/raw data/interim data/processed \
             artifacts/checkpoints artifacts/metrics artifacts/figures artifacts/runs

# GPU image only — uncomment on a CUDA base:
# RUN uv sync --frozen --group dev --extra eval --extra graph --extra llm \
#     && uv pip install --find-links https://data.pyg.org/whl/torch-2.4.0+cu121.html \
#        torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3

CMD ["make", "smoke"]
