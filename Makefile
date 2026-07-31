.DEFAULT_GOAL := help
SHELL := /bin/bash
UV ?= uv
RUN := $(UV) run

# Phases that have not landed yet print this and exit 0. A placeholder that silently
# succeeds is worse than one that says why it did nothing.
define not_implemented
	@echo ""
	@echo "  [$(1)] is not implemented until Phase $(2)."
	@echo "  $(3)"
	@echo ""
	@exit 1
endef

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------ setup ---

.PHONY: install
install:  ## Install the light (CPU-only) environment + dev tools
	$(UV) sync --group dev
	$(RUN) pre-commit install || true

.PHONY: install-eval
install-eval:  ## Add the CPU evaluation metrics stack
	$(UV) sync --group dev --extra eval

.PHONY: install-gpu
install-gpu: install-pyg  ## Add graph + llm extras (needs CUDA 12.1, see README)

.PHONY: install-gpu-base
install-gpu-base:
	$(UV) sync --group dev --extra eval --extra graph --extra llm --extra human

# torch-scatter/sparse/cluster are not in uv.lock: their PyPI sdists import torch at
# build time and cannot be resolved. These are the prebuilt wheels for torch 2.4.0+cu121.
PYG_WHEELS := https://data.pyg.org/whl/torch-2.4.0+cu121.html

.PHONY: install-pyg
install-pyg: install-gpu-base  ## Install the PyG companion wheels (torch 2.4.0+cu121)
	$(UV) pip install --find-links $(PYG_WHEELS) \
		torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3

.PHONY: lock
lock:  ## Re-resolve uv.lock
	$(UV) lock

# ------------------------------------------------------------------ checks ---

.PHONY: lint
lint:  ## ruff check + format check
	$(RUN) ruff check src tests scripts
	$(RUN) ruff format --check src tests scripts

.PHONY: format
format:  ## Apply ruff formatting and autofixes
	$(RUN) ruff check --fix src tests scripts
	$(RUN) ruff format src tests scripts

.PHONY: typecheck
typecheck:  ## mypy --strict over facts/ and eval/ only
	$(RUN) mypy

.PHONY: test
test:  ## Unit + integration tests with coverage
	$(RUN) pytest

.PHONY: test-fast
test-fast:  ## Unit tests only, no slow/gpu/network markers
	$(RUN) pytest tests/unit -m "not slow and not gpu and not network"

.PHONY: smoke
smoke: lint typecheck test  ## The CI gate: lint + typecheck + tests + e2e smoke run
	$(RUN) python scripts/smoke.py
	@echo "smoke OK"

# ---------------------------------------------------------------- pipeline ---
# Filled in by later phases. Order below is the dependency order.

.PHONY: data
data:  ## [Phase 1] Ingest and normalise raw substrates
	$(call not_implemented,make data,1,Will ingest IBM AMLworld and Elliptic2 into data/interim.)

.PHONY: splits
splits:  ## [Phase 2] Build the frozen temporal split manifests
	$(call not_implemented,make splits,2,Writes committed ID lists + content hashes to schemas/splits/.)

.PHONY: facts
facts:  ## [Phase 3] Extract case_facts records from subgraphs
	$(call not_implemented,make facts,3,The measurement instrument. See invariant 1 before touching it.)

.PHONY: bronze
bronze:  ## [Phase 4] Render deterministic template narratives
	$(call not_implemented,make bronze,4,Template-rendered narratives, faithful by construction.)

.PHONY: silver
silver:  ## [Phase 5] Verified LLM rewrites of the Bronze tier
	$(call not_implemented,make silver,5,Rewrites gated by the same verifier used at eval time.)

.PHONY: train-encoder
train-encoder:  ## [Phase 7] Train the GAT graph encoder
	$(call not_implemented,make train-encoder,7,Needs the graph extra and a CUDA device.)

.PHONY: train-generator
train-generator:  ## [Phase 8/9] QLoRA-finetune the generator with graph fusion
	$(call not_implemented,make train-generator,9,Needs the llm extra and >=24 GB VRAM.)

.PHONY: eval
eval:  ## [Phase 10] Faithfulness + surface metrics
	$(call not_implemented,make eval,10,Runs the fact verifier in reverse over generated narratives.)

.PHONY: matrix
matrix:  ## [Phase 12] Full experiment matrix across substrates and ablations
	$(call not_implemented,make matrix,12,Hydra multirun over the ablation grid.)

.PHONY: release
release:  ## [Phase 14] Package artifacts and figures for submission
	$(call not_implemented,make release,14,Bundles metrics, figures and run contexts for the paper.)

# ----------------------------------------------------------------- hygiene ---

.PHONY: clean
clean:  ## Remove caches and build products (never touches data/ or artifacts/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info src/*.egg-info
	rm -rf .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@echo "clean OK (data/ and artifacts/ left untouched -- see invariant 6)"
