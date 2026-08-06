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

.PHONY: install-stats
install-stats:  ## Add the pure-CPU statistics stack (scipy/statsmodels; no torch)
	$(UV) sync --group dev --extra stats

.PHONY: install-eval
install-eval:  ## Add the CPU evaluation metrics stack (includes bert-score, so torch)
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
data:  ## [Phase 1] Ingest and normalise raw substrates into data/interim
	$(RUN) python scripts/01_ingest.py
	$(RUN) python scripts/01_ingest.py data=elliptic2

.PHONY: data-amlworld
data-amlworld:  ## [Phase 1] Ingest AMLworld only (override the variant with data.size=...)
	$(RUN) python scripts/01_ingest.py

.PHONY: data-elliptic2
data-elliptic2:  ## [Phase 1] Ingest Elliptic2 only (skips cleanly if access not granted)
	$(RUN) python scripts/01_ingest.py data=elliptic2

.PHONY: cases
cases:  ## [Phase 2] Extract cases, split temporally, audit for leakage
	$(RUN) python scripts/02_build_cases.py

.PHONY: splits
splits: cases  ## [Phase 2] Alias for `cases`: the split manifests are its output

.PHONY: cases-debug
cases-debug:  ## [Phase 2] A 400-case build, for smoke runs
	$(RUN) python scripts/02_build_cases.py cases=debug

.PHONY: sensitivity
sensitivity:  ## [Phase 2] k x n_max sensitivity grid -> artifacts/metrics/sensitivity/
	$(RUN) python scripts/02b_sensitivity.py

.PHONY: audit
audit:  ## [Phase 2] Re-run the leakage audit over the committed split manifest
	$(RUN) python scripts/02c_audit.py

.PHONY: facts
facts:  ## [Phase 3] Extract case_facts records from subgraphs
	$(RUN) python scripts/03_extract_facts.py

.PHONY: facts-elliptic2
facts-elliptic2:  ## [Phase 3] Extract facts for Elliptic2 (skips cleanly without access)
	$(RUN) python scripts/03_extract_facts.py data=elliptic2

.PHONY: facts-gate
facts-gate:  ## [Phase 3] The gate: 1,000-case round trip + the independent oracle
	$(RUN) pytest tests/integration/test_facts_roundtrip.py tests/golden -v

.PHONY: facts-golden
facts-golden:  ## [Phase 3] Regenerate golden fact records. Read the diff before committing.
	$(RUN) python -m tests.golden.test_golden_case_facts --regenerate

.PHONY: bronze
bronze:  ## [Phase 4] Render deterministic template narratives + the ten-point gate
	$(RUN) python scripts/04_build_bronze.py

.PHONY: bronze-elliptic2
bronze-elliptic2:  ## [Phase 4] Bronze for Elliptic2 (skips cleanly without access)
	$(RUN) python scripts/04_build_bronze.py data=elliptic2

.PHONY: bronze-gate
bronze-gate:  ## [Phase 4] The gate: renderer, ten-point harness, corpus integration
	$(RUN) pytest tests/unit/test_bronze_format.py tests/unit/test_bronze_templates.py \
		tests/unit/test_bronze_renderer.py tests/unit/test_corpus_validate.py \
		tests/unit/test_corpus_support.py tests/integration/test_bronze_pipeline.py -v

.PHONY: silver
silver:  ## [Phase 5] Verified LLM rewrites of Bronze. COSTS MONEY: dry-run first.
	$(RUN) python scripts/05_build_silver.py corpus=silver

.PHONY: silver-dry-run
silver-dry-run:  ## [Phase 5] 20 records, projected cost and quality, nothing written
	$(RUN) python scripts/05_build_silver.py corpus=silver corpus.generation.dry_run=true

.PHONY: silver-resume
silver-resume:  ## [Phase 5] Continue an interrupted build from its checkpoint
	$(RUN) python scripts/05_build_silver.py corpus=silver corpus.generation.resume=true

.PHONY: silver-gate
silver-gate:  ## [Phase 5] The gate: prompts, extraction, the loop, cache, budget, resume
	$(RUN) pytest tests/unit/test_silver_prompts.py tests/unit/test_silver_extraction.py \
		tests/unit/test_silver_generate.py tests/unit/test_silver_api_client.py \
		tests/unit/test_silver_quality.py tests/integration/test_silver_pipeline.py -v

.PHONY: gold-sample
gold-sample:  ## [Phase 6] Draw the Gold sample and reserve it test-only in the manifest
	$(RUN) python scripts/06_sample_gold_cases.py corpus=gold

.PHONY: annotate
annotate:  ## [Phase 6] Launch the annotation interface (ANNOTATOR=annotator-01)
	@test -n "$(ANNOTATOR)" || { echo "set ANNOTATOR, e.g. make annotate ANNOTATOR=annotator-01"; exit 1; }
	$(RUN) streamlit run src/g2t_aml/human/annotation_ui.py -- \
		--annotator $(ANNOTATOR) --dataset $${DATASET:-amlworld_hi_small}

.PHONY: calibrate
calibrate:  ## [Phase 6] Annotate the calibration set (ANNOTATOR=annotator-01)
	@test -n "$(ANNOTATOR)" || { echo "set ANNOTATOR, e.g. make calibrate ANNOTATOR=annotator-01"; exit 1; }
	$(RUN) streamlit run src/g2t_aml/human/annotation_ui.py -- \
		--annotator $(ANNOTATOR) --dataset $${DATASET:-amlworld_hi_small} --calibration

.PHONY: gold
gold:  ## [Phase 6] Ingest reviewed annotations -> gold.jsonl + the ten-point gate
	$(RUN) python scripts/06b_ingest_gold.py corpus=gold

.PHONY: guidelines-pdf
guidelines-pdf:  ## [Phase 6] Render the annotation guidelines and recruitment brief to PDF
	$(RUN) python scripts/06c_export_guidelines.py

.PHONY: gold-gate
gold-gate:  ## [Phase 6] The gate: sampling, reservation, panel, graph, validation, ingest
	$(RUN) pytest tests/unit/test_human_sampling.py tests/unit/test_human_reservation.py \
		tests/unit/test_human_factpanel.py tests/unit/test_human_graphview.py \
		tests/unit/test_human_validation.py tests/unit/test_human_store_review.py \
		tests/unit/test_human_agreement.py tests/unit/test_human_calibration.py \
		tests/integration/test_gold_pipeline.py -v

.PHONY: train-encoder
train-encoder:  ## [Phase 7] Train all six encoder arms x three seeds and run the gate
	$(RUN) python scripts/07_train_encoder.py experiment=encoder_sweep

.PHONY: encoder-debug
encoder-debug:  ## [Phase 7] Two arms, one seed, few epochs -- a wiring check, not a result
	$(RUN) python scripts/07_train_encoder.py experiment=encoder_sweep \
		'experiment.arms=[gatv2,mlp]' 'training.seeds=[42]' training.epochs=3 \
		training.n_bootstrap=200 \
		experiment.ablations.positional_encodings=false \
		experiment.ablations.edge_features=false \
		experiment.ablations.loss_function=false

.PHONY: encoder-features
encoder-features:  ## [Phase 7] Build the feature cache from the frozen splits
	$(RUN) python -c "import logging, sys; logging.basicConfig(level=logging.INFO); \
		sys.path.insert(0, 'src'); \
		from g2t_aml.models.encoder.dataset import build_feature_cache as b; \
		b(cases_dir='data/processed/amlworld_hi_small/cases', \
		  realistic_dir='data/processed/amlworld_hi_small/cases/realistic_test', \
		  interim_dir='data/interim/amlworld_hi_small', \
		  splits_dir='schemas/splits/amlworld', \
		  facts_parquet='data/processed/amlworld_hi_small/facts.parquet', \
		  out_dir='data/processed/amlworld_hi_small/encoder/features', \
		  log=logging.getLogger('cache'))"

.PHONY: score-cases
score-cases:  ## [Phase 7] Write model_signal back into every fact record
	$(RUN) python scripts/07b_score_cases.py

.PHONY: encoder-gate
encoder-gate:  ## [Phase 7] The gate: features, arms, losses, metrics, splits, write-back
	$(RUN) pytest tests/unit/test_encoder_features.py tests/unit/test_encoder_arms.py \
		tests/unit/test_encoder_metrics.py tests/unit/test_encoder_dataset.py \
		tests/unit/test_encoder_writeback.py tests/integration/test_encoder_pipeline.py -v

.PHONY: train-generator
train-generator:  ## [Phase 9] Train S1 then A1 -- the treatment and its control, in order
	$(RUN) python scripts/09_train_generator.py experiment=generator_s1
	$(RUN) python scripts/09_train_generator.py experiment=generator_a1

.PHONY: train-s1 train-a1 train-s2 train-b7 train-b8
train-s1:  ## [Phase 9] Priority 1: GAT + F2 + full text (the primary arm)
	$(RUN) python scripts/09_train_generator.py experiment=generator_s1
train-a1:  ## [Phase 9] Priority 2: the shuffled control. Do not skip this to save GPU time
	$(RUN) python scripts/09_train_generator.py experiment=generator_a1
train-s2:  ## [Phase 9] Priority 3: GAT + F2, text_mode=none (the headline arm)
	$(RUN) python scripts/09_train_generator.py experiment=generator_s2
train-b7:  ## [Phase 9] Priority 4: text-only QLoRA, no fusion (the threatening baseline)
	$(RUN) python scripts/09_train_generator.py experiment=generator_b7
train-b8:  ## [Phase 9] Priority 5: G-Retriever-style F1 + full text
	$(RUN) python scripts/09_train_generator.py experiment=generator_b8

.PHONY: generator-debug
generator-debug:  ## [Phase 9] A wiring run on a small slice -- never report a number from it
	$(RUN) python scripts/09_train_generator.py experiment=generator_debug

.PHONY: gate8
gate8:  ## [Phase 9] THE DECISION POINT: compare S1 against A1 and state the verdict
	@test -n "$(S1)" -a -n "$(A1)" || \
		(echo "usage: make gate8 S1=<history_S1.jsonl> A1=<history_A1.jsonl>" && exit 1)
	$(RUN) python scripts/09b_compare_arms.py --treatment "$(S1)" --control "$(A1)" \
		--out artifacts/metrics/generator/gate8.json

.PHONY: generator-gate
generator-gate:  ## [Phase 9] The gate: fusion, loss masking, overfit, Gold hold-out, guard
	$(RUN) pytest tests/unit/test_fusion.py tests/unit/test_generator_harness.py \
		tests/unit/test_generator_guard.py tests/integration/test_generator_pipeline.py -v

.PHONY: eval
eval:  ## [Phase 10] Score every configured system -> artifacts/metrics/eval/<run>/
	$(RUN) python scripts/10_evaluate.py

.PHONY: eval-bronze
eval-bronze:  ## [Phase 10] The Bronze-only run: the self-consistency gate, no model needed
	$(RUN) python scripts/10_evaluate.py eval.surface.bertscore=false

.PHONY: eval-debug
eval-debug:  ## [Phase 10] 200 records, few resamples, no BERTScore -- a wiring check
	$(RUN) python scripts/10_evaluate.py eval.limit=200 eval.surface.bertscore=false \
		eval.stats.bootstrap_samples=1000

.PHONY: eval-gate
eval-gate:  ## [Phase 10] The gate: metrics, both extractors, taxonomy, statistics, end-to-end
	$(RUN) pytest tests/unit/test_eval_layer1.py tests/unit/test_eval_layer2.py \
		tests/unit/test_eval_extraction.py tests/unit/test_eval_taxonomy.py \
		tests/unit/test_eval_statistics.py tests/unit/test_eval_report.py \
		tests/integration/test_eval_end_to_end.py \
		--cov=g2t_aml.eval --cov-report=term-missing --cov-fail-under=85 -v

# ---------------------------------------------------------------------------
# Phase 11 -- the experiment matrix
#
# START WITH `make matrix-plan`. It prints the plan, the resource split and the seed
# policy without touching a GPU or spending an API dollar. `make matrix` refuses to run
# the GPU and API arms unless you pass ALLOW_GPU=1 / ALLOW_API=1, because on this machine
# they cannot succeed and a run that fails for four hours is worse than one that does not
# start (D-068).
# ---------------------------------------------------------------------------

.PHONY: matrix-plan
matrix-plan:  ## [Phase 11] Print the matrix plan: runs, order, resources, seed policy
	uv run python scripts/11_run_matrix.py --dry-run

.PHONY: matrix
matrix:  ## [Phase 11] Run the experiment matrix (ALLOW_GPU=1 ALLOW_API=1 SYSTEMS=B1,B2)
	uv run python scripts/11_run_matrix.py \
		$(if $(SYSTEMS),--systems $(SYSTEMS),) \
		$(if $(ALLOW_GPU),--allow-gpu,) \
		$(if $(ALLOW_API),--allow-api,) \
		$(if $(FORCE),--force,)

.PHONY: matrix-cpu
matrix-cpu:  ## [Phase 11] Run only the arms that need no GPU and no network (B1, B2)
	uv run python scripts/11_run_matrix.py --systems B1,B2

.PHONY: aggregate
aggregate:  ## [Phase 11] Aggregate the matrix: tidy table, CIs, significance, LaTeX, figures
	uv run python scripts/11a_aggregate.py

.PHONY: qualitative
qualitative:  ## [Phase 11] Side-by-side cases, worst cases, and S1-vs-B7 disagreements
	uv run python scripts/11b_qualitative.py

.PHONY: matrix-gate
matrix-gate:  ## [Phase 11] Run the Phase 11 tests
	uv run pytest \
		tests/unit/test_experiment_registry.py \
		tests/unit/test_experiment_runner.py \
		tests/unit/test_experiment_aggregate.py \
		tests/unit/test_experiment_baselines.py \
		tests/unit/test_experiment_figures.py \
		tests/integration/test_matrix_pipeline.py -v

.PHONY: study-build
study-build:  ## [Phase 12] Build the blinded block design, blind key and narrative pool
	$(RUN) python scripts/12_build_study.py

.PHONY: study-rate
study-rate:  ## [Phase 12] Run the rating interface (RATER=rater-01 required)
	@test -n "$(RATER)" || { echo "usage: make study-rate RATER=rater-01"; exit 2; }
	$(RUN) streamlit run src/g2t_aml/human/study_ui.py -- \
		--rater $(RATER) \
		--design artifacts/human_study/design.json \
		--narratives artifacts/human_study/narratives.jsonl \
		--responses artifacts/human_study/responses \
		--processed data/processed/amlworld_hi_small \
		--interim data/interim/amlworld_hi_small

.PHONY: study-analyse
study-analyse:  ## [Phase 12] Unblind and analyse: agreement, ranks, times, edits, correlation
	$(RUN) python scripts/12b_analyse_study.py

.PHONY: study-release
study-release:  ## [Phase 12] Prepare the anonymised response data for public release
	$(RUN) python scripts/12c_release_study.py

.PHONY: study-gate
study-gate:  ## [Phase 12] The gate: design, blinding, timer, edit capture, statistics, release
	$(RUN) pytest tests/unit/test_study_design.py tests/unit/test_study_ui.py \
		tests/unit/test_study_analysis.py tests/unit/test_study_release.py -v

# --------------------------------------------------------------- phase 13 ---

.PHONY: benchmark
benchmark:  ## [Phase 13] Measure efficiency: latency, VRAM, throughput, cost, deployability
	$(RUN) python scripts/13_benchmark.py

.PHONY: benchmark-quick
benchmark-quick:  ## [Phase 13] Wiring check: 5 warm-up / 10 measured, ~30 s
	$(RUN) python scripts/13_benchmark.py --quick

.PHONY: benchmark-gate
benchmark-gate:  ## [Phase 13] The gate: protocol, percentiles, cost model, table rendering
	$(RUN) pytest tests/unit/test_eval_efficiency.py -v

# --------------------------------------------------------------- phase 14 ---
# START WITH `make quickstart`. It reproduces one published result from a clean clone in
# about two seconds, with no data download, no GPU and no network.
#
# The two release bundles carry DIFFERENT LICENCES and must never be merged: the corpus is
# CDLA-Sharing-1.0 Enhanced Data, everything else is Apache-2.0 Results (D-098).
# ---------------------------------------------------------------------------

.PHONY: quickstart
quickstart:  ## [Phase 14] Reproduce one published result from a clean clone, in ~2 s
	$(RUN) python scripts/14_quickstart.py

.PHONY: quickstart-golden
quickstart-golden:  ## [Phase 14] Regenerate the quickstart golden file. Read the diff.
	$(RUN) python scripts/14_quickstart.py --regenerate-golden

.PHONY: verify-release
verify-release:  ## [Phase 14] The nine release checks against a clean `git archive` export
	$(RUN) python scripts/14_verify_release.py \
		--json artifacts/metrics/release_verification.json

.PHONY: verify-release-docker
verify-release-docker:  ## [Phase 14] The same, inside a freshly built CPU container
	$(RUN) python scripts/14_verify_release.py --in-docker

.PHONY: secret-scan
secret-scan:  ## [Phase 14] gitleaks over the FULL git history, not just the working tree
	@command -v gitleaks >/dev/null || { \
		echo "gitleaks not found. Install it:"; \
		echo "  https://github.com/gitleaks/gitleaks/releases"; exit 1; }
	gitleaks detect --source . --config .gitleaks.toml --redact --no-banner

.PHONY: docker-cpu docker-gpu
docker-cpu:  ## [Phase 14] Build the CPU image (runs the quickstart at build time)
	docker build -f docker/Dockerfile.cpu -t g2t-aml:cpu .
docker-gpu:  ## [Phase 14] Build the GPU image (torch 2.4.0+cu121, PyG 2.6.1)
	docker build -f docker/Dockerfile.gpu -t g2t-aml:gpu .

.PHONY: release-plan
release-plan:  ## [Phase 14] Show what each licence bundle would contain. Writes nothing.
	$(RUN) python scripts/14_package_release.py --dry-run

.PHONY: release
release:  ## [Phase 14] Package the two licence-separated bundles into dist/
	$(RUN) python scripts/14_package_release.py --out dist --version $(VERSION)

VERSION ?= v0.1.0

.PHONY: elliptic2-reconstruct
elliptic2-reconstruct:  ## [Phase 14] Rebuild Elliptic2 artifacts from YOUR licensed copy
	$(RUN) python scripts/14_reconstruct_elliptic2.py

.PHONY: release-gate
release-gate:  ## [Phase 14] The gate: quickstart, packaging, reconstruction, hygiene
	$(RUN) pytest tests/unit/test_release_packaging.py \
		tests/integration/test_quickstart.py -v

# ----------------------------------------------------------------- hygiene ---

.PHONY: clean
clean:  ## Remove caches and build products (never touches data/ or artifacts/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info src/*.egg-info
	rm -rf .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@echo "clean OK (data/ and artifacts/ left untouched -- see invariant 6)"
