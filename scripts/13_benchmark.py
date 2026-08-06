#!/usr/bin/env python
"""Phase 13: measure the efficiency and deployability of every system in the matrix.

Runs the protocol declared in :mod:`g2t_aml.eval.efficiency` -- warm up 20, measure 100,
report the distribution rather than the mean, bin by case size, and time the whole path
rather than the decoder -- against every system that can be measured on this machine, and
records every system that cannot with the reason it cannot.

**What is measurable here and what is not.** Phase 11 has not run: no generator arm has
been trained, this machine is a 4 GB RTX 2050 with 7 GB of system RAM, and Llama-3.1-8B at
nf4 does not fit on it before a single activation (D-068). So the 8B-backbone systems
(B6-B8, S1, S2, A1-A6) have no checkpoint to load and no card to load it onto, and the API
baselines (B3-B5) have no credentials. Those thirteen rows are written as absences with
their blockers, exactly as Phase 11 writes its non-runs.

What *is* measurable is not nothing, and it is not a proxy. B1 and B2 are complete systems
that run end to end on this hardware, and the components they share with every other row --
case extraction, the fact layer, serialisation, the graph encoder, and the guard's
verification pass -- are the same code the 8B arms call. Measuring them here means that on
the day a card arrives, the only unmeasured stage is the decoder.

The stages timed per narrative are the real deployment path::

    alert names an account
      -> case extraction   cut the subgraph from the indexed 5M-edge graph
      -> fact extraction   build the checkable record
      -> serialisation     render it into the model's context
      -> encoding          GATv2 forward + pooled tokens        [B2, and every S/A arm]
      -> generation        template render, or decoder          [B1, B2 measured]
      -> guard             verify N candidates, select, repair  [measured as a component]

Graph loading is timed too, but as cold start rather than per narrative: the index is built
once per process and serves every case, so charging it to each narrative would multiply a
one-off by the corpus size.

Usage:
    uv run python scripts/13_benchmark.py
    uv run python scripts/13_benchmark.py --n-measured 100 --n-warmup 20
    uv run python scripts/13_benchmark.py --quick          # 10/5, for a wiring check
    uv run python scripts/13_benchmark.py --no-encoder     # skip B2 and the GPU path
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from g2t_aml.data.canonical import CanonicalGraph
from g2t_aml.data.case_extraction import (
    ExtractionParams,
    GraphIndex,
    TimeWindow,
    cut_case,
    materialise_cut,
)
from g2t_aml.eval.efficiency import (
    DEFAULT_N_MEASURED,
    DEFAULT_N_WARMUP,
    DEFAULT_NODE_BINS,
    INTERACTIVE_BATCH,
    THROUGHPUT_BATCH,
    BenchmarkSample,
    CostAssumptions,
    DeploymentProfile,
    EfficiencyTable,
    EndToEndTimer,
    LatencySummary,
    MemoryProfile,
    ModelFootprint,
    NodeBin,
    Stage,
    SystemEfficiency,
    api_cost_per_1000,
    capture_hardware,
    count_parameters,
    directory_size_bytes,
    local_cost_per_1000,
    measure_cold_start,
    measure_peak_memory,
    run_benchmark,
    summarise_by_node_bin,
)
from g2t_aml.experiments.registry import Executor, Resource, all_systems
from g2t_aml.facts.extractor import extract_facts
from g2t_aml.facts.serialiser import serialise_facts
from g2t_aml.utils.io import read_json, read_jsonl, write_json, write_jsonl
from g2t_aml.utils.logging import configure_logging, get_logger
from g2t_aml.utils.run_context import RunContext

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = "amlworld_hi_small"

log = get_logger("benchmark")


# ------------------------------------------------------------------ the cost model ---

# EVERY FIGURE BELOW IS A DECLARED ASSUMPTION, NOT A MEASUREMENT. They are list prices a
# reader substitutes their own numbers into; see DECISIONS.md D-093 for the model and its
# limits. Two of them matter more than the rest and are the ones a sceptical reader should
# attack first: `utilisation`, because assuming a dedicated box runs flat out divides the
# hourly cost by roughly four, and `depreciation_years`, because a longer life makes any
# capital purchase look better.

#: The configuration this project's own arms are sized for: one 24 GB professional card in
#: a modest host. 24 GB is the smallest memory that trains Llama-3.1-8B under QLoRA at the
#: Phase 9 sequence length without gradient-checkpointing gymnastics, which is why it is
#: the reference rather than a larger card.
SERVING_ASSUMPTIONS = CostAssumptions(
    hardware_capital_usd=9000.0,
    depreciation_years=3.0,
    utilisation=0.5,
    power_draw_w=550.0,
    pue=1.5,
    electricity_usd_per_kwh=0.12,
    source=(
        "List prices, 2026-08. Single 24 GB professional GPU (approx. USD 6,000) in a "
        "2-socket host (approx. USD 3,000); 3-year straight-line depreciation, the common "
        "convention for GPU compute; 50% utilisation, because an AML batch workload does "
        "not run flat out; US commercial electricity at USD 0.12/kWh; PUE 1.5, a common "
        "enterprise figure (a hyperscale facility is nearer 1.1, an older server room is "
        "worse). No figure here is a claim about what any institution actually pays."
    ),
)

#: The CPU-only configuration B1 and B2 need. B2 runs the graph encoder, which is 1.2M
#: parameters and runs on a CPU perfectly well -- the accelerator buys latency, not
#: feasibility.
CPU_ASSUMPTIONS = CostAssumptions(
    hardware_capital_usd=2500.0,
    depreciation_years=4.0,
    utilisation=0.5,
    power_draw_w=150.0,
    pue=1.5,
    electricity_usd_per_kwh=0.12,
    source=(
        "List prices, 2026-08. Commodity 2-socket server, no accelerator, 4-year "
        "straight-line depreciation (longer than the GPU box: a CPU server is not "
        "obsoleted by the next generation the same way)."
    ),
)

#: Frontier-API pricing, per million tokens. Recorded with its date because a stale price
#: is a desk-reject at this venue and a silently stale one is worse.
API_ASSUMPTIONS = CostAssumptions(
    api_input_usd_per_mtok=15.0,
    api_output_usd_per_mtok=75.0,
    source=(
        "Published list pricing as of 2026-08 for the frontier tier named in the registry. "
        "Prices move; substitute current figures before submission. Batch and cached-input "
        "discounts are not applied, and would reduce these figures."
    ),
)

#: Token counts for the API cost estimate. Measured from the Bronze corpus's serialised
#: records and narrative lengths, not guessed -- see `_api_token_estimate`.
_API_CALLS_PER_NARRATIVE = {
    Executor.API_ZERO_SHOT: 1.0,
    Executor.API_FEW_SHOT: 1.0,
    # B5 generates, self-verifies and repairs, and is deliberately given more inference
    # compute than any of our arms (D-084). Charging it for one call would price it as
    # something it is not. Three is the mean call count its own trace declares.
    Executor.API_AGENTIC: 3.0,
}


# ------------------------------------------------------------------ deployability ---

_ON_PREM_EXECUTORS = frozenset(
    {
        Executor.TEMPLATE,
        Executor.CLASSIFIER_TEMPLATE,
        Executor.LOCAL_ZERO_SHOT,
        Executor.TRAINED_GENERATOR,
    }
)

_API_REGULATORY_NOTE = (
    "Sending customer transaction records to a third-party endpoint puts them outside the "
    "institution's direct control. Institutions weigh this against GLBA's Safeguards Rule "
    "(16 CFR Part 314) in the US, the GDPR's Chapter V restrictions on transfers outside "
    "the EEA, and internal data-governance and vendor-risk policy. Whether a given "
    "arrangement is permissible depends on the jurisdiction, the contract and the "
    "institution's own controls, and is a question for its counsel -- not one this paper "
    "answers. What is factual is that the data leaves the perimeter."
)

_LOCAL_REGULATORY_NOTE = (
    "No customer data leaves the institution's network at inference time, so the "
    "cross-border-transfer and third-party-processor questions do not arise. The "
    "remaining data-governance obligations -- retention, access control, model "
    "documentation -- are the institution's ordinary internal ones."
)


def _deployment_for(spec: Any) -> DeploymentProfile:
    """Assemble the deployability assessment for one system.

    Args:
        spec: The registry spec.

    Returns:
        The profile. On-premise is decided by the executor and nothing else: it is a
        property of where the computation happens, not of how well it performs.
    """
    on_prem = spec.executor in _ON_PREM_EXECUTORS
    if not on_prem:
        return DeploymentProfile(
            on_premise=False,
            data_leaves_perimeter=(
                "The serialised fact record for every alert: account identifiers, "
                "counterparty counts, transaction amounts, currencies and timestamps."
            ),
            min_viable_hardware="None. A network egress path and a vendor contract.",
            recommended_hardware="n/a -- the compute is the vendor's.",
            regulatory_context=_API_REGULATORY_NOTE,
            notes=(
                "Quality aside, this row is unavailable to an institution that cannot "
                "export transaction data, which is the constraint the paper's deployment "
                "argument turns on."
            ),
        )

    if spec.base_model is None:
        # The template systems. B2 additionally runs a 1.2M-parameter encoder.
        return DeploymentProfile(
            on_premise=True,
            data_leaves_perimeter="Nothing.",
            min_viable_hardware=(
                "A commodity CPU server. No accelerator. Measured on the recorded host "
                "with no GPU involvement in the template path."
            ),
            recommended_hardware="Any 8-core server with 16 GB RAM.",
            regulatory_context=_LOCAL_REGULATORY_NOTE,
            notes="Runs anywhere the fact layer runs, which is anywhere Python runs.",
        )

    # The 8B arms. The minimum is a memory question and it is answerable from the model's
    # own arithmetic, which is stated rather than measured and labelled as such below.
    return DeploymentProfile(
        on_premise=True,
        data_leaves_perimeter="Nothing.",
        min_viable_hardware=(
            "One 16 GB accelerator for inference at nf4 (weights approx. 5.6 GB plus KV "
            "cache and activations); one 24 GB accelerator to train under QLoRA at the "
            "Phase 9 sequence length. NOT MEASURED -- no such card was available."
        ),
        recommended_hardware=(
            "One 24 GB professional card for a mid-size institution's volume; see "
            "docs/deployability.md for the sizing at 10,000 alerts/month."
        ),
        regulatory_context=_LOCAL_REGULATORY_NOTE,
        notes=(
            "The base model is an open-weights download the institution hosts itself; "
            "what this project adds on top is the adapter, the encoder and the projector."
        ),
    )


# ------------------------------------------------------------------ the pipeline ---


class Pipeline:
    """The measurable end-to-end path, assembled once and driven per case.

    Holds the graph index, the case manifest, the vocabulary and (optionally) the encoder,
    because every one of those is a per-process cost that the cold-start figure carries and
    the per-narrative figure must not.
    """

    def __init__(self, *, root: Path, with_encoder: bool, device: str) -> None:
        """Build the pipeline and time the loads that make up cold start.

        Args:
            root: Repository root.
            with_encoder: Whether to load the Phase 7 GATv2 checkpoint.
            device: Torch device string for the encoder.

        Raises:
            FileNotFoundError: If the interim graph or the case manifest is absent.
        """
        self.root = root
        self.device_str = device
        interim = root / "data" / "interim" / DATASET
        processed = root / "data" / "processed" / DATASET

        started = time.perf_counter()
        self.graph = CanonicalGraph.load(interim)
        self.graph_load_s = time.perf_counter() - started
        log.info(
            "graph loaded: %d nodes, %d edges in %.2f s",
            self.graph.num_nodes,
            self.graph.num_edges,
            self.graph_load_s,
        )

        started = time.perf_counter()
        self.index = GraphIndex(self.graph)
        self.index_build_s = time.perf_counter() - started
        log.info("traversal index built in %.2f s", self.index_build_s)

        self.cases = {
            str(row["case_id"]): row
            for row in read_jsonl(processed / "cases" / "cases.jsonl")
            if isinstance(row, dict)
        }
        manifest = read_json(processed / "cases" / "cases_manifest.json")
        params = manifest["extraction_params"]
        self.params = ExtractionParams(
            k_hops=int(params["k_hops"]),
            n_max=int(params["n_max"]),
            prune_rule=str(params["prune_rule"]),
            preserve_laundering_paths=bool(params["preserve_laundering_paths"]),
            seed=int(params["seed"]),
            max_neighbours_per_node=int(params["max_neighbours_per_node"]),
        )

        # Bronze's renderer and its vocabulary, loaded once. `render_bronze` would load the
        # vocabulary per call otherwise, which would put a file read into every measured
        # narrative and describe the disk rather than the renderer.
        from g2t_aml.corpus.bronze.renderer import render_bronze
        from g2t_aml.facts.vocab import load_vocabulary

        self._render = render_bronze
        self.vocabulary = load_vocabulary()

        self.encoder: Any = None
        self.feature_space: Any = None
        self.encoder_load_s: float | None = None
        self.encoder_params: tuple[int, int] | None = None
        self.encoder_checkpoint: Path | None = None
        if with_encoder:
            self._load_encoder(root)

    def _load_encoder(self, root: Path) -> None:
        """Load the Phase 7 GATv2 checkpoint onto the configured device.

        Args:
            root: Repository root.
        """
        import torch
        from omegaconf import OmegaConf

        from g2t_aml.models.encoder.features import FeatureSpace
        from g2t_aml.models.encoder.registry import build_encoder

        checkpoint = root / "artifacts" / "checkpoints" / "encoder" / "gatv2" / "gatv2_seed42.pt"
        if not checkpoint.is_file():
            log.warning("no encoder checkpoint at %s; B2 will be recorded as blocked", checkpoint)
            return

        started = time.perf_counter()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        space = FeatureSpace.from_dict(payload["feature_space"])
        model = build_encoder(OmegaConf.create(payload["encoder_config"]), space)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        model = model.to(self.device_str)
        if self.device_str.startswith("cuda"):
            torch.cuda.synchronize()
        self.encoder_load_s = time.perf_counter() - started

        self.encoder = model
        self.feature_space = space
        self.encoder_checkpoint = checkpoint
        self.encoder_params = count_parameters(model)
        log.info(
            "encoder loaded on %s in %.2f s (%d params)",
            self.device_str,
            self.encoder_load_s,
            self.encoder_params[0],
        )

    @property
    def cold_start_s(self) -> float:
        """Return the total process-start-to-ready time.

        Returns:
            Graph load plus index build plus, where applicable, encoder load.
        """
        return self.graph_load_s + self.index_build_s + (self.encoder_load_s or 0.0)

    def case_ids(self, split: str = "test", *, seed: int | None = None) -> list[str]:
        """Return the case ids in a split.

        **Shuffled under a fixed seed when one is given, not taken in manifest order.**
        The manifest is ordered by extraction, which correlates with case size, so the
        first hundred entries are not a sample of the corpus -- they are a sample of its
        smallest cases, and a p95 measured over them describes a workload nobody has. The
        seed is recorded in the run context, so the draw is reproducible.

        Args:
            split: Which frozen split to draw from.
            seed: Shuffle seed. None keeps manifest order, which is what the per-bin draw
                wants because it is stratifying explicitly.

        Returns:
            Case ids present both in the split manifest and in the case store.
        """
        manifest = self.root / "schemas" / "splits" / "amlworld" / f"{split}.txt"
        ids = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines()]
        present = [cid for cid in ids if cid and cid in self.cases]
        if seed is not None:
            random.Random(seed).shuffle(present)
        return present

    def case_ids_by_bin(
        self, case_ids: Sequence[str], bins: Sequence[NodeBin] = DEFAULT_NODE_BINS
    ) -> dict[str, list[str]]:
        """Group case ids by the size band their node count falls in.

        Read from the case manifest rather than by extracting each case, so the grouping
        costs nothing and the benchmark measures only what it means to measure.

        Args:
            case_ids: The candidate cases.
            bins: The size bands.

        Returns:
            Bin label to case ids. A band this corpus does not populate maps to an empty
            list, which the size table reports as an empty band rather than omitting.
        """
        grouped: dict[str, list[str]] = {b.label: [] for b in bins}
        for cid in case_ids:
            n_nodes = int(self.cases[cid].get("n_nodes", 0))
            for node_bin in bins:
                if node_bin.contains(n_nodes):
                    grouped[node_bin.label].append(cid)
                    break
        return grouped

    def window(self, row: dict[str, Any]) -> TimeWindow:
        """Rebuild a case's time window from its manifest row.

        Args:
            row: The case manifest row.

        Returns:
            The window the case was cut with.
        """
        return TimeWindow(
            start=datetime.fromisoformat(str(row["window_start"])),
            end=datetime.fromisoformat(str(row["window_end"])),
        )

    def measure(
        self,
        case_id: str,
        *,
        with_encoder: bool = False,
        with_guard: bool = False,
        n_guard_candidates: int = 4,
    ) -> BenchmarkSample:
        """Run one narrative end to end and return its timing.

        This is the function the protocol calls a hundred times. Every stage is timed
        inside it, and the total is the sum of the per-narrative stages -- not a separate
        clock, so the breakdown always adds up to the headline.

        Args:
            case_id: Which case.
            with_encoder: Run the graph encoder forward pass (the B2 path, and the stage
                every S/A arm pays).
            with_guard: Verify candidates against the fact record and select.
            n_guard_candidates: How many candidates the guard scores. Four is what the
                Phase 9 guard samples.

        Returns:
            The measured sample.
        """
        row = self.cases[case_id]
        timer = EndToEndTimer()

        with timer.stage(Stage.CASE_EXTRACTION):
            cut = cut_case(
                self.graph, str(row["seed_node"]), self.window(row), self.params, index=self.index
            )
            case = materialise_cut(cut, self.index)

        with timer.stage(Stage.FACT_EXTRACTION):
            facts = extract_facts(case)

        with timer.stage(Stage.SERIALISATION):
            # The value is discarded: what is being measured is the cost of producing the
            # model's textual context, which every text-mode arm pays whether or not this
            # particular benchmark consumes it.
            serialise_facts(facts, style="compact")

        if with_encoder and self.encoder is not None:
            self._time_encoding(timer, case, str(row["seed_node"]))

        with timer.stage(Stage.GENERATION):
            narrative = self._render(facts, vocabulary=self.vocabulary)

        if with_guard:
            self._time_guard(timer, case_id, narrative, facts, n_guard_candidates)

        return BenchmarkSample(
            case_id=case_id,
            n_nodes=case.num_nodes,
            n_edges=case.num_edges,
            seconds=timer.total(),
            stage_seconds=timer.stages,
            n_output_tokens=len(narrative.text.split()),
        )

    def _time_encoding(self, timer: EndToEndTimer, case: Any, seed_node: str) -> None:
        """Time featurisation and the encoder forward pass.

        CUDA is synchronised before the clock stops. Without that the measurement is of
        how fast Python can queue kernels, which on a small graph is most of the apparent
        latency and is not a property of the model.

        Args:
            timer: The timer to attribute to.
            case: The materialised case.
            seed_node: The account the case was built around.
        """
        import torch
        from torch_geometric.data import Batch

        from g2t_aml.models.encoder.features import build_case_data

        started = time.perf_counter()
        data = build_case_data(case, self.feature_space, seed_node=seed_node)
        batch = Batch.from_data_list([data]).to(self.device_str)
        with torch.no_grad():
            self.encoder(batch)
        if self.device_str.startswith("cuda"):
            torch.cuda.synchronize()
        timer.record(Stage.ENCODING, time.perf_counter() - started)

    def _time_guard(
        self, timer: EndToEndTimer, case_id: str, bronze: Any, facts: Any, n_candidates: int
    ) -> None:
        """Time the guard's verification and selection over N candidates.

        The extractor is the one ``scripts/09_train_generator.py`` builds for the guard --
        ``SlotAlignmentExtractor`` bound to the Bronze rendering of this record -- so this
        is the guard's real code path and not a stand-in for it.

        **What this measures and what it does not.** The guard's cost is the claim
        extractor plus the Phase 3 checker, run once per candidate, plus selection. None of
        that depends on which model produced the text, which is why it is measurable on a
        machine that cannot run the model. What it excludes is the four *generations* the
        guard requests, and that omission is stated on every guard figure rather than
        folded in: the measured number is the verification half of the overhead, and the
        generation half is unmeasured because generation is.

        Repair is disabled: a repair costs one more generation plus one more verification,
        and it fires only on a contradicted candidate. Bronze is faithful by construction,
        so it would never fire here, and leaving it enabled would report a repair rate of
        zero as if it were a property of the guard rather than of the input.

        Args:
            timer: The timer to attribute to.
            case_id: The case.
            bronze: The rendered Bronze narrative, used both as the candidate text and as
                the extractor's alignment reference.
            facts: The fact record.
            n_candidates: How many candidates to score.
        """
        from g2t_aml.corpus.silver.claim_extraction import SlotAlignmentExtractor
        from g2t_aml.facts.checkers import CheckContext
        from g2t_aml.models.generator.guard import InferenceGuard

        with timer.stage(Stage.GUARD):
            extractor = SlotAlignmentExtractor(bronze, vocabulary=self.vocabulary)
            InferenceGuard(allow_regeneration=False).run(
                case_id,
                [bronze.text] * n_candidates,
                facts,
                extractor,
                context=CheckContext(facts=facts),
            )


# ------------------------------------------------------------------ assembly ---


def _measured_row(
    spec: Any,
    samples: Sequence[BenchmarkSample],
    *,
    footprint: ModelFootprint,
    memory: MemoryProfile,
    cold_start: LatencySummary,
    assumptions: CostAssumptions,
    batch_size: int,
    n_warmup: int,
    guard_samples: Sequence[BenchmarkSample] = (),
    binned_samples: Sequence[BenchmarkSample] = (),
) -> SystemEfficiency:
    """Assemble one measured row from its samples.

    Args:
        spec: The registry spec.
        samples: Guard-off samples.
        footprint: Parameters and bytes.
        memory: Peak memory.
        cold_start: Load-to-ready distribution.
        assumptions: The cost model.
        batch_size: Which batch size these samples describe.
        n_warmup: Runs discarded.
        guard_samples: Guard-on samples, where they were taken.
        binned_samples: The size-stratified draw. Kept separate from ``samples`` on
            purpose: the headline distribution must come from a representative draw, and
            the per-band table must come from a draw that populates every band. Using one
            for both means either the bands are empty or the headline p95 describes a
            corpus whose size mix nobody has.

    Returns:
        The row.
    """
    seconds = [s.seconds for s in samples]
    summary = LatencySummary.from_samples(seconds)
    narratives_per_s = 1.0 / summary.mean_s if summary.mean_s > 0 else 0.0
    total_tokens = sum(s.n_output_tokens for s in samples)
    total_seconds = sum(seconds)
    return SystemEfficiency(
        system_id=spec.system_id,
        role=spec.role,
        measured=True,
        footprint=footprint,
        memory=memory,
        latency_guard_off=summary,
        latency_guard_on=(
            LatencySummary.from_samples([s.seconds for s in guard_samples])
            if guard_samples
            else None
        ),
        latency_by_node_bin=summarise_by_node_bin(binned_samples or samples),
        cold_start=cold_start,
        narratives_per_second=narratives_per_s * batch_size if batch_size > 1 else narratives_per_s,
        tokens_per_second=(total_tokens / total_seconds) if total_seconds > 0 else 0.0,
        batch_size=batch_size,
        cost=local_cost_per_1000(narratives_per_s, assumptions),
        deployment=_deployment_for(spec),
        stage_means=_stage_means(samples, guard_samples=guard_samples),
        n_runs=len(samples),
        n_warmup=n_warmup,
    )


def _stage_means(
    samples: Sequence[BenchmarkSample], *, guard_samples: Sequence[BenchmarkSample] = ()
) -> dict[str, float]:
    """Return the mean seconds per stage across a sample set.

    The guard stage is taken from the guard-on draw and merged in, so the breakdown shows
    what the guard costs even though the headline latency for a ``guard=False`` system
    correctly excludes it. Without this the guard's cost is visible only as the difference
    between two totals, which is exactly the arithmetic a reader should not have to do.

    Args:
        samples: The measured samples.
        guard_samples: The guard-on samples, if any.

    Returns:
        Stage name to mean seconds. A stage no sample recorded is absent rather than zero.
    """
    totals: dict[str, float] = {}
    for sample in samples:
        for stage, seconds in sample.stage_seconds.items():
            totals[stage] = totals.get(stage, 0.0) + seconds
    n = len(samples)
    means = {k: v / n for k, v in totals.items()} if n else {}
    if guard_samples:
        guard_total = sum(s.stage_seconds.get(str(Stage.GUARD), 0.0) for s in guard_samples)
        means[str(Stage.GUARD)] = guard_total / len(guard_samples)
    return means


def _blocker_for(spec: Any) -> str:
    """Return why a system could not be measured on this machine.

    Every blocker names the specific missing input, because "not run" is not a reason and
    a reader six months from now cannot tell an oversight from a constraint.

    Args:
        spec: The registry spec.

    Returns:
        The blocker text.
    """
    if spec.resource is Resource.API:
        return (
            "No API credentials; zero calls have been made. Cost is estimated from "
            "published pricing and is labelled as an estimate, not a measurement."
        )
    if spec.trained:
        return (
            "No trained checkpoint: Phase 11 has not run and Gate 8 is open. This machine "
            "is a 4 GB RTX 2050 with 7 GB system RAM; Llama-3.1-8B at nf4 is 4.5-5.6 GB of "
            "weights before a single activation, and CPU offload is closed by the RAM "
            "(D-068)."
        )
    return (
        "Requires the 8B backbone on an accelerator this machine does not have "
        "(4 GB RTX 2050, 7 GB system RAM; D-068)."
    )


def _api_token_estimate(
    pipeline: Pipeline, case_ids: Sequence[str], n: int = 50
) -> tuple[float, float]:
    """Measure the mean prompt and completion length the API baselines would pay for.

    The token counts are measured from this corpus rather than guessed: the prompt is the
    serialised fact record the API executors send, and the completion is a Bronze narrative,
    which is the right order of magnitude for a SAR narrative because it is one.

    Args:
        pipeline: The assembled pipeline.
        case_ids: Cases to sample.
        n: How many to sample.

    Returns:
        ``(mean_input_tokens, mean_output_tokens)``. A word count scaled by 1.3, the
        conventional English word-to-BPE-token ratio; this is an estimate and the cost
        figures derived from it are labelled as estimates.
    """
    ratio = 1.3
    inputs: list[float] = []
    outputs: list[float] = []
    for case_id in case_ids[:n]:
        row = pipeline.cases[case_id]
        cut = cut_case(
            pipeline.graph,
            str(row["seed_node"]),
            pipeline.window(row),
            pipeline.params,
            index=pipeline.index,
        )
        facts = extract_facts(materialise_cut(cut, pipeline.index))
        inputs.append(len(serialise_facts(facts, style="compact").split()) * ratio)
        outputs.append(
            len(pipeline._render(facts, vocabulary=pipeline.vocabulary).text.split()) * ratio
        )
    mean_in = sum(inputs) / len(inputs) if inputs else 0.0
    mean_out = sum(outputs) / len(outputs) if outputs else 0.0
    # A prompt is the record plus a system instruction and, for the few-shot arm, exemplars.
    # 400 tokens of instruction is the Phase 11 baseline prompt's own length.
    return mean_in + 400.0, mean_out


#: Training steps discarded before the step-time clock starts, for the same reason the
#: per-narrative protocol discards twenty: the first steps allocate the optimiser state
#: and build the autograd graph, and they are not what a steady-state step costs.
_TRAIN_WARMUP_STEPS = 5


def _encoder_training_memory(
    pipeline: Pipeline, case_ids: Sequence[str], device: str, *, n_steps: int = 20
) -> dict[str, Any]:
    """Measure the graph encoder's training-step peak memory and step time.

    **Reported as a component, not as a row.** B2 does not train: it reads the Phase 7
    checkpoint, and its registry spec says ``trained=False``. Putting a training-VRAM
    figure in B2's row would assert a cost that system does not pay. But the encoder *is*
    trained, every jointly-trained arm (A4, and S1 at the encoder learning rate) pays this
    cost, and it is measurable here -- so it is recorded where it is true.

    Args:
        pipeline: The assembled pipeline, carrying the loaded encoder.
        case_ids: Cases to build a training batch from.
        device: Device to measure on.
        n_steps: Forward-backward steps to run.

    Returns:
        The measurement, or a stated absence when no encoder is loaded.
    """
    if pipeline.encoder is None:
        return {"measured": False, "blocker": "no encoder checkpoint loaded"}

    import torch
    from torch_geometric.data import Batch

    from g2t_aml.models.encoder.features import build_case_data

    graphs = []
    for cid in case_ids[:32]:
        row = pipeline.cases[cid]
        case = materialise_cut(
            cut_case(
                pipeline.graph,
                str(row["seed_node"]),
                pipeline.window(row),
                pipeline.params,
                index=pipeline.index,
            ),
            pipeline.index,
        )
        graphs.append(
            build_case_data(case, pipeline.feature_space, seed_node=str(row["seed_node"]))
        )
    batch = Batch.from_data_list(graphs).to(device)

    model = pipeline.encoder
    model.train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-5)
    seconds: list[float] = []
    with measure_peak_memory(device) as mem:
        for step in range(n_steps):
            began = time.perf_counter()
            optimiser.zero_grad(set_to_none=True)
            out = model(batch)
            # A surrogate loss over the risk head. What is being measured is the memory and
            # time of one forward-backward-step cycle; the real objective would allocate the
            # same tensors, and using it here would need labels this measurement does not.
            loss = out.risk_logits.float().pow(2).mean()
            loss.backward()
            optimiser.step()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            # The first steps allocate the optimiser state and the autograd graph; they are
            # the same warm-up the per-narrative protocol discards, for the same reason.
            if step >= _TRAIN_WARMUP_STEPS:
                seconds.append(time.perf_counter() - began)
    model.eval()
    return {
        "measured": True,
        "batch_size": len(graphs),
        "n_steps": len(seconds),
        "train_allocated_gb": mem.get("allocated_gb"),
        "train_reserved_gb": mem.get("reserved_gb"),
        "step_seconds": LatencySummary.from_samples(seconds).to_dict(),
        "device": device,
        "note": (
            "The Phase 7 GATv2 encoder under AdamW at batch 32, measured on the recorded "
            "hardware. The loss is a surrogate (mean squared logit): what is being measured "
            "is the memory and time of a forward-backward-step cycle, not convergence. "
            "This is the cost every jointly-trained arm pays for the encoder half; the "
            "8B half is unmeasured."
        ),
    }


def _faithfulness_from_phase10(root: Path, stream: str = "balanced") -> dict[str, float]:
    """Read measured faithfulness scores for the frontier figure's y-axis.

    The Phase 11 aggregate is empty -- no matrix system has run -- so the only measured
    Zero-Hallucination Rate in the project is Phase 10's, over Bronze. Bronze *is* B1: the
    same deterministic renderer over the same fact records, which is why that score can be
    read onto B1's row rather than left blank. It is the same metric computed by the same
    code, read from a different file, and the figure says so in its caption.

    Args:
        root: Repository root.
        stream: Which stream to read. Streams are never pooled.

    Returns:
        System id to score. Empty when no evaluation has been written.
    """
    eval_dir = root / "artifacts" / "metrics" / "eval"
    if not eval_dir.is_dir():
        return {}
    candidates = sorted(eval_dir.glob("*/evaluation.json"))
    if not candidates:
        return {}
    payload = read_json(candidates[-1])
    if not isinstance(payload, dict):
        return {}
    # The evaluation file keys systems as "<system>/<stream>" in one flat string, not as
    # a nested mapping. Reading it as nested returns nothing and fails silently, which is
    # how this figure rendered empty the first time it was run.
    systems = payload.get("systems", {})
    bronze = systems.get(f"bronze/{stream}", {})
    score = bronze.get("faithfulness", {}).get("zero_hallucination_rate")
    return {"B1": float(score)} if score is not None else {}


# ------------------------------------------------------------------ main ---


def main() -> int:  # noqa: PLR0912, PLR0915 -- one linear measure-then-assemble pass
    """Run the benchmark and write the efficiency table.

    Returns:
        0 on success, 1 if the substrate the benchmark needs is absent. A system that
        cannot be measured is not a failure -- it is a row with a blocker.
    """
    parser = argparse.ArgumentParser(description="Phase 13: benchmark every system.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--n-warmup", type=int, default=DEFAULT_N_WARMUP)
    parser.add_argument("--n-measured", type=int, default=DEFAULT_N_MEASURED)
    parser.add_argument("--quick", action="store_true", help="5 warm-up / 10 measured")
    parser.add_argument("--no-encoder", action="store_true", help="skip the encoder path")
    parser.add_argument("--device", default="cuda", help="device for the encoder")
    parser.add_argument(
        "--draw-seed", type=int, default=13, help="seed for the representative case draw"
    )
    parser.add_argument(
        "--n-per-bin", type=int, default=40, help="measured runs per case-size band"
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "metrics" / "phase13")
    args = parser.parse_args()

    configure_logging()
    n_warmup = 5 if args.quick else args.n_warmup
    n_measured = 10 if args.quick else args.n_measured

    hardware = capture_hardware()
    log.info("hardware: %s", hardware.describe())

    device = args.device
    if device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                log.warning("no CUDA device; the encoder path will run on CPU")
                device = "cpu"
        except ImportError:
            device = "cpu"

    try:
        pipeline = Pipeline(root=args.root, with_encoder=not args.no_encoder, device=device)
    except FileNotFoundError as exc:
        log.error("cannot benchmark: %s", exc)
        return 1

    case_ids = pipeline.case_ids("test", seed=args.draw_seed)
    log.info("%d test cases available (shuffled under seed %d)", len(case_ids), args.draw_seed)
    by_bin = pipeline.case_ids_by_bin(case_ids)
    log.info("case-size bands: %s", {k: len(v) for k, v in by_bin.items()})
    if len(case_ids) < n_warmup + n_measured:
        log.warning(
            "only %d cases for %d runs; the protocol will cycle, which measures a warmer "
            "cache than a non-repeating draw would",
            len(case_ids),
            n_warmup + n_measured,
        )

    errors: list[dict[str, str]] = []

    def _record(case_id: str, exc: Exception) -> None:
        errors.append({"case_id": case_id, "error": f"{type(exc).__name__}: {exc}"})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- cold start. Measured with no warm-up, deliberately: warm-up is the thing cold
    # start is defined as the absence of.
    cold_start = measure_cold_start(
        lambda: CanonicalGraph.load(args.root / "data" / "interim" / DATASET), n_repeats=3
    )
    log.info("cold start (graph load) p50 %.2f s", cold_start.p50_s)

    table = EfficiencyTable(
        hardware=hardware,
        generated_at=datetime.now(UTC).isoformat(),
        protocol={
            "n_warmup": n_warmup,
            "n_measured": n_measured,
            "batch_sizes": [INTERACTIVE_BATCH, THROUGHPUT_BATCH],
            "split": "test",
            "n_cases_available": len(case_ids),
            "cases_cycled": len(case_ids) < n_warmup + n_measured,
            "draw_seed": args.draw_seed,
            "draw": "seeded shuffle of the frozen test split, representative of the corpus",
            "n_per_bin": args.n_per_bin,
            "binned_draw": "size-stratified, reported only in the by-size table",
            "cases_per_band": {k: len(v) for k, v in by_bin.items()},
            "percentile_method": "nearest-rank over observed samples, never interpolated",
            "stages_timed": [str(s) for s in Stage],
            "graph_load_s": pipeline.graph_load_s,
            "index_build_s": pipeline.index_build_s,
            "encoder_load_s": pipeline.encoder_load_s,
            "cold_start_total_s": pipeline.cold_start_s,
            "guard_candidates": 4,
            "device": device,
        },
    )

    all_samples: dict[str, list[BenchmarkSample]] = {}

    # ---- B1: template, no encoder, no guard. A complete system, measured end to end.
    log.info("benchmarking B1 (template) at batch 1 ...")
    with measure_peak_memory(device) as b1_mem:
        b1 = run_benchmark(
            lambda cid: pipeline.measure(cid),
            case_ids,
            n_warmup=n_warmup,
            n_measured=n_measured,
            on_error=_record,
        )
    all_samples["B1"] = b1
    log.info("B1: n=%d p50 %.4f s p95 %.4f s", len(b1), *_p(b1))

    # ---- the guard's verification cost, measured over the same cases. Reported against
    # B1 as its guard-on counterpart, which is what makes the 4x claim a measurement.
    log.info("benchmarking B1 + guard (4 candidates) ...")
    b1_guard = run_benchmark(
        lambda cid: pipeline.measure(cid, with_guard=True),
        case_ids,
        n_warmup=n_warmup,
        n_measured=n_measured,
        on_error=_record,
    )
    all_samples["B1_guard"] = b1_guard
    log.info("B1+guard: n=%d p50 %.4f s p95 %.4f s", len(b1_guard), *_p(b1_guard))

    # ---- B2: encoder + template.
    b2: list[BenchmarkSample] = []
    b2_mem: dict[str, float] = {}
    if pipeline.encoder is not None:
        log.info("benchmarking B2 (encoder + template) on %s ...", device)
        with measure_peak_memory(device) as b2_mem:
            b2 = run_benchmark(
                lambda cid: pipeline.measure(cid, with_encoder=True),
                case_ids,
                n_warmup=n_warmup,
                n_measured=n_measured,
                on_error=_record,
            )
        all_samples["B2"] = b2
        log.info("B2: n=%d p50 %.4f s p95 %.4f s", len(b2), *_p(b2))

    # ---- the size-stratified draw. The headline distribution above comes from a
    # representative sample, which is right for p95 and wrong for the size table: this
    # corpus is dominated by small cases, so a representative draw leaves the large bands
    # nearly empty and the "latency by case size" table with nothing in its right-hand
    # columns. This draw fills every populated band explicitly and is reported only in
    # that table, never pooled into the headline.
    def _binned(system: str, *, with_encoder: bool) -> list[BenchmarkSample]:
        """Run the size-stratified draw for one system.

        Args:
            system: Label for the log line and the raw-sample file.
            with_encoder: Whether to run the encoder stage.

        Returns:
            Every band's samples, concatenated.
        """
        log.info("benchmarking %s across %d case-size bands ...", system, len(by_bin))
        out: list[BenchmarkSample] = []
        for label, ids in by_bin.items():
            if not ids:
                log.info("  band %s: no cases in this corpus", label)
                continue
            n_bin = min(5, len(ids)) if args.quick else min(args.n_per_bin, len(ids))
            band = run_benchmark(
                lambda cid: pipeline.measure(cid, with_encoder=with_encoder),
                ids,
                n_warmup=min(n_warmup, len(ids)),
                n_measured=n_bin,
                on_error=_record,
            )
            out.extend(band)
            log.info("  band %s: n=%d p50 %.4f s p95 %.4f s", label, len(band), *_p(band))
        all_samples[f"{system}_binned"] = out
        return out

    binned = _binned("B1", with_encoder=False)
    binned_b2 = _binned("B2", with_encoder=True) if pipeline.encoder is not None else []

    # ---- batch 32. The template path is not batched internally, so this measures the
    # throughput of a 32-case queue rather than a batched forward pass, and it is labelled
    # that way rather than implied to be the second thing.
    log.info("benchmarking B1 at batch %d ...", THROUGHPUT_BATCH)
    batch_samples = _measure_batch(pipeline, case_ids, THROUGHPUT_BATCH, n_warmup, n_measured)

    # ---- footprints.
    encoder_bytes = (
        directory_size_bytes(pipeline.encoder_checkpoint) if pipeline.encoder_checkpoint else 0
    )
    template_footprint = ModelFootprint(
        total_params=0,
        trainable_params=0,
        notes="No learned parameters: deterministic template rendering from the record.",
    )
    encoder_footprint = ModelFootprint(
        total_params=pipeline.encoder_params[0] if pipeline.encoder_params else 0,
        trainable_params=pipeline.encoder_params[1] if pipeline.encoder_params else 0,
        encoder_bytes=encoder_bytes,
        notes=(
            "The Phase 7 GATv2 checkpoint. The on-disk figure includes the optimiser state "
            "and the fitted feature space the checkpoint carries, not weights alone."
        ),
    )

    cpu_memory = MemoryProfile(
        host_ram_peak_gb=b1_mem.get("host_ram_peak_gb"),
        measured_on="cpu",
    )
    encoder_memory = MemoryProfile(
        inference_allocated_gb=b2_mem.get("allocated_gb"),
        inference_reserved_gb=b2_mem.get("reserved_gb"),
        host_ram_peak_gb=b2_mem.get("host_ram_peak_gb"),
        measured_on=device,
    )

    log.info("measuring the encoder's training-step footprint ...")
    encoder_training = _encoder_training_memory(pipeline, case_ids, device)
    if encoder_training.get("measured"):
        log.info(
            "encoder training: peak reserved %.3f GB, step p50 %.4f s",
            encoder_training["train_reserved_gb"] or 0.0,
            encoder_training["step_seconds"]["p50_s"],
        )

    api_in, api_out = _api_token_estimate(pipeline, case_ids)
    log.info("API prompt/completion estimate: %.0f / %.0f tokens", api_in, api_out)

    # ---- assemble every row in the registry, measured or not.
    specs = {s.system_id: s for s in all_systems()}
    for system_id, spec in specs.items():
        if system_id == "B1" and b1:
            table.add(
                _measured_row(
                    spec,
                    b1,
                    footprint=template_footprint,
                    memory=cpu_memory,
                    cold_start=cold_start,
                    assumptions=CPU_ASSUMPTIONS,
                    batch_size=INTERACTIVE_BATCH,
                    n_warmup=n_warmup,
                    guard_samples=b1_guard,
                    binned_samples=binned,
                )
            )
        elif system_id == "B2" and b2:
            table.add(
                _measured_row(
                    spec,
                    b2,
                    footprint=encoder_footprint,
                    memory=encoder_memory,
                    cold_start=LatencySummary.from_samples([pipeline.cold_start_s] * 3),
                    assumptions=CPU_ASSUMPTIONS,
                    batch_size=INTERACTIVE_BATCH,
                    n_warmup=n_warmup,
                    binned_samples=binned_b2,
                )
            )
        else:
            cost = None
            if spec.resource is Resource.API:
                cost = api_cost_per_1000(
                    api_in,
                    api_out,
                    API_ASSUMPTIONS,
                    calls_per_narrative=_API_CALLS_PER_NARRATIVE.get(spec.executor, 1.0),
                )
            table.add(
                SystemEfficiency(
                    system_id=system_id,
                    role=spec.role,
                    measured=False,
                    blocker=_blocker_for(spec),
                    cost=cost,
                    deployment=_deployment_for(spec),
                )
            )

    # ---- write everything.
    table.write_json(out_dir / "efficiency.json")
    (out_dir / "table_efficiency.tex").write_text(table.to_latex(), encoding="utf-8")
    (out_dir / "table_guard_cost.tex").write_text(table.guard_table_to_latex(), encoding="utf-8")
    (out_dir / "table_latency_by_size.tex").write_text(
        table.node_bin_table_to_latex(), encoding="utf-8"
    )
    write_jsonl(
        out_dir / "raw_samples.jsonl",
        [{"system": k, **s.to_dict()} for k, v in all_samples.items() for s in v],
    )
    write_json(out_dir / "batch_throughput.json", batch_samples)
    write_json(
        out_dir / "components.json",
        {
            "note": (
                "Per-stage measurements that are not themselves systems. Every 8B arm pays "
                "these on top of its decoder, so when a card arrives the only stage still "
                "unmeasured is generation."
            ),
            "encoder_training": encoder_training,
            "encoder_inference_load_s": pipeline.encoder_load_s,
            "graph_load_s": pipeline.graph_load_s,
            "index_build_s": pipeline.index_build_s,
            "guard_verification": {
                "n_candidates": 4,
                "repair_enabled": False,
                "mean_seconds": (
                    sum(s.stage_seconds.get(str(Stage.GUARD), 0.0) for s in b1_guard)
                    / len(b1_guard)
                    if b1_guard
                    else None
                ),
                "note": (
                    "Verification only. The four generations the guard requests are not "
                    "included, because generation is not measurable on this machine."
                ),
            },
        },
    )
    if errors:
        write_jsonl(out_dir / "errors.jsonl", errors)
        log.warning("%d runs failed; see errors.jsonl", len(errors))

    write_json(
        out_dir / "run_context.json",
        RunContext.capture(
            experiment_name="phase13_benchmark",
            cfg=dict(table.protocol),
            seeds={"extraction": pipeline.params.seed},
            repo_root=REPO_ROOT,
            hardware=hardware.to_dict(),
            n_measured=n_measured,
            n_warmup=n_warmup,
        ).to_dict(),
    )

    # ---- the frontier figure. Rendered here rather than by the Phase 11 aggregator so it
    # picks up this run's measurements without waiting on the matrix.
    try:
        from g2t_aml.experiments.aggregate import aggregate_matrix
        from g2t_aml.experiments.figures import efficiency_frontier
    except ImportError:
        log.warning("matplotlib is not installed; skipping the frontier figure")
    else:
        figures_dir = args.root / "artifacts" / "figures" / "phase13"
        figures_dir.mkdir(parents=True, exist_ok=True)
        figure = efficiency_frontier(
            aggregate_matrix(args.root / "artifacts" / "matrix"),
            figures_dir / "fig_efficiency_frontier.pdf",
            efficiency=table.to_dict(),
            faithfulness=_faithfulness_from_phase10(args.root),
        )
        log.info("frontier figure -> %s", figure)

    cov = table.coverage()
    log.info(
        "efficiency table written: %d systems, %d measured, %d blocked",
        cov["n_systems"],
        cov["n_measured"],
        cov["n_blocked"],
    )
    return 0


def _p(samples: Sequence[BenchmarkSample]) -> tuple[float, float]:
    """Return p50 and p95 of a sample set, for a log line.

    Args:
        samples: The measured samples.

    Returns:
        ``(p50, p95)`` in seconds.
    """
    summary = LatencySummary.from_samples([s.seconds for s in samples])
    return summary.p50_s, summary.p95_s


def _measure_batch(
    pipeline: Pipeline,
    case_ids: Sequence[str],
    batch_size: int,
    n_warmup: int,
    n_measured: int,
) -> dict[str, Any]:
    """Measure sustained throughput over batches of cases.

    **This is queue throughput, not a batched forward pass.** The template path has no
    batch dimension to exploit; what a batch buys it is amortised per-process setup and
    better cache behaviour, and calling the result "batch 32 latency" would imply a
    parallelism that is not there. The 8B arms would batch genuinely, and that is one more
    thing this machine cannot measure.

    Args:
        pipeline: The assembled pipeline.
        case_ids: Cases to draw from.
        batch_size: Cases per batch.
        n_warmup: Warm-up batches.
        n_measured: Measured batches. Capped so the run stays proportionate.

    Returns:
        A mapping carrying the per-batch distribution and the derived throughput.
    """
    n_batches = max(1, min(n_measured // 4, 25))
    seconds: list[float] = []
    for b in range(n_warmup // 4 + n_batches):
        start = (b * batch_size) % max(1, len(case_ids) - batch_size)
        chunk = case_ids[start : start + batch_size]
        began = time.perf_counter()
        for cid in chunk:
            try:
                pipeline.measure(cid)
            except Exception:
                continue
        elapsed = time.perf_counter() - began
        if b >= n_warmup // 4:
            seconds.append(elapsed)
    summary = LatencySummary.from_samples(seconds)
    return {
        "batch_size": batch_size,
        "n_batches": len(seconds),
        "per_batch": summary.to_dict(),
        "narratives_per_second": (batch_size / summary.mean_s) if summary.mean_s > 0 else 0.0,
        "note": (
            "Queue throughput over a batch of independent cases, not a batched forward "
            "pass: the template path has no batch dimension."
        ),
    }


if __name__ == "__main__":
    sys.exit(main())
