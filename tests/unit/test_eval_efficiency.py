"""Phase 13: the efficiency instrument.

The measurements this module produces go into a table a reader sizes hardware from, so the
tests here are about the properties that make such a table trustworthy rather than about
whether a stopwatch runs. Four of them exist because getting them wrong is silent:

- **A zero must not print as an absence** and an absence must not print as a zero. B1 has
  exactly zero learned parameters, and a table that renders that as ``--`` tells a reader
  the number was not measured.
- **Percentiles are nearest-rank.** An interpolated p95 is a latency that never happened,
  and a capacity planner sizing a queue against it is sizing against a fiction.
- **An unmeasured row must carry its blocker**, enforced at construction, because an
  absence without a reason reads as an oversight six months later (invariant 7).
- **The warm-up must actually be discarded.** A protocol that says it discards twenty and
  keeps them reports a cold system's latency under a warm system's name.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from g2t_aml.eval.efficiency import (
    DEFAULT_NODE_BINS,
    PER_NARRATIVE_STAGES,
    BenchmarkSample,
    CostAssumptions,
    CostEstimate,
    DeploymentProfile,
    EfficiencyTable,
    EndToEndTimer,
    HardwareConfig,
    LatencySummary,
    MemoryProfile,
    ModelFootprint,
    NodeBin,
    Stage,
    SystemEfficiency,
    api_cost_per_1000,
    capture_hardware,
    directory_size_bytes,
    local_cost_per_1000,
    measure_cold_start,
    measure_peak_memory,
    percentile,
    run_benchmark,
    summarise_by_node_bin,
)


def _sample(case_id: str = "c1", *, n_nodes: int = 10, seconds: float = 0.01) -> BenchmarkSample:
    """Build a benchmark sample.

    Args:
        case_id: The case id.
        n_nodes: Node count, which is the binning axis.
        seconds: Wall time.

    Returns:
        The sample.
    """
    return BenchmarkSample(case_id=case_id, n_nodes=n_nodes, n_edges=n_nodes * 2, seconds=seconds)


class TestPercentile:
    """Nearest-rank, over observed samples, reproducible from the published raw data."""

    def test_nearest_rank_returns_an_observed_value(self) -> None:
        """Every percentile is a member of the sample, never a value between two."""
        samples = [1.0, 2.0, 3.0, 4.0, 100.0]
        for q in (0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0):
            assert percentile(samples, q) in samples

    def test_p95_of_a_hundred_is_the_ninety_fifth_observation(self) -> None:
        """The definition a reader can reproduce: ceil(q*n) on the sorted sample."""
        samples = [float(i) for i in range(1, 101)]
        assert percentile(samples, 0.95) == 95.0
        assert percentile(samples, 0.99) == 99.0
        assert percentile(samples, 0.50) == 50.0

    def test_unsorted_input_is_sorted_first(self) -> None:
        """The caller is not required to sort."""
        assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], 0.5) == 3.0

    def test_empty_sample_is_zero_not_an_error(self) -> None:
        """A system that produced no runs still occupies a row."""
        assert percentile([], 0.95) == 0.0

    def test_quantile_out_of_range_raises(self) -> None:
        """A quantile above one is a caller bug, not something to clamp silently."""
        with pytest.raises(ValueError, match="quantile"):
            percentile([1.0], 1.5)


class TestLatencySummary:
    """The distribution, because the mean is the number that hides the problem."""

    def test_summary_reports_the_tail(self) -> None:
        """p95 and p99 come from the tail, not from the bulk."""
        samples = [0.01] * 95 + [1.0] * 5
        summary = LatencySummary.from_samples(samples)
        assert summary.n == 100
        assert summary.p50_s == pytest.approx(0.01)
        assert summary.p95_s == pytest.approx(0.01)
        assert summary.p99_s == pytest.approx(1.0)
        assert summary.max_s == pytest.approx(1.0)

    def test_two_systems_with_one_mean_have_different_tails(self) -> None:
        """The property the table exists to show: equal means, unequal tails."""
        steady = [0.1] * 100
        spiky = [0.05] * 98 + [2.6, 2.5]
        assert LatencySummary.from_samples(steady).mean_s == pytest.approx(
            LatencySummary.from_samples(spiky).mean_s, rel=1e-6
        )
        assert LatencySummary.from_samples(spiky).p99_s > LatencySummary.from_samples(steady).p99_s

    def test_p99_at_n_100_is_the_second_worst_run_and_misses_a_lone_outlier(self) -> None:
        """The caveat the module documents, pinned so nobody reads p99 as the worst case.

        Nearest-rank p99 over exactly 100 observations is ``ordered[98]`` -- the
        second-worst run. A distribution with a single catastrophic outlier therefore has a
        p99 that does not see it, and only ``max_s`` does. That is not a bug in the
        percentile; it is what n=100 can resolve, and it is why the summary publishes
        ``max_s`` beside p99 rather than stopping at the tail quantile.
        """
        one_outlier = LatencySummary.from_samples([0.05] * 99 + [10.0])
        assert one_outlier.p99_s == pytest.approx(0.05)
        assert one_outlier.max_s == pytest.approx(10.0)

    def test_empty_is_all_zero_at_n_zero(self) -> None:
        """An absent distribution is n=0, distinguishable from a fast one."""
        summary = LatencySummary.from_samples([])
        assert summary.n == 0
        assert summary.p95_s == 0.0

    def test_single_sample_has_zero_stdev_rather_than_raising(self) -> None:
        """One observation is a valid, if weak, measurement."""
        assert LatencySummary.from_samples([0.5]).std_s == 0.0


class TestNodeBinning:
    """A 150-node case does not cost what a 20-node case costs."""

    def test_bins_are_closed_open(self) -> None:
        """The upper bound belongs to the next bin, so no case is counted twice."""
        node_bin = NodeBin(25, 50)
        assert not node_bin.contains(24)
        assert node_bin.contains(25)
        assert node_bin.contains(49)
        assert not node_bin.contains(50)

    def test_zero_high_is_unbounded(self) -> None:
        """The top band has no ceiling."""
        assert NodeBin(100, 0).contains(10_000)
        assert NodeBin(100, 0).label == "100+"

    def test_default_bins_partition_every_node_count(self) -> None:
        """Every case lands in exactly one band, or the size table double-counts."""
        for n in range(0, 400):
            hits = [b for b in DEFAULT_NODE_BINS if b.contains(n)]
            assert len(hits) == 1, f"{n} nodes landed in {len(hits)} bands"

    def test_empty_band_is_reported_not_omitted(self) -> None:
        """An unpopulated band is information about the corpus, not a gap to hide."""
        summaries = summarise_by_node_bin([_sample(n_nodes=5)])
        assert set(summaries) == {b.label for b in DEFAULT_NODE_BINS}
        assert summaries["100+"].n == 0
        assert summaries["0-24"].n == 1

    def test_larger_cases_are_binned_separately(self) -> None:
        """The whole point: the bands separate, so a size effect is visible."""
        samples = [_sample(n_nodes=10, seconds=0.01), _sample(n_nodes=60, seconds=0.05)]
        summaries = summarise_by_node_bin(samples)
        assert summaries["0-24"].p50_s == pytest.approx(0.01)
        assert summaries["50-99"].p50_s == pytest.approx(0.05)


class TestBenchmarkProtocol:
    """Warm up, discard, measure. The discard is the part that must be real."""

    def test_warmup_runs_are_executed_and_discarded(self) -> None:
        """Twenty runs happen and none of them reach the result."""
        calls: list[str] = []

        def measure(case_id: str) -> BenchmarkSample:
            calls.append(case_id)
            return _sample(case_id, seconds=float(len(calls)))

        samples = run_benchmark(measure, ["a", "b", "c"], n_warmup=20, n_measured=10)
        assert len(calls) == 30
        assert len(samples) == 10
        # The measured samples are the last ten, so the warm-up's cheap first runs are gone.
        assert min(s.seconds for s in samples) == 21.0

    def test_a_failing_case_is_recorded_and_measurement_continues(self) -> None:
        """A benchmark that aborts on one bad case reports nothing about the good ones."""
        seen: list[str] = []

        def measure(case_id: str) -> BenchmarkSample:
            if case_id == "bad":
                raise RuntimeError("boom")
            return _sample(case_id)

        samples = run_benchmark(
            measure,
            ["ok", "bad"],
            n_warmup=0,
            n_measured=10,
            on_error=lambda cid, exc: seen.append(cid),
        )
        assert len(samples) == 5
        assert seen == ["bad"] * 5

    def test_empty_case_list_raises(self) -> None:
        """Benchmarking nothing is a caller bug, not an empty result."""
        with pytest.raises(ValueError, match="no cases"):
            run_benchmark(lambda _: _sample(), [], n_warmup=0, n_measured=1)

    def test_cold_start_takes_no_warmup(self) -> None:
        """Warm-up is the thing cold start is defined as the absence of."""
        loads = 0

        def load() -> int:
            nonlocal loads
            loads += 1
            return loads

        summary = measure_cold_start(load, n_repeats=3)
        assert loads == 3
        assert summary.n == 3


class TestEndToEndTimer:
    """Latency is end-to-end or it is not reported."""

    def test_stages_accumulate_on_re_entry(self) -> None:
        """A guard that verifies four candidates enters its stage four times."""
        timer = EndToEndTimer()
        for _ in range(4):
            with timer.stage(Stage.GUARD):
                pass
        assert len(timer.stages) == 1
        assert timer.stages[str(Stage.GUARD)] >= 0.0

    def test_graph_load_is_excluded_from_the_per_narrative_total(self) -> None:
        """A once-per-process cost charged to every narrative multiplies a one-off."""
        timer = EndToEndTimer()
        timer.record(Stage.GRAPH_LOAD, 5.0)
        timer.record(Stage.FACT_EXTRACTION, 0.1)
        timer.record(Stage.GENERATION, 0.2)
        assert timer.total() == pytest.approx(0.3)
        assert Stage.GRAPH_LOAD not in PER_NARRATIVE_STAGES

    def test_the_total_is_the_sum_of_the_breakdown(self) -> None:
        """The headline and the breakdown cannot disagree, because one is the other."""
        timer = EndToEndTimer()
        for stage in PER_NARRATIVE_STAGES:
            timer.record(stage, 0.25)
        assert timer.total() == pytest.approx(0.25 * len(PER_NARRATIVE_STAGES))


class TestFootprint:
    """Parameters and bytes, broken out, because the base model is not the contribution."""

    def test_shipped_bytes_excludes_the_base_model(self) -> None:
        """What the project distributes is three orders smaller than what it runs on."""
        footprint = ModelFootprint(
            total_params=8_030_000_000,
            trainable_params=42_000_000,
            base_model_bytes=5_600_000_000,
            adapter_bytes=160_000_000,
            encoder_bytes=2_500_000,
            fusion_bytes=8_000_000,
        )
        assert footprint.total_bytes == 5_770_500_000
        assert footprint.shipped_bytes == 170_500_000
        assert footprint.trainable_fraction == pytest.approx(42 / 8030, rel=1e-3)

    def test_zero_parameters_is_a_value_not_a_division_error(self) -> None:
        """A template system has no parameters and still has a footprint."""
        assert ModelFootprint(total_params=0, trainable_params=0).trainable_fraction == 0.0

    def test_directory_size_counts_every_file(self, tmp_path: Path) -> None:
        """A checkpoint directory is the sum of what is in it."""
        (tmp_path / "a.bin").write_bytes(b"x" * 100)
        nested = tmp_path / "sub"
        nested.mkdir()
        (nested / "b.bin").write_bytes(b"y" * 50)
        assert directory_size_bytes(tmp_path) == 150
        assert directory_size_bytes(tmp_path / "a.bin") == 100

    def test_a_missing_path_is_zero_not_an_error(self, tmp_path: Path) -> None:
        """A footprint can be assembled before the checkpoint exists."""
        assert directory_size_bytes(tmp_path / "absent") == 0


class TestCostModel:
    """Cost is a declared model, and the declaration is the deliverable."""

    def test_utilisation_is_the_number_that_moves_the_answer(self) -> None:
        """Assuming a dedicated box runs flat out divides the hourly cost by four."""
        half = CostAssumptions(hardware_capital_usd=9000.0, depreciation_years=3.0, utilisation=0.5)
        eighth = CostAssumptions(
            hardware_capital_usd=9000.0, depreciation_years=3.0, utilisation=0.125
        )
        assert eighth.amortised_usd_per_hour == pytest.approx(
            4.0 * half.amortised_usd_per_hour, rel=1e-9
        )

    def test_amortisation_is_straight_line_over_serving_hours(self) -> None:
        """The arithmetic a reader can check by hand."""
        assumptions = CostAssumptions(
            hardware_capital_usd=8766.0, depreciation_years=1.0, utilisation=1.0
        )
        # 365.25 * 24 = 8766 serving hours in one year at full utilisation.
        assert assumptions.amortised_usd_per_hour == pytest.approx(1.0, rel=1e-3)

    def test_power_includes_the_cooling_multiplier(self) -> None:
        """PUE is not decoration: at 1.5 it is a third of the energy bill."""
        assumptions = CostAssumptions(power_draw_w=1000.0, pue=1.5, electricity_usd_per_kwh=0.10)
        assert assumptions.power_usd_per_hour == pytest.approx(0.15)

    def test_no_capital_declared_is_zero_not_a_division_error(self) -> None:
        """A default-constructed model costs nothing rather than raising."""
        assert CostAssumptions().hourly_cost() == 0.0

    def test_local_cost_scales_inversely_with_throughput(self) -> None:
        """Twice the throughput is half the cost per thousand."""
        assumptions = CostAssumptions(hardware_capital_usd=9000.0, utilisation=0.5)
        slow = local_cost_per_1000(1.0, assumptions)
        fast = local_cost_per_1000(2.0, assumptions)
        assert fast.usd_per_1000 == pytest.approx(slow.usd_per_1000 / 2.0, rel=1e-9)
        assert slow.basis == "amortised_local"

    def test_zero_throughput_is_infinite_not_a_crash(self) -> None:
        """A system that produces nothing has no cost per thousand."""
        assert math.isinf(local_cost_per_1000(0.0, CostAssumptions()).usd_per_1000)

    def test_api_cost_charges_every_call_the_system_makes(self) -> None:
        """An agentic comparator priced at one call is priced as something it is not."""
        assumptions = CostAssumptions(api_input_usd_per_mtok=15.0, api_output_usd_per_mtok=75.0)
        one = api_cost_per_1000(1000.0, 500.0, assumptions, calls_per_narrative=1.0)
        three = api_cost_per_1000(1000.0, 500.0, assumptions, calls_per_narrative=3.0)
        assert one.basis == "api_marginal"
        # 1000 input at 15/Mtok + 500 output at 75/Mtok = 0.0525 per call, 52.50 per 1000.
        assert one.usd_per_1000 == pytest.approx(52.5)
        assert three.usd_per_1000 == pytest.approx(157.5)

    def test_breakeven_is_none_when_the_api_is_cheaper_at_every_volume(self) -> None:
        """The table must not imply a crossover that does not exist."""
        estimate = local_cost_per_1000(
            1.0, CostAssumptions(hardware_capital_usd=9000.0, utilisation=0.5)
        )
        assert estimate.breakeven_narratives_per_month(0.0) is None

    def test_breakeven_is_reported_for_an_api_marginal_estimate_as_none(self) -> None:
        """Only a local estimate has capital to amortise against a volume."""
        api = CostEstimate(usd_per_1000=52.5, basis="api_marginal")
        assert api.breakeven_narratives_per_month(100.0) is None


class TestSystemEfficiencyRow:
    """An absence must state its reason. Invariant 7, enforced at construction."""

    def test_an_unmeasured_row_without_a_blocker_raises(self) -> None:
        """An absence with no reason reads as an oversight, and later nobody can tell."""
        with pytest.raises(ValueError, match="blocker"):
            SystemEfficiency(system_id="S1", measured=False)

    def test_an_unmeasured_row_with_a_blocker_is_valid(self) -> None:
        """A blocked row is a first-class value, not a gap."""
        row = SystemEfficiency(system_id="S1", measured=False, blocker="no checkpoint")
        assert row.to_dict()["blocker"] == "no checkpoint"

    def test_a_measured_row_needs_no_blocker(self) -> None:
        """The constraint applies only where it means something."""
        assert SystemEfficiency(system_id="B1", measured=True).measured

    def test_absent_measurements_serialise_as_none_not_zero(self) -> None:
        """A reader must be able to tell an unmeasured latency from a fast one."""
        payload = SystemEfficiency(system_id="S1", measured=False, blocker="x").to_dict()
        assert payload["latency_guard_off"] is None
        assert payload["narratives_per_second"] is None


class TestEfficiencyTable:
    """The table, and the two ways it could lie about what it contains."""

    @staticmethod
    def _table() -> EfficiencyTable:
        """Build a table with one measured row and one blocked one.

        Returns:
            The table.
        """
        table = EfficiencyTable(
            hardware=HardwareConfig(
                gpu_name="Test GPU",
                gpu_vram_gb=4.0,
                gpu_driver="1.0",
                cuda_runtime="12.1",
                cuda_capability="8.6",
                torch_version="2.4.0",
                cpu_model="Test CPU",
                cpu_count=8,
                ram_gb=7.0,
                platform="linux",
                python_version="3.11.14",
            )
        )
        table.add(
            SystemEfficiency(
                system_id="B1",
                role="Faithfulness ceiling",
                measured=True,
                footprint=ModelFootprint(total_params=0, trainable_params=0),
                memory=MemoryProfile(measured_on="cpu"),
                latency_guard_off=LatencySummary.from_samples([0.01] * 100),
                latency_guard_on=LatencySummary.from_samples([0.04] * 100),
                narratives_per_second=100.0,
                cost=local_cost_per_1000(100.0, CostAssumptions(hardware_capital_usd=2500.0)),
                deployment=DeploymentProfile(
                    on_premise=True, data_leaves_perimeter="Nothing.", min_viable_hardware="CPU"
                ),
                latency_by_node_bin=summarise_by_node_bin([_sample(n_nodes=10)]),
                n_runs=100,
            )
        )
        table.add(
            SystemEfficiency(
                system_id="S1",
                role="Full system",
                measured=False,
                blocker="No checkpoint: Gate 8 is open and the card is 4 GB.",
                deployment=DeploymentProfile(
                    on_premise=True, data_leaves_perimeter="Nothing.", min_viable_hardware="16 GB"
                ),
            )
        )
        return table

    def test_coverage_states_how_much_of_the_table_is_real(self) -> None:
        """'The efficiency table' and 'one measured row of two' are different claims."""
        assert self._table().coverage() == {"n_systems": 2, "n_measured": 1, "n_blocked": 1}

    def test_zero_parameters_prints_as_zero_not_as_a_dash(self) -> None:
        """B1 has exactly no learned parameters and the table must say so."""
        latex = self._table().to_latex()
        b1_row = next(line for line in latex.splitlines() if line.startswith("B1"))
        assert "0.0 & 0.0" in b1_row, b1_row

    def test_an_unmeasured_cell_prints_as_a_dash_with_a_footnote(self) -> None:
        """The reader sees which numbers exist and why the others do not, on one page."""
        latex = self._table().to_latex()
        s1_row = next(line for line in latex.splitlines() if line.startswith("S1"))
        assert "--" in s1_row
        assert "\\textsuperscript{1}" in s1_row
        assert "Gate 8 is open" in latex

    def test_the_cost_column_does_not_collapse_two_different_values(self) -> None:
        """Costs span five orders of magnitude; fixed decimals render 0.00053 and 0.00116
        as the same cell, and two systems differing by a factor of two look identical."""
        table = EfficiencyTable()
        table.add(
            SystemEfficiency(
                system_id="LOW",
                measured=True,
                cost=CostEstimate(usd_per_1000=0.00053, basis="amortised_local"),
            )
        )
        table.add(
            SystemEfficiency(
                system_id="HIGH",
                measured=True,
                cost=CostEstimate(usd_per_1000=0.00116, basis="amortised_local"),
            )
        )
        table.add(
            SystemEfficiency(
                system_id="API",
                measured=True,
                cost=CostEstimate(usd_per_1000=22.645, basis="api_marginal"),
            )
        )
        latex = table.to_latex()
        low = next(line for line in latex.splitlines() if line.startswith("LOW"))
        high = next(line for line in latex.splitlines() if line.startswith("HIGH"))
        api = next(line for line in latex.splitlines() if line.startswith("API"))
        assert "0.00053" in low
        assert "0.00116" in high
        assert "22.6" in api

    def test_the_caption_names_the_hardware_and_the_coverage(self) -> None:
        """A latency table whose caption omits the machine is one a reviewer cannot use."""
        latex = self._table().to_latex()
        assert "Test GPU" in latex
        assert "1 of 2 rows are measured" in latex

    def test_latex_special_characters_are_escaped(self) -> None:
        """A system id with an underscore must not break the build."""
        table = EfficiencyTable()
        table.add(SystemEfficiency(system_id="A3_F3", measured=False, blocker="none & nothing"))
        latex = table.to_latex()
        assert "A3\\_F3" in latex
        assert "none \\& nothing" in latex

    def test_guard_table_reports_both_columns_and_the_ratio(self) -> None:
        """Guard-on and guard-off are two measurements and the table shows both."""
        latex = self._table().guard_table_to_latex()
        assert "0.010" in latex
        assert "0.040" in latex
        assert "4.00" in latex

    def test_guard_caption_says_generations_are_excluded(self) -> None:
        """The measured overhead is verification only, and the caption must not overclaim."""
        latex = self._table().guard_table_to_latex()
        assert "NOT" in latex
        assert "verification" in latex.lower()

    def test_guard_table_states_its_own_absence(self) -> None:
        """An empty table is a statement, not a missing file."""
        table = EfficiencyTable()
        table.add(SystemEfficiency(system_id="S1", measured=False, blocker="x"))
        assert "No system has both measurements" in table.guard_table_to_latex()

    def test_size_table_renders_every_band_including_empty_ones(self) -> None:
        """An unpopulated band is reported as n=0 rather than dropped."""
        latex = self._table().node_bin_table_to_latex()
        for node_bin in DEFAULT_NODE_BINS:
            assert node_bin.label in latex

    def test_json_round_trips_through_disk(self, tmp_path: Path) -> None:
        """The stored table carries its rows, its hardware and its coverage."""
        import json

        path = self._table().write_json(tmp_path / "efficiency.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["coverage"]["n_measured"] == 1
        assert payload["hardware"]["gpu_name"] == "Test GPU"
        assert {r["system_id"] for r in payload["rows"]} == {"B1", "S1"}


class TestHardwareCapture:
    """A measurement without its hardware is not a measurement."""

    def test_capture_reads_the_machine_rather_than_a_constant(self) -> None:
        """Every field is read; a hand-recorded config disagrees with its own machine."""
        hardware = capture_hardware()
        assert hardware.cpu_count > 0
        assert hardware.python_version.startswith("3.")
        assert hardware.platform

    def test_describe_is_a_single_caption_line(self) -> None:
        """The string a table caption embeds."""
        described = capture_hardware().describe()
        assert "RAM" in described
        assert "Python" in described

    def test_peak_memory_on_cpu_yields_without_cuda_keys(self) -> None:
        """A CPU benchmark runs unchanged rather than branching at every call site."""
        with measure_peak_memory("cpu") as stats:
            pass
        assert "allocated_gb" not in stats
