"""Phase 13: efficiency, footprint and cost — the measurements the deployment argument rests on.

The strongest answer to "why not just prompt a frontier model?" is not a quality argument.
It is that a bank cannot send customer transaction data to a third-party endpoint, and an
8B model running inside the institution's own perimeter is categorically different from an
API call regardless of what either scores. This module produces the numbers that make that
difference quantitative rather than rhetorical.

Five things here are decisions rather than descriptions, each documented at its definition:

- **Latency is end-to-end or it is not reported** (:class:`Stage`). The measured path is
  graph load → case extraction → fact extraction → encoding → generation → guard
  verification. Timing only the decoder describes a component, not a system, and it
  flatters us: the fact layer and the guard are ours and they cost real milliseconds.
  :meth:`EndToEndTimer.total` sums the per-narrative stages and excludes the
  once-per-process graph load, which is carried by the cold-start figure instead.
- **Guard-on and guard-off are two measurements, never one** (:class:`SystemEfficiency`).
  The guard samples four candidates and verifies each, so it is roughly 4x generation plus
  verification. That trade-off is the thing a deployment reader wants priced.
- **Peak VRAM is reported as allocated *and* reserved** (:class:`MemoryProfile`), because
  the reserved figure is the one that decides whether a run fits on a given card, and the
  allocated figure is the one that is a property of the model rather than of the allocator.
- **Percentiles are nearest-rank over the observed samples, never interpolated**
  (:func:`percentile`). p95 is used for capacity planning, and a capacity planner wants a
  latency that actually happened.
- **Cost is a declared model, not a number** (:class:`CostAssumptions`). Every input — the
  capital figure, the depreciation life, the utilisation factor, the power draw, the
  electricity rate — is a field with a source, and :meth:`CostAssumptions.hourly_cost`
  shows its working. Comparing a local system's amortised cost against an API's marginal
  price without stating the amortisation is the standard way this table misleads.

An unmeasured row is a first-class value, not a gap. :class:`SystemEfficiency` carries
``measured`` and ``blocker``, the LaTeX writer renders an unmeasured cell as ``--`` with a
footnote, and :meth:`EfficiencyTable.coverage` reports how much of the matrix is real.
Invariant 7: absences are a deliverable.

Nothing at module scope imports torch. This module is read by the aggregator, the figure
code and the deployability document on machines that have no accelerator; the accelerator
paths are imported inside the two functions that need them.
"""

from __future__ import annotations

import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from g2t_aml.utils.io import write_json

__all__ = [
    "DEFAULT_NODE_BINS",
    "DEFAULT_N_MEASURED",
    "DEFAULT_N_WARMUP",
    "INTERACTIVE_BATCH",
    "THROUGHPUT_BATCH",
    "BenchmarkSample",
    "CostAssumptions",
    "CostEstimate",
    "DeploymentProfile",
    "EfficiencyTable",
    "EndToEndTimer",
    "HardwareConfig",
    "LatencySummary",
    "MemoryProfile",
    "ModelFootprint",
    "NodeBin",
    "Stage",
    "SystemEfficiency",
    "capture_hardware",
    "count_parameters",
    "directory_size_bytes",
    "measure_peak_memory",
    "percentile",
    "run_benchmark",
    "summarise_by_node_bin",
]


# ------------------------------------------------------------------ the protocol ---

#: Discarded before measurement begins. The first calls pay for import-time module
#: caching, lazy Polars kernel compilation, CUDA context creation and the allocator's
#: first arena. Including them reports a warm system's latency as if it were cold, which
#: is wrong in both directions: it inflates the mean and it hides the genuine cold-start
#: cost, which is measured separately and deliberately (:func:`measure_cold_start`).
DEFAULT_N_WARMUP = 20

#: Measured runs per system per configuration. A hundred is the floor at which a p95 is a
#: measurement rather than an anecdote: it is the 95th of 100 observations, so it is an
#: observed sample and not an extrapolation. p99 at n=100 is the single worst-but-one run
#: and is reported with that caveat attached rather than silently.
DEFAULT_N_MEASURED = 100

#: Interactive: one alert, one investigator waiting. This is the number a user feels.
INTERACTIVE_BATCH = 1

#: Batch processing: an overnight queue. This is the number that sizes a nightly window.
THROUGHPUT_BATCH = 32


class Stage(StrEnum):
    """One stage of the end-to-end path, timed separately and summed for the total.

    The stage list is the argument. A system latency that omits ``FACT_EXTRACTION`` and
    ``GUARD`` is reporting the decoder's speed under the system's name, and those two
    stages are ours — they are where the faithfulness comes from, and they are not free.

    Attributes:
        GRAPH_LOAD: Materialising the substrate graph and its traversal index. Paid once
            per process, not once per narrative; carried in the cold-start figure and
            excluded from the per-narrative total. See :meth:`EndToEndTimer.total`.
        CASE_EXTRACTION: Cutting the case subgraph around the seed account.
        FACT_EXTRACTION: Building the checkable fact record.
        SERIALISATION: Rendering the record into the model's textual context.
        ENCODING: The graph encoder forward pass and the fusion projection.
        GENERATION: Decoding the narrative.
        GUARD: Candidate verification, selection and any repair.
    """

    GRAPH_LOAD = "graph_load"
    CASE_EXTRACTION = "case_extraction"
    FACT_EXTRACTION = "fact_extraction"
    SERIALISATION = "serialisation"
    ENCODING = "encoding"
    GENERATION = "generation"
    GUARD = "guard"


#: The stages that are paid once per narrative. ``GRAPH_LOAD`` is not among them: the
#: index is built once and serves every case in the process, so charging it to each
#: narrative would multiply a one-off cost by the corpus size. It is reported as
#: cold start instead, which is what it is.
PER_NARRATIVE_STAGES: frozenset[Stage] = frozenset(
    {
        Stage.CASE_EXTRACTION,
        Stage.FACT_EXTRACTION,
        Stage.SERIALISATION,
        Stage.ENCODING,
        Stage.GENERATION,
        Stage.GUARD,
    }
)


@dataclass(frozen=True)
class NodeBin:
    """A case-size band, because a 150-node case does not cost what a 20-node case costs.

    Reporting one mean over a corpus whose case sizes span an order of magnitude describes
    no case in it. The bins are closed-open on node count.

    Attributes:
        low: Inclusive lower bound on node count.
        high: Exclusive upper bound, or 0 for unbounded.
    """

    low: int
    high: int

    def contains(self, n_nodes: int) -> bool:
        """Report whether a case falls in this bin.

        Args:
            n_nodes: The case's node count.

        Returns:
            True when ``low <= n_nodes < high``, treating ``high == 0`` as unbounded.
        """
        if n_nodes < self.low:
            return False
        return True if self.high == 0 else n_nodes < self.high

    @property
    def label(self) -> str:
        """Return the bin as a human-readable range.

        Returns:
            ``"25-49"``, or ``"100+"`` for the unbounded bin.
        """
        return f"{self.low}+" if self.high == 0 else f"{self.low}-{self.high - 1}"


#: Chosen against this corpus's actual distribution rather than round numbers: the case
#: budget is n_max=150 (Phase 2), and the mass sits below 50 nodes. Four bins keeps every
#: bin populated at n=100, which a finer grid would not.
DEFAULT_NODE_BINS: tuple[NodeBin, ...] = (
    NodeBin(0, 25),
    NodeBin(25, 50),
    NodeBin(50, 100),
    NodeBin(100, 0),
)


# ------------------------------------------------------------------ the hardware ---


@dataclass(frozen=True)
class HardwareConfig:
    """Exactly what the numbers were measured on.

    A latency without its hardware is not a measurement, and "an NVIDIA GPU" is not a
    hardware description. Every field here is read from the machine rather than typed in,
    because a hand-recorded configuration is one that will disagree with the machine it
    claims to describe.

    Attributes:
        gpu_name: The card, or None when no accelerator is present.
        gpu_vram_gb: Total device memory.
        gpu_driver: NVIDIA driver version.
        cuda_runtime: The CUDA runtime torch was built against.
        cuda_capability: Compute capability, which decides whether bf16 and
            ``flash_attention_2`` are available at all.
        torch_version: The torch build.
        cpu_model: The host processor.
        cpu_count: Logical cores.
        ram_gb: Total system memory. On this project it is a binding constraint, not a
            footnote: it is what closes CPU offload.
        platform: OS string.
        python_version: Interpreter version.
    """

    gpu_name: str | None
    gpu_vram_gb: float | None
    gpu_driver: str | None
    cuda_runtime: str | None
    cuda_capability: str | None
    torch_version: str | None
    cpu_model: str
    cpu_count: int
    ram_gb: float
    platform: str
    python_version: str

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a mapping.

        Returns:
            Every field.
        """
        return asdict(self)

    def describe(self) -> str:
        """Return a one-line description for a table caption.

        Returns:
            A compact string naming the accelerator, its memory, the host and the driver.
        """
        gpu = (
            f"{self.gpu_name} ({self.gpu_vram_gb:.0f} GB)"
            if self.gpu_name and self.gpu_vram_gb
            else "no accelerator"
        )
        return (
            f"{gpu}, driver {self.gpu_driver or 'n/a'}, CUDA {self.cuda_runtime or 'n/a'}, "
            f"torch {self.torch_version or 'n/a'}, {self.cpu_model} x{self.cpu_count}, "
            f"{self.ram_gb:.0f} GB RAM, Python {self.python_version}"
        )


def _nvidia_smi(query: str) -> str | None:
    """Read one field from ``nvidia-smi``.

    Args:
        query: The ``--query-gpu`` field name.

    Returns:
        The first GPU's value, or None when the tool is absent or fails. A missing
        accelerator is a normal state on this project, not an error.
    """
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        out = subprocess.run(
            [binary, f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    first = out.stdout.strip().splitlines()
    return first[0].strip() if first else None


def _cpu_model() -> str:
    """Read the host processor's model name.

    Returns:
        The model string, or :func:`platform.processor`'s answer when ``/proc/cpuinfo``
        is unavailable or does not carry one.
    """
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            for line in cpuinfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _ram_gb() -> float:
    """Read total system memory in GB.

    Returns:
        Total RAM, or 0.0 when it cannot be determined.
    """
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 0.0
    return float(pages) * float(page_size) / 1024**3


def capture_hardware() -> HardwareConfig:
    """Read the current machine's configuration.

    Every field is read from the machine. torch is imported inside this function so the
    module stays importable on a host that has no torch at all.

    Returns:
        The configuration, with the accelerator fields None when no CUDA device is
        visible.
    """
    torch_version: str | None = None
    cuda_runtime: str | None = None
    capability: str | None = None
    gpu_name: str | None = None
    vram: float | None = None
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_runtime = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            gpu_name = props.name
            vram = props.total_memory / 1024**3
            capability = f"{props.major}.{props.minor}"
    except (ImportError, RuntimeError):  # pragma: no cover -- no-torch hosts
        pass

    if gpu_name is None:
        gpu_name = _nvidia_smi("name")

    return HardwareConfig(
        gpu_name=gpu_name,
        gpu_vram_gb=vram,
        gpu_driver=_nvidia_smi("driver_version"),
        cuda_runtime=cuda_runtime,
        cuda_capability=capability,
        torch_version=torch_version,
        cpu_model=_cpu_model(),
        cpu_count=os.cpu_count() or 0,
        ram_gb=_ram_gb(),
        platform=platform.platform(),
        python_version=sys.version.split()[0],
    )


# ------------------------------------------------------------------ distributions ---


def percentile(samples: Sequence[float], q: float) -> float:
    """Return the nearest-rank percentile of a sample.

    **Nearest-rank, not interpolated.** An interpolated p95 is a number that never
    happened; capacity planning sizes a queue against a latency that did. The rank is
    ``ceil(q * n)`` on the sorted sample, clamped into range, which is the definition a
    reader can reproduce from the published raw samples.

    Args:
        samples: The observations. Need not be sorted.
        q: Quantile in ``[0, 1]``.

    Returns:
        The observation at the nearest rank, or 0.0 for an empty sample.

    Raises:
        ValueError: If ``q`` is outside ``[0, 1]``.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {q}")
    if not samples:
        return 0.0
    ordered = sorted(samples)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


@dataclass(frozen=True)
class LatencySummary:
    """A latency distribution, reported as a distribution.

    The mean alone is the number that hides the problem: a system whose mean is 0.4 s and
    whose p99 is 11 s needs different hardware from one whose mean is 0.4 s and whose p99
    is 0.5 s, and the two are indistinguishable in a table of means.

    Attributes:
        n: Measured runs, excluding warm-up.
        mean_s: Arithmetic mean, seconds.
        std_s: Sample standard deviation, or 0.0 at n < 2.
        min_s: Fastest observed run.
        p50_s: Median.
        p95_s: The capacity-planning figure.
        p99_s: The tail. At n=100 this is the second-worst observed run and should be read
            as such.
        max_s: Slowest observed run.
    """

    n: int
    mean_s: float
    std_s: float
    min_s: float
    p50_s: float
    p95_s: float
    p99_s: float
    max_s: float

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> LatencySummary:
        """Summarise a sample of per-run wall times.

        Args:
            samples: Seconds per run.

        Returns:
            The summary. An empty sample yields an all-zero summary at ``n = 0`` rather
            than raising, so a system that produced no runs still occupies a row.
        """
        if not samples:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return cls(
            n=len(samples),
            mean_s=statistics.fmean(samples),
            std_s=statistics.stdev(samples) if len(samples) > 1 else 0.0,
            min_s=min(samples),
            p50_s=percentile(samples, 0.50),
            p95_s=percentile(samples, 0.95),
            p99_s=percentile(samples, 0.99),
            max_s=max(samples),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as a mapping.

        Returns:
            Every field.
        """
        return asdict(self)


# ------------------------------------------------------------------ the timer ---


class EndToEndTimer:
    """Times the stages of one narrative and refuses to total a partial path.

    Used as a context manager per stage::

        timer = EndToEndTimer()
        with timer.stage(Stage.CASE_EXTRACTION):
            case = extract_case(...)
        with timer.stage(Stage.FACT_EXTRACTION):
            facts = extract_facts(case)

    A stage entered twice accumulates, which is what a guard that verifies four candidates
    needs.
    """

    def __init__(self) -> None:
        """Build an empty timer."""
        self._seconds: dict[Stage, float] = {}

    def stage(self, name: Stage) -> Any:
        """Time a block and attribute it to a stage.

        Args:
            name: The stage.

        Returns:
            A context manager. Re-entering an already-timed stage accumulates rather than
            replacing, so repeated work inside one narrative is counted once in total.
        """

        @contextmanager
        def _timed() -> Iterator[None]:
            started = time.perf_counter()
            try:
                yield
            finally:
                elapsed = time.perf_counter() - started
                self._seconds[name] = self._seconds.get(name, 0.0) + elapsed

        return _timed()

    def record(self, name: Stage, seconds: float) -> None:
        """Attribute a measured duration to a stage directly.

        For a stage timed elsewhere — a CUDA-synchronised region, or a call that already
        returns its own wall time.

        Args:
            name: The stage.
            seconds: Duration to add.
        """
        self._seconds[name] = self._seconds.get(name, 0.0) + seconds

    @property
    def stages(self) -> dict[str, float]:
        """Return the per-stage seconds recorded so far.

        Returns:
            Stage name to seconds, for the stages that were entered.
        """
        return {str(k): v for k, v in self._seconds.items()}

    def total(self) -> float:
        """Return the per-narrative total.

        ``GRAPH_LOAD`` is excluded: it is a per-process cost carried by the cold-start
        figure, and charging it to every narrative would multiply a one-off by the corpus
        size.

        Returns:
            The sum over :data:`PER_NARRATIVE_STAGES`.
        """
        return sum(v for k, v in self._seconds.items() if k in PER_NARRATIVE_STAGES)


@dataclass(frozen=True)
class BenchmarkSample:
    """One measured run, with the case size that explains it.

    Attributes:
        case_id: The case, so a tail run can be inspected rather than wondered about.
        n_nodes: Case node count. The binning axis.
        n_edges: Case edge count.
        seconds: End-to-end wall time for this narrative.
        stage_seconds: Per-stage breakdown.
        n_output_tokens: Tokens produced, for the tokens/sec figure.
    """

    case_id: str
    n_nodes: int
    n_edges: int
    seconds: float
    stage_seconds: Mapping[str, float] = field(default_factory=dict)
    n_output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the sample as a mapping.

        Returns:
            Every field, with the stage breakdown as a plain dict.
        """
        return {
            "case_id": self.case_id,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "seconds": self.seconds,
            "stage_seconds": dict(self.stage_seconds),
            "n_output_tokens": self.n_output_tokens,
        }


def run_benchmark(
    measure: Callable[[str], BenchmarkSample],
    case_ids: Sequence[str],
    *,
    n_warmup: int = DEFAULT_N_WARMUP,
    n_measured: int = DEFAULT_N_MEASURED,
    on_error: Callable[[str, Exception], None] | None = None,
) -> list[BenchmarkSample]:
    """Run the protocol: warm up, discard, then measure.

    The warm-up runs are executed against the same callable and thrown away. They are not
    optional and they are not a formality: on this pipeline the first case pays for
    Polars' lazy kernel compilation and, on the GPU path, for CUDA context creation, and
    including them moves the mean by more than any architectural difference the paper
    reports.

    Args:
        measure: Called with a case id, returns one measured sample.
        case_ids: Cases to draw from. Cycled if shorter than ``n_warmup + n_measured``,
            which is recorded by the caller rather than hidden — a benchmark that reuses
            cases measures a warmer cache than one that does not.
        n_warmup: Runs to discard.
        n_measured: Runs to keep.
        on_error: Called with ``(case_id, exception)`` when a run raises. The run is
            skipped and measurement continues; a benchmark that aborts on one bad case
            reports nothing about the ninety-nine good ones.

    Returns:
        The measured samples, warm-up excluded. May be shorter than ``n_measured`` if runs
        failed; the caller reports the shortfall rather than padding it.

    Raises:
        ValueError: If ``case_ids`` is empty or either count is negative.
    """
    if not case_ids:
        raise ValueError("no cases to benchmark")
    if n_warmup < 0 or n_measured < 0:
        raise ValueError(f"counts must be non-negative, got {n_warmup=} {n_measured=}")

    def _draw(i: int) -> str:
        return case_ids[i % len(case_ids)]

    for i in range(n_warmup):
        try:
            measure(_draw(i))
        except Exception as exc:
            if on_error is not None:
                on_error(_draw(i), exc)

    samples: list[BenchmarkSample] = []
    for i in range(n_measured):
        case_id = _draw(n_warmup + i)
        try:
            samples.append(measure(case_id))
        except Exception as exc:
            if on_error is not None:
                on_error(case_id, exc)
    return samples


def summarise_by_node_bin(
    samples: Sequence[BenchmarkSample],
    bins: Sequence[NodeBin] = DEFAULT_NODE_BINS,
) -> dict[str, LatencySummary]:
    """Summarise latency separately in each case-size band.

    Args:
        samples: The measured samples.
        bins: The bands.

    Returns:
        Bin label to summary, including empty bins at ``n = 0`` — an empty band is
        information (this corpus produced no case that large), not something to omit.
    """
    out: dict[str, LatencySummary] = {}
    for node_bin in bins:
        seconds = [s.seconds for s in samples if node_bin.contains(s.n_nodes)]
        out[node_bin.label] = LatencySummary.from_samples(seconds)
    return out


def measure_cold_start(load: Callable[[], Any], *, n_repeats: int = 3) -> LatencySummary:
    """Measure model-load-to-ready, with no warm-up.

    Warm-up is deliberately absent: cold start is precisely the un-warmed cost, and a
    warmed cold-start measurement is a contradiction. The OS page cache still warms across
    repeats, which is stated with the number rather than corrected away — a service
    restarting on a host that has been running is the realistic case.

    Args:
        load: Performs the load and returns whatever it loads.
        n_repeats: How many times to load.

    Returns:
        The distribution over the repeats.
    """
    seconds: list[float] = []
    for _ in range(max(1, n_repeats)):
        started = time.perf_counter()
        load()
        seconds.append(time.perf_counter() - started)
    return LatencySummary.from_samples(seconds)


# ------------------------------------------------------------------ footprint ---


@dataclass(frozen=True)
class ModelFootprint:
    """Parameters and bytes, broken out by component.

    Reported per component because that is the question a deployment reader is actually
    asking: the base model is a one-time download shared across every arm, whereas the
    adapter, the encoder and the fusion projector are what this project ships. A single
    "16 GB" figure hides that the contribution is 40 MB of it.

    Attributes:
        total_params: Every parameter, trainable or frozen.
        trainable_params: Parameters that receive gradients. Under QLoRA this is the LoRA
            adapters plus the fp32 fusion projector plus, where jointly trained, the
            encoder.
        base_model_bytes: The quantised backbone on disk, or 0 where no backbone runs.
        adapter_bytes: LoRA adapter weights.
        encoder_bytes: The Phase 7 graph encoder checkpoint.
        fusion_bytes: The Phase 8 projector.
        notes: Anything a reader needs to read the numbers correctly.
    """

    total_params: int
    trainable_params: int
    base_model_bytes: int = 0
    adapter_bytes: int = 0
    encoder_bytes: int = 0
    fusion_bytes: int = 0
    notes: str = ""

    @property
    def total_bytes(self) -> int:
        """Return the total on-disk footprint.

        Returns:
            The sum of the four components.
        """
        return self.base_model_bytes + self.adapter_bytes + self.encoder_bytes + self.fusion_bytes

    @property
    def shipped_bytes(self) -> int:
        """Return the footprint excluding the base model.

        The base model is a public download an institution already has or fetches once.
        What this project distributes is the rest, and it is three orders of magnitude
        smaller — which is a deployment argument in itself.

        Returns:
            Adapter plus encoder plus fusion.
        """
        return self.adapter_bytes + self.encoder_bytes + self.fusion_bytes

    @property
    def trainable_fraction(self) -> float:
        """Return the fraction of parameters that are trained.

        Returns:
            Trainable over total, or 0.0 when there are no parameters.
        """
        return self.trainable_params / self.total_params if self.total_params else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return the footprint with its derived figures.

        Returns:
            Every field plus ``total_bytes``, ``shipped_bytes`` and
            ``trainable_fraction``.
        """
        return {
            **asdict(self),
            "total_bytes": self.total_bytes,
            "shipped_bytes": self.shipped_bytes,
            "trainable_fraction": self.trainable_fraction,
        }


def directory_size_bytes(path: str | Path) -> int:
    """Return the total size of a file, or of every file under a directory.

    Args:
        path: A file or directory. A missing path is 0 rather than an error, so a
            footprint can be assembled for a system whose checkpoint does not exist yet.

    Returns:
        Size in bytes.
    """
    target = Path(path)
    if target.is_file():
        return target.stat().st_size
    if not target.is_dir():
        return 0
    return sum(p.stat().st_size for p in target.rglob("*") if p.is_file())


def count_parameters(module: Any) -> tuple[int, int]:
    """Count a module's total and trainable parameters.

    Args:
        module: Anything exposing ``parameters()`` in the torch sense. Typed loosely
            because this module does not import torch.

    Returns:
        ``(total, trainable)``.
    """
    total = 0
    trainable = 0
    for param in module.parameters():
        n = int(param.numel())
        total += n
        if bool(param.requires_grad):
            trainable += n
    return total, trainable


@dataclass(frozen=True)
class MemoryProfile:
    """Peak device memory, allocated and reserved, for training and for inference.

    Both figures, always. ``max_memory_allocated`` is what the process asked for and is a
    property of the model; ``max_memory_reserved`` is what the caching allocator held and
    is what decides whether the job fits on a given card. Publishing only the first
    understates the hardware requirement, and publishing only the second describes the
    allocator as much as the model.

    Attributes:
        train_allocated_gb: Peak allocated during training, or None if not measured.
        train_reserved_gb: Peak reserved during training.
        inference_allocated_gb: Peak allocated during inference.
        inference_reserved_gb: Peak reserved during inference.
        host_ram_peak_gb: Peak resident set size, which is the binding constraint whenever
            CPU offload is in play.
        measured_on: The device the figures came from.
    """

    train_allocated_gb: float | None = None
    train_reserved_gb: float | None = None
    inference_allocated_gb: float | None = None
    inference_reserved_gb: float | None = None
    host_ram_peak_gb: float | None = None
    measured_on: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the profile as a mapping.

        Returns:
            Every field.
        """
        return asdict(self)


@contextmanager
def measure_peak_memory(device: str = "cuda") -> Iterator[dict[str, float]]:
    """Measure peak device memory over a block.

    torch is imported inside this function. On a host with no accelerator the yielded
    mapping is simply left without the CUDA keys, so a CPU benchmark runs unchanged rather
    than branching at every call site.

    Args:
        device: ``cuda``, ``cuda:N`` or ``cpu``.

    Yields:
        A mapping filled on exit with ``allocated_gb`` and ``reserved_gb`` when a CUDA
        device was used, and always with ``host_ram_peak_gb`` where the platform reports
        it.
    """
    stats: dict[str, float] = {}
    on_cuda = device.startswith("cuda")
    torch_mod: Any = None
    if on_cuda:
        try:
            import torch

            on_cuda = torch.cuda.is_available()
            torch_mod = torch
        except ImportError:  # pragma: no cover -- no-torch hosts
            on_cuda = False
    if on_cuda and torch_mod is not None:
        torch_mod.cuda.reset_peak_memory_stats()
        torch_mod.cuda.synchronize()
    try:
        yield stats
    finally:
        if on_cuda and torch_mod is not None:
            torch_mod.cuda.synchronize()
            stats["allocated_gb"] = torch_mod.cuda.max_memory_allocated() / 1024**3
            stats["reserved_gb"] = torch_mod.cuda.max_memory_reserved() / 1024**3
        try:
            import resource

            # ru_maxrss is kilobytes on Linux, bytes on macOS. Linux is what this project
            # runs on and what the recorded hardware says; the platform check keeps a
            # macOS reading from being a thousand times too large.
            divisor = 1024**2 if sys.platform == "linux" else 1024**3
            stats["host_ram_peak_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor
        except (ImportError, OSError):  # pragma: no cover -- non-POSIX hosts
            pass


# ------------------------------------------------------------------ the cost model ---


@dataclass(frozen=True)
class CostAssumptions:
    """The amortisation model, declared in full so the comparison can be argued with.

    **Comparing a local system's amortised cost against an API's marginal price without
    stating the amortisation is the standard way this table misleads.** An API price is
    marginal: one more narrative costs one more narrative's tokens, and zero narratives
    cost nothing. A local cost is capital already spent, and it is only competitive at
    volume. Both are reported here, both carry their assumptions, and
    :meth:`CostEstimate.breakeven_narratives_per_month` states where the crossover is
    instead of leaving a reader to assume it is anywhere in particular.

    Every figure is a list-price input a reader can substitute. None is a claim about what
    any particular institution pays: hardware is bought at negotiated prices, electricity
    is regional, and an institution with an existing GPU estate has already sunk the
    capital.

    Attributes:
        hardware_capital_usd: Purchase price of the accelerator and its host.
        depreciation_years: Straight-line life. Three years is the common convention for
            GPU compute and is what the figure below assumes.
        utilisation: Fraction of wall-clock hours the hardware is actually serving. A
            dedicated box at 100% is not a realistic assumption for a batch AML workload,
            and assuming it would divide the hourly cost by roughly four.
        power_draw_w: Sustained system draw under load, accelerator plus host.
        pue: Data-centre power usage effectiveness — the multiplier for cooling and
            distribution. 1.5 is a common enterprise figure; a modern hyperscale facility
            is nearer 1.1 and an older server room is worse.
        electricity_usd_per_kwh: Commercial rate.
        engineer_usd_per_hour: Loaded rate, used only for the training-cost figure where
            a run needs supervision. Zero by default: an unattended run costs no labour,
            and including a notional figure would inflate a number that is already an
            estimate.
        api_input_usd_per_mtok: Hosted-model input price per million tokens.
        api_output_usd_per_mtok: Hosted-model output price per million tokens.
        source: Where the figures came from, recorded so a stale price is visible as
            stale.
    """

    hardware_capital_usd: float = 0.0
    depreciation_years: float = 3.0
    utilisation: float = 0.5
    power_draw_w: float = 0.0
    pue: float = 1.5
    electricity_usd_per_kwh: float = 0.12
    engineer_usd_per_hour: float = 0.0
    api_input_usd_per_mtok: float = 0.0
    api_output_usd_per_mtok: float = 0.0
    source: str = ""

    @property
    def amortised_usd_per_hour(self) -> float:
        """Return the capital cost charged per serving hour.

        Straight-line over the depreciation life, divided by the hours the hardware is
        actually serving rather than by the hours in the period. At 50% utilisation a box
        costs twice as much per useful hour as the naive division suggests, and that
        factor is the single most misreportable number in this table.

        Returns:
            USD per serving hour, or 0.0 when no capital was declared.
        """
        if self.hardware_capital_usd <= 0 or self.depreciation_years <= 0:
            return 0.0
        serving_hours = 365.25 * 24.0 * self.depreciation_years * max(self.utilisation, 1e-9)
        return self.hardware_capital_usd / serving_hours

    @property
    def power_usd_per_hour(self) -> float:
        """Return the energy cost per serving hour, including cooling.

        Returns:
            USD per hour of draw at the declared rate and PUE.
        """
        return (self.power_draw_w / 1000.0) * self.pue * self.electricity_usd_per_kwh

    def hourly_cost(self) -> float:
        """Return the total cost of one serving hour.

        Returns:
            Amortised capital plus power.
        """
        return self.amortised_usd_per_hour + self.power_usd_per_hour

    def to_dict(self) -> dict[str, Any]:
        """Return the assumptions with the derived hourly figures.

        Returns:
            Every field plus the three derived rates, so a stored table carries its own
            working and a reader never has to recompute it to check it.
        """
        return {
            **asdict(self),
            "amortised_usd_per_hour": self.amortised_usd_per_hour,
            "power_usd_per_hour": self.power_usd_per_hour,
            "hourly_cost_usd": self.hourly_cost(),
        }


@dataclass(frozen=True)
class CostEstimate:
    """What a thousand narratives cost, and what training cost.

    Attributes:
        usd_per_1000: Marginal cost of a thousand narratives under the declared model.
        basis: ``amortised_local`` or ``api_marginal``. The two are not the same kind of
            number and the field exists so a table never implies they are.
        training_usd: GPU-hours times rate for this system's training run, or 0.0 for a
            system that trains nothing.
        training_gpu_hours: The hours themselves, reported separately because the rate is
            the contestable part and the hours are not.
        assumptions: The model that produced the figures.
    """

    usd_per_1000: float
    basis: str
    training_usd: float = 0.0
    training_gpu_hours: float = 0.0
    assumptions: CostAssumptions = field(default_factory=CostAssumptions)

    def breakeven_narratives_per_month(self, api_usd_per_1000: float) -> float | None:
        """Return the monthly volume at which local serving costs less than the API.

        Args:
            api_usd_per_1000: The hosted alternative's marginal cost per thousand.

        Returns:
            Narratives per month at the crossover, or None when this estimate is not a
            local one or the API is cheaper at every volume. The capital is charged
            monthly at the declared amortisation; below the crossover the API is cheaper
            and the table should not pretend otherwise.
        """
        if self.basis != "amortised_local" or api_usd_per_1000 <= 0:
            return None
        monthly_capital = (
            self.assumptions.amortised_usd_per_hour * 730.0 * self.assumptions.utilisation
        )
        marginal_gap = api_usd_per_1000 - self.usd_per_1000
        if marginal_gap <= 0:
            return None
        return 1000.0 * monthly_capital / marginal_gap

    def to_dict(self) -> dict[str, Any]:
        """Return the estimate as a mapping.

        Returns:
            Every field, with the assumptions expanded.
        """
        return {
            "usd_per_1000": self.usd_per_1000,
            "basis": self.basis,
            "training_usd": self.training_usd,
            "training_gpu_hours": self.training_gpu_hours,
            "assumptions": self.assumptions.to_dict(),
        }


def local_cost_per_1000(
    narratives_per_second: float,
    assumptions: CostAssumptions,
    *,
    training_gpu_hours: float = 0.0,
    training_usd_per_gpu_hour: float = 0.0,
) -> CostEstimate:
    """Cost a thousand narratives on owned hardware.

    Args:
        narratives_per_second: Measured throughput.
        assumptions: The amortisation model.
        training_gpu_hours: Hours the training run consumed.
        training_usd_per_gpu_hour: Rate for the training hours. Kept separate from the
            serving assumptions because training is commonly rented and serving owned, and
            conflating the two rates is a category error.

    Returns:
        The estimate, at basis ``amortised_local``. Zero throughput yields an infinite
        cost rather than a division error, which is the honest reading: a system that
        produces nothing has no cost per thousand.
    """
    if narratives_per_second <= 0:
        per_1000 = math.inf
    else:
        hours_per_1000 = 1000.0 / (narratives_per_second * 3600.0)
        per_1000 = hours_per_1000 * assumptions.hourly_cost()
    return CostEstimate(
        usd_per_1000=per_1000,
        basis="amortised_local",
        training_usd=training_gpu_hours * training_usd_per_gpu_hour,
        training_gpu_hours=training_gpu_hours,
        assumptions=assumptions,
    )


def api_cost_per_1000(
    input_tokens: float,
    output_tokens: float,
    assumptions: CostAssumptions,
    *,
    calls_per_narrative: float = 1.0,
) -> CostEstimate:
    """Cost a thousand narratives against a hosted endpoint's published pricing.

    Args:
        input_tokens: Mean prompt tokens per call.
        output_tokens: Mean completion tokens per call.
        assumptions: Carries the per-million prices.
        calls_per_narrative: Calls each narrative costs. The agentic comparator makes
            several, and charging it for one would price it as something it is not.

    Returns:
        The estimate, at basis ``api_marginal``.
    """
    per_call = (
        input_tokens * assumptions.api_input_usd_per_mtok
        + output_tokens * assumptions.api_output_usd_per_mtok
    ) / 1_000_000.0
    return CostEstimate(
        usd_per_1000=per_call * calls_per_narrative * 1000.0,
        basis="api_marginal",
        assumptions=assumptions,
    )


# ------------------------------------------------------------------ deployability ---


@dataclass(frozen=True)
class DeploymentProfile:
    """Whether a system can run inside an institution's perimeter, and on what.

    **Factual, never legal.** This records what a system does with data and what hardware
    it needs. It does not assert what any institution may or may not lawfully do: that is
    a question for the institution's counsel, it varies by jurisdiction and by contract,
    and a paper that asserts it will be right about some readers and wrong about others.
    ``regulatory_context`` describes the landscape and cites it; ``on_premise`` is a
    statement about the software.

    Attributes:
        on_premise: Whether every component can run inside the institution's network with
            no outbound call at inference time. This is a property of the system, and it
            is the column the deployment argument turns on.
        data_leaves_perimeter: What, if anything, is transmitted outside. ``"nothing"``
            for a local system; for a hosted one, named plainly.
        min_viable_hardware: The cheapest configuration that runs it at all, which is not
            the configuration it was measured on.
        recommended_hardware: What a mid-size institution would actually buy for the
            stated volume.
        regulatory_context: The constraints a deployment reader would weigh, described and
            sourced rather than concluded.
        notes: Anything else material.
    """

    on_premise: bool
    data_leaves_perimeter: str
    min_viable_hardware: str
    recommended_hardware: str = ""
    regulatory_context: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the profile as a mapping.

        Returns:
            Every field.
        """
        return asdict(self)


# ------------------------------------------------------------------ the row ---


@dataclass(frozen=True)
class SystemEfficiency:
    """One row of the efficiency table.

    ``measured`` is the field that keeps this table honest. A row assembled from an
    analytic estimate, a vendor figure or a published benchmark is not a measurement of
    this system, and the LaTeX writer renders it as absent rather than as a number a
    reader would take for one.

    Attributes:
        system_id: The registry id, so this table joins to the results table on one key.
        role: What the system is in the paper's argument.
        measured: Whether the latency and memory figures were measured on the recorded
            hardware by this project.
        blocker: Why not, when ``measured`` is False. Required in that case.
        footprint: Parameters and bytes.
        memory: Peak device and host memory.
        latency_guard_off: End-to-end distribution with the guard disabled.
        latency_guard_on: End-to-end distribution with the guard enabled. Separate row,
            never averaged with the above.
        latency_by_node_bin: Guard-off distribution per case-size band.
        cold_start: Process start to first token.
        narratives_per_second: Sustained throughput at the recorded batch size.
        tokens_per_second: Generation throughput.
        batch_size: Which batch size the throughput figures describe.
        cost: The cost estimate and its assumptions.
        deployment: The deployability assessment.
        stage_means: Mean seconds per stage, which is where a total is explained.
        n_runs: Measured runs behind the distributions.
        n_warmup: Runs discarded before measurement.
    """

    system_id: str
    role: str = ""
    measured: bool = False
    blocker: str = ""
    footprint: ModelFootprint | None = None
    memory: MemoryProfile | None = None
    latency_guard_off: LatencySummary | None = None
    latency_guard_on: LatencySummary | None = None
    latency_by_node_bin: Mapping[str, LatencySummary] = field(default_factory=dict)
    cold_start: LatencySummary | None = None
    narratives_per_second: float | None = None
    tokens_per_second: float | None = None
    batch_size: int = INTERACTIVE_BATCH
    cost: CostEstimate | None = None
    deployment: DeploymentProfile | None = None
    stage_means: Mapping[str, float] = field(default_factory=dict)
    n_runs: int = 0
    n_warmup: int = 0

    def __post_init__(self) -> None:
        """Refuse an unmeasured row that does not say why.

        Raises:
            ValueError: If ``measured`` is False and no blocker was given. An absence
                without its reason is the thing invariant 7 exists to prevent: it reads as
                an oversight, and six months later nobody can tell whether it was one.
        """
        if not self.measured and not self.blocker:
            raise ValueError(
                f"system {self.system_id!r} is unmeasured and carries no blocker; "
                "an absence must state its reason (invariant 7)"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the row as a JSON-serialisable mapping.

        Returns:
            Every field, with the nested dataclasses expanded and None preserved so a
            reader can tell an absent measurement from a zero.
        """
        return {
            "system_id": self.system_id,
            "role": self.role,
            "measured": self.measured,
            "blocker": self.blocker,
            "footprint": self.footprint.to_dict() if self.footprint else None,
            "memory": self.memory.to_dict() if self.memory else None,
            "latency_guard_off": (
                self.latency_guard_off.to_dict() if self.latency_guard_off else None
            ),
            "latency_guard_on": (
                self.latency_guard_on.to_dict() if self.latency_guard_on else None
            ),
            "latency_by_node_bin": {k: v.to_dict() for k, v in self.latency_by_node_bin.items()},
            "cold_start": self.cold_start.to_dict() if self.cold_start else None,
            "narratives_per_second": self.narratives_per_second,
            "tokens_per_second": self.tokens_per_second,
            "batch_size": self.batch_size,
            "cost": self.cost.to_dict() if self.cost else None,
            "deployment": self.deployment.to_dict() if self.deployment else None,
            "stage_means": dict(self.stage_means),
            "n_runs": self.n_runs,
            "n_warmup": self.n_warmup,
        }


# ------------------------------------------------------------------ the table ---

_LATEX_ESCAPES = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
}


def _tex(text: str) -> str:
    """Escape LaTeX special characters in a cell.

    Args:
        text: Raw cell text.

    Returns:
        The escaped text.
    """
    return "".join(_LATEX_ESCAPES.get(c, c) for c in text)


def _sig(value: float | None, digits: int = 3, *, absent: str = "--") -> str:
    """Format a number to a fixed number of significant figures.

    Used for the cost column, whose values span five orders of magnitude: a fixed number
    of *decimal* places renders USD 0.00053 and USD 0.00116 as the same cell, which makes
    two systems that differ by a factor of two look identical in a headline table.

    Args:
        value: The number, or None.
        digits: Significant figures.
        absent: What to print for None or a non-finite value.

    Returns:
        The formatted cell, with trailing zeros stripped from the fractional part.
    """
    if value is None or not math.isfinite(value):
        return absent
    if value == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(value)))
    decimals = max(0, digits - 1 - magnitude)
    text = f"{value:.{decimals}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _fmt(value: float | None, spec: str = ".2f", *, absent: str = "--") -> str:
    """Format a possibly-absent number for a table cell.

    Args:
        value: The number, or None.
        spec: Format specification.
        absent: What to print for None.

    Returns:
        The formatted cell. Infinity prints as ``--`` too: an infinite cost per thousand
        is the arithmetic of zero throughput, not a measurement.
    """
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return absent
    return format(value, spec)


@dataclass
class EfficiencyTable:
    """Every system's efficiency row, plus the hardware they were measured on.

    Attributes:
        rows: One per system, in registry order.
        hardware: The machine. One configuration for the whole table; a table whose rows
            were measured on different machines is not a comparison, and
            :meth:`add` does not police that — the caller runs one benchmark per host.
        protocol: How the benchmark was run, recorded alongside the numbers.
        generated_at: ISO timestamp.
    """

    rows: list[SystemEfficiency] = field(default_factory=list)
    hardware: HardwareConfig | None = None
    protocol: Mapping[str, Any] = field(default_factory=dict)
    generated_at: str = ""

    def add(self, row: SystemEfficiency) -> None:
        """Append a row.

        Args:
            row: The system's efficiency record.
        """
        self.rows.append(row)

    def coverage(self) -> dict[str, int]:
        """Report how much of the table is measured.

        Returns:
            ``n_systems``, ``n_measured`` and ``n_blocked``. Printed at the head of the
            table and into PHASE_LOG, because "the efficiency table" and "the efficiency
            table, two rows of seventeen of which are measurements" are different claims.
        """
        measured = sum(1 for r in self.rows if r.measured)
        return {
            "n_systems": len(self.rows),
            "n_measured": measured,
            "n_blocked": len(self.rows) - measured,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the whole table as a JSON-serialisable mapping.

        Returns:
            The rows, the hardware, the protocol and the coverage summary.
        """
        return {
            "generated_at": self.generated_at,
            "hardware": self.hardware.to_dict() if self.hardware else None,
            "protocol": dict(self.protocol),
            "coverage": self.coverage(),
            "rows": [r.to_dict() for r in self.rows],
        }

    def write_json(self, path: str | Path) -> Path:
        """Write the table to disk atomically.

        Args:
            path: Destination JSON file.

        Returns:
            The path written.
        """
        return write_json(path, self.to_dict())

    def to_latex(self, *, caption: str = "", label: str = "tab:efficiency") -> str:
        """Render the main efficiency table as LaTeX.

        Unmeasured cells render as ``--`` and the row carries a superscript marker keyed
        to a blocker note beneath the table. That is deliberate: a reader scanning the
        table sees which numbers exist, and a reader who wants to know why finds the
        reason on the same page rather than in a repository.

        Args:
            caption: Table caption. The hardware line is appended to it automatically,
                because a table of latencies whose caption does not name the machine is a
                table a reviewer cannot use.
            label: LaTeX label.

        Returns:
            A complete ``table`` environment.
        """
        header = (
            "System & Role & Params (M) & Train. (M) & VRAM (GB) & "
            "p50 (s) & p95 (s) & Narr./s & USD/1k & On-prem \\\\"
        )
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{llrrrrrrrc}",
            "\\toprule",
            header,
            "\\midrule",
        ]

        blockers: list[tuple[str, str]] = []
        for row in self.rows:
            marker = ""
            if not row.measured:
                blockers.append((row.system_id, row.blocker))
                marker = f"\\textsuperscript{{{len(blockers)}}}"

            # A present footprint of zero parameters is a fact about a template system,
            # not a missing measurement, and it must not print as one. Only an absent
            # footprint is a dash. Parameters are in millions throughout so the template
            # rows and an 8B row share a scale a reader can compare down.
            fp = row.footprint
            params_m = fp.total_params / 1e6 if fp is not None else None
            train_m = fp.trainable_params / 1e6 if fp is not None else None
            vram = row.memory.inference_reserved_gb if row.memory else None
            p50 = row.latency_guard_off.p50_s if row.latency_guard_off else None
            p95 = row.latency_guard_off.p95_s if row.latency_guard_off else None
            on_prem = (
                ("\\checkmark" if row.deployment.on_premise else "$\\times$")
                if row.deployment
                else "--"
            )
            cells = [
                f"{_tex(row.system_id)}{marker}",
                _tex(row.role[:28]),
                _fmt(params_m, ".1f"),
                _fmt(train_m, ".1f"),
                _fmt(vram, ".2f"),
                _fmt(p50, ".3f"),
                _fmt(p95, ".3f"),
                _fmt(row.narratives_per_second, ".2f"),
                _sig(row.cost.usd_per_1000 if row.cost else None),
                on_prem,
            ]
            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")

        cov = self.coverage()
        full_caption = caption or "System efficiency and deployability."
        full_caption += (
            f" {cov['n_measured']} of {cov['n_systems']} rows are measured;"
            " a dash is an absent measurement, never a zero."
        )
        if self.hardware is not None:
            full_caption += f" Measured on: {_tex(self.hardware.describe())}."
        lines.append(f"\\caption{{{full_caption}}}")
        lines.append(f"\\label{{{label}}}")
        for i, (system_id, blocker) in enumerate(blockers, start=1):
            lines.append(
                f"\\par\\footnotesize\\textsuperscript{{{i}}}{_tex(system_id)}: " f"{_tex(blocker)}"
            )
        lines.append("\\end{table}")
        return "\n".join(lines) + "\n"

    def guard_table_to_latex(self, *, label: str = "tab:guard-cost") -> str:
        """Render the guard-on versus guard-off comparison.

        Its own table, because it is its own claim: what the guard costs is the trade-off
        a deployment reader is deciding about, and burying it as two adjacent columns in a
        ten-column table is how it stops being read.

        Args:
            label: LaTeX label.

        Returns:
            A complete ``table`` environment, or a stated absence when no system has both
            measurements.
        """
        pairs = [
            r
            for r in self.rows
            if r.latency_guard_off is not None and r.latency_guard_on is not None
        ]
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "System & p50 off (s) & p50 on (s) & p95 off (s) & p95 on (s) & Overhead \\\\",
            "\\midrule",
        ]
        for row in pairs:
            off = row.latency_guard_off
            on = row.latency_guard_on
            assert off is not None and on is not None
            overhead = on.p50_s / off.p50_s if off.p50_s > 0 else None
            lines.append(
                " & ".join(
                    [
                        _tex(row.system_id),
                        _fmt(off.p50_s, ".3f"),
                        _fmt(on.p50_s, ".3f"),
                        _fmt(off.p95_s, ".3f"),
                        _fmt(on.p95_s, ".3f"),
                        _fmt(overhead, ".2f") + ("$\\times$" if overhead else ""),
                    ]
                )
                + " \\\\"
            )
        if not pairs:
            lines.append("\\multicolumn{6}{c}{No system has both measurements.} \\\\")
        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\caption{Cost of the inference guard's \\emph{verification} pass: claim "
                "extraction and the Phase~3 checker, run once per candidate, plus "
                "selection. The four \\emph{generations} the guard also requests are NOT "
                "included -- on the rows measured here generation is a template render, "
                "and on the rows where it is a decoder it is unmeasured. The full guard "
                "overhead is therefore this figure plus roughly three additional "
                "generations. Guarded and unguarded are reported as separate systems "
                "throughout (D-073).}",
                f"\\label{{{label}}}",
                "\\end{table}",
            ]
        )
        return "\n".join(lines) + "\n"

    def node_bin_table_to_latex(self, *, label: str = "tab:latency-by-size") -> str:
        """Render latency binned by case size.

        Args:
            label: LaTeX label.

        Returns:
            A complete ``table`` environment. Only measured systems appear: an unmeasured
            system has no distribution to bin.
        """
        bins = [b.label for b in DEFAULT_NODE_BINS]
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{l" + "rr" * len(bins) + "}",
            "\\toprule",
            "& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{b} nodes}}" for b in bins) + " \\\\",
            "System & " + " & ".join("$n$ & p50 (s)" for _ in bins) + " \\\\",
            "\\midrule",
        ]
        drawn = 0
        for row in self.rows:
            if not row.latency_by_node_bin:
                continue
            drawn += 1
            cells = [_tex(row.system_id)]
            for label_ in bins:
                summary = row.latency_by_node_bin.get(label_)
                cells.append(str(summary.n) if summary else "0")
                cells.append(_fmt(summary.p50_s if summary and summary.n else None, ".3f"))
            lines.append(" & ".join(cells) + " \\\\")
        if drawn == 0:
            lines.append(
                f"\\multicolumn{{{1 + 2 * len(bins)}}}{{c}}" "{No binned measurements.} \\\\"
            )
        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\caption{End-to-end latency by case size. A 150-node case does not cost "
                "what a 20-node case costs, and a single mean over the corpus describes "
                "neither.}",
                f"\\label{{{label}}}",
                "\\end{table}",
            ]
        )
        return "\n".join(lines) + "\n"
