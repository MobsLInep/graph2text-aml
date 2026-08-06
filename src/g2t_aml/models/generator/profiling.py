"""Peak VRAM, throughput and wall time — captured now because Phase 13 needs it and it is free.

Three numbers, recorded per arm and per phase (training, generation, guarded generation):
peak device memory, tokens per second, and seconds per epoch. They cost nothing to collect
while a run is happening and cannot be recovered afterwards, which is the whole argument
for doing it now rather than in Phase 13.

**Peak memory is read from the allocator, not from ``nvidia-smi``.**
``torch.cuda.max_memory_allocated`` reports what this process actually asked for, whereas
``nvidia-smi`` reports the caching allocator's reservation plus every other process on the
card. On a shared or desktop GPU the second number is not a property of the model at all.
Both are recorded — the reserved figure is what determines whether a run fits on a given
card, so it is the one a reader sizing hardware needs — but they are labelled distinctly
and never conflated.

**The guarded system is profiled separately.** It generates four candidates and sometimes a
fifth, so its throughput is roughly a quarter of the raw model's. Reporting one throughput
figure for "the system" would describe neither.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from g2t_aml.utils.io import write_json

__all__ = ["DeviceInfo", "PhaseProfile", "RunProfile", "device_info", "profile_phase"]


@dataclass(frozen=True)
class DeviceInfo:
    """What the numbers were measured on.

    A throughput figure without its hardware is not a measurement, and a VRAM figure
    without the card's capacity does not tell a reader whether it will fit on theirs.

    Attributes:
        name: Device name, or ``cpu``.
        total_memory_gb: Card capacity, or None on CPU.
        capability: CUDA compute capability, which determines whether
            ``flash_attention_2`` and bf16 are available at all.
        torch_version: The torch build.
        platform: OS string.
    """

    name: str
    total_memory_gb: float | None
    capability: str | None
    torch_version: str
    platform: str

    def to_dict(self) -> dict[str, Any]:
        """Return the device description as a mapping.

        Returns:
            The fields.
        """
        return asdict(self)


def device_info(device: str = "cuda") -> DeviceInfo:
    """Describe the device a run is measured on.

    Args:
        device: ``cuda``, ``cuda:N`` or ``cpu``.

    Returns:
        The description. Falls back to a CPU description when CUDA is unavailable, rather
        than raising, so a CPU smoke run still produces a complete profile.
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return DeviceInfo(
            name="cpu",
            total_memory_gb=None,
            capability=None,
            torch_version=torch.__version__,
            platform=platform.platform(),
        )
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return DeviceInfo(
        name=props.name,
        total_memory_gb=props.total_memory / 1024**3,
        capability=f"{props.major}.{props.minor}",
        torch_version=torch.__version__,
        platform=platform.platform(),
    )


@dataclass
class PhaseProfile:
    """One measured phase of a run.

    Attributes:
        phase: What was measured — ``train``, ``generate``, ``guarded_generate``.
        arm: Which system, so two arms' profiles cannot be confused.
        seconds: Wall time.
        n_tokens: Tokens processed, however the phase counts them: sequence tokens for
            training, generated tokens for inference.
        n_examples: Examples processed.
        peak_allocated_gb: Peak memory this process allocated.
        peak_reserved_gb: Peak the caching allocator reserved. **This is the number that
            determines whether the run fits on a given card**, and it is larger than the
            allocated figure — reporting only the smaller one understates the requirement.
        oom: Whether the phase hit an out-of-memory error.
        notes: Free-form, used to record a deviation such as a reduced sequence length.
    """

    phase: str
    arm: str
    seconds: float = 0.0
    n_tokens: int = 0
    n_examples: int = 0
    peak_allocated_gb: float | None = None
    peak_reserved_gb: float | None = None
    oom: bool = False
    notes: str = ""

    @property
    def tokens_per_second(self) -> float:
        """Return throughput in tokens per second.

        Returns:
            Tokens divided by wall time, or 0.0 when nothing was measured.
        """
        return self.n_tokens / self.seconds if self.seconds > 0 else 0.0

    @property
    def examples_per_second(self) -> float:
        """Return throughput in examples per second.

        Returns:
            Examples divided by wall time, or 0.0 when nothing was measured.
        """
        return self.n_examples / self.seconds if self.seconds > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return the profile with its derived rates.

        Returns:
            The fields plus ``tokens_per_second`` and ``examples_per_second``.
        """
        return {
            **asdict(self),
            "tokens_per_second": self.tokens_per_second,
            "examples_per_second": self.examples_per_second,
        }


@contextmanager
def profile_phase(
    phase: str, *, arm: str, device: str = "cuda", notes: str = ""
) -> Iterator[PhaseProfile]:
    """Measure wall time and peak memory over a block, and record an OOM if one happens.

    The profile object is yielded so the caller can fill in ``n_tokens`` and
    ``n_examples`` as it goes; the timing and memory fields are filled on exit, including
    on the exception path — a run that died of OOM at 14 GB is a data point Phase 13 wants,
    and a profiler that only records successful runs discards exactly the measurements a
    reader sizing hardware needs.

    Args:
        phase: What is being measured.
        arm: Which system.
        device: Device to read memory statistics from.
        notes: Free-form note, for recording a deviation.

    Yields:
        The profile, to be filled in by the caller.

    Raises:
        Exception: Re-raises whatever the block raised, after recording the profile.
    """
    on_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    profile = PhaseProfile(phase=phase, arm=arm, notes=notes)
    started = time.perf_counter()
    try:
        yield profile
    except torch.cuda.OutOfMemoryError:
        profile.oom = True
        raise
    finally:
        if on_cuda:
            torch.cuda.synchronize()
            profile.peak_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3
            profile.peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
        profile.seconds = time.perf_counter() - started


@dataclass
class RunProfile:
    """Every phase measured in one run, plus the hardware it ran on.

    Attributes:
        arm: Which system.
        device: The hardware description.
        phases: The measured phases, in order.
        deviations: Hyperparameter deviations forced by memory — a reduced
            ``max_seq_len``, a lower LoRA rank, fewer query tokens. **Recorded here and in
            PHASE_LOG.md**, because such a change affects comparability across arms and
            has to be applied uniformly or not at all.
    """

    arm: str
    device: DeviceInfo
    phases: list[PhaseProfile] = field(default_factory=list)
    deviations: list[str] = field(default_factory=list)

    def add(self, profile: PhaseProfile) -> None:
        """Record a measured phase.

        Args:
            profile: The phase profile.
        """
        self.phases.append(profile)

    def note_deviation(self, description: str) -> None:
        """Record a hyperparameter deviation forced by memory pressure.

        Args:
            description: What was changed and why, e.g. ``"max_seq_len 2048 -> 1536, OOM
                at batch size 2"``.
        """
        self.deviations.append(description)

    def to_dict(self) -> dict[str, Any]:
        """Return the whole profile as a JSON-serialisable mapping.

        Returns:
            The arm, the device, every phase and any deviations.
        """
        return {
            "arm": self.arm,
            "device": self.device.to_dict(),
            "phases": [p.to_dict() for p in self.phases],
            "deviations": list(self.deviations),
        }

    def write(self, path: str | Path) -> Path:
        """Write the profile to disk atomically.

        Args:
            path: Destination JSON file.

        Returns:
            The path written.
        """
        return write_json(path, self.to_dict())
