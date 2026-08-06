#!/usr/bin/env python
"""Build the Phase 12 study: the block design, the blind key, and the narrative pool.

This is the **only** place that holds the design, the blind key and the generated corpora
at the same time. It emits three files and the separation between them is the whole
blinding argument:

- ``design.json`` — who rates what, in what order. No system labels. Given to the interface.
- ``narratives.jsonl`` — item id to narrative text. No system labels. Given to the interface.
- ``key.json`` — item id to system. **Never given to the interface.** Read only by
  ``12b_analyse_study.py`` and ``12c_release_study.py``.

**Exits non-zero when fewer than two systems have generations**, because a study of one
system against itself is not a study and producing a design for one would waste a panel's
time before anyone noticed. As of 2026-08-05 only Bronze exists, so this script refuses on
this machine — which is the correct behaviour and is what the Phase 12 log records.

Usage:
    uv run python scripts/12_build_study.py
    uv run python scripts/12_build_study.py study.n_raters=10 study.items_per_rater=60
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from g2t_aml.human.reservation import load_reservation
from g2t_aml.human.study_design import DesignError, build_design
from g2t_aml.utils.io import atomic_path

#: Systems the study compares, in the order the paper's table uses. Bronze is the template
#: baseline every paired test is against.
DEFAULT_SYSTEMS = ("S1", "S2", "B7", "B3", "Bronze")

#: Fewer than this many arms with real generations and the comparison is not worth running.
MIN_SYSTEMS = 2

#: The Phase 12 gate's floor, stated in cases-per-system rather than in pool size. An
#: incomplete design leaves cells empty by construction, so the pool can be 100 cases while
#: an arm is only ever seen on 60 of them.
GATE_CASES_PER_SYSTEM = 80


def _load_generations(root: Path, system: str) -> dict[str, str]:
    """Read one system's narratives, keyed by case id.

    Args:
        root: Directory holding ``<system>.jsonl``, one object per line with ``case_id``
            and ``narrative``.
        system: The system id.

    Returns:
        Case id to narrative. Empty when the system has produced nothing.
    """
    path = root / f"{system}.jsonl"
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        narrative = str(payload.get("narrative", "")).strip()
        if narrative:
            out[str(payload["case_id"])] = narrative
    return out


#: Exit code captured out of the Hydra-decorated entrypoint. `@hydra.main` discards the
#: wrapped function's return value; see D-051.
_EXIT_CODE: list[int] = []


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def _run(cfg: DictConfig) -> None:
    """Build and write the study.

    Args:
        cfg: The composed Hydra config.

    Returns:
        Nothing. The exit code is recorded in :data:`_EXIT_CODE`: 2 when fewer than
        :data:`MIN_SYSTEMS` systems have generations, 3 when the parameters admit no valid
        design, 0 otherwise.
    """
    dataset = str(cfg.data.name)
    processed = Path(cfg.paths.processed_dir) / dataset
    generations_dir = processed / "generations"
    out_dir = Path(cfg.paths.artifacts_dir) / "human_study"
    out_dir.mkdir(parents=True, exist_ok=True)

    systems = list(getattr(cfg.get("study", {}), "systems", DEFAULT_SYSTEMS))
    n_raters = int(getattr(cfg.get("study", {}), "n_raters", 10))
    items_per_rater = int(getattr(cfg.get("study", {}), "items_per_rater", 60))
    n_cases = int(getattr(cfg.get("study", {}), "n_cases", 100))

    available: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for system in systems:
        generations = _load_generations(generations_dir, system)
        if generations:
            available[system] = generations
        else:
            missing.append(system)

    print(f"Systems with generations: {sorted(available) or 'NONE'}")
    if missing:
        print(f"Systems with NO generations: {missing}")

    if len(available) < MIN_SYSTEMS:
        print(
            f"\nREFUSING to build a study over {len(available)} system(s).\n"
            f"  Expected generations under {generations_dir}/<system>.jsonl\n"
            "  A comparison needs at least two arms; a design built over one would spend a\n"
            "  panel's time to compare a system against itself.\n"
            "  Blockers: S1/S2 need a GPU (D-068); B7/B3 and Silver need API credentials.",
            file=sys.stderr,
        )
        _EXIT_CODE.append(2)
        return

    # Cases are the Gold reservation intersected with what every available system covers.
    # Intersecting rather than unioning is deliberate: a case one arm cannot render is a
    # case whose row in the design would be structurally missing for that arm only, which
    # is an imbalance the design validator would reject and the analysis would misread.
    reserved = load_reservation(Path(cfg.paths.splits_dir) / dataset).case_ids
    covered = set.intersection(*(set(g) for g in available.values()))
    case_ids = sorted(c for c in reserved if c in covered)[:n_cases]
    print(f"Cases: {len(case_ids)} (reserved {len(reserved)}, covered by all arms {len(covered)})")

    try:
        design, key = build_design(
            case_ids,
            sorted(available),
            [f"rater-{i:02d}" for i in range(1, n_raters + 1)],
            dataset=dataset,
            items_per_rater=items_per_rater,
        )
    except DesignError as exc:
        print(f"\nCannot build a valid design: {exc}", file=sys.stderr)
        _EXIT_CODE.append(3)
        return

    design.write(out_dir / "design.json", key, out_dir / "key.json")

    pool: list[dict[str, Any]] = []
    for item in design.items:
        narrative = available[key.system_for(item.item_id)][item.case_id]
        pool.append({"item_id": item.item_id, "narrative": narrative})
    with atomic_path(out_dir / "narratives.jsonl") as tmp:
        tmp.write_text(
            "".join(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n" for p in pool),
            encoding="utf-8",
        )

    print()
    print(design.report.summary())
    print(f"\nWrote {out_dir}/design.json, narratives.jsonl")
    print(f"Wrote {out_dir}/key.json  <-- NEVER give this to the rating interface")

    floor = min(design.report.cases_per_system.values())
    if floor < GATE_CASES_PER_SYSTEM:
        print(
            f"\nWARNING: the least-covered arm reaches {floor} cases, under the Phase 12 "
            f"gate's {GATE_CASES_PER_SYSTEM}. Raise study.n_raters or study.items_per_rater.",
            file=sys.stderr,
        )


def main() -> int:
    """Run the Hydra entrypoint and return its exit code.

    ``@hydra.main`` discards its wrapped function's return value, so a code produced
    inside ``_run`` has to travel out through :data:`_EXIT_CODE` (D-051).

    Returns:
        The code ``_run`` produced, or 0 when it produced none.
    """
    _run()
    return _EXIT_CODE[-1] if _EXIT_CODE else 0


if __name__ == "__main__":
    sys.exit(main())
