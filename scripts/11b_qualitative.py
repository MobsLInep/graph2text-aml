#!/usr/bin/env python
"""Phase 11: the qualitative materials -- side-by-side cases, worst cases, and disagreements.

The qualitative section is often what makes an applications paper feel real, and it is
also the section most easily produced badly: a set of examples chosen after reading the
results illustrates the conclusion instead of testing it. Every selection rule here is
therefore **fixed before the numbers exist and mechanical**:

- **Side-by-side** uses the SAME ten cases for every system, and those ten are chosen by a
  stratified rule over typology plus a hash of the case id, never by hand.
- **Worst-case** ranks by contradicted claims, then critical findings, then Fact F1
  ascending -- the ordering ``eval.report.worst_cases`` already fixes, reused rather than
  restated.
- **Disagreements** are the cases where one system's Zero-Hallucination differs from
  another's, taken from both directions in equal number. Taking only the cases where our
  system wins is how an error analysis becomes a highlight reel.

Everything is written as both JSONL (for further analysis) and markdown (for reading), with
the fact record beside every narrative, because a narrative without its record cannot be
judged faithful or otherwise by a reader.

Usage:
    uv run python scripts/11b_qualitative.py
    uv run python scripts/11b_qualitative.py --treatment S1 --comparator B7 --n 10
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from g2t_aml.experiments.runner import COMPLETION_MARKER
from g2t_aml.facts.serialiser import serialise_facts
from g2t_aml.utils.io import read_json, read_jsonl, write_json, write_jsonl
from g2t_aml.utils.logging import configure_logging, get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]

#: How many cases the side-by-side section covers. Ten, from the brief.
N_SIDE_BY_SIDE = 10

#: How many worst cases per system.
N_WORST = 5

#: How many disagreements in each direction.
N_DISAGREEMENTS = 5


def _latest_run_dir(root: Path, system: str, seed: int) -> Path | None:
    """Find a system's most recently completed run directory at one seed.

    Args:
        root: The matrix root.
        system: The system id.
        seed: The seed.

    Returns:
        The directory, or None when the system has no completed run.
    """
    seed_dir = root / system / f"seed{seed}"
    if not seed_dir.is_dir():
        return None
    completed = [d for d in sorted(seed_dir.iterdir()) if (d / COMPLETION_MARKER).is_file()]
    if not completed:
        return None
    return max(completed, key=lambda d: (d / COMPLETION_MARKER).stat().st_mtime)


def _load_generations(root: Path, system: str, seed: int) -> dict[str, str]:
    """Read one system's narratives, keyed by case id.

    Args:
        root: The matrix root.
        system: The system id.
        seed: The seed.

    Returns:
        Case id to narrative. Empty when the system has not run.
    """
    directory = _latest_run_dir(root, system, seed)
    if directory is None:
        return {}
    path = directory / "generations.jsonl"
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for row in read_jsonl(path):
        if isinstance(row, dict) and "case_id" in row:
            text = row.get("narrative") or row.get("target_narrative") or ""
            out[str(row["case_id"])] = str(text)
    return out


def _load_scores(root: Path, system: str, seed: int, metric: str) -> dict[str, float]:
    """Read one system's per-case scores on a metric.

    Args:
        root: The matrix root.
        system: The system id.
        seed: The seed.
        metric: The metric.

    Returns:
        Case id to value. Empty when absent.
    """
    directory = _latest_run_dir(root, system, seed)
    if directory is None:
        return {}
    path = directory / "per_case.jsonl"
    if not path.is_file():
        return {}
    out: dict[str, float] = {}
    for row in read_jsonl(path):
        if isinstance(row, dict) and isinstance(row.get(metric), int | float):
            out[str(row["case_id"])] = float(row[metric])
    return out


def select_side_by_side(facts: Mapping[str, Any], n: int = N_SIDE_BY_SIDE) -> tuple[str, ...]:
    """Choose the cases every system is shown on, stratified by typology.

    **Fixed before any system is scored.** One case per typology in a stable order, then
    the remainder filled by a hash-ordered draw, so the ten cases are the same ten
    whichever systems happen to have run and whatever they scored.

    Args:
        facts: Case id to fact record.
        n: How many cases.

    Returns:
        The selected case ids.
    """
    by_typology: dict[str, list[str]] = {}
    for case_id, record in sorted(facts.items()):
        by_typology.setdefault(record.typology.label, []).append(case_id)

    selected: list[str] = []
    for typology in sorted(by_typology):
        candidates = sorted(by_typology[typology], key=_stable_key)
        if candidates:
            selected.append(candidates[0])
        if len(selected) >= n:
            break
    if len(selected) < n:
        remaining = sorted(set(facts) - set(selected), key=_stable_key)
        selected.extend(remaining[: n - len(selected)])
    return tuple(selected[:n])


def _stable_key(case_id: str) -> tuple[str, str]:
    """Return a process-independent ordering key for a case id.

    **Not** Python's ``hash``: string hashing is salted per process unless PYTHONHASHSEED
    is pinned, so a selection ordered by ``hash(case_id)`` would draw a different ten
    cases on every invocation. A rule that claims to be fixed before the numbers exist
    has to actually be fixed, and a per-process salt would make the qualitative section
    unreproducible in exactly the way that claim rules out.

    Args:
        case_id: The case.

    Returns:
        ``(digest, case_id)``, the second element breaking any digest collision.
    """
    from g2t_aml.utils.hashing import canonical_json, hash_id_list

    return (hash_id_list([case_id]), canonical_json(case_id))


def _render_side_by_side(
    case_ids: Sequence[str],
    facts: Mapping[str, Any],
    generations: Mapping[str, Mapping[str, str]],
) -> str:
    """Render the side-by-side markdown.

    Args:
        case_ids: The cases, in order.
        facts: Case id to fact record.
        generations: System id to case-id-to-narrative.

    Returns:
        The markdown document.
    """
    lines = [
        "# Side-by-side outputs",
        "",
        "The same cases for every system, selected by a stratified rule fixed before any",
        "system was scored. Each case shows its fact record first: a narrative without",
        "its record cannot be judged faithful or otherwise.",
        "",
    ]
    if not generations:
        lines += [
            "**No system has produced generations.** The selection rule and the rendering",
            "path are exercised; the outputs are not available.",
            "",
        ]
    for case_id in case_ids:
        record = facts.get(case_id)
        lines += [
            f"## Case `{case_id}`",
            "",
            f"**Typology:** {record.typology.label if record else 'unknown'}",
            "",
            "<details><summary>Fact record</summary>",
            "",
            "```",
            serialise_facts(record, style="verbose") if record else "(no record)",
            "```",
            "",
            "</details>",
            "",
        ]
        for system in sorted(generations):
            narrative = generations[system].get(case_id)
            lines += [
                f"### {system}",
                "",
                narrative if narrative else "_(not run)_",
                "",
            ]
    return "\n".join(lines)


def main() -> int:  # noqa: PLR0912, PLR0915 -- three numbered sections in one linear
    # pass; splitting them separates each artifact from the rule that selects it.
    """Produce every qualitative artifact.

    Returns:
        0 always. The materials are produced from whatever has run; an empty matrix
        yields the selection, the fact records and stated absences, which is the correct
        output at that point rather than a failure.
    """
    parser = argparse.ArgumentParser(description="Phase 11 qualitative materials.")
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT / "artifacts" / "matrix", help="matrix root"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "metrics" / "phase11" / "qualitative",
        help="output directory",
    )
    parser.add_argument(
        "--bronze",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "amlworld_hi_small" / "corpus" / "bronze.jsonl",
        help="Bronze corpus, for the fact records",
    )
    parser.add_argument("--treatment", default="S1", help="the arm the analysis centres on")
    parser.add_argument("--comparator", default="B7", help="the arm it is contrasted with")
    parser.add_argument("--seed", type=int, default=42, help="which seed's outputs to read")
    parser.add_argument("--n", type=int, default=N_SIDE_BY_SIDE, help="side-by-side cases")
    args = parser.parse_args()

    configure_logging()
    log = get_logger("qualitative")

    if not args.bronze.is_file():
        log.error("Bronze corpus not found at %s; run `make bronze` first", args.bronze)
        return 1

    from g2t_aml.corpus.factsio import facts_from_dict

    facts: dict[str, Any] = {}
    splits: dict[str, str] = {}
    for row in read_jsonl(args.bronze):
        if isinstance(row, dict):
            case_id = str(row["case_id"])
            facts[case_id] = facts_from_dict(dict(row["facts"]))
            splits[case_id] = str(row.get("split", "unknown"))
    test_facts = {c: f for c, f in facts.items() if splits.get(c) == "test"}
    log.info("%d test cases available", len(test_facts))

    from g2t_aml.experiments.registry import system_ids

    generations = {}
    for system in system_ids():
        rows = _load_generations(args.root, system, args.seed)
        if rows:
            generations[system] = rows
    log.info(
        "systems with generations at seed %d: %s",
        args.seed,
        ", ".join(sorted(generations)) or "(none)",
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Side by side, same ten cases for everybody.
    selected = select_side_by_side(test_facts, n=args.n)
    (out / "side_by_side.md").write_text(
        _render_side_by_side(selected, test_facts, generations), encoding="utf-8"
    )
    write_json(
        out / "side_by_side_selection.json",
        {
            "case_ids": list(selected),
            "rule": (
                "one case per typology in sorted order, then hash-ordered fill; fixed "
                "before any system was scored"
            ),
            "n_systems_with_output": len(generations),
        },
    )
    log.info("side_by_side -> %s (%d cases)", out / "side_by_side.md", len(selected))

    # 2. Worst cases per system, by the ordering eval.report.worst_cases fixes.
    worst_rows: list[dict[str, Any]] = []
    for system in sorted(generations):
        directory = _latest_run_dir(args.root, system, args.seed)
        metrics_path = directory / "metrics.json" if directory else None
        if metrics_path is None or not metrics_path.is_file():
            log.warning("%s has generations but no metrics.json; skipping worst cases", system)
            continue
        report = read_json(metrics_path)
        if not isinstance(report, dict):
            continue
        for key, block in (report.get("systems") or {}).items():
            if not isinstance(block, dict):
                continue
            for entry in (block.get("worst_cases") or [])[:N_WORST]:
                if isinstance(entry, dict):
                    worst_rows.append({"system": system, "slice": key, **entry})
    write_jsonl(out / "worst_cases.jsonl", worst_rows)
    log.info("worst_cases -> %s (%d rows)", out / "worst_cases.jsonl", len(worst_rows))

    # 3. Where the treatment beats the comparator and where it loses -- BOTH directions,
    #    in equal number. Taking only the wins is how an error analysis becomes a
    #    highlight reel, and it is the specific failure this rule exists to prevent.
    metric = "zero_hallucination"
    treatment = _load_scores(args.root, args.treatment, args.seed, metric)
    comparator = _load_scores(args.root, args.comparator, args.seed, metric)
    shared = sorted(set(treatment) & set(comparator))
    wins = [c for c in shared if treatment[c] > comparator[c]][:N_DISAGREEMENTS]
    losses = [c for c in shared if treatment[c] < comparator[c]][:N_DISAGREEMENTS]
    disagreements = [
        {
            "case_id": case_id,
            "direction": direction,
            "treatment": args.treatment,
            "comparator": args.comparator,
            "treatment_score": treatment[case_id],
            "comparator_score": comparator[case_id],
            "typology": test_facts[case_id].typology.label if case_id in test_facts else None,
            "treatment_narrative": generations.get(args.treatment, {}).get(case_id),
            "comparator_narrative": generations.get(args.comparator, {}).get(case_id),
            "serialised_facts": (
                serialise_facts(test_facts[case_id], style="verbose")
                if case_id in test_facts
                else None
            ),
        }
        for direction, group in (("treatment_wins", wins), ("comparator_wins", losses))
        for case_id in group
    ]
    write_jsonl(out / "disagreements.jsonl", disagreements)
    write_json(
        out / "disagreements_summary.json",
        {
            "treatment": args.treatment,
            "comparator": args.comparator,
            "metric": metric,
            "n_shared_cases": len(shared),
            "n_treatment_wins": len([c for c in shared if treatment[c] > comparator[c]]),
            "n_comparator_wins": len([c for c in shared if treatment[c] < comparator[c]]),
            "n_sampled_each_direction": N_DISAGREEMENTS,
            "note": (
                "Both directions are sampled in equal number by construction. The full "
                "win/loss counts are reported above the sample so the sample cannot be "
                "mistaken for the distribution."
            ),
        },
    )
    log.info(
        "disagreements -> %s (%d shared cases, %d sampled)",
        out / "disagreements.jsonl",
        len(shared),
        len(disagreements),
    )

    if not generations:
        log.warning(
            "No system has generations. The selection rules, the fact records and the "
            "rendering paths are exercised; the narratives are absent and are recorded "
            "as such."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
