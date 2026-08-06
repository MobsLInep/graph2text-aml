#!/usr/bin/env python
"""Extract case_facts records for every case in a built corpus.

Writes three things, all under the run's processed directory:

- ``facts/<case_id>.json`` — one validated record per case (invariant 3: each carries the
  frozen schema version and the resolved extractor config).
- ``facts.parquet`` — a flattened aggregate for analysis. Not a substitute for the JSON:
  it drops the availability sentinels' reasons and the field-producer map, so it is for
  plotting, never for generation.
- ``facts_coverage.json`` — what fraction of schema fields is populated per substrate.
  This is the number that makes invariant 4 auditable: an Elliptic2 run SHOULD show low
  coverage on the monetary families, and a run that showed high coverage there would mean
  the mask had stopped being consulted.

Usage:
    uv run python scripts/03_extract_facts.py
    uv run python scripts/03_extract_facts.py data=elliptic2
    uv run python scripts/03_extract_facts.py facts.limit=1000
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

import hydra
import polars as pl
from omegaconf import DictConfig

from g2t_aml.data.canonical import CanonicalGraph
from g2t_aml.data.case_extraction import GraphIndex
from g2t_aml.data.case_sampling import CaseCollection
from g2t_aml.facts.config import FactConfig
from g2t_aml.facts.extractor import extract_facts
from g2t_aml.facts.salience import salience_report
from g2t_aml.facts.schema import (
    CASE_FACTS_SCHEMA_VERSION,
    EXTRACTOR_VERSION,
    CaseFacts,
    Unavailable,
    facts_to_dict,
    validate_facts,
)
from g2t_aml.utils.io import write_json
from g2t_aml.utils.logging import configure_logging, get_logger, stage
from g2t_aml.utils.run_context import RunContext

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def _leaf_paths(payload: Any, prefix: str = "") -> dict[str, bool]:
    """Map every leaf field path to whether it carries a value.

    A field is populated when it is neither an availability sentinel nor a bare null. The
    distinction is what makes the coverage report meaningful rather than a count of keys.
    """
    populated: dict[str, bool] = {}
    if isinstance(payload, dict):
        if "available" in payload and payload["available"] is False:
            return {prefix: False}
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict) and set(value) == {"value", "currency"}:
                populated[path] = True
            elif isinstance(value, dict):
                populated |= _leaf_paths(value, path)
            elif isinstance(value, list):
                populated[path] = len(value) > 0
            else:
                populated[path] = value is not None
    return populated


def _flatten(facts: CaseFacts) -> dict[str, Any]:
    """Flatten a record into one Parquet row, sentinels collapsed to nulls."""
    payload = facts_to_dict(facts)
    row: dict[str, Any] = {
        "case_id": facts.case_id,
        "dataset": facts.dataset,
        "schema_version": facts.schema_version,
        "extractor_version": facts.extractor_version,
        "typology": facts.typology.label,
        "typology_source": facts.typology.source,
        "typology_confidence": facts.typology.confidence,
        "focal_id": facts.focal_entity.id,
        "focal_role": facts.focal_entity.role,
    }
    for group in ("structure", "focal_entity"):
        for key, value in payload[group].items():
            if not isinstance(value, dict | list):
                row[f"{group}.{key}"] = value

    temporal = facts.temporal
    row["temporal.available"] = not isinstance(temporal, Unavailable)
    if not isinstance(temporal, Unavailable):
        row["temporal.span_hours"] = temporal.span_hours
        row["temporal.n_transactions"] = temporal.n_transactions
        row["temporal.burst_detected"] = temporal.burst_detected
        row["temporal.burst_window_hours"] = temporal.burst_window_hours
        row["temporal.burst_txn_count"] = temporal.burst_txn_count
        row["temporal.event_ordering"] = ">".join(temporal.event_ordering)

    flow = facts.flow
    row["flow.available"] = not isinstance(flow, Unavailable)
    if not isinstance(flow, Unavailable):
        for name, amount in (
            ("total_inflow", flow.total_inflow),
            ("total_outflow", flow.total_outflow),
            ("retained", flow.retained),
            ("max_single_transfer", flow.max_single_transfer),
        ):
            single = not isinstance(amount, Unavailable)
            row[f"flow.{name}"] = amount.value if single else None
            row[f"flow.{name}_currency"] = amount.currency if single else None
        row["flow.n_transfers_near_threshold"] = flow.n_transfers_near_threshold
        row["flow.n_currencies"] = len(flow.currencies_involved)
        row["flow.n_distinct_banks"] = (
            None if isinstance(flow.n_distinct_banks, Unavailable) else flow.n_distinct_banks
        )

    labels = facts.labels
    row["labels.available"] = not isinstance(labels, Unavailable)
    if not isinstance(labels, Unavailable):
        row["labels.n_illicit_counterparties"] = labels.n_illicit_counterparties
        row["labels.n_licit_counterparties"] = labels.n_licit_counterparties
        row["labels.n_unknown_counterparties"] = labels.n_unknown_counterparties
        row["labels.n_counterparties"] = labels.n_counterparties
        row["labels.min_hops_to_known_illicit"] = labels.min_hops_to_known_illicit
        row["labels.n_illicit_transactions"] = labels.n_illicit_transactions
        row["labels.illicit_inflow_share"] = (
            None
            if isinstance(labels.illicit_inflow_share, Unavailable)
            else labels.illicit_inflow_share
        )

    for name, motif in facts.motifs.as_mapping().items():
        row[f"motifs.{name}.present"] = motif.present
        for descriptor, value in motif.descriptors.items():
            if not isinstance(value, list):
                row[f"motifs.{name}.{descriptor}"] = value
    return row


#: Exit code captured out of the Hydra-decorated entrypoint.
#:
#: ``@hydra.main`` **discards its wrapped function's return value** — it returns None
#: regardless — so the long-standing ``sys.exit(main())`` always exited 0 and every
#: documented "exits non-zero when the gate fails" in this repository was silently untrue.
#: A failing gate looked identical to a passing one to CI, to ``make``, and to any caller
#: checking ``$?``. Capturing the code out of a module-level cell is the smallest fix that
#: keeps the Hydra entrypoint shape. Found in Phase 5; see PHASE_LOG.
_EXIT_CODE: list[int] = []


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def _run(cfg: DictConfig) -> None:  # noqa: PLR0915 -- two extra statements from the
    # exit-code capture; the body is one linear extract-validate-write pass.
    """Extract fact records for a built case corpus.

    Args:
        cfg: Composed Hydra configuration.

    Returns:
        Nothing. The exit code is recorded in `_EXIT_CODE`; 0 on success, 1 when the corpus
        is absent or a record fails validation.
    """
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    configure_logging(log_file=run_dir / "facts.log")
    log = get_logger(__name__)

    interim_dir = Path(cfg.paths.interim_dir) / str(cfg.data.interim_name)
    processed_dir = Path(cfg.paths.processed_dir) / str(cfg.data.interim_name)
    cases_dir = processed_dir / "cases"
    facts_dir = processed_dir / "facts"

    declared = str(cfg.schema_version.case_facts)
    if declared != CASE_FACTS_SCHEMA_VERSION:
        log.error(
            "config declares case_facts schema %s but the code is frozen at %s " "(invariant 3)",
            declared,
            CASE_FACTS_SCHEMA_VERSION,
        )
        _EXIT_CODE.append(1)
        return

    with stage("facts", log, cases_dir=str(cases_dir), out=str(facts_dir)) as summary:
        if not (cases_dir / "cases.jsonl").is_file():
            log.warning("no case corpus at %s; run `make cases` first", cases_dir)
            summary["status"] = "skipped: no case corpus"
            _EXIT_CODE.append(0)
            return

        collection = CaseCollection.load(cases_dir)
        index = GraphIndex(CanonicalGraph.load(interim_dir))
        config = FactConfig.from_hydra(cfg.facts)
        facts_dir.mkdir(parents=True, exist_ok=True)

        limit = int(cfg.facts.limit or 0)
        case_ids = collection.case_ids[:limit] if limit else collection.case_ids
        log.info("extracting facts for %d cases", len(case_ids))

        rows: list[dict[str, Any]] = []
        field_populated: Counter[str] = Counter()
        field_seen: Counter[str] = Counter()
        typologies: Counter[str] = Counter()
        salience_required: Counter[int] = Counter()

        for n, case_id in enumerate(case_ids, start=1):
            facts = extract_facts(collection.materialise(case_id, index), config)
            payload = facts_to_dict(facts)
            validate_facts(payload)
            write_json(facts_dir / f"{case_id}.json", payload, canonical=True)

            rows.append(_flatten(facts))
            typologies[facts.typology.label] += 1
            salience_required[len(salience_report(facts).required)] += 1
            for path, populated in _leaf_paths(payload).items():
                field_seen[path] += 1
                if populated:
                    field_populated[path] += 1
            if n % 2000 == 0:
                log.info("  %d / %d", n, len(case_ids))

        frame = pl.DataFrame(rows, infer_schema_length=None)
        frame.write_parquet(processed_dir / "facts.parquet", compression="zstd")

        coverage = {
            path: round(field_populated[path] / field_seen[path], 6) for path in sorted(field_seen)
        }
        report = {
            "dataset": str(cfg.data.interim_name),
            "schema_version": CASE_FACTS_SCHEMA_VERSION,
            "extractor_version": EXTRACTOR_VERSION,
            "n_cases": len(case_ids),
            "availability_mask": index.graph.availability.to_dict(),
            "field_population_rate": coverage,
            "fully_populated_fields": sorted(p for p, r in coverage.items() if r == 1.0),
            "never_populated_fields": sorted(p for p, r in coverage.items() if r == 0.0),
            "typology_counts": dict(typologies.most_common()),
            "salient_fields_required_histogram": dict(sorted(salience_required.items())),
            "fact_config": config.to_dict(),
        }
        write_json(processed_dir / "facts_coverage.json", report, canonical=True)
        write_json(run_dir / "facts_coverage.json", report, canonical=True)
        RunContext.capture(
            experiment_name=cfg.experiment.name,
            cfg=cfg,
            seeds={"global": int(cfg.seed)},
            repo_root=Path(__file__).resolve().parents[1],
            phase="3",
        ).save(run_dir)

        summary["n_cases"] = len(case_ids)
        summary["mean_field_population"] = round(sum(coverage.values()) / len(coverage), 4)
        summary["status"] = "ok"

    _EXIT_CODE.append(0)
    return


def main() -> int:
    """Run the Hydra entrypoint and return its exit code.

    Returns:
        The code ``_run`` produced, or 0 when it produced none.
    """
    _run()
    return _EXIT_CODE[-1] if _EXIT_CODE else 0


if __name__ == "__main__":
    sys.exit(main())
