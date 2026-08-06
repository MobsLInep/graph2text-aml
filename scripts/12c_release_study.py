#!/usr/bin/env python
"""Prepare the anonymised response data for public release.

Reads the blind key, because the release is where system labels are revealed. Everything
else it does is subtraction: free-text comments dropped, identifiers re-pseudonymised under
a salt not shared with the design, timestamps removed, and any correction that trips the
identifier scanner withheld and named in the manifest.

The re-identification map is written **outside** the release directory and marked PRIVATE.
Do not move it in. See ``docs/human_study/data_management_plan.md`` §5.

Usage:
    uv run python scripts/12c_release_study.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from g2t_aml.human.study_analysis import load_blind_key
from g2t_aml.human.study_design import load_design
from g2t_aml.human.study_release import prepare_release
from g2t_aml.human.study_ui import ResponseStore

#: Exit code captured out of the Hydra-decorated entrypoint. `@hydra.main` discards the
#: wrapped function's return value; see D-051.
_EXIT_CODE: list[int] = []


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def _run(cfg: DictConfig) -> None:
    """Write the release bundle.

    Args:
        cfg: The composed Hydra config.

    Returns:
        Nothing. The exit code is recorded in :data:`_EXIT_CODE`: 2 when there are no
        responses to release, 0 otherwise.
    """
    study_dir = Path(cfg.paths.artifacts_dir) / "human_study"
    responses = ResponseStore(root=study_dir / "responses").read_all()
    if not responses:
        print(f"No responses under {study_dir / 'responses'}; nothing to release.", file=sys.stderr)
        _EXIT_CODE.append(2)
        return

    report = prepare_release(
        responses,
        load_blind_key(study_dir / "key.json"),
        load_design(study_dir / "design.json"),
        study_dir / "release",
    )

    print(f"Released {report.n_released} responses from {report.n_raters} raters")
    print(f"Systems: {', '.join(report.systems)}")
    if report.n_withheld:
        print(f"\nWithheld {report.n_withheld} response(s):")
        for item_id, reason in sorted(report.withheld_items.items()):
            print(f"  {item_id}: {reason}")
    if report.n_server_timed:
        print(
            f"\n{report.n_server_timed} released rows were timed by the server clock and "
            "their times are upper bounds. The README says so."
        )
    print(f"\nWrote {study_dir / 'release'}")
    print(
        f"Wrote {study_dir / 'release_rater_map.PRIVATE.json'}\n"
        "  ^ NOT part of the release. Store it with the consent forms and destroy it on "
        "publication."
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
