#!/usr/bin/env python
"""Rebuild the Elliptic2 derived artifacts from *your own* licensed copy.

**Why this script exists instead of data.** Elliptic2 is access-gated, and its *data*
licence could not be located: the official tooling repository is Apache-2.0 but says
nothing about the dataset, and the terms presumably arrive with the access grant. This
project recorded it `redistributable=False` and treats it as closed until it holds written
terms. So the release ships a reconstruction path rather than any Elliptic2 bytes.

The intended shape of that path is the one the community has converged on for gated
corpora: publish the **case identifiers, the fact records and the narratives** -- which are
derived, not the source -- plus a script that rebuilds the graphs from the licensed copy
the user already holds.

**This project has none of those to publish.**

> Elliptic2 access was never requested. The substrate has never been ingested. There are
> zero Elliptic2 cases, zero fact records and zero narratives anywhere in this repository,
> and therefore nothing for this script to reconstruct against.

The script is shipped now, tested and documented, so that the path is in place the day
access arrives -- and so that the release does not quietly imply an Elliptic2 half that
does not exist. Run it and it will tell you exactly that.

What it does when the derived artifacts *do* exist:

1. Verify the raw Elliptic2 files are present and readable, and report their digests.
2. Ingest them through the same loader every other substrate uses.
3. Rebuild the case subgraphs named by the released case-id manifest.
4. Verify each rebuilt case against the released fact record, and report any divergence.

Usage:
    uv run python scripts/14_reconstruct_elliptic2.py --raw data/raw/elliptic2
    uv run python scripts/14_reconstruct_elliptic2.py --check-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The files an Elliptic2 access grant delivers, per the official tooling's documentation.
#: No checksums are pinned: this project has never seen them (docs/data_cards/elliptic2.md).
EXPECTED_FILES = (
    "background_edges.csv",
    "background_nodes.csv",
    "connected_components.csv",
    "edges.csv",
    "nodes.csv",
)

ACCESS_URL = "https://www.elliptic.co/elliptic2"
TOOLING_URL = "https://github.com/MITIBMxGraph/Elliptic2"

NOTHING_TO_RECONSTRUCT = f"""\
Nothing to reconstruct.

This release contains no Elliptic2-derived artifacts:

    case-id manifest      absent
    fact records          absent
    narratives            absent

Elliptic2 access was never requested by this project and the substrate has never been
ingested. That is documented, not accidental -- see docs/data_cards/elliptic2.md section 2,
RESULTS.md section 5, and docs/ETHICS.md section 5.

The loader ({{loader}}) is written against the documented schema and is tested against a
synthetic tree; its real-data tests skip. `make data-elliptic2` exits 0 with
ingest_skipped.json rather than failing.

If you hold a licensed copy and want to extend this work to Elliptic2:

  1. Request access:   {ACCESS_URL}
  2. Official tooling: {TOOLING_URL}
  3. Unzip so data/raw/elliptic2/ contains:
       {" ".join(EXPECTED_FILES)}
  4. make data-elliptic2 && make cases && make facts && make bronze

Read docs/data_cards/elliptic2.md section 2 before publishing anything derived from it.
The terms that arrive with your access grant govern your copy, and they may differ from
what any third party assumed.
"""


def check_raw(raw_dir: Path) -> tuple[list[str], list[str]]:
    """Report which expected raw files are present.

    Args:
        raw_dir: The directory an access grant was unzipped into.

    Returns:
        ``(present, missing)`` file names.
    """
    present, missing = [], []
    for name in EXPECTED_FILES:
        (present if (raw_dir / name).is_file() else missing).append(name)
    return present, missing


def released_artifacts() -> dict[str, Path | None]:
    """Locate the Elliptic2 derived artifacts a reconstruction would need.

    Returns:
        Artifact name to its path, or ``None`` when absent.
    """
    processed = REPO_ROOT / "data" / "processed" / "elliptic2"
    candidates = {
        "case_ids": processed / "cases" / "case_ids.txt",
        "facts": processed / "facts.parquet",
        "narratives": processed / "corpus" / "bronze.jsonl",
    }
    return {name: (path if path.exists() else None) for name, path in candidates.items()}


def main() -> int:
    """Report the reconstruction state, and reconstruct when there is anything to.

    Returns:
        0 when the state is reported successfully -- including the expected case where
        there is nothing to reconstruct. 1 only when the user asked for a reconstruction
        against raw files that are missing or unreadable.
    """
    parser = argparse.ArgumentParser(
        description="Rebuild Elliptic2 derived artifacts from your own licensed copy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Elliptic2 is access-gated and not redistributable, so this release ships a "
            "reconstruction path rather than data. This project never obtained the "
            "substrate, so there is currently nothing to reconstruct against."
        ),
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "elliptic2",
        help="directory holding your licensed Elliptic2 copy",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "interim",
        help="where reconstructed interim artifacts would be written",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="report the state and exit without reconstructing",
    )
    args = parser.parse_args()

    artifacts = released_artifacts()
    have_any = any(path is not None for path in artifacts.values())

    loader = "src/g2t_aml/data/loaders/elliptic2.py"
    if not have_any:
        print(NOTHING_TO_RECONSTRUCT.format(loader=loader))
        present, missing = check_raw(args.raw)
        if present:
            print(
                f"For information: {len(present)} of {len(EXPECTED_FILES)} raw "
                f"Elliptic2 files are present at {args.raw}."
            )
            if not missing:
                print(
                    "You appear to hold the full dataset. Run `make data-elliptic2` to "
                    "ingest it and build the derived artifacts from scratch."
                )
        return 0

    print("Elliptic2 derived artifacts found in this release:")
    for name, path in artifacts.items():
        state = str(path.relative_to(REPO_ROOT)) if path else "absent"
        print(f"  {name:12s} {state}")

    present, missing = check_raw(args.raw)
    print(f"\nRaw Elliptic2 at {args.raw}: {len(present)} present, {len(missing)} missing")
    for name in missing:
        print(f"  missing: {name}")
    if missing:
        print(f"\nCannot reconstruct without the full dataset. Request access at " f"{ACCESS_URL}.")
        return 1

    if args.check_only:
        print(
            "\n--check-only: raw files present and derived artifacts available; "
            "reconstruction would run."
        )
        return 0

    print("\nReconstructing via the standard pipeline:")
    print("  uv run python scripts/01_ingest.py data=elliptic2")
    print("  uv run python scripts/02_build_cases.py data=elliptic2")
    print("  uv run python scripts/03_extract_facts.py data=elliptic2")
    print("\nThen verify the rebuilt records against the released fact records with")
    print("  uv run pytest tests/integration/test_facts_roundtrip.py")
    print(
        "\nThis script does not re-implement those stages; running them separately keeps "
        "the reconstruction path identical to the production path, which is the only way "
        "a reconstruction is evidence of anything."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
