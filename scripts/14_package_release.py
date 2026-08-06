#!/usr/bin/env python
"""Package the release artifacts into licence-separated bundles.

**The separation is the point of this script.** AMLworld data is CDLA-Sharing-1.0 with a
share-alike obligation; CDLA-Sharing-1.0 §3.5 exempts *Results* -- trained models, metrics
and figures -- from that obligation. So there are two bundles and they must not be mixed:

- ``code-and-results``  Apache-2.0. Weights, metrics, figures, run contexts, docs.
- ``corpus-and-facts``  **CDLA-Sharing-1.0.** Narratives, fact records, cases.

The corpus goes in the second bundle rather than the first, even though "generated
narratives" reads like a §3.5 Result, because **these narratives quote the source Data**:
account identifiers, timestamps, currencies and transaction amounts appear verbatim in the
text, and each record embeds the fact record they were rendered from. That is more than a
de-minimis portion of the original Data, which makes the corpus Enhanced Data. The
conservative reading costs nothing -- CDLA-Sharing-1.0 permits redistribution -- and the
unconservative one is a licence breach. See D-098.

Each bundle is written with its licence text, an attribution NOTICE where one is owed, a
manifest with a SHA-256 per file, and a README saying what it is and is not.

**Nothing Elliptic2-derived is ever packaged**: the substrate is access-gated, its data
licence could not be located, and this project never obtained it. The Elliptic2 path is a
reconstruction script (``scripts/14_reconstruct_elliptic2.py``), not data.

Usage:
    uv run python scripts/14_package_release.py --dry-run
    uv run python scripts/14_package_release.py --out dist/
    uv run python scripts/14_package_release.py --out dist/ --bundle corpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CDLA_URL = "https://cdla.dev/sharing-1-0/"

ATTRIBUTION = """\
Data Provider
-------------
Altman, E., Blanusa, J., von Niederhausern, L., Egressy, B., Anghel, A., Atasu, K.
"Realistic Synthetic Financial Transactions for Anti-Money Laundering Models."
Advances in Neural Information Processing Systems 36 (Datasets and Benchmarks), 2023.
Source: https://github.com/IBM/AML-Data
"""


@dataclass(frozen=True)
class Bundle:
    """One licence-homogeneous release archive.

    Attributes:
        name: Archive basename.
        licence: SPDX identifier governing every file in it.
        summary: One line for the bundle README.
        sources: ``(repo-relative source, path inside the archive)`` pairs. A source that
            is a directory is copied whole; a missing source is reported, never skipped
            silently.
        notice: Extra text appended to the bundle NOTICE, or empty.
    """

    name: str
    licence: str
    summary: str
    sources: tuple[tuple[str, str], ...]
    notice: str = ""


#: Apache-2.0. Results under CDLA-Sharing-1.0 s3.5: nothing here embeds source Data.
CODE_AND_RESULTS = Bundle(
    name="graph2text-aml-code-and-results",
    licence="Apache-2.0",
    summary=(
        "Model checkpoints, evaluation metrics, figures, run provenance and documentation. "
        "Results under CDLA-Sharing-1.0 s3.5; no source Data is embedded."
    ),
    sources=(
        ("artifacts/checkpoints", "checkpoints"),
        ("artifacts/metrics", "metrics"),
        ("artifacts/figures", "figures"),
        ("RESULTS.md", "RESULTS.md"),
        ("DECISIONS.md", "DECISIONS.md"),
        ("PHASE_LOG.md", "PHASE_LOG.md"),
        ("CITATION.cff", "CITATION.cff"),
        ("CHANGELOG.md", "CHANGELOG.md"),
        ("docs", "docs"),
        ("schemas", "schemas"),
    ),
)

#: CDLA-Sharing-1.0. Enhanced Data: every narrative quotes the source Data.
CORPUS_AND_FACTS = Bundle(
    name="graph2text-aml-corpus-and-facts",
    licence="CDLA-Sharing-1.0",
    summary=(
        "The Bronze narrative corpus, the case_facts records and the case store. "
        "Enhanced Data under CDLA-Sharing-1.0: the narratives quote AMLworld identifiers, "
        "timestamps and amounts verbatim."
    ),
    sources=(
        ("data/processed/amlworld_hi_small/corpus", "corpus"),
        ("data/processed/amlworld_hi_small/facts.parquet", "facts.parquet"),
        ("data/processed/amlworld_hi_small/facts", "facts"),
        ("data/processed/amlworld_hi_small/cases", "cases"),
        ("data/processed/amlworld_hi_small/facts_coverage.json", "facts_coverage.json"),
        ("data/processed/amlworld_hi_small/leakage_audit.json", "leakage_audit.json"),
        # The split manifests, the JSON Schema and the vocabulary are deliberately NOT
        # here. They are lists of derived case identifiers and a schema -- Apache-2.0, and
        # they ship in the code bundle. Copying them in would make this bundle
        # licence-heterogeneous, which is the one thing it must not be.
    ),
    notice=ATTRIBUTION
    + """
Changes made to the original Data (CDLA-Sharing-1.0 s3.2)
---------------------------------------------------------
Nothing in this bundle is a copy of the original files. Every record is derived:

1. Cases were extracted as 2-hop, 48-hour-windowed subgraphs around seed accounts, capped
   at 150 nodes and 64 neighbours per node, pruned by descending amount.
2. `case_facts` records were computed from those subgraphs -- degrees, flows, motifs,
   temporal structure -- against the frozen schema `case_facts_v1.json` at version 1.0.0.
3. Bronze narratives were rendered from those records by deterministic template. Amounts
   are rounded for display and cross-currency aggregates are withheld as typed sentinels,
   because the source carries no exchange rates.
4. Split manifests are lists of derived case identifiers, not of source rows.

No original transaction row is reproduced in full anywhere in this bundle. Account
identifiers, timestamps, currencies and amounts do appear, which is why this bundle is
Enhanced Data and carries this NOTICE.

NOT INCLUDED
------------
Nothing derived from Elliptic2. That substrate is access-gated, its data licence could not
be located, and this project never obtained it. Use
`scripts/14_reconstruct_elliptic2.py` against your own licensed copy.
""",
)

BUNDLES = {"code": CODE_AND_RESULTS, "corpus": CORPUS_AND_FACTS}


def sha256_file(path: Path) -> str:
    """Digest a file.

    Args:
        path: The file.

    Returns:
        Lowercase hex SHA-256.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(bundle: Bundle) -> tuple[list[tuple[Path, str]], list[str]]:
    """Resolve a bundle's sources into concrete files.

    Args:
        bundle: The bundle definition.

    Returns:
        ``(files, missing)`` where ``files`` is ``(absolute source, archive-relative
        path)`` and ``missing`` names sources that do not exist.
    """
    files: list[tuple[Path, str]] = []
    missing: list[str] = []
    for src, dest in bundle.sources:
        origin = REPO_ROOT / src
        if not origin.exists():
            missing.append(src)
            continue
        if origin.is_file():
            files.append((origin, dest))
            continue
        for path in sorted(origin.rglob("*")):
            if not path.is_file() or path.name == ".gitkeep":
                continue
            files.append((path, f"{dest}/{path.relative_to(origin).as_posix()}"))
    return files, missing


def write_bundle(bundle: Bundle, out_dir: Path, *, version: str, compress: bool) -> dict:
    """Assemble one bundle and, unless disabled, tar it.

    Args:
        bundle: The bundle definition.
        out_dir: Directory to write into.
        version: Release version string, recorded in the manifest.
        compress: Write a ``.tar.gz`` beside the staged directory.

    Returns:
        The manifest, also written into the bundle as ``MANIFEST.json``.
    """
    staged = out_dir / bundle.name
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)

    files, missing = collect(bundle)
    entries = []
    for origin, rel in files:
        target = staged / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        entries.append(
            {
                "path": rel,
                "sha256": sha256_file(target),
                "size_bytes": target.stat().st_size,
            }
        )

    licence_src = REPO_ROOT / "LICENSE"
    if bundle.licence == "Apache-2.0" and licence_src.is_file():
        shutil.copy2(licence_src, staged / "LICENSE")
    elif bundle.licence == "CDLA-Sharing-1.0":
        (staged / "LICENSE").write_text(
            "This bundle is licensed under the Community Data License Agreement -\n"
            f"Sharing, Version 1.0 (CDLA-Sharing-1.0).\n\nFull text: {CDLA_URL}\n\n"
            "The agreement text must accompany any redistribution of this bundle, and\n"
            "redistribution must be under an unmodified form of the same agreement.\n"
            "No additional restrictions may be imposed.\n",
            encoding="utf-8",
        )

    if bundle.notice:
        (staged / "NOTICE").write_text(bundle.notice, encoding="utf-8")

    manifest = {
        "bundle": bundle.name,
        "version": version,
        "licence": bundle.licence,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_files": len(entries),
        "total_bytes": sum(int(e["size_bytes"]) for e in entries),
        "missing_sources": missing,
        "files": entries,
    }
    (staged / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (staged / "README.md").write_text(
        f"# {bundle.name}\n\n"
        f"**Licence: {bundle.licence}**\n\n{bundle.summary}\n\n"
        f"Version `{version}`. {len(entries)} files, "
        f"{manifest['total_bytes'] / 1e6:.1f} MB.\n\n"
        "Every file's SHA-256 is in `MANIFEST.json`.\n\n"
        "**This bundle is licence-homogeneous and must stay that way.** The companion "
        "bundle carries the other licence; do not merge them. See `DECISIONS.md` D-098 "
        "and `docs/dataset_cards/README.md`.\n",
        encoding="utf-8",
    )

    if compress:
        archive = out_dir / f"{bundle.name}-{version}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(staged, arcname=staged.name)
        manifest["archive"] = archive.name
        manifest["archive_sha256"] = sha256_file(archive)
        manifest["archive_bytes"] = archive.stat().st_size
    return manifest


def main() -> int:
    """Package the requested bundles.

    Returns:
        0 on success, 1 when a bundle would be empty or a source is missing under
        ``--strict``.
    """
    parser = argparse.ArgumentParser(
        description="Package release artifacts into licence-separated bundles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Two bundles, two licences, never mixed: code-and-results is Apache-2.0, "
            "corpus-and-facts is CDLA-Sharing-1.0. Nothing Elliptic2-derived is ever "
            "packaged. See DECISIONS.md D-098."
        ),
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "dist")
    parser.add_argument(
        "--bundle",
        choices=[*BUNDLES, "all"],
        default="all",
        help="which bundle to build (default: all)",
    )
    parser.add_argument("--version", default="v0.1.0")
    parser.add_argument("--no-compress", action="store_true", help="stage only, no tarball")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what each bundle would contain and exit without writing",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any declared source is missing",
    )
    args = parser.parse_args()

    selected = list(BUNDLES.values()) if args.bundle == "all" else [BUNDLES[args.bundle]]
    exit_code = 0

    if args.dry_run:
        for bundle in selected:
            files, missing = collect(bundle)
            total = sum(p.stat().st_size for p, _ in files)
            print(f"\n{bundle.name}  [{bundle.licence}]")
            print(f"  {len(files)} files, {total / 1e6:.1f} MB")
            for src, _ in bundle.sources:
                present = (REPO_ROOT / src).exists()
                print(f"    {'ok     ' if present else 'MISSING'} {src}")
            if missing and args.strict:
                exit_code = 1
        print("\ndry run: nothing written")
        return exit_code

    args.out.mkdir(parents=True, exist_ok=True)
    for bundle in selected:
        manifest = write_bundle(
            bundle, args.out, version=args.version, compress=not args.no_compress
        )
        print(f"\n{bundle.name}  [{bundle.licence}]")
        print(f"  {manifest['n_files']} files, {manifest['total_bytes'] / 1e6:.1f} MB")
        if manifest.get("archive"):
            print(
                f"  archive: {manifest['archive']} " f"({manifest['archive_bytes'] / 1e6:.1f} MB)"
            )
            print(f"  sha256:  {manifest['archive_sha256']}")
        if manifest["missing_sources"]:
            print(f"  MISSING SOURCES ({len(manifest['missing_sources'])}):")
            for src in manifest["missing_sources"]:
                print(f"    {src}")
            if args.strict:
                exit_code = 1
        if manifest["n_files"] == 0:
            print("  bundle is EMPTY")
            exit_code = 1

    print(f"\nwrote bundles to {args.out}")
    print("The two bundles carry different licences and must not be merged (D-098).")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
