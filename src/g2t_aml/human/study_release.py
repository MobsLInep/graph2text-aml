"""Turning a run study into something publishable without publishing a person.

The response data is the most valuable artifact Phase 12 produces and the most dangerous.
It is valuable because a decision-setting study that nobody can re-analyse is a table of
numbers a reviewer has to take on faith. It is dangerous because it contains free text
written by identifiable professionals about their own working practice, and because
"anonymised" is a claim about what can be *re-identified*, not about whether a name field
was deleted.

Four things happen here, and each is a distinct risk:

**Rater ids are re-pseudonymised.** ``rater-03`` is already a pseudonym, but it is the same
pseudonym that appears in the recruitment records, the payment schedule and the consent
forms. Anyone holding those and the released data can join them. The release therefore
re-maps to fresh ``R01``-style labels through a keyed digest, and **the mapping is not
written to the release**. It is emitted separately so that the study team can answer a
query about a specific participant, and it lives wherever the consent forms live.

**Free text is dropped, not scrubbed.** Comments and the rater's corrected narratives are
the two free-text fields. A comment like "I've seen this pattern at my last employer" is
identifying, and no regular expression finds the general case. Comments are therefore
removed entirely rather than filtered. The corrected narratives are *kept*, because the
edit distance between presented and corrected is a headline measurement and a release that
cannot reproduce it is not reproducible — but they are passed through the Phase 4 PII
scanner first, and any response whose correction trips it is withheld with its item id
listed, so the withholding is visible in the release rather than silent.

**System labels are revealed.** This is the point at which the study stops being blind. Each
released row carries the system that produced its narrative, joined from the blind key —
which is why the key is an input here and nowhere in the interface.

**Timing provenance travels with the times.** A row timed by the server clock cannot have
had tab-hidden time removed from it. Released rows keep ``timing_source`` so a re-analyst
can drop them, and the manifest counts them.

The output is a directory: the responses as JSONL, the design, the scale definitions, a
manifest with the counts and content hashes, and a README. Nothing in it is a summary — the
statistics are recomputed from these files by ``notebooks/12_human_study.ipynb``, which is
what makes the published numbers checkable rather than merely stated.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from g2t_aml.human.study_design import BlindKey, StudyDesign
from g2t_aml.human.study_ui import LIKERT_DIMENSIONS, RatingResponse
from g2t_aml.utils.io import atomic_path

__all__ = [
    "RELEASE_SCHEMA_VERSION",
    "ReleaseReport",
    "prepare_release",
    "pseudonymise",
]

#: Version of the released record shape. Independent of the internal response schema: the
#: release drops fields and adds the system label, so the two are not the same object and
#: versioning them together would make an internal change look like a release change.
RELEASE_SCHEMA_VERSION = "1.0.0"

#: Fields dropped from every released row, with the reason each is unsafe.
#:
#: ``comment`` is unstructured text about the rater's own professional experience and cannot
#: be scrubbed reliably. ``rater_id`` is replaced rather than dropped -- see
#: :func:`pseudonymise`. ``submitted_at`` is a wall-clock timestamp, and a sequence of them
#: is a record of when a named professional was working, which is more than the study needs
#: and more than a participant agreed to publish.
_DROPPED = ("comment", "rater_id", "submitted_at")


def pseudonymise(rater_ids: Sequence[str], salt: str) -> dict[str, str]:
    """Return a fresh label for each rater, ordered so the labels leak nothing.

    Sorted by digest rather than by the original id, so ``R01`` is not ``rater-01``.
    Labelling in input order would make the re-pseudonymisation cosmetic: anyone holding the
    recruitment list could invert it by sorting.

    Args:
        rater_ids: The internal pseudonyms.
        salt: The release salt. Must not be the design salt -- a shared salt lets someone
            holding the design reproduce the mapping.

    Returns:
        Internal id to released label.
    """
    digested = sorted(
        rater_ids,
        key=lambda r: hashlib.blake2b(f"{salt}|{r}".encode(), digest_size=8).hexdigest(),
    )
    return {rater_id: f"R{i:02d}" for i, rater_id in enumerate(digested, start=1)}


@dataclass(frozen=True)
class ReleaseReport:
    """What the release contains and what it withheld.

    Attributes:
        n_released: Rows written.
        n_withheld: Rows withheld, with the reason recorded per item.
        withheld_items: Item ids withheld, and why. Published, because a release that drops
            rows silently is a release whose n cannot be checked against the study's own
            design.
        n_raters: Raters in the release.
        n_server_timed: Rows whose times could not have tab-hidden periods removed.
        systems: The systems represented.
        files: Relative path to sha256 for every file written.
    """

    n_released: int
    n_withheld: int
    withheld_items: dict[str, str]
    n_raters: int
    n_server_timed: int
    systems: tuple[str, ...]
    files: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised report.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "n_released": self.n_released,
            "n_withheld": self.n_withheld,
            "withheld_items": self.withheld_items,
            "n_raters": self.n_raters,
            "n_server_timed": self.n_server_timed,
            "systems": list(self.systems),
            "files": self.files,
        }


def _scan_for_pii(text: str) -> str:
    """Return the reason this text cannot be released, or an empty string.

    Delegates to the Phase 4 scanner rather than reimplementing one. That scanner is
    deliberately blunt -- it would rather refuse a clean narrative than pass an identifier --
    which is the correct bias for a corpus check and the correct bias here too.

    Args:
        text: The rater's corrected narrative.

    Returns:
        A short reason, or ``""`` when the text is clean.
    """
    if not text.strip():
        return ""
    from g2t_aml.corpus.pii import scan_for_identifiers

    try:
        hits = scan_for_identifiers(text)
    except Exception as exc:  # a scanner failure must withhold, never pass
        return f"PII scan failed: {exc}"
    if hits:
        return f"PII scanner flagged {sorted({name for name, _ in hits})}"
    return ""


def prepare_release(
    responses: Sequence[RatingResponse],
    key: BlindKey,
    design: StudyDesign,
    out_dir: Path,
    *,
    salt: str = "g2t-aml-phase12-release",
) -> ReleaseReport:
    """Write the anonymised release bundle.

    Args:
        responses: Every response, repeats included. Repeats are released and flagged, since
            intra-rater reliability has to be recomputable.
        key: The blind key. Unblinding happens here.
        design: The design, released so the block structure is inspectable.
        out_dir: Directory to write. Created if absent.
        salt: Release salt for :func:`pseudonymise`. Must differ from the design salt.

    Returns:
        The report, also written as ``manifest.json``.

    Raises:
        ValueError: If the release salt equals the design salt, which would make the
            re-pseudonymisation invertible by anyone holding the design.
        OSError: If a write fails.
    """
    if salt == key.salt:
        raise ValueError(
            "the release salt must differ from the design salt: sharing them lets anyone "
            "holding the design invert the released rater labels"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pseudonymise(sorted({r.rater_id for r in responses}), salt)
    rows: list[dict[str, Any]] = []
    withheld: dict[str, str] = {}

    for response in sorted(responses, key=lambda r: (labels[r.rater_id], r.position)):
        reason = _scan_for_pii(response.corrected_narrative)
        if reason:
            withheld[response.item_id] = reason
            continue
        payload = response.to_dict()
        for field in _DROPPED:
            payload.pop(field, None)
        payload["rater"] = labels[response.rater_id]
        payload["system"] = key.system_for(response.item_id)
        payload["schema_version"] = RELEASE_SCHEMA_VERSION
        rows.append(payload)

    files: dict[str, str] = {}

    def write(name: str, text: str) -> None:
        path = out_dir / name
        with atomic_path(path) as tmp:
            tmp.write_text(text, encoding="utf-8")
        files[name] = hashlib.sha256(text.encode("utf-8")).hexdigest()

    write(
        "responses.jsonl",
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
    )
    write(
        "design.json",
        json.dumps(design.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    write(
        "scales.json",
        json.dumps([d.to_dict() for d in LIKERT_DIMENSIONS], ensure_ascii=False, indent=2) + "\n",
    )
    write("README.md", _readme(len(rows), len(withheld)))

    report = ReleaseReport(
        n_released=len(rows),
        n_withheld=len(withheld),
        withheld_items=withheld,
        n_raters=len({r["rater"] for r in rows}),
        n_server_timed=sum(1 for r in rows if r.get("timing_source") == "server"),
        systems=tuple(sorted({r["system"] for r in rows})),
        files=files,
    )
    write(
        "manifest.json",
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    # The rater mapping is written OUTSIDE the release directory, beside it. Putting it in
    # the bundle would undo the entire re-pseudonymisation the moment the bundle is zipped
    # and uploaded, which is exactly the accident this layout exists to prevent.
    mapping_path = out_dir.parent / f"{out_dir.name}_rater_map.PRIVATE.json"
    with atomic_path(mapping_path) as tmp:
        tmp.write_text(
            json.dumps(
                {
                    "WARNING": (
                        "This file re-identifies participants. It is NOT part of the "
                        "release. Store it with the consent forms and never publish it."
                    ),
                    "salt": salt,
                    "mapping": labels,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return report


def _readme(n_released: int, n_withheld: int) -> str:
    """Return the release README.

    Args:
        n_released: Rows in the release.
        n_withheld: Rows withheld.

    Returns:
        Markdown.
    """
    return f"""# Graph2Text AML — decision-setting study, anonymised responses

{n_released} rating responses from a blinded, balanced incomplete block study in which
AML-literate raters judged SAR narratives produced by several systems, corrected each to a
state they would file, and were timed while doing so.

## Files

| File | What it is |
|---|---|
| `responses.jsonl` | One row per rating. System labels revealed. |
| `design.json` | The block design: who saw what, in what order. No system labels. |
| `scales.json` | The five ordinal scales with their verbatim anchors. |
| `manifest.json` | Counts, withheld items and content hashes. |

## Fields

`item_id` opaque item key · `rater` released pseudonym · `case_id` the case ·
`system` the arm that produced the narrative · `position` order in that rater's session ·
`is_repeat` a planted repeat, for intra-rater reliability ·
the five scales, each an integer 1-7 · `would_file` the binary decision ·
`seconds_to_usable_draft` **active** seconds · `timing_source` `browser` or `server` ·
`hidden_seconds`, `n_blurs` tab-visibility accounting ·
`presented_narrative`, `corrected_narrative` the two texts the edit distance is computed
between.

## Read this before using the times

`timing_source` is `browser` for rows whose time excludes periods when the tab was hidden,
and `server` for rows where the visibility component did not load. **A `server` row's time
is an upper bound** and includes any interruption. Drop them or report them separately.

## Read this before using the ratings

- The design is **incomplete**: no rater saw the same case twice, so most (case, system)
  cells are rated once. Inter-rater agreement is estimable only on the anchor cells that
  every rater rated — `design.json` records how many there are.
- **Do not report a Likert mean without an agreement statistic.**
- Friedman's test needs complete blocks and this design has none. Block by rater, or use
  Durbin's test on the case blocks. See `notebooks/12_human_study.ipynb`.

## Withheld

{n_withheld} response(s) were withheld because the corrected narrative tripped the PII
scanner. Their item ids and the reason are in `manifest.json`, so the released n can be
reconciled against the design.

## Anonymisation

Rater labels are re-derived from the internal pseudonyms through a keyed digest under a
salt not shared with the design, and sorted by digest so the ordering carries nothing.
Free-text comments and submission timestamps are dropped entirely. The mapping back to
internal pseudonyms is not part of this release.

## Licence

See the repository's `LICENSE` and `docs/data_cards/`. Note that AMLworld-derived data is
CDLA-Sharing-1.0; narratives and metrics are exempt under §3.5.
"""
