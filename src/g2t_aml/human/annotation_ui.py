r"""The writing interface: graph on the left, facts and the box on the right.

Run it with::

    uv run streamlit run src/g2t_aml/human/annotation_ui.py -- \\
        --annotator annotator-01 --dataset amlworld_hi_small

Everything this module does that matters is decided elsewhere and only *displayed* here:
the fact panel is :mod:`g2t_aml.human.factpanel`, the graph is
:mod:`g2t_aml.human.graphview`, the live flags are :mod:`g2t_aml.human.validation`, and
what is written down is :mod:`g2t_aml.human.store`. That split is what lets the rules an
annotator works under be unit-tested without a browser, and it is why the Elliptic2
masking test does not need Streamlit installed.

**No model output, ever.** :class:`~g2t_aml.human.caseloader.AnnotationCase` has no
narrative field of any kind, and the store refuses a record carrying one. There is
therefore no code path in this file that could show an annotator a Bronze rendering, a
Silver rewrite or a generated narrative — not as a suggestion, not as a placeholder, not
as a "reference". A Gold set written next to a draft is a set of edits to that draft, and
it cannot be used to evaluate the system that produced it.

**Submission is a two-step.** The first press runs the Phase 3 checker over the draft and
shows every CONTRADICTED claim in place; the annotator either fixes them or presses again
to submit with them standing, and either way it is recorded. Live flags never block — see
:mod:`g2t_aml.human.validation` for why that is the more useful design — but a
contradiction against the record is shown before the item can be saved, because it is the
one class of error the annotator cannot detect by re-reading their own text.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from g2t_aml.corpus.silver.claim_extraction import canonicalise_narrative
from g2t_aml.facts.checkers import CheckContext, CheckResult, Verdict, check_narrative_text
from g2t_aml.facts.vocab import load_vocabulary
from g2t_aml.human.caseloader import AnnotationCase, CaseSource
from g2t_aml.human.factpanel import FactPanel, build_fact_panel
from g2t_aml.human.graphview import DEFAULT_MAX_NODES, build_graph_view, to_plotly_figure
from g2t_aml.human.store import Annotation, AnnotationStore, FlagOutcome
from g2t_aml.human.validation import SECTION_HEADINGS, Severity, ValidationSummary, validate_draft

__all__ = ["AnnotationSession", "build_parser", "main", "starter_template"]

#: The empty four-part scaffold the box starts with. Headings only — no sentence, no
#: example phrasing, nothing an annotator could complete by filling gaps. A scaffold is
#: structure; a draft is influence.
#:
#: **Title case, not upper case.** The PII scanner's SWIFT/BIC pattern is any run of eight
#: or more uppercase letters (`corpus/pii.py`, deliberately blunt — it would rather refuse
#: a narrative than miss a real identifier), and `ACTIVITY OBSERVED` matches it. An
#: all-caps scaffold would therefore fail check 8 of the ten-point harness on **every**
#: Gold record, for the heading the guidelines told the annotator to write. Section
#: detection upper-cases before comparing, so the canonical names in
#: :data:`~g2t_aml.human.validation.SECTION_HEADINGS` still match.
_STARTER = "\n\n".join(f"[{i}] {h.title()}\n" for i, h in enumerate(SECTION_HEADINGS, start=1))

#: Severity to the Streamlit callout used for it.
_CALLOUT = {Severity.CRITICAL: "error", Severity.WARNING: "warning", Severity.INFO: "info"}


def starter_template() -> str:
    """Return the empty four-part scaffold a new item opens with.

    Returns:
        The four headings, blank beneath each.
    """
    return _STARTER


@dataclass
class AnnotationSession:
    """One annotator's working state for one item.

    Kept as a plain object rather than in Streamlit's session state so that the
    submission logic — when a draft counts as revised, what the checker is run over, what
    is recorded — is testable without a running app.

    Attributes:
        case: The item being annotated.
        annotator_id: The pseudonym.
        started_at: Monotonic clock reading when the item was opened.
        revision_count: Substantive changes to the draft so far.
        last_draft: The draft as of the last recorded revision.
        checked_once: Whether the checker has been run on the current draft, which is
            what makes submission a two-step.
        is_calibration: Whether this item belongs to the calibration set.
    """

    case: AnnotationCase
    annotator_id: str
    started_at: float
    revision_count: int = 0
    last_draft: str = ""
    checked_once: bool = False
    is_calibration: bool = False

    @property
    def seconds_spent(self) -> float:
        """Return wall-clock seconds since the item was opened.

        Returns:
            The elapsed time.
        """
        return max(0.0, time.monotonic() - self.started_at)

    def note_draft(self, draft: str) -> None:
        """Record a draft, counting it as a revision when it is substantively different.

        Whitespace-only edits do not count. Without that, the revision count measures
        typing rather than rethinking, and the signal it is kept for — that an item was
        hard — disappears into keystroke noise.

        Args:
            draft: The current draft.
        """
        if canonicalise_narrative(draft) != canonicalise_narrative(self.last_draft):
            self.revision_count += 1
            self.checked_once = False
        self.last_draft = draft

    def validate(self, draft: str) -> ValidationSummary:
        """Run the live validation over a draft.

        Args:
            draft: The current draft.

        Returns:
            The summary shown beneath the box.
        """
        return validate_draft(
            draft,
            self.case.facts,
            salient_fields=self.case.salience.required,
        )

    def check(self, draft: str) -> list[CheckResult]:
        """Run the Phase 3 checker over a draft and return only the adverse verdicts.

        Only the text-level checks run here. A claim-level check needs a span alignment
        back to fact fields, which for a human-written narrative is built at ingestion by
        the slot aligner; running a weaker approximation of it in the editor would show
        the annotator verdicts that ingestion then disagrees with, and they would learn to
        distrust both.

        Args:
            draft: The current draft.

        Returns:
            Every CONTRADICTED result, in document order.
        """
        context = CheckContext(facts=self.case.facts, vocabulary=load_vocabulary())
        results = check_narrative_text(canonicalise_narrative(draft), context)
        return [r for r in results if r.verdict is Verdict.CONTRADICTED]

    def build_annotation(
        self,
        draft: str,
        summary: ValidationSummary,
        panel: FactPanel,
        graph_digest: dict[str, Any],
        *,
        typology_assigned: str,
        difficulty: int | None,
        comment: str,
        overrides: dict[str, bool],
        notes: dict[str, str],
    ) -> Annotation:
        """Assemble the record to be stored.

        Args:
            draft: The submitted narrative, canonicalised before storage so the character
                spans ingestion computes hold against exactly these bytes.
            summary: The validation summary at submission.
            panel: The fact panel the annotator was shown.
            graph_digest: What the graph view showed.
            typology_assigned: The annotator's typology judgement.
            difficulty: Their 1-5 rating, or None.
            comment: Free text.
            overrides: Flag rule to whether it was overridden.
            notes: Flag rule to the annotator's reason.

        Returns:
            The annotation, ready to append.
        """
        return Annotation(
            case_id=self.case.case_id,
            dataset=self.case.facts.dataset,
            annotator_id=self.annotator_id,
            narrative=canonicalise_narrative(draft),
            seconds_spent=self.seconds_spent,
            revision_count=self.revision_count,
            flags=tuple(
                FlagOutcome(
                    flag=flag,
                    overridden=overrides.get(flag.rule, True),
                    annotator_note=notes.get(flag.rule, ""),
                )
                for flag in summary.flags
            ),
            typology_assigned=typology_assigned,
            difficulty=difficulty,
            annotator_comment=comment,
            panel_digest=panel.to_dict(),
            graph_digest=graph_digest,
            checker_summary=summary.to_dict(),
            is_calibration=self.is_calibration,
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser. Arguments follow ``--`` on the ``streamlit run`` command line.
    """
    parser = argparse.ArgumentParser(description="Graph2Text AML Gold annotation interface")
    parser.add_argument("--annotator", required=True, help="pseudonym, e.g. annotator-01")
    parser.add_argument("--dataset", default="amlworld_hi_small", help="substrate key")
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument(
        "--queue",
        default="",
        help="file of case ids to annotate, one per line; defaults to the Gold reservation",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="annotate the calibration set rather than the Gold corpus",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=DEFAULT_MAX_NODES,
        help="display cap; above it the graph is truncated and says so",
    )
    return parser


def _require_streamlit() -> Any:
    """Import Streamlit, or explain how to install it.

    Returns:
        The ``streamlit`` module.

    Raises:
        ImportError: If Streamlit is not installed, with the command that installs it.
    """
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the annotation interface needs Streamlit. Install the human extra: "
            "`uv sync --group dev --extra human`"
        ) from exc
    return st


def _compose(root: Path, dataset: str) -> Any:
    """Compose the Hydra configuration, so no directory root is written in this file.

    Streamlit owns the process, so the usual ``@hydra.main`` entrypoint is unavailable;
    composing explicitly is how the interface still reaches every path through
    ``cfg.paths.*`` rather than assembling one from string literals. The convention is not
    decoration — a hardcoded ``data/processed`` here would break every deployment that
    puts the corpus somewhere else, and the annotation interface is the component most
    likely to be run on a machine that does.

    **``HydraConfig`` has to be populated by hand, and that is load-bearing.**
    ``configs/paths/default.yaml`` resolves ``root`` as
    ``${oc.env:G2T_AML_ROOT,${hydra:runtime.cwd}}``. OmegaConf resolves a nested
    interpolation in a resolver's *argument list* eagerly, so the ``hydra:`` fallback is
    evaluated whether or not the environment variable is set — and a bare ``compose()``
    leaves the ``HydraConfig`` singleton empty, so every ``cfg.paths.*`` lookup raises
    ``InterpolationResolutionError`` and the interface dies on its first page load.
    ``return_hydra_config=True`` plus ``set_config`` is what ``@hydra.main`` does; this is
    that, done explicitly because Streamlit owns the process and there is no decorator to
    hang it on.

    Found by loading the running app in a browser. No unit test would have caught it:
    every other config test in this repository goes through ``@hydra.main``, which sets
    the singleton as a side effect, so the failure exists only on the one entrypoint that
    cannot use it.

    Args:
        root: Repository root.
        dataset: The substrate key, used to pick the data config.

    Returns:
        The composed configuration, with every path interpolation resolvable.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig

    substrate = "elliptic2" if dataset.startswith("elliptic2") else "amlworld"
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
        cfg = compose(
            config_name="config",
            overrides=[f"data={substrate}", "corpus=gold"],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
        return cfg


def _queue_for(args: argparse.Namespace, manifest_dir: Path) -> list[str]:
    """Return the case ids this annotator works through, in order.

    Args:
        args: Parsed arguments.
        manifest_dir: The substrate's frozen split manifest directory.

    Returns:
        The queue.

    Raises:
        FileNotFoundError: If no queue file and no reservation can be found — the
            interface must never fall back to "any case", because a case outside the Gold
            sample is one nobody reserved and nobody will ingest.
    """
    if args.queue:
        return Path(args.queue).read_text(encoding="utf-8").split()
    reserved = manifest_dir / "gold_reserved.txt"
    if not reserved.is_file():
        raise FileNotFoundError(
            f"no annotation queue given and no Gold reservation at {reserved}. Run "
            "`make gold-sample` first: annotating a case outside the reserved sample "
            "produces an item that is neither held out nor ingested."
        )
    return reserved.read_text(encoding="utf-8").split()


def _render_flags(st: Any, summary: ValidationSummary) -> tuple[dict[str, bool], dict[str, str]]:
    """Render the live flag panel and collect override decisions.

    Args:
        st: The Streamlit module.
        summary: The current validation summary.

    Returns:
        ``(overrides, notes)`` keyed by flag rule.
    """
    overrides: dict[str, bool] = {}
    notes: dict[str, str] = {}
    if not summary.flags:
        st.success("No flags. Check the salience list before you submit.")
        return overrides, notes

    for i, flag in enumerate(summary.flags):
        getattr(st, _CALLOUT[flag.severity])(
            f"**{flag.rule}** — {flag.message}" + (f"\n\n> {flag.excerpt}" if flag.excerpt else "")
        )
        if flag.severity is not Severity.INFO:
            notes[flag.rule] = st.text_input(
                "Why is this correct as written? (optional)",
                key=f"note-{i}-{flag.rule}",
            )
            overrides[flag.rule] = True
    return overrides, notes


def _render_panel(st: Any, panel: FactPanel) -> None:
    """Render the fact record as a readable structured summary.

    Args:
        st: The Streamlit module.
        panel: The panel to render.
    """
    st.caption(f"{panel.case_id} · {panel.dataset}")
    if panel.masked_families:
        st.warning(
            "**Not available on this substrate — do not write about it:**\n\n"
            + "\n".join(f"- {m}" for m in panel.masked_families)
        )
    if panel.typology_source == "inferred":
        st.info(
            f"Typology **{panel.typology}** was inferred from motif detection, not read "
            "from ground truth. It must appear inside a hedge."
        )
    elif panel.typology_scope == "stream_membership":
        st.info(
            f"This case is *part of* a **{panel.typology}** stream and may not show it in "
            "full. Do not describe the scheme as complete."
        )

    for section in panel.sections:
        with st.expander(section.name, expanded=section.name in ("Subject", "Scope")):
            if section.blurb:
                st.caption(section.blurb)
            for row in section.rows:
                marker = "**·**" if row.salient else ""
                suffix = f"  \n<small>{row.note}</small>" if row.note else ""
                st.markdown(
                    f"{marker} {row.label}: **{row.value}**{suffix}",
                    unsafe_allow_html=True,
                )


def main(argv: list[str] | None = None) -> None:  # noqa: PLR0915 - one linear page; the
    # layout is the function, and splitting it would separate a widget from the state it
    # reads.  # pragma: no cover - the app entrypoint
    """Run the annotation interface.

    Args:
        argv: Command-line arguments. Streamlit passes everything after ``--``.

    Raises:
        ImportError: If Streamlit or plotly is not installed.
    """
    st = _require_streamlit()
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()

    st.set_page_config(page_title="Gold annotation", layout="wide")

    cfg = _compose(root, args.dataset)
    processed_dir = Path(cfg.paths.processed_dir) / args.dataset
    interim_dir = Path(cfg.paths.interim_dir) / args.dataset
    manifest_dir = Path(cfg.data.split.manifest_dir)

    @st.cache_resource
    def _source(processed: str, interim: str) -> CaseSource:
        return CaseSource(processed_dir=Path(processed), interim_dir=Path(interim))

    source = _source(str(processed_dir), str(interim_dir))
    queue = _queue_for(args, manifest_dir)
    store = AnnotationStore(root=processed_dir / "gold" / "annotations")
    done = {a.case_id for a in store.read(args.annotator) if a.is_calibration == args.calibration}
    remaining = [c for c in queue if c not in done]

    with st.sidebar:
        st.metric("Annotator", args.annotator)
        st.metric("Completed", f"{len(done)} / {len(queue)}")
        if not remaining:
            st.success("Queue complete.")
            st.stop()
        case_id = st.selectbox("Case", remaining, index=0)
        st.divider()
        st.caption("Allowed hedges")
        for hedge in load_vocabulary().hedging_allowed:
            st.markdown(f"- {hedge}")

    if st.session_state.get("case_id") != case_id:
        st.session_state["case_id"] = case_id
        st.session_state["session"] = AnnotationSession(
            case=source.load(case_id),
            annotator_id=args.annotator,
            started_at=time.monotonic(),
            is_calibration=args.calibration,
        )
        st.session_state["draft"] = starter_template()

    session: AnnotationSession = st.session_state["session"]
    panel = build_fact_panel(session.case.facts)
    view = build_graph_view(session.case.view, session.case.focal_id, max_nodes=args.max_nodes)

    left, right = st.columns([5, 4], gap="large")

    with left:
        st.subheader("The case")
        until = None
        if len(view.timeline) > 1:
            step = st.select_slider(
                "Show transactions up to",
                options=list(view.timeline),
                value=view.timeline[-1],
                format_func=lambda t: t.strftime("%Y-%m-%d %H:%M"),
            )
            until = step
        st.plotly_chart(to_plotly_figure(view, until=until), use_container_width=True)
        if view.truncated:
            st.error(view.caption)

    with right:
        st.subheader("The record")
        _render_panel(st, panel)

    st.divider()
    st.subheader("The narrative")
    draft = st.text_area(
        "Four parts. Assert only what the record supports.",
        key="draft",
        height=380,
        label_visibility="visible",
    )
    session.note_draft(draft)
    summary = session.validate(draft)

    counters = st.columns(4)
    counters[0].metric("Tokens", summary.n_tokens, delta=None if summary.length_ok else "outside")
    counters[1].metric("Sections", f"{len(summary.sections_found)} / 4")
    counters[2].metric(
        "Salient covered",
        f"{len(summary.salient_mentioned)} / "
        f"{len(summary.salient_mentioned) + len(summary.salient_missing)}",
    )
    counters[3].metric("Flags", len(summary.flags), delta=summary.n_critical or None)

    overrides, notes = _render_flags(st, summary)

    st.divider()
    controls = st.columns([2, 1, 3])
    typology = controls[0].selectbox(
        "Typology you judge this to be",
        options=sorted({*load_vocabulary().typologies["amlworld"]["members"]}),
    )
    difficulty = controls[1].slider("Difficulty", 1, 5, 3)
    comment = controls[2].text_input("Anything the form does not capture")

    contradictions = session.check(draft) if session.checked_once else []
    if st.button("Check against the record", type="secondary"):
        contradictions = session.check(draft)
        session.checked_once = True
        if contradictions:
            for result in contradictions:
                st.error(f"CONTRADICTED — {result.reason}")
        else:
            st.success("No contradictions against the fact record.")

    if st.button("Submit", type="primary", disabled=not session.checked_once):
        annotation = session.build_annotation(
            draft,
            summary,
            panel,
            view.to_dict(),
            typology_assigned=typology,
            difficulty=difficulty,
            comment=comment,
            overrides=overrides,
            notes=notes,
        )
        store.append(annotation)
        st.success(f"Saved {annotation.case_id} after {annotation.seconds_spent / 60:.1f} minutes.")
        st.session_state.pop("case_id", None)
        st.rerun()


if __name__ == "__main__":  # pragma: no cover
    main()
