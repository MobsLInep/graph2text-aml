"""Phase 6: the Gold tier — sampling, annotation, calibration, agreement, ingestion.

Gold is the only corpus tier written by people, and the only one that breaks the
circularity of evaluating LLM output against LLM-written references. Everything here
exists to make that claim survivable: which cases (:mod:`sampling`), held out where
(:mod:`reservation`), written how (:mod:`annotation_ui`, :mod:`factpanel`,
:mod:`graphview`, :mod:`validation`), by whom and how well (:mod:`calibration`,
:mod:`agreement`), checked by a second reader (:mod:`review`), and turned into records the
same ten-point harness gates (:mod:`gold_ingest`).

Two rules run through all of it. **An annotator is never shown generated text** — not
Bronze, not Silver, not a model's output — because a reference written beside a draft is a
set of edits to that draft. And **a Gold test item is never trained on**, enforced by
:func:`g2t_aml.corpus.training_data.load_training_records` rather than by anyone
remembering.

Phase 12 adds the decision-setting study to the same package: :mod:`study_design` (who
rates what, blinded), :mod:`study_ui` (the rating interface and its two clocks),
:mod:`study_analysis` (the only module that unblinds) and :mod:`study_release` (the
anonymised deposit). It inverts Phase 6's rule -- a Phase 12 rater is shown generated text
and nothing else, because judging it is the task -- but keeps the same separation of
decision from display, so the block design and every statistic are testable without a
browser.

Re-exported here are the types that cross a phase boundary. The Streamlit entrypoints are
deliberately not among them: importing one pulls in the ``human`` extra, and nothing in
Phases 7-14 should need a UI installed to read an agreement report. Neither is
:func:`~g2t_aml.human.study_analysis.load_blind_key` -- unblinding should cost an explicit
import from the module that documents what it is for.
"""

from g2t_aml.human.agreement import (
    AgreementReport,
    PairAgreement,
    cohens_kappa,
    is_double_annotated,
    measure_agreement,
)
from g2t_aml.human.calibration import (
    AnnotatorCalibration,
    CalibrationItem,
    CalibrationSet,
    build_calibration_set,
    score_annotator,
)
from g2t_aml.human.caseloader import AnnotationCase, CaseSource
from g2t_aml.human.factpanel import FactPanel, build_fact_panel
from g2t_aml.human.gold_ingest import GoldIngestReport, ingest_annotations
from g2t_aml.human.graphview import GraphView, build_graph_view
from g2t_aml.human.reservation import (
    GoldReservation,
    ReservationError,
    assert_not_reserved,
    load_reservation,
    write_reservation,
)
from g2t_aml.human.review import Adjudication, Review, ReviewLog, ReviewVerdict
from g2t_aml.human.sampling import (
    GoldCandidate,
    GoldSample,
    GoldSamplingParams,
    sample_gold_cases,
)
from g2t_aml.human.store import Annotation, AnnotationStore, FlagOutcome
from g2t_aml.human.study_analysis import (
    StudyAnalysis,
    analyse_study,
    durbin_test,
    friedman_test,
    krippendorff_alpha_ordinal,
    nemenyi_posthoc,
    normalised_levenshtein,
)
from g2t_aml.human.study_design import (
    BlindKey,
    DesignError,
    DesignReport,
    StudyDesign,
    StudyItem,
    build_design,
    load_design,
    validate_design,
)
from g2t_aml.human.study_release import ReleaseReport, prepare_release
from g2t_aml.human.study_ui import (
    LIKERT_DIMENSIONS,
    BlurAwareTimer,
    LikertDimension,
    RatingResponse,
    ResponseStore,
)
from g2t_aml.human.validation import LiveFlag, Severity, ValidationSummary, validate_draft

__all__ = [
    "Adjudication",
    "AgreementReport",
    "Annotation",
    "AnnotationCase",
    "AnnotationStore",
    "AnnotatorCalibration",
    "BlindKey",
    "BlurAwareTimer",
    "CalibrationItem",
    "CalibrationSet",
    "CaseSource",
    "DesignError",
    "DesignReport",
    "FactPanel",
    "FlagOutcome",
    "GoldCandidate",
    "GoldIngestReport",
    "GoldReservation",
    "GoldSample",
    "GoldSamplingParams",
    "GraphView",
    "LIKERT_DIMENSIONS",
    "LikertDimension",
    "LiveFlag",
    "PairAgreement",
    "RatingResponse",
    "ReleaseReport",
    "ReservationError",
    "ResponseStore",
    "Review",
    "ReviewLog",
    "ReviewVerdict",
    "Severity",
    "StudyAnalysis",
    "StudyDesign",
    "StudyItem",
    "ValidationSummary",
    "analyse_study",
    "assert_not_reserved",
    "build_calibration_set",
    "build_design",
    "build_fact_panel",
    "build_graph_view",
    "cohens_kappa",
    "durbin_test",
    "friedman_test",
    "ingest_annotations",
    "is_double_annotated",
    "krippendorff_alpha_ordinal",
    "load_design",
    "load_reservation",
    "measure_agreement",
    "nemenyi_posthoc",
    "normalised_levenshtein",
    "prepare_release",
    "sample_gold_cases",
    "score_annotator",
    "validate_design",
    "validate_draft",
    "write_reservation",
]
