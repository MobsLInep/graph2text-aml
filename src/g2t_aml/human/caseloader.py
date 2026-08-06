"""Loading one case for annotation: the record, the subgraph, and nothing else.

The interface needs three things per item — the fact record, the case subgraph, and the
salience list. This module assembles them, and its narrow public surface is the point:
:class:`AnnotationCase` has no field for a narrative of any kind, so there is no route by
which a Bronze rendering or a model output reaches the screen. Gold's independence is a
property of what can be loaded, not of what the UI chooses to display.

**The interim graph is loaded once and shared.** Case membership is stored as index lists
into the whole 515,088-account graph (Phase 2 writes two columnar tables, not 30,000
files), so materialising a case means holding that graph in memory. At roughly 400 MB it
is loaded once per session and reused, which is why :class:`CaseSource` is a long-lived
object rather than a function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from g2t_aml.corpus.factsio import load_case_facts_file
from g2t_aml.data.canonical import CanonicalGraph
from g2t_aml.data.case_extraction import GraphIndex
from g2t_aml.data.case_sampling import CaseCollection
from g2t_aml.facts.caseview import CaseView, build_case_view
from g2t_aml.facts.salience import SalienceReport, salience_report
from g2t_aml.facts.schema import CaseFacts
from g2t_aml.facts.vocab import ControlledVocabulary

__all__ = ["AnnotationCase", "CaseSource", "CaseSourceError"]


class CaseSourceError(RuntimeError):
    """Raised when a case cannot be assembled for annotation."""


@dataclass(frozen=True)
class AnnotationCase:
    """Everything an annotator may be shown about one case.

    Attributes:
        case_id: The case.
        facts: The fact record.
        view: The subgraph, as the fact layer sees it.
        salience: The required and excused salient fields for this case.

    There is deliberately no narrative field of any kind.
    """

    case_id: str
    facts: CaseFacts
    view: CaseView
    salience: SalienceReport

    @property
    def focal_id(self) -> str:
        """Return the account the case is about.

        Returns:
            The focal account identifier.
        """
        return self.facts.focal_entity.id


@dataclass
class CaseSource:
    """Assembles annotation cases from the processed and interim trees.

    Attributes:
        processed_dir: The substrate's processed directory, holding ``facts/`` and
            ``cases/``.
        interim_dir: The substrate's interim directory, holding the whole canonical graph.
        vocabulary: The controlled vocabulary, for the salience lists.
    """

    processed_dir: Path
    interim_dir: Path
    vocabulary: ControlledVocabulary | None = None
    _graph_index: GraphIndex | None = field(default=None, repr=False)
    _cases: CaseCollection | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Resolve the two directories.

        Raises:
            CaseSourceError: If either is absent, naming the make target that builds it.
        """
        self.processed_dir = Path(self.processed_dir)
        self.interim_dir = Path(self.interim_dir)
        if not (self.processed_dir / "facts").is_dir():
            raise CaseSourceError(
                f"no fact records under {self.processed_dir}; run `make facts` first"
            )
        if not (self.processed_dir / "cases" / "cases.jsonl").is_file():
            raise CaseSourceError(
                f"no case corpus under {self.processed_dir}; run `make cases` first"
            )
        if not self.interim_dir.is_dir():
            raise CaseSourceError(f"no ingested graph at {self.interim_dir}; run `make data` first")

    def _index(self) -> GraphIndex:
        """Return the shared index over the whole substrate graph.

        Built on first use and kept for the source's lifetime: the graph is hundreds of
        megabytes and every case materialisation needs it, so loading per case would make
        the interface unusable. Cached on the instance rather than with ``lru_cache``
        because the cache would key on ``self`` and this class is mutable.

        Returns:
            The graph index.
        """
        if self._graph_index is None:
            self._graph_index = GraphIndex(CanonicalGraph.load(self.interim_dir))
        return self._graph_index

    def _collection(self) -> CaseCollection:
        """Return the case index, loading it once.

        Returns:
            The collection.
        """
        if self._cases is None:
            self._cases = CaseCollection.load(self.processed_dir / "cases")
        return self._cases

    def case_ids(self) -> tuple[str, ...]:
        """Return every case the source can supply.

        Returns:
            The case ids, in build order.
        """
        return tuple(self._collection().case_ids)

    def load(self, case_id: str) -> AnnotationCase:
        """Assemble one case for annotation.

        Args:
            case_id: The case.

        Returns:
            The record, the subgraph and the salience list.

        Raises:
            CaseSourceError: If the case has no fact record, or is not in the case index.
        """
        facts_path = self.processed_dir / "facts" / f"{case_id}.json"
        if not facts_path.is_file():
            raise CaseSourceError(
                f"case {case_id!r} has no fact record at {facts_path}. An annotator must "
                "never be shown a case whose facts have not been extracted: there would "
                "be nothing to check the narrative against."
            )
        try:
            graph = self._collection().materialise(case_id, self._index())
        except KeyError as exc:
            raise CaseSourceError(f"case {case_id!r} is not in the case index") from exc

        facts = load_case_facts_file(facts_path)
        return AnnotationCase(
            case_id=case_id,
            facts=facts,
            view=build_case_view(graph),
            salience=salience_report(facts, self.vocabulary),
        )
