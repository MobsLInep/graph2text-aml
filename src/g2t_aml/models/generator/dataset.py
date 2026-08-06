"""The dataset, the collator, and the curriculum — with the Gold hold-out enforced here.

Three jobs.

**Refuse Gold test items.** :func:`~g2t_aml.corpus.training_data.load_training_records`
already refuses a reserved case and refuses the Gold tier on a training split, and this
module goes through it rather than reading the JSONL itself. :func:`assert_no_gold_test`
then re-asserts the reservation against the assembled dataset, because the curriculum
concatenates several corpus files and a guarantee that holds for each input separately is
not automatically a guarantee about the concatenation.

**Mask the loss.** Handled position-by-position in
:mod:`~g2t_aml.models.generator.prompts`; this module pads and stacks without disturbing
it, and :func:`loss_mask_report` exists so the test suite can assert that the loss is zero
on prompt and soft-token positions rather than trusting that it is.

**Batch variable-sized graphs.** Case subgraphs range from 2 to 150 nodes, so the graph
side is a PyG ``Batch`` while the text side is a padded rectangle. The collator produces
both and keeps their case order identical — a permutation between them pairs every
narrative with the wrong graph, which is exactly what the A1 control does *deliberately*,
and would otherwise turn every arm into A1 without anyone noticing.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from g2t_aml.corpus.training_data import load_training_records
from g2t_aml.human.reservation import GoldReservation, ReservationError
from g2t_aml.models.generator.prompts import IGNORE_INDEX, BuiltPrompt, PromptBuilder

#: How many offending case ids an error message lists before truncating.
_MAX_LISTED = 5

__all__ = [
    "CurriculumStage",
    "GeneratorBatch",
    "GeneratorDataset",
    "GraphCollator",
    "assert_no_gold_test",
    "build_curriculum",
    "loss_mask_report",
]


@dataclass(frozen=True)
class CurriculumStage:
    """One stage of the training curriculum.

    The default curriculum is Bronze+Silver for epoch 1, Silver only for epochs 2-3, and
    an optional short Gold-train tail. It is expressed as data rather than as branches in
    the trainer so that the ablation — does the curriculum matter at all? — is a config
    change and costs one run rather than a code change.

    Attributes:
        name: Stage name, used in logs and the run record.
        tiers: Which corpus tiers this stage draws from.
        epochs: How many epochs this stage runs for. Fractional values are permitted for
            the Gold tail, where the brief asks for a fixed step count rather than an
            epoch.
        max_steps: Hard step cap for this stage, or None for none. The Gold tail uses
            this.
    """

    name: str
    tiers: tuple[str, ...]
    epochs: float = 1.0
    max_steps: int | None = None


def build_curriculum(cfg: Any) -> tuple[CurriculumStage, ...]:
    """Resolve the curriculum from a config node.

    Args:
        cfg: A sequence of stage mappings, each with ``name``, ``tiers`` and optionally
            ``epochs`` and ``max_steps``. None or empty yields the default three-epoch
            curriculum.

    Returns:
        The stages, in order.

    Raises:
        ValueError: If a stage names no tiers, which would train on nothing.
    """
    if not cfg:
        return (
            CurriculumStage("mixed", ("bronze", "silver"), epochs=1.0),
            CurriculumStage("silver_only", ("silver",), epochs=2.0),
        )
    stages: list[CurriculumStage] = []
    for raw in cfg:
        tiers = tuple(str(t) for t in raw["tiers"])
        if not tiers:
            raise ValueError(f"curriculum stage {raw.get('name')!r} names no tiers")
        stages.append(
            CurriculumStage(
                name=str(raw["name"]),
                tiers=tiers,
                epochs=float(raw.get("epochs", 1.0)),
                max_steps=(int(raw["max_steps"]) if raw.get("max_steps") is not None else None),
            )
        )
    return tuple(stages)


def assert_no_gold_test(
    case_ids: Sequence[str], reservation: GoldReservation | None, *, where: str
) -> None:
    """Assert that no reserved Gold test case appears in a training population.

    The loader already refuses these one file at a time. This re-checks the assembled
    dataset, because the curriculum concatenates corpora and because this is the assertion
    the Phase 9 brief asks to exist in the data loader. It is cheap and the failure it
    catches is silent in every metric the project reports — a memorised target scores
    *better* on faithfulness, adequacy and overlap, so nothing looks wrong.

    Args:
        case_ids: Every case id about to be trained on.
        reservation: The Gold reservation, or None when none has been made.
        where: Description of the population, used in the error message.

    Raises:
        ReservationError: If any reserved case is present.
    """
    if reservation is None:
        return
    reserved = set(reservation.case_ids)
    found = sorted(set(case_ids) & reserved)
    if found:
        raise ReservationError(
            f"{len(found)} reserved Gold test case(s) reached {where}: "
            f"{found[:_MAX_LISTED]}{'...' if len(found) > _MAX_LISTED else ''}. "
            "These are the held-out reference; "
            "training on them inflates every number in the paper."
        )


class GeneratorDataset:
    """Training records for one split, tokenised on demand and paired with their graphs.

    Tokenising lazily rather than up front is deliberate at this corpus size: 15,707
    records at 2048 tokens is ~1.2 GB of int64 held for the whole run on a machine that
    is already tight, and the tokenisation is not the bottleneck.
    """

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        builder: PromptBuilder,
        graphs: dict[str, Any] | None = None,
        reservation: GoldReservation | None = None,
        for_training: bool = True,
    ) -> None:
        """Build the dataset.

        Args:
            records: Training records, already filtered to one split.
            builder: The prompt builder.
            graphs: Case id to PyG ``Data``, from the Phase 7 feature cache. None on a
                text-only arm.
            reservation: The Gold reservation, asserted against every record.
            for_training: Whether to include the target narrative as a completion.

        Raises:
            ReservationError: If a reserved Gold test case is present and this is a
                training dataset.
            KeyError: If a record has no graph and graphs were supplied.
        """
        self.records = list(records)
        self.builder = builder
        self.graphs = graphs
        self.for_training = for_training

        if for_training:
            assert_no_gold_test(
                [str(r["case_id"]) for r in self.records],
                reservation,
                where="the generator training set",
            )
        if graphs is not None:
            missing = [str(r["case_id"]) for r in self.records if str(r["case_id"]) not in graphs]
            if missing:
                raise KeyError(
                    f"{len(missing)} record(s) have no cached graph, first {missing[:3]}; "
                    "rebuild the encoder feature cache or drop them from the corpus"
                )

    def __len__(self) -> int:
        """Return how many examples the dataset holds.

        Returns:
            The record count.
        """
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[BuiltPrompt, Any]:
        """Return one tokenised example and its graph.

        Args:
            index: Position in the split.

        Returns:
            ``(prompt, graph)``; the graph is None on a text-only arm.
        """
        record = self.records[index]
        prompt = self.builder.build(record, for_training=self.for_training)
        graph = None if self.graphs is None else self.graphs[str(record["case_id"])]
        return prompt, graph

    def __iter__(self) -> Iterator[tuple[BuiltPrompt, Any]]:
        """Iterate over the dataset in order.

        Returns:
            An iterator of ``(prompt, graph)`` pairs.
        """
        return (self[i] for i in range(len(self)))

    @property
    def case_ids(self) -> tuple[str, ...]:
        """Return the case ids, in dataset order.

        Returns:
            The ids.
        """
        return tuple(str(r["case_id"]) for r in self.records)


@dataclass
class GeneratorBatch:
    """One collated batch.

    Attributes:
        input_ids: ``[B, T]``, right-padded.
        attention_mask: ``[B, T]``.
        labels: ``[B, T]``, ``-100`` on system, prompt, soft-token and pad positions.
        soft_mask: ``[B, T]``, True at exactly the reserved graph positions.
        graph_batch: A PyG ``Batch``, or None on a text-only arm. **In the same case
            order as the text side.**
        case_ids: The case ids, in batch order, so a diagnostic can name the case.
        completion_starts: Per-row index of the first completion token, for restricting
            the attention-mass diagnostic to generated positions.
        n_facts_truncated: Total fact tokens dropped to fit the sequence budget.
    """

    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor
    soft_mask: Tensor
    graph_batch: Any = None
    case_ids: tuple[str, ...] = ()
    completion_starts: tuple[int, ...] = ()
    n_facts_truncated: int = 0

    def to(self, device: str | torch.device) -> GeneratorBatch:
        """Move the tensor fields to a device.

        Args:
            device: Target device.

        Returns:
            This batch, moved in place and returned for chaining.
        """
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.labels = self.labels.to(device)
        self.soft_mask = self.soft_mask.to(device)
        if self.graph_batch is not None:
            self.graph_batch = self.graph_batch.to(device)
        return self

    @property
    def n_supervised_tokens(self) -> int:
        """Return how many positions contribute to the loss.

        Returns:
            The count of labels that are not ``-100``. Logged per batch: a value of zero
            means an entire batch trained on nothing, which happens when truncation eats
            the completion and is otherwise invisible in a loss that simply reads ``nan``.
        """
        return int((self.labels != IGNORE_INDEX).sum())


@dataclass
class GraphCollator:
    """Pads the text side, batches the graph side, and keeps the two in the same order.

    Attributes:
        pad_token_id: Id used for padding. Masked out of both attention and loss.
        pad_to_multiple_of: Round the padded length up to this multiple. 8 keeps tensor
            cores fed; it costs a few tokens of padding that are masked anyway.
    """

    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, items: Sequence[tuple[BuiltPrompt, Any]]) -> GeneratorBatch:
        """Collate a list of examples into a batch.

        Args:
            items: ``(prompt, graph)`` pairs from :class:`GeneratorDataset`.

        Returns:
            The collated batch.

        Raises:
            ValueError: If the batch is empty, or if some examples carry a graph and
                others do not — a mixed batch cannot be spliced consistently.
        """
        if not items:
            raise ValueError("cannot collate an empty batch")

        prompts = [p for p, _ in items]
        graphs = [g for _, g in items]
        have_graphs = [g is not None for g in graphs]
        if any(have_graphs) and not all(have_graphs):
            raise ValueError(
                "batch mixes examples with and without graphs; the fusion layer would "
                "splice a different number of positions per row"
            )

        width = max(len(p.input_ids) for p in prompts)
        if self.pad_to_multiple_of > 1:
            remainder = width % self.pad_to_multiple_of
            if remainder:
                width += self.pad_to_multiple_of - remainder

        n = len(prompts)
        input_ids = torch.full((n, width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((n, width), dtype=torch.long)
        # -100 rather than 0: padding must never contribute to the loss, and a 0 here
        # would train the model to predict token 0 at every padded position.
        labels = torch.full((n, width), IGNORE_INDEX, dtype=torch.long)
        soft_mask = torch.zeros((n, width), dtype=torch.bool)

        for row, prompt in enumerate(prompts):
            length = len(prompt.input_ids)
            input_ids[row, :length] = torch.tensor(prompt.input_ids, dtype=torch.long)
            attention_mask[row, :length] = 1
            labels[row, :length] = torch.tensor(prompt.labels, dtype=torch.long)
            soft_mask[row, :length] = torch.tensor(prompt.soft_mask, dtype=torch.bool)

        graph_batch = None
        if all(have_graphs):
            from torch_geometric.data import Batch

            graph_batch = Batch.from_data_list(list(graphs))

        return GeneratorBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            soft_mask=soft_mask,
            graph_batch=graph_batch,
            case_ids=tuple(p.case_id for p in prompts),
            completion_starts=tuple(p.completion_start for p in prompts),
            n_facts_truncated=sum(p.n_facts_truncated for p in prompts),
        )


def loss_mask_report(batch: GeneratorBatch) -> dict[str, int]:
    """Count supervised positions by role, so the masking can be asserted rather than assumed.

    Used by the test suite to check the brief's requirement directly: the loss must be
    zero on the prompt and on the soft-token positions.

    Args:
        batch: A collated batch.

    Returns:
        Counts of supervised positions overall, on soft-token positions, on padding, and
        the number of rows with no supervision at all. Every value but
        ``n_supervised`` must be zero in a correct batch.
    """
    supervised = batch.labels != IGNORE_INDEX
    return {
        "n_supervised": int(supervised.sum()),
        "n_supervised_soft": int((supervised & batch.soft_mask).sum()),
        "n_supervised_pad": int((supervised & (batch.attention_mask == 0)).sum()),
        "n_rows_unsupervised": int((~supervised.any(dim=1)).sum()),
    }


def load_curriculum_records(
    corpus_dir: str | Path,
    *,
    split: str,
    tiers: Sequence[str],
    reservation: GoldReservation | None = None,
) -> list[dict[str, Any]]:
    """Load and concatenate the corpora one curriculum stage draws from.

    Args:
        corpus_dir: Directory holding ``bronze.jsonl``, ``silver.jsonl``, ``gold.jsonl``.
        split: The split to load.
        tiers: Which tiers this stage uses.
        reservation: The Gold reservation, passed to the loader and re-asserted on the
            concatenation.

    Returns:
        The records, in tier order then file order.

    Raises:
        FileNotFoundError: If a named tier's corpus file is absent. Silent-skip would let
            a Silver-only stage train on nothing and report a healthy loss over an empty
            set.
        ReservationError: If a reserved case survives into the concatenation.
    """
    root = Path(corpus_dir)
    records: list[dict[str, Any]] = []
    for tier in tiers:
        path = root / f"{tier}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"curriculum stage needs tier {tier!r} but {path} does not exist; "
                "build it or remove the tier from the curriculum"
            )
        records.extend(load_training_records(path, split=split, reservation=reservation).records)
    assert_no_gold_test(
        [str(r["case_id"]) for r in records], reservation, where="the assembled curriculum stage"
    )
    return records
