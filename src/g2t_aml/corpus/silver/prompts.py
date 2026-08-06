"""The rewrite and repair prompts: loaded from versioned files, hashed, rendered.

**The prompt is an artifact, not a string literal.** It lives in ``prompts/`` as a file
with a content hash, and every record Silver writes carries the hash of the exact prompt
that produced it. API model behaviour changes underneath a project without announcement;
six weeks after a run, a shifted distribution has to be attributable to the prompt or to
the model, and without the hash it is attributable to neither.

**The rewriter sees the fact record and the Bronze narrative. It never sees the raw
graph.** That is enforced here, by what this module is able to put in a prompt: there is no
code path from a case subgraph to a rendered prompt, and :func:`build_rewrite_prompt` takes
a :class:`~g2t_aml.facts.schema.CaseFacts` and a narrative string. It is a structural limit
on hallucination rather than an instruction the model may ignore — a model cannot invent
the shape of something it was not shown, and if it does, the claim is unaligned to any slot
and the extractor treats it as a candidate addition.

**Availability is stated positively and negatively.** Invariant 4 says nothing may assert a
fact that does not exist for its substrate. Telling a model only what it may write leaves
it to infer the complement; telling it explicitly that this substrate has no entity types
and no jurisdictions, and that those must be omitted rather than hedged, is the form that
survives a model deciding to be helpful.

**The system/user split is a cost control, not formatting.** Everything identical across
the corpus — the role, the vocabulary, the SAR structure, the rewrite rules — is in the
system message, and everything per-case is in the user message. Over ~12k calls per teacher
that prefix is a prompt-cache hit rather than ~900 tokens of fresh input on every request.
:func:`assert_system_is_case_invariant` exists because the saving is silent when it breaks:
a per-case value that drifts up into the system message turns every request into a cache
write, and nothing about the output would look wrong.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from g2t_aml.facts.schema import CaseFacts
from g2t_aml.facts.serialiser import serialise_facts
from g2t_aml.facts.vocab import ControlledVocabulary, load_vocabulary

__all__ = [
    "CASE_INVARIANT_PLACEHOLDERS",
    "PROMPTS_DIR",
    "REPAIR_PROMPT_NAME",
    "REWRITE_PROMPT_NAME",
    "STYLE_DIRECTIVES",
    "PromptRenderError",
    "PromptTemplate",
    "RenderedPrompt",
    "Violation",
    "assert_system_is_case_invariant",
    "build_repair_prompt",
    "build_rewrite_prompt",
    "load_prompt",
    "prompt_hash",
    "style_directive_for",
]

#: Where the versioned prompt files live. Resolved from this module's position because a
#: prompt is source, not data: it is committed, reviewed and hashed alongside the code
#: that renders it, and the two must never disagree.
PROMPTS_DIR = Path(__file__).resolve().parents[4] / "prompts"

REWRITE_PROMPT_NAME = "silver_rewrite_v1"
REPAIR_PROMPT_NAME = "silver_repair_v1"

#: Section markers inside a prompt file. Both sections are hashed together, so a change to
#: either changes the recorded hash.
#:
#: **A marker counts only on a line of its own.** Matching it anywhere would let a header
#: comment that *names* the markers split the file at the comment — which is exactly what
#: happened the first time this was run, silently producing a 141-character system message
#: and moving the whole instruction block into the per-case half. The failure was invisible
#: in the rendered prompt and would have shown up only as a prompt-cache miss rate.
_SYSTEM_MARKER = "<<<SYSTEM>>>"
_USER_MARKER = "<<<USER>>>"
_SECTION_RE = re.compile(r"^<<<(SYSTEM|USER)>>>[ \t]*$", re.MULTILINE)

#: Placeholders are lower-case identifiers only. Deliberately narrow: the prompt body
#: quotes ``{field.path|value}`` from the Bronze annotated form, and a wider pattern would
#: try to substitute it. A placeholder that fails to match is a placeholder that renders
#: literally, so the grammar is small enough that the failure cannot be subtle.
_PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")

#: Word band offered to the teacher. Narrower than the harness's [80, 400] token gate, in
#: words rather than tokens, and deliberately inside it: a model asked for the limit writes
#: to the limit, and a corpus whose every narrative is maximal teaches the generator that a
#: report is always maximal.
DEFAULT_MIN_WORDS = 130
DEFAULT_MAX_WORDS = 260

#: Placeholders the system message is allowed to use. Every one is a function of the
#: controlled vocabulary and the configured word band — constant across the corpus — so
#: substituting them leaves the system message byte-identical for every case, which is what
#: makes it a prompt-cache prefix.
CASE_INVARIANT_PLACEHOLDERS = frozenset(
    {"hedging_block", "forbidden_block", "min_words", "max_words"}
)

#: Surface directives, one drawn deterministically per case.
#:
#: **This is the diversity mechanism for teachers that cannot accept sampling parameters.**
#: The frontier Anthropic models reject ``temperature`` and ``top_p`` outright (a 400, not a
#: warning), so a two-teacher corpus cannot rely on temperature alone for surface variety
#: without excluding those models. A directive selected by a hash of the case id gives
#: variation that is reproducible, auditable and recorded — the rendered prompt hash on
#: every record covers it — rather than variation nobody can reconstruct. Each directive
#: constrains surface form only; none of them licenses a change of fact.
STYLE_DIRECTIVES: tuple[str, ...] = (
    "Open the subject paragraph with the account rather than with the case. Prefer active "
    "voice throughout.",
    "Lead the activity paragraph with the time window before the amounts. Keep sentences " "short.",
    "Use one longer, subordinated sentence per paragraph among the shorter ones. Avoid "
    "beginning consecutive sentences with the same word.",
    "Write the pattern paragraph as an observation followed by its structural evidence, "
    "rather than the reverse.",
    "Prefer nominal phrasing for the counterparty counts and verbal phrasing for the "
    "movements of value.",
    "Place the strongest quantitative evidence last in the basis paragraph, as the "
    "closing clause.",
    "Vary paragraph length deliberately: one noticeably shorter paragraph among the four.",
    "Introduce the typology by what the data shows before naming it, not after.",
)


class PromptRenderError(ValueError):
    """Raised when a prompt cannot be rendered.

    Always a bug in the caller or in the prompt file: an unsupplied placeholder, an
    unknown one, or a malformed file. Never a data condition — a fact record that cannot
    be described is caught by the renderer in Phase 4, long before a prompt is built.
    """


@dataclass(frozen=True)
class PromptTemplate:
    """One versioned prompt file, parsed and hashed.

    Attributes:
        name: The file stem, e.g. ``"silver_rewrite_v1"``.
        path: Where it was loaded from.
        system: The system message, verbatim.
        user: The user message, with placeholders unsubstituted.
        content_hash: SHA-256 of the whole file, header comment included. **This is the
            hash recorded on every generated record.** The header is inside it on purpose:
            editing the rationale for a prompt is editing the artifact, and a run that
            cannot be told apart from a run under different reasoning is not reproducible.
        placeholders: Every placeholder the file uses, across both sections, sorted.
        system_placeholders: Those the system message uses. Asserted to be a subset of
            :data:`CASE_INVARIANT_PLACEHOLDERS` at load time.
    """

    name: str
    path: Path
    system: str
    user: str
    content_hash: str
    placeholders: tuple[str, ...]
    system_placeholders: tuple[str, ...] = ()

    def render(self, values: dict[str, str]) -> RenderedPrompt:
        """Substitute values into both sections.

        Args:
            values: Placeholder name to replacement text. Must cover
                :attr:`placeholders` exactly.

        Returns:
            The rendered prompt, carrying this template's content hash, a hash of the
            rendered bytes, and a hash of the system message alone.

        Raises:
            PromptRenderError: If a placeholder is unsupplied or an unknown key is
                offered. Strict in both directions: a missing key would render a prompt
                with a literal ``{salient_block}`` in it, and an extra key means the
                caller believes it is controlling something the prompt does not read.
        """
        supplied = set(values)
        expected = set(self.placeholders)
        if missing := sorted(expected - supplied):
            raise PromptRenderError(f"prompt {self.name!r} needs values for {missing}")
        if extra := sorted(supplied - expected):
            raise PromptRenderError(
                f"prompt {self.name!r} was given {extra}, which it does not use; the "
                "caller is controlling something the prompt does not read"
            )
        system = _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], self.system)
        user = _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], self.user)
        return RenderedPrompt(
            template_name=self.name,
            prompt_hash=self.content_hash,
            system=system,
            user=user,
            rendered_hash=prompt_hash(system + "\n" + user),
            system_hash=prompt_hash(system),
        )


@dataclass(frozen=True)
class RenderedPrompt:
    """A prompt ready to send, with both hashes it needs to be reproducible.

    Attributes:
        template_name: Which prompt file.
        prompt_hash: Content hash of that file. Recorded on the generated record.
        system: System message.
        user: User message, fully substituted.
        rendered_hash: Hash of the exact text sent. Part of the cache key, so a change to
            the fact record or the Bronze draft misses the cache even when the template
            has not moved.
        system_hash: Hash of the rendered system message alone. Recorded so a prompt-cache
            miss rate can be explained after the fact: if this varies across a run, the
            case-invariance of the system prefix has broken and every request paid a cache
            write.
    """

    template_name: str
    prompt_hash: str
    system: str
    user: str
    rendered_hash: str
    system_hash: str = ""

    def to_provenance(self) -> dict[str, str]:
        """Return the prompt fields recorded on a generated record.

        Returns:
            Template name and the three hashes. The prompt text itself is deliberately not
            included: it embeds the whole fact record, which the record already carries.
        """
        return {
            "prompt_name": self.template_name,
            "prompt_hash": self.prompt_hash,
            "rendered_prompt_hash": self.rendered_hash,
            "system_prompt_hash": self.system_hash,
        }


@dataclass(frozen=True)
class Violation:
    """One verification failure, phrased for a repair prompt.

    Attributes:
        field_path: The fact field the claim was about, or None for a text-level
            violation such as a forbidden phrase.
        quoted: The text in the narrative that failed.
        verdict: ``"contradicted"`` or ``"unverifiable"``.
        hallucination_class: The H-class, when the checker assigned one.
        reason: The checker's explanation.
    """

    field_path: str | None
    quoted: str
    verdict: str
    hallucination_class: str | None
    reason: str

    def to_line(self) -> str:
        """Render the violation as one line of the repair prompt.

        **The expected value is deliberately absent.** A model handed
        ``expected: 26,779.82`` pastes that string; a model told that
        ``flow.total_inflow`` disagrees with the record has to go and read the record,
        which is the behaviour that generalises to the claims the checker did not catch.

        Returns:
            One bullet.
        """
        where = f"the claim about `{self.field_path}`" if self.field_path else "the text"
        marker = f" [{self.hallucination_class}]" if self.hallucination_class else ""
        return (
            f"- {self.verdict.upper()}{marker}: {where}, written as {self.quoted!r}. {self.reason}"
        )


def prompt_hash(text: str) -> str:
    """Return the SHA-256 of a prompt string.

    Args:
        text: The text to hash.

    Returns:
        The hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=8)
def load_prompt(name: str, prompts_dir: str | None = None) -> PromptTemplate:
    """Load, parse, hash and cache a prompt file.

    Args:
        name: The file stem, without ``.txt``.
        prompts_dir: Override for :data:`PROMPTS_DIR`. Present for tests; production code
            passes nothing.

    Returns:
        The parsed template.

    Raises:
        FileNotFoundError: If the prompt file is missing.
        PromptRenderError: If the file does not hold exactly one system section followed
            by exactly one user section, each on a line of its own. A prompt file that
            parsed leniently would silently mis-split and change model behaviour with no
            diff to point at.
    """
    root = Path(prompts_dir) if prompts_dir is not None else PROMPTS_DIR
    path = root / f"{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"prompt {name!r} not found at {path}")
    raw = path.read_text(encoding="utf-8")

    markers = list(_SECTION_RE.finditer(raw))
    if [m.group(1) for m in markers] != ["SYSTEM", "USER"]:
        raise PromptRenderError(
            f"prompt file {path} must hold exactly one {_SYSTEM_MARKER} line followed by "
            f"exactly one {_USER_MARKER} line, each alone on its line; found "
            f"{[m.group(1) for m in markers]}"
        )
    system = raw[markers[0].end() : markers[1].start()]
    user = raw[markers[1].end() :]

    system_text, user_text = system.strip(), user.strip()
    system_placeholders = tuple(sorted(set(_PLACEHOLDER_RE.findall(system_text))))
    assert_system_is_case_invariant(name, system_placeholders)

    return PromptTemplate(
        name=name,
        path=path,
        system=system_text,
        user=user_text,
        content_hash=prompt_hash(raw),
        placeholders=tuple(
            sorted(
                set(_PLACEHOLDER_RE.findall(system_text)) | set(_PLACEHOLDER_RE.findall(user_text))
            )
        ),
        system_placeholders=system_placeholders,
    )


def assert_system_is_case_invariant(name: str, system_placeholders: tuple[str, ...]) -> None:
    """Refuse a prompt whose system message depends on the case.

    The system message is the prompt-cache prefix across every call in a run. A per-case
    placeholder in it — the fact record, the Bronze draft, the salient list — makes the
    prefix different on every request, so every call pays a cache *write* instead of a
    read. Nothing about the generated text would look wrong; only the bill would change,
    and only in aggregate. That is exactly the class of mistake that has to fail loudly at
    load time rather than be noticed at the end of a run.

    Args:
        name: The prompt name, for the error message.
        system_placeholders: Placeholders found in the system section.

    Raises:
        PromptRenderError: If the system section uses a placeholder outside
            :data:`CASE_INVARIANT_PLACEHOLDERS`.
    """
    if offending := sorted(set(system_placeholders) - CASE_INVARIANT_PLACEHOLDERS):
        raise PromptRenderError(
            f"prompt {name!r} uses {offending} in its system message, but those vary per "
            "case. The system message is the prompt-cache prefix for the whole run; a "
            "per-case value in it turns every request into a cache write. Move them into "
            "the user section."
        )


def style_directive_for(case_id: str) -> str:
    """Select this case's surface directive, deterministically.

    Args:
        case_id: The case being rewritten.

    Returns:
        One of :data:`STYLE_DIRECTIVES`, chosen by a SHA-256 of the case id so the same
        case draws the same directive on every machine and every re-run — the same
        discipline :func:`g2t_aml.corpus.bronze.renderer.select_variant` applies to
        template realisations, and for the same reason: a corpus whose surface varies with
        a global seed cannot be regenerated from a case manifest.
    """
    digest = hashlib.sha256(f"{case_id}|silver-style".encode()).digest()
    return STYLE_DIRECTIVES[int.from_bytes(digest[:8], "big") % len(STYLE_DIRECTIVES)]


def _availability_blocks(facts: CaseFacts) -> tuple[str, str]:
    """Render the substrate's availability mask as two explicit lists.

    Invariant 4 in the form a language model will act on. Stating only what is available
    leaves the complement to inference; naming the unavailable classes and saying that
    they must be omitted rather than hedged is what survives a model trying to be helpful
    about a missing amount.

    Args:
        facts: The record whose mask is described.

    Returns:
        ``(unavailable_block, available_block)``, each a bullet list.
    """
    mask = facts.availability.to_dict()
    unavailable = [
        f"- `{flag}` — this substrate has no such data" for flag, ok in mask.items() if not ok
    ]
    available = [f"- `{flag}`" for flag, ok in mask.items() if ok]
    return (
        "\n".join(unavailable) or "- (none — every fact class is available)",
        "\n".join(available) or "- (none)",
    )


def _hedging_block(vocabulary: ControlledVocabulary) -> str:
    """Render the allowed hedging phrases.

    Args:
        vocabulary: The controlled vocabulary.

    Returns:
        A bullet list of permitted phrases.
    """
    return "\n".join(f"- {phrase}" for phrase in vocabulary.hedging_allowed)


def _forbidden_block(vocabulary: ControlledVocabulary) -> str:
    """Render the forbidden-phrase lists, grouped, with their hallucination classes.

    Grouped rather than flattened because the group name carries the reason: a model told
    that ``mixer`` sits under ``entity_type [H4]`` alongside ``shell company`` has been
    told what kind of statement is banned, not merely which twelve strings.

    Args:
        vocabulary: The controlled vocabulary.

    Returns:
        A bullet list, one line per group.
    """
    lines = []
    for group, (hallucination_class, phrases) in sorted(vocabulary.forbidden.items()):
        rendered = ", ".join(f'"{p}"' for p in sorted(phrases))
        lines.append(f"- **{group}** [{hallucination_class}]: {rendered}")
    return "\n".join(lines)


def _salient_block(facts: CaseFacts, vocabulary: ControlledVocabulary) -> str:
    """Render the salient fields for this record's typology, filtered by availability.

    Filtered rather than declared wholesale: an Elliptic2 case must not be told to mention
    ``flow.total_outflow``, because no narrative could mention it faithfully. This is the
    same filter :func:`g2t_aml.facts.salience.required_fields` applies when adequacy is
    scored, so the instruction and the measurement read one list.

    Args:
        facts: The record.
        vocabulary: The controlled vocabulary.

    Returns:
        A bullet list of field paths.
    """
    from g2t_aml.facts.salience import required_fields  # local: avoids a cycle at import

    required, _ = required_fields(facts, vocabulary)
    return "\n".join(f"- `{path}`" for path in required) or "- (none)"


def _inferred_hedge_block(facts: CaseFacts, vocabulary: ControlledVocabulary) -> str:
    """Render the hedge requirement for an inferred typology, or a note that none applies.

    Args:
        facts: The record.
        vocabulary: The controlled vocabulary.

    Returns:
        A paragraph, always non-empty so the prompt never renders a bare gap.
    """
    if facts.typology.source != "inferred":
        return (
            "This case's typology comes from ground truth rather than inference, so no "
            "additional hedge is required on the typology statement itself."
        )
    phrases = ", ".join(f'"{p}"' for p in vocabulary.required_for_inferred)
    return (
        "**This case's typology is inferred, not observed.** The typology statement must "
        f"appear inside one of these hedges: {phrases}. An unhedged inferred typology is "
        "an H5 violation and the narrative will be rejected."
    )


def build_rewrite_prompt(
    facts: CaseFacts,
    bronze_narrative: str,
    bronze_annotated: str,
    *,
    vocabulary: ControlledVocabulary | None = None,
    min_words: int = DEFAULT_MIN_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
    prompts_dir: str | None = None,
) -> RenderedPrompt:
    """Build the rewrite prompt for one case.

    Note what this signature cannot accept: there is no parameter for the case subgraph,
    the transaction table or the node inventory beyond what the fact record carries. The
    rewriter's view of the world is the fact record and the Bronze draft, and that is a
    property of the call, not of the wording inside it.

    Args:
        facts: The fact record. The only source of facts.
        bronze_narrative: The Bronze draft, plain text.
        bronze_annotated: The Bronze draft with its slots marked, so the teacher can see
            which spans are load-bearing.
        vocabulary: The controlled vocabulary. Loaded from disk when omitted.
        min_words: Lower end of the target band.
        max_words: Upper end of the target band.
        prompts_dir: Override for the prompt directory. Tests only.

    Returns:
        The rendered prompt.

    Raises:
        PromptRenderError: If the template and this function disagree about placeholders.
        FileNotFoundError: If the prompt file is missing.
    """
    vocab = vocabulary if vocabulary is not None else load_vocabulary()
    unavailable, available = _availability_blocks(facts)
    template = load_prompt(REWRITE_PROMPT_NAME, prompts_dir)
    return template.render(
        {
            "fact_record": serialise_facts(facts, style="verbose"),
            "unavailable_block": unavailable,
            "available_block": available,
            "bronze_narrative": bronze_narrative,
            "bronze_annotated": bronze_annotated,
            "typology_label": facts.typology.label,
            "salient_block": _salient_block(facts, vocab),
            "hedging_block": _hedging_block(vocab),
            "forbidden_block": _forbidden_block(vocab),
            "inferred_hedge_block": _inferred_hedge_block(facts, vocab),
            "style_directive": style_directive_for(facts.case_id),
            "min_words": str(min_words),
            "max_words": str(max_words),
        }
    )


def build_repair_prompt(
    facts: CaseFacts,
    draft_narrative: str,
    violations: list[Violation],
    *,
    vocabulary: ControlledVocabulary | None = None,
    min_words: int = DEFAULT_MIN_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
    prompts_dir: str | None = None,
) -> RenderedPrompt:
    """Build the repair prompt for a rewrite that failed verification.

    Args:
        facts: The fact record.
        draft_narrative: The rewrite that failed.
        violations: What the checker found. Must be non-empty — a repair prompt with no
            violations would ask a model to change a narrative that passed.
        vocabulary: The controlled vocabulary. Loaded from disk when omitted.
        min_words: Lower end of the target band.
        max_words: Upper end of the target band.
        prompts_dir: Override for the prompt directory. Tests only.

    Returns:
        The rendered prompt.

    Raises:
        PromptRenderError: If ``violations`` is empty, or the template and this function
            disagree about placeholders.
        FileNotFoundError: If the prompt file is missing.
    """
    if not violations:
        raise PromptRenderError(
            "a repair prompt needs at least one violation; repairing a narrative that "
            "passed verification can only make it worse"
        )
    vocab = vocabulary if vocabulary is not None else load_vocabulary()
    unavailable, _ = _availability_blocks(facts)
    template = load_prompt(REPAIR_PROMPT_NAME, prompts_dir)
    return template.render(
        {
            "fact_record": serialise_facts(facts, style="verbose"),
            "unavailable_block": unavailable,
            "draft_narrative": draft_narrative,
            "violation_block": "\n".join(v.to_line() for v in violations),
            "hedging_block": _hedging_block(vocab),
            "forbidden_block": _forbidden_block(vocab),
            "min_words": str(min_words),
            "max_words": str(max_words),
        }
    )
