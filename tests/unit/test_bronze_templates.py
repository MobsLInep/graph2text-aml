"""The template pack: families, variants, and the vocabulary it is allowed to use.

Two properties matter here and neither is visible by reading the pack. The first is that
every surface form the renderer can produce is one the *controlled vocabulary* licenses —
a template that invented a way to name a role would put an unlicensed phrase into fifteen
thousand training examples. The second is that the display maps are bijections, because a
claim parsed out of the text is only recoverable if the surface form maps back to exactly
one vocabulary member.
"""

from __future__ import annotations

import re

import pytest

from g2t_aml.corpus.bronze.templates import (
    FAMILIES,
    FAMILY_FOR_TYPOLOGY,
    PHASE_DISPLAY,
    ROLE_DISPLAY,
    SALIENCE_SENTENCES,
    TYPOLOGY_DISPLAY,
    family_for,
    n_variants,
)
from g2t_aml.facts.temporal import PHASES
from g2t_aml.facts.vocab import load_vocabulary

MIN_FAMILIES = 8
MIN_VARIANTS = 4
MAX_VARIANTS = 6

_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


def _all_segments():
    for family in FAMILIES.values():
        for index in range(family.n_realisations):
            realisation = family.realisation(index)
            for section in ("subject", "activity", "pattern", "basis"):
                yield family, getattr(realisation, section)


class TestAcceptanceCriteria:
    def test_at_least_eight_families(self) -> None:
        assert len(FAMILIES) >= MIN_FAMILIES

    def test_every_family_has_four_to_six_structural_realisations(self) -> None:
        for key in FAMILIES:
            assert MIN_VARIANTS <= n_variants(key) <= MAX_VARIANTS, key

    def test_composition_multiplies_the_realisation_count(self) -> None:
        """The fix for the 0.81 self-BLEU: sections compose independently (D-042)."""
        for family in FAMILIES.values():
            assert family.n_realisations >= 100 * family.n_surface_variants

    def test_every_amlworld_typology_has_a_family(self) -> None:
        vocabulary = load_vocabulary()
        members = set(vocabulary.typologies["amlworld"]["members"]) - {"unclassified"}
        assert members <= set(FAMILY_FOR_TYPOLOGY)

    def test_a_family_key_is_never_reused_for_two_typologies_ambiguously(self) -> None:
        for key, family in FAMILIES.items():
            assert family.key == key


class TestSurfaceFormsComeFromTheVocabulary:
    def test_every_role_display_form_is_licensed(self) -> None:
        """The renderer may not invent a way of naming a role."""
        vocabulary = load_vocabulary()
        for role, forms in ROLE_DISPLAY.items():
            licensed = {
                variant.lower() for variant in vocabulary.entity_roles[role]["phrase_variants"]
            }
            for form in forms:
                stripped = re.sub(r"^(the|a|an) ", "", form.lower())
                assert (
                    form.lower() in licensed or stripped in licensed
                ), f"{form!r} is not a phrase_variant of role {role!r} in vocab_v1.yaml"

    def test_role_display_covers_every_controlled_role(self) -> None:
        assert set(ROLE_DISPLAY) == set(load_vocabulary().role_names())

    def test_typology_display_covers_every_controlled_typology(self) -> None:
        vocabulary = load_vocabulary()
        members = set(vocabulary.typologies["amlworld"]["members"])
        assert members <= set(TYPOLOGY_DISPLAY)

    def test_phase_display_covers_the_closed_phase_vocabulary(self) -> None:
        assert set(PHASE_DISPLAY) == set(PHASES)


class TestDisplayMapsAreBijections:
    """A surface form that maps back to two members makes a parsed claim ambiguous."""

    @pytest.mark.parametrize(
        "mapping", [PHASE_DISPLAY, TYPOLOGY_DISPLAY], ids=["phase", "typology"]
    )
    def test_scalar_map_is_injective(self, mapping: dict[str, str]) -> None:
        assert len(set(mapping.values())) == len(mapping)

    def test_role_map_is_injective_across_all_forms(self) -> None:
        forms = [form for variants in ROLE_DISPLAY.values() for form in variants]
        assert len(set(forms)) == len(forms)


class TestSegmentsAreWellFormed:
    def test_every_placeholder_is_recognised(self) -> None:
        kinds = {
            "count",
            "money",
            "duration",
            "share",
            "density",
            "timestamp",
            "entity",
            "role",
            "typology",
            "ordering",
            "set",
            "bool",
        }
        for _family, section in _all_segments():
            for segment in section:
                for body in _PLACEHOLDER.findall(segment.text):
                    if body.startswith(("~", "_")):
                        continue
                    _path, _, rest = body.partition(":")
                    kind = rest.partition(":")[0]
                    assert kind in kinds, f"{body!r} uses unknown kind {kind!r}"

    def test_every_section_has_at_least_one_segment(self) -> None:
        for _family, section in _all_segments():
            assert section

    def test_no_segment_contains_a_forbidden_phrase(self) -> None:
        """A forbidden phrase baked into a template would fail every record that used it."""
        vocabulary = load_vocabulary()
        for _family, section in _all_segments():
            for segment in section:
                hit = vocabulary.forbidden_hit(segment.text)
                assert hit is None, f"{segment.text!r} contains forbidden phrase {hit}"

    def test_salience_sentences_are_also_clean_and_annotated(self) -> None:
        vocabulary = load_vocabulary()
        for path, sentence in SALIENCE_SENTENCES.items():
            assert vocabulary.forbidden_hit(sentence) is None
            assert f"{{{path}:" in sentence, f"{path} fallback must annotate its own field"


class TestTopologyFamilyIsSubstrateSafe:
    def test_it_names_no_monetary_temporal_or_label_field(self) -> None:
        """Elliptic2's family cannot assert a masked fact because it cannot name one."""
        family = family_for("topology_only")
        forbidden_prefixes = ("flow.", "temporal.", "labels.")
        for index in range(family.n_realisations):
            realisation = family.realisation(index)
            for name in ("subject", "activity", "pattern", "basis"):
                for segment in getattr(realisation, name):
                    for body in _PLACEHOLDER.findall(segment.text):
                        assert not body.startswith(forbidden_prefixes), body

    def test_it_declares_no_availability_requirement(self) -> None:
        assert family_for("topology_only").requires_mask == ()

    def test_every_other_family_requires_amounts(self) -> None:
        for key, family in FAMILIES.items():
            if key == "topology_only":
                continue
            assert "monetary_amounts" in family.requires_mask
