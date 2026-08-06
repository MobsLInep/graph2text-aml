"""Contract tests: schema, code and vocabulary must not drift apart.

These are the tests that fail when someone adds a field and forgets the rest of the
system. Each one guards a specific way the three artifacts could silently disagree.
"""

from __future__ import annotations

import json

import pytest
from tests.factories import fan_out_case

from g2t_aml.data.canonical import TYPOLOGY_VOCABULARY
from g2t_aml.facts.checkers import CHECKER_REGISTRY, checkable_field_paths
from g2t_aml.facts.config import FactConfig
from g2t_aml.facts.extractor import ROLE_VOCABULARY, extract_facts
from g2t_aml.facts.schema import (
    CASE_FACTS_SCHEMA_VERSION,
    MOTIF_NAMES,
    facts_to_dict,
    load_case_facts_schema,
)
from g2t_aml.facts.taxonomy import CRITICAL_CLASSES, HallucinationClass, Severity, class_by_id
from g2t_aml.facts.vocab import VOCAB_VERSION, load_vocabulary, parse_condition

#: Groups whose leaves are metadata rather than assertable facts. A narrative does not
#: make claims about the schema version or the provenance block, so they need no checker.
_UNCHECKABLE_GROUPS = {
    "schema_version",
    "case_id",
    "dataset",
    "extractor_version",
    "availability",
    "provenance",
}


def _is_leaf_object(value: dict) -> bool:
    """Report whether a nested object is semantically a single value.

    An availability sentinel and a Money amount are both objects in JSON but single
    facts in the record: a narrative claims "USD 482,300", not a value and a currency
    separately, and the checker is registered at that level.
    """
    keys = set(value)
    return "available" in keys or keys == {"value", "currency"}


def _leaf_paths(payload: dict, prefix: str = "") -> set[str]:
    """Collect dotted paths to every assertable leaf of a serialised record."""
    paths: set[str] = set()
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and not _is_leaf_object(value):
            paths |= _leaf_paths(value, path)
        else:
            paths.add(path)
    return paths


# ------------------------------------------------------- checker coverage ---


def test_every_checkable_schema_field_has_a_registered_checker():
    # THE coverage test the brief asks for. A field added to the schema without a checker
    # would be silently unverifiable, which inflates nothing and hides everything.
    payload = facts_to_dict(extract_facts(fan_out_case(width=5)))
    leaves = _leaf_paths(payload)
    checkable = {p for p in leaves if p.split(".")[0] not in _UNCHECKABLE_GROUPS}

    missing = sorted(p for p in checkable if p not in CHECKER_REGISTRY)
    assert not missing, f"schema fields with no registered checker: {missing}"


def test_no_checker_is_registered_for_a_field_that_does_not_exist():
    # The other direction: a stale registration would silently never fire.
    payload = facts_to_dict(extract_facts(fan_out_case(width=5)))
    leaves = _leaf_paths(payload)
    # model_signal leaves collapse when the block is null, so allow its declared paths.
    allowed = leaves | {
        "model_signal.gnn_risk_score",
        "model_signal.score_percentile",
        "model_signal.top_contributing_nodes",
        "model_signal.model_version",
    }
    stale = sorted(p for p in CHECKER_REGISTRY if p not in allowed)
    assert not stale, f"checkers registered for non-existent fields: {stale}"


def test_registry_has_no_duplicate_registration():
    assert len(checkable_field_paths()) == len(set(checkable_field_paths()))


# ------------------------------------------------------ vocabulary <-> code ---


def test_role_vocabulary_agrees_across_schema_code_and_yaml():
    vocab = load_vocabulary()
    schema = load_case_facts_schema()
    schema_roles = set(schema["$defs"]["entity_role"]["enum"])
    assert set(ROLE_VOCABULARY) == schema_roles == set(vocab.entity_roles)


def test_entity_type_terms_are_absent_from_the_role_vocabulary():
    # D-029: the exclusion is deliberate. No substrate carries an entity-type column, so
    # these words must not be sayable at all rather than corrected after the fact.
    forbidden = {"mixer", "exchange", "tumbler", "merchant", "shell", "casino"}
    assert not (forbidden & {r.lower() for r in ROLE_VOCABULARY})


def test_entity_type_terms_are_on_the_forbidden_list_as_h4():
    vocab = load_vocabulary()
    hallucination_class, phrases = vocab.forbidden["entity_type"]
    assert hallucination_class == "H4"
    assert "mixer" in phrases
    assert HallucinationClass.H4.is_critical


def test_typology_vocabulary_matches_the_canonical_one():
    schema = load_case_facts_schema()
    schema_typologies = set(schema["properties"]["typology"]["properties"]["label"]["enum"])
    assert schema_typologies == set(TYPOLOGY_VOCABULARY)


def test_motif_names_match_the_schema():
    schema = load_case_facts_schema()
    assert set(MOTIF_NAMES) == set(schema["properties"]["motifs"]["properties"])


def test_every_risk_descriptor_binds_to_a_real_checkable_field():
    # A descriptor bound to a misspelled path would be permanently UNVERIFIABLE, and that
    # is invisible in an aggregate rate.
    vocab = load_vocabulary()
    for descriptor in vocab.risk_descriptors.values():
        assert (
            descriptor.binds_to in CHECKER_REGISTRY
        ), f"{descriptor.name} binds to {descriptor.binds_to}, which has no checker"


def test_every_binding_condition_parses():
    vocab = load_vocabulary()
    for descriptor in vocab.risk_descriptors.values():
        operator, threshold = parse_condition(descriptor.condition)
        assert operator
        assert isinstance(threshold, float)


def test_burst_descriptor_threshold_is_strictly_tighter_than_the_detector_cap():
    # burst_window_hours is bounded above by burst_window_hours BY CONSTRUCTION, so a
    # binding at or above the cap is satisfied by every burst the detector can report --
    # a descriptor that always holds is not a claim. See D-026.
    config = FactConfig()
    vocab = load_vocabulary()
    for descriptor in vocab.risk_descriptors.values():
        if descriptor.binds_to != "temporal.burst_window_hours":
            continue
        operator, threshold = parse_condition(descriptor.condition)
        assert operator in {"<", "<="}
        assert threshold < config.burst_window_hours, (
            f"{descriptor.name} binds {descriptor.condition!r} but bursts can never "
            f"exceed {config.burst_window_hours}h, so the condition is vacuous"
        )


def test_every_salience_path_is_a_real_field():
    vocab = load_vocabulary()
    payload = facts_to_dict(extract_facts(fan_out_case(width=5)))
    leaves = _leaf_paths(payload)
    for typology, fields in vocab.salience.items():
        for path in fields:
            assert (
                path in leaves or path in CHECKER_REGISTRY
            ), f"salience list for {typology} names {path}, which is not a fact field"


def test_salience_covers_every_typology_in_the_vocabulary():
    vocab = load_vocabulary()
    for typology in TYPOLOGY_VOCABULARY:
        assert vocab.salient_fields(typology), typology


def test_regulatory_whitelist_is_non_empty_and_well_formed():
    vocab = load_vocabulary()
    assert vocab.regulatory
    for reference in vocab.regulatory.values():
        assert reference.threshold > 0
        assert reference.citation
        assert reference.phrase_variants


# ------------------------------------------------------------- versioning ---


def test_schema_version_is_frozen_and_consistent_everywhere():
    schema = load_case_facts_schema()
    assert CASE_FACTS_SCHEMA_VERSION == "1.0.0"
    assert schema["properties"]["schema_version"]["const"] == CASE_FACTS_SCHEMA_VERSION
    assert load_vocabulary().schema_version == CASE_FACTS_SCHEMA_VERSION


def test_package_constant_matches_the_facts_module():
    import g2t_aml

    assert g2t_aml.CASE_FACTS_SCHEMA_VERSION == CASE_FACTS_SCHEMA_VERSION


def test_hydra_config_declares_the_frozen_schema_version(repo_root):
    text = (repo_root / "configs" / "config.yaml").read_text()
    assert f'case_facts: "{CASE_FACTS_SCHEMA_VERSION}"' in text


def test_vocab_version_matches():
    assert load_vocabulary().version == VOCAB_VERSION


def test_schema_is_strict_everywhere_it_defines_an_object(repo_root):
    # additionalProperties:false is what stops a typo'd field being silently accepted.
    schema = json.loads((repo_root / "schemas" / "case_facts_v1.json").read_text())

    def walk(node, path="$"):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                assert node.get("additionalProperties") is False, path
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    walk(schema)


# --------------------------------------------------------------- taxonomy ---


def test_taxonomy_has_nine_classes_and_three_critical():
    assert len(list(HallucinationClass)) == 9
    assert {h.ident for h in CRITICAL_CLASSES} == {"H4", "H6", "H7"}


def test_severity_is_ordered():
    assert Severity.CRITICAL.rank > Severity.HIGH.rank > Severity.MEDIUM.rank > Severity.LOW.rank


def test_class_lookup_is_strict():
    assert class_by_id("h4") is HallucinationClass.H4
    with pytest.raises(KeyError, match="unknown hallucination class"):
        class_by_id("H99")


def test_every_hallucination_class_has_a_definition():
    for h in HallucinationClass:
        assert h.definition and h.title


# ------------------------------------------------------- schema strictness ---


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("extra top-level field", lambda d: d.update({"bogus": 1})),
        ("extra nested field", lambda d: d["structure"].update({"bogus": 1})),
        ("wrong schema version", lambda d: d.update({"schema_version": "2.0.0"})),
        # The entity-type exclusion, enforced by the schema and not only by the checker.
        ("role outside the vocabulary", lambda d: d["focal_entity"].update({"role": "mixer"})),
        (
            "typology outside the vocabulary",
            lambda d: d["typology"].update({"label": "smurfing"}),
        ),
        # An amount without its unit is a fact a narrative can restate wrongly for free.
        ("money without a currency", lambda d: d["flow"].update({"total_outflow": {"value": 1.0}})),
        # A sentinel whose `available` is true would defeat the whole absence design.
        (
            "sentinel claiming availability",
            lambda d: d["flow"].update({"cross_border": {"available": True, "reason": "x"}}),
        ),
        ("negative node count", lambda d: d["structure"].update({"n_nodes": -1})),
        ("density above one", lambda d: d["structure"].update({"density": 2.0})),
        (
            "phase outside the vocabulary",
            lambda d: d["temporal"].update({"event_ordering": ["laundering_phase"]}),
        ),
    ],
)
def test_schema_rejects_malformed_records(name, mutate):
    # `additionalProperties: false` and the enums are only worth having if they actually
    # fire. A schema that silently accepted a typo'd field would let a whole fact family
    # go missing without anything noticing.
    import jsonschema

    from g2t_aml.facts.schema import validate_facts

    payload = json.loads(json.dumps(facts_to_dict(extract_facts(fan_out_case(width=5)))))
    mutate(payload)
    with pytest.raises(jsonschema.ValidationError):
        validate_facts(payload)
