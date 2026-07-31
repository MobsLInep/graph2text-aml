from __future__ import annotations

import logging

import pytest

from g2t_aml.utils import logging as g2t_logging
from g2t_aml.utils.logging import configure_logging, get_logger, log_mapping, stage


@pytest.fixture(autouse=True)
def _reset_logging_state():
    yield
    g2t_logging._CONFIGURED = False
    for handler in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(handler)


def test_configure_logging_returns_package_logger():
    assert configure_logging(force=True).name == "g2t_aml"


def test_configure_logging_tees_to_file(tmp_path):
    log_file = tmp_path / "nested" / "run.log"
    log = configure_logging(log_file=log_file, force=True)
    log.info("hello from the pipeline")
    logging.shutdown()
    assert "hello from the pipeline" in log_file.read_text()


def test_configure_logging_is_idempotent_without_force():
    configure_logging(force=True)
    before = len(logging.getLogger().handlers)
    configure_logging()
    assert len(logging.getLogger().handlers) == before


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("__main__", "g2t_aml.__main__"),
        ("scripts.smoke", "g2t_aml.smoke"),
        ("g2t_aml.facts.extract", "g2t_aml.facts.extract"),
    ],
)
def test_get_logger_namespaces_under_the_package(given, expected):
    assert get_logger(given).name == expected


def test_log_mapping_emits_one_record_per_key(caplog):
    log = logging.getLogger("g2t_aml.test")
    with caplog.at_level(logging.INFO, logger="g2t_aml.test"):
        log_mapping(log, "config:", {"seed": 42, "data": "amlworld"})
    assert "config:" in caplog.text
    assert "seed" in caplog.text
    assert "'amlworld'" in caplog.text


def test_log_mapping_handles_empty_mapping(caplog):
    log = logging.getLogger("g2t_aml.test")
    with caplog.at_level(logging.INFO, logger="g2t_aml.test"):
        log_mapping(log, "config:", {})
    assert "config:" in caplog.text


def test_stage_logs_start_end_and_elapsed(caplog):
    log = logging.getLogger("g2t_aml.test")
    with caplog.at_level(logging.INFO, logger="g2t_aml.test"), stage("facts", log, seed=42) as s:
        s["n_cases"] = 7
    assert "=== START facts ===" in caplog.text
    assert "=== END facts" in caplog.text
    assert "n_cases" in caplog.text
    assert "elapsed_seconds" in caplog.text


def test_stage_without_context_still_brackets(caplog):
    log = logging.getLogger("g2t_aml.test")
    with caplog.at_level(logging.INFO, logger="g2t_aml.test"), stage("bare", log):
        pass
    assert "=== END bare" in caplog.text


def test_stage_logs_and_reraises_on_failure(caplog):
    log = logging.getLogger("g2t_aml.test")
    with (
        caplog.at_level(logging.INFO, logger="g2t_aml.test"),
        pytest.raises(ValueError, match="boom"),
        stage("facts", log),
    ):
        raise ValueError("boom")
    assert "=== FAILED facts" in caplog.text
    assert "=== END" not in caplog.text


def test_stage_defaults_to_the_package_logger(caplog):
    with caplog.at_level(logging.INFO, logger="g2t_aml"), stage("default-logger"):
        pass
    assert "=== START default-logger ===" in caplog.text
