"""
Offline tests for the optional evaluation: vertical-owned datasets and an explicit, safe judge choice.

Nothing here imports RAGAS (the eval stack is optional) and nothing makes a provider call.
"""

import json
from pathlib import Path

import pytest

from app_core.evaluation.config import (
    JUDGE_API_KEY_ENV,
    JUDGE_BASE_URL_ENV,
    JUDGE_MODEL_ENV,
    EvalConfigError,
    JudgeConfig,
    parse_args,
    resolve_judge_config,
)
from app_core.evaluation.dataset import EvalDatasetError, load_eval_dataset
from app_core.llm.client import OPENAI_DEFAULT_BASE_URL

ROOT = Path(__file__).resolve().parents[1]
LOCAL = "http://localhost:11434/v1"

_ENV = (
    "LLM_API_KEY", "LLM_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL",
    JUDGE_BASE_URL_ENV, JUDGE_API_KEY_ENV, JUDGE_MODEL_ENV, "RAG_CHAT_MODEL",
)


@pytest.fixture
def env(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# ---- dataset: explicit and vertical-owned ----


def _write(tmp_path, payload):
    path = tmp_path / "eval.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


def test_dataset_loads_cases_and_name(tmp_path):
    path = _write(tmp_path, {"name": " Mine ", "cases": [{"question": " Q1? ", "ground_truth": " A1 ", "id": "extra ok"}]})

    dataset = load_eval_dataset(path)

    assert dataset.name == "Mine" and dataset.path == path.resolve()
    assert [(c.question, c.ground_truth) for c in dataset.cases] == [("Q1?", "A1")]


def test_dataset_name_defaults_to_the_file_stem(tmp_path):
    assert load_eval_dataset(_write(tmp_path, {"cases": [{"question": "q", "ground_truth": "a"}]})).name == "eval"


@pytest.mark.parametrize(
    "payload, message",
    [
        ("{not json", "not valid JSON"),
        ("[]", 'non-empty "cases" list'),
        ({"cases": []}, 'non-empty "cases" list'),
        ({"cases": "nope"}, 'non-empty "cases" list'),
        ({"cases": ["just a string"]}, r"cases\[0\]\): expected an object"),
        ({"cases": [{"question": "q"}]}, r'cases\[0\]\): "ground_truth" must be a non-empty string'),
        ({"cases": [{"question": "q", "ground_truth": "  "}]}, r'"ground_truth" must be a non-empty string'),
        ({"cases": [{"question": "ok", "ground_truth": "ok"}, {"ground_truth": "a"}]}, r'cases\[1\]\): "question"'),
        ({"cases": [{"question": 5, "ground_truth": "a"}]}, r'"question" must be a non-empty string'),
    ],
)
def test_malformed_datasets_fail_clearly(tmp_path, payload, message):
    with pytest.raises(EvalDatasetError, match=message):
        load_eval_dataset(_write(tmp_path, payload))


def test_missing_dataset_file_fails_clearly(tmp_path):
    with pytest.raises(EvalDatasetError, match="not found"):
        load_eval_dataset(tmp_path / "missing.json")


def test_the_cli_has_no_default_dataset(capsys):
    with pytest.raises(SystemExit) as raised:
        parse_args([])

    assert raised.value.code == 2
    assert "--dataset" in capsys.readouterr().err
    assert parse_args(["--dataset", "x.json"]).dataset == "x.json"


def test_the_evaluation_script_embeds_no_vertical_data_and_no_hosted_default(legal_vocabulary):
    source = (ROOT / "evaluate_ragas.py").read_text(encoding="utf-8")

    assert not legal_vocabulary.search(source)
    assert "EVALUATION_QUESTIONS" not in source and "EVALUATION_GROUND_TRUTHS" not in source
    assert 'getenv("OPENAI_API_KEY")' not in source  # the judge comes from resolve_judge_config only


# ---- judge: never silently hosted ----


def test_default_judge_is_the_applications_own_endpoint_even_with_a_stray_hosted_key(env):
    env.setenv("LLM_API_KEY", "local-placeholder")
    env.setenv("LLM_BASE_URL", LOCAL)
    env.setenv("OPENAI_API_KEY", "sk-stray-hosted-key")
    env.setenv("RAG_CHAT_MODEL", "gemma3:4b")

    judge = resolve_judge_config()

    assert (judge.base_url, judge.api_key, judge.model, judge.explicit) == (LOCAL, "local-placeholder", "gemma3:4b", False)
    assert "api.openai.com" not in judge.describe() and "sk-stray" not in judge.describe() + repr(judge)


def test_local_endpoint_with_only_a_legacy_key_still_judges_locally(env):
    env.setenv("LLM_BASE_URL", LOCAL)
    env.setenv("OPENAI_API_KEY", "legacy-key")

    assert resolve_judge_config().base_url == LOCAL


def test_stale_legacy_base_url_does_not_redirect_a_new_style_judge(env):
    env.setenv("LLM_API_KEY", "key")
    env.setenv("OPENAI_BASE_URL", "http://stale-gateway.invalid/v1")

    assert resolve_judge_config().base_url == OPENAI_DEFAULT_BASE_URL  # the application's own (default) endpoint


def test_nothing_configured_stops_before_any_call(env):
    with pytest.raises(EvalConfigError, match="No evaluation judge configured"):
        resolve_judge_config()


@pytest.mark.parametrize("only", [JUDGE_BASE_URL_ENV, JUDGE_API_KEY_ENV])
def test_a_separate_judge_must_be_configured_completely(env, only):
    env.setenv("LLM_API_KEY", "app-key")
    env.setenv(only, "https://judge.example/v1" if only == JUDGE_BASE_URL_ENV else "judge-key")

    with pytest.raises(EvalConfigError, match="needs both"):
        resolve_judge_config()


def test_explicit_hosted_judge_never_receives_the_applications_key(env):
    env.setenv("LLM_API_KEY", "app-key-private")
    env.setenv("LLM_BASE_URL", LOCAL)
    env.setenv(JUDGE_BASE_URL_ENV, OPENAI_DEFAULT_BASE_URL)
    env.setenv(JUDGE_API_KEY_ENV, "judge-key")

    judge = resolve_judge_config()

    assert (judge.base_url, judge.api_key, judge.explicit) == (OPENAI_DEFAULT_BASE_URL, "judge-key", True)
    assert "app-key-private" not in repr(judge) + judge.describe()


def test_judge_model_precedence(env):
    env.setenv("LLM_API_KEY", "k")
    assert resolve_judge_config().model == "gpt-4o-mini"
    env.setenv("RAG_CHAT_MODEL", "chat-m")
    assert resolve_judge_config().model == "chat-m"
    env.setenv(JUDGE_MODEL_ENV, "judge-m")
    assert resolve_judge_config().model == "judge-m"
    assert resolve_judge_config("cli-m").model == "cli-m"


def test_describe_hides_credentials_and_url_details():
    judge = JudgeConfig(model="m", base_url="https://user:pw@judge.example:443/v1?token=abc#x", api_key="SECRET", explicit=True)

    text = judge.describe() + repr(judge)

    for leaked in ("SECRET", "user", "pw@", "token", "abc"):
        assert leaked not in text
    assert "https://judge.example/v1" in text and "separate judge endpoint" in text
