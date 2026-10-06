"""
Evaluation command-line and judge configuration (no RAGAS import, no provider access).

Evaluation is optional and manual. It sends the questions, the retrieved context, the model's
answers and the reference answers to a "judge" model, so where that judge runs must never be a
surprise:

* By default the judge is the application's own configured endpoint (LLM_API_KEY, LLM_BASE_URL),
  the same one that already receives the queries and the retrieved context. A local/private
  configuration therefore gets a local judge; nothing is ever sent to a hosted provider because
  a stray OPENAI_API_KEY happens to be set.
* A different judge (typically a stronger hosted model) is used only when it is configured
  explicitly and completely: RAG_EVAL_JUDGE_BASE_URL and RAG_EVAL_JUDGE_API_KEY together. The
  application's key is never sent to a judge endpoint it was not issued for.
* With no judge configured at all, evaluation stops with an explanation before any call is made.
* RAGAS' own anonymous usage analytics are switched off by default (see `apply_privacy_defaults`):
  the only connection evaluation makes is to the judge endpoint named above.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional

from app_core.lifecycle import normalize_endpoint
from app_core.llm.client import OPENAI_DEFAULT_BASE_URL, resolve_api_key, resolve_base_url

JUDGE_BASE_URL_ENV = "RAG_EVAL_JUDGE_BASE_URL"
JUDGE_API_KEY_ENV = "RAG_EVAL_JUDGE_API_KEY"
JUDGE_MODEL_ENV = "RAG_EVAL_JUDGE_MODEL"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"  # same default as the pipeline's RAG_CHAT_MODEL
RAGAS_TELEMETRY_ENV = "RAGAS_DO_NOT_TRACK"  # RAGAS reads it when it first records an analytics event


class EvalConfigError(ValueError):
    """The evaluation judge is not (or not consistently) configured."""


@dataclass(frozen=True)
class JudgeConfig:
    model: str
    base_url: str = field(repr=False)  # always explicit (the SDKs' own OPENAI_BASE_URL fallbacks never apply); may carry user info
    api_key: str = field(repr=False)
    explicit: bool  # True: a separate judge endpoint from RAG_EVAL_JUDGE_*; False: the application's endpoint

    def describe(self) -> str:
        """One line for the console, safe to print: no key, no user info/query in the URL."""
        where = "separate judge endpoint (RAG_EVAL_JUDGE_*)" if self.explicit else "the application's own endpoint"
        return f"model={self.model} endpoint={normalize_endpoint(self.base_url)} ({where})"


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def apply_privacy_defaults() -> None:
    """
    Opt out of RAGAS' anonymous usage analytics unless the user has made an explicit choice.

    Call it from the evaluation entry point after the repository `.env` is loaded (so a value set there
    counts as the user's choice) and before RAGAS runs. An explicit value, even `false`, is never
    overwritten; a blank value counts as unset. The configured judge itself is not affected.
    """
    if not _env(RAGAS_TELEMETRY_ENV):
        os.environ[RAGAS_TELEMETRY_ENV] = "true"


def resolve_judge_config(model_override: Optional[str] = None) -> JudgeConfig:
    model = (model_override or "").strip() or _env(JUDGE_MODEL_ENV) or _env("RAG_CHAT_MODEL") or DEFAULT_CHAT_MODEL

    explicit_url, explicit_key = _env(JUDGE_BASE_URL_ENV), _env(JUDGE_API_KEY_ENV)
    if explicit_url or explicit_key:
        if not (explicit_url and explicit_key):
            raise EvalConfigError(
                f"A separate evaluation judge needs both {JUDGE_BASE_URL_ENV} and {JUDGE_API_KEY_ENV} "
                f"(for hosted OpenAI use {JUDGE_BASE_URL_ENV}={OPENAI_DEFAULT_BASE_URL}). "
                "Unset both to judge with the application's own endpoint."
            )
        return JudgeConfig(model=model, base_url=explicit_url, api_key=explicit_key, explicit=True)

    api_key = resolve_api_key()
    if not api_key:
        raise EvalConfigError(
            "No evaluation judge configured: set LLM_API_KEY (and LLM_BASE_URL for a local endpoint) to judge "
            f"with the application's own endpoint, or {JUDGE_BASE_URL_ENV} + {JUDGE_API_KEY_ENV} for a separate judge."
        )
    return JudgeConfig(
        model=model, base_url=resolve_base_url() or OPENAI_DEFAULT_BASE_URL, api_key=api_key, explicit=False
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_ragas.py",
        description=(
            "Optional, manual RAGAS evaluation of the configured RAG application. The dataset is vertical-owned "
            "and must be named explicitly; the judge is the application's own endpoint unless RAG_EVAL_JUDGE_* "
            "configures another one. Needs requirements-eval.txt; every metric makes judge-model calls."
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        metavar="PATH",
        help='JSON file {"cases": [{"question": ..., "ground_truth": ...}]}, e.g. examples/equipment_manual/eval.json',
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        metavar="NAME",
        help=f"judge model (default: {JUDGE_MODEL_ENV}, else RAG_CHAT_MODEL)",
    )
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)
