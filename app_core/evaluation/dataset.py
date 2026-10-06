"""
Evaluation dataset loading.

An evaluation dataset belongs to a vertical, never to the core: there is no built-in default and
no registry, the caller always names a file. Format (JSON):

    {
      "name": "optional label",
      "cases": [
        {"question": "...", "ground_truth": "reference answer used by the reference-based metrics"}
      ]
    }

Loading is pure and offline: nothing here needs RAGAS, a provider or an API key.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, Union


class EvalDatasetError(ValueError):
    """The evaluation dataset is missing or malformed."""


@dataclass(frozen=True)
class EvalCase:
    question: str
    ground_truth: str


@dataclass(frozen=True)
class EvalDataset:
    name: str
    path: Path
    cases: Tuple[EvalCase, ...]


def load_eval_dataset(path: Union[str, Path]) -> EvalDataset:
    """Read and validate a dataset file; raises EvalDatasetError describing exactly what is wrong."""
    file = Path(path).expanduser()
    if not file.is_file():
        raise EvalDatasetError(f"Evaluation dataset not found: {file.resolve()}")
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise EvalDatasetError(f"Cannot read evaluation dataset {file}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EvalDatasetError(f"Evaluation dataset {file} is not valid JSON: {exc}") from exc

    raw_cases = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(raw_cases, list) or not raw_cases:
        raise EvalDatasetError(f'Evaluation dataset {file} must be an object with a non-empty "cases" list')

    cases = []
    for i, raw in enumerate(raw_cases):
        where = f"{file} (cases[{i}])"
        if not isinstance(raw, dict):
            raise EvalDatasetError(f"{where}: expected an object")
        for key in ("question", "ground_truth"):
            value: Any = raw.get(key)
            if not isinstance(value, str) or not value.strip():
                raise EvalDatasetError(f'{where}: "{key}" must be a non-empty string')
        cases.append(EvalCase(question=raw["question"].strip(), ground_truth=raw["ground_truth"].strip()))

    name = data.get("name")
    return EvalDataset(
        name=name.strip() if isinstance(name, str) and name.strip() else file.stem,
        path=file.resolve(),
        cases=tuple(cases),
    )
