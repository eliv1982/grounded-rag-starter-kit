"""
Static boundary checks for the reusable core: it is domain-neutral and never imports the layers above it.

These are deliberately simple text/AST scans of `app_core/`. The vertical-owned material (legal profile,
statute boundaries, legal labels, eval datasets) lives under `examples/`.
"""

import ast
from pathlib import Path

from app_core.config.profile import DEFAULT_PROFILE

ROOT = Path(__file__).resolve().parents[1]
CORE_FILES = sorted((ROOT / "app_core").rglob("*.py"))


def _imported_top_level_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


def test_core_files_are_found():
    assert len(CORE_FILES) >= 10  # guards the scans below against silently scanning nothing


def test_core_contains_no_legal_domain_vocabulary(legal_vocabulary):
    offenders = {
        str(path.relative_to(ROOT)): sorted({m.group(0).lower() for m in legal_vocabulary.finditer(path.read_text(encoding="utf-8"))})
        for path in CORE_FILES
    }
    assert {name: words for name, words in offenders.items() if words} == {}


def test_core_never_imports_root_wrappers_entry_points_or_verticals():
    above_core = {p.stem for p in ROOT.glob("*.py")} | {"web", "examples", "scripts", "tests"}
    offenders = {
        str(path.relative_to(ROOT)): sorted(set(_imported_top_level_modules(path)) & above_core)
        for path in CORE_FILES
    }
    assert {name: mods for name, mods in offenders.items() if mods} == {}


def test_default_profile_is_neutral(legal_vocabulary):
    assert DEFAULT_PROFILE.system_prompt_extra == ""
    assert dict(DEFAULT_PROFILE.section_boundaries) == {}
    assert dict(DEFAULT_PROFILE.kind_labels) == {}
    assert DEFAULT_PROFILE.sentence_language == "en"
    assert not legal_vocabulary.search(DEFAULT_PROFILE.chunk_header)
    # Unknown kinds are shown as written, never translated into a domain label.
    assert DEFAULT_PROFILE.kind_label("law") == "law"
    assert DEFAULT_PROFILE.section_pattern("statute") is None


def test_non_legal_verticals_contain_no_legal_wording(legal_vocabulary):
    # The data a vertical supplies (documents, manifest, profile, eval cases); READMEs may carry disclaimers.
    checked = 0
    for folder in ("sample_corpus", "examples/equipment_manual"):
        for path in sorted((ROOT / folder).iterdir()):
            if path.suffix in {".txt", ".json"}:
                assert not legal_vocabulary.search(path.read_text(encoding="utf-8")), str(path.relative_to(ROOT))
                checked += 1
    assert checked >= 5
