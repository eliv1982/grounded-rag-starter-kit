"""
Domain profile: the small, vertical-owned input the reusable core accepts.

The core is domain-neutral. Everything that makes a vertical a vertical (a policy
handbook, an equipment manual, a compliance assistant) is supplied as plain data in one
`DomainProfile`, normally from the optional `"profile"` object of the corpus
manifest (see `app_core/config/knowledge.py`):

    name                 vertical identifier (shown in stats; not part of any fingerprint)
    system_prompt_extra  domain instructions appended AFTER the core grounding rules; they supplement
                         those rules and can never replace them (`build_system_prompt`)
    section_boundaries   doc_type -> regex that marks where a new section starts (zero-width, e.g.
                         "(?m)^(?=Procedure\\s+\\d+\\.)"); doc types without an entry are not split
    kind_labels          source_kind -> label used in the chunk header (unknown kinds show as-is)
    chunk_header         template for the header put in front of every chunk; fields:
                         {source_display} {kind_label} {heading}
    sentence_language    pysbd language code for splitting over-long paragraphs

The default profile is neutral: no section boundaries, no labels, an English
header, English sentence rules and no extra prompt. A profile is data, not code,
so everything that changes how chunks are built (boundaries, labels, header,
language) is fingerprinted into the index manifest automatically and the
system_prompt_extra into the answer fingerprint (app_core/lifecycle.py): editing
a profile rebuilds the index or misses the cache without any manual version bump.
The profile comes from the repository's own configuration and is trusted like
the corpus manifest; it is never user input.
"""

import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Pattern

DEFAULT_PROFILE_NAME = "generic"
DEFAULT_CHUNK_HEADER = "[Source: {source_display} | {kind_label}]\n[Section: {heading}]\n\n"
DEFAULT_SENTENCE_LANGUAGE = "en"
MAX_SYSTEM_PROMPT_EXTRA_CHARS = 4000

HEADER_FIELDS = ("source_display", "kind_label", "heading")
PROFILE_KEYS = (
    "name",
    "system_prompt_extra",
    "section_boundaries",
    "kind_labels",
    "chunk_header",
    "sentence_language",
)


class ProfileError(ValueError):
    """A domain profile is malformed."""


def _text(name: str, value: Any, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ProfileError(f'profile field "{name}" must be {"a string" if allow_empty else "a non-empty string"}')
    return value


def _str_map(name: str, value: Any) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        raise ProfileError(f'profile field "{name}" must be an object of strings')
    out: Dict[str, str] = {}
    for key, item in value.items():
        _text(f"{name} key", key)
        out[key] = _text(f'{name}["{key}"]', item)
    return out


@dataclass(frozen=True)
class DomainProfile:
    name: str = DEFAULT_PROFILE_NAME
    system_prompt_extra: str = ""
    section_boundaries: Mapping[str, str] = field(default_factory=dict)
    kind_labels: Mapping[str, str] = field(default_factory=dict)
    chunk_header: str = DEFAULT_CHUNK_HEADER
    sentence_language: str = DEFAULT_SENTENCE_LANGUAGE

    def __post_init__(self) -> None:
        _text("name", self.name)
        extra = _text("system_prompt_extra", self.system_prompt_extra, allow_empty=True)
        if len(extra) > MAX_SYSTEM_PROMPT_EXTRA_CHARS:
            raise ProfileError(f'profile field "system_prompt_extra" is longer than {MAX_SYSTEM_PROMPT_EXTRA_CHARS} characters')
        object.__setattr__(self, "section_boundaries", _str_map("section_boundaries", self.section_boundaries))
        object.__setattr__(self, "kind_labels", _str_map("kind_labels", self.kind_labels))
        _text("sentence_language", self.sentence_language)

        for doc_type, pattern in self.section_boundaries.items():
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ProfileError(f'profile field section_boundaries["{doc_type}"] is not a valid regex: {exc}') from exc

        template = _text("chunk_header", self.chunk_header)
        try:
            fields = {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}
            unknown = sorted(fields - set(HEADER_FIELDS))
            if not unknown:
                template.format(**{name: "" for name in HEADER_FIELDS})
        except (ValueError, IndexError, KeyError) as exc:
            raise ProfileError(f'profile field "chunk_header" is not a valid template: {exc!r}') from exc
        if unknown:
            raise ProfileError(
                f'profile field "chunk_header" uses unknown fields {unknown}; allowed: {", ".join(HEADER_FIELDS)}'
            )

    # -- what the chunker asks the profile --

    def section_pattern(self, doc_type: str) -> Optional[Pattern[str]]:
        """Compiled section-boundary regex for this doc type, or None if the doc type is not split."""
        pattern = self.section_boundaries.get(doc_type)
        return re.compile(pattern) if pattern else None

    def kind_label(self, source_kind: str) -> str:
        return self.kind_labels.get(source_kind, source_kind)

    def format_header(self, source_display: str, source_kind: str, heading: str) -> str:
        return self.chunk_header.format(
            source_display=source_display, kind_label=self.kind_label(source_kind), heading=heading
        )

    # -- what the lifecycle fingerprints --

    def chunking_identity(self) -> Dict[str, Any]:
        """Everything in the profile that changes the stored chunks (the name and the prompt extra do not)."""
        return {
            "section_boundaries": dict(self.section_boundaries),
            "kind_labels": dict(self.kind_labels),
            "chunk_header": self.chunk_header,
            "sentence_language": self.sentence_language,
        }


DEFAULT_PROFILE = DomainProfile()


def profile_from_dict(data: Any) -> DomainProfile:
    """Build a profile from a JSON-style object; unknown keys are rejected so typos cannot go unnoticed."""
    if not isinstance(data, Mapping):
        raise ProfileError("a profile must be an object")
    unknown = sorted(set(data) - set(PROFILE_KEYS))
    if unknown:
        raise ProfileError(f"unknown profile fields {unknown}; allowed: {', '.join(PROFILE_KEYS)}")
    return DomainProfile(**{key: data[key] for key in PROFILE_KEYS if key in data})
