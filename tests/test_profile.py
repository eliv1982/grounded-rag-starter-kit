"""
Offline tests for the domain profile: validation, manifest loading, profile-driven chunking, the
header-size cap, and how the profile takes part in the index/cache lifecycle.
"""

import json

import pytest

from app_core.config.knowledge import CORPUS_CONFIG_ENV, CorpusConfigError, default_profile, load_corpus_config, load_profile_config
from app_core.config.profile import DEFAULT_PROFILE, DomainProfile, ProfileError, profile_from_dict
from app_core.retrieval.vector_store import VectorStore

LEGAL_MANIFEST = "examples/independent_guarantees/corpus.json"
EQUIPMENT_MANIFEST = "examples/equipment_manual/corpus.json"

SECTIONED = (
    "Section 1. Alpha\nThe alpha section says that alpha things happen on alpha days every week.\n\n"
    "Section 2. Beta\nThe beta section says that beta things happen on beta days every month."
)
SECTIONED_PROFILE = DomainProfile(
    name="sectioned",
    section_boundaries={"manual": r"(?m)^(?=Section\s+\d+\.)"},
    kind_labels={"guide": "user guide"},
    chunk_header="<{kind_label} / {source_display}> {heading}\n",
)


def _store(profile=DEFAULT_PROFILE, chunk_size=800, overlap=200, min_len=20):
    vs = VectorStore.__new__(VectorStore)
    vs.profile = profile
    vs.chunk_size, vs.chunk_overlap, vs.min_chunk_len = chunk_size, overlap, min_len
    return vs


def _chunks(vs, text, *, kind="guide", doc_type="manual", display="Doc"):
    return vs._build_chunks_for_file(text, source="s", source_display=display, source_kind=kind, doc_type=doc_type)


# ---- validation ----


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"name": ""}, '"name"'),
        ({"system_prompt_extra": 5}, "system_prompt_extra"),
        ({"system_prompt_extra": "x" * 4001}, "longer than"),
        ({"section_boundaries": {"manual": "("}}, "valid regex"),
        ({"section_boundaries": {"manual": 3}}, "section_boundaries"),
        ({"section_boundaries": ["a"]}, "object of strings"),
        ({"kind_labels": {"guide": ""}}, "kind_labels"),
        ({"chunk_header": "[{unknown}]"}, "unknown fields"),
        ({"chunk_header": "[{0}]"}, "unknown fields"),
        ({"chunk_header": "{heading.__class__}"}, "unknown fields"),
        ({"chunk_header": "[{heading"}, "not a valid template"),
        ({"chunk_header": "{heading!x}"}, "not a valid template"),
        ({"sentence_language": " "}, "sentence_language"),
    ],
)
def test_invalid_profiles_are_rejected_with_a_precise_message(kwargs, message):
    with pytest.raises(ProfileError, match=message):
        DomainProfile(**kwargs)


def test_profile_from_dict_rejects_typos_and_non_objects():
    with pytest.raises(ProfileError, match="unknown profile fields"):
        profile_from_dict({"sistem_prompt_extra": "typo"})
    with pytest.raises(ProfileError, match="must be an object"):
        profile_from_dict(["not", "an", "object"])


def test_profile_from_dict_round_trips_a_full_profile():
    profile = profile_from_dict(
        {
            "name": "x",
            "system_prompt_extra": "Be brief.",
            "section_boundaries": {"manual": "(?m)^(?=A)"},
            "kind_labels": {"guide": "user guide"},
            "chunk_header": "{heading}|{source_display}|{kind_label}\n",
            "sentence_language": "de",
        }
    )
    assert profile.name == "x" and profile.sentence_language == "de"
    assert profile.format_header("Doc", "guide", "Head") == "Head|Doc|user guide\n"
    assert profile.format_header("Doc", "other", "Head") == "Head|Doc|other\n"  # unknown kind shown as written


# ---- manifest loading ----


def _manifest(tmp_path, payload):
    (tmp_path / "doc.txt").write_text("Some document text that is long enough to be a chunk of its own.", encoding="utf-8")
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


ENTRY = {"path": "doc.txt", "source": "d", "source_display": "Doc"}


def test_manifest_without_a_profile_gets_the_neutral_default(tmp_path):
    assert load_profile_config(_manifest(tmp_path, {"entries": [ENTRY]})) is DEFAULT_PROFILE


def test_manifest_profile_is_loaded(tmp_path):
    profile = load_profile_config(
        _manifest(tmp_path, {"profile": {"name": "mine", "system_prompt_extra": "Hi."}, "entries": [ENTRY]})
    )
    assert (profile.name, profile.system_prompt_extra) == ("mine", "Hi.")


@pytest.mark.parametrize("bad", ["a string", {"sistem_prompt_extra": "typo"}, {"chunk_header": "{nope}"}])
def test_malformed_manifest_profile_is_a_clear_corpus_config_error(tmp_path, bad):
    with pytest.raises(CorpusConfigError, match="profile"):
        load_profile_config(_manifest(tmp_path, {"profile": bad, "entries": [ENTRY]}))


def test_loading_the_profile_does_not_require_the_corpus_files(tmp_path):
    payload = {"profile": {"name": "p"}, "entries": [{**ENTRY, "path": "missing.txt"}]}
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_profile_config(path).name == "p"
    with pytest.raises(CorpusConfigError, match="не найден"):
        load_corpus_config(path)


def test_default_profile_follows_the_corpus_config_variable(monkeypatch):
    monkeypatch.delenv(CORPUS_CONFIG_ENV, raising=False)
    assert default_profile() is DEFAULT_PROFILE
    monkeypatch.setenv(CORPUS_CONFIG_ENV, EQUIPMENT_MANIFEST)
    assert default_profile().name == "equipment_manual"


def test_shipped_verticals_have_valid_profiles():
    assert load_profile_config(LEGAL_MANIFEST).name == "independent_guarantees"
    assert load_profile_config(EQUIPMENT_MANIFEST).name == "equipment_manual"
    assert load_profile_config("sample_corpus/corpus.json") is DEFAULT_PROFILE
    assert len(load_corpus_config(EQUIPMENT_MANIFEST)) == 1  # a profile does not disturb entry loading


# ---- profile-driven chunking ----


def test_neutral_default_never_splits_by_doc_type_or_labels_kinds():
    rows = _chunks(_store(), SECTIONED, kind="law", doc_type="statute")

    assert len(rows) == 1  # one section: nothing in the core knows what a statute is
    text, meta = rows[0]
    assert text.startswith("[Source: Doc | law]\n[Section: Section 1. Alpha]\n\n")
    assert meta["section_heading"] == "Section 1. Alpha"


def test_profile_boundaries_labels_and_header_shape_the_chunks():
    rows = _chunks(_store(SECTIONED_PROFILE), SECTIONED)

    assert [meta["section_heading"] for _, meta in rows] == ["Section 1. Alpha", "Section 2. Beta"]
    assert rows[0][0].startswith("<user guide / Doc> Section 1. Alpha\nSection 1. Alpha\n")
    assert all(text.startswith("<user guide / Doc>") for text, _ in rows)


def test_boundaries_apply_only_to_their_doc_type():
    rows = _chunks(_store(SECTIONED_PROFILE), SECTIONED, doc_type="overview")

    assert len(rows) == 1


def test_header_len_metadata_marks_exactly_the_header():
    for profile, header in (
        (DEFAULT_PROFILE, "[Source: Doc | guide]\n[Section: Section 1. Alpha]\n\n"),
        (SECTIONED_PROFILE, "<user guide / Doc> Section 1. Alpha\n"),
    ):
        text, meta = _chunks(_store(profile), SECTIONED)[0]
        assert text[: int(meta["header_len"])] == header
        assert text[int(meta["header_len"]):].startswith("Section 1. Alpha")


def test_chunking_is_deterministic():
    vs = _store(SECTIONED_PROFILE)
    assert _chunks(vs, SECTIONED) == _chunks(vs, SECTIONED)


def test_sentence_language_comes_from_the_profile(monkeypatch):
    import pysbd

    seen = []

    class Recorder:
        def __init__(self, language, clean):
            seen.append(language)

        def segment(self, text):
            return [text]

    monkeypatch.setattr(pysbd, "Segmenter", Recorder)
    _store(DomainProfile(sentence_language="ru"))._split_sentences("One. Two.")
    _store()._split_sentences("One. Two.")

    assert seen == ["ru", "en"]


# ---- an oversized header can no longer starve the body (audit finding) ----


def test_oversized_header_cannot_explode_the_chunks():
    vs = _store(chunk_size=200, overlap=40, min_len=20)
    text = "Heading line\n\n" + "Plain body sentence number one. " * 12
    display = "A very long source display name " * 4

    rows = vs._enforce_hard_chunk_limit(_chunks(vs, text, display=display, kind="policy", doc_type="overview"), 200, 40)

    texts = [t for t, _ in rows]
    assert 1 <= len(rows) <= 6  # this input used to become 48 rows, most of them cut-up header
    assert len(set(texts)) == len(texts)
    for text_, meta in rows:
        assert len(text_) <= 200  # the hard limit still holds
        assert len(text_) - int(meta["header_len"]) >= 40  # a real body, never a one-character sliver
    assert rows == vs._enforce_hard_chunk_limit(_chunks(vs, text, display=display, kind="policy", doc_type="overview"), 200, 40)


def test_header_is_capped_at_half_the_chunk_size_by_shortening_the_heading():
    vs = _store(chunk_size=400, overlap=80, min_len=20)
    text = "H" * 300 + "\n\n" + "Body sentence here. " * 20

    rows = _chunks(vs, text, display="D" * 80, kind="policy", doc_type="overview")

    assert rows
    for chunk, meta in rows:
        assert meta["header_len"] == "200"  # exactly the cap
        assert chunk[:200].endswith("…]\n\n")  # the heading was cut inside the header...
        assert meta["section_heading"] == "H" * 160 + "…"  # ...but not in the metadata
        assert len(chunk) <= 400


def test_header_that_cannot_fit_is_dropped_and_the_source_stays_in_metadata():
    vs = _store(chunk_size=100, overlap=20, min_len=20)

    rows = _chunks(vs, "Body sentence here. " * 10, display="D" * 60, kind="policy", doc_type="overview")

    assert rows
    for chunk, meta in rows:
        assert meta["header_len"] == "0" and not chunk.startswith("[")
        assert meta["source_display"] == "D" * 60 and len(chunk) <= 100


def test_hard_split_parts_after_the_first_have_no_header_offset():
    vs = _store(chunk_size=100, overlap=20)
    long_row = ("[Source: Doc | guide]\n[Section: x]\n\n" + "w" * 300, {"header_len": "36"})

    parts = vs._enforce_hard_chunk_limit([long_row], 100, 20)

    assert [m["header_len"] for _, m in parts] == ["36"] + ["0"] * (len(parts) - 1)


# ---- the profile in the index / cache lifecycle ----


def _real_store(chroma_dir, profile=None):
    return VectorStore(persist_directory=chroma_dir, profile=profile)


def test_changing_the_chunking_rules_of_the_profile_rebuilds_the_index(index_env, corpus, chroma_dir):
    _real_store(chroma_dir).ensure_index(corpus.entries)

    status = _real_store(chroma_dir, DomainProfile(chunk_header="[{heading}]\n\n")).ensure_index(corpus.entries)

    assert status.action == "rebuilt"
    assert status.reasons == ("domain profile (chunking rules) changed",)


def test_profile_name_and_prompt_extra_do_not_touch_the_index(index_env, corpus, chroma_dir):
    _real_store(chroma_dir, DomainProfile(name="a", system_prompt_extra="One.")).ensure_index(corpus.entries)

    status = _real_store(chroma_dir, DomainProfile(name="b", system_prompt_extra="Two.")).ensure_index(corpus.entries)

    assert status.action == "reused"


def test_stored_chunks_carry_the_profile_header_and_header_length(index_env, corpus, chroma_dir):
    store = _real_store(chroma_dir, DomainProfile(chunk_header="<<{source_display}>>\n"))
    store.ensure_index(corpus.entries)

    stored = store.collection.get(include=["documents", "metadatas"])

    assert stored["documents"]
    for document, meta in zip(stored["documents"], stored["metadatas"]):
        assert document.startswith("<<") and document[: int(meta["header_len"])].endswith(">>\n")


def test_system_prompt_extra_scopes_the_answer_cache(pipeline_factory, corpus):
    question = "What does the alpha document say?"
    rules_a, rules_b = DomainProfile(system_prompt_extra="Rules A."), DomainProfile(system_prompt_extra="Rules B.")

    first, first_llm = pipeline_factory(corpus_entries=corpus.entries, profile=rules_a)
    assert first.query(question)["from_cache"] is False
    assert first.query(question)["from_cache"] is True and len(first_llm.calls) == 1

    other, other_llm = pipeline_factory(corpus_entries=corpus.entries, profile=rules_b)
    assert other.answer_fingerprint != first.answer_fingerprint
    assert other.query(question)["from_cache"] is False and len(other_llm.calls) == 1

    again, again_llm = pipeline_factory(corpus_entries=corpus.entries, profile=rules_a)
    assert again.answer_fingerprint == first.answer_fingerprint
    assert again.query(question)["from_cache"] is True and again_llm.calls == []
