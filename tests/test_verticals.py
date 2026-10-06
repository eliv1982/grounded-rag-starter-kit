"""
Reuse proof: two different verticals run through the same unchanged `app_core/`.

* equipment_manual: a tiny synthetic non-legal vertical, configured purely by data (examples/equipment_manual).
  It is loaded the way the real entry points load a vertical: only RAG_CORPUS_CONFIG is set, and the pipeline
  resolves the corpus and the profile from that manifest.
* independent_guarantees: the legal demo. Its source texts are private (not in the repo), so its profile is
  exercised on synthetic text; the profile itself is the real one from examples/independent_guarantees.

Fake embeddings and a fake chat model: no provider, no network.
"""

import re

import pytest

from app_core.config.knowledge import CORPUS_CONFIG_ENV, load_corpus_config, load_profile_config
from app_core.evaluation.dataset import load_eval_dataset
from app_core.generation.prompts import DEFAULT_RAG_SYSTEM_PROMPT, build_system_prompt
from app_core.retrieval.vector_store import VectorStore

EQUIPMENT_MANIFEST = "examples/equipment_manual/corpus.json"
LEGAL_MANIFEST = "examples/independent_guarantees/corpus.json"
QUESTION = "How often must the boiler be descaled?"


# ---- a non-legal vertical, loaded from its manifest ----


@pytest.fixture
def equipment(index_env, pipeline_factory):
    index_env.setenv(CORPUS_CONFIG_ENV, EQUIPMENT_MANIFEST)
    for name, value in {"RAG_CHUNK_SIZE": "800", "RAG_CHUNK_OVERLAP": "200", "RAG_MIN_CHUNK_LEN": "80"}.items():
        index_env.setenv(name, value)
    return pipeline_factory()  # no corpus_entries, no profile: everything comes from the manifest


def test_equipment_vertical_corpus_loads_and_indexes_with_its_own_profile(equipment):
    pipeline, _ = equipment

    stored = pipeline.vector_store.collection.get(include=["documents", "metadatas"])
    procedures = [(d, m) for d, m in zip(stored["documents"], stored["metadatas"]) if m["section_heading"].startswith("Procedure")]

    assert pipeline.profile.name == "equipment_manual"
    assert sorted(m["section_heading"][:12] for _, m in procedures) == [f"Procedure {n}." for n in (1, 2, 3, 4)]
    for document, meta in procedures:
        assert document.startswith("[service manual: Aurora X2 Service Manual (sample)]\n[Procedure: Procedure ")
        assert meta["source_kind"] == "service_manual" and meta["doc_type"] == "manual"
        assert len(document) <= 800 and document[int(meta["header_len"]):].startswith("Procedure ")


def test_equipment_vertical_retrieval_works(equipment):
    pipeline, _ = equipment

    hits = pipeline.vector_store.search(QUESTION, top_k=10)

    assert 4 <= len(hits) <= 10
    assert all(hit["text"] and hit["metadata"]["source"] == "aurora_x2_manual" for hit in hits)
    assert all(0 <= hit["distance"] <= 2 for hit in hits)


def test_equipment_vertical_answers_through_the_unchanged_pipeline(equipment):
    pipeline, llm = equipment

    result = pipeline.query(QUESTION)

    assert result["answer"] == "fake llm answer" and result["from_cache"] is False
    assert 1 <= len(result["context_docs"]) <= 3 and len(llm.calls) == 1
    system, user = (m["content"] for m in llm.calls[0]["messages"])
    assert system == build_system_prompt(pipeline.profile.system_prompt_extra)
    assert "Aurora X2" in system and DEFAULT_RAG_SYSTEM_PROMPT in system  # vertical added, core rules intact
    assert re.search(r'<retrieved_fragment number="1" source="Aurora X2 Service Manual \(sample\)" type="service_manual"', user)
    assert pipeline.get_stats()["profile"] == "equipment_manual"


def test_nothing_legal_leaks_into_the_non_legal_vertical(equipment, legal_vocabulary):
    pipeline, llm = equipment
    pipeline.query(QUESTION)

    stored = pipeline.vector_store.collection.get(include=["documents", "metadatas"])
    everything = [*stored["documents"], *(str(m) for m in stored["metadatas"])]
    everything += [m["content"] for m in llm.calls[0]["messages"]]

    assert everything
    assert [text for text in everything if legal_vocabulary.search(text)] == []


def test_equipment_vertical_eval_dataset_loads():
    dataset = load_eval_dataset("examples/equipment_manual/eval.json")

    assert dataset.name == "equipment_manual" and len(dataset.cases) == 3
    assert all("Procedure" in case.ground_truth for case in dataset.cases)


# ---- the legal vertical, through its own profile ----

STATUTE_TEXT = (
    "Статья 368. Понятие независимой гарантии\n"
    "По независимой гарантии гарант принимает на себя обязательство уплатить бенефициару денежную сумму.\n\n"
    "Статья 369. Вознаграждение за выдачу гарантии\n"
    "Принципал уплачивает гаранту вознаграждение за выдачу независимой гарантии, если иное не предусмотрено.\n\n"
    "§ 3. Независимая гарантия\n"
    "Отдельный параграф о независимых гарантиях в составе главы, достаточно длинный для отдельного фрагмента."
)


def _legal_store():
    vs = VectorStore.__new__(VectorStore)
    vs.profile = load_profile_config(LEGAL_MANIFEST)
    vs.chunk_size, vs.chunk_overlap, vs.min_chunk_len = 800, 200, 80
    return vs


def _chunks(text, *, kind, doc_type, display="ГК РФ"):
    return _legal_store()._build_chunks_for_file(text, source="gk", source_display=display, source_kind=kind, doc_type=doc_type)


def test_legal_profile_splits_statutes_at_articles_and_paragraph_signs():
    rows = _chunks(STATUTE_TEXT, kind="law", doc_type="statute")

    assert [m["section_heading"] for _, m in rows] == [
        "Статья 368. Понятие независимой гарантии",
        "Статья 369. Вознаграждение за выдачу гарантии",
        "§ 3. Независимая гарантия",
    ]
    assert rows[0][0].startswith(
        "[Источник: ГК РФ | закон РФ]\n[Фрагмент: Статья 368. Понятие независимой гарантии]\n\nСтатья 368."
    )


def test_legal_profile_splits_only_statutes_and_labels_every_known_kind():
    assert len(_chunks(STATUTE_TEXT, kind="rules", doc_type="rules")) == 1
    assert _chunks(STATUTE_TEXT, kind="rules", doc_type="rules")[0][0].startswith("[Источник: ГК РФ | правила (URDG)]")
    assert _chunks(STATUTE_TEXT, kind="case_law_summary", doc_type="overview")[0][0].startswith(
        "[Источник: ГК РФ | обзор судебной практики]"
    )
    assert _chunks(STATUTE_TEXT, kind="unlisted", doc_type="overview")[0][0].startswith("[Источник: ГК РФ | unlisted]")


def test_legal_profile_carries_the_legal_prompt_addition_and_russian_sentence_rules():
    profile = load_profile_config(LEGAL_MANIFEST)

    assert profile.sentence_language == "ru"
    assert "not legal advice" in profile.system_prompt_extra
    assert build_system_prompt(profile.system_prompt_extra).startswith(DEFAULT_RAG_SYSTEM_PROMPT)


def test_legal_vertical_runs_end_to_end_through_the_same_core(pipeline_factory, tmp_path):
    (tmp_path / "gk.txt").write_text(STATUTE_TEXT, encoding="utf-8")
    entries = [
        {"path": tmp_path / "gk.txt", "source": "gk", "source_display": "ГК РФ", "source_kind": "law", "doc_type": "statute"}
    ]
    pipeline, llm = pipeline_factory(corpus_entries=entries, profile=load_profile_config(LEGAL_MANIFEST))

    stored = pipeline.vector_store.collection.get(include=["documents"])
    result = pipeline.query("Что такое независимая гарантия?")

    assert len(stored["documents"]) == 3 and all(d.startswith("[Источник: ГК РФ | закон РФ]") for d in stored["documents"])
    assert result["answer"] == "fake llm answer"
    system = llm.calls[0]["messages"][0]["content"]
    assert "independent guarantees" in system and DEFAULT_RAG_SYSTEM_PROMPT in system
    assert pipeline.get_stats()["profile"] == "independent_guarantees"


def test_legal_vertical_eval_dataset_is_vertical_owned_data():
    dataset = load_eval_dataset("examples/independent_guarantees/eval.json")

    assert dataset.name == "independent_guarantees" and len(dataset.cases) == 5
    assert all(case.ground_truth for case in dataset.cases)


def test_shipped_manifests_still_describe_their_documents():
    assert [e["source"] for e in load_corpus_config(EQUIPMENT_MANIFEST)] == ["aurora_x2_manual"]
