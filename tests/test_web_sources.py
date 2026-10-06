"""
Offline tests for how the web page presents retrieved context: honest wording, numbering that matches the
prompt's fragment numbers, header-free previews and escaped output.
"""

import re

import pytest
from fastapi.testclient import TestClient

import web.app as web_app
from app_core.generation.prompts import build_rag_prompt
from web.routes import SNIPPET_CHARS, source_cards

HEADER = "[Источник: Doc | policy]\n[Фрагмент: Head]\n\n"


def _doc(i, body, *, header=HEADER, header_len="auto", label=None, heading="Head"):
    meta = {
        "source_display": label or f"Source {i}",
        "source_kind": "policy",
        "doc_type": "overview",
        "section_heading": heading,
    }
    if header_len == "auto":
        meta["header_len"] = str(len(header))
    elif header_len is not None:
        meta["header_len"] = header_len
    return {"id": f"doc_{i}", "text": header + body, "metadata": meta}


@pytest.fixture
def render(monkeypatch):
    """POST /ask against the real app with a pipeline stub that returns the given context documents."""

    def _render(docs):
        class Stub:
            def query(self, question):
                return {"answer": "the answer", "context_docs": docs, "from_cache": False, "model": "m", "cached_at": ""}

        monkeypatch.setattr(web_app, "load_repo_env", lambda *a, **k: False)
        monkeypatch.setattr(web_app, "RAGPipeline", Stub)
        with TestClient(web_app.app) as client:
            response = client.post("/ask", data={"question": "a question"})
        assert response.status_code == 200
        return response.text

    return _render


def test_card_numbers_match_the_fragment_numbers_in_the_prompt():
    docs = [_doc(i, f"Body number {i}.") for i in (1, 2, 3)]

    prompt_numbers = [int(n) for n in re.findall(r'<retrieved_fragment number="(\d+)"', build_rag_prompt("q", docs))]
    card_numbers = [card["number"] for card in source_cards(docs)]

    assert card_numbers == prompt_numbers == [1, 2, 3]


def test_cards_are_numbered_in_the_page_in_selection_order(render):
    page = render([_doc(i, f"Body number {i}.", label=f"Label {i}") for i in (1, 2, 3)])

    positions = [page.index(f"Фрагмент {i} · Label {i}") for i in (1, 2, 3)]
    assert positions == sorted(positions)


def test_wording_describes_retrieved_context_not_verified_citations(render):
    page = render([_doc(1, "Body.")])

    assert "Извлечённые фрагменты: контекст, переданный модели" in page
    assert "не проверяются автоматически" in page
    assert "Использованные источники" not in page


def test_no_context_section_without_context(render):
    assert "Извлечённые фрагменты" not in render([])


def test_internal_chunk_header_is_not_shown_in_the_preview(render):
    page = render([_doc(1, "The actual evidence text.")])

    assert "The actual evidence text." in page
    assert "[Фрагмент: Head]" not in page and "[Источник: Doc | policy]" not in page


@pytest.mark.parametrize("header_len", [None, "abc", "", "99999", "-5", "0"])
def test_missing_or_invalid_header_length_falls_back_to_the_full_text(render, header_len):
    page = render([_doc(1, "Evidence text.", header_len=header_len)])

    assert "Evidence text." in page
    assert "[Фрагмент: Head]" in page  # shown whole rather than cut at a wrong offset


def test_preview_is_limited_to_the_body_not_the_header():
    card = source_cards([_doc(1, "x" * 1000)])[0]

    assert card["snippet"] == "x" * SNIPPET_CHARS + "..."
    assert source_cards([_doc(1, "short")])[0]["snippet"] == "short"


def test_untrusted_chunk_text_and_metadata_are_escaped(render):
    page = render(
        [
            _doc(
                1,
                "<b>bold</b> <script>alert(1)</script>",
                label="<script>alert('label')</script>",
                heading="<img src=x onerror=alert(1)>",
            )
        ]
    )

    assert "<script>alert" not in page and "<img src=x" not in page and "<b>bold</b>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page and "&lt;b&gt;bold&lt;/b&gt;" in page
