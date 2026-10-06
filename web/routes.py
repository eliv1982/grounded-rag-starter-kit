from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from rag_pipeline import RAGPipeline

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

SNIPPET_CHARS = 280


def _snippet(doc: Dict[str, Any], limit: int = SNIPPET_CHARS) -> str:
    """Preview of a chunk's body: the internal chunk header (metadata `header_len`) is not shown."""
    text = str(doc.get("text") or "")
    try:
        header_len = int((doc.get("metadata") or {}).get("header_len") or 0)
    except (TypeError, ValueError):
        header_len = 0
    body = (text[header_len:] if 0 < header_len < len(text) else text).strip()
    return body[:limit] + "..." if len(body) > limit else body


def source_cards(context_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    View models for the retrieved-context cards.

    The numbering is the position in the selected context, which is exactly the `number` attribute of
    the matching <retrieved_fragment> in the prompt (app_core/generation/prompts.py). The cards show what
    the model was given, not which fragments its answer actually relies on.
    """
    cards = []
    for number, doc in enumerate(context_docs, start=1):
        meta = doc.get("metadata") or {}
        cards.append(
            {
                "number": number,
                "label": meta.get("source_display") or meta.get("source") or "Источник",
                "source_kind": meta.get("source_kind") or "",
                "doc_type": meta.get("doc_type") or "",
                "heading": meta.get("section_heading") or "",
                "snippet": _snippet(doc),
            }
        )
    return cards


def _get_pipeline(request: Request) -> RAGPipeline:
    """The pipeline created once by the application lifespan (see web/app.py)."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise RuntimeError("RAG pipeline is not initialized: application startup did not complete")
    return pipeline


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "question": "",
            "answer": "",
            "source_cards": [],
            "error": "",
            "from_cache": None,
            "model": "",
            "cached_at": "",
        },
    )


@router.get("/ask")
def ask_get():
    return RedirectResponse(url="/", status_code=303)


@router.post("/ask", response_class=HTMLResponse)
def ask(request: Request, question: str = Form(default="")):
    answer = ""
    context_docs: List[Dict[str, Any]] = []
    error = ""
    from_cache = None
    model = ""
    cached_at = ""
    normalized_question = question.strip()

    if not normalized_question:
        error = "Пожалуйста, введите вопрос."
    else:
        try:
            result = _get_pipeline(request).query(normalized_question)
            answer = result.get("answer", "")
            context_docs = result.get("context_docs") or []
            from_cache = result.get("from_cache")
            model = result.get("model", "")
            cached_at = result.get("cached_at", "")
        except Exception as exc:
            error = str(exc)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "question": normalized_question,
            "answer": answer,
            "source_cards": source_cards(context_docs),
            "error": error,
            "from_cache": from_cache,
            "model": model,
            "cached_at": cached_at,
        },
    )

