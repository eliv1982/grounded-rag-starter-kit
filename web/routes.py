from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from rag_pipeline import RAGPipeline

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
_pipeline: Optional[RAGPipeline] = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "question": "",
            "answer": "",
            "context_docs": [],
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
            result = _get_pipeline().query(normalized_question)
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
            "context_docs": context_docs,
            "error": error,
            "from_cache": from_cache,
            "model": model,
            "cached_at": cached_at,
        },
    )

