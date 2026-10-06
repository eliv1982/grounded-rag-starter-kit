from contextlib import asynccontextmanager

from fastapi import FastAPI

from app_core.config.env import load_repo_env
from rag_pipeline import RAGPipeline
from web.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # .env из корня репозитория; уже заданные переменные окружения имеют приоритет.
    load_repo_env()
    # The pipeline is built exactly once here, before the first request, and shared via app.state.
    # Building it validates the persisted vector index against the corpus/embedding/chunking
    # configuration and rebuilds it if needed (that is the only provider access at startup).
    # A missing or broken configuration therefore stops startup instead of failing the first /ask.
    try:
        app.state.pipeline = RAGPipeline()
    except Exception as exc:
        raise RuntimeError(f"RAG pipeline failed to initialize: {exc}") from exc
    yield


app = FastAPI(title="RAG Assistant Web", lifespan=lifespan)
app.include_router(router)
