from contextlib import asynccontextmanager

from fastapi import FastAPI

from app_core.config.env import load_repo_env
from web.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # .env из корня репозитория; уже заданные переменные окружения имеют приоритет.
    # RAGPipeline создаётся лениво при первом /ask, поэтому запуск не обращается к провайдерам.
    load_repo_env()
    yield


app = FastAPI(title="RAG Assistant Web", lifespan=lifespan)
app.include_router(router)
