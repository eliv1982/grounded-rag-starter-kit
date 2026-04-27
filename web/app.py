from fastapi import FastAPI

from web.routes import router

app = FastAPI(title="RAG Assistant Web")
app.include_router(router)

