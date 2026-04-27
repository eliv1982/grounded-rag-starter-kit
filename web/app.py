import logging

from fastapi import FastAPI

from web.routes import router

app = FastAPI(title="RAG Assistant Web")
app.include_router(router)


@app.on_event("startup")
def announce_web_url():
    logger = logging.getLogger("uvicorn.error")
    logger.info("Web interface: http://127.0.0.1:8010")
    for route in app.routes:
        logger.info("Route: %s %s", route.path, getattr(route, "methods", ""))

