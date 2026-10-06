from fastapi.testclient import TestClient

import web.app as web_app
import web.routes as web_routes


def test_lifespan_starts_and_index_renders_offline(monkeypatch):
    env_loads = []
    monkeypatch.setattr(web_app, "load_repo_env", lambda *a, **k: env_loads.append(True) or False)

    def _no_pipeline(*args, **kwargs):
        raise AssertionError("RAGPipeline (provider access) must not be created on startup or GET /")

    monkeypatch.setattr(web_routes, "RAGPipeline", _no_pipeline)
    monkeypatch.setattr(web_routes, "_pipeline", None)

    with TestClient(web_app.app) as client:  # entering the context runs the lifespan
        assert env_loads == [True]
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'name="question"' in response.text
