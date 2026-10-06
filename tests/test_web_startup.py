from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import web.app as web_app


class FakePipeline:
    """Counts constructions so tests can prove the web app builds exactly one pipeline."""

    created = []

    def __init__(self):
        self.queries = []
        FakePipeline.created.append(self)

    def query(self, question):
        self.queries.append(question)
        return {
            "answer": f"answer to {question}",
            "context_docs": [],
            "from_cache": False,
            "model": "fake-model",
            "cached_at": "",
        }


@pytest.fixture
def web(monkeypatch):
    """The real app with env loading stubbed and RAGPipeline replaced by FakePipeline."""
    FakePipeline.created = []
    events = []
    monkeypatch.setattr(web_app, "load_repo_env", lambda *a, **k: events.append("env") or False)

    class _Recording(FakePipeline):
        def __init__(self):
            events.append("pipeline")
            super().__init__()

    monkeypatch.setattr(web_app, "RAGPipeline", _Recording)
    return events


def test_lifespan_loads_env_then_creates_one_pipeline_before_any_request(web):
    with TestClient(web_app.app) as client:  # entering the context runs the lifespan
        assert web == ["env", "pipeline"]
        assert len(FakePipeline.created) == 1
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'name="question"' in response.text
    assert len(FakePipeline.created) == 1  # serving the page did not build another one


def test_requests_reuse_the_pipeline_created_at_startup(web):
    with TestClient(web_app.app) as client:
        for question in ("first", "second", "third"):
            response = client.post("/ask", data={"question": question})
            assert response.status_code == 200
            assert f"answer to {question}" in response.text

    assert len(FakePipeline.created) == 1
    assert FakePipeline.created[0].queries == ["first", "second", "third"]


def test_simultaneous_requests_share_one_pipeline(web):
    with TestClient(web_app.app) as client:
        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(lambda i: client.post("/ask", data={"question": f"q{i}"}), range(16)))

    assert all(r.status_code == 200 for r in responses)
    assert len(FakePipeline.created) == 1
    assert sorted(FakePipeline.created[0].queries) == sorted(f"q{i}" for i in range(16))


def test_each_application_start_builds_its_own_single_pipeline(web):
    with TestClient(web_app.app):
        pass
    with TestClient(web_app.app) as client:
        client.post("/ask", data={"question": "q"})

    assert len(FakePipeline.created) == 2  # one per start, never one per request
    assert FakePipeline.created[0].queries == [] and FakePipeline.created[1].queries == ["q"]


def test_startup_fails_clearly_when_the_pipeline_cannot_initialize(monkeypatch):
    monkeypatch.setattr(web_app, "load_repo_env", lambda *a, **k: False)

    def _broken():
        raise ValueError("RAG_CORPUS_CONFIG is not set")

    monkeypatch.setattr(web_app, "RAGPipeline", _broken)

    with pytest.raises(RuntimeError, match="RAG pipeline failed to initialize: RAG_CORPUS_CONFIG is not set") as raised:
        with TestClient(web_app.app):
            pytest.fail("the application must not start serving")

    assert isinstance(raised.value.__cause__, ValueError)


def test_request_without_startup_reports_a_clear_error(web, monkeypatch):
    monkeypatch.setattr(web_app.app.state, "pipeline", None, raising=False)
    client = TestClient(web_app.app)  # no `with`: the lifespan does not run

    response = client.post("/ask", data={"question": "q"})

    assert response.status_code == 200
    assert "RAG pipeline is not initialized" in response.text
    assert FakePipeline.created == []
