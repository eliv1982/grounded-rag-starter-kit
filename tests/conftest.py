import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Tests must stay offline: keep ChromaDB's anonymous telemetry from phoning home.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# Everything the cache/index lifecycle reads from the environment. Lifecycle tests start from none of it.
_LIFECYCLE_ENV = (
    "LLM_API_KEY",
    "OPENAI_API_KEY",
    "LLM_BASE_URL",
    "OPENAI_BASE_URL",
    "RAG_CHAT_MODEL",
    "RAG_EMBEDDING_MODEL",
    "RAG_CORPUS_VERSION",
    "RAG_CORPUS_CONFIG",
    "RAG_COLLECTION_NAME",
    "RAG_CHROMA_PATH",
    "RAG_CHUNK_SIZE",
    "RAG_CHUNK_OVERLAP",
    "RAG_MIN_CHUNK_LEN",
    "RAG_EMBED_BATCH_SIZE",
    "RAG_RAW_TOP_K",
    "RAG_MAX_DISTANCE",
    "RAG_FINAL_TOP_K",
    "RAG_TOP_K",
    "RAG_MAX_TOKENS",
    "RAG_TEMPERATURE",
)


class FakeEmbeddings:
    """
    Stands in for the OpenAI client (`.embeddings.create`): deterministic offline vectors.

    Vectors depend only on the text, so two embedding "models" produce same-dimension vectors and a
    model swap can only be caught by the index manifest. `fail_on_call` makes the Nth call raise.
    """

    DIM = 8

    def __init__(self):
        self.calls = []  # one (model, [texts]) per embeddings.create call
        self.fail_on_call = None  # 1-based number of the call that should raise
        self.embeddings = SimpleNamespace(create=self._create)

    @staticmethod
    def vector(text):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(b + 1) / 256 for b in digest[: FakeEmbeddings.DIM]]

    def _create(self, *, input, model):
        texts = list(input) if isinstance(input, list) else [input]
        self.calls.append((model, texts))
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("simulated embedding failure")
        data = [SimpleNamespace(index=i, embedding=self.vector(t)) for i, t in enumerate(texts)]
        return SimpleNamespace(data=data)

    @property
    def embedded_texts(self):
        return [t for _, texts in self.calls for t in texts]


class FakeLLM:
    """Records every chat-completion call; never touches the network."""

    def __init__(self, answer="fake llm answer"):
        self.answer = answer
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=f"  {self.answer}  ")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture(scope="session")
def legal_vocabulary():
    """Vocabulary that must never appear in the reusable core or in a non-legal vertical."""
    import re

    return re.compile(
        r"статья|urdg|закон|гк рф|гарант|statute|legal|юрид|\blaw\b|судеб|бенефициар", re.IGNORECASE
    )


@pytest.fixture
def lifecycle_env(monkeypatch):
    """Hermetic configuration: no lifecycle-relevant variable leaks in from the developer's shell."""
    for name in _LIFECYCLE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "test-key-not-used")
    return monkeypatch


@pytest.fixture
def fake_embeddings(lifecycle_env):
    """Route VectorStore embeddings to FakeEmbeddings (real Chroma, no provider)."""
    fake = FakeEmbeddings()
    lifecycle_env.setattr("app_core.retrieval.vector_store.get_llm_client", lambda: fake)
    return fake


@pytest.fixture
def chroma_dir(tmp_path):
    """A throw-away Chroma persist directory; the in-process client cache is dropped afterwards."""
    yield str(tmp_path / "chroma")
    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass


def _document(title, paragraphs):
    body = [f"{title} clause {i}. " + "The quick brown fox jumps over the lazy dog. " * 4 for i in range(paragraphs)]
    return title + "\n\n" + "\n\n".join(body)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    # Bytes, not write_text(): the files must hold LF on every platform.
    (root / "alpha.txt").write_bytes(_document("Alpha", 6).encode("utf-8"))
    (root / "beta.txt").write_bytes(_document("Beta", 6).encode("utf-8"))
    entries = [
        {
            "path": root / "alpha.txt",
            "source": "alpha",
            "source_display": "Alpha doc",
            "source_kind": "policy",
            "doc_type": "overview",
        },
        {
            "path": root / "beta.txt",
            "source": "beta",
            "source_display": "Beta doc",
            "source_kind": "policy",
            "doc_type": "overview",
        },
    ]
    return SimpleNamespace(root=root, entries=entries)


@pytest.fixture
def index_env(fake_embeddings, lifecycle_env):
    """Small chunks and tiny embedding batches so a rebuild makes several provider calls."""
    for name, value in {
        "RAG_CHUNK_SIZE": "400",
        "RAG_CHUNK_OVERLAP": "80",
        "RAG_MIN_CHUNK_LEN": "30",
        "RAG_EMBED_BATCH_SIZE": "2",
        "RAG_EMBEDDING_MODEL": "embed-a",
    }.items():
        lifecycle_env.setenv(name, value)
    return lifecycle_env


@pytest.fixture
def fake_llm_class():
    """The recording chat client class (pytest fixtures are the supported way to share it)."""
    return FakeLLM


@pytest.fixture
def pipeline_factory(index_env, fake_llm_class, chroma_dir, tmp_path):
    """
    Build `RAGPipeline`s over a real temporary Chroma index, with fake embeddings and a recording fake
    chat client. Returns `make(**RAGPipeline kwargs) -> (pipeline, llm)`; cache and index files are shared
    between calls, so a second call behaves like a restart. A cutoff of 2 lets every fake-vector chunk qualify.
    """
    import rag_pipeline

    index_env.setenv("RAG_MAX_DISTANCE", "2")
    index_env.setenv("RAG_FINAL_TOP_K", "3")

    def make(**kwargs):
        llm = fake_llm_class()
        index_env.setattr(rag_pipeline, "get_llm_client", lambda: llm)
        pipeline = rag_pipeline.RAGPipeline(
            cache_db_path=str(tmp_path / "cache.db"), persist_directory=chroma_dir, **kwargs
        )
        return pipeline, llm

    return make
