"""
Модуль работы с векторным хранилищем ChromaDB.
Загрузка нескольких источников с метаданными и нарезка на чанки. Предметно-зависимое (границы разделов,
подписи типов источников, заголовок чанка, язык разбиения на предложения) приходит из DomainProfile
(app_core/config/profile.py); без профиля действует нейтральный профиль по умолчанию.
"""

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Tuple

import chromadb
import time
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError
from openai import APIConnectionError, APITimeoutError

from app_core.config.knowledge import default_collection_name
from app_core.config.profile import DEFAULT_PROFILE, DomainProfile
from app_core.lifecycle import (
    STATE_COMPLETE,
    STATE_INCOMPLETE,
    build_index_identity,
    corpus_fingerprint,
    endpoint_identity,
    fingerprint,
    index_problems,
    manifest_metadata,
    parse_manifest,
)
from app_core.llm.client import get_llm_client, resolve_base_url

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Persisted indexes depend on how chunks are built. The chunk_size/overlap/min_chunk_len settings and the
# profile's chunking rules are recorded in the index manifest automatically. Bump CHUNKING_VERSION in
# app_core/lifecycle.py whenever the chunking algorithm or chunk layout below changes, so existing indexes
# are rebuilt.


class IndexBuildError(RuntimeError):
    """Ingestion failed; the collection stays marked incomplete and is rebuilt on the next start."""


@dataclass(frozen=True)
class IndexStatus:
    """Outcome of `VectorStore.ensure_index`."""

    action: str  # "reused" (valid manifest, nothing embedded) or "rebuilt"
    identity: Dict[str, Any]  # the index identity the collection now matches
    chunk_count: int
    reasons: Tuple[str, ...] = ()  # why a rebuild was needed


class VectorStore:
    """Векторное хранилище на основе ChromaDB."""

    _ADD_BATCH_SIZE = 100  # chunks per collection.add() call

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        profile: Optional[DomainProfile] = None,
    ):
        self.profile = profile or DEFAULT_PROFILE
        self.collection_name = collection_name or default_collection_name()
        collection_name = self.collection_name
        if persist_directory is None:
            persist_directory = str(PROJECT_ROOT / "runtime" / "chroma_db")
        self.persist_directory = persist_directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Chroma's anonymous usage telemetry is on by default; this store must not contact anything but the
        # configured provider endpoint, so it is switched off explicitly (it carries no document text).
        self.client = chromadb.PersistentClient(
            path=persist_directory, settings=ChromaSettings(anonymized_telemetry=False)
        )

        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Коллекция '{collection_name}' загружена. Документов: {self.collection.count()}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"Создана новая коллекция '{collection_name}'")

        self.llm_client = get_llm_client()
        self.embedding_model = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
        # Embeddings go through the same OpenAI-compatible endpoint as chat (configured identity only).
        self.embedding_endpoint = endpoint_identity(resolve_base_url())
        self.corpus_version = os.getenv("RAG_CORPUS_VERSION", "1")
        self.chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "800"))
        self.chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
        self.min_chunk_len = int(os.getenv("RAG_MIN_CHUNK_LEN", "80"))

    def _split_sentences(self, text: str) -> List[str]:
        try:
            from pysbd import Segmenter

            segmenter = Segmenter(language=self.profile.sentence_language, clean=False)
            return [s.strip() for s in segmenter.segment(text) if s.strip()]
        except Exception:
            return self._split_sentences_regex(text)

    def _split_sentences_regex(self, text: str) -> List[str]:
        parts = re.split(r"([.!?]+\s+)", text)
        full: List[str] = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts):
                full.append((parts[i] + parts[i + 1]).strip())
                i += 2
            else:
                if parts[i].strip():
                    full.append(parts[i].strip())
                i += 1
        return [s for s in full if s]

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        if len(text) <= overlap_size:
            return text
        overlap_candidate = text[-overlap_size:]
        sentence_starts = [". ", "! ", "? ", "\n"]
        best_start = 0
        for delimiter in sentence_starts:
            pos = overlap_candidate.find(delimiter)
            if pos != -1 and pos > best_start:
                best_start = pos + len(delimiter)
        if best_start > 0:
            return overlap_candidate[best_start:].strip()
        return overlap_candidate.strip()

    def _split_long_block(self, paragraph: str, chunk_size: int, overlap: int) -> List[str]:
        sentences = self._split_sentences(paragraph)
        chunks: List[str] = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= chunk_size:
                current = (current + " " + sentence).strip() if current else sentence
            else:
                if current:
                    chunks.append(current)
                    overlap_text = self._get_overlap_text(current, overlap)
                    current = (overlap_text + " " + sentence).strip() if overlap_text else sentence
                else:
                    current = sentence
        if current:
            chunks.append(current)
        return chunks

    def _chunk_semantic(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(current) + len(paragraph) + 2 <= chunk_size:
                current = current + "\n\n" + paragraph if current else paragraph
            elif current:
                chunks.append(current)
                overlap_text = self._get_overlap_text(current, overlap)
                current = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
            else:
                if len(paragraph) > chunk_size:
                    sent_chunks = self._split_long_block(paragraph, chunk_size, overlap)
                    if sent_chunks:
                        chunks.extend(sent_chunks[:-1])
                        current = sent_chunks[-1]
                else:
                    current = paragraph
        if current:
            chunks.append(current)
        return [c for c in chunks if len(c) >= self.min_chunk_len]

    def _split_oversized_chunk(self, text: str, max_len: int, overlap: int) -> List[str]:
        cleaned = (text or "").strip()
        if not cleaned:
            return []
        if max_len <= 0:
            return [cleaned]
        if len(cleaned) <= max_len:
            return [cleaned]

        overlap = max(0, min(overlap, max_len // 2)) if max_len > 1 else 0
        step = max(1, max_len - overlap)
        min_natural_cut = max(1, int(max_len * 0.5))
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

        chunks: List[str] = []
        start = 0
        total_len = len(cleaned)

        while start < total_len:
            end = min(start + max_len, total_len)
            if end >= total_len:
                tail = cleaned[start:].strip()
                if tail:
                    chunks.append(tail)
                break

            window = cleaned[start:end]
            cut = -1
            for sep in separators:
                idx = window.rfind(sep)
                if idx >= min_natural_cut:
                    cut = start + idx + len(sep)
                    break

            if cut <= start:
                cut = end

            piece = cleaned[start:cut].strip()
            if piece:
                chunks.append(piece)
            else:
                cut = end

            next_start = max(0, cut - overlap)
            if next_start <= start:
                next_start = start + step
            start = next_start

        return chunks

    def _enforce_hard_chunk_limit(
        self, rows: List[Tuple[str, Dict[str, str]]], max_len: int, overlap: int
    ) -> List[Tuple[str, Dict[str, str]]]:
        if max_len <= 0:
            return rows

        out: List[Tuple[str, Dict[str, str]]] = []
        for doc_text, meta in rows:
            if len(doc_text) <= max_len:
                out.append((doc_text, meta))
                continue

            split_parts = self._split_oversized_chunk(doc_text, max_len, overlap)
            if not split_parts:
                continue

            for idx, part in enumerate(split_parts):
                if not part:
                    continue
                new_meta = dict(meta)
                new_meta["hard_split_index"] = str(idx)
                if idx > 0:
                    new_meta["header_len"] = "0"  # only the first part still starts with the chunk header
                out.append((part, new_meta))

        return out

    @staticmethod
    def _split_sections(text: str, boundary: Pattern[str]) -> List[str]:
        """Split at the profile's section boundaries; text without any boundary stays one section."""
        parts = [p.strip() for p in boundary.split(text) if p.strip()]
        if len(parts) <= 1:
            stripped = text.strip()
            return [stripped] if stripped else []
        return parts

    def _section_heading(self, section: str, max_len: int = 160) -> str:
        line = section.strip().split("\n", 1)[0].strip()
        if len(line) > max_len:
            return line[:max_len] + "…"
        return line

    def _fit_header(self, source_display: str, source_kind: str, heading: str) -> str:
        """
        The chunk header, capped at half the chunk size so the body always keeps a sane budget.

        An over-long header first loses the end of the heading; if the rest of the header alone is still
        too long (a tiny chunk size, a long source name or template) the chunk gets no header: the
        source stays available in the chunk metadata.
        """
        cap = self.chunk_size // 2
        header = self.profile.format_header(source_display, source_kind, heading)
        if len(header) <= cap:
            return header
        cut = len(header) - cap + 1  # characters to drop, one of them replaced by the ellipsis
        if len(heading) > cut:
            header = self.profile.format_header(source_display, source_kind, heading[: len(heading) - cut] + "…")
            if len(header) <= cap:
                return header
        return ""

    def _build_chunks_for_file(
        self,
        text: str,
        source: str,
        source_display: str,
        source_kind: str,
        doc_type: str,
    ) -> List[Tuple[str, Dict[str, str]]]:
        chunk_size = self.chunk_size
        overlap = self.chunk_overlap

        boundary = self.profile.section_pattern(doc_type)
        if boundary is not None:
            sections = self._split_sections(text, boundary)
        else:
            sections = [text.strip()] if text.strip() else []

        out: List[Tuple[str, Dict[str, str]]] = []
        for section in sections:
            heading = self._section_heading(section)
            prefix = self._fit_header(source_display, source_kind, heading)
            max_body_len = max(1, chunk_size - len(prefix))
            # header_len lets presentation layers show the chunk body without the header boilerplate.
            base_meta = {
                "source": source,
                "source_display": source_display,
                "source_kind": source_kind,
                "doc_type": doc_type,
                "section_heading": heading,
                "header_len": str(len(prefix)),
            }
            if len(section) <= chunk_size:
                meta = {**base_meta, "subchunk_index": ""}
                for body_part in self._split_oversized_chunk(section, max_body_len, overlap):
                    doc_text = f"{prefix}{body_part}"
                    if len(doc_text) < self.min_chunk_len:
                        continue
                    out.append((doc_text, meta))
                continue

            subchunks = self._chunk_semantic(section, chunk_size, overlap)
            for i, sub in enumerate(subchunks):
                body = sub.strip()
                if len(body) < self.min_chunk_len:
                    continue
                meta = {**base_meta, "subchunk_index": str(i)}
                for body_part in self._split_oversized_chunk(body, max_body_len, overlap):
                    doc_text = f"{prefix}{body_part}"
                    if len(doc_text) < self.min_chunk_len:
                        continue
                    out.append((doc_text, meta))
        return out

    def _resolve_path(self, path: Path, base_dir: Path) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return (base_dir / p).resolve()

    def _read_sources(self, corpus_entries: List[Dict[str, Any]], base: Path) -> List[Dict[str, Any]]:
        """Read every corpus file once; the same text feeds the corpus fingerprint and the chunker."""
        sources: List[Dict[str, Any]] = []
        for entry in corpus_entries:
            path = self._resolve_path(Path(entry["path"]), base)
            if not path.exists():
                raise FileNotFoundError(f"Файл корпуса не найден: {path}")

            # Same newline handling as a text-mode open(): "\r\n" and "\r" become "\n".
            text = path.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
            sources.append(
                {
                    "source": str(entry["source"]),
                    "source_display": str(entry["source_display"]),
                    "source_kind": str(entry.get("source_kind", "unknown")),
                    "doc_type": str(entry.get("doc_type", "overview")),
                    "file_name": path.name,
                    "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "text": text,
                }
            )
        return sources

    def _build_rows(self, sources: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, str]]]:
        """Chunk the corpus. Pure and local: nothing is embedded or written."""
        all_rows: List[Tuple[str, Dict[str, str]]] = []
        for source in sources:
            rows = self._build_chunks_for_file(
                source["text"],
                source=source["source"],
                source_display=source["source_display"],
                source_kind=source["source_kind"],
                doc_type=source["doc_type"],
            )
            print(f"  {source['file_name']}: {len(rows)} чанков")
            all_rows.extend(rows)

        if not all_rows:
            raise ValueError("Корпус пуст после нарезки")

        before_total = len(all_rows)
        before_max_len = max((len(doc) for doc, _ in all_rows), default=0)
        print(
            "[INFO] Chunk summary before hard-limit pass: "
            f"total={before_total} max_len={before_max_len}"
        )

        all_rows = self._enforce_hard_chunk_limit(
            all_rows, max_len=self.chunk_size, overlap=self.chunk_overlap
        )

        if not all_rows:
            raise ValueError("Корпус пуст после hard-limit pass")

        after_total = len(all_rows)
        after_max_len = max((len(doc) for doc, _ in all_rows), default=0)
        print(
            "[INFO] Chunk summary after hard-limit pass: "
            f"total={after_total} max_len={after_max_len}"
        )
        return all_rows

    def ensure_index(
        self,
        corpus_entries: List[Dict[str, Any]],
        base_dir: Optional[Path] = None,
    ) -> IndexStatus:
        """
        Make this store's collection a complete index of `corpus_entries`, or fail.

        A non-empty collection is NOT trusted by itself: it is reused only if its manifest is complete
        and matches the corpus (version + manifest/content fingerprint), the embedding model and
        endpoint, and the chunking settings, and the stored chunk count agrees. Otherwise this
        store's own collection (and only that one) is dropped and rebuilt:

          1. the corpus is read and chunked locally (a bad corpus fails before anything is deleted);
          2. the collection is recreated with a manifest marked "incomplete";
          3. chunks are embedded and written;
          4. only then is the manifest marked "complete".

        If anything fails after step 2 the collection stays "incomplete" and the next start rebuilds.
        corpus_entries: path (Path | str), source, source_display, source_kind, doc_type
        """
        if not corpus_entries:
            raise ValueError("Не заданы источники корпуса: список corpus_entries пуст")

        base = base_dir or Path(__file__).resolve().parent
        sources = self._read_sources(corpus_entries, base)
        identity = build_index_identity(
            corpus_version=self.corpus_version,
            corpus_fingerprint=corpus_fingerprint(sources),
            embedding_model=self.embedding_model,
            embedding_endpoint=self.embedding_endpoint,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            min_chunk_len=self.min_chunk_len,
            profile_fingerprint=fingerprint(self.profile.chunking_identity()),
        )

        count = self.collection.count()
        problems = index_problems(identity, self.collection.metadata, count)
        if not problems:
            print(f"[INDEX] '{self.collection_name}': manifest is complete and matches, {count} chunks reused")
            return IndexStatus("reused", identity, count)

        print(f"[INDEX] '{self.collection_name}' will be rebuilt: {'; '.join(problems)}")
        effective_embed_batch_size = max(1, int(os.getenv("RAG_EMBED_BATCH_SIZE", "16")))
        print(
            "[INFO] Effective config: "
            f"embedding_model={self.embedding_model} "
            f"chunk_size={self.chunk_size} "
            f"chunk_overlap={self.chunk_overlap} "
            f"min_chunk_len={self.min_chunk_len} "
            f"embed_batch_size={effective_embed_batch_size}"
        )

        rows = self._build_rows(sources)
        self._reset_collection(identity, expected_count=len(rows))
        try:
            self._ingest(rows)
            written = self.collection.count()
            if written != len(rows):
                raise RuntimeError(f"expected {len(rows)} chunks in the collection, found {written}")
            self.collection.modify(
                metadata=manifest_metadata(identity, state=STATE_COMPLETE, chunk_count=len(rows))
            )
        except Exception as exc:
            raise IndexBuildError(
                f"Indexing of collection '{self.collection_name}' did not finish ({exc.__class__.__name__}: {exc}). "
                "The index stays marked incomplete and will be rebuilt on the next start."
            ) from exc

        print(f"Загружено {len(rows)} фрагментов в '{self.collection_name}'")
        return IndexStatus("rebuilt", identity, len(rows), tuple(problems))

    def _reset_collection(self, identity: Dict[str, Any], expected_count: int) -> None:
        """Drop and recreate this store's own collection, marked incomplete. Other collections are never touched."""
        try:
            self.client.delete_collection(self.collection_name)
        except NotFoundError:
            pass
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                **manifest_metadata(identity, state=STATE_INCOMPLETE, chunk_count=expected_count),
            },
        )

    def _ingest(self, rows: List[Tuple[str, Dict[str, str]]]) -> None:
        print(f"Всего чанков: {len(rows)}. Создание эмбеддингов…")
        documents = [r[0] for r in rows]
        metadatas = [r[1] for r in rows]
        embeddings = self._create_embeddings_batched(documents)

        ids = [f"doc_{i}" for i in range(len(documents))]
        batch = self._ADD_BATCH_SIZE
        for start in range(0, len(documents), batch):
            end = min(start + batch, len(documents))
            self.collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )
            print(f"  В Chroma записано {end}/{len(documents)}")

    def index_manifest(self) -> Optional[Dict[str, Any]]:
        """The index manifest stored with the collection, or None if it has none."""
        return parse_manifest(self.collection.metadata)

    def load_documents(self, file_path: str, base_dir: Optional[Path] = None) -> IndexStatus:
        """Обратная совместимость: один файл как корпус из одного источника."""
        base = base_dir or Path(__file__).resolve().parent
        p = self._resolve_path(Path(file_path), base)
        return self.ensure_index(
            [
                {
                    "path": p,
                    "source": "single_file",
                    "source_display": p.name,
                    "source_kind": "unknown",
                    "doc_type": "overview",
                }
            ],
            base_dir=base,
        )

    @staticmethod
    def _embedding_connection_hint(exc: BaseException) -> str:
        cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
        tail = f" Детали: {cause}" if cause else ""
        return (
            "Не удалось связаться с API OpenAI (Connection error / таймаут). "
            "Проверьте интернет, VPN (если API недоступен из вашей сети), файрвол и корпоративный прокси. "
            "Для прокси задайте HTTPS_PROXY в системе или в PowerShell: "
            "$env:HTTPS_PROXY='http://127.0.0.1:ПОРТ'. "
            "Можно увеличить OPENAI_TIMEOUT (сек) и уменьшить RAG_EMBED_BATCH_SIZE. "
            "При использовании зеркала/шлюза укажите OPENAI_BASE_URL."
            + tail
        )

    def _create_embeddings_batched(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        if batch_size is None:
            batch_size = int(os.getenv("RAG_EMBED_BATCH_SIZE", "16"))
        batch_size = max(1, batch_size)
        embed_retries = max(1, int(os.getenv("OPENAI_EMBED_RETRIES", "5")))
        all_emb: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            last_err: Optional[BaseException] = None
            for attempt in range(embed_retries):
                try:
                    response = self.llm_client.embeddings.create(
                        input=batch,
                        model=self.embedding_model,
                    )
                    by_index = sorted(response.data, key=lambda d: d.index)
                    all_emb.extend([d.embedding for d in by_index])
                    break
                except (APIConnectionError, APITimeoutError) as e:
                    last_err = e
                    wait = min(30, 2**attempt)
                    print(
                        f"  Сеть/API: попытка {attempt + 1}/{embed_retries} не удалась, "
                        f"пауза {wait} с… ({e.__class__.__name__})"
                    )
                    time.sleep(wait)
                except Exception:
                    batch_lengths = [len(item) for item in batch]
                    max_batch_len = max(batch_lengths) if batch_lengths else 0
                    min_batch_len = min(batch_lengths) if batch_lengths else 0
                    print(
                        "[ERROR] Embedding batch failed: "
                        f"start={i} batch_size={len(batch)} "
                        f"min_item_len={min_batch_len} max_item_len={max_batch_len} "
                        f"model={self.embedding_model}"
                    )
                    raise
            else:
                raise RuntimeError(self._embedding_connection_hint(last_err)) from last_err
        return all_emb

    def _create_embedding(self, text: str) -> List[float]:
        response = self.llm_client.embeddings.create(
            input=text,
            model=self.embedding_model,
        )
        return response.data[0].embedding

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self._create_embedding(query)
        n = min(top_k, max(1, self.collection.count()))
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
        )
        documents: List[Dict[str, Any]] = []
        if results["documents"] and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                meta = {}
                if results.get("metadatas") and results["metadatas"][0]:
                    meta = dict(results["metadatas"][0][i] or {})
                documents.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "distance": results["distances"][0][i] if results.get("distances") else None,
                        "metadata": meta,
                    }
                )
        return documents

    def get_collection_stats(self) -> Dict[str, Any]:
        manifest = self.index_manifest() or {}
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "index_state": manifest.get("state"),
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }


if __name__ == "__main__":
    import sys

    from app_core.config.env import load_repo_env

    from app_core.config.knowledge import default_corpus_entries, default_profile
    from app_core.llm.client import resolve_api_key

    load_repo_env()

    if not resolve_api_key():
        print("Ошибка: задайте LLM_API_KEY (или устаревший OPENAI_API_KEY)")
        sys.exit(1)
    if len(sys.argv) < 2:
        print('Использование: python -m app_core.retrieval.vector_store "вопрос по корпусу из RAG_CORPUS_CONFIG"')
        sys.exit(2)

    # Deliberately a collection of its own, separate from the app's.
    vs = VectorStore(collection_name="test_collection", profile=default_profile())
    vs.ensure_index(default_corpus_entries())

    r = vs.search(" ".join(sys.argv[1:]), top_k=4)
    for i, doc in enumerate(r, 1):
        print(f"\n{i}. {doc['metadata'].get('source_display', '')} | {doc['text'][:180]}…")

