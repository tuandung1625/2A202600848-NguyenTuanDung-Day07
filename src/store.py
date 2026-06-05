from __future__ import annotations

import os
from pathlib import Path
import json
from typing import Any, Callable
from uuid import uuid4

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
        persist_directory: str | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0
        self._storage_backend = "in_memory"

        try:
            import chromadb

            if persist_directory:
                os.makedirs(persist_directory, exist_ok=True)
                client = chromadb.PersistentClient(path=persist_directory)
                chroma_collection_name = collection_name
            else:
                client = chromadb.Client()
                # Use an instance-scoped collection for non-persistent stores so
                # separate EmbeddingStore objects do not leak state across tests.
                chroma_collection_name = f"{collection_name}_{uuid4().hex}"
            self._collection = client.get_or_create_collection(name=chroma_collection_name)
            self._use_chroma = True
            self._storage_backend = "chromadb"
        except Exception:
            self._use_chroma = False
            self._collection = None
            self._storage_backend = "json_persisted_store" if persist_directory else "in_memory"
            self._load_persisted_store()

    def _json_db_path(self) -> Path | None:
        if not self._persist_directory:
            return None
        return Path(self._persist_directory) / f"{self._collection_name}.json"

    def _load_persisted_store(self) -> None:
        db_path = self._json_db_path()
        if db_path is None or not db_path.exists():
            return

        payload = json.loads(db_path.read_text(encoding="utf-8"))
        self._store = payload.get("records", [])
        self._next_index = payload.get("next_index", len(self._store))

    def _save_persisted_store(self) -> None:
        db_path = self._json_db_path()
        if db_path is None:
            return

        db_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "collection_name": self._collection_name,
            "next_index": self._next_index,
            "records": self._store,
        }
        db_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def _make_record(self, doc: Document) -> dict[str, Any]:
        return {
            "id": f"{doc.id}_{self._next_index}",
            "content": doc.content,
            "embedding": self._embedding_fn(doc.content),
            "metadata": {
                **(doc.metadata or {}),
                "doc_id": doc.id,
            },
        }

    def _search_records(
        self, query: str, records: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        query_vec = self._embedding_fn(query)

        scored = []
        for record in records:
            score = _dot(query_vec, record["embedding"])
            scored.append((score, record))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "score": score,
                **record,
            }
            for score, record in scored[:top_k]
        ]

    def add_documents(self, docs: list[Document]) -> None:
        if not docs:
            return

        if self._use_chroma:
            ids = []
            documents = []
            embeddings = []
            metadatas = []

            for doc in docs:
                record = self._make_record(doc)
                ids.append(record["id"])
                documents.append(record["content"])
                embeddings.append(record["embedding"])
                metadatas.append(record["metadata"])

                self._next_index += 1

            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        else:
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)
                self._next_index += 1
            self._save_persisted_store()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not query:
            return []

        if self._use_chroma:
            query_vec = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
            )
            return self._format_chroma_results(results)

        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(
        self, query: str, top_k: int = 3, metadata_filter: dict = None
    ) -> list[dict]:
        if not metadata_filter:
            return self.search(query, top_k)

        if self._use_chroma:
            query_vec = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                where=metadata_filter,
            )
            return self._format_chroma_results(results)

        # In-memory filtering
        filtered = [
            r
            for r in self._store
            if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())
        ]

        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        if self._use_chroma:
            # Get all IDs with matching doc_id
            results = self._collection.get(
                where={"doc_id": doc_id}
            )

            ids = results.get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
                return True
            return False

        original_len = len(self._store)

        self._store = [
            r for r in self._store if r["metadata"].get("doc_id") != doc_id
        ]

        deleted = len(self._store) < original_len
        if deleted:
            self._save_persisted_store()
        return deleted

    def reset(self) -> None:
        if self._use_chroma:
            results = self._collection.get()
            ids = results.get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
            self._next_index = 0
            return

        self._store = []
        self._next_index = 0
        self._save_persisted_store()

    def _format_chroma_results(self, results: dict[str, Any]) -> list[dict[str, Any]]:
        output = []
        distances = results.get("distances", [[]])[0]

        for i in range(len(results["documents"][0])):
            distance = distances[i] if i < len(distances) else None
            output.append(
                {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    # Chroma returns smaller distances for better matches; convert
                    # them into a descending score so the API is consistent with
                    # the in-memory implementation and the tests.
                    "score": -distance if distance is not None else None,
                }
            )

        output.sort(
            key=lambda record: float("-inf") if record["score"] is None else record["score"],
            reverse=True,
        )
        return output
