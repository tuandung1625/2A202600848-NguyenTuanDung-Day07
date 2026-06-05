from __future__ import annotations

from typing import Any, Callable

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
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

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

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not query:
            return []

        if self._use_chroma:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
            )

            output = []
            for i in range(len(results["documents"][0])):
                output.append(
                    {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": results["distances"][0][i]
                        if "distances" in results
                        else None,
                    }
                )
            return output

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
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=metadata_filter,
            )

            output = []
            for i in range(len(results["documents"][0])):
                output.append(
                    {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": results["distances"][0][i]
                        if "distances" in results
                        else None,
                    }
                )
            return output

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

        return len(self._store) < original_len
