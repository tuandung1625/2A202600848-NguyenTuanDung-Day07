from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from main import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION,
    DEFAULT_DISEASE_DIR,
    DEFAULT_LLM_PROVIDER,
    build_chunker,
    build_embedder,
    build_llm,
    create_store,
    load_disease_documents,
)
from src.agent import KnowledgeBaseAgent

load_dotenv(override=False)

app = Flask(__name__)


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def _collection_embedding_dimension(store) -> int | None:
    if not getattr(store, "_use_chroma", False) or store.get_collection_size() == 0:
        return None

    try:
        result = store._collection.get(limit=1, include=["embeddings"])
        embeddings = result.get("embeddings") or []
        if embeddings and embeddings[0] is not None:
            return len(embeddings[0])
    except Exception:
        return None
    return None


def _embedder_dimension(embedder) -> int:
    return len(embedder("dimension check"))


def _validate_store_embedding_dimension(store, embedder, provider_name: str) -> None:
    collection_dim = _collection_embedding_dimension(store)
    if collection_dim is None:
        return

    query_dim = _embedder_dimension(embedder)
    if collection_dim != query_dim:
        raise RuntimeError(
            "Embedding dimension mismatch: the persisted collection was built with "
            f"{collection_dim}-dim vectors, but the current web demo is using "
            f"'{provider_name}' embeddings with {query_dim} dimensions. "
            "Set WEB_EMBEDDING_PROVIDER or EMBEDDING_PROVIDER to the same backend "
            "used when loading the database, or rebuild the collection with the "
            "current backend."
        )


@lru_cache(maxsize=1)
def get_runtime() -> dict[str, object]:
    embedding_provider = _env("WEB_EMBEDDING_PROVIDER", _env("EMBEDDING_PROVIDER", "mock"))
    llm_provider = _env("WEB_LLM_PROVIDER", _env("LLM_PROVIDER", DEFAULT_LLM_PROVIDER))
    collection = _env("WEB_COLLECTION", DEFAULT_COLLECTION)
    persist_dir = _env("WEB_PERSIST_DIR", _env("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_DIR))
    data_dir = _env("WEB_DATA_DIR", DEFAULT_DISEASE_DIR)
    top_k = int(_env("WEB_TOP_K", "3"))

    embedder = build_embedder(embedding_provider)
    llm = build_llm(llm_provider)
    store = create_store(
        collection,
        embedding_fn=embedder,
        persist_directory=persist_dir,
    )

    # Make the demo easy to run: if the collection is empty, preload the disease data.
    if store.get_collection_size() == 0:
        chunker = build_chunker(
            "recursive",
            chunk_size=500,
            overlap=50,
            max_sentences=3,
        )
        docs = load_disease_documents(data_dir, chunker=chunker)
        store.add_documents(docs)

    _validate_store_embedding_dimension(store, embedder, embedding_provider)

    agent = KnowledgeBaseAgent(store=store, llm_fn=llm)
    return {
        "agent": agent,
        "store": store,
        "top_k": top_k,
        "collection": collection,
        "persist_dir": persist_dir,
        "embedding_backend": getattr(embedder, "_backend_name", embedder.__class__.__name__),
        "llm_backend": getattr(llm, "_backend_name", llm.__class__.__name__),
    }


@app.get("/")
def index():
    try:
        runtime = get_runtime()
    except Exception as exc:
        return render_template(
            "chat.html",
            collection=_env("WEB_COLLECTION", DEFAULT_COLLECTION),
            persist_dir=_env("WEB_PERSIST_DIR", _env("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_DIR)),
            embedding_backend=_env("WEB_EMBEDDING_PROVIDER", _env("EMBEDDING_PROVIDER", "mock")),
            llm_backend=_env("WEB_LLM_PROVIDER", _env("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)),
            stored_records="unknown",
            startup_error=str(exc),
        )

    store = runtime["store"]
    return render_template(
        "chat.html",
        collection=runtime["collection"],
        persist_dir=runtime["persist_dir"],
        embedding_backend=runtime["embedding_backend"],
        llm_backend=runtime["llm_backend"],
        stored_records=store.get_collection_size(),
        startup_error=None,
    )


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        runtime = get_runtime()
        top_k = int(runtime["top_k"])
        store = runtime["store"]
        agent = runtime["agent"]
        results = store.search(question, top_k=top_k)
        answer = agent.answer(question, top_k=top_k)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    sources = []
    for item in results:
        metadata = item.get("metadata", {})
        preview = item.get("content", "").replace("\n", " ").strip()
        sources.append(
            {
                "score": item.get("score"),
                "disease": metadata.get("disease"),
                "source_file": metadata.get("source_file"),
                "chunk_index": metadata.get("chunk_index"),
                "preview": preview[:240],
            }
        )

    return jsonify(
        {
            "answer": answer,
            "sources": sources,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
