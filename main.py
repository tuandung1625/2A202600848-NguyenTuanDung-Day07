from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import (
    ChunkingStrategyComparator,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
)
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.llm import GEMINI_CHAT_MODEL, OPENAI_CHAT_MODEL, GeminiChatLLM, MockLLM, OpenAIChatLLM
from src.models import Document
from src.store import EmbeddingStore

DEFAULT_CHROMA_DIR = "./chroma_data"
DEFAULT_DISEASE_DIR = "data/desease_data"
DEFAULT_COLLECTION = "disease_chunks"
DEFAULT_LLM_PROVIDER = "mock"

SAMPLE_FILES = [
    "data/python_intro.txt",
    "data/vector_store_notes.md",
    "data/rag_system_design.md",
    "data/customer_support_playbook.txt",
    "data/chunking_experiment_report.md",
    "data/vi_retrieval_notes.md",
]


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


def build_chunker(
    strategy: str,
    *,
    chunk_size: int,
    overlap: int,
    max_sentences: int,
):
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    if strategy == "sentences":
        return SentenceChunker(max_sentences_per_chunk=max_sentences)
    if strategy == "recursive":
        return RecursiveChunker(chunk_size=chunk_size)
    raise ValueError(f"Unsupported chunking strategy: {strategy}")


def build_embedder(provider: str):
    provider = provider.strip().lower()

    if provider == "local":
        return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
    if provider == "openai":
        return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
    return _mock_embed


def build_llm(provider: str):
    provider = provider.strip().lower()

    if provider == "openai":
        return OpenAIChatLLM(model_name=os.getenv("OPENAI_CHAT_MODEL", OPENAI_CHAT_MODEL))
    if provider == "gemini":
        return GeminiChatLLM(model_name=os.getenv("GEMINI_CHAT_MODEL", GEMINI_CHAT_MODEL))
    return MockLLM()


def load_disease_documents(
    data_dir: str,
    *,
    chunker=None,
) -> list[Document]:
    base_dir = Path(data_dir)
    docs: list[Document] = []

    for path in sorted(base_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        source_line = text.splitlines()[0] if text.strip() else ""
        source_url = source_line.replace("Source URL:", "").strip() if source_line.startswith("Source URL:") else ""
        disease_name = path.stem

        if chunker is None:
            docs.append(
                Document(
                    id=disease_name,
                    content=text,
                    metadata={
                        "disease": disease_name,
                        "source_file": path.name,
                        "source_url": source_url,
                    },
                )
            )
            continue

        chunks = chunker.chunk(text)
        for index, chunk in enumerate(chunks):
            docs.append(
                Document(
                    id=f"{disease_name}_{index}",
                    content=chunk,
                    metadata={
                        "disease": disease_name,
                        "source_file": path.name,
                        "source_url": source_url,
                        "chunk_index": index,
                    },
                )
            )

    return docs


def create_store(
    collection_name: str,
    *,
    embedding_fn,
    persist_directory: str | None,
) -> EmbeddingStore:
    return EmbeddingStore(
        collection_name=collection_name,
        embedding_fn=embedding_fn,
        persist_directory=persist_directory,
    )


def print_search_results(results: list[dict], *, preview_chars: int = 200) -> None:
    if not results:
        print("No results found.")
        return

    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        preview = result.get("content", "")[:preview_chars].replace("\n", " ")
        print(f"{index}. score={result.get('score')}")
        print(f"   disease={metadata.get('disease')} source_file={metadata.get('source_file')} chunk_index={metadata.get('chunk_index')}")
        print(f"   preview={preview}...")


def run_sample_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        return 1

    embedder = build_embedder(os.getenv(EMBEDDING_PROVIDER_ENV, "mock"))
    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = create_store("manual_test_store", embedding_fn=embedder, persist_directory=None)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    print_search_results(search_results, preview_chars=120)

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=MockLLM())
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0


def run_compare_chunking(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"Missing file: {path}")
        return 1

    text = path.read_text(encoding="utf-8")
    result = ChunkingStrategyComparator().compare(text, chunk_size=args.chunk_size)

    print(f"=== Chunking Comparison: {path.name} ===")
    for strategy, stats in result.items():
        print(
            f"{strategy}: count={stats['count']}, "
            f"avg_length={stats['avg_length']:.1f}, "
            f"min={stats['min_length']}, max={stats['max_length']}"
        )
    return 0


def run_load_disease_db(args: argparse.Namespace) -> int:
    chunker = build_chunker(
        args.chunking,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_sentences=args.max_sentences,
    )
    embedder = build_embedder(args.embedding_provider)
    store = create_store(
        args.collection,
        embedding_fn=embedder,
        persist_directory=args.persist_dir,
    )

    if args.reset:
        store.reset()

    docs = load_disease_documents(args.data_dir, chunker=chunker)
    store.add_documents(docs)

    print("=== Disease Data Loaded ===")
    print(f"data_dir={args.data_dir}")
    print(f"chunking={args.chunking}")
    print(f"chunk_size={args.chunk_size}")
    if args.chunking == "fixed":
        print(f"overlap={args.overlap}")
    if args.chunking == "sentences":
        print(f"max_sentences={args.max_sentences}")
    print(f"embedding_backend={getattr(embedder, '_backend_name', embedder.__class__.__name__)}")
    print(f"storage_backend={store._storage_backend}")
    print(f"collection={args.collection}")
    print(f"persist_dir={args.persist_dir}")
    print(f"stored_records={store.get_collection_size()}")
    return 0


def run_search_disease_db(args: argparse.Namespace) -> int:
    embedder = build_embedder(args.embedding_provider)
    store = create_store(
        args.collection,
        embedding_fn=embedder,
        persist_directory=args.persist_dir,
    )

    metadata_filter = None
    if args.filter_disease:
        metadata_filter = {"disease": args.filter_disease}

    if metadata_filter:
        results = store.search_with_filter(args.query, top_k=args.top_k, metadata_filter=metadata_filter)
    else:
        results = store.search(args.query, top_k=args.top_k)

    print("=== Disease Search Results ===")
    print(f"query={args.query}")
    print(f"storage_backend={store._storage_backend}")
    print(f"collection={args.collection}")
    print(f"persist_dir={args.persist_dir}")
    if metadata_filter:
        print(f"metadata_filter={metadata_filter}")
    print_search_results(results)
    return 0


def run_ask_disease_db(args: argparse.Namespace) -> int:
    embedder = build_embedder(args.embedding_provider)
    llm = build_llm(args.llm_provider)
    store = create_store(
        args.collection,
        embedding_fn=embedder,
        persist_directory=args.persist_dir,
    )
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm)

    print("=== Disease QA ===")
    print(f"question={args.question}")
    print(f"embedding_backend={getattr(embedder, '_backend_name', embedder.__class__.__name__)}")
    print(f"llm_backend={getattr(llm, '_backend_name', llm.__class__.__name__)}")
    print(f"storage_backend={store._storage_backend}")
    print(f"collection={args.collection}")
    print(f"persist_dir={args.persist_dir}")
    print("\nAnswer:")
    print(agent.answer(args.question, top_k=args.top_k))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilities for chunking, storing, and querying disease documents.")
    subparsers = parser.add_subparsers(dest="command")

    demo_parser = subparsers.add_parser("demo", help="Run the original sample-data demo.")
    demo_parser.add_argument("question", nargs="?", default=None)

    compare_parser = subparsers.add_parser("compare-chunking", help="Compare chunking strategies on one file.")
    compare_parser.add_argument("--file", required=True, help="Path to the file to analyze.")
    compare_parser.add_argument("--chunk-size", type=int, default=500)

    load_parser = subparsers.add_parser("load-disease-db", help="Chunk disease data and store it in ChromaDB or in-memory store.")
    load_parser.add_argument("--data-dir", default=DEFAULT_DISEASE_DIR)
    load_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    load_parser.add_argument("--persist-dir", default=os.getenv("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_DIR))
    load_parser.add_argument("--chunking", choices=["fixed", "sentences", "recursive"], default="recursive")
    load_parser.add_argument("--chunk-size", type=int, default=500)
    load_parser.add_argument("--overlap", type=int, default=50)
    load_parser.add_argument("--max-sentences", type=int, default=3)
    load_parser.add_argument("--embedding-provider", choices=["mock", "local", "openai"], default=os.getenv(EMBEDDING_PROVIDER_ENV, "mock"))
    load_parser.add_argument("--reset", action="store_true", help="Delete existing records in the collection before loading.")

    search_parser = subparsers.add_parser("search-disease-db", help="Query disease chunks from the stored database.")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    search_parser.add_argument("--persist-dir", default=os.getenv("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_DIR))
    search_parser.add_argument("--top-k", type=int, default=3)
    search_parser.add_argument("--embedding-provider", choices=["mock", "local", "openai"], default=os.getenv(EMBEDDING_PROVIDER_ENV, "mock"))
    search_parser.add_argument("--filter-disease", default=None, help="Optional metadata filter by disease name, e.g. coloboma.")

    ask_parser = subparsers.add_parser("ask-disease-db", help="Ask a question over the stored disease chunks using KnowledgeBaseAgent.")
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    ask_parser.add_argument("--persist-dir", default=os.getenv("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_DIR))
    ask_parser.add_argument("--top-k", type=int, default=3)
    ask_parser.add_argument("--embedding-provider", choices=["mock", "local", "openai"], default=os.getenv(EMBEDDING_PROVIDER_ENV, "mock"))
    ask_parser.add_argument("--llm-provider", choices=["mock", "openai", "gemini"], default=os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER))

    return parser


def main() -> int:
    load_dotenv(override=False)
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
        return run_sample_demo(question=question)

    if args.command == "demo":
        return run_sample_demo(question=args.question)
    if args.command == "compare-chunking":
        return run_compare_chunking(args)
    if args.command == "load-disease-db":
        return run_load_disease_db(args)
    if args.command == "search-disease-db":
        return run_search_disease_db(args)
    if args.command == "ask-disease-db":
        return run_ask_disease_db(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
