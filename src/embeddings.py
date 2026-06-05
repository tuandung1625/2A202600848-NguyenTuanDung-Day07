from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"


class MockEmbedder:
    """Deterministic embedding backend used by tests and default classroom runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        model_path = self._resolve_model_path(model_name)
        self._backend_name = str(model_path)
        self.model = SentenceTransformer(str(model_path), local_files_only=True)

    def __call__(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]

    @staticmethod
    def _resolve_model_path(model_name: str) -> Path | str:
        if Path(model_name).exists():
            return model_name

        hf_home = os.getenv("HF_HOME")
        candidate_roots = []
        if hf_home:
            candidate_roots.append(Path(hf_home))
        candidate_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

        model_dir_name = f"models--sentence-transformers--{model_name}"
        for root in candidate_roots:
            model_root = root / model_dir_name
            refs_main = model_root / "refs" / "main"
            if refs_main.exists():
                revision = refs_main.read_text(encoding="utf-8").strip()
                snapshot_dir = model_root / "snapshots" / revision
                if snapshot_dir.exists():
                    return snapshot_dir

            snapshots_dir = model_root / "snapshots"
            if snapshots_dir.exists():
                snapshot_dirs = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()])
                if snapshot_dirs:
                    return snapshot_dirs[-1]

        return model_name


class OpenAIEmbedder:
    """OpenAI embeddings API-backed embedder."""

    def __init__(self, model_name: str = OPENAI_EMBEDDING_MODEL) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self._backend_name = model_name
        self.client = OpenAI()

    def __call__(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return [float(value) for value in response.data[0].embedding]


_mock_embed = MockEmbedder()
