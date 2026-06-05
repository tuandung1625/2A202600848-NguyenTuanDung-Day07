from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self._store = store
        self._llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        if not question:
            return "Please provide a valid question."

        # 1. Retrieve relevant chunks
        results = self._store.search(question, top_k=top_k)

        if not results:
            return "No relevant information found in knowledge base."

        # 2. Build context
        context_parts = []
        for i, r in enumerate(results):
            content = r.get("content", "")
            context_parts.append(f"[Chunk {i+1}]\n{content}")

        context = "\n\n".join(context_parts)

        # 3. Build prompt
        prompt = f"""
You are a helpful AI assistant. Answer the question based only on the provided context.

Context:
{context}

Question:
{question}

Instructions:
- Use only the information from the context above.
- If the answer is not in the context, say "I don't know".
- Be concise and clear.

Answer:
""".strip()

        # 4. Call LLM
        response = self._llm_fn(prompt)

        return response
