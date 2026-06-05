from __future__ import annotations

import json
import os
from urllib import parse, request

OPENAI_CHAT_MODEL = "gpt-4o-mini"
GEMINI_CHAT_MODEL = "gemini-2.5-flash"


class MockLLM:
    """Simple fallback LLM used for offline demos."""

    def __init__(self) -> None:
        self._backend_name = "mock llm fallback"

    def __call__(self, prompt: str) -> str:
        preview = prompt[:300].replace("\n", " ")
        return f"[MOCK LLM] Answer generated from prompt preview: {preview}..."


class OpenAIChatLLM:
    """OpenAI chat-completions backed callable used by KnowledgeBaseAgent."""

    def __init__(self, model_name: str = OPENAI_CHAT_MODEL) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self._backend_name = model_name
        self.client = OpenAI()

    def __call__(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
        )
        return response.output_text


class GeminiChatLLM:
    """Gemini REST-backed callable used by KnowledgeBaseAgent."""

    def __init__(self, model_name: str = GEMINI_CHAT_MODEL, api_key: str | None = None) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY for GeminiChatLLM.")
        self._backend_name = model_name

    def __call__(self, prompt: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={parse.quote(self.api_key)}"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                    ]
                }
            ]
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))

        candidates = data.get("candidates", [])
        if not candidates:
            return "[GEMINI] No response candidates returned."

        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [part.get("text", "") for part in parts if part.get("text")]
        return "".join(text_parts).strip() or "[GEMINI] Empty text response."
