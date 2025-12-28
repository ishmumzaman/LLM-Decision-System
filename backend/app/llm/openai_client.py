from __future__ import annotations

from openai import AsyncOpenAI

from app.settings import Settings


def build_openai_client(settings: Settings) -> AsyncOpenAI:
    if not settings.openai_api_key:
        raise ValueError("Missing OPENAI_API_KEY")
    return AsyncOpenAI(api_key=settings.openai_api_key)
