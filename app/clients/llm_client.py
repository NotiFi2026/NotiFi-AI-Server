import json
from typing import Any

from openai import AsyncOpenAI

from app.config import settings

_client = AsyncOpenAI(api_key=settings.openai_api_key)


async def complete_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """LLM을 호출하고 JSON 응답을 파싱해 반환한다."""
    response = await _client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=512,
    )
    return json.loads(response.choices[0].message.content)


async def complete_text(system_prompt: str, user_prompt: str) -> str:
    """LLM을 호출하고 텍스트 응답을 반환한다."""
    response = await _client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()
