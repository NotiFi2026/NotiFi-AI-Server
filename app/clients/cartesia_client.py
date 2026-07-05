"""Cartesia TTS 클라이언트 — 한국어/일본어 음성 출력."""
import asyncio
import io
import wave

import numpy as np
import sounddevice as sd

from app.common.logging_config import logger
from app.config import settings

INITIAL_PROMPT = "괜찮으세요? 도움이 필요하시면 도와줘라고 말씀해주세요."
RETRY_PROMPT = "응답이 잘 들리지 않았습니다. 괜찮으시면 괜찮아, 도움이 필요하시면 도와줘라고 말씀해주세요."

_INITIAL_PROMPT_MAP = {
    "ko": INITIAL_PROMPT,
    "ja": "大丈夫ですか？助けが必要な場合は「助けて」とおっしゃってください。",
}
_RETRY_PROMPT_MAP = {
    "ko": RETRY_PROMPT,
    "ja": "お返事が聞こえませんでした。大丈夫な場合は「大丈夫」、助けが必要な場合は「助けて」とおっしゃってください。",
}

_CARTESIA_URL = "https://api.cartesia.ai/tts/bytes"
_HEADERS = {
    "X-API-Key": settings.cartesia_api_key,
    "Cartesia-Version": "2024-06-10",
    "Content-Type": "application/json",
}
_VOICE_MAP = {
    "ko": settings.cartesia_voice_id_ko,
    "ja": settings.cartesia_voice_id_ja,
}


async def speak(text: str, language: str = "ko") -> None:
    """Cartesia TTS로 음성 생성 후 스피커 재생."""
    import httpx

    voice_id = _VOICE_MAP.get(language, settings.cartesia_voice_id_ko)

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            _CARTESIA_URL,
            json={
                "model_id": "sonic-2",
                "transcript": text,
                "voice": {"mode": "id", "id": voice_id},
                "output_format": {
                    "container": "wav",
                    "encoding": "pcm_s16le",
                    "sample_rate": 22050,
                },
                "language": language,
            },
            headers=_HEADERS,
            timeout=15.0,
        )
        resp.raise_for_status()
        wav_bytes = resp.content

    await asyncio.to_thread(_play_wav, wav_bytes)
    logger.info(
        "TTS 재생 완료",
        extra={"action": "tts_speak", "language": language, "text_length": len(text)},
    )


def _play_wav(wav_bytes: bytes) -> None:
    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as wf:
            sample_rate = wf.getframerate()
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    sd.play(audio, sample_rate)
    sd.wait()
