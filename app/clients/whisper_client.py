"""Whisper STT 클라이언트 — 마이크 녹음 후 OpenAI Whisper 전사."""
import asyncio
import io
import wave

import numpy as np
import sounddevice as sd
from openai import AsyncOpenAI

from app.common.logging_config import logger
from app.config import settings

_SAMPLE_RATE = 16000
_CHANNELS = 1

_client = AsyncOpenAI(api_key=settings.openai_api_key)


async def listen(timeout_seconds: float = 8.0, language: str | None = None) -> str | None:
    """마이크 녹음 → Whisper 전사 → 텍스트 반환. 무응답이면 None.

    language=None이면 Whisper가 자동 감지한다.
    """
    logger.info(
        "STT 녹음 시작",
        extra={"action": "stt_listen", "timeout": timeout_seconds, "language": language or "auto"},
    )

    audio = await asyncio.to_thread(_record, timeout_seconds)
    wav_bytes = _to_wav_bytes(audio)

    buf = io.BytesIO(wav_bytes)
    buf.name = "recording.wav"

    kwargs = {"model": "whisper-1", "file": buf}
    if language:
        kwargs["language"] = language

    result = await _client.audio.transcriptions.create(**kwargs)
    text = result.text.strip()

    logger.info(
        "STT 완료",
        extra={"action": "stt_completed", "text": text or "(무응답)"},
    )
    return text if text else None


def _record(duration: float) -> np.ndarray:
    audio = sd.rec(
        int(duration * _SAMPLE_RATE),
        samplerate=_SAMPLE_RATE,
        channels=_CHANNELS,
        dtype="int16",
    )
    sd.wait()
    return audio


def _to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()
