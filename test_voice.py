"""TTS + STT 단독 테스트 — python test_voice.py

1. Cartesia TTS로 질문 음성 재생
2. 마이크로 8초 녹음
3. Whisper로 전사
4. 키워드 분류
"""
import asyncio

from app.agent import response_classifier
from app.clients import cartesia_client, whisper_client


async def main() -> None:
    print("▶ TTS 재생 중...")
    await cartesia_client.speak(cartesia_client.INITIAL_PROMPT, language="ko")

    print("🎙  녹음 중 (8초)... 말씀해주세요")
    stt_text = await whisper_client.listen(timeout_seconds=8.0, language="ko")
    print(f"📝 STT 결과: {stt_text!r}")

    result = await response_classifier.classify(stt_text)
    print(f"✅ 분류 결과: {result.value}")


if __name__ == "__main__":
    asyncio.run(main())
