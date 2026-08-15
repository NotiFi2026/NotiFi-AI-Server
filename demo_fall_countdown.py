"""촬영용 — 혼자 찍을 때 쓰는 낙상 트리거. 5초 조용히 기다렸다가 이벤트를 보낸다.

실행하고 나서 카메라 앞으로 이동해 5초 안에 쓰러지면 된다.
그 다음은 test_voice_full.py와 동일 — 실제 TTS/STT + Spring 연동.

사용법:
    python demo_fall_countdown.py        # 한국어 (기본)
    python demo_fall_countdown.py ja     # 일본어
"""
import asyncio
import json
import sys
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")

from app.agent import escalation_agent
from app.agent.schemas import EventType, ModelResult, RiskLevel
from app.clients import cartesia_client

COUNTDOWN_SECONDS = 5.0
START_WORD_MAP = {"ko": "시작한다", "ja": "始めます"}


async def main(language: str = "ko") -> None:
    start_word = START_WORD_MAP.get(language, START_WORD_MAP["ko"])
    print(f"▶ {start_word} — {COUNTDOWN_SECONDS:.0f}초 후 낙상 이벤트가 전송된다")
    await cartesia_client.speak(start_word, language=language)
    await asyncio.sleep(COUNTDOWN_SECONDS)

    model_result = ModelResult(
        care_target_id=1,
        device_id=None,
        event_type=EventType.FALL,
        label="fall_simulated",
        risk_level=RiskLevel.DANGER,
        confidence=0.91,
        risk_score=91,
        model_version="notifi-csi-v1",
        detected_at=datetime.now(timezone.utc),
        context_features={"no_movement_seconds_after_event": 20, "post_event_movement_level": 0.03},
        language=language,
    )

    print("▶ 낙상 이벤트 전송 (실제 TTS/STT + Spring 연동)")
    print("=" * 60)

    final = await escalation_agent.run(model_result)

    print("\n" + "=" * 60)
    print("최종 결과 JSON:")
    print(json.dumps({
        "sensing_event_id":      final["sensing_event_id"],
        "escalation_id":         final["escalation_id"],
        "language":              final["language"],
        "response_policy":       final["response_policy"],
        "stt_text":              final["stt_text"],
        "voice_response_result": final["voice_response_result"],
        "notification_level":    final["notification_level"],
        "escalation_continued":  final["escalation_continued"],
        "guardian_message":      final["guardian_message"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    lang = sys.argv[1] if len(sys.argv) > 1 else "ko"
    asyncio.run(main(lang))
