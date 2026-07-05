"""실제 TTS/STT + Spring 연동 full 흐름

사용법:
    python test_voice_full.py        # 한국어 (기본)
    python test_voice_full.py ko     # 한국어
    python test_voice_full.py ja     # 일본어
"""
import asyncio
import json
import sys
from datetime import datetime, timezone

from app.agent import escalation_agent
from app.agent.schemas import EventType, ModelResult, RiskLevel


async def main(language: str = "ko") -> None:
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

    print(f"▶ 에스컬레이션 시작 (언어={language}, 실제 TTS/STT + Spring 연동)")
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
