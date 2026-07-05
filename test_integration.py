"""실제 Spring 서버와 end-to-end 통합 테스트 — python test_integration.py

TTS/STT만 mock (하드웨어 없음), Spring I1/I2/I3은 실제 호출.
실행 전 Spring + DB 가동 필요: docker-compose up -d postgres redis && gradlew bootRun
"""
import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from app.agent import escalation_agent
from app.agent.schemas import EventType, ModelResult, RiskLevel


def _model(risk_level: RiskLevel, care_target_id: int = 1) -> ModelResult:
    return ModelResult(
        care_target_id=care_target_id,
        device_id=None,
        event_type=EventType.FALL,
        label="fall_simulated",
        risk_level=risk_level,
        confidence=0.91,
        risk_score=91,
        model_version="notifi-csi-v1",
        detected_at=datetime.now(timezone.utc),
        context_features={
            "no_movement_seconds_after_event": 20,
            "post_event_movement_level": 0.03,
        },
    )


async def main() -> None:
    # ── DANGER + 음성 무응답 ──────────────────────────────────
    print("=" * 60)
    print("DANGER + 음성 무응답 (실제 Spring 연동)")
    print("=" * 60)

    with patch("app.clients.whisper_client.listen", AsyncMock(return_value=None)), \
         patch("app.clients.cartesia_client.speak", AsyncMock()):
        final = await escalation_agent.run(_model(RiskLevel.DANGER))

    print(json.dumps({
        "sensing_event_id": final["sensing_event_id"],
        "escalation_id": final["escalation_id"],
        "voice_response_result": final["voice_response_result"],
        "notification_level": final["notification_level"],
        "escalation_continued": final["escalation_continued"],
        "guardian_message": final["guardian_message"],
    }, ensure_ascii=False, indent=2))

    # ── SAFE ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SAFE (I1만 호출, 이후 종료)")
    print("=" * 60)

    final_safe = await escalation_agent.run(_model(RiskLevel.SAFE))
    print(json.dumps({
        "sensing_event_id": final_safe["sensing_event_id"],
        "escalation_triggered": final_safe["escalation_triggered"],
        "response_policy": final_safe["response_policy"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
