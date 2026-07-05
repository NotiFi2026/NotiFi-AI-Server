"""자연어 알림 생성 테스트 — python test_message_generator.py"""
import asyncio
import json

from app.agent.message_generator import generate_daily_report_summary, generate_guardian_message
from app.agent.schemas import (
    DailyReportInput,
    DailyReportMetrics,
    EventType,
    RiskLevel,
    VoiceResponseResult,
)


async def main() -> None:
    print("=" * 60)
    print("케이스 1: danger + NO_RESPONSE (낙상 / 음성 무응답)")
    print("=" * 60)
    msg = await generate_guardian_message(
        event_type=EventType.FALL,
        label="fall_simulated",
        risk_level=RiskLevel.DANGER,
        confidence=0.91,
        response_result=VoiceResponseResult.NO_RESPONSE,
        context_features={
            "room": "침실",
            "no_movement_seconds_after_event": 20,
            "post_event_movement_level": 0.03,
            "breathing_signal_strength": 0.12,
        },
    )
    print(json.dumps(msg.model_dump(), ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("케이스 2: danger + USER_OK (낙상 / 괜찮다고 응답)")
    print("=" * 60)
    msg2 = await generate_guardian_message(
        event_type=EventType.FALL,
        label="fall_simulated",
        risk_level=RiskLevel.DANGER,
        confidence=0.88,
        response_result=VoiceResponseResult.USER_OK,
        context_features={"room": "거실"},
    )
    print(json.dumps(msg2.model_dump(), ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("케이스 3: warning (장시간 무활동)")
    print("=" * 60)
    msg3 = await generate_guardian_message(
        event_type=EventType.INACTIVITY,
        label="long_inactivity",
        risk_level=RiskLevel.WARNING,
        confidence=0.79,
        response_result=None,
        context_features={"room": "침실", "no_movement_seconds_after_event": 3600},
    )
    print(json.dumps(msg3.model_dump(), ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("케이스 4: 일일 리포트")
    print("=" * 60)
    report_input = DailyReportInput(
        care_target_id=45,
        report_date="2026-06-23",
        metrics=DailyReportMetrics(
            activity_level=0.55,
            activity_change_percent=-8.3,
            total_inactivity_minutes=124,
            longest_inactive_minutes=47,
            warning_event_count=1,
            danger_event_count=0,
            respiration_abnormal_count=2,
            avg_breathing_rate=16.2,
        ),
    )
    output = await generate_daily_report_summary(report_input)
    print(json.dumps(output.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
