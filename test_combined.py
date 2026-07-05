"""LangGraph 에스컬레이션 + 일일 리포트 JSON 출력 — python test_combined.py

[케이스 1] DANGER + 음성 무응답
  → VOICE_CHECK/EXECUTED, VOICE_CHECK/NO_RESPONSE, GUARDIAN_NOTIFY, EMERGENCY_CALL

[케이스 2] DANGER + 사용자 OK
  → VOICE_CHECK/EXECUTED, VOICE_CHECK/RESPONDED, GUARDIAN_NOTIFY(INFO, 종료)

[케이스 3] SAFE
  → I1만 호출, 이후 종료

[케이스 4] 일일 리포트 생성 + I3 페이로드
"""
import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent import escalation_agent
from app.agent.message_generator import generate_daily_report_summary
from app.agent.schemas import (
    DailyReportInput,
    DailyReportMetrics,
    EventType,
    ModelResult,
    RiskLevel,
)


def _i1_response(triggered: bool = True, escalation_id: int = 31) -> dict:
    return {
        "sensing_event_id": 882,
        "risk_assessment_id": 451,
        "escalation_triggered": triggered,
        "escalation_id": escalation_id if triggered else None,
    }


def _model(
    risk_level: RiskLevel = RiskLevel.DANGER,
    event_type: EventType = EventType.FALL,
) -> ModelResult:
    return ModelResult(
        care_target_id=45,
        device_id=2,
        event_type=event_type,
        label="fall_simulated",
        risk_level=risk_level,
        confidence=0.91,
        risk_score=91,
        model_version="notifi-csi-v1",
        detected_at=datetime.now(timezone.utc),
        context_features={
            "no_movement_seconds_after_event": 20,
            "post_event_movement_level": 0.03,
            "breathing_signal_strength": 0.12,
        },
    )


def _print_i2_calls(mock_record: AsyncMock) -> None:
    print(f"I2 호출 횟수: {mock_record.call_count}회")
    for i, c in enumerate(mock_record.call_args_list, 1):
        kw = c.kwargs
        entry = {
            "호출": i,
            "step_type": kw.get("step_type"),
            "step_order": kw.get("step_order"),
            "status": kw.get("status"),
            "response_detail": kw.get("response_detail"),
            "guardian_message": kw["guardian_message"].model_dump()
            if kw.get("guardian_message") else None,
        }
        print(json.dumps(entry, ensure_ascii=False, indent=2))


async def main() -> None:
    # ── 케이스 1: DANGER + 음성 무응답 ──────────────────────
    print("=" * 60)
    print("케이스 1: DANGER + 음성 무응답")
    print("=" * 60)

    mock_send = AsyncMock(return_value=_i1_response(triggered=True))
    mock_record = AsyncMock()

    with patch("app.clients.spring_client.send_sensing_event", mock_send), \
         patch("app.clients.spring_client.record_escalation_step", mock_record), \
         patch("app.clients.whisper_client.listen", AsyncMock(return_value=None)), \
         patch("app.clients.cartesia_client.speak", AsyncMock()):
        final = await escalation_agent.run(_model(RiskLevel.DANGER))

    print(f"최종 response_policy: {final['response_policy']}")
    print(f"최종 voice_response_result: {final['voice_response_result']}")
    print(f"최종 notification_level: {final['notification_level']}")
    print(f"최종 escalation_continued: {final['escalation_continued']}")
    _print_i2_calls(mock_record)

    # ── 케이스 2: DANGER + 사용자 OK ────────────────────────
    print("\n" + "=" * 60)
    print("케이스 2: DANGER + 사용자 OK")
    print("=" * 60)

    mock_send2 = AsyncMock(return_value=_i1_response(triggered=True, escalation_id=32))
    mock_record2 = AsyncMock()

    with patch("app.clients.spring_client.send_sensing_event", mock_send2), \
         patch("app.clients.spring_client.record_escalation_step", mock_record2), \
         patch("app.clients.whisper_client.listen", AsyncMock(return_value="괜찮아요")), \
         patch("app.clients.cartesia_client.speak", AsyncMock()):
        final2 = await escalation_agent.run(_model(RiskLevel.DANGER))

    print(f"최종 voice_response_result: {final2['voice_response_result']}")
    print(f"최종 notification_level: {final2['notification_level']}")
    print(f"최종 escalation_continued: {final2['escalation_continued']}")
    _print_i2_calls(mock_record2)

    # ── 케이스 3: SAFE ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("케이스 3: SAFE (I1만 호출, 이후 종료)")
    print("=" * 60)

    mock_send3 = AsyncMock(return_value=_i1_response(triggered=False))
    mock_record3 = AsyncMock()

    with patch("app.clients.spring_client.send_sensing_event", mock_send3), \
         patch("app.clients.spring_client.record_escalation_step", mock_record3):
        final3 = await escalation_agent.run(_model(RiskLevel.SAFE))

    print(f"I1 호출 횟수: {mock_send3.call_count}회")
    print(f"I2 호출 횟수: {mock_record3.call_count}회 (0이어야 함)")
    print(f"response_policy: {final3['response_policy']}")

    # ── 케이스 4: 일일 리포트 ───────────────────────────────
    print("\n" + "=" * 60)
    print("케이스 4: 일일 리포트 생성 + I3 페이로드")
    print("=" * 60)

    report_input = DailyReportInput(
        care_target_id=45,
        report_date="2026-07-02",
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
    i3_payload = {
        "care_target_id": output.care_target_id,
        "report_date": output.report_date,
        "summary_text": output.summary_text,
        "metrics": output.metrics.model_dump(),
        "generated_at": output.generated_at.isoformat(),
    }
    print(json.dumps(i3_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
