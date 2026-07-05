"""보호자 알림 문구 및 일일 리포트 문장을 생성한다."""
from __future__ import annotations

import json
from typing import Any, Optional

from app.agent.schemas import (
    DailyReportInput,
    DailyReportOutput,
    EventType,
    GuardianMessage,
    RiskLevel,
    VoiceResponseResult,
)
from app.clients.llm_client import complete_json, complete_text
from app.common.logging_config import logger

_FORBIDDEN = [
    "심정지로 판단",
    "의식이 없는 상태",
    "호흡부전",
    "생명이 위험",
    "질병이 발생",
    "응급처치가 필요",
]

_SYSTEM_PROMPT_GUARDIAN = """\
당신은 노인 안전 모니터링 서비스의 보호자 알림 문구 생성기입니다.

규칙:
1. 감지된 사실만 말한다.
2. 의학적 진단, 질병명, 치료 지시를 생성하지 않는다.
3. 입력에 없는 정보를 만들지 않는다.
4. "의심", "감지", "확인이 필요" 같은 표현을 사용한다.
5. 과도한 공포를 유발하는 표현을 피한다.
6. 금지 표현: "심정지로 판단됩니다", "의식이 없는 상태", "호흡부전", "생명이 위험", "질병이 발생", "응급처치가 필요"

응답 형식 (JSON):
{
  "title": "한 문장 제목",
  "body": "감지된 상황을 구체적으로 설명하는 본문",
  "recommendation": "보호자가 취할 수 있는 행동 제안"
}"""

_SYSTEM_PROMPT_REPORT = """\
당신은 노인 생활 데이터를 보호자가 이해하기 쉽게 요약하는 리포트 생성기입니다.

규칙:
1. 입력 지표에 없는 내용을 생성하지 않는다.
2. 질병명, 진단명, 치료 지시를 생성하지 않는다.
3. 변화가 있는 항목 중심으로 요약한다.
4. 보호자가 할 수 있는 확인 행동을 제안한다.
5. "악화", "질환", "치료 필요" 같은 단정 표현을 피한다.

하루 생활 패턴 변화를 2~4문장으로 요약한다."""


def _event_type_label(event_type: EventType) -> str:
    return {
        EventType.FALL: "낙상",
        EventType.INACTIVITY: "장시간 무활동",
        EventType.RESPIRATION_ABNORMAL: "호흡 이상",
        EventType.ANOMALY: "이상 행동",
        EventType.SENSOR_ERROR: "센서 오류",
        EventType.NORMAL: "정상",
    }.get(event_type, event_type.value)


def _build_guardian_user_prompt(
    event_type: EventType,
    label: str,
    risk_level: RiskLevel,
    confidence: float,
    response_result: Optional[VoiceResponseResult],
    context_features: Optional[dict[str, Any]],
) -> str:
    parts = [
        f"이벤트 유형: {_event_type_label(event_type)}",
        f"위험도: {risk_level.value}",
        f"모델 신뢰도: {confidence:.0%}",
    ]

    if context_features:
        if "no_movement_seconds_after_event" in context_features:
            parts.append(f"이벤트 후 무움직임 지속: {context_features['no_movement_seconds_after_event']}초")
        if "post_event_movement_level" in context_features:
            level = context_features["post_event_movement_level"]
            parts.append(f"이벤트 후 움직임 수준: {'매우 낮음' if level < 0.1 else '낮음' if level < 0.3 else '보통'}")
        if "breathing_signal_strength" in context_features:
            strength = context_features["breathing_signal_strength"]
            parts.append(f"호흡 신호 강도: {'약함' if strength < 0.2 else '보통'}")

    if response_result:
        result_label = {
            VoiceResponseResult.USER_OK: "사용자가 음성 확인에 괜찮다고 응답함",
            VoiceResponseResult.USER_NEEDS_HELP: "사용자가 음성 확인에 도움을 요청함",
            VoiceResponseResult.NO_RESPONSE: "음성 확인에 응답 없음",
        }.get(response_result, "")
        if result_label:
            parts.append(f"음성 확인 결과: {result_label}")

    return "\n".join(parts)


def _check_forbidden(text: str) -> bool:
    """금지 표현이 포함되어 있으면 True를 반환한다."""
    return any(f in text for f in _FORBIDDEN)


async def generate_guardian_message(
    event_type: EventType,
    label: str,
    risk_level: RiskLevel,
    confidence: float,
    response_result: Optional[VoiceResponseResult] = None,
    context_features: Optional[dict[str, Any]] = None,
) -> GuardianMessage:
    user_prompt = _build_guardian_user_prompt(
        event_type, label, risk_level, confidence, response_result, context_features
    )

    logger.info(
        "보호자 메시지 생성 시작",
        extra={
            "action": "guardian_message_generating",
            "event_type": event_type.value,
            "risk_level": risk_level.value,
            "response_result": response_result.value if response_result else None,
        },
    )

    raw = await complete_json(_SYSTEM_PROMPT_GUARDIAN, user_prompt)
    message = GuardianMessage(**raw)

    if any(_check_forbidden(v) for v in [message.title, message.body, message.recommendation]):
        logger.warning(
            "금지 표현 감지 — 재생성 시도",
            extra={"action": "guardian_message_forbidden_detected"},
        )
        raw = await complete_json(
            _SYSTEM_PROMPT_GUARDIAN + "\n\n※ 이전 응답에 금지 표현이 포함되어 있었습니다. 반드시 피해주세요.",
            user_prompt,
        )
        message = GuardianMessage(**raw)

    logger.info(
        "보호자 메시지 생성 완료",
        extra={"action": "guardian_message_generated"},
    )
    return message


async def generate_daily_report_summary(report_input: DailyReportInput) -> DailyReportOutput:
    m = report_input.metrics
    sign = "+" if m.activity_change_percent >= 0 else ""
    user_prompt = "\n".join([
        f"날짜: {report_input.report_date}",
        f"활동 수준: {m.activity_level:.0%} (전일 대비 {sign}{m.activity_change_percent:.1f}%)",
        f"평균 호흡 수: 분당 {m.avg_breathing_rate:.1f}회",
        f"전체 무활동 시간: {m.total_inactivity_minutes}분",
        f"최장 연속 무활동: {m.longest_inactive_minutes}분",
        f"주의 이벤트: {m.warning_event_count}건",
        f"위험 이벤트: {m.danger_event_count}건",
        f"호흡 이상 이벤트: {m.respiration_abnormal_count}건",
    ])

    logger.info(
        "일일 리포트 생성 시작",
        extra={
            "action": "daily_report_generating",
            "care_target_id": report_input.care_target_id,
            "report_date": report_input.report_date,
        },
    )

    summary_text = await complete_text(_SYSTEM_PROMPT_REPORT, user_prompt)

    logger.info(
        "일일 리포트 생성 완료",
        extra={"action": "daily_report_generated"},
    )

    from datetime import datetime, timezone
    return DailyReportOutput(
        care_target_id=report_input.care_target_id,
        report_date=report_input.report_date,
        summary_text=summary_text,
        metrics=report_input.metrics,
        generated_at=datetime.now(timezone.utc),
    )
