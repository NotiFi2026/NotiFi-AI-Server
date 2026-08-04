"""보호자 알림 문구 및 일일 리포트 문장을 생성한다."""
from __future__ import annotations

import json
from typing import Any, Optional

from app.agent.schemas import (
    DailyReportInput,
    DailyReportMetrics,
    DailyReportOutput,
    DailyReportSection,
    EventType,
    GuardianMessage,
    ReportTag,
    RiskLevel,
    VoiceResponseResult,
)
from app.clients.llm_client import complete_json
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
당신은 노인 생활 데이터를 보호자가 이해하기 쉽게 태그별로 요약하는 리포트 생성기입니다.

규칙:
1. 입력 지표에 없는 내용을 생성하지 않는다.
2. 질병명, 진단명, 치료 지시를 생성하지 않는다.
3. 각 태그의 위험도는 이미 결정되어 입력에 주어진다 — 임의로 바꾸지 않고 그 톤에 맞춰 서술한다.
4. body는 상황 설명만 한다. 보호자가 할 행동은 body에 넣지 말고 recommended_action에 따로 쓴다.
5. risk_level이 safe면 recommended_action은 null로 둔다. warning/danger면 recommended_action에 구체적인 확인 행동을 한 문장으로 쓴다.
6. "악화", "질환", "치료 필요" 같은 단정 표현을 피한다.
7. 각 섹션의 title은 10자 내외, body는 1~2문장으로 작성한다.
8. safe_class_counts가 주어지면, 그중 눈에 띄는 활동 1~2개를 골라 실제 횟수를 숫자로 포함해 자연스러운 문장으로 body에 넣는다 (예: "걷기 14회"). absence(부재)는 언급하지 않는다.

응답 형식 (JSON):
{
  "risk_event": {"title": "...", "body": "...", "recommended_action": "..." 또는 null}
}"""


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


def _report_risk_levels(m: DailyReportMetrics) -> dict[ReportTag, RiskLevel]:
    """지표 임계값으로 태그별 위험도를 결정한다 — LLM이 아니라 코드가 판단한다.

    TODO: 모델 가중치 미반영 상태의 임시 규칙. 정확도 개선된 가중치가 들어오면
    (팀원 전달 예정 2026-08-09) 모델 출력 기준으로 교체할 것.
    """

    def _level(danger: bool, warning: bool) -> RiskLevel:
        if danger:
            return RiskLevel.DANGER
        if warning:
            return RiskLevel.WARNING
        return RiskLevel.SAFE

    return {
        ReportTag.RISK_EVENT: _level(
            m.danger_event_count >= 1, m.warning_event_count >= 1
        ),
    }


def _build_report_user_prompt(
    report_input: DailyReportInput, risk_levels: dict[ReportTag, RiskLevel]
) -> str:
    m = report_input.metrics
    lines = [
        f"날짜: {report_input.report_date}",
        f"[risk_event] 위험도: {risk_levels[ReportTag.RISK_EVENT].value} | "
        f"주의 이벤트: {m.warning_event_count}건, 위험 이벤트: {m.danger_event_count}건",
    ]

    safe_counts = {
        k: v for k, v in m.safe_class_counts.items() if k != "absence" and v > 0
    }
    if safe_counts:
        counts_str = ", ".join(f"{k} {v}회" for k, v in safe_counts.items())
        lines.append(f"[risk_event] 오늘의 정상 활동: {counts_str}")

    return "\n".join(lines)


async def generate_daily_report_summary(report_input: DailyReportInput) -> DailyReportOutput:
    risk_levels = _report_risk_levels(report_input.metrics)
    user_prompt = _build_report_user_prompt(report_input, risk_levels)

    logger.info(
        "일일 리포트 생성 시작",
        extra={
            "action": "daily_report_generating",
            "care_target_id": report_input.care_target_id,
            "report_date": report_input.report_date,
        },
    )

    raw = await complete_json(_SYSTEM_PROMPT_REPORT, user_prompt)
    sections = [
        DailyReportSection(
            tag=tag,
            risk_level=risk_levels[tag],
            title=raw[tag.value]["title"],
            body=raw[tag.value]["body"],
            recommended_action=raw[tag.value].get("recommended_action"),
        )
        for tag in ReportTag
    ]

    logger.info(
        "일일 리포트 생성 완료",
        extra={"action": "daily_report_generated"},
    )

    from datetime import datetime, timezone
    return DailyReportOutput(
        care_target_id=report_input.care_target_id,
        report_date=report_input.report_date,
        sections=sections,
        metrics=report_input.metrics,
        generated_at=datetime.now(timezone.utc),
    )
