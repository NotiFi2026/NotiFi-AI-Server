"""Spring 백엔드 내부 API 클라이언트 (I1·I2·I3)."""
from datetime import datetime
from typing import Any, Optional

import httpx

from app.agent.schemas import GuardianMessage
from app.common.logging_config import logger
from app.config import settings

_HEADERS = {"X-Internal-Key": settings.spring_internal_key}
_BASE = f"{settings.spring_base_url}/internal/v1"


async def send_sensing_event(payload: dict[str, Any]) -> dict[str, Any]:
    """I1: 감지 이벤트 + 위험도 적재. escalation_id 포함 응답 반환."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_BASE}/sensing-events",
            json=payload,
            headers=_HEADERS,
            timeout=10.0,
        )
        resp.raise_for_status()
    data = resp.json()["data"]
    logger.info(
        "감지 이벤트 적재 완료",
        extra={
            "action": "sensing_event_saved",
            "sensing_event_id": data.get("sensing_event_id"),
            "escalation_triggered": data.get("escalation_triggered"),
        },
    )
    return data


async def record_escalation_step(
    escalation_id: int,
    step_type: str,
    step_order: int,
    status: str,
    executed_at: datetime,
    responded_at: Optional[datetime] = None,
    response_detail: Optional[dict[str, Any]] = None,
    guardian_message: Optional[GuardianMessage] = None,
) -> dict[str, Any]:
    """I2: 에스컬레이션 단계 진행 기록. GUARDIAN_NOTIFY 단계면 guardian_message 포함.

    응답 data(기록된 step + escalation_status)를 반환한다 —
    에이전트가 EMERGENCY_CALL 진행 전 해소 여부를 판단하는 데 쓴다.
    """
    body: dict[str, Any] = {
        "step_type": step_type,
        "step_order": step_order,
        "status": status,
        "executed_at": executed_at.isoformat(),
    }
    if responded_at:
        body["responded_at"] = responded_at.isoformat()
    if response_detail:
        body["response_detail"] = response_detail
    if guardian_message:
        body["guardian_message"] = guardian_message.model_dump()

    logger.info(
        "I2 요청",
        extra={
            "action": "spring_i2_requested",
            "escalation_id": escalation_id,
            "step_type": step_type,
            "status": status,
        },
    )
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_BASE}/escalations/{escalation_id}/steps",
            json=body,
            headers=_HEADERS,
            timeout=10.0,
        )
        resp.raise_for_status()
    data = resp.json()["data"]
    logger.info(
        "에스컬레이션 단계 기록 완료",
        extra={
            "action": "escalation_step_recorded",
            "escalation_id": escalation_id,
            "step_type": step_type,
            "status": data.get("status", status),
            "escalation_status": data.get("escalation_status"),
        },
    )
    return data


async def save_daily_report(
    care_target_id: int,
    report_date: str,
    summary_text: str,
    metrics: dict[str, Any],
) -> None:
    """I3: 일일 리포트 적재."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_BASE}/reports",
            json={
                "care_target_id": care_target_id,
                "report_date": report_date,
                "summary_text": summary_text,
                "metrics": metrics,
            },
            headers=_HEADERS,
            timeout=10.0,
        )
        resp.raise_for_status()
    logger.info(
        "일일 리포트 적재 완료",
        extra={
            "action": "spring_i3_completed",
            "care_target_id": care_target_id,
            "report_date": report_date,
        },
    )
