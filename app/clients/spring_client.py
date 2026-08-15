"""Spring 백엔드 내부 API 클라이언트 (I1·I2·I3·I4·I5·I6)."""
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


async def get_daily_metrics(care_target_id: int, report_date: str) -> dict[str, Any]:
    """I6: 리포트 생성용 하루치 집계 조회.

    이벤트 카운트의 단일 출처는 Spring DB다 — 에이전트 서버는 저장소가 없으므로
    하루치를 자체 유지할 수 없고, 여기서 읽은 값과 Spring이 저장한 이벤트가 어긋나지 않는다.
    하루 경계는 Spring이 Asia/Seoul 기준으로 자른다.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_BASE}/care-targets/{care_target_id}/daily-metrics",
            params={"date": report_date},
            headers=_HEADERS,
            timeout=10.0,
        )
        resp.raise_for_status()
    data = resp.json()["data"]
    logger.info(
        "일일 집계 조회 완료",
        extra={
            "action": "spring_i6_completed",
            "care_target_id": care_target_id,
            "report_date": report_date,
            "warning_event_count": data.get("warning_event_count"),
            "danger_event_count": data.get("danger_event_count"),
        },
    )
    return data


async def save_daily_report(
    care_target_id: int,
    report_date: str,
    sections: list[dict[str, Any]],
    metrics: dict[str, Any],
    generated_at: Optional[datetime] = None,
) -> None:
    """I3: 일일 리포트 적재. (care_target_id, report_date) 기준 UPSERT.

    `generated_at`을 생략하면 Spring이 수신 시각으로 채운다. 생성 시각을 아는 쪽은
    우리이므로 넘기는 편이 정확하다 — 선택 인자인 것은 기존 호출부 호환 때문이다.
    """
    body: dict[str, Any] = {
        "care_target_id": care_target_id,
        "report_date": report_date,
        "sections": sections,
        "metrics": metrics,
    }
    if generated_at:
        body["generated_at"] = generated_at.isoformat()

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_BASE}/reports",
            json=body,
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
            # 신규 생성일 때만 보호자 푸시가 나간다 — 재적재는 갱신(200).
            # 키 이름에 `created`를 쓰면 안 된다. LogRecord의 예약 속성이라
            # extra로 덮는 순간 KeyError로 로깅이 터진다(적재는 이미 끝난 뒤라 더 헷갈린다).
            "newly_created": resp.status_code == 201,
        },
    )


async def send_heartbeat(device_uid: str) -> bool:
    """I4: 노드 생존 신호. `tb_device.last_seen_at`을 갱신한다.

    보드가 살아 있다는 사실을 아는 건 CSI 라인을 받는 우리뿐이다. 이걸 보내지 않으면
    Spring의 `last_seen_at`이 영원히 null이고, 보호자 앱 디바이스 화면은 노드가 멀쩡히
    송신 중인데도 "신호 없음"으로 표시한다.

    **404를 예외로 올리지 않는다.** Spring은 등록되지 않은 device_uid에 404를 준다.
    보드는 켰는데 앱에서 등록을 안 한 상태가 설치 현장에서 가장 흔한데, 그때마다
    예외가 올라오면 주기마다 스택트레이스가 쌓인다. 호출부가 MAC당 한 번만 경고하도록
    성공 여부만 돌려준다.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_BASE}/devices/{device_uid}/heartbeat",
            headers=_HEADERS,
            timeout=5.0,
        )
    if resp.status_code == 404:
        return False
    resp.raise_for_status()
    # 성공은 로그를 남기지 않는다 — 보드 수 × 주기만큼 데몬 로그가 하트비트로 덮인다
    return True


async def send_pose_clip(
    sensing_event_id: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """I5: 복원 스켈레톤 클립 적재. 이벤트당 1건 멱등 — 재요청 시 기존 id를 반환한다."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_BASE}/sensing-events/{sensing_event_id}/pose-clip",
            json=payload,
            headers=_HEADERS,
            timeout=10.0,
        )
        resp.raise_for_status()
    data = resp.json()["data"]
    logger.info(
        "포즈 클립 적재 완료",
        extra={
            "action": "pose_clip_saved",
            "sensing_event_id": sensing_event_id,
            "pose_clip_id": data.get("pose_clip_id"),
        },
    )
    return data
