"""추론 결과를 Spring까지 밀어 넣는 경로. **HTTP 엔드포인트와 수집 데몬이 공유한다.**

두 벌로 갈라두면 한쪽만 고치는 사고가 난다 — NORMAL 절감·클립 적재·에스컬레이션 트리거는
전부 순서와 실패 처리가 얽혀 있어서, 복사본이 조용히 달라지면 이벤트가 유실된다.

FastAPI를 import하지 않는다. 예외는 그대로 올려보내고 HTTP 매핑은 라우터가 한다
(데몬은 같은 예외를 "윈도 버리기"로 처리한다).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from app.agent import escalation_agent
from app.agent.payload_builder import build_sensing_event_payload
from app.agent.schemas import EventType, ModelResult, RiskLevel
from app.clients import spring_client
from app.common.logging_config import logger
from app.model import pipeline

#: danger 판정 시 에이전트를 어떻게 띄울지는 호출자가 정한다.
#: HTTP는 BackgroundTasks, 데몬은 asyncio.create_task — 여기서 정하면 둘 중 하나가 어색해진다.
DangerScheduler = Callable[[ModelResult, dict[str, Any]], None]


async def deliver(
    pred: dict[str, Any],
    *,
    device_id: str,
    care_target_id: int,
    spring_device_id: int | None,
    detected_at: datetime,
    schedule_danger: DangerScheduler,
) -> dict[str, Any]:
    """추론 dict → I1 적재 → (비정상이면) I5 클립 → (danger면) 에스컬레이션.

    Raises:
        ModelContractError: 모델 출력이 계약을 벗어남 (서버 문제)
        httpx.HTTPError: Spring 적재 실패 — 삼키면 호출자가 재시도하지 못한다
    """
    model_result = pipeline.to_model_result(pred, care_target_id, spring_device_id, detected_at)
    activity_class = model_result.activity_class

    if not pipeline.should_send(device_id, model_result.event_type, activity_class, detected_at):
        logger.info(
            "NORMAL 절감 — 전송 생략",
            extra={
                "action": "model_ingest_throttled",
                "device_id": device_id,
                "activity_class": activity_class,
            },
        )
        return {"sent": False, "reason": "normal_throttled", "activity_class": activity_class}

    # 여기서 예외로 빠져나가면 mark_sent를 하지 않으므로 다음 윈도가 절감으로 막히지 않는다
    saved = await spring_client.send_sensing_event(
        build_sensing_event_payload(escalation_agent.initial_state(model_result))
    )
    pose_clip_id = None
    if model_result.event_type is not EventType.NORMAL and saved.get("sensing_event_id"):
        # 클립을 에이전트보다 먼저 보낸다 — danger 흐름은 음성확인·대기로 수 분이라
        # 클립이 늦으면 보호자 알림 시점에 리플레이가 없다
        clip = await spring_client.send_pose_clip(
            saved["sensing_event_id"],
            pipeline.build_pose_clip_payload(
                pred, pipeline.window_start(detected_at, pred), detected_at
            ),
        )
        pose_clip_id = clip.get("pose_clip_id")

    pipeline.mark_sent(device_id, activity_class, detected_at)

    if model_result.risk_level is RiskLevel.DANGER:
        schedule_danger(model_result, saved)

    return {
        "sent": True,
        "sensing_event_id": saved.get("sensing_event_id"),
        "event_type": model_result.event_type.value,
        "activity_class": activity_class,
        "risk_level": model_result.risk_level.value,
        "risk_score": model_result.risk_score,
        "escalation_triggered": saved.get("escalation_triggered", False),
        "pose_clip_id": pose_clip_id,
    }
