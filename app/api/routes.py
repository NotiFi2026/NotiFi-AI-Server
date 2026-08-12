"""AI Agent 진입점 — sensing server가 모델 결과를 POST하면 에스컬레이션 실행."""
from fastapi import APIRouter, BackgroundTasks, Header

from app.agent import escalation_agent
from app.agent.schemas import ModelResult
from app.api.auth import check_internal_key
from app.api.status import update_risk_level
from app.common.logging_config import logger

router = APIRouter(prefix="/internal/agent")


async def run_agent_safely(
    model_result: ModelResult,
    prefetched: dict | None = None,
) -> None:
    """백그라운드 실행 래퍼 — 예외를 로그로 흡수.

    prefetched: 호출자가 이미 I1을 보냈다면 그 응답(추론 파이프라인 경로).
    """
    try:
        await escalation_agent.run(model_result, prefetched=prefetched)
    except Exception as exc:
        logger.error(
            "에스컬레이션 실행 오류",
            extra={
                "action": "agent_error",
                "care_target_id": model_result.care_target_id,
                "error": str(exc),
            },
            exc_info=True,
        )


@router.post("/run", status_code=202)
async def run_agent(
    model_result: ModelResult,
    background_tasks: BackgroundTasks,
    x_internal_key: str = Header(default=""),
) -> dict:
    """모델 결과를 받아 에스컬레이션 흐름을 백그라운드로 실행한다.

    sensing server → POST /internal/agent/run
    → 202 즉시 반환 → 백그라운드에서 LangGraph 실행
    """
    check_internal_key(x_internal_key)

    update_risk_level(model_result.risk_level.value)
    background_tasks.add_task(run_agent_safely, model_result)

    logger.info(
        "에스컬레이션 요청 수신",
        extra={
            "action": "model_result_received",
            "care_target_id": model_result.care_target_id,
            "risk_level": model_result.risk_level.value,
            "event_type": model_result.event_type.value,
        },
    )
    return {
        "accepted": True,
        "care_target_id": model_result.care_target_id,
        "risk_level": model_result.risk_level.value,
    }
