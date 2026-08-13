"""AI Agent 진입점 — sensing server가 모델 결과를 POST하면 에스컬레이션 실행."""
from datetime import date
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Header
from pydantic import BaseModel

from app.agent import escalation_agent, report_service
from app.agent.schemas import DailyReportOutput, ModelResult
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


class DailyReportRunRequest(BaseModel):
    care_target_id: int
    #: 생략하면 KST 기준 어제. 리포트는 지난 하루를 요약하는 것이 기본 동작이다.
    report_date: Optional[date] = None


@router.post("/reports/run", response_model=DailyReportOutput)
async def run_daily_report(
    body: DailyReportRunRequest,
    x_internal_key: str = Header(default=""),
) -> DailyReportOutput:
    """일일 리포트를 생성해 Spring에 적재하고 생성 결과를 반환한다.

    에스컬레이션(`/run`)과 달리 **동기 실행**이다. LLM 호출 1회라 수 초면 끝나고,
    호출자가 생성된 문장과 실패 여부를 그 자리에서 확인할 수 있어야 한다.

    같은 (노인, 날짜)로 다시 부르면 Spring이 UPSERT로 갱신한다 — 보호자 푸시는
    최초 생성 때만 나가므로 재실행이 알림을 중복 발송하지 않는다.
    """
    check_internal_key(x_internal_key)

    report_date = body.report_date or report_service.default_report_date()
    return await report_service.run_daily_report(body.care_target_id, report_date)
