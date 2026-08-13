"""AI Agent 진입점 — sensing server가 모델 결과를 POST하면 에스컬레이션 실행."""
from datetime import date
from typing import Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from openai import APIError
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


def _spring_failure(exc: httpx.HTTPStatusError, care_target_id: int) -> HTTPException:
    """Spring 내부 API 실패를 호출자가 고칠 수 있는 형태로 옮긴다.

    전부 500으로 흘리면 운영자가 원인을 알 수 없다 — 특히 키 불일치는 설정 한 줄
    문제인데 "Internal Server Error"만 보면 서버가 죽은 줄 안다.
    """
    status = exc.response.status_code
    if status == 404:
        return HTTPException(
            status_code=404,
            detail=f"care_target {care_target_id}을 Spring에서 찾을 수 없다(삭제됐거나 잘못된 ID)",
        )
    if status == 401:
        return HTTPException(
            status_code=502,
            detail="Spring 내부 API 인증 실패 — SPRING_INTERNAL_KEY가 Spring의 INTERNAL_API_KEY와 다르다",
        )
    return HTTPException(status_code=502, detail=f"Spring 내부 API 실패 (status={status})")


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
    try:
        return await report_service.run_daily_report(body.care_target_id, report_date)
    except httpx.HTTPStatusError as exc:
        raise _spring_failure(exc, body.care_target_id) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502, detail=f"Spring 서버에 연결할 수 없다: {exc}"
        ) from exc
    except APIError as exc:
        # 의존 서비스(OpenAI) 장애를 우리 버그(500)와 구분한다
        raise HTTPException(
            status_code=502, detail=f"리포트 문장 생성 실패: {exc}"
        ) from exc
