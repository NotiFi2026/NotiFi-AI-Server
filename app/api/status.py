"""발표용 실시간 상태 엔드포인트."""
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

RiskLevel = Literal["SAFE", "WARNING", "DANGER"]

_current_risk_level: RiskLevel = "SAFE"


def update_risk_level(level: str) -> None:
    """agent run 시 자동으로 상태를 갱신한다."""
    global _current_risk_level
    normalized = level.upper()
    if normalized in ("SAFE", "WARNING", "DANGER"):
        _current_risk_level = normalized  # type: ignore[assignment]


class StatusResponse(BaseModel):
    risk_level: RiskLevel


class DemoOverride(BaseModel):
    risk_level: RiskLevel


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """앱이 3초마다 폴링하는 현재 위험도 반환."""
    return StatusResponse(risk_level=_current_risk_level)


@router.post("/status/demo", response_model=StatusResponse)
async def set_status_demo(body: DemoOverride) -> StatusResponse:
    """발표 중 수동으로 위험도를 변경한다. 인증 없음 — 데모 전용."""
    update_risk_level(body.risk_level)
    return StatusResponse(risk_level=_current_risk_level)
