"""라이브 캘리브레이션 — 수집 데몬 버퍼에서 트라이얼을 떠서 프로필을 만든다.

기존 `/calibrate`(NPZ 업로드)는 CSI를 직접 읽는 쪽만 쓸 수 있어서 앱이 캘리브레이션을
시작할 방법이 없었다. 여기는 "지금 이 10.13초를 한 장 담아라"만 요구하므로,
앱은 안내 문구를 띄우고 버튼만 누르면 된다.

수집 순서(모델 문서 §3.4): 무인 10초 × 12 → 기본 8동작 × 2 → 낙상 5종(선택).
"""
from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.api.auth import check_internal_key
from app.api.model_routes._common import get_pump, get_runtime, require_device_id
from app.api.model_routes.devices import calibration_warnings
from app.common.logging_config import logger
from app.model.errors import InferenceBusyError
# collector가 아니라 errors에서 받는다 — collector는 notifi_ai를 끌어와 부팅을 깬다
from app.stream.errors import NotEnoughSignal

router = APIRouter()


class CaptureRequest(BaseModel):
    """action_id가 없으면 무인(absence) 트라이얼."""

    action_id: int | None = Field(default=None, ge=0)


@router.post("/devices/{device_id}/calibration/trials", status_code=201)
async def capture_trial(
    request: Request,
    device_id: str,
    body: CaptureRequest,
    x_internal_key: str = Header(default=""),
) -> dict:
    """지금 버퍼의 마지막 한 윈도를 트라이얼로 담는다.

    호출 시점 기준 **직전** 10.13초를 담으므로, 앱은 동작을 마친 뒤에 부르면 된다
    (카운트다운 → 동작 → 호출).
    """
    check_internal_key(x_internal_key)
    require_device_id(device_id)
    pump = get_pump(request)
    runtime = get_runtime(request)

    if body.action_id is not None and body.action_id >= len(runtime.spec.action_labels):
        raise HTTPException(
            status_code=400,
            detail=f"action_id는 0~{len(runtime.spec.action_labels) - 1} 범위여야 한다",
        )

    try:
        index, trial = pump.sessions.capture(device_id, body.action_id)
    except NotEnoughSignal as exc:
        # 보드가 꺼졌거나 방금 시작했다 — 실패 이유를 그대로 알려줘야 조치할 수 있다
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    logger.info(
        "캘리브레이션 트라이얼 캡처",
        extra={
            "action": "calibration_trial_captured",
            "device_id": device_id,
            "kind": trial.kind,
            "action_id": trial.action_id,
            "link_coverage": trial.coverage(),
        },
    )
    return trial.summary(index)


@router.get("/devices/{device_id}/calibration")
async def session_progress(
    request: Request,
    device_id: str,
    x_internal_key: str = Header(default=""),
) -> dict:
    """진행 상황 — 위저드가 어디까지 찍었는지 보여주고 재진입을 판단한다."""
    check_internal_key(x_internal_key)
    require_device_id(device_id)
    return get_pump(request).sessions.get(device_id).progress()


@router.delete("/devices/{device_id}/calibration/trials/{index}")
async def drop_trial(
    request: Request,
    device_id: str,
    index: int,
    x_internal_key: str = Header(default=""),
) -> dict:
    """재촬영 — 신호가 나빴던 트라이얼을 빼고 다시 찍는다."""
    check_internal_key(x_internal_key)
    require_device_id(device_id)
    if not get_pump(request).sessions.drop_trial(device_id, index):
        raise HTTPException(status_code=404, detail="Trial not found")
    return {"device_id": device_id, "index": index, "deleted": True}


@router.post("/devices/{device_id}/calibration/fit")
async def fit_session(
    request: Request,
    device_id: str,
    x_internal_key: str = Header(default=""),
) -> dict:
    """모아둔 트라이얼로 프로필을 학습·저장하고 세션을 비운다."""
    check_internal_key(x_internal_key)
    require_device_id(device_id)
    pump = get_pump(request)
    runtime = get_runtime(request)

    session = pump.sessions.get(device_id)
    absence = session.absence()
    if not absence:
        # absence가 캘리브레이션의 기준선이다 — 없으면 학습 자체가 성립하지 않는다
        raise HTTPException(status_code=400, detail="무인(absence) 트라이얼이 최소 1건 필요하다")

    try:
        summary = await run_in_threadpool(
            runtime.fit_calibration_arrays, device_id, absence, session.support(runtime.spec)
        )
    except InferenceBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="Device not registered") from exc
    except Exception as exc:
        logger.error(
            "라이브 캘리브레이션 실패",
            extra={"action": "calibration_failed", "device_id": device_id, "error": str(exc)},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Calibration failed") from exc

    # 성공했을 때만 비운다 — 실패했는데 지워버리면 몇 분치 수집이 날아간다
    pump.sessions.clear(device_id)

    warnings = calibration_warnings(summary)
    return {"device_id": device_id, "usable": not warnings, "warnings": warnings, **summary}
