"""모델 추론 엔드포인트 — 캘리브레이션된 디바이스의 CSI 윈도를 판정한다."""
from fastapi import APIRouter, File, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from app.common.logging_config import logger
from app.config import settings
from app.model import ModelRuntime

router = APIRouter(prefix="/internal/model")


def _runtime(request: Request) -> ModelRuntime:
    runtime = getattr(request.app.state, "model_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model runtime is not loaded")
    return runtime


def _check_key(x_internal_key: str) -> None:
    if x_internal_key != settings.spring_internal_key:
        raise HTTPException(status_code=401, detail="Invalid internal key")


@router.get("/health")
async def model_health(request: Request) -> dict:
    """모델 로드 상태와 메타데이터(17행동·3위험도)를 반환한다. 인증 없음 — 헬스체크용."""
    runtime = getattr(request.app.state, "model_runtime", None)
    if runtime is None:
        return {"loaded": False, "enabled": settings.notifi_model_enabled}
    return {"loaded": True, "enabled": True, "devices": runtime.list_devices(), **runtime.describe()}


@router.post("/devices/{device_id}/predict")
async def predict(
    request: Request,
    device_id: str,
    file: bytes = File(...),
    include_pose: bool = False,
    x_internal_key: str = Header(default=""),
) -> dict:
    """쿼리 NPZ(csi [T,3,114,2] + link_mask [T,3])를 추론한다.

    캘리브레이션 프로필이 없는 디바이스는 400 — 무보정 추론은 허용하지 않는다.
    """
    _check_key(x_internal_key)
    runtime = _runtime(request)
    try:
        return await run_in_threadpool(
            runtime.predict_npz, device_id, file, include_pose
        )
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        logger.warning(
            "모델 추론 실패",
            extra={"action": "model_predict_failed", "device_id": device_id, "error": str(exc)},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
