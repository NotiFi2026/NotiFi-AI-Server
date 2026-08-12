"""모델 추론 엔드포인트 — 캘리브레이션된 디바이스의 CSI 윈도를 판정한다."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from fastapi import APIRouter, File, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from app.api.auth import check_internal_key
from app.common.logging_config import logger
from app.config import settings

if TYPE_CHECKING:
    # 런타임에 import하면 torch·notifi_ai가 기동 시 무조건 로드된다.
    # 모델 미설치 환경에서도 서버는 떠야 하므로 타입 검사에서만 참조한다.
    from app.model import ModelRuntime

router = APIRouter(prefix="/internal/model")

# 레지스트리 경로를 만들기 전에 막는다 — device_id는 경로 세그먼트로 쓰인다
_DEVICE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# 쿼리 NPZ는 304프레임 기준 ~1MB. 여유를 두되 무제한 적재는 막는다
_MAX_UPLOAD_BYTES = 8 * 1024 * 1024


def _runtime(request: Request) -> "ModelRuntime":
    runtime = getattr(request.app.state, "model_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model runtime is not loaded")
    return runtime


@router.get("/health")
async def model_health(request: Request, x_internal_key: str = Header(default="")) -> dict:
    """모델 로드 상태를 반환한다. 인증 없이 호출 가능하되 최소 정보만 노출한다.

    enabled인데 로드되지 않았으면 503 — 모니터링이 "모델 없는 서버"를 정상으로 보면 안 된다.
    """
    runtime = getattr(request.app.state, "model_runtime", None)
    if runtime is None:
        if settings.notifi_model_enabled:
            raise HTTPException(status_code=503, detail="Model failed to load")
        return {"loaded": False, "enabled": False}

    described = runtime.describe()
    body = {
        "loaded": True,
        "enabled": True,
        "model_name": described["model_name"],
        "device": described["device"],
        "actions": described["actions"],
        "risks": described["risks"],
    }
    # 설치 경로·성능지표·등록 디바이스는 인증된 호출에만
    if x_internal_key and x_internal_key == settings.spring_internal_key:
        body["artifact_dir"] = described["artifact_dir"]
        body["metadata"] = described["metadata"]
        body["devices"] = runtime.list_devices()
    return body


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
    check_internal_key(x_internal_key)
    if not _DEVICE_ID.match(device_id):
        raise HTTPException(status_code=400, detail="Invalid device_id")
    if len(file) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload too large")

    runtime = _runtime(request)
    try:
        return await run_in_threadpool(
            runtime.predict_npz, device_id, file, include_pose
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        # 입력 계열만 400. RuntimeError(CUDA OOM 등)는 서버 장애이므로 500으로 올린다
        logger.warning(
            "모델 추론 입력 오류",
            extra={"action": "model_predict_rejected", "device_id": device_id, "error": str(exc)},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(
            "모델 추론 실패",
            extra={"action": "model_predict_failed", "device_id": device_id, "error": str(exc)},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Inference failed") from exc
