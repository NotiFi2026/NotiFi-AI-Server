"""모델 라우터가 공유하는 가드와 추론 실행기.

여기서 `notifi_ai`·torch를 런타임에 import하면 모델 미설치 환경에서 서버가
아예 부팅되지 않는다. ModelRuntime은 타입 검사에서만 참조한다.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from app.api.auth import check_internal_key
from app.common.logging_config import logger
from app.model.errors import InferenceBusyError

if TYPE_CHECKING:
    from app.model.runtime import ModelRuntime

# 레지스트리 경로를 만들기 전에 막는다 — device_id는 경로 세그먼트로 쓰인다
DEVICE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# 쿼리 NPZ는 304프레임 기준 ~1MB. 여유를 두되 무제한 적재는 막는다
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


def get_runtime(request: Request) -> "ModelRuntime":
    runtime = getattr(request.app.state, "model_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model runtime is not loaded")
    return runtime


def require_device_id(device_id: str) -> None:
    if not DEVICE_ID.match(device_id):
        raise HTTPException(status_code=400, detail="Invalid device_id")


def get_pump(request: Request) -> Any:
    """수집 데몬. 꺼져 있으면 라이브 캡처 계열 엔드포인트는 성립하지 않는다."""
    pump = getattr(request.app.state, "stream_pump", None)
    if pump is None:
        raise HTTPException(status_code=503, detail="CSI stream is not running")
    return pump


def refresh_stream_mapping(request: Request, runtime: "ModelRuntime", dropped: str | None = None) -> None:
    """등록·삭제 후 수집 데몬의 보드 매핑을 갱신한다.

    안 하면 방금 등록한 보드의 패킷이 "등록되지 않은 보드"로 계속 버려진다 —
    설치 현장에서 원인 찾기 제일 어려운 종류의 침묵이다.
    """
    pump = getattr(request.app.state, "stream_pump", None)
    if pump is None:
        return
    if dropped is not None:
        pump.buffers.drop(dropped)
    pump.router.reload(runtime.list_device_configs())


def guard_inference(x_internal_key: str, device_id: str, file: bytes) -> None:
    """두 추론 엔드포인트가 공유하는 입력 가드 — 한쪽만 고치는 사고를 막는다."""
    check_internal_key(x_internal_key)
    require_device_id(device_id)
    if len(file) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload too large")


async def run_inference(
    runtime: "ModelRuntime",
    device_id: str,
    file: bytes,
    include_pose: bool,
    action: str,
) -> dict[str, Any]:
    """추론 실행 + 실패 분류. 응답 본문에는 내부 경로를 담지 않는다."""
    try:
        return await run_in_threadpool(runtime.predict_npz, device_id, file, include_pose)
    except InferenceBusyError as exc:
        # 502(Spring 장애)와 구분한다 — 이건 이 서버가 바쁜 것이고, 호출자는 윈도를 버리면 된다
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (FileNotFoundError, KeyError, ValueError) as exc:
        # 입력 계열만 400. 예외 메시지에는 calibration.pt 절대경로가 들어 있어 로그로만 남긴다
        logger.warning(
            "추론 입력 오류",
            extra={"action": f"{action}_rejected", "device_id": device_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=400, detail="Invalid query NPZ or missing calibration profile"
        ) from exc
    except Exception as exc:
        # ModelContractError·CUDA 오류 등 서버 문제는 500 (400으로 내리면 호출자가 재시도하지 않는다)
        logger.error(
            "추론 실패",
            extra={"action": f"{action}_failed", "device_id": device_id, "error": str(exc)},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Inference failed") from exc
