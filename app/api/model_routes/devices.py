"""설치 생명주기 — 디바이스 등록·상태·삭제와 캘리브레이션.

설치 순서는 등록 → 캘리브레이션 → 추론이다. 프로필 없이는 추론이 400이므로
이 엔드포인트들이 없으면 새 가구를 붙일 수 없다.
"""
from __future__ import annotations

import threading
from typing import Any

from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.api.auth import check_internal_key
from app.api.model_routes._common import get_runtime, require_device_id
from app.common.logging_config import logger
from app.config import settings
from app.model import pipeline
from app.model.errors import InferenceBusyError

router = APIRouter()

# 캘리브레이션은 업로드·압축해제만으로 요청당 수십 MB를 쓴다. 모델 락은 학습 구간만
# 막으므로, 그 앞단까지 포함해 동시 1건으로 제한한다.
_calibration_slot = threading.Semaphore(1)

# 모델 문서 §3.4 수집 규모
_ABSENCE_EXPECTED = 12
_BASIC_ACTIONS = (0, 1, 2, 3, 4, 5, 7, 8)
_BASIC_REPEATS = 2


class DeviceRegisterRequest(BaseModel):
    """설치 1채(RX 1 + TX 3 보드)의 등록 정보.

    device_id는 서버가 care_target_id에서 파생한다 — 앱이 정하지 않는다.
    보드 방향(RX=North/TX1=South/TX2=West/TX3=East)은 모델이 고정 계약으로 강제하므로
    선택지로 노출하지 않는다.
    """

    care_target_id: int = Field(gt=0)
    rx_id: str = Field(min_length=1, max_length=64)
    tx1_id: str = Field(min_length=1, max_length=64)
    tx2_id: str = Field(min_length=1, max_length=64)
    tx3_id: str = Field(min_length=1, max_length=64)
    firmware_version: str = "unknown"
    notes: str = ""


def _calibration_warnings(summary: dict[str, Any]) -> list[str]:
    """설치 품질·수집 규모 경고. 거부하지 않고 알린다 — 재시도로 덮어쓸 수 있다."""
    warnings: list[str] = []
    valid_links = sum(1 for ok in summary.get("baseline_link_valid", []) if ok)
    if valid_links < 2:
        warnings.append(
            f"유효 링크 {valid_links}개 — 최소 2개 필요. 케이블·전원·안테나·sender ID를 확인하라"
        )
    weak = [
        index for index, coverage in enumerate(summary.get("link_coverage", []))
        if coverage < 0.35
    ]
    if weak:
        warnings.append(f"링크 커버리지 부족: TX{[i + 1 for i in weak]}")

    absence = summary.get("absence_trials")
    if absence is not None and absence < _ABSENCE_EXPECTED:
        warnings.append(f"무인 트라이얼 {absence}회 — 권장 {_ABSENCE_EXPECTED}회(각 10초)")

    counts = summary.get("support_action_counts")
    if counts:
        missing = [
            action for action in _BASIC_ACTIONS
            if action < len(counts) and counts[action] < _BASIC_REPEATS
        ]
        if missing:
            warnings.append(f"기본 동작 수집 부족(각 {_BASIC_REPEATS}회 권장): action {missing}")
    return warnings


@router.post("/devices", status_code=201)
async def register_device(
    request: Request,
    body: DeviceRegisterRequest,
    x_internal_key: str = Header(default=""),
) -> dict:
    """설치 1채를 등록한다. 캘리브레이션보다 먼저 호출해야 한다."""
    check_internal_key(x_internal_key)
    runtime = get_runtime(request)

    device_id = pipeline.device_id_for(body.care_target_id)
    config = {
        "device_id": device_id,
        "rx_id": body.rx_id,
        "tx1_id": body.tx1_id,
        "tx2_id": body.tx2_id,
        "tx3_id": body.tx3_id,
        "firmware_version": body.firmware_version,
        "notes": body.notes,
    }
    try:
        result = await run_in_threadpool(runtime.register_device, config)
    except (TypeError, ValueError) as exc:
        # 보드 ID 중복·공백 등 — 계약 위반 내용은 그대로 알려줘야 고칠 수 있다
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"care_target_id": body.care_target_id, **result}


@router.delete("/devices/{device_id}")
async def delete_device(
    request: Request,
    device_id: str,
    x_internal_key: str = Header(default=""),
) -> dict:
    """등록·캘리브레이션을 함께 제거한다. 재설치와 파생 상태 정리에 쓴다."""
    check_internal_key(x_internal_key)
    require_device_id(device_id)
    runtime = get_runtime(request)
    if not await run_in_threadpool(runtime.delete_device, device_id):
        raise HTTPException(status_code=404, detail="Device not found")
    return {"device_id": device_id, "deleted": True}


@router.get("/devices/{device_id}")
async def device_status(
    request: Request,
    device_id: str,
    x_internal_key: str = Header(default=""),
) -> dict:
    """등록·캘리브레이션 진행 상태. 위저드가 어디부터 시작할지 판단한다."""
    check_internal_key(x_internal_key)
    require_device_id(device_id)
    runtime = get_runtime(request)
    status = await run_in_threadpool(runtime.device_status, device_id)
    if status.get("calibration"):
        status["warnings"] = _calibration_warnings(status["calibration"])
        status["usable"] = not status["warnings"]
    return status


@router.post("/devices/{device_id}/calibrate")
async def calibrate(
    request: Request,
    device_id: str,
    file: UploadFile = File(...),
    x_internal_key: str = Header(default=""),
) -> dict:
    """캘리브레이션 NPZ로 프로필을 학습한다.

    absence_csi/absence_mask 필수, support_* 선택. 트라이얼당 ~0.79MiB라
    추론(8MiB)과 다른 상한을 쓴다. 업로드를 메모리에 통째로 올리지 않도록
    UploadFile(1MiB 초과분은 디스크 스풀)을 그대로 넘긴다.
    """
    check_internal_key(x_internal_key)
    require_device_id(device_id)
    limit = settings.notifi_calibration_max_upload_mb * 1024 * 1024
    if file.size is not None and file.size > limit:
        raise HTTPException(status_code=413, detail="Upload too large")

    runtime = get_runtime(request)
    if not _calibration_slot.acquire(blocking=False):
        # 업로드·압축해제만으로 요청당 수십 MB다. 모델 락은 학습 구간만 막으므로
        # 여기서 막지 않으면 동시 요청이 메모리를 밀어낸다.
        raise HTTPException(status_code=503, detail="Another calibration is in progress")
    try:
        summary = await run_in_threadpool(
            runtime.fit_calibration_npz, device_id, file.file
        )
    except InferenceBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (FileNotFoundError, KeyError, ValueError) as exc:
        # 미등록 디바이스·NPZ 계약 위반. 경로가 섞일 수 있어 본문은 일반 메시지로 둔다
        logger.warning(
            "캘리브레이션 입력 오류",
            extra={"action": "calibration_rejected", "device_id": device_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=400,
            detail="Device not registered or invalid calibration NPZ",
        ) from exc
    except Exception as exc:
        logger.error(
            "캘리브레이션 실패",
            extra={"action": "calibration_failed", "device_id": device_id, "error": str(exc)},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Calibration failed") from exc
    finally:
        _calibration_slot.release()

    warnings = _calibration_warnings(summary)
    return {"device_id": device_id, "usable": not warnings, "warnings": warnings, **summary}
