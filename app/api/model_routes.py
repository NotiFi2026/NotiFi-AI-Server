"""모델 추론 엔드포인트 — 캘리브레이션된 디바이스의 CSI 윈도를 판정한다."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, BackgroundTasks, File, Form, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from app.agent import escalation_agent
from app.agent.payload_builder import build_sensing_event_payload
from app.agent.schemas import EventType
from app.api.auth import check_internal_key
from app.clients import spring_client
from app.common.logging_config import logger
from app.config import settings
from app.model import pipeline

if TYPE_CHECKING:
    # 런타임에 import하면 torch·notifi_ai가 기동 시 무조건 로드된다.
    # 모델 미설치 환경에서도 서버는 떠야 하므로 타입 검사에서만 참조한다.
    from app.model.runtime import ModelRuntime

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


@router.post("/devices/{device_id}/ingest", status_code=202)
async def ingest(
    request: Request,
    background_tasks: BackgroundTasks,
    device_id: str,
    file: bytes = File(...),
    care_target_id: int = Form(...),
    spring_device_id: int | None = Form(default=None),
    window_end_at: datetime | None = Form(default=None),
    x_internal_key: str = Header(default=""),
) -> dict:
    """추론 → I1 적재 → (비정상이면) I5 클립 → (danger면) 에스컬레이션.

    care_target_id는 호출자가 준다 — 모델 레지스트리의 device_id와 Spring의 노인 ID를
    잇는 수단이 아직 없다. window_end_at도 모델이 시각을 모르므로 호출자가 준다(기본 now).
    """
    check_internal_key(x_internal_key)
    if not _DEVICE_ID.match(device_id):
        raise HTTPException(status_code=400, detail="Invalid device_id")
    if len(file) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload too large")

    runtime = _runtime(request)
    try:
        pred = await run_in_threadpool(runtime.predict_npz, device_id, file, True)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.warning(
            "인제스트 입력 오류",
            extra={"action": "model_ingest_rejected", "device_id": device_id, "error": str(exc)},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(
            "인제스트 추론 실패",
            extra={"action": "model_ingest_failed", "device_id": device_id, "error": str(exc)},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    detected_at = window_end_at or datetime.now(timezone.utc)
    model_result = pipeline.to_model_result(pred, care_target_id, spring_device_id, detected_at)
    activity_class = pred["action_label"].upper()

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

    try:
        saved = await spring_client.send_sensing_event(
            build_sensing_event_payload(escalation_agent.initial_state(model_result))
        )
        pose_clip_id = None
        if model_result.event_type is not EventType.NORMAL and saved.get("sensing_event_id"):
            clip = await spring_client.send_pose_clip(
                saved["sensing_event_id"],
                pipeline.build_pose_clip_payload(
                    pred, pipeline.window_start(detected_at, pred), detected_at
                ),
            )
            pose_clip_id = clip.get("pose_clip_id")
    except httpx.HTTPError as exc:
        # Spring 장애를 삼키면 호출자가 재시도하지 못한다
        logger.error(
            "Spring 적재 실패",
            extra={"action": "spring_ingest_failed", "device_id": device_id, "error": str(exc)},
            exc_info=True,
        )
        raise HTTPException(status_code=502, detail="Spring ingest failed") from exc

    if model_result.risk_level.value == "danger":
        background_tasks.add_task(_run_escalation, model_result, saved)

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


async def _run_escalation(model_result, prefetched: dict) -> None:
    """danger 흐름은 음성확인·대기로 수 분 걸린다 — 응답을 막지 않도록 백그라운드."""
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
