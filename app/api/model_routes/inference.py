"""추론 — 상태 조회, 순수 추론 프로브, Spring 적재 파이프라인.

predict는 판정 결과를 호출자에게 돌려주기만 한다(디버깅·검증용).
ingest는 그 결과를 Spring 적재·에스컬레이션까지 이어붙인 운영 경로다.
"""
from __future__ import annotations

from datetime import datetime, timezone

import httpx
from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    Header,
    HTTPException,
    Request,
)

from app.agent import escalation_agent
from app.agent.payload_builder import build_sensing_event_payload
from app.agent.schemas import EventType, RiskLevel
from app.api.auth import check_internal_key
from app.api.model_routes._common import get_runtime, guard_inference, run_inference
from app.api.routes import run_agent_safely
from app.clients import spring_client
from app.common.logging_config import logger
from app.config import settings
from app.model import pipeline
from app.model.errors import ModelContractError

router = APIRouter()


@router.get("/health")
async def model_health(request: Request, x_internal_key: str = Header(default="")) -> dict:
    """모델 로드 상태를 반환한다. 인증 없이 호출 가능하되 최소 정보만 노출한다.

    enabled인데 로드되지 않았으면 503 — 모니터링이 "모델 없는 서버"를 정상으로 보면 안 된다.
    추론이 멈춘 경우에도 503 — 멈춘 CUDA 연산은 되돌릴 수 없으므로 재시작이 유일한 대응이고,
    이 신호가 없으면 서버가 사실상 죽은 채로 200을 반환한다.
    """
    runtime = getattr(request.app.state, "model_runtime", None)
    if runtime is None:
        if settings.notifi_model_enabled:
            raise HTTPException(status_code=503, detail="Model failed to load")
        return {"loaded": False, "enabled": False}

    inflight = runtime.inflight_seconds()
    if inflight is not None and inflight > settings.notifi_inference_stuck_seconds:
        logger.error(
            "추론 멈춤 감지",
            extra={"action": "inference_stuck", "inflight_seconds": inflight},
        )
        raise HTTPException(
            status_code=503, detail=f"Inference stuck for {inflight:.0f}s"
        )

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
        body["inflight_seconds"] = inflight
        body["last_success_age_seconds"] = runtime.last_success_age_seconds()
    return body


@router.get("/spec")
async def model_spec(request: Request, x_internal_key: str = Header(default="")) -> dict:
    """모델이 말하는 계약 — 행동 라벨·위험 등급·관절 스키마·입력 형상.

    모델은 계속 갱신되므로 하위(Spring·앱·수집 데몬)가 자기 가정이 아직 맞는지
    런타임에 확인할 수 있어야 한다. 스모크 테스트도 이걸 보고 어긋남을 조기 발견한다.
    """
    check_internal_key(x_internal_key)
    return get_runtime(request).spec.as_dict()


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
    guard_inference(x_internal_key, device_id, file)
    return await run_inference(
        get_runtime(request), device_id, file, include_pose, "model_predict"
    )


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
    guard_inference(x_internal_key, device_id, file)
    derived = pipeline.care_target_id_from(device_id)
    if derived is not None and derived != care_target_id:
        # device_id=care-5인데 care_target_id=7이면 응급 이벤트가 엉뚱한 노인에게 적재된다
        raise HTTPException(
            status_code=400,
            detail=f"device_id targets care_target {derived}, not {care_target_id}",
        )
    if window_end_at is not None and window_end_at.tzinfo is None:
        # naive 시각을 받아들이면 로컬시간으로 해석돼 detected_at이 조용히 어긋난다.
        # detected_at은 멱등키의 일부라 중복 판정까지 깨지므로 거부한다.
        raise HTTPException(
            status_code=400, detail="window_end_at requires a timezone offset"
        )

    # 항상 포즈까지 받는다 — 추론 전에는 비정상 여부를 알 수 없고,
    # 배열 직렬화는 수 ms로 추론 210ms 대비 무시할 수준이다.
    pred = await run_inference(
        get_runtime(request), device_id, file, True, "model_ingest"
    )

    detected_at = window_end_at or datetime.now(timezone.utc)
    try:
        model_result = pipeline.to_model_result(pred, care_target_id, spring_device_id, detected_at)
    except ModelContractError as exc:
        logger.error(
            "모델 계약 위반",
            extra={"action": "model_contract_error", "device_id": device_id, "error": str(exc)},
        )
        raise HTTPException(status_code=500, detail="Unexpected model output") from exc

    activity_class = model_result.activity_class

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
        # Spring 장애를 삼키면 호출자가 재시도하지 못한다.
        # 여기서 빠져나가면 mark_sent를 하지 않으므로 다음 윈도가 절감으로 막히지 않는다.
        logger.error(
            "Spring 적재 실패",
            extra={"action": "spring_ingest_failed", "device_id": device_id, "error": str(exc)},
            exc_info=True,
        )
        raise HTTPException(status_code=502, detail="Spring ingest failed") from exc

    pipeline.mark_sent(device_id, activity_class, detected_at)

    if model_result.risk_level is RiskLevel.DANGER:
        background_tasks.add_task(run_agent_safely, model_result, saved)

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
