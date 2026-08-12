"""모델 추론 엔드포인트 — 캘리브레이션된 디바이스의 CSI 윈도를 판정한다."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import APIRouter, BackgroundTasks, File, Form, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.agent import escalation_agent
from app.agent.payload_builder import build_sensing_event_payload
from app.agent.schemas import EventType, RiskLevel
from app.api.auth import check_internal_key
from app.api.routes import run_agent_safely
from app.clients import spring_client
from app.common.logging_config import logger
from app.config import settings
from app.model import pipeline
from app.model.errors import InferenceBusyError, ModelContractError

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


def _guard(x_internal_key: str, device_id: str, file: bytes) -> None:
    """두 추론 엔드포인트가 공유하는 입력 가드 — 한쪽만 고치는 사고를 막는다."""
    check_internal_key(x_internal_key)
    if not _DEVICE_ID.match(device_id):
        raise HTTPException(status_code=400, detail="Invalid device_id")
    if len(file) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload too large")


async def _infer(
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


def _calibration_warnings(summary: dict[str, Any], absence_expected: int = 12) -> list[str]:
    """설치 품질 경고. 거부하지 않고 알린다 — 재시도로 덮어쓸 수 있다."""
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
    return warnings


@router.post("/devices", status_code=201)
async def register_device(
    request: Request,
    body: DeviceRegisterRequest,
    x_internal_key: str = Header(default=""),
) -> dict:
    """설치 1채를 등록한다. 캘리브레이션보다 먼저 호출해야 한다."""
    check_internal_key(x_internal_key)
    runtime = _runtime(request)

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
        await run_in_threadpool(runtime.register_device, config)
    except (TypeError, ValueError) as exc:
        # 보드 ID 중복·공백 등 — 계약 위반 내용은 그대로 알려줘야 고칠 수 있다
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"device_id": device_id, "care_target_id": body.care_target_id}


@router.get("/devices/{device_id}")
async def device_status(
    request: Request,
    device_id: str,
    x_internal_key: str = Header(default=""),
) -> dict:
    """등록·캘리브레이션 진행 상태. 위저드가 어디부터 시작할지 판단한다."""
    check_internal_key(x_internal_key)
    if not _DEVICE_ID.match(device_id):
        raise HTTPException(status_code=400, detail="Invalid device_id")
    runtime = _runtime(request)
    status = await run_in_threadpool(runtime.device_status, device_id)
    if status.get("calibration"):
        status["warnings"] = _calibration_warnings(status["calibration"])
        status["usable"] = not status["warnings"]
    return status


@router.post("/devices/{device_id}/calibrate")
async def calibrate(
    request: Request,
    device_id: str,
    file: bytes = File(...),
    x_internal_key: str = Header(default=""),
) -> dict:
    """캘리브레이션 NPZ로 프로필을 학습한다.

    absence_csi/absence_mask 필수, support_* 선택. 트라이얼당 ~0.79MiB라
    추론(8MiB)과 다른 상한을 쓴다.
    """
    check_internal_key(x_internal_key)
    if not _DEVICE_ID.match(device_id):
        raise HTTPException(status_code=400, detail="Invalid device_id")
    limit = settings.notifi_calibration_max_upload_mb * 1024 * 1024
    if len(file) > limit:
        raise HTTPException(status_code=413, detail="Upload too large")

    runtime = _runtime(request)
    try:
        summary = await run_in_threadpool(runtime.fit_calibration_npz, device_id, file)
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

    warnings = _calibration_warnings(summary)
    return {"device_id": device_id, "usable": not warnings, "warnings": warnings, **summary}


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
    _guard(x_internal_key, device_id, file)
    return await _infer(_runtime(request), device_id, file, include_pose, "model_predict")


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
    _guard(x_internal_key, device_id, file)
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
    pred = await _infer(_runtime(request), device_id, file, True, "model_ingest")

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
