"""추론 결과 → Spring 적재 계약 변환.

`notifi_ai`를 import하지 않는다 — 런타임이 실어 보낸 plain dict만 다룬다.
모델·GPU 없이 테스트되고, 라우터가 이 모듈을 import해도 부팅에 torch가 끌려오지 않는다.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from app.agent.schemas import EventType, ModelResult, RiskLevel
from app.config import settings

# 행동의 정적 카테고리(safe 9 / warning 3 / danger 5) → Spring event_type
_RISK_ID_TO_EVENT = (EventType.NORMAL, EventType.ANOMALY, EventType.FALL)

JOINT_SCHEMA = "smpl-22"

# Spring은 risk_probability를 @Digits(fraction=3)로 받는다 — 4자리 이상이면 400
_PROBABILITY_DIGITS = 3


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), _PROBABILITY_DIGITS)


def to_model_result(
    pred: dict[str, Any],
    care_target_id: int,
    spring_device_id: int | None,
    detected_at: datetime,
) -> ModelResult:
    """추론 dict를 에이전트·I1이 쓰는 ModelResult로 변환한다.

    저품질(low_quality) 윈도는 danger 판정이라도 WARNING으로 강등한다 —
    링크 부족·프레임 결손만으로 자동 경보를 울리지 않는다는 원칙.
    """
    quality: dict[str, Any] = pred.get("quality") or {}
    risk_probability: list[float] = pred["risk_probability"]

    event_type = _RISK_ID_TO_EVENT[pred["action_risk_id"]]
    risk_level = RiskLevel(pred["risk_label"])
    # 확정 산식: warning 확률은 절반, danger 확률은 전체 가중
    risk_score = round(50 * risk_probability[1] + 100 * risk_probability[2])

    features: dict[str, Any] = {
        "activity_class": pred["action_label"].upper(),
        "action_probability": _round(max(pred["action_probability"])),
    }

    if quality.get("low_quality"):
        features["quality"] = quality
        if risk_level is RiskLevel.DANGER:
            features["degraded_from"] = RiskLevel.DANGER.value
            risk_level = RiskLevel.WARNING

    return ModelResult(
        care_target_id=care_target_id,
        device_id=spring_device_id,
        event_type=event_type,
        label=pred["action_label"],
        risk_level=risk_level,
        confidence=_round(quality.get("risk_confidence", 0.0)),
        risk_score=max(0, min(100, risk_score)),
        model_version=pred.get("model_name", "NotiFi_AI_v1"),
        detected_at=detected_at,
        features=features,
    )


def build_pose_clip_payload(
    pred: dict[str, Any],
    window_start_at: datetime,
    window_end_at: datetime,
) -> dict[str, Any]:
    """I5 클립 페이로드. frames는 top-level JSON 객체여야 한다(배열 불가)."""
    pose_rel = pred["pose_rel"]
    fps = pred["fps"]
    frame_count = len(pose_rel)
    return {
        "model_version": pred.get("model_name", "NotiFi_AI_v1"),
        "joint_schema": JOINT_SCHEMA,
        "fps": round(fps),
        "frame_count": frame_count,
        "duration_ms": round(frame_count / fps * 1000),
        "window_start_at": to_iso_ms(window_start_at),
        "window_end_at": to_iso_ms(window_end_at),
        "frames": {
            "joints": pred["joints"],
            "pose_rel": pose_rel,
            "root": pred["root"],
            "frame_valid": pred["frame_valid"],
        },
    }


def window_start(window_end_at: datetime, pred: dict[str, Any]) -> datetime:
    """윈도 시작 = 종료 - 프레임수/fps. 모델은 시각을 모르므로 종료시각에서 역산한다."""
    seconds = len(pred["pose_rel"]) / pred["fps"]
    return window_end_at - timedelta(seconds=seconds)


def to_iso_ms(value: datetime) -> str:
    """계약: 시각은 ms 정밀도. 마이크로초를 그대로 보내면 멱등키 정밀도가 흔들린다."""
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds")


# ── NORMAL 전송 절감 ────────────────────────────────────────────────────────
# 10초 윈도를 상시 추론하면 NORMAL이 폭증한다. 계약: 클래스가 바뀌면 즉시,
# 같은 클래스가 이어지면 5분에 1건만 보낸다. 비정상 이벤트는 절대 거르지 않는다.

_lock = threading.Lock()
_last_sent: dict[str, tuple[str, datetime]] = {}


def should_send(
    device_id: str,
    event_type: EventType,
    activity_class: str,
    now: datetime,
) -> bool:
    if event_type is not EventType.NORMAL:
        return True

    with _lock:
        previous = _last_sent.get(device_id)
        if previous is not None:
            last_class, last_at = previous
            interval = timedelta(seconds=settings.notifi_normal_interval_seconds)
            if last_class == activity_class and now - last_at < interval:
                return False
        _last_sent[device_id] = (activity_class, now)
    return True


def reset_throttle() -> None:
    """테스트용 — 프로세스 전역 상태를 비운다."""
    with _lock:
        _last_sent.clear()
