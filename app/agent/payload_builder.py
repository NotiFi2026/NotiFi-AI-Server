"""Spring 내부 API 요청 payload 조립."""
from typing import Any

# Spring은 risk_probability를 @Digits(integer=1, fraction=3)로 받는다 — 4자리 이상이면 400
_PROBABILITY_DIGITS = 3


def build_sensing_event_payload(state: dict[str, Any]) -> dict[str, Any]:
    """I1 payload 조립 — spec 11.4 기준."""
    features: dict[str, Any] = {}
    if state.get("label"):
        features["label"] = state["label"]
    if state.get("context_features"):
        features["context_features"] = state["context_features"]
    if state.get("features"):
        features.update(state["features"])

    # 세부 행동은 정식 필드로 보낸다 — features에만 있으면 Spring이 집계에 못 쓴다
    activity_class = features.pop("activity_class", None)
    confidence = round(float(state["confidence"]), _PROBABILITY_DIGITS)

    return {
        "care_target_id": state["care_target_id"],
        "device_id": state.get("device_id"),
        "event_type": state["event_type"],
        "activity_class": activity_class,
        "risk_probability": confidence,
        "anomaly_score": None,
        "trend_score": None,
        "sensor_status": "OK",
        "model_version": state["model_version"],
        "features": features or None,
        "detected_at": state["detected_at"],
        "risk_score": state["risk_score"],
        "risk_level": state["risk_level"].upper(),
        "score_breakdown": {
            "confidence": confidence,
            "source": "model_risk_output",
        },
    }
