"""Spring 내부 API 요청 payload 조립."""
from typing import Any

from app.model.pipeline import round_probability


def build_sensing_event_payload(state: dict[str, Any]) -> dict[str, Any]:
    """I1 payload 조립 — spec 11.4 기준."""
    features: dict[str, Any] = {}
    if state.get("label"):
        features["label"] = state["label"]
    if state.get("context_features"):
        features["context_features"] = state["context_features"]
    if state.get("features"):
        features.update(state["features"])

    confidence = round_probability(state["confidence"])

    return {
        "care_target_id": state["care_target_id"],
        "device_id": state.get("device_id"),
        "event_type": state["event_type"],
        # 세부 행동은 정식 필드로 보낸다 — features에만 있으면 Spring이 집계에 못 쓴다
        "activity_class": state.get("activity_class"),
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
