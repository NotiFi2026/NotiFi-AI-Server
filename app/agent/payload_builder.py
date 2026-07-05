"""Spring 내부 API 요청 payload 조립."""
from typing import Any


def build_sensing_event_payload(state: dict[str, Any]) -> dict[str, Any]:
    """I1 payload 조립 — spec 11.4 기준."""
    features: dict[str, Any] = {}
    if state.get("label"):
        features["label"] = state["label"]
    if state.get("context_features"):
        features["context_features"] = state["context_features"]

    return {
        "care_target_id": state["care_target_id"],
        "device_id": state.get("device_id"),
        "event_type": state["event_type"],
        "risk_probability": state["confidence"],
        "anomaly_score": None,
        "trend_score": None,
        "sensor_status": "OK",
        "model_version": state["model_version"],
        "features": features or None,
        "detected_at": state["detected_at"],
        "risk_score": state["risk_score"],
        "risk_level": state["risk_level"].upper(),
        "score_breakdown": {
            "confidence": state["confidence"],
            "source": "model_risk_output",
        },
    }
