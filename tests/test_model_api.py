"""모델 엔드포인트 — 모델을 로드하지 않은 상태에서의 계약."""
import sys

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from main import app

KEY = settings.spring_internal_key


@pytest.fixture
def client():
    # lifespan이 돌아도 NOTIFI_MODEL_ENABLED=false라 모델은 로드되지 않는다
    with TestClient(app) as c:
        yield c


def test_boots_without_notifi_ai(monkeypatch):
    """모델 패키지가 없어도 앱은 import·기동된다.

    타입 힌트용 top-level import가 되살아나면 여기서 잡힌다 — 실제로 났던 회귀다.
    """
    for name in [m for m in sys.modules if m == "main" or m.startswith("app.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "notifi_ai", None)

    import main as reloaded

    assert reloaded.app is not None


def test_health_503_when_model_enabled_but_not_loaded(client, monkeypatch):
    monkeypatch.setattr(settings, "notifi_model_enabled", True)
    assert client.get("/internal/model/health").status_code == 503


def test_health_reports_disabled(client):
    body = client.get("/internal/model/health").json()
    assert body == {"loaded": False, "enabled": False}


def test_predict_rejects_wrong_key(client):
    res = client.post(
        "/internal/model/devices/home-001/predict",
        headers={"X-Internal-Key": "wrong"},
        files={"file": ("q.npz", b"x")},
    )
    assert res.status_code == 401


def test_predict_rejects_empty_configured_key(client, monkeypatch):
    """키가 설정되지 않았으면 헤더 없는 요청도 통과시키면 안 된다."""
    monkeypatch.setattr(settings, "spring_internal_key", "")
    res = client.post(
        "/internal/model/devices/home-001/predict",
        files={"file": ("q.npz", b"x")},
    )
    assert res.status_code == 401


@pytest.mark.parametrize("device_id", ["..", r"..\..\Windows", "a/b", "x" * 65, "bad id"])
def test_predict_rejects_bad_device_id(client, device_id):
    """device_id는 경로 세그먼트로 쓰이므로 경로 탈출 문자를 막는다."""
    res = client.post(
        f"/internal/model/devices/{device_id}/predict",
        headers={"X-Internal-Key": KEY},
        files={"file": ("q.npz", b"x")},
    )
    assert res.status_code in (400, 404)


def test_predict_503_when_runtime_missing(client):
    res = client.post(
        "/internal/model/devices/home-001/predict",
        headers={"X-Internal-Key": KEY},
        files={"file": ("q.npz", b"x")},
    )
    assert res.status_code == 503


class StuckRuntime:
    """추론이 멈춘 런타임 흉내 — 모델 없이 health 판정만 검증한다."""

    def __init__(self, inflight: float) -> None:
        self._inflight = inflight

    def inflight_seconds(self):
        return self._inflight

    def last_success_age_seconds(self):
        return None

    def describe(self):
        raise AssertionError("멈춤이면 describe까지 가지 않아야 한다")


def test_health_503_when_inference_stuck(client, monkeypatch):
    """멈춘 CUDA 연산은 되돌릴 수 없다 — 재시작을 유도할 유일한 신호가 이 503이다."""
    monkeypatch.setattr(settings, "notifi_inference_stuck_seconds", 60.0)
    client.app.state.model_runtime = StuckRuntime(inflight=120.0)
    try:
        assert client.get("/internal/model/health").status_code == 503
    finally:
        client.app.state.model_runtime = None


def test_health_ok_while_inference_is_merely_running(client, monkeypatch):
    """진행 중이라고 아프다고 하면 안 된다 — 한도를 넘겨야 멈춤이다."""
    monkeypatch.setattr(settings, "notifi_inference_stuck_seconds", 60.0)

    class Running(StuckRuntime):
        def describe(self):
            return {
                "model_name": "NotiFi_AI_v1", "device": "cpu",
                "actions": [], "risks": [], "artifact_dir": "/x", "metadata": {},
            }

    client.app.state.model_runtime = Running(inflight=1.0)
    try:
        assert client.get("/internal/model/health").status_code == 200
    finally:
        client.app.state.model_runtime = None


def test_ingest_rejects_wrong_key(client):
    res = client.post(
        "/internal/model/devices/home-001/ingest",
        headers={"X-Internal-Key": "wrong"},
        files={"file": ("q.npz", b"x")},
        data={"care_target_id": "1"},
    )
    assert res.status_code == 401


def test_ingest_requires_care_target_id(client):
    res = client.post(
        "/internal/model/devices/home-001/ingest",
        headers={"X-Internal-Key": KEY},
        files={"file": ("q.npz", b"x")},
    )
    assert res.status_code == 422


def test_ingest_rejects_bad_device_id(client):
    res = client.post(
        r"/internal/model/devices/..\..\x/ingest",
        headers={"X-Internal-Key": KEY},
        files={"file": ("q.npz", b"x")},
        data={"care_target_id": "1"},
    )
    assert res.status_code in (400, 404)


def test_ingest_503_when_runtime_missing(client):
    res = client.post(
        "/internal/model/devices/home-001/ingest",
        headers={"X-Internal-Key": KEY},
        files={"file": ("q.npz", b"x")},
        data={"care_target_id": "1"},
    )
    assert res.status_code == 503


def test_agent_run_rejects_wrong_key(client):
    """기존 엔드포인트도 같은 인증 헬퍼를 쓴다.

    본문 검증이 인증보다 먼저라 유효한 본문을 보내야 인증 경로에 닿는다(선존재 동작).
    """
    res = client.post(
        "/internal/agent/run",
        headers={"X-Internal-Key": "wrong"},
        json={
            "care_target_id": 1,
            "event_type": "FALL",
            "label": "fall_from_standing",
            "risk_level": "danger",
            "confidence": 0.9,
            "risk_score": 90,
            "model_version": "NotiFi_AI_v1",
            "detected_at": "2026-08-12T03:22:00.123Z",
        },
    )
    assert res.status_code == 401
