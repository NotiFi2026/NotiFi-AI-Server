"""디바이스 등록·캘리브레이션 API. 모델·GPU 없이 돈다."""
import pytest
from fastapi.testclient import TestClient

from app.api.model_routes import devices
from app.api.model_routes.devices import calibration_warnings
from app.config import settings
from app.model import pipeline
from main import app

KEY = settings.spring_internal_key


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ── device_id 파생 ──────────────────────────────────────────────────────────

def test_device_id_round_trip():
    assert pipeline.device_id_for(42) == "care-42"
    assert pipeline.care_target_id_from("care-42") == 42


@pytest.mark.parametrize("device_id", ["home-001", "care-", "care-abc", "cares-1", ""])
def test_care_target_id_from_non_conforming_is_none(device_id):
    """규약 밖 ID는 None — 수동 등록된 레거시 디바이스를 막지 않는다."""
    assert pipeline.care_target_id_from(device_id) is None


# ── 인증·상태 ───────────────────────────────────────────────────────────────

def test_register_requires_key(client):
    res = client.post("/internal/model/devices", json={
        "care_target_id": 1, "rx_id": "RX", "tx1_id": "T1", "tx2_id": "T2", "tx3_id": "T3",
    })
    assert res.status_code == 401


def test_calibrate_requires_key(client):
    res = client.post(
        "/internal/model/devices/care-1/calibrate",
        files={"file": ("c.npz", b"x")},
    )
    assert res.status_code == 401


def test_register_rejects_non_positive_care_target(client):
    res = client.post("/internal/model/devices", headers={"X-Internal-Key": KEY}, json={
        "care_target_id": 0, "rx_id": "RX", "tx1_id": "T1", "tx2_id": "T2", "tx3_id": "T3",
    })
    assert res.status_code == 422


def test_register_503_when_runtime_missing(client):
    res = client.post("/internal/model/devices", headers={"X-Internal-Key": KEY}, json={
        "care_target_id": 1, "rx_id": "RX", "tx1_id": "T1", "tx2_id": "T2", "tx3_id": "T3",
    })
    assert res.status_code == 503


def test_calibrate_503_when_runtime_missing(client):
    res = client.post(
        "/internal/model/devices/care-1/calibrate",
        headers={"X-Internal-Key": KEY},
        files={"file": ("c.npz", b"x")},
    )
    assert res.status_code == 503


def test_device_status_rejects_bad_device_id(client):
    res = client.get(
        r"/internal/model/devices/..\..\x",
        headers={"X-Internal-Key": KEY},
    )
    assert res.status_code in (400, 404)


# ── ingest ID 불일치 ────────────────────────────────────────────────────────

def test_ingest_rejects_care_target_mismatch(client):
    """device_id가 가리키는 노인과 care_target_id가 다르면 거부한다.

    통과시키면 낙상 이벤트가 엉뚱한 노인에게 적재된다.
    """
    res = client.post(
        "/internal/model/devices/care-5/ingest",
        headers={"X-Internal-Key": KEY},
        files={"file": ("q.npz", b"x")},
        data={"care_target_id": "7"},
    )
    assert res.status_code == 400
    assert "care_target" in res.json()["detail"]


def test_ingest_allows_matching_ids(client):
    """일치하면 통과해 다음 단계(런타임 미로드 503)로 간다."""
    res = client.post(
        "/internal/model/devices/care-5/ingest",
        headers={"X-Internal-Key": KEY},
        files={"file": ("q.npz", b"x")},
        data={"care_target_id": "5"},
    )
    assert res.status_code == 503


def test_ingest_allows_legacy_device_id(client):
    """규약 밖 device_id는 검사하지 않는다 — home-001 같은 기존 등록을 막지 않는다."""
    res = client.post(
        "/internal/model/devices/home-001/ingest",
        headers={"X-Internal-Key": KEY},
        files={"file": ("q.npz", b"x")},
        data={"care_target_id": "5"},
    )
    assert res.status_code == 503


# ── 품질 경고 판정 ──────────────────────────────────────────────────────────

HEALTHY = {
    "baseline_link_valid": [True, True, True],
    "link_coverage": [0.9, 0.8, 0.7],
    "absence_trials": 12,
    "support_action_counts": [2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
}


def test_warns_when_fewer_than_two_links_alive():
    warnings = calibration_warnings({
        **HEALTHY,
        "baseline_link_valid": [True, False, False],
        "link_coverage": [0.9, 0.0, 0.0],
    })
    assert any("최소 2개" in w for w in warnings)


def test_no_warnings_when_installation_is_healthy():
    assert calibration_warnings(HEALTHY) == []


def test_warns_when_absence_trials_are_insufficient():
    """README에 12회를 계약으로 써놓고 검증하지 않으면 계약이 아니다."""
    warnings = calibration_warnings({**HEALTHY, "absence_trials": 3})
    assert any("무인 트라이얼 3회" in w for w in warnings)


def test_warns_when_basic_actions_are_missing():
    warnings = calibration_warnings({**HEALTHY, "support_action_counts": [0] * 17})
    assert any("기본 동작" in w for w in warnings)


def test_absence_count_absent_is_not_warned():
    """저장된 프로필 요약에는 트라이얼 수가 없다 — 없다고 경고하면 오탐이다."""
    summary = {k: v for k, v in HEALTHY.items() if k != "absence_trials"}
    assert calibration_warnings(summary) == []


# ── 삭제 ────────────────────────────────────────────────────────────────────

def test_delete_requires_key(client):
    assert client.delete("/internal/model/devices/care-1").status_code == 401


def test_delete_503_when_runtime_missing(client):
    res = client.delete("/internal/model/devices/care-1", headers={"X-Internal-Key": KEY})
    assert res.status_code == 503


def test_delete_rejects_bad_device_id(client):
    res = client.delete(r"/internal/model/devices/..\..\x", headers={"X-Internal-Key": KEY})
    assert res.status_code in (400, 404)


# ── 동시 캘리브레이션 제한 ──────────────────────────────────────────────────

def test_second_concurrent_calibration_is_rejected(client):
    """업로드·압축해제는 모델 락 밖이라 여기서 막지 않으면 메모리가 밀린다."""
    class NeverCalled:
        def fit_calibration_npz(self, *args):
            raise AssertionError("슬롯이 찼으면 학습까지 가면 안 된다")

    client.app.state.model_runtime = NeverCalled()
    devices._calibration_slot.acquire()
    try:
        res = client.post(
            "/internal/model/devices/care-1/calibrate",
            headers={"X-Internal-Key": KEY},
            files={"file": ("c.npz", b"x")},
        )
        assert res.status_code == 503
        assert "another calibration" in res.json()["detail"].lower()
    finally:
        devices._calibration_slot.release()
        client.app.state.model_runtime = None


def test_calibration_slot_released_after_failure(client):
    """실패해도 슬롯이 남으면 이후 모든 캘리브레이션이 503이 된다."""
    client.post(
        "/internal/model/devices/care-1/calibrate",
        headers={"X-Internal-Key": KEY},
        files={"file": ("c.npz", b"x")},
    )
    assert devices._calibration_slot.acquire(blocking=False)
    devices._calibration_slot.release()
