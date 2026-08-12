"""추론 락 대기 한도와 멈춤 관측.

notifi_ai가 없는 환경(부팅은 되어야 한다)에서는 통째로 건너뛴다.
아티팩트·GPU는 필요 없다 — ModelRuntime 생성자가 (model, registry)를 받으므로 가짜로 채운다.
"""
import io
import threading
import time

import numpy as np
import pytest

pytest.importorskip("notifi_ai")

from app.config import settings  # noqa: E402
from app.model.errors import InferenceBusyError  # noqa: E402
from app.model.runtime import ModelRuntime  # noqa: E402


def make_npz() -> bytes:
    """load_query_npz가 읽는 최소 쿼리 NPZ — csi·link_mask 키만 있으면 된다."""
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        csi=np.zeros((8, 3, 114, 2), np.float32),
        link_mask=np.ones((8, 3), bool),
    )
    return buffer.getvalue()


class FakePrediction:
    action_id = 0
    action_label = "walking"
    risk_label = "safe"
    quality = {"low_quality": False}

    def to_dict(self, include_pose: bool = False) -> dict:
        return {"action_label": self.action_label}


class FakeModel:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.started = threading.Event()

    def predict(self, csi, link_mask, calibration=None):
        self.started.set()
        time.sleep(self.delay)
        return FakePrediction()


class FakeRegistry:
    def load_calibration(self, device_id: str):
        return None


@pytest.fixture
def npz():
    return make_npz()


def test_busy_when_lock_held_beyond_timeout(monkeypatch, npz):
    """앞 추론이 길면 뒤 요청은 무한 대기 대신 한도 안에 포기한다."""
    monkeypatch.setattr(settings, "notifi_inference_lock_timeout_seconds", 0.2)
    model = FakeModel(delay=2.0)
    runtime = ModelRuntime(model, FakeRegistry())

    blocker = threading.Thread(target=runtime.predict_npz, args=("home-001", npz))
    blocker.start()
    assert model.started.wait(timeout=5), "첫 추론이 시작되지 않았다"

    started_at = time.monotonic()
    with pytest.raises(InferenceBusyError):
        runtime.predict_npz("home-001", npz)
    waited = time.monotonic() - started_at

    assert waited < 1.0, f"한도(0.2s)를 크게 넘겨 기다렸다: {waited}s"
    blocker.join(timeout=10)


def test_inflight_seconds_reports_running_inference(npz):
    model = FakeModel(delay=1.0)
    runtime = ModelRuntime(model, FakeRegistry())
    assert runtime.inflight_seconds() is None

    worker = threading.Thread(target=runtime.predict_npz, args=("home-001", npz))
    worker.start()
    assert model.started.wait(timeout=5)

    assert runtime.inflight_seconds() is not None
    worker.join(timeout=10)


def test_lock_released_and_state_cleared_after_success(npz):
    runtime = ModelRuntime(FakeModel(), FakeRegistry())
    runtime.predict_npz("home-001", npz)

    assert runtime.inflight_seconds() is None
    assert runtime.last_success_age_seconds() is not None
    # 락이 풀렸으므로 다음 추론이 곧바로 된다
    runtime.predict_npz("home-001", npz)


def test_lock_released_when_inference_raises(npz):
    class Exploding(FakeModel):
        def predict(self, csi, link_mask, calibration=None):
            raise RuntimeError("cuda blew up")

    runtime = ModelRuntime(Exploding(), FakeRegistry())
    with pytest.raises(RuntimeError):
        runtime.predict_npz("home-001", npz)

    assert runtime.inflight_seconds() is None
    # 실패해도 락이 남지 않아야 한다
    runtime._model = FakeModel()
    runtime.predict_npz("home-001", npz)
