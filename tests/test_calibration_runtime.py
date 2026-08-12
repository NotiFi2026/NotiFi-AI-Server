"""캘리브레이션 런타임 — 실제 레지스트리(임시 디렉터리) + 가짜 모델.

아티팩트·GPU는 필요 없다. notifi_ai가 없는 환경에서는 통째로 건너뛴다.
"""
import io

import numpy as np
import pytest

pytest.importorskip("notifi_ai")

from notifi_ai.registry import DeviceRegistry  # noqa: E402

from app.model.runtime import ModelRuntime  # noqa: E402

BOARDS = {"rx_id": "RX-1", "tx1_id": "T1", "tx2_id": "T2", "tx3_id": "T3"}


class FakeProfile:
    def __init__(self, device_id: str) -> None:
        self.device_id = device_id

    def save(self, path):
        path.write_bytes(b"profile")
        return path

    def summary(self) -> dict:
        return {
            "device_id": self.device_id,
            "baseline_link_valid": [True, True, True],
            "link_coverage": [0.9, 0.9, 0.9],
            "support_action_counts": [0] * 17,
        }


class FakeModel:
    def fit_calibration(self, device_id, absence, support):
        return FakeProfile(device_id)


def make_calibration_npz(trials: int = 2) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        absence_csi=np.zeros((trials, 8, 3, 114, 2), np.float32),
        absence_mask=np.ones((trials, 8, 3), bool),
    )
    return buffer.getvalue()


@pytest.fixture
def runtime(tmp_path):
    return ModelRuntime(FakeModel(), DeviceRegistry(tmp_path))


def register(runtime, **overrides) -> dict:
    return runtime.register_device({"device_id": "care-1", **BOARDS, **overrides})


# ── 재등록 시 프로필 폐기 ───────────────────────────────────────────────────

def test_reregister_with_different_boards_discards_calibration(runtime, tmp_path):
    """보드가 바뀌면 이전 baseline은 무효다 — 그대로 두면 낙상 판정이 조용히 틀어진다."""
    register(runtime)
    runtime.fit_calibration_npz("care-1", make_calibration_npz())
    assert (tmp_path / "care-1" / "calibration.pt").exists()

    result = register(runtime, tx2_id="T2-NEW")

    assert result["calibration_invalidated"] is True
    assert not (tmp_path / "care-1" / "calibration.pt").exists()


def test_reregister_with_same_boards_keeps_calibration(runtime, tmp_path):
    """오타 수정·펌웨어 갱신 같은 재등록까지 재캘리브레이션을 강요하지 않는다."""
    register(runtime)
    runtime.fit_calibration_npz("care-1", make_calibration_npz())

    result = register(runtime, firmware_version="v2.0")

    assert result["calibration_invalidated"] is False
    assert (tmp_path / "care-1" / "calibration.pt").exists()


def test_first_registration_reports_no_invalidation(runtime):
    assert register(runtime)["calibration_invalidated"] is False


# ── 원자적 저장 ─────────────────────────────────────────────────────────────

def test_failed_save_keeps_previous_profile(runtime, tmp_path):
    """저장이 중단돼도 기존 프로필이 살아 있어야 한다 — 깨지면 그 가구는 영구 400이다."""
    register(runtime)
    runtime.fit_calibration_npz("care-1", make_calibration_npz())
    target = tmp_path / "care-1" / "calibration.pt"
    original = target.read_bytes()

    class ExplodingProfile(FakeProfile):
        def save(self, path):
            path.write_bytes(b"partial")
            raise OSError("disk full")

    runtime._model.fit_calibration = lambda d, a, s: ExplodingProfile(d)
    with pytest.raises(OSError):
        runtime.fit_calibration_npz("care-1", make_calibration_npz())

    assert target.read_bytes() == original
    assert not (tmp_path / "care-1" / "calibration.pt.tmp").exists()


def test_summary_reports_trial_counts(runtime):
    register(runtime)
    summary = runtime.fit_calibration_npz("care-1", make_calibration_npz(trials=3))
    assert summary["absence_trials"] == 3
    assert summary["support_trials"] == 0


# ── 삭제 ────────────────────────────────────────────────────────────────────

def test_delete_removes_registration_and_profile(runtime, tmp_path):
    register(runtime)
    runtime.fit_calibration_npz("care-1", make_calibration_npz())

    assert runtime.delete_device("care-1") is True
    assert not (tmp_path / "care-1").exists()
    assert runtime.device_status("care-1")["registered"] is False


def test_delete_unknown_device_returns_false(runtime):
    assert runtime.delete_device("care-999") is False
