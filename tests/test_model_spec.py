"""모델 계약(ModelSpec)과 그걸 노출하는 /spec.

모델은 계속 갱신된다. 여기 테스트는 **새 모델이 왔을 때 하위 가정이 조용히 어긋나는 것**을
막는 게 목적이다 — 어긋나면 부팅에서 터져야지, 응급 판정을 틀리게 내면 안 된다.
notifi_ai 없이 돈다(ModelSpec은 순수 타입).
"""
import dataclasses

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from main import app
from tests.conftest import FAKE_SPEC

KEY = settings.spring_internal_key


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


# ── 계약 검증 ───────────────────────────────────────────────────────────────

def test_window_seconds_derives_from_frames_and_fps():
    """데몬 윈도 길이가 여기서 나온다 — 304/30 = 10.13초"""
    assert FAKE_SPEC.window_seconds == pytest.approx(10.133, abs=1e-3)


def test_action_risk_mapping_must_cover_every_action():
    """행동이 늘었는데 매핑이 그대로면 IndexError가 추론 도중에 터진다."""
    with pytest.raises(ValueError, match="위험 매핑"):
        dataclasses.replace(FAKE_SPEC, action_labels=FAKE_SPEC.action_labels + ("action17",))


def test_risk_levels_must_stay_three():
    """pipeline._RISK_ID_TO_EVENT가 safe/warning/danger 3칸 고정이다."""
    with pytest.raises(ValueError, match="위험 등급"):
        dataclasses.replace(FAKE_SPEC, risk_labels=("safe", "warning", "danger", "critical"))


def test_action_risk_id_must_stay_in_range():
    broken = (0,) * 16 + (3,)  # 위험 등급은 0~2뿐
    with pytest.raises(ValueError, match="범위"):
        dataclasses.replace(FAKE_SPEC, action_to_risk=broken)


@pytest.mark.parametrize("field,value", [("frames", 0), ("fps", 0.0), ("links", 0)])
def test_input_shape_must_be_positive(field, value):
    with pytest.raises(ValueError, match="입력 형상"):
        dataclasses.replace(FAKE_SPEC, **{field: value})


def test_spec_is_immutable():
    """로드 후 누가 바꾸면 추론 결과 라벨이 도중에 달라진다."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        FAKE_SPEC.joint_schema = "smpl-24"  # type: ignore[misc]


# ── /spec 노출 ──────────────────────────────────────────────────────────────

def test_spec_requires_key(client):
    assert client.get("/internal/model/spec").status_code == 401


def test_spec_returns_503_without_model(client):
    """모델이 없으면 계약도 없다 — 빈 응답으로 있는 척하지 않는다."""
    res = client.get("/internal/model/spec", headers={"X-Internal-Key": KEY})
    assert res.status_code == 503


def test_spec_body_carries_downstream_contract(client, monkeypatch):
    class StubRuntime:
        spec = FAKE_SPEC

    monkeypatch.setattr(app.state, "model_runtime", StubRuntime(), raising=False)
    body = client.get("/internal/model/spec", headers={"X-Internal-Key": KEY}).json()

    # 앱 렌더러·Spring 적재·데몬 윈도가 각각 의존하는 값
    assert body["joint_schema"] == "smpl-22"
    assert len(body["joint_names"]) == 22
    assert len(body["action_labels"]) == 17
    assert body["risk_labels"] == ["safe", "warning", "danger"]
    assert body["frames"] == 304 and body["fps"] == 30.0
    assert body["window_seconds"] == pytest.approx(10.133, abs=1e-3)


def test_joint_names_must_not_be_empty():
    """관절 이름은 앱이 렌더 순서를 정하는 근거다 — 비면 그릴 수 없다."""
    with pytest.raises(ValueError, match="관절 목록"):
        dataclasses.replace(FAKE_SPEC, joint_names=())


def test_adapter_refuses_to_name_unknown_joint_count():
    """관절 수가 바뀌면 스키마 이름을 지어내지 않는다.

    지어내면 앱이 "아는 스키마"로 착각하고 엉뚱한 뼈대로 그린다. 모르면 부팅에서 멈추고,
    adapter._JOINT_SCHEMAS에 새 스키마를 등록하게 만드는 편이 안전하다.
    """
    pytest.importorskip("notifi_ai")
    from app.model import adapter

    assert adapter._JOINT_SCHEMAS.get(22) == "smpl-22"
    assert adapter._JOINT_SCHEMAS.get(24) is None
