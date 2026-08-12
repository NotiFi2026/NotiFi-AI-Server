"""추론 결과 → Spring 계약 변환. 모델·GPU 없이 돈다."""
from datetime import datetime, timedelta, timezone

import pytest

from app.agent.payload_builder import build_sensing_event_payload
from app.agent.schemas import EventType, RiskLevel
from app.model import pipeline

NOW = datetime(2026, 8, 12, 3, 22, 0, 123456, tzinfo=timezone.utc)

# 모델 패키지의 ACTION_LABELS 순서 — 앞 9개 safe, 다음 3개 warning, 마지막 5개 danger
ACTIONS = [
    "walking", "standing_still", "sitting_still", "lying_still", "lie_to_stand",
    "stand_to_lie_normal", "absence", "sit_to_stand", "stand_to_sit",
    "unstable_walking", "stumble_recover", "bed_exit_failed",
    "fall_from_standing", "fall_while_walking", "bed_exit_fall", "bed_fall", "chair_exit_fall",
]


def make_pred(
    action_label="walking",
    action_risk_id=0,
    risk_label="safe",
    risk_probability=(1.0, 0.0, 0.0),
    low_quality=False,
    frames=4,
):
    return {
        "action_label": action_label,
        "action_risk_id": action_risk_id,
        "action_probability": [0.9876543] + [0.0] * 16,
        "risk_label": risk_label,
        "risk_probability": list(risk_probability),
        "quality": {"risk_confidence": 0.98765432, "low_quality": low_quality, "active_links": 3},
        "model_name": "NotiFi_AI_v1",
        "pose_rel": [[[0.0, 0.0, 0.0]] * 22] * frames,
        "root": [[0.0, 0.0, 0.0]] * frames,
        "frame_valid": [True] * frames,
        "joints": [f"j{i}" for i in range(22)],
        "fps": 30.0,
    }


@pytest.fixture(autouse=True)
def clear_throttle():
    pipeline.reset_throttle()


# ── 변환 ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("index,label", list(enumerate(ACTIONS)))
def test_event_type_mapping_covers_all_17_actions(index, label):
    risk_id = 0 if index < 9 else 1 if index < 12 else 2
    expected = [EventType.NORMAL, EventType.ANOMALY, EventType.FALL][risk_id]

    result = pipeline.to_model_result(
        make_pred(action_label=label, action_risk_id=risk_id), 1, None, NOW
    )

    assert result.event_type is expected
    # 정식 필드로 실린다 — features dict의 매직 키가 아니다
    assert result.activity_class == label.upper()


def test_risk_score_formula():
    """확정 산식: round(50 x P(warning) + 100 x P(danger))"""
    result = pipeline.to_model_result(
        make_pred(risk_probability=(0.1, 0.3, 0.6)), 1, None, NOW
    )
    assert result.risk_score == round(50 * 0.3 + 100 * 0.6)


def test_risk_score_stays_within_spring_bounds():
    result = pipeline.to_model_result(
        make_pred(risk_probability=(0.0, 0.0, 1.0)), 1, None, NOW
    )
    assert 0 <= result.risk_score <= 100


def test_low_quality_downgrades_danger_to_warning():
    """링크 부족·프레임 결손만으로 자동 경보를 울리지 않는다."""
    result = pipeline.to_model_result(
        make_pred(risk_label="danger", risk_probability=(0.0, 0.1, 0.9), low_quality=True),
        1, None, NOW,
    )
    assert result.risk_level is RiskLevel.WARNING
    assert result.features["degraded_from"] == "danger"
    assert result.features["quality"]["low_quality"] is True


def test_low_quality_does_not_upgrade_safe():
    result = pipeline.to_model_result(make_pred(low_quality=True), 1, None, NOW)
    assert result.risk_level is RiskLevel.SAFE
    assert "degraded_from" not in result.features


def test_confidence_rounded_to_three_decimals():
    """Spring risk_probability는 @Digits(fraction=3) — 4자리 이상이면 400."""
    result = pipeline.to_model_result(make_pred(), 1, None, NOW)
    assert result.confidence == 0.988


# ── I1 페이로드 ─────────────────────────────────────────────────────────────

def test_payload_sends_activity_class_as_field_not_feature():
    from app.agent import escalation_agent

    model_result = pipeline.to_model_result(
        make_pred(action_label="fall_from_standing", action_risk_id=2, risk_label="danger"),
        1, None, NOW,
    )
    payload = build_sensing_event_payload(escalation_agent.initial_state(model_result))

    assert payload["activity_class"] == "FALL_FROM_STANDING"
    assert payload["event_type"] == "FALL"
    assert payload["risk_level"] == "DANGER"
    assert "activity_class" not in (payload["features"] or {})


def test_unknown_risk_label_raises_contract_error():
    """모델이 계약 밖 값을 내면 입력 오류(400)가 아니라 서버 오류로 올라가야 한다."""
    from app.model.errors import ModelContractError

    with pytest.raises(ModelContractError):
        pipeline.to_model_result(make_pred(risk_label="explosive"), 1, None, NOW)


def test_naive_datetime_is_rejected():
    """naive 시각을 astimezone()에 넣으면 로컬시간으로 해석돼 9시간 어긋난다."""
    naive = datetime(2026, 8, 12, 3, 22, 0)
    with pytest.raises(ValueError):
        pipeline.to_iso_ms(naive)


def test_model_result_rejects_naive_detected_at():
    """계약을 스키마에서 강제한다 — 두 진입점(agent/run, ingest)을 한 곳에서 덮는다."""
    from app.agent.schemas import EventType as ET, ModelResult, RiskLevel as RL

    with pytest.raises(ValueError):
        ModelResult(
            care_target_id=1,
            event_type=ET.FALL,
            label="fall_from_standing",
            risk_level=RL.DANGER,
            confidence=0.9,
            risk_score=90,
            model_version="NotiFi_AI_v1",
            detected_at=datetime(2026, 8, 12, 3, 22, 0),
        )


def test_payload_detected_at_is_millisecond_precision():
    from app.agent import escalation_agent

    model_result = pipeline.to_model_result(make_pred(), 1, None, NOW)
    payload = build_sensing_event_payload(escalation_agent.initial_state(model_result))

    assert payload["detected_at"] == "2026-08-12T03:22:00.123+00:00"


def test_payload_risk_probability_has_at_most_three_decimals():
    from app.agent import escalation_agent

    model_result = pipeline.to_model_result(make_pred(), 1, None, NOW)
    payload = build_sensing_event_payload(escalation_agent.initial_state(model_result))

    assert str(payload["risk_probability"])[::-1].find(".") <= 3


# ── I5 포즈 클립 ────────────────────────────────────────────────────────────

def test_pose_clip_payload_shape():
    pred = make_pred(frames=304)
    end = NOW
    start = pipeline.window_start(end, pred)
    payload = pipeline.build_pose_clip_payload(pred, start, end)

    assert payload["joint_schema"] == "smpl-22"
    assert payload["fps"] == 30
    assert payload["frame_count"] == 304
    assert payload["duration_ms"] == 10133
    assert len(payload["frames"]["joints"]) == 22
    # frames는 top-level JSON 객체여야 한다 (Spring Map<String,Object>)
    assert isinstance(payload["frames"], dict)
    assert payload["window_start_at"] < payload["window_end_at"]


def test_window_start_derived_from_frame_count():
    pred = make_pred(frames=304)
    assert pipeline.window_start(NOW, pred) == NOW - timedelta(seconds=304 / 30.0)


# ── NORMAL 절감 ─────────────────────────────────────────────────────────────

def test_first_normal_is_sent():
    assert pipeline.should_send("d1", EventType.NORMAL, "WALKING", NOW) is True


def test_same_class_within_interval_is_skipped():
    pipeline.mark_sent("d1", "WALKING", NOW)
    assert pipeline.should_send("d1", EventType.NORMAL, "WALKING", NOW + timedelta(seconds=60)) is False


def test_class_change_is_sent_immediately():
    pipeline.mark_sent("d1", "WALKING", NOW)
    assert pipeline.should_send("d1", EventType.NORMAL, "SITTING_STILL", NOW + timedelta(seconds=1)) is True


def test_same_class_after_interval_is_sent():
    pipeline.mark_sent("d1", "WALKING", NOW)
    assert pipeline.should_send("d1", EventType.NORMAL, "WALKING", NOW + timedelta(seconds=301)) is True


@pytest.mark.parametrize("event_type", [EventType.FALL, EventType.ANOMALY])
def test_abnormal_events_are_never_throttled(event_type):
    for _ in range(3):
        assert pipeline.should_send("d1", event_type, "FALL_FROM_STANDING", NOW) is True


def test_throttle_is_per_device():
    pipeline.mark_sent("d1", "WALKING", NOW)
    assert pipeline.should_send("d2", EventType.NORMAL, "WALKING", NOW) is True


def test_should_send_does_not_record_by_itself():
    """판정만으로 기록되면 Spring 적재가 실패해도 다음 윈도가 막혀 2건이 유실된다."""
    assert pipeline.should_send("d1", EventType.NORMAL, "WALKING", NOW) is True
    # 적재가 실패해 mark_sent를 부르지 않은 상황 — 다음 윈도는 여전히 보내야 한다
    assert pipeline.should_send("d1", EventType.NORMAL, "WALKING", NOW + timedelta(seconds=10)) is True


def test_tracked_devices_are_bounded():
    """device_id는 호출자가 정하는 문자열이라 상한이 없으면 무한히 쌓인다."""
    for i in range(pipeline._MAX_TRACKED_DEVICES + 50):
        pipeline.mark_sent(f"device-{i}", "WALKING", NOW)
    assert len(pipeline._last_sent) <= pipeline._MAX_TRACKED_DEVICES


# ── 에이전트 연계 ───────────────────────────────────────────────────────────
# 합성 CSI로는 모델이 danger를 내지 않아(absence로 판정) E2E로 태울 수 없다.

@pytest.mark.asyncio
async def test_agent_does_not_repost_i1_when_prefetched(monkeypatch):
    """파이프라인이 I5를 위해 이미 I1을 보냈으면 에이전트는 다시 보내지 않는다."""
    from app.agent import escalation_agent
    from app.clients import spring_client

    called = []

    async def fail_if_called(payload):
        called.append(payload)
        return {}

    monkeypatch.setattr(spring_client, "send_sensing_event", fail_if_called)

    model_result = pipeline.to_model_result(make_pred(), 1, None, NOW)
    state = await escalation_agent.run(
        model_result,
        prefetched={"sensing_event_id": 40, "escalation_triggered": False, "escalation_id": None},
    )

    assert called == []
    assert state["sensing_event_id"] == 40


@pytest.mark.asyncio
async def test_danger_without_escalation_id_does_not_call_i2(monkeypatch):
    """멱등 재적재로 escalation_id가 없으면 /escalations/None/steps 로 POST하면 안 된다."""
    from app.agent import escalation_agent
    from app.clients import spring_client

    steps = []

    async def record(*args, **kwargs):
        steps.append(kwargs or args)
        return {}

    monkeypatch.setattr(spring_client, "record_escalation_step", record)

    model_result = pipeline.to_model_result(
        make_pred(action_label="fall_from_standing", action_risk_id=2,
                  risk_label="danger", risk_probability=(0.0, 0.1, 0.9)),
        1, None, NOW,
    )
    await escalation_agent.run(
        model_result,
        prefetched={"sensing_event_id": 41, "escalation_triggered": True, "escalation_id": None},
    )

    assert steps == []
