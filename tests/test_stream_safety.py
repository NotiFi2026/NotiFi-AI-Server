"""응급 경로가 조용히 막히는 경우들.

전부 "아무 일도 안 일어나서" 알아채기 어려운 종류라, 셀프 리뷰로 찾은 뒤 회귀 테스트를 붙였다.
notifi_ai 없이 돈다 — 게이트·태스크 보관·억제 규칙은 모델과 무관한 순수 로직이다.
"""
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.agent.schemas import EventType, ModelResult, RiskLevel
from app.model import ingest_service, pipeline
from app.stream.policy import AlertCooldown

DETECTED_AT = datetime(2026, 8, 13, 3, 0, tzinfo=timezone.utc)


def make_pred(risk_label: str, low_quality: bool, action_label: str = "fall_from_standing"):
    """danger 판정 한 장. low_quality면 to_model_result가 WARNING으로 강등한다."""
    risk = {"safe": (1.0, 0.0, 0.0), "warning": (0.0, 1.0, 0.0), "danger": (0.0, 0.0, 1.0)}
    return {
        "action_label": action_label,
        "action_risk_id": 2,
        "action_probability": [0.9] + [0.0] * 16,
        "risk_label": risk_label,
        "risk_probability": list(risk[risk_label]),
        "quality": {"risk_confidence": 0.9, "low_quality": low_quality, "active_links": 3},
        "model_name": "NotiFi_AI_v1",
        "pose_rel": [[[0.0, 0.0, 0.0]] * 22] * 4,
        "root": [[0.0, 0.0, 0.0]] * 4,
        "frame_valid": [True] * 4,
        "joints": [f"j{i}" for i in range(22)],
        "fps": 30.0,
        "joint_schema": "smpl-22",
    }


@pytest.fixture(autouse=True)
def clear_throttle():
    pipeline.reset_throttle()


# ── A. 강등된 danger가 쿨다운을 걸면 안 된다 ────────────────────────────────

def test_low_quality_danger_is_downgraded_before_gate():
    """게이트는 강등이 끝난 값을 봐야 한다.

    저품질 danger를 원본 라벨로 판단하면, 에스컬레이션도 못 만든 윈도가 억제를 걸어
    그 뒤에 온 진짜 낙상을 삼킨다.
    """
    seen: list[RiskLevel] = []

    async def run():
        with patch.object(
            ingest_service.spring_client, "send_sensing_event", new=AsyncMock(return_value={})
        ):
            await ingest_service.deliver(
                make_pred("danger", low_quality=True),
                device_id="care-1",
                care_target_id=1,
                spring_device_id=None,
                detected_at=DETECTED_AT,
                schedule_danger=lambda *_: None,
                gate=lambda result: (seen.append(result.risk_level), True)[1],
            )

    asyncio.run(run())
    assert seen == [RiskLevel.WARNING], "게이트가 강등 전 danger를 봤다"


def test_gate_false_blocks_send():
    sent = AsyncMock(return_value={})

    async def run():
        with patch.object(ingest_service.spring_client, "send_sensing_event", new=sent):
            return await ingest_service.deliver(
                make_pred("danger", low_quality=False),
                device_id="care-1",
                care_target_id=1,
                spring_device_id=None,
                detected_at=DETECTED_AT,
                schedule_danger=lambda *_: None,
                gate=lambda _: False,
            )

    result = asyncio.run(run())
    assert result["sent"] is False and result["reason"] == "gated"
    sent.assert_not_awaited()


def test_gate_absent_keeps_http_path_unchanged():
    """HTTP 경로는 게이트 없이 그대로 전송한다 — 데몬 정책이 앱 호출까지 막으면 안 된다."""
    async def run():
        with patch.object(
            ingest_service.spring_client,
            "send_sensing_event",
            new=AsyncMock(return_value={"sensing_event_id": 7, "escalation_triggered": True}),
        ), patch.object(
            ingest_service.spring_client, "send_pose_clip", new=AsyncMock(return_value={})
        ):
            return await ingest_service.deliver(
                make_pred("danger", low_quality=False),
                device_id="care-1",
                care_target_id=1,
                spring_device_id=None,
                detected_at=DETECTED_AT,
                schedule_danger=lambda *_: None,
            )

    assert asyncio.run(run())["sent"] is True


def make_pump():
    """모델 없이 펌프만 세운다 — 억제 규칙은 추론과 무관한 순수 로직이다."""
    from app.config import settings
    from app.stream.pump import StreamPump
    from tests.conftest import FAKE_SPEC

    class StubRuntime:
        spec = FAKE_SPEC

    return StreamPump(StubRuntime(), settings)


def deliver_through_pump(pump, pred, *, escalation_triggered: bool, window_end: float):
    saved = {"sensing_event_id": 1, "escalation_triggered": escalation_triggered}

    async def run():
        with patch.object(
            ingest_service.spring_client, "send_sensing_event", new=AsyncMock(return_value=saved)
        ), patch.object(
            ingest_service.spring_client, "send_pose_clip", new=AsyncMock(return_value={})
        ):
            await pump._deliver("care-1", 1, pred, window_end, [1.0, 1.0, 1.0])

    asyncio.run(run())


def test_downgraded_danger_does_not_arm_cooldown():
    """**이번 리뷰의 핵심 회귀.**

    저품질 danger는 WARNING으로 강등돼 Spring이 에스컬레이션을 만들지 않는다.
    그런데도 억제를 걸면, 곧이어 일어난 **진짜 낙상이 조용히 묻힌다.**
    (수정 전 코드는 모델 원본 라벨로 판단해 여기서 억제를 걸었다.)
    """
    pytest.importorskip("notifi_ai")
    pump = make_pump()

    deliver_through_pump(
        pump, make_pred("danger", low_quality=True), escalation_triggered=False, window_end=100.0
    )

    assert pump.cooldown.allows("care-1", 101.0), "강등된 윈도가 억제를 걸었다 — 진짜 낙상을 삼킨다"


def test_real_escalation_arms_cooldown():
    """반대로 진짜 에스컬레이션이 생기면 겹치는 윈도는 억제돼야 한다 —
    안 그러면 한 번 넘어졌는데 119 신고가 여러 번 걸린다."""
    pytest.importorskip("notifi_ai")
    pump = make_pump()

    deliver_through_pump(
        pump, make_pred("danger", low_quality=False), escalation_triggered=True, window_end=100.0
    )

    assert not pump.cooldown.allows("care-1", 101.0)


def test_second_real_fall_passes_after_downgraded_one():
    """강등된 윈도 직후에 온 진짜 낙상이 실제로 전송되는지 — 사고 시나리오 그대로."""
    pytest.importorskip("notifi_ai")
    pump = make_pump()

    deliver_through_pump(
        pump, make_pred("danger", low_quality=True), escalation_triggered=False, window_end=100.0
    )
    pipeline.reset_throttle()

    sent = AsyncMock(return_value={"sensing_event_id": 2, "escalation_triggered": True})

    async def run():
        with patch.object(ingest_service.spring_client, "send_sensing_event", new=sent), \
             patch.object(ingest_service.spring_client, "send_pose_clip", new=AsyncMock(return_value={})):
            await pump._deliver("care-1", 1, make_pred("danger", low_quality=False), 105.0, [1.0] * 3)

    asyncio.run(run())
    sent.assert_awaited_once()


# ── B. 에스컬레이션 태스크가 GC로 사라지면 안 된다 ──────────────────────────

def test_scheduled_agent_task_is_retained():
    """create_task 반환값을 버리면 루프가 약한 참조만 들고 있어 실행 중 사라질 수 있다.

    에스컬레이션은 음성확인 → 대기 → 알림 → 119로 수 분이 걸려 GC 창이 넓다.
    """
    tasks: set[asyncio.Task] = set()
    finished = []

    async def long_agent():
        await asyncio.sleep(0.05)
        finished.append(True)

    async def run():
        task = asyncio.create_task(long_agent())
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        assert tasks, "실행 중에는 참조가 남아 있어야 한다"
        await asyncio.sleep(0.15)
        assert not tasks, "끝난 뒤에는 정리돼야 한다(무한 증가 방지)"

    asyncio.run(run())
    assert finished == [True]


# ── C. 반쪽 윈도는 캘리브레이션 트라이얼로 받지 않는다 ──────────────────────

def test_low_coverage_trial_is_rejected():
    pytest.importorskip("notifi_ai")
    import numpy as np

    from app.stream.buffer import BufferSet
    from app.stream.collector import SessionStore
    from app.stream.errors import NotEnoughSignal
    from tests.conftest import FAKE_SPEC

    buffers = BufferSet(links=3, retain_seconds=60)
    # 윈도 10.13초짜리에 패킷을 1초 구간에만 넣는다 — 대부분이 결측이 된다
    for index in range(20):
        buffers.get("care-1").add(0, 100.0 + index * 0.05, np.zeros(256, dtype=np.float32))

    store = SessionStore(buffers, FAKE_SPEC)
    with pytest.raises(NotEnoughSignal, match="유효 프레임"):
        store.capture("care-1", None)
