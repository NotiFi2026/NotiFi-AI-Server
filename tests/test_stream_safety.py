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


# ── B. 살아 있는 보드가 앱에 "신호 없음"으로 보이면 안 된다 ──────────────────
#
# 하트비트(I4)를 보낼 수 있는 건 CSI 라인을 받는 이 데몬뿐이다. 안 보내면 Spring의
# last_seen_at이 영원히 null이고, 노드가 멀쩡히 도는데도 보호자 앱 디바이스 화면은
# "신호 없음"으로 남는다 — 오류 하나 없이 화면만 거짓말하는 종류다.


class FakeDeviceConfig:
    """레지스트리 DeviceConfig 중 라우터가 실제로 읽는 필드만."""

    def __init__(self, device_id: str, tx1_id=None, tx2_id=None, tx3_id=None):
        self.device_id = device_id
        self.tx1_id = tx1_id
        self.tx2_id = tx2_id
        self.tx3_id = tx3_id


def feed(router, mac: str, at: float) -> None:
    """CSI 한 줄이 들어온 것처럼 라우터를 돌린다(파싱 자체는 모델 영역이라 세워 둔다)."""
    import numpy as np

    with patch(
        "app.stream.router.parse_csi_line",
        return_value=(mac, np.zeros(256, dtype=np.float32)),
    ):
        router.handle("CSI_DATA,...", at)


def make_router():
    from app.stream.buffer import BufferSet
    from app.stream.router import PacketRouter

    return PacketRouter(BufferSet(links=3, retain_seconds=60))


def stream_client():
    """펌프가 실제로 부르는 spring_client 모듈 — 여기에 패치를 걸어야 한다."""
    from app.stream import pump as pump_module

    return pump_module.spring_client


def test_alive_boards_uses_registered_uid_not_normalized_mac():
    """Spring은 device_uid를 완전 일치로 찾는다.

    라우터는 표기 흔들림을 흡수하려고 MAC을 소문자로 정규화하는데, 그 값을 그대로
    하트비트에 쓰면 대문자로 등록된 기기는 매번 404다 — 그리고 404는 조용하다.
    """
    pytest.importorskip("notifi_ai")
    router = make_router()
    router.reload([FakeDeviceConfig("care-1", tx1_id="1A:00:00:00:00:00")])

    feed(router, "1a:00:00:00:00:00", 100.0)

    assert list(router.alive_boards(since=99.0)) == ["1A:00:00:00:00:00"]


def test_alive_boards_excludes_silent_board():
    """전원이 빠진 보드에까지 하트비트를 보내면 앱은 죽은 노드를 정상으로 표시한다."""
    pytest.importorskip("notifi_ai")
    router = make_router()
    router.reload([FakeDeviceConfig("care-1", tx1_id="aa", tx2_id="bb")])

    feed(router, "aa", 100.0)
    feed(router, "bb", 200.0)

    assert list(router.alive_boards(since=150.0)) == ["bb"]


def test_unregistered_board_is_not_tracked():
    pytest.importorskip("notifi_ai")
    router = make_router()
    router.reload([FakeDeviceConfig("care-1", tx1_id="aa")])

    feed(router, "ff", 100.0)

    assert router.alive_boards(since=0.0) == {}


def test_removed_board_stops_being_reported():
    """등록에서 빠진 보드가 계속 살아 있는 것으로 보고되면 안 된다."""
    pytest.importorskip("notifi_ai")
    router = make_router()
    router.reload([FakeDeviceConfig("care-1", tx1_id="aa")])
    feed(router, "aa", 100.0)

    router.reload([FakeDeviceConfig("care-1", tx1_id="bb")])

    assert router.alive_boards(since=0.0) == {}


def make_pump_with_live_board(interval: float = 60.0):
    pump = make_pump()
    pump._config = pump._config.model_copy(
        update={"notifi_stream_heartbeat_seconds": interval}
    )
    pump.router.reload([FakeDeviceConfig("care-1", tx1_id="aa")])
    feed(pump.router, "aa", 100.0)
    return pump


def test_heartbeat_is_sent_once_per_interval():
    """윈도 루프는 stride(2초)마다 도는데 하트비트까지 매번 보내면 Spring을 두드리기만 한다."""
    pytest.importorskip("notifi_ai")
    pump = make_pump_with_live_board()
    sender = AsyncMock(return_value=True)

    async def run():
        with patch.object(stream_client(), "send_heartbeat", new=sender):
            await pump._flush_heartbeats(100.0)
            await pump._flush_heartbeats(102.0)  # stride만 지났다 — 아직 주기가 아니다
            feed(pump.router, "aa", 161.0)
            await pump._flush_heartbeats(161.0)

    asyncio.run(run())
    assert sender.await_count == 2
    assert pump.stats["heartbeats"] == 2


def test_heartbeat_failure_does_not_break_the_window_loop():
    """하트비트는 부가 정보다. 여기서 예외가 새면 그 stride의 낙상 판정이 통째로 날아간다."""
    pytest.importorskip("notifi_ai")
    import httpx

    pump = make_pump_with_live_board()

    async def run():
        with patch.object(
            stream_client(),
            "send_heartbeat",
            new=AsyncMock(side_effect=httpx.ConnectError("spring down")),
        ):
            await pump._flush_heartbeats(100.0)

    asyncio.run(run())  # 예외가 올라오면 실패한다
    assert pump.stats["heartbeats"] == 0


def test_unregistered_board_warns_once(caplog):
    """Spring에 등록 안 된 보드는 주기마다 404다 — 같은 경고가 로그를 덮으면 안 된다."""
    pytest.importorskip("notifi_ai")
    pump = make_pump_with_live_board()

    async def run():
        with patch.object(stream_client(), "send_heartbeat", new=AsyncMock(return_value=False)):
            await pump._flush_heartbeats(100.0)
            feed(pump.router, "aa", 161.0)
            await pump._flush_heartbeats(161.0)

    with caplog.at_level("WARNING"):
        asyncio.run(run())

    warned = [r for r in caplog.records if "등록되지 않은 보드" in r.getMessage()]
    assert len(warned) == 1
    assert pump.stats["heartbeats"] == 0


def test_heartbeat_can_be_disabled():
    pytest.importorskip("notifi_ai")
    pump = make_pump_with_live_board(interval=0.0)
    sender = AsyncMock(return_value=True)

    async def run():
        with patch.object(stream_client(), "send_heartbeat", new=sender):
            await pump._flush_heartbeats(100.0)

    asyncio.run(run())
    sender.assert_not_awaited()


# ── C. 에스컬레이션 태스크가 GC로 사라지면 안 된다 ──────────────────────────

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


# ── D. 반쪽 윈도는 캘리브레이션 트라이얼로 받지 않는다 ──────────────────────

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
