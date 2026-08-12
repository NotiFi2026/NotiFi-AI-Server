"""수집 데몬의 순수 부분 — 링버퍼·재신고 억제·리플레이 소스.

전부 notifi_ai 없이 돈다. 모델을 타는 부분(라우터·펌프)은 실보드/리플레이 검증으로 확인한다.
"""
import csv

import numpy as np
import pytest

from app.stream.buffer import BufferSet, DeviceBuffer
from app.stream.policy import AlertCooldown
from app.stream.sources import ReplaySource, build_source


def iq(value: float = 1.0) -> np.ndarray:
    return np.full(256, value, dtype=np.float32)


# ── 링버퍼 ──────────────────────────────────────────────────────────────────

def test_window_returns_only_requested_span():
    buffer = DeviceBuffer("care-1", links=3, retain_seconds=60)
    for index in range(10):
        buffer.add(0, float(index), iq(index))

    times, _ = buffer.window(3.0, 6.0)[0][0], buffer.window(3.0, 6.0)[1][0]
    assert times.tolist() == [3.0, 4.0, 5.0, 6.0]


def test_old_packets_are_dropped():
    """무한히 쌓이면 메모리를 먹는다 — 윈도 밖은 버린다."""
    buffer = DeviceBuffer("care-1", links=3, retain_seconds=5.0)
    for index in range(20):
        buffer.add(0, float(index), iq())

    assert buffer.counts()[0] == 6  # 14.0~19.0
    times, _ = buffer.window(0.0, 100.0)
    assert times[0][0] == 14.0


def test_missing_link_yields_empty_arrays():
    """링크 하나가 죽어도 윈도는 만들어져야 한다 — 결측은 link_mask가 표현한다."""
    buffer = DeviceBuffer("care-1", links=3, retain_seconds=60)
    buffer.add(0, 1.0, iq())

    times, values = buffer.window(0.0, 2.0)
    assert len(times[0]) == 1
    assert len(times[1]) == 0 and len(times[2]) == 0
    assert values[1].shape[0] == 0


def test_newest_packet_at_spans_links():
    buffer = DeviceBuffer("care-1", links=3, retain_seconds=60)
    assert buffer.newest_packet_at() is None
    buffer.add(0, 1.0, iq())
    buffer.add(2, 5.0, iq())
    assert buffer.newest_packet_at() == 5.0


def test_buffer_set_isolates_devices_and_drops():
    buffers = BufferSet(links=3, retain_seconds=60)
    buffers.get("care-1").add(0, 1.0, iq())
    buffers.get("care-2").add(0, 2.0, iq())
    assert {b.device_id for b in buffers.active()} == {"care-1", "care-2"}

    buffers.drop("care-1")
    assert {b.device_id for b in buffers.active()} == {"care-2"}


# ── 재신고 억제 ─────────────────────────────────────────────────────────────

def test_cooldown_blocks_repeat_alert_within_window():
    """겹치는 윈도가 같은 낙상을 여러 번 신고하면 119가 여러 번 걸린다
    (Spring은 DANGER마다 에스컬레이션을 새로 만든다)."""
    cooldown = AlertCooldown(120.0)
    assert cooldown.allows("care-1", 100.0)

    cooldown.mark("care-1", 100.0)
    assert not cooldown.allows("care-1", 102.0)
    assert not cooldown.allows("care-1", 219.0)
    assert cooldown.allows("care-1", 220.0)


def test_cooldown_is_per_device():
    cooldown = AlertCooldown(120.0)
    cooldown.mark("care-1", 100.0)
    assert cooldown.allows("care-2", 101.0)


def test_cooldown_zero_disables_suppression():
    cooldown = AlertCooldown(0.0)
    cooldown.mark("care-1", 100.0)
    assert cooldown.allows("care-1", 100.0)


def test_remaining_reports_time_left():
    cooldown = AlertCooldown(120.0)
    cooldown.mark("care-1", 100.0)
    assert cooldown.remaining("care-1", 130.0) == pytest.approx(90.0)
    assert cooldown.remaining("care-2", 130.0) == 0.0


# ── 소스 ────────────────────────────────────────────────────────────────────

def test_replay_source_yields_only_csi_lines(tmp_path):
    path = tmp_path / "csi.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc_time_ms", "raw_line"])
        writer.writerow([1000, "CSI_DATA,x,aa:bb,[1,2,3]"])
        writer.writerow([1010, "boot: something"])  # 부팅 로그 등 잡음
        writer.writerow([1020, "CSI_DATA,y,aa:bb,[4,5,6]"])

    lines = list(ReplaySource(path, speed=1000).lines())
    assert len(lines) == 2
    assert all(line.startswith("CSI_DATA") for line in lines)


def test_replay_source_requires_path():
    with pytest.raises(ValueError, match="REPLAY_PATH"):
        build_source("replay", port="", baud=0, replay_path="", replay_speed=1.0, replay_loop=False)


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="알 수 없는 수집 소스"):
        build_source("udp", port="", baud=0, replay_path="", replay_speed=1.0, replay_loop=False)
