"""링크별 CSI 패킷 링버퍼 — 실시간 신호와 모델 입력(고정 텐서) 사이의 간극을 메운다.

ESP는 링크마다 제각각인 시각에 패킷을 뱉는다. 모델은 `[frames, links, subcarriers, 2]`
한 장을 요구한다. 그 사이를 잇는 게 이 버퍼와 windower다.

**notifi_ai를 import하지 않는다** — 순수 자료구조라 모델 없이 전부 테스트된다.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Iterable

import numpy as np


class LinkBuffer:
    """링크 하나의 최근 패킷. 오래된 것은 버린다(무한히 쌓이면 메모리를 먹는다)."""

    def __init__(self, retain_seconds: float) -> None:
        self._retain = retain_seconds
        self._items: deque[tuple[float, np.ndarray]] = deque()

    def add(self, at: float, iq: np.ndarray) -> None:
        self._items.append((at, iq))
        cutoff = at - self._retain
        while self._items and self._items[0][0] < cutoff:
            self._items.popleft()

    def slice(self, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
        """[start, end] 구간의 (시각, IQ). 비어 있으면 빈 배열 — 호출자가 링크 결측으로 다룬다."""
        picked = [(at, iq) for at, iq in self._items if start <= at <= end]
        if not picked:
            return np.zeros(0, dtype=np.float64), np.zeros((0, 0), dtype=np.float32)
        times = np.fromiter((at for at, _ in picked), dtype=np.float64, count=len(picked))
        return times, np.stack([iq for _, iq in picked])

    def __len__(self) -> int:
        return len(self._items)

    @property
    def newest(self) -> float | None:
        return self._items[-1][0] if self._items else None


class DeviceBuffer:
    """디바이스 1채(RX 1 + TX 3)의 링크별 버퍼.

    수신 스레드가 쓰고 윈도 루프가 읽으므로 락으로 감싼다 — deque는 append/popleft만
    원자적이고, 여기 slice는 전체 순회라 그 보장 밖이다.
    """

    def __init__(self, device_id: str, links: int, retain_seconds: float) -> None:
        self.device_id = device_id
        self._lock = threading.Lock()
        self._links = [LinkBuffer(retain_seconds) for _ in range(links)]

    def add(self, link_index: int, at: float, iq: np.ndarray) -> None:
        with self._lock:
            self._links[link_index].add(at, iq)

    def window(self, start: float, end: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """`packets_to_grid`가 받는 형태 — 링크 순서대로 (시각 배열, IQ 배열)."""
        with self._lock:
            sliced = [link.slice(start, end) for link in self._links]
        return [times for times, _ in sliced], [iq for _, iq in sliced]

    def newest_packet_at(self) -> float | None:
        """가장 최근 패킷 시각. 윈도를 끊을 기준이자, 신호가 끊겼는지 판단하는 근거."""
        with self._lock:
            stamps = [link.newest for link in self._links]
        present = [value for value in stamps if value is not None]
        return max(present) if present else None

    def counts(self) -> list[int]:
        with self._lock:
            return [len(link) for link in self._links]


class BufferSet:
    """디바이스별 버퍼 모음. 등록된 보드에서 온 패킷만 받는다."""

    def __init__(self, links: int, retain_seconds: float) -> None:
        self._links = links
        self._retain = retain_seconds
        self._lock = threading.Lock()
        self._devices: dict[str, DeviceBuffer] = {}

    def get(self, device_id: str) -> DeviceBuffer:
        with self._lock:
            buffer = self._devices.get(device_id)
            if buffer is None:
                buffer = DeviceBuffer(device_id, self._links, self._retain)
                self._devices[device_id] = buffer
            return buffer

    def active(self) -> Iterable[DeviceBuffer]:
        with self._lock:
            return list(self._devices.values())

    def drop(self, device_id: str) -> None:
        """디바이스가 삭제되면 CSI 파생 상태도 남기지 않는다."""
        with self._lock:
            self._devices.pop(device_id, None)
