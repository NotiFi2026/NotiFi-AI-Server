"""실시간 판정 정책 — 겹치는 윈도가 같은 사고를 여러 번 신고하지 않게 한다.

데몬은 stride(예: 2초)마다 10.13초 윈도를 뜬다. 낙상 하나가 5개 윈도에 걸쳐 잡히므로,
그대로 두면 **한 번 넘어졌는데 119 신고가 5번 걸린다.**

NORMAL 절감(pipeline.should_send)과는 다른 층이다. 그쪽은 "같은 평온이 이어질 때 전송을 줄이는"
것이고, 이쪽은 "같은 사고를 중복 신고하지 않는" 것이다.

notifi_ai를 import하지 않는다 — 순수 규칙이라 모델 없이 테스트된다.
"""
from __future__ import annotations

import threading


class AlertCooldown:
    """디바이스별 위험 판정 재신고 억제.

    쿨다운 중에도 **NORMAL·경고 이벤트는 그대로 흘려보낸다.** 막는 건 위험 판정 하나뿐이라,
    쿨다운 동안 사람이 일어났다는 사실(walking 등)은 정상적으로 적재된다.
    """

    def __init__(self, cooldown_seconds: float) -> None:
        self._cooldown = cooldown_seconds
        self._lock = threading.Lock()
        self._last_alert_at: dict[str, float] = {}

    def allows(self, device_id: str, now: float) -> bool:
        if self._cooldown <= 0:
            return True
        with self._lock:
            previous = self._last_alert_at.get(device_id)
        return previous is None or now - previous >= self._cooldown

    def mark(self, device_id: str, now: float) -> None:
        """실제로 신고한 뒤에만 호출한다 — 적재 실패까지 쿨다운으로 세면 사고가 통째로 유실된다."""
        with self._lock:
            self._last_alert_at[device_id] = now

    def clear(self, device_id: str) -> None:
        with self._lock:
            self._last_alert_at.pop(device_id, None)

    def remaining(self, device_id: str, now: float) -> float:
        with self._lock:
            previous = self._last_alert_at.get(device_id)
        if previous is None:
            return 0.0
        return max(0.0, self._cooldown - (now - previous))
