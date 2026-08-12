"""캘리브레이션 트라이얼을 **라이브 버퍼에서 뜬다.**

기존 경로는 "완성된 NPZ 업로드 → fit"인데, NPZ를 만들 수 있는 건 CSI를 직접 읽는 쪽뿐이라
앱은 캘리브레이션을 시작할 수가 없었다. 데몬이 링버퍼를 들고 있으니 캡처는 사실상 공짜다 —
"지금 이 순간의 10.13초를 한 장 떠서 세션에 넣어라"만 있으면 된다.

앱 캘리브레이션 위저드가 이 API 위에 올라간다. NPZ 업로드 경로는 오프라인용으로 남겨 둔다.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.model.adapter import ModelSpec, SupportTrial, window_from_packets
from app.stream.buffer import BufferSet
from app.stream.errors import NotEnoughSignal


@dataclass
class Trial:
    """캡처된 트라이얼 한 건. absence면 action_id가 없다."""

    csi: np.ndarray
    link_mask: np.ndarray
    action_id: int | None
    captured_at: float

    @property
    def kind(self) -> str:
        return "absence" if self.action_id is None else "support"

    def coverage(self) -> list[float]:
        """링크별 유효 프레임 비율. 위저드가 재촬영을 권할지 판단하는 근거."""
        return [round(float(self.link_mask[:, i].mean()), 3) for i in range(self.link_mask.shape[1])]

    def summary(self, index: int) -> dict[str, Any]:
        return {
            "index": index,
            "kind": self.kind,
            "action_id": self.action_id,
            "link_coverage": self.coverage(),
            "any_link_coverage": round(float(self.link_mask.any(axis=1).mean()), 3),
        }


@dataclass
class CalibrationSession:
    """디바이스 하나의 진행 중인 캘리브레이션. 메모리에만 둔다.

    서버가 재시작되면 사라진다 — 수집은 사람이 방에서 움직이는 몇 분짜리 작업이라
    중간 상태를 디스크에 남길 값어치가 없고, 남기면 낡은 세션이 조용히 섞인다.
    """

    device_id: str
    trials: list[Trial] = field(default_factory=list)

    def absence(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(t.csi, t.link_mask) for t in self.trials if t.action_id is None]

    def support(self, spec: ModelSpec) -> list[SupportTrial]:
        return [
            SupportTrial(
                csi=t.csi,
                link_mask=t.link_mask,
                action_id=t.action_id,
                # 위험 등급은 행동에서 파생된다 — 호출자가 따로 정하면 라벨이 어긋날 수 있다
                risk_id=spec.action_to_risk[t.action_id],
            )
            for t in self.trials
            if t.action_id is not None
        ]

    def progress(self) -> dict[str, Any]:
        counts: dict[int, int] = {}
        for trial in self.trials:
            if trial.action_id is not None:
                counts[trial.action_id] = counts.get(trial.action_id, 0) + 1
        return {
            "device_id": self.device_id,
            "absence_trials": len(self.absence()),
            "support_trials": sum(counts.values()),
            "support_action_counts": counts,
            "trials": [trial.summary(index) for index, trial in enumerate(self.trials)],
        }


#: 이 밑이면 트라이얼로 받지 않는다. 반쪽짜리 윈도가 absence 기준선에 섞이면
#: 이후 모든 판정이 조용히 틀어진다 — 커버리지를 응답에 실어 보내는 것만으로는 부족하다.
#: 위저드가 경고를 무시하면 그대로 들어가기 때문이다.
MIN_TRIAL_COVERAGE = 0.3


class SessionStore:
    """디바이스별 캘리브레이션 세션. 캡처는 버퍼에서 윈도를 한 장 뜨는 것뿐이다."""

    def __init__(self, buffers: BufferSet, spec: ModelSpec) -> None:
        self._buffers = buffers
        self._spec = spec
        self._lock = threading.Lock()
        self._sessions: dict[str, CalibrationSession] = {}

    def get(self, device_id: str) -> CalibrationSession:
        with self._lock:
            session = self._sessions.get(device_id)
            if session is None:
                session = CalibrationSession(device_id)
                self._sessions[device_id] = session
            return session

    def capture(self, device_id: str, action_id: int | None) -> tuple[int, Trial]:
        """지금 버퍼의 마지막 한 윈도를 트라이얼로 담는다.

        Raises:
            NotEnoughSignal: 윈도를 채울 만큼 패킷이 없다
        """
        spec = self._spec
        buffer = self._buffers.get(device_id)
        end = buffer.newest_packet_at()
        if end is None:
            raise NotEnoughSignal("수신된 CSI 패킷이 없다")

        start = end - spec.window_seconds
        per_link_times, per_link_iq = buffer.window(start, end)
        if not any(len(times) for times in per_link_times):
            raise NotEnoughSignal("윈도 구간에 패킷이 없다")

        csi, link_mask = window_from_packets(per_link_times, per_link_iq, spec, end)

        # 데몬이 막 켜졌거나 신호가 끊기면 윈도의 대부분이 결측이다. 그런 트라이얼을 담으면
        # 기준선이 오염되고, 그 뒤 판정이 전부 틀어지는데 원인을 찾기가 매우 어렵다.
        usable = float(link_mask.any(axis=1).mean())
        if usable < MIN_TRIAL_COVERAGE:
            raise NotEnoughSignal(
                f"유효 프레임 {usable:.0%} — {MIN_TRIAL_COVERAGE:.0%} 이상 필요. "
                "보드 전원·거리를 확인하고 잠시 뒤 다시 시도하라"
            )

        trial = Trial(csi=csi, link_mask=link_mask, action_id=action_id, captured_at=time.monotonic())

        session = self.get(device_id)
        with self._lock:
            session.trials.append(trial)
            index = len(session.trials) - 1
        return index, trial

    def drop_trial(self, device_id: str, index: int) -> bool:
        """재촬영 — 마음에 안 드는 트라이얼을 빼고 다시 찍게 한다."""
        session = self.get(device_id)
        with self._lock:
            if not 0 <= index < len(session.trials):
                return False
            session.trials.pop(index)
            return True

    def clear(self, device_id: str) -> None:
        with self._lock:
            self._sessions.pop(device_id, None)
