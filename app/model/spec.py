"""모델이 스스로 말하는 계약 — **notifi_ai를 import하지 않는 순수 타입.**

여기 두는 이유는 두 가지다.
  - 모델 없이 도는 테스트가 스펙을 만들 수 있어야 한다
  - 데몬·변환 계층이 타입만 쓰려다 torch를 통째로 끌어오면 안 된다

값을 채우는 건 `app.model.adapter`(notifi_ai를 아는 유일한 모듈)의 몫이다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    action_labels: tuple[str, ...]
    #: 행동별 정적 위험 카테고리(0 safe / 1 warning / 2 danger). risk_labels의 인덱스다
    action_to_risk: tuple[int, ...]
    risk_labels: tuple[str, ...]
    joint_names: tuple[str, ...]
    joint_schema: str
    fps: float
    #: 모델 입력 한 장의 프레임 수
    frames: int
    links: int
    subcarriers: int
    #: 이 간격 안에 실측 패킷이 없으면 해당 프레임의 링크를 무효로 본다
    max_gap_seconds: float

    def __post_init__(self) -> None:
        """모델이 바뀌었는데 하위 가정이 못 따라간 경우를 **로드 시점에** 터뜨린다.

        조용히 틀린 판정을 내는 것보다 부팅에서 실패하는 편이 낫다 — 응급 판정 경로다.
        """
        if len(self.action_to_risk) != len(self.action_labels):
            raise ValueError(
                f"행동 {len(self.action_labels)}종인데 위험 매핑은 {len(self.action_to_risk)}칸"
            )
        if len(self.risk_labels) != 3:
            # pipeline._RISK_ID_TO_EVENT가 3칸(safe/warning/danger) 고정이다
            raise ValueError(f"위험 등급이 3종이 아니다: {self.risk_labels}")
        if self.action_to_risk and max(self.action_to_risk) >= len(self.risk_labels):
            raise ValueError("행동→위험 매핑이 위험 등급 범위를 벗어난다")
        if not self.joint_names:
            raise ValueError("관절 목록이 비었다")
        if self.frames <= 0 or self.fps <= 0 or self.links <= 0:
            raise ValueError("입력 형상이 비정상이다")

    @property
    def window_seconds(self) -> float:
        """모델 입력 한 장의 길이. 수집 데몬의 윈도 길이이기도 하다."""
        return self.frames / self.fps

    def as_dict(self) -> dict[str, Any]:
        """`GET /internal/model/spec` 응답. 하위(Spring·앱·데몬)가 계약을 확인한다."""
        return {
            "model_name": self.model_name,
            "action_labels": list(self.action_labels),
            "action_to_risk": list(self.action_to_risk),
            "risk_labels": list(self.risk_labels),
            "joint_schema": self.joint_schema,
            "joint_names": list(self.joint_names),
            "fps": self.fps,
            "frames": self.frames,
            "links": self.links,
            "subcarriers": self.subcarriers,
            "window_seconds": round(self.window_seconds, 3),
            "max_gap_seconds": self.max_gap_seconds,
        }
