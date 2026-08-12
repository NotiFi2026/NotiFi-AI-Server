"""notifi_ai 결합을 가두는 경계. **이 파일만 notifi_ai를 import한다.**

모델은 계속 갱신된다(v2·v3…). 새 버전이 왔을 때 고칠 곳이 여기 하나가 되도록,
나머지 코드는 `ModelSpec`(모델이 스스로 말하는 계약)과 순수 dict만 본다.

주의: top-level에서 notifi_ai·torch를 끌어오므로 **부팅 경로에서 import하면 안 된다.**
모델 미설치 환경에서 서버가 아예 뜨지 않게 된다(실제로 두 번 밟았다).
lifespan 안에서만 import하고, 라우터는 TYPE_CHECKING으로만 참조한다.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from notifi_ai import parser_contract as C
from notifi_ai.constants import ACTION_TO_RISK, JOINT_NAMES, MAX_FRAMES
from notifi_ai.csi_parser import default_grid, packets_to_grid, parse_csi_line
from notifi_ai.io import load_calibration_npz, load_query_npz
from notifi_ai.registry import DeviceRegistry
from notifi_ai.schemas import DeviceConfig

from app.model.spec import ModelSpec

__all__ = [
    "DeviceConfig",
    "DeviceRegistry",
    "ModelSpec",
    "build_spec",
    "load_calibration_npz",
    "load_model",
    "load_query_npz",
    "parse_csi_line",
    "window_from_packets",
]

# 관절 수 → 공개 스키마 이름. 모델이 joint_schema를 직접 알려주지 않아 여기서 판정한다
# (AI팀에 describe() 확장을 요청해 둔 상태 — 오면 이 표 대신 모델 값을 쓴다).
# 모르는 관절 수는 이름을 지어내지 않는다. 앱이 모르는 스키마를 그리지 않고 안내를 띄우는 쪽이
# 엉뚱한 자세를 그리는 것보다 안전하다.
_JOINT_SCHEMAS = {22: "smpl-22"}


def load_model(artifact_dir: str | None, device: str | None, class_name: str) -> Any:
    """모델 인스턴스를 만든다. 클래스명이 설정인 이유는 그게 버전을 달고 있기 때문이다
    (`NotiFiAIv1` → v2는 `NotiFiAIv2`). 환경변수 한 줄로 갈아끼운다."""
    module = importlib.import_module("notifi_ai")
    try:
        factory = getattr(module, class_name)
    except AttributeError as exc:
        available = [name for name in dir(module) if name.startswith("NotiFiAI")]
        raise RuntimeError(
            f"notifi_ai에 {class_name}가 없다. 설치된 클래스: {available or '없음'}"
        ) from exc
    return factory(artifact_dir=artifact_dir, device=device)


def build_spec(model: Any) -> ModelSpec:
    """모델에서 계약을 읽어 온다.

    describe()가 주는 것(model_name·actions·risks)은 모델에서, 아직 안 주는 것은
    패키지 상수에서 가져온다. 어느 쪽이든 **이 함수 밖에서는 출처를 알 필요가 없다.**
    """
    described = model.describe()
    joint_names = tuple(JOINT_NAMES)
    schema = _JOINT_SCHEMAS.get(len(joint_names))
    if schema is None:
        raise RuntimeError(
            f"관절 {len(joint_names)}개짜리 스키마 이름을 모른다 — adapter._JOINT_SCHEMAS에 추가하라"
        )

    # 정합성 검사는 ModelSpec.__post_init__이 한다 — 어긋나면 여기서 터진다
    return ModelSpec(
        model_name=described["model_name"],
        action_labels=tuple(described["actions"]),
        action_to_risk=tuple(int(value) for value in ACTION_TO_RISK),
        risk_labels=tuple(described["risks"]),
        joint_names=joint_names,
        joint_schema=schema,
        fps=float(C.TARGET_FPS),
        frames=int(MAX_FRAMES),
        links=int(C.N_LINKS),
        subcarriers=len(C.LIVE_SUBCARRIERS),
        max_gap_seconds=float(C.MAX_GAP_S),
    )


def window_from_packets(
    per_link_times: list[np.ndarray],
    per_link_iq: list[np.ndarray],
    spec: ModelSpec,
    window_end_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """링크별 비동기 패킷 → 모델 입력 한 장 `(csi[F,L,S,2], link_mask[F,L])`.

    전처리는 반드시 모델 패키지의 `packets_to_grid`를 쓴다 — 학습과 배포가 공유하는
    유일한 경로라, 다시 구현하면 재현 불가능한 어긋남이 생긴다.

    격자는 0부터 시작하므로 윈도 시작 시각만큼 평행이동해 패킷 시각과 축을 맞춘다.
    """
    grid = default_grid(spec.window_seconds, spec.fps)
    grid = grid + (window_end_seconds - spec.window_seconds)
    return packets_to_grid(per_link_times, per_link_iq, grid, spec.max_gap_seconds)
