"""notifi-ai 모델 런타임 — 프로세스당 1개 인스턴스를 들고 추론을 직렬화한다.

notifi_ai 패키지의 create_app(api.py)이 하던 구성(모델 + 레지스트리 + Lock)을 옮겼다.
번들 구현이 빠뜨린 두 가지를 여기서 보강한다:
  - 기동 시 warmup 호출 (첫 CUDA 요청이 수십 초 걸리는 것을 없앤다)
  - 블로킹 추론을 이벤트 루프 밖에서 실행 (호출부에서 run_in_threadpool)
"""
from __future__ import annotations

import threading
import time
from typing import Any

from notifi_ai import NotiFiAIv1
from notifi_ai.constants import ACTION_TO_RISK, JOINT_NAMES, TARGET_FPS
from notifi_ai.io import load_query_npz
from notifi_ai.registry import DeviceRegistry

from app.common.logging_config import logger
from app.config import Settings, settings
from app.model.errors import InferenceBusyError


class ModelRuntime:
    """로드된 모델 1개 + 디바이스 레지스트리 + 추론 직렬화 락."""

    def __init__(self, model: NotiFiAIv1, registry: DeviceRegistry) -> None:
        self._model = model
        self._registry = registry
        # 모델은 스레드 안전하지 않다 — 모든 추론을 직렬화한다
        self._lock = threading.Lock()
        # 진행 상태. 단일 대입/읽기라 GIL 하에서 원자적이고, 관측용이라
        # 약간의 경합은 무해하므로 별도 락을 두지 않는다.
        self._inflight_since: float | None = None
        self._last_success_at: float | None = None

    @classmethod
    def load(cls, settings: Settings) -> "ModelRuntime":
        """아티팩트를 로드하고 warmup까지 마친 런타임을 반환한다."""
        model = NotiFiAIv1(
            artifact_dir=settings.notifi_artifact_dir,
            device=settings.notifi_model_device,
        )
        model.warmup()
        return cls(model, DeviceRegistry(settings.notifi_registry_root))

    def describe(self) -> dict[str, Any]:
        return self._model.describe()

    def list_devices(self) -> list[str]:
        return self._registry.list_devices()

    def inflight_seconds(self) -> float | None:
        """진행 중인 추론의 경과 시간. 유휴면 None.

        실행 중인 CUDA 연산은 파이썬에서 중단시킬 수 없다 — 스레드를 죽일 수도,
        asyncio 타임아웃으로 되돌릴 수도 없다. 그래서 멈춘 추론을 "고치는" 방법은
        없고, 관측해서 외부가 프로세스를 재시작하게 하는 것이 유일한 대응이다.
        """
        started = self._inflight_since
        return None if started is None else time.monotonic() - started

    def last_success_age_seconds(self) -> float | None:
        finished = self._last_success_at
        return None if finished is None else time.monotonic() - finished

    def predict_npz(
        self,
        device_id: str,
        payload: bytes,
        include_pose: bool = False,
    ) -> dict[str, Any]:
        """쿼리 NPZ를 추론한다. 캘리브레이션 프로필이 없으면 FileNotFoundError.

        무보정 추론은 허용하지 않는다 — 확정된 연동 계약.
        """
        profile = self._registry.load_calibration(device_id)
        csi, link_mask = load_query_npz(payload)

        timeout = settings.notifi_inference_lock_timeout_seconds
        if not self._lock.acquire(timeout=timeout):
            # 무한 대기하면 스레드풀이 채워지며 서버 전체가 조용히 멎는다.
            # 포기하고 503을 주면 호출자가 해당 윈도를 버리고 다음 윈도로 넘어간다.
            logger.warning(
                "추론 락 획득 실패 — 앞 추론이 지연/정지 중",
                extra={
                    "action": "inference_busy",
                    "device_id": device_id,
                    "inflight_seconds": self.inflight_seconds(),
                },
            )
            raise InferenceBusyError(f"inference busy (waited {timeout}s)")
        try:
            self._inflight_since = time.monotonic()
            prediction = self._model.predict(csi, link_mask, profile)
            self._last_success_at = time.monotonic()
        finally:
            self._inflight_since = None
            self._lock.release()

        logger.info(
            "모델 추론 완료",
            extra={
                "action": "model_predict",
                "device_id": device_id,
                "action_label": prediction.action_label,
                "risk_label": prediction.risk_label,
                "low_quality": prediction.quality.get("low_quality"),
            },
        )
        result = prediction.to_dict(include_pose=include_pose)
        # 변환 계층이 notifi_ai에 의존하지 않도록 필요한 상수를 여기서 실어 보낸다.
        # action_risk_id는 행동의 정적 카테고리(0 safe/1 warning/2 danger)로,
        # 독립 위험도 헤드(risk_label)와 다르다 — event_type 매핑에 쓴다.
        result["action_risk_id"] = ACTION_TO_RISK[prediction.action_id]
        if include_pose:
            result["joints"] = list(JOINT_NAMES)
            result["fps"] = TARGET_FPS
        return result
