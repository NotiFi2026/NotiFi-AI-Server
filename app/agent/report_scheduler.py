"""일일 리포트 스케줄러 — 매일 정해진 KST 시각에 감시 중인 가구의 리포트를 만든다.

`report_service.run_daily_report`는 있었지만 부르는 게 수동 API 라우트뿐이라
"매일 아침 어제 리포트가 도착한다"는 약속이 실제로는 지켜지지 않았다. 이 모듈이 그걸 잇는다.

**notifi_ai를 import하지 않는다.** 리포트는 I6 조회 → LLM 생성 → I3 적재라 모델이 필요 없는데,
디바이스 목록을 얻겠다고 `app.model.adapter`(=`DeviceRegistry`)를 끌어오면 torch가 딸려와
부팅 경로가 깨진다 — adapter 헤더가 "부팅 경로에서 import하면 안 된다"고 못박아 둔 그 문제다.
그래서 레지스트리 클래스 대신 **레지스트리 루트의 디렉터리 이름**을 읽는다. 모델 로드가
실패한 서버에서도 리포트는 돌아야 하므로 이 분리는 성능이 아니라 가용성 문제다.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from app.agent.report_service import default_report_date, run_daily_report
from app.common.logging_config import logger
from app.config import Settings
from app.model.pipeline import care_target_id_from

#: 리포트의 "하루" 경계 — report_service·Spring I6와 같은 존이어야 날짜가 어긋나지 않는다.
KST = ZoneInfo("Asia/Seoul")


def next_run_at(now_utc: datetime, hour: int, minute: int) -> datetime:
    """다음 실행 시각(UTC). KST 기준 오늘 hour:minute이 이미 지났으면 내일이다."""
    now_kst = now_utc.astimezone(KST)
    candidate = now_kst.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now_kst:
        candidate += timedelta(days=1)
    return candidate.astimezone(timezone.utc)


def list_care_target_ids(registry_root: str) -> list[int]:
    """이 서버가 감시 중인 가구의 노인 ID.

    레지스트리는 device_id마다 디렉터리 하나를 두고, device_id는 `care-{노인ID}` 규약이다.
    규약을 안 따르는 항목(수동 등록된 `home-001` 등)은 노인 ID를 알 수 없으므로 건너뛴다.
    """
    root = Path(registry_root)
    if not root.is_dir():
        return []

    ids: list[int] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        care_target_id = care_target_id_from(entry.name)
        if care_target_id is not None:
            ids.append(care_target_id)
    return ids


class ReportScheduler:
    """KST 고정 시각에 하루 한 번 리포트 배치를 돌린다.

    APScheduler를 넣지 않는다 — 잡이 하나뿐이라 "다음 시각까지 잠든다" 루프면 충분하고,
    이 레포에 스케줄러 의존성을 새로 들이지 않는 편이 낫다.
    """

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        self._stop.clear()
        # 반환값을 반드시 붙잡는다 — 루프는 약한 참조만 들고 있어 놓아두면 실행 중 GC된다
        # (같은 실수를 스트림 데몬에서 이미 한 번 했다)
        self._task = asyncio.create_task(self._loop(), name="daily-report-scheduler")
        logger.info(
            "리포트 스케줄러 기동",
            extra={
                "action": "report_scheduler_started",
                "hour_kst": self._config.notifi_report_hour_kst,
                "minute_kst": self._config.notifi_report_minute_kst,
            },
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while not self._stop.is_set():
            now = datetime.now(timezone.utc)
            target = next_run_at(
                now,
                self._config.notifi_report_hour_kst,
                self._config.notifi_report_minute_kst,
            )
            wait_seconds = (target - now).total_seconds()
            logger.info(
                "다음 리포트 배치 대기",
                extra={
                    "action": "report_batch_scheduled",
                    "run_at": target.isoformat(),
                    "wait_seconds": round(wait_seconds),
                },
            )
            try:
                await asyncio.sleep(wait_seconds)
            except asyncio.CancelledError:
                raise

            await self.run_batch()

    async def run_batch(self) -> int:
        """감시 중인 모든 가구의 어제 리포트를 만든다. 생성에 성공한 가구 수를 반환한다.

        가구 하나가 실패해도 나머지는 돈다 — LLM 타임아웃이나 삭제된 노인(I6 404) 하나로
        그날 전체 리포트가 사라지면 안 된다.

        I3는 `(노인, 날짜)` UPSERT라 재실행이 안전하다. 같은 날 두 번 돌아도 행은 하나이고
        FCM은 신규 생성일 때만 나간다.
        """
        report_date = default_report_date()
        care_target_ids = list_care_target_ids(self._config.notifi_registry_root)

        if not care_target_ids:
            logger.info(
                "리포트 대상 가구가 없다 — 배치를 건너뛴다",
                extra={"action": "report_batch_empty", "report_date": str(report_date)},
            )
            return 0

        succeeded = 0
        for care_target_id in care_target_ids:
            try:
                await run_daily_report(care_target_id, report_date)
                succeeded += 1
            except Exception as exc:  # noqa: BLE001 - 한 가구의 실패가 배치를 멈추면 안 된다
                logger.error(
                    "리포트 생성 실패 — 다음 가구로 넘어간다",
                    extra={
                        "action": "report_generation_failed",
                        "care_target_id": care_target_id,
                        "report_date": str(report_date),
                        "error": str(exc),
                    },
                    exc_info=True,
                )

        logger.info(
            "리포트 배치 완료",
            extra={
                "action": "report_batch_done",
                "report_date": str(report_date),
                "succeeded": succeeded,
                "total": len(care_target_ids),
            },
        )
        return succeeded
