"""일일 리포트 스케줄러 — 실행 시각 계산·대상 가구 수집·실패 격리. LLM·Spring 없이 돈다."""
from datetime import datetime, timezone

import pytest

from app.agent import report_scheduler
from app.agent.report_scheduler import (
    KST,
    ReportScheduler,
    list_care_target_ids,
    next_run_at,
)
from app.config import settings


# ── 다음 실행 시각 ────────────────────────────────────────────────────────────

def test_next_run_is_today_when_target_still_ahead():
    now = datetime(2026, 8, 14, 1, 0, tzinfo=timezone.utc)  # KST 10:00
    run_at = next_run_at(now, hour=20, minute=0).astimezone(KST)

    assert (run_at.year, run_at.month, run_at.day) == (2026, 8, 14)
    assert (run_at.hour, run_at.minute) == (20, 0)


def test_next_run_rolls_to_tomorrow_after_target_passed():
    """KST 23:50에 08:00을 계산하면 내일이어야 한다 — 자정을 넘는 경계."""
    now = datetime(2026, 8, 14, 14, 50, tzinfo=timezone.utc)  # KST 23:50
    run_at = next_run_at(now, hour=8, minute=0).astimezone(KST)

    assert (run_at.year, run_at.month, run_at.day) == (2026, 8, 15)
    assert (run_at.hour, run_at.minute) == (8, 0)


def test_next_run_rolls_over_when_exactly_on_target():
    """정각에 재시작해도 그날 것을 또 돌리지 않는다 — 이미 돈 배치를 반복하면 안 된다."""
    now = datetime(2026, 8, 13, 23, 0, tzinfo=timezone.utc)  # KST 08-14 08:00 정각
    run_at = next_run_at(now, hour=8, minute=0).astimezone(KST)

    assert run_at.day == 15


# ── 대상 가구 수집 ────────────────────────────────────────────────────────────

def test_lists_only_care_prefixed_devices(tmp_path):
    """레지스트리엔 규약을 안 따르는 항목도 있다(수동 등록 home-001 등)."""
    (tmp_path / "care-34").mkdir()
    (tmp_path / "care-7").mkdir()
    (tmp_path / "home-001").mkdir()  # 노인 ID를 알 수 없다
    (tmp_path / "care-abc").mkdir()  # 숫자가 아니다
    (tmp_path / "stray.json").write_text("{}", encoding="utf-8")  # 디렉터리가 아니다

    assert sorted(list_care_target_ids(str(tmp_path))) == [7, 34]


def test_missing_registry_root_is_not_an_error(tmp_path):
    """레지스트리가 아직 없는 서버(디바이스 등록 전)에서도 그냥 비어 있어야 한다."""
    assert list_care_target_ids(str(tmp_path / "없는경로")) == []


# ── 배치 실행 ────────────────────────────────────────────────────────────────

@pytest.fixture
def registry(tmp_path, monkeypatch):
    (tmp_path / "care-1").mkdir()
    (tmp_path / "care-2").mkdir()
    (tmp_path / "care-3").mkdir()
    monkeypatch.setattr(settings, "notifi_registry_root", str(tmp_path))
    return tmp_path


@pytest.mark.asyncio
async def test_batch_runs_every_household(registry, monkeypatch):
    called: list[int] = []

    async def fake_run(care_target_id, report_date):
        called.append(care_target_id)

    monkeypatch.setattr(report_scheduler, "run_daily_report", fake_run)

    succeeded = await ReportScheduler(settings).run_batch()

    assert called == [1, 2, 3]
    assert succeeded == 3


@pytest.mark.asyncio
async def test_one_household_failure_does_not_stop_the_batch(registry, monkeypatch):
    """LLM 타임아웃이나 삭제된 노인(I6 404) 하나로 그날 전체 리포트가 사라지면 안 된다."""
    called: list[int] = []

    async def fake_run(care_target_id, report_date):
        called.append(care_target_id)
        if care_target_id == 2:
            raise RuntimeError("LLM timeout")

    monkeypatch.setattr(report_scheduler, "run_daily_report", fake_run)

    succeeded = await ReportScheduler(settings).run_batch()

    assert called == [1, 2, 3]  # 2에서 멈추지 않았다
    assert succeeded == 2


@pytest.mark.asyncio
async def test_batch_skips_when_no_household_registered(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "notifi_registry_root", str(tmp_path))

    async def fail(*args, **kwargs):
        raise AssertionError("대상이 없으면 부르면 안 된다")

    monkeypatch.setattr(report_scheduler, "run_daily_report", fail)

    assert await ReportScheduler(settings).run_batch() == 0


# ── 태스크 수명 ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_start_holds_a_strong_task_reference(monkeypatch):
    """create_task 반환값을 안 붙잡으면 루프가 약한 참조만 들고 있어 실행 중 GC된다."""
    monkeypatch.setattr(settings, "notifi_report_hour_kst", 8)
    scheduler = ReportScheduler(settings)

    scheduler.start()
    assert scheduler._task is not None
    assert not scheduler._task.done()

    await scheduler.stop()
    assert scheduler._task is None
