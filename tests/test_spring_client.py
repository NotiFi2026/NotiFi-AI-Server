"""Spring 클라이언트 — 요청 형태와 로깅까지 실제로 태운다.

`report_service` 테스트는 이 함수들을 통째로 목킹하므로 여기서만 잡히는 결함이 있다.
실제로 `extra={"created": ...}` 가 LogRecord 예약 속성과 충돌해 **적재가 끝난 뒤**
로깅에서 500이 나는 사고가 있었다 — 호출자는 실패로 보는데 데이터는 들어가 있었다.
"""
from datetime import date, datetime, timezone

import httpx
import pytest

from app.clients import spring_client


def stub_client(handler, monkeypatch):
    """httpx.AsyncClient를 MockTransport로 갈아끼운다 — 네트워크 없이 요청을 관찰한다."""
    original = httpx.AsyncClient

    def factory(*args, **kwargs):
        return original(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(spring_client.httpx, "AsyncClient", factory)


@pytest.mark.asyncio
async def test_get_daily_metrics_sends_date_and_key(monkeypatch):
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("X-Internal-Key")
        return httpx.Response(200, json={"success": True, "data": {
            "care_target_id": 18, "date": "2026-08-12",
            "warning_event_count": 0, "danger_event_count": 2,
            "activity_class_counts": {"WALKING": 2},
        }})

    stub_client(handler, monkeypatch)
    data = await spring_client.get_daily_metrics(18, "2026-08-12")

    assert "/internal/v1/care-targets/18/daily-metrics" in seen["url"]
    assert "date=2026-08-12" in seen["url"]
    assert seen["key"] == spring_client.settings.spring_internal_key
    assert data["danger_event_count"] == 2


@pytest.mark.asyncio
async def test_save_daily_report_includes_generated_at(monkeypatch):
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"success": True, "data": {
            "daily_report_id": 1, "care_target_id": 18, "created": True,
        }})

    stub_client(handler, monkeypatch)
    await spring_client.save_daily_report(
        care_target_id=18,
        report_date="2026-08-12",
        sections=[{"tag": "risk_event", "risk_level": "danger", "title": "t", "body": "b"}],
        metrics={"warning_event_count": 0, "danger_event_count": 2},
        generated_at=datetime(2026, 8, 13, 0, 10, tzinfo=timezone.utc),
    )

    assert seen["body"]["generated_at"] == "2026-08-13T00:10:00+00:00"


@pytest.mark.asyncio
async def test_save_daily_report_omits_generated_at_when_absent(monkeypatch):
    """생략하면 Spring이 수신 시각으로 채운다 — null을 보내면 검증에 걸린다."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"success": True, "data": {"daily_report_id": 1}})

    stub_client(handler, monkeypatch)
    await spring_client.save_daily_report(
        care_target_id=18, report_date="2026-08-12", sections=[{}], metrics={},
    )

    assert "generated_at" not in seen["body"]


@pytest.mark.parametrize("status", [200, 201])
@pytest.mark.asyncio
async def test_save_daily_report_logs_without_reserved_key_clash(monkeypatch, status):
    """신규(201)·갱신(200) 어느 쪽이든 로깅이 터지지 않아야 한다.

    LogRecord 예약 속성(`created` 등)을 extra 키로 쓰면 적재 성공 후 KeyError가 난다.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"success": True, "data": {"daily_report_id": 1}})

    stub_client(handler, monkeypatch)
    await spring_client.save_daily_report(
        care_target_id=18, report_date="2026-08-12", sections=[{}], metrics={},
    )


@pytest.mark.asyncio
async def test_get_daily_metrics_raises_on_error_status(monkeypatch):
    """404(없는 노인)를 조용히 삼키면 빈 리포트가 생성된다."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"success": False, "error": {"code": "CARE_TARGET_NOT_FOUND"}})

    stub_client(handler, monkeypatch)
    with pytest.raises(httpx.HTTPStatusError):
        await spring_client.get_daily_metrics(99999, "2026-08-12")
