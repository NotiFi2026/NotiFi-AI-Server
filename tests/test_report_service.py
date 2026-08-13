"""일일 리포트 생성 — I6 조회 → LLM → I3 적재. OpenAI·Spring·모델 없이 돈다."""
from datetime import date, datetime, timezone

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIError

from app.agent import report_service
from app.agent.schemas import (
    DailyReportMetrics,
    DailyReportOutput,
    DailyReportSection,
    ReportTag,
    RiskLevel,
)
from app.config import settings
from main import app

KEY = settings.spring_internal_key


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def i6_response(**overrides):
    """Spring I6가 실제로 돌려주는 형태 — activity_class_counts는 대문자 17종."""
    body = {
        "care_target_id": 18,
        "date": "2026-08-12",
        "warning_event_count": 3,
        "danger_event_count": 1,
        "activity_class_counts": {"WALKING": 120, "SITTING_STILL": 340},
    }
    body.update(overrides)
    return body


def fake_output(care_target_id=18, report_date="2026-08-12"):
    return DailyReportOutput(
        care_target_id=care_target_id,
        report_date=report_date,
        sections=[
            DailyReportSection(
                tag=ReportTag.RISK_EVENT,
                risk_level=RiskLevel.DANGER,
                title="낙상이 감지됐어요",
                body="오후에 넘어지신 순간이 있었어요.",
                recommended_action="전화로 안부를 확인해 주세요.",
            )
        ],
        metrics=DailyReportMetrics(warning_event_count=3, danger_event_count=1),
        generated_at=datetime(2026, 8, 13, 0, 10, tzinfo=timezone.utc),
    )


# ── I6 → 생성기 입력 변환 ────────────────────────────────────────────────────


def test_metrics_lowercases_activity_class_keys():
    """프롬프트 생성기가 모델 표기(소문자)를 전제하므로 경계에서 맞춘다."""
    metrics = report_service._to_metrics(i6_response())

    assert metrics.safe_class_counts == {"walking": 120, "sitting_still": 340}


def test_metrics_drops_non_safe_classes():
    """낙상·불안정 보행이 '오늘의 정상 활동'으로 서술되면 안 된다."""
    metrics = report_service._to_metrics(i6_response(activity_class_counts={
        "WALKING": 120,
        "UNSTABLE_WALKING": 2,       # warning
        "FALL_FROM_STANDING": 1,     # danger
        "BED_FALL": 1,               # danger
    }))

    assert metrics.safe_class_counts == {"walking": 120}


def test_metrics_keeps_absence():
    """absence는 safe 9종이다. 프롬프트 문장에서 빼는 건 생성기 몫이지 여기가 아니다."""
    metrics = report_service._to_metrics(i6_response(activity_class_counts={"ABSENCE": 40}))

    assert metrics.safe_class_counts == {"absence": 40}


def test_metrics_passes_event_counts_through():
    metrics = report_service._to_metrics(i6_response())

    assert metrics.warning_event_count == 3
    assert metrics.danger_event_count == 1


def test_metrics_handles_empty_day():
    """이벤트가 없는 날도 리포트는 생성된다 — 카운트는 0, 분류는 빈 dict."""
    metrics = report_service._to_metrics({
        "care_target_id": 18, "date": "2026-08-12",
        "warning_event_count": 0, "danger_event_count": 0, "activity_class_counts": {},
    })

    assert metrics.warning_event_count == 0
    assert metrics.safe_class_counts == {}


# ── 대상일 산정 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "now_utc, expected",
    [
        # KST 09:30 (같은 날) → 어제는 08-12
        (datetime(2026, 8, 13, 0, 30, tzinfo=timezone.utc), date(2026, 8, 12)),
        # KST 익일 00:30 — UTC로 계산하면 08-11이 나온다. 여기가 틀리면 I6 일 경계와 어긋난다
        (datetime(2026, 8, 12, 15, 30, tzinfo=timezone.utc), date(2026, 8, 12)),
        # KST 08:59 — UTC 날짜가 아직 안 넘어간 시각
        (datetime(2026, 8, 12, 23, 59, tzinfo=timezone.utc), date(2026, 8, 12)),
    ],
)
def test_default_report_date_uses_kst(now_utc, expected):
    assert report_service.default_report_date(now_utc) == expected


# ── 배선 ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def wired(monkeypatch):
    """I6 조회·LLM 생성·I3 적재를 전부 대체하고 호출 인자를 기록한다."""
    calls: dict[str, object] = {}

    async def fake_get_metrics(care_target_id, report_date):
        calls["i6"] = (care_target_id, report_date)
        return i6_response()

    async def fake_generate(report_input):
        calls["generate"] = report_input
        return fake_output()

    async def fake_save(**kwargs):
        calls["i3"] = kwargs

    monkeypatch.setattr(report_service.spring_client, "get_daily_metrics", fake_get_metrics)
    monkeypatch.setattr(report_service, "generate_daily_report_summary", fake_generate)
    monkeypatch.setattr(report_service.spring_client, "save_daily_report", fake_save)
    return calls


@pytest.mark.asyncio
async def test_run_wires_i6_to_generator_to_i3(wired):
    await report_service.run_daily_report(18, date(2026, 8, 12))

    assert wired["i6"] == (18, "2026-08-12")
    # 생성기는 I6에서 변환된 metrics를 받는다
    assert wired["generate"].metrics.safe_class_counts == {"walking": 120, "sitting_still": 340}
    assert wired["i3"]["care_target_id"] == 18
    assert wired["i3"]["report_date"] == "2026-08-12"


@pytest.mark.asyncio
async def test_run_forwards_generated_at(wired):
    """Spring은 생략 시 수신 시각으로 채운다 — 생성 시각을 아는 쪽이 넘겨야 정확하다."""
    await report_service.run_daily_report(18, date(2026, 8, 12))

    assert wired["i3"]["generated_at"] == datetime(2026, 8, 13, 0, 10, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_run_serializes_sections_to_plain_json(wired):
    """enum이 객체째 실려 나가면 httpx 직렬화에서 터진다."""
    await report_service.run_daily_report(18, date(2026, 8, 12))

    section = wired["i3"]["sections"][0]
    assert section["tag"] == "risk_event"
    assert section["risk_level"] == "danger"
    assert section["recommended_action"] == "전화로 안부를 확인해 주세요."


@pytest.mark.asyncio
async def test_run_propagates_failure(monkeypatch):
    """단건 실행에서 실패는 호출자에게 그대로 보여야 한다 — 삼키지 않는다."""
    async def boom(care_target_id, report_date):
        raise RuntimeError("I6 down")

    monkeypatch.setattr(report_service.spring_client, "get_daily_metrics", boom)

    with pytest.raises(RuntimeError):
        await report_service.run_daily_report(18, date(2026, 8, 12))


# ── 엔드포인트 ───────────────────────────────────────────────────────────────


def test_endpoint_requires_internal_key(client):
    res = client.post("/internal/agent/reports/run", json={"care_target_id": 18})

    assert res.status_code == 401


def test_endpoint_returns_generated_sections(client, wired):
    res = client.post(
        "/internal/agent/reports/run",
        json={"care_target_id": 18, "report_date": "2026-08-12"},
        headers={"X-Internal-Key": KEY},
    )

    assert res.status_code == 200
    body = res.json()
    # 데모에서 눌러 생성 문장을 바로 확인하는 것이 이 엔드포인트의 목적이다
    assert body["sections"][0]["title"] == "낙상이 감지됐어요"
    assert body["sections"][0]["risk_level"] == "danger"


def raise_from_service(monkeypatch, exc: Exception):
    async def boom(care_target_id, report_date):
        raise exc

    monkeypatch.setattr(report_service, "run_daily_report", boom)


def spring_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "http://spring/internal/v1/care-targets/18/daily-metrics")
    return httpx.HTTPStatusError(
        "boom", request=request, response=httpx.Response(status, request=request)
    )


def test_endpoint_maps_unknown_care_target_to_404(client, monkeypatch):
    """500으로 흘리면 ID 오타인지 서버가 죽은 건지 구분이 안 된다."""
    raise_from_service(monkeypatch, spring_error(404))

    res = client.post(
        "/internal/agent/reports/run",
        json={"care_target_id": 99999},
        headers={"X-Internal-Key": KEY},
    )

    assert res.status_code == 404
    assert "99999" in res.json()["detail"]


def test_endpoint_maps_internal_key_mismatch_to_actionable_502(client, monkeypatch):
    """설정 한 줄 문제이므로 무엇을 고쳐야 하는지 응답에 담는다."""
    raise_from_service(monkeypatch, spring_error(401))

    res = client.post(
        "/internal/agent/reports/run",
        json={"care_target_id": 18},
        headers={"X-Internal-Key": KEY},
    )

    assert res.status_code == 502
    assert "INTERNAL_API_KEY" in res.json()["detail"]


def test_endpoint_maps_spring_unreachable_to_502(client, monkeypatch):
    raise_from_service(monkeypatch, httpx.ConnectError("connection refused"))

    res = client.post(
        "/internal/agent/reports/run",
        json={"care_target_id": 18},
        headers={"X-Internal-Key": KEY},
    )

    assert res.status_code == 502


def test_endpoint_maps_llm_failure_to_502(client, monkeypatch):
    """의존 서비스 장애를 우리 버그(500)와 구분한다."""
    raise_from_service(monkeypatch, APIError("rate limited", request=None, body=None))

    res = client.post(
        "/internal/agent/reports/run",
        json={"care_target_id": 18},
        headers={"X-Internal-Key": KEY},
    )

    assert res.status_code == 502


def test_endpoint_defaults_report_date_to_yesterday(client, wired, monkeypatch):
    monkeypatch.setattr(report_service, "default_report_date", lambda: date(2026, 8, 11))

    res = client.post(
        "/internal/agent/reports/run",
        json={"care_target_id": 18},
        headers={"X-Internal-Key": KEY},
    )

    assert res.status_code == 200
    assert wired["i6"] == (18, "2026-08-11")
