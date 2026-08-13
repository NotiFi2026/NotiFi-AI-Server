"""일일 리포트 생성 — I6 조회 → LLM 생성 → I3 적재.

에이전트 서버는 저장소가 없다. 하루치 카운트는 Spring이 단일 출처로 들고 있으므로
I6로 읽어와 문장만 만들고 다시 Spring에 넣는다. 이 모듈은 그 세 단계를 잇기만 한다.
"""
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from app.agent.message_generator import generate_daily_report_summary
from app.agent.schemas import DailyReportInput, DailyReportMetrics, DailyReportOutput
from app.clients import spring_client
from app.common.logging_config import logger

#: 리포트의 "하루" 경계. Spring I6도 같은 존으로 자르므로 여기서 어긋나면
#: 요청한 날짜와 집계된 날짜가 달라진다.
KST = ZoneInfo("Asia/Seoul")

#: 모델 17행동 중 safe 9종(모델이 쓰는 소문자 표기).
#:
#: notifi_ai에서 가져오지 않는다 — `notifi_ai/__init__.py`가 torch를 import해서
#: 상수 하나만 끌어와도 모델 미설치 환경의 부팅이 깨진다. 이 목록은 모델 내부가 아니라
#: api-spec에 공개된 계약이므로, `pipeline._RISK_ID_TO_EVENT`와 같은 방식으로 복제한다.
_SAFE_ACTIVITY_CLASSES = frozenset({
    "walking",
    "standing_still",
    "sitting_still",
    "lying_still",
    "lie_to_stand",
    "stand_to_lie_normal",
    "absence",
    "sit_to_stand",
    "stand_to_sit",
})


def default_report_date(now: datetime | None = None) -> date:
    """리포트 대상일 = KST 기준 어제.

    UTC로 계산하면 한국시간 09:00에 날짜가 넘어가 I6의 일 경계와 어긋난다.
    """
    moment = now or datetime.now(timezone.utc)
    return moment.astimezone(KST).date() - timedelta(days=1)


def _to_metrics(data: dict[str, Any]) -> DailyReportMetrics:
    """I6 응답 → 리포트 생성기 입력.

    두 가지를 여기서 맞춘다. 안 하면 프롬프트가 조용히 틀어진다.

    1. **대소문자** — I6는 Spring 저장 표기(대문자)로 주는데, 프롬프트 생성기는
       모델 표기(소문자)를 전제로 `absence`를 걸러낸다. 대문자로 넘기면 그 필터가
       비껴가 부재 시간이 "오늘의 정상 활동"으로 서술된다.
    2. **safe 필터** — I6는 17종 전부를 준다. 낙상까지 넘기면 같은 프롬프트가 위에서는
       "위험 이벤트 1건", 아래에서는 "정상 활동: fall_from_standing 1회"라고 말하게 된다.
    """
    counts: dict[str, int] = {}
    dropped: list[str] = []
    for name, count in (data.get("activity_class_counts") or {}).items():
        key = name.lower()
        if key in _SAFE_ACTIVITY_CLASSES:
            counts[key] = count
        else:
            dropped.append(key)

    if dropped:
        # 버린 것을 조용히 넘기지 않는다 — 모르는 클래스가 늘면 여기서 보인다
        logger.debug(
            "safe 외 activity_class 제외",
            extra={"action": "report_metrics_filtered", "dropped": sorted(dropped)},
        )

    return DailyReportMetrics(
        warning_event_count=data.get("warning_event_count", 0),
        danger_event_count=data.get("danger_event_count", 0),
        safe_class_counts=counts,
    )


async def run_daily_report(care_target_id: int, report_date: date) -> DailyReportOutput:
    """한 가구의 하루치 리포트를 생성해 적재하고, 생성 결과를 반환한다.

    예외를 삼키지 않는다 — 단건 실행에서는 실패가 호출자에게 그대로 보여야 하고,
    여러 가구를 도는 쪽(스케줄러)이 가구별로 감싸는 것이 맞는 층이다.
    """
    report_date_str = report_date.isoformat()
    logger.info(
        "일일 리포트 생성 시작",
        extra={
            "action": "daily_report_run_started",
            "care_target_id": care_target_id,
            "report_date": report_date_str,
        },
    )

    metrics_data = await spring_client.get_daily_metrics(care_target_id, report_date_str)
    metrics = _to_metrics(metrics_data)

    output = await generate_daily_report_summary(
        DailyReportInput(
            care_target_id=care_target_id,
            report_date=report_date_str,
            metrics=metrics,
        )
    )

    await spring_client.save_daily_report(
        care_target_id=care_target_id,
        report_date=report_date_str,
        # mode="json"이라야 tag·risk_level enum이 문자열로 풀린다
        sections=[section.model_dump(mode="json") for section in output.sections],
        metrics=output.metrics.model_dump(mode="json"),
        generated_at=output.generated_at,
    )

    logger.info(
        "일일 리포트 생성 완료",
        extra={
            "action": "daily_report_run_completed",
            "care_target_id": care_target_id,
            "report_date": report_date_str,
            "section_count": len(output.sections),
        },
    )
    return output
