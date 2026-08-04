"""일일 리포트 자연어 생성 테스트 — python test_daily_report.py"""
import asyncio
import json

from app.agent.message_generator import generate_daily_report_summary
from app.agent.schemas import DailyReportInput, DailyReportMetrics


async def main() -> None:
    report_input = DailyReportInput(
        care_target_id=1,
        report_date="2026-07-02",
        metrics=DailyReportMetrics(
            warning_event_count=2,
            danger_event_count=1,
        ),
    )

    print("▶ 일일 리포트 생성 중...")
    print("=" * 60)

    result = await generate_daily_report_summary(report_input)

    print(json.dumps({
        "care_target_id":  result.care_target_id,
        "report_date":     result.report_date,
        "sections":        [s.model_dump() for s in result.sections],
        "generated_at":    result.generated_at.isoformat(),
        "metrics": {
            "warning_event_count":       result.metrics.warning_event_count,
            "danger_event_count":        result.metrics.danger_event_count,
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
