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
            activity_level=0.32,
            activity_change_percent=-18.5,
            total_inactivity_minutes=210,
            longest_inactive_minutes=95,
            warning_event_count=2,
            danger_event_count=1,
            respiration_abnormal_count=0,
            avg_breathing_rate=16.4,
        ),
    )

    print("▶ 일일 리포트 생성 중...")
    print("=" * 60)

    result = await generate_daily_report_summary(report_input)

    print(json.dumps({
        "care_target_id":  result.care_target_id,
        "report_date":     result.report_date,
        "summary_text":    result.summary_text,
        "generated_at":    result.generated_at.isoformat(),
        "metrics": {
            "activity_level":            result.metrics.activity_level,
            "activity_change_percent":   result.metrics.activity_change_percent,
            "total_inactivity_minutes":  result.metrics.total_inactivity_minutes,
            "longest_inactive_minutes":  result.metrics.longest_inactive_minutes,
            "warning_event_count":       result.metrics.warning_event_count,
            "danger_event_count":        result.metrics.danger_event_count,
            "respiration_abnormal_count":result.metrics.respiration_abnormal_count,
            "avg_breathing_rate":        result.metrics.avg_breathing_rate,
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
