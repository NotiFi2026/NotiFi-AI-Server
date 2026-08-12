from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class RiskLevel(str, Enum):
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"


class EventType(str, Enum):
    FALL = "FALL"
    INACTIVITY = "INACTIVITY"
    RESPIRATION_ABNORMAL = "RESPIRATION_ABNORMAL"
    ANOMALY = "ANOMALY"
    SENSOR_ERROR = "SENSOR_ERROR"
    NORMAL = "NORMAL"


class VoiceResponseResult(str, Enum):
    USER_OK = "USER_OK"
    USER_NEEDS_HELP = "USER_NEEDS_HELP"
    NO_RESPONSE = "NO_RESPONSE"
    UNCLEAR = "UNCLEAR"


class NotificationLevel(str, Enum):
    INFO = "INFO"
    EMERGENCY = "EMERGENCY"
    DAILY_REPORT = "DAILY_REPORT"
    SYSTEM = "SYSTEM"


class ModelResult(BaseModel):
    care_target_id: int
    device_id: Optional[int] = None
    event_type: EventType
    # AI v1 17행동 세부 분류(대문자). I1의 정식 필드로 나간다 —
    # features dict에 실어 보내면 키 이름이 어긋나도 조용히 null로 적재된다.
    activity_class: Optional[str] = None
    label: str
    risk_level: RiskLevel
    confidence: float
    risk_score: int
    model_version: str
    detected_at: datetime
    context_features: Optional[dict[str, Any]] = None
    features: Optional[dict[str, Any]] = None
    language: str = "ko"


class GuardianMessage(BaseModel):
    title: str
    body: str
    recommendation: str


class AgentContext(BaseModel):
    model_result: ModelResult
    escalation_id: Optional[int] = None
    sensing_event_id: Optional[int] = None
    risk_assessment_id: Optional[int] = None
    escalation_triggered: bool = False
    voice_response_result: Optional[VoiceResponseResult] = None
    stt_text: Optional[str] = None
    guardian_message: Optional[GuardianMessage] = None
    notification_level: Optional[NotificationLevel] = None
    escalation_continued: bool = False


class DailyReportMetrics(BaseModel):
    warning_event_count: int = 0
    danger_event_count: int = 0
    safe_class_counts: dict[str, int] = {}


class DailyReportInput(BaseModel):
    care_target_id: int
    report_date: str
    metrics: DailyReportMetrics


class ReportTag(str, Enum):
    RISK_EVENT = "risk_event"


class DailyReportSection(BaseModel):
    tag: ReportTag
    risk_level: RiskLevel
    title: str
    body: str
    recommended_action: Optional[str] = None


class DailyReportOutput(BaseModel):
    care_target_id: int
    report_date: str
    sections: list[DailyReportSection]
    metrics: DailyReportMetrics
    generated_at: datetime


