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
    activity_level: float = 0.0
    activity_change_percent: float = 0.0
    total_inactivity_minutes: int = 0
    longest_inactive_minutes: int = 0
    warning_event_count: int = 0
    danger_event_count: int = 0
    respiration_abnormal_count: int = 0
    avg_breathing_rate: float = 0.0


class DailyReportInput(BaseModel):
    care_target_id: int
    report_date: str
    metrics: DailyReportMetrics


class DailyReportOutput(BaseModel):
    care_target_id: int
    report_date: str
    summary_text: str
    metrics: DailyReportMetrics
    generated_at: datetime


