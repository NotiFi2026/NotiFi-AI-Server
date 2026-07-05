"""risk_level에 따라 대응 정책을 선택한다."""

_POLICY_MAP = {
    "safe": "safe_record_only",
    "warning": "warning_record_report",
    "danger": "voice_check_required",
}


def select_policy(risk_level: str) -> str:
    return _POLICY_MAP.get(risk_level.lower(), "safe_record_only")
