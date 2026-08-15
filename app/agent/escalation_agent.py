"""응급 대응 흐름 — LangGraph StateGraph.

LangSmith 추적: LANGCHAIN_TRACING_V2=true, LANGCHAIN_API_KEY=<key>,
LANGCHAIN_PROJECT=notifi-ai 를 .env에 설정하면 자동 활성화된다.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from app.agent import response_classifier
from app.agent.message_generator import generate_guardian_message
from app.agent.payload_builder import build_sensing_event_payload
from app.agent.response_policy import select_policy
from app.agent.schemas import EventType, GuardianMessage, NotificationLevel, RiskLevel, VoiceResponseResult, ModelResult
from app.clients import cartesia_client, spring_client, whisper_client
from app.common.logging_config import logger
from app.config import settings
from app.model.pipeline import to_iso_ms


# ── State ─────────────────────────────────────────────────────────────────────

class EscalationState(TypedDict):
    # 모델 결과 (입력)
    care_target_id: int
    device_id: Optional[int]
    event_type: str
    activity_class: Optional[str]
    label: str
    risk_level: str
    confidence: float
    risk_score: int
    model_version: str
    detected_at: str
    context_features: Optional[dict]
    features: Optional[dict]
    language: str
    # I1 응답
    sensing_event_id: Optional[int]
    risk_assessment_id: Optional[int]
    escalation_id: Optional[int]
    escalation_triggered: bool
    # 대응 정책
    response_policy: Optional[str]
    # 음성 확인
    voice_check_executed_at: Optional[str]
    stt_text: Optional[str]
    voice_response_result: Optional[str]
    retry_count: int
    # 보호자 알림
    guardian_message: Optional[dict]
    notification_level: Optional[str]
    escalation_continued: bool
    # step 순서 추적
    next_step_order: int


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def _submit_sensing_event(state: EscalationState) -> dict:
    if state.get("sensing_event_id") is not None:
        # 파이프라인이 I5를 위해 이미 I1을 보내고 결과를 넘겨준 경우 — 재전송하지 않는다
        return {}

    logger.info(
        "I1 요청",
        extra={
            "action": "spring_i1_requested",
            "care_target_id": state["care_target_id"],
            "risk_level": state["risk_level"],
            "event_type": state["event_type"],
        },
    )
    payload = build_sensing_event_payload(state)
    data = await spring_client.send_sensing_event(payload)
    logger.info(
        "I1 완료",
        extra={
            "action": "spring_i1_completed",
            "sensing_event_id": data.get("sensing_event_id"),
            "escalation_triggered": data.get("escalation_triggered"),
            "escalation_id": data.get("escalation_id"),
        },
    )
    if data.get("escalation_triggered") and data.get("escalation_id"):
        logger.info(
            "에스컬레이션 생성",
            extra={
                "action": "escalation_started",
                "escalation_id": data.get("escalation_id"),
                "care_target_id": state["care_target_id"],
            },
        )
    return {
        "sensing_event_id": data.get("sensing_event_id"),
        "risk_assessment_id": data.get("risk_assessment_id"),
        "escalation_triggered": data.get("escalation_triggered", False),
        "escalation_id": data.get("escalation_id"),
    }


async def _select_response_policy(state: EscalationState) -> dict:
    policy = select_policy(state["risk_level"])

    if (
        policy == "voice_check_required"
        and state.get("escalation_id")
        and not state.get("escalation_triggered")
    ):
        # Spring이 새로 만들지 않고 진행 중인 건에 합쳤다 — 그 건에는 이미 대응이 붙어 있다.
        # 여기서 또 음성 확인을 시작하면 어르신은 "괜찮다고 했는데 또 물어본다"를 겪는다.
        # (id가 있으면서 triggered=false인 경우는 이 재사용뿐이다. 멱등 재적재는 true를 준다.)
        logger.info(
            "이미 진행 중인 대응 — 음성 확인 생략",
            extra={
                "action": "escalation_reused",
                "care_target_id": state["care_target_id"],
                "escalation_id": state.get("escalation_id"),
            },
        )
        policy = "safe_record_only"

    if policy == "voice_check_required" and not state.get("escalation_id"):
        # 멱등 재적재 등으로 Spring이 escalation_id를 주지 않은 경우.
        # 그대로 진행하면 /escalations/None/steps 로 POST한다.
        logger.error(
            "danger인데 escalation_id 없음 — 단계 기록 없이 종료",
            extra={
                "action": "escalation_id_missing",
                "care_target_id": state["care_target_id"],
                "sensing_event_id": state.get("sensing_event_id"),
            },
        )
        policy = "safe_record_only"

    logger.info(
        "대응 정책 선택",
        extra={
            "action": "response_policy_selected",
            "care_target_id": state["care_target_id"],
            "event_type": state["event_type"],
            "risk_level": state["risk_level"],
            "response_policy": policy,
        },
    )
    return {"response_policy": policy}


def _route_by_risk_level(state: EscalationState) -> str:
    return state.get("response_policy") or "safe_record_only"


async def _safe_record_only(state: EscalationState) -> dict:
    logger.info(
        "safe — 기록 후 종료",
        extra={
            "action": "model_result_received",
            "care_target_id": state["care_target_id"],
            "risk_level": "safe",
            "sensing_event_id": state.get("sensing_event_id"),
        },
    )
    return {}


async def _warning_record_report(state: EscalationState) -> dict:
    logger.info(
        "warning — Spring이 보호자 주의 알림 처리",
        extra={
            "action": "model_result_received",
            "care_target_id": state["care_target_id"],
            "risk_level": "warning",
            "sensing_event_id": state.get("sensing_event_id"),
        },
    )
    return {}


async def _danger_voice_check(state: EscalationState) -> dict:
    """VOICE_CHECK 단계: EXECUTED step 기록(최초 1회), TTS 출력, STT 수집."""
    now = datetime.now(timezone.utc)
    is_retry = state["retry_count"] > 0
    language = state.get("language", "ko")

    prompt_map = cartesia_client._RETRY_PROMPT_MAP if is_retry else cartesia_client._INITIAL_PROMPT_MAP
    prompt = prompt_map.get(language, prompt_map["ko"])

    if not is_retry:
        await spring_client.record_escalation_step(
            escalation_id=state["escalation_id"],
            step_type="VOICE_CHECK",
            step_order=state["next_step_order"],
            status="EXECUTED",
            executed_at=now,
            response_detail={
                "tts_provider": "cartesia",
                "language": language,
                "prompt": prompt,
            },
        )
        logger.info(
            "음성 확인 시작",
            extra={
                "action": "voice_check_executed",
                "escalation_id": state["escalation_id"],
                "step_order": state["next_step_order"],
                "language": language,
            },
        )

    await cartesia_client.speak(prompt, language=language)
    stt_text = await whisper_client.listen(timeout_seconds=10.0)

    updates: dict = {"stt_text": stt_text}
    if not is_retry:
        updates["voice_check_executed_at"] = now.isoformat()
    return updates


async def _classify_voice_response(state: EscalationState) -> dict:
    result = await response_classifier.classify(state.get("stt_text"))

    if result == VoiceResponseResult.UNCLEAR:
        if state["retry_count"] >= 1:
            # 재시도 후에도 UNCLEAR → NO_RESPONSE 처리
            logger.info(
                "재시도 후 UNCLEAR — NO_RESPONSE로 처리",
                extra={"action": "voice_response_classified", "result": "NO_RESPONSE_AFTER_UNCLEAR"},
            )
            return {"voice_response_result": VoiceResponseResult.NO_RESPONSE.value}
        # 첫 UNCLEAR: 재질문 예약
        return {"retry_count": state["retry_count"] + 1}

    return {"voice_response_result": result.value}


def _route_after_classify(state: EscalationState) -> str:
    """voice_response_result가 아직 없으면 재질문."""
    return "retry" if state.get("voice_response_result") is None else "continue"


async def _record_voice_check_response(state: EscalationState) -> dict:
    """VOICE_CHECK/RESPONDED 또는 NO_RESPONSE step 기록."""
    voice_result = VoiceResponseResult(state["voice_response_result"])
    executed_at = datetime.fromisoformat(state["voice_check_executed_at"])
    now = datetime.now(timezone.utc)

    status = "NO_RESPONSE" if voice_result == VoiceResponseResult.NO_RESPONSE else "RESPONDED"

    await spring_client.record_escalation_step(
        escalation_id=state["escalation_id"],
        step_type="VOICE_CHECK",
        step_order=state["next_step_order"],
        status=status,
        executed_at=executed_at,
        responded_at=now if status == "RESPONDED" else None,
        response_detail={
            "stt_provider": "whisper",
            "response_result": voice_result.value,
            "stt_text": state.get("stt_text") or "",
        },
    )
    logger.info(
        "음성 확인 응답 기록",
        extra={
            "action": "spring_i2_completed",
            "step_type": "VOICE_CHECK",
            "status": status,
        },
    )
    return {"next_step_order": state["next_step_order"] + 1}


async def _generate_guardian_message(state: EscalationState) -> dict:
    voice_result = VoiceResponseResult(state["voice_response_result"])

    msg = await generate_guardian_message(
        event_type=EventType(state["event_type"]),
        label=state["label"],
        risk_level=RiskLevel(state["risk_level"]),
        confidence=state["confidence"],
        response_result=voice_result,
        context_features=state.get("context_features"),
    )

    if voice_result == VoiceResponseResult.USER_OK:
        level = NotificationLevel.INFO
        escalation_continued = False
    else:
        level = NotificationLevel.EMERGENCY
        escalation_continued = True

    logger.info(
        "보호자 메시지 생성 완료",
        extra={
            "action": "guardian_message_generated",
            "notification_level": level.value,
            "escalation_continued": escalation_continued,
        },
    )
    return {
        "guardian_message": msg.model_dump(),
        "notification_level": level.value,
        "escalation_continued": escalation_continued,
    }


async def _submit_guardian_notify_step(state: EscalationState) -> dict:
    now = datetime.now(timezone.utc)
    voice_result = VoiceResponseResult(state["voice_response_result"])
    msg = GuardianMessage(**state["guardian_message"])
    step_order = state["next_step_order"]

    await spring_client.record_escalation_step(
        escalation_id=state["escalation_id"],
        step_type="GUARDIAN_NOTIFY",
        step_order=step_order,
        status="EXECUTED",
        executed_at=now,
        response_detail={
            "reason": voice_result.value,
            "notification_level": state["notification_level"],
            "escalation_continued": state["escalation_continued"],
        },
        guardian_message=msg,
    )
    logger.info(
        "보호자 알림 단계 기록 완료",
        extra={
            "action": "spring_i2_completed",
            "step_type": "GUARDIAN_NOTIFY",
            "escalation_id": state["escalation_id"],
            "notification_level": state["notification_level"],
        },
    )
    return {"next_step_order": step_order + 1}


def _route_after_guardian_notify(state: EscalationState) -> str:
    return "emergency_call" if state["escalation_continued"] else "finish"


async def _wait_and_emergency_call(state: EscalationState) -> dict:
    """보호자 확인 대기 후 EMERGENCY_CALL 진행.

    대기 중 보호자가 앱에서 확인(E3 resolve)하면 Spring이 step을 SKIPPED로
    강제 기록하므로, 응답의 escalation_status/status로 중단 여부를 확인한다.
    """
    delay = settings.emergency_call_delay_seconds
    logger.info(
        "긴급 연락 전 보호자 확인 대기",
        extra={
            "action": "emergency_call_waiting",
            "escalation_id": state["escalation_id"],
            "delay_seconds": delay,
        },
    )
    await asyncio.sleep(delay)

    data = await spring_client.record_escalation_step(
        escalation_id=state["escalation_id"],
        step_type="EMERGENCY_CALL",
        step_order=state["next_step_order"],
        status="EXECUTED",
        executed_at=datetime.now(timezone.utc),
    )

    if data.get("status") == "SKIPPED" or data.get("escalation_status") != "IN_PROGRESS":
        logger.info(
            "보호자 확인으로 긴급 연락 중단",
            extra={
                "action": "emergency_call_skipped",
                "escalation_id": state["escalation_id"],
                "escalation_status": data.get("escalation_status"),
            },
        )
    else:
        logger.info(
            "긴급 연락 단계 기록 완료",
            extra={"action": "spring_i2_completed", "step_type": "EMERGENCY_CALL"},
        )
    return {"next_step_order": state["next_step_order"] + 1}


async def _finish(state: EscalationState) -> dict:
    logger.info(
        "에스컬레이션 완료",
        extra={
            "action": "escalation_finished",
            "care_target_id": state["care_target_id"],
            "escalation_id": state.get("escalation_id"),
            "escalation_continued": state.get("escalation_continued"),
        },
    )
    return {}


# ── Graph 조립 ─────────────────────────────────────────────────────────────────

_builder = StateGraph(EscalationState)

_builder.add_node("submit_sensing_event", _submit_sensing_event)
_builder.add_node("select_response_policy", _select_response_policy)
_builder.add_node("safe_record_only", _safe_record_only)
_builder.add_node("warning_record_report", _warning_record_report)
_builder.add_node("danger_voice_check", _danger_voice_check)
_builder.add_node("classify_voice_response", _classify_voice_response)
_builder.add_node("record_voice_check_response", _record_voice_check_response)
_builder.add_node("generate_guardian_message", _generate_guardian_message)
_builder.add_node("submit_guardian_notify_step", _submit_guardian_notify_step)
_builder.add_node("wait_and_emergency_call", _wait_and_emergency_call)
_builder.add_node("finish", _finish)

_builder.set_entry_point("submit_sensing_event")
_builder.add_edge("submit_sensing_event", "select_response_policy")
_builder.add_conditional_edges(
    "select_response_policy",
    _route_by_risk_level,
    {
        "safe_record_only": "safe_record_only",
        "warning_record_report": "warning_record_report",
        "voice_check_required": "danger_voice_check",
    },
)
_builder.add_edge("safe_record_only", END)
_builder.add_edge("warning_record_report", END)
_builder.add_edge("danger_voice_check", "classify_voice_response")
_builder.add_conditional_edges(
    "classify_voice_response",
    _route_after_classify,
    {
        "retry": "danger_voice_check",
        "continue": "record_voice_check_response",
    },
)
_builder.add_edge("record_voice_check_response", "generate_guardian_message")
_builder.add_edge("generate_guardian_message", "submit_guardian_notify_step")
_builder.add_conditional_edges(
    "submit_guardian_notify_step",
    _route_after_guardian_notify,
    {
        "emergency_call": "wait_and_emergency_call",
        "finish": "finish",
    },
)
_builder.add_edge("wait_and_emergency_call", "finish")
_builder.add_edge("finish", END)

graph = _builder.compile()


def initial_state(
    model_result: ModelResult,
    prefetched: dict | None = None,
) -> EscalationState:
    """ModelResult를 그래프 state로 편다. I1 payload 조립도 이 형태를 입력으로 받는다."""
    return {
        "care_target_id": model_result.care_target_id,
        "device_id": model_result.device_id,
        "event_type": model_result.event_type.value,
        "activity_class": model_result.activity_class,
        "label": model_result.label,
        "risk_level": model_result.risk_level.value,
        "confidence": model_result.confidence,
        "risk_score": model_result.risk_score,
        "model_version": model_result.model_version,
        "detected_at": to_iso_ms(model_result.detected_at),
        "context_features": model_result.context_features,
        "features": model_result.features,
        "language": model_result.language,
        "sensing_event_id": (prefetched or {}).get("sensing_event_id"),
        "risk_assessment_id": (prefetched or {}).get("risk_assessment_id"),
        "escalation_id": (prefetched or {}).get("escalation_id"),
        "escalation_triggered": (prefetched or {}).get("escalation_triggered", False),
        "response_policy": None,
        "voice_check_executed_at": None,
        "stt_text": None,
        "voice_response_result": None,
        "retry_count": 0,
        "guardian_message": None,
        "notification_level": None,
        "escalation_continued": False,
        "next_step_order": 1,
    }


async def run(
    model_result: ModelResult,
    prefetched: dict | None = None,
) -> EscalationState:
    """모델 결과를 받아 에스컬레이션 흐름을 실행한다.

    prefetched: 호출자가 이미 I1을 보냈다면 그 응답(sensing_event_id·escalation_id 등).
    주면 I1을 다시 보내지 않는다 — 추론 파이프라인은 I5 전송을 위해 먼저 I1을 호출한다.
    """
    logger.info(
        "모델 결과 수신",
        extra={
            "action": "model_result_received",
            "care_target_id": model_result.care_target_id,
            "event_type": model_result.event_type.value,
            "risk_level": model_result.risk_level.value,
            "confidence": model_result.confidence,
        },
    )
    return await graph.ainvoke(initial_state(model_result, prefetched))
