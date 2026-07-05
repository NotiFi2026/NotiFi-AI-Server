"""사용자 음성 응답 분류 — 키워드 우선, UNCLEAR는 LLM 보조."""
from app.agent.schemas import VoiceResponseResult
from app.clients.llm_client import complete_json
from app.common.logging_config import logger

_HELP_KEYWORDS = [
    # 한국어
    "도와", "도와줘", "살려", "아파", "못일어나", "못 일어나",
    "119", "응급", "힘들어",
    # 일본어
    "助けて", "助け", "痛い", "起き上がれ", "救急", "苦しい", "倒れ",
]

_OK_KEYWORDS = [
    # 한국어
    "괜찮아", "괜찮아요", "괜찮습니다", "문제없어", "문제 없어요", "괜찮음",
    # 일본어
    "大丈夫", "だいじょうぶ", "問題ない", "平気",
]

_SYSTEM_PROMPT = """\
노인 안전 모니터링 서비스에서 사용자 음성 응답을 분류합니다. 한국어와 일본어 응답 모두 처리합니다.

분류 기준:
- USER_OK: 괜찮다는 응답 (예: 괜찮아, 문제없어, 大丈夫, 問題ない)
- USER_NEEDS_HELP: 도움 요청 (예: 도와줘, 살려줘, 아파, 힘들어, 助けて, 痛い, 苦しい)
- NO_RESPONSE: 응답 없음 또는 빈 텍스트
- UNCLEAR: 위 기준에 해당하지 않는 경우

JSON 형식으로만 응답하세요:
{"response_result": "USER_OK" | "USER_NEEDS_HELP" | "NO_RESPONSE" | "UNCLEAR"}"""


def _keyword_classify(stt_text: str) -> VoiceResponseResult | None:
    if not stt_text or not stt_text.strip():
        return VoiceResponseResult.NO_RESPONSE

    normalized = stt_text.replace(" ", "").lower()

    if any(k.replace(" ", "") in normalized for k in _HELP_KEYWORDS):
        return VoiceResponseResult.USER_NEEDS_HELP

    if any(k.replace(" ", "") in normalized for k in _OK_KEYWORDS):
        return VoiceResponseResult.USER_OK

    return None  # UNCLEAR — LLM로 위임


async def classify(stt_text: str | None) -> VoiceResponseResult:
    """키워드 기반 분류 → UNCLEAR이면 LLM 보조 분류."""
    if stt_text is None:
        logger.info(
            "음성 응답 분류 완료",
            extra={"action": "voice_response_classified", "method": "keyword", "result": "NO_RESPONSE"},
        )
        return VoiceResponseResult.NO_RESPONSE

    # 1차: 키워드
    result = _keyword_classify(stt_text)
    if result is not None:
        logger.info(
            "음성 응답 분류 완료",
            extra={
                "action": "voice_response_classified",
                "method": "keyword",
                "result": result.value,
            },
        )
        return result

    # 2차: LLM 보조
    raw = await complete_json(_SYSTEM_PROMPT, f'STT 결과: "{stt_text}"')
    label = raw.get("response_result", "UNCLEAR")
    try:
        result = VoiceResponseResult(label)
    except ValueError:
        result = VoiceResponseResult.UNCLEAR

    logger.info(
        "음성 응답 분류 완료",
        extra={
            "action": "voice_response_classified",
            "method": "llm_fallback",
            "result": result.value,
        },
    )
    return result
