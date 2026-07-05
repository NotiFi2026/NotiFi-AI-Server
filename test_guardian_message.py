"""보호자 메시지 생성 테스트 — python test_guardian_message.py"""
import asyncio
import json

from app.agent.message_generator import generate_guardian_message
from app.agent.schemas import EventType, RiskLevel, VoiceResponseResult


async def main() -> None:
    cases = [
        ("USER_OK",         VoiceResponseResult.USER_OK),
        ("USER_NEEDS_HELP", VoiceResponseResult.USER_NEEDS_HELP),
        ("NO_RESPONSE",     VoiceResponseResult.NO_RESPONSE),
    ]

    context = {
        "no_movement_seconds_after_event": 20,
        "post_event_movement_level": 0.03,
    }

    for label, result in cases:
        print(f"\n{'='*60}")
        print(f"▶ [{label}]")
        msg = await generate_guardian_message(
            event_type=EventType.FALL,
            label="fall_detected",
            risk_level=RiskLevel.DANGER,
            confidence=0.91,
            response_result=result,
            context_features=context,
        )
        print(json.dumps(msg.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
