"""모델·GPU 없이 도는 테스트 환경.

설정은 import 시점에 고정되므로 app import 전에 환경변수를 세팅한다.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("SPRING_INTERNAL_KEY", "test-key")
os.environ.setdefault("NOTIFI_MODEL_ENABLED", "false")

from app.model.spec import ModelSpec  # noqa: E402 - 위 환경변수 설정이 먼저여야 한다

#: 실모델 v1과 같은 형태의 계약. ModelSpec은 notifi_ai에 의존하지 않으므로
#: 모델이 설치되지 않은 환경에서도 만들 수 있다.
FAKE_SPEC = ModelSpec(
    model_name="NotiFi_AI_v1",
    action_labels=tuple(f"action{index}" for index in range(17)),
    action_to_risk=tuple(0 if index < 9 else 1 if index < 12 else 2 for index in range(17)),
    risk_labels=("safe", "warning", "danger"),
    joint_names=tuple(f"j{index}" for index in range(22)),
    joint_schema="smpl-22",
    fps=30.0,
    frames=304,
    links=3,
    subcarriers=114,
    max_gap_seconds=0.1,
)
