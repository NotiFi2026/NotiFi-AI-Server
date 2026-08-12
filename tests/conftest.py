"""모델·GPU 없이 도는 테스트 환경.

설정은 import 시점에 고정되므로 app import 전에 환경변수를 세팅한다.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("SPRING_INTERNAL_KEY", "test-key")
os.environ.setdefault("NOTIFI_MODEL_ENABLED", "false")
