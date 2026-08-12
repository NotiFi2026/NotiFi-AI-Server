"""모델 런타임 예외.

runtime.py에 두면 라우터가 예외를 잡으려고 import하는 순간 notifi_ai·torch가
기동 시 끌려와 모델 미설치 환경의 부팅이 깨진다. 그래서 의존성 없는 모듈로 분리한다.
"""


class InferenceBusyError(RuntimeError):
    """대기 한도 안에 추론 락을 잡지 못했다 — 앞 추론이 오래 걸리거나 멈췄다."""
