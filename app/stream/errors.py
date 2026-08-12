"""수집 계층 예외 — **notifi_ai를 끌어오지 않는 별도 모듈.**

collector.py에 두면 라우터가 예외를 잡으려고 import하는 순간 notifi_ai·torch가 딸려와
모델 미설치 환경의 부팅이 깨진다(app/model/errors.py와 같은 이유이자, 실제로 밟은 회귀다).
"""
from __future__ import annotations


class NotEnoughSignal(RuntimeError):
    """버퍼에 윈도를 채울 만큼 데이터가 없다 — 보드가 꺼졌거나 방금 시작했다."""
