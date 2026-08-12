"""모델 런타임·변환 계층.

여기서 runtime을 re-export하지 않는다 — `app.model.pipeline`만 import해도
패키지 초기화를 타고 notifi_ai·torch가 끌려와 모델 미설치 환경의 부팅이 깨진다.
런타임이 필요한 곳에서 `from app.model.runtime import ModelRuntime`으로 직접 가져온다.
"""
