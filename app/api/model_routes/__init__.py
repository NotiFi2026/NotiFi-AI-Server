"""모델 라우터 조립.

로직·상수를 여기 두지 않는다 — 패키지 초기화만으로 무거운 것이 끌려오면
모델 미설치 환경의 부팅이 깨진다(`app/model/__init__.py`에서 실제로 겪었다).
"""
from fastapi import APIRouter

from app.api.model_routes import devices, inference

router = APIRouter(prefix="/internal/model")
router.include_router(devices.router)
router.include_router(inference.router)

__all__ = ["router"]
