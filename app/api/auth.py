"""내부 API 인증 — Spring과 공유하는 X-Internal-Key 검증."""
from fastapi import HTTPException

from app.config import settings


def check_internal_key(x_internal_key: str) -> None:
    """키가 다르면 401. 키가 설정되지 않았으면 모든 요청을 거부한다.

    빈 키를 유효한 키로 취급하면 헤더 없는 요청이 그대로 통과해
    내부 API 전체가 무인증으로 열린다.
    """
    if not settings.spring_internal_key:
        raise HTTPException(status_code=401, detail="Internal key is not configured")
    if x_internal_key != settings.spring_internal_key:
        raise HTTPException(status_code=401, detail="Invalid internal key")
