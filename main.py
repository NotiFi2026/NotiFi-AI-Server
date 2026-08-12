from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.model_routes import router as model_router
from app.api.routes import router
from app.api.status import router as status_router
from app.common.logging_config import logger
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """기동 시 모델을 로드하고 warmup까지 끝낸다.

    로드에 실패해도 서버는 뜬다 — 에스컬레이션 에이전트는 모델 없이도 동작해야 한다.
    """
    app.state.model_runtime = None
    if settings.notifi_model_enabled:
        try:
            from app.model.runtime import ModelRuntime

            app.state.model_runtime = ModelRuntime.load(settings)
            logger.info(
                "모델 로드 완료",
                extra={"action": "model_loaded", **app.state.model_runtime.describe()},
            )
        except Exception as exc:
            logger.error(
                "모델 로드 실패 — 추론 없이 기동한다",
                extra={"action": "model_load_failed", "error": str(exc)},
                exc_info=True,
            )
    else:
        logger.info("모델 로드 건너뜀", extra={"action": "model_disabled"})
    yield


app = FastAPI(title="Notifi AI Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(status_router)
app.include_router(model_router)
