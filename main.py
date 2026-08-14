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
            described = app.state.model_runtime.describe()
            # describe() 전체를 실으면 성능지표·절대경로까지 담긴 거대한 한 줄이 된다
            logger.info(
                "모델 로드 완료",
                extra={
                    "action": "model_loaded",
                    "model_name": described["model_name"],
                    "device": described["device"],
                },
            )
        except Exception as exc:
            logger.error(
                "모델 로드 실패 — 추론 없이 기동한다",
                extra={"action": "model_load_failed", "error": str(exc)},
                exc_info=True,
            )
    else:
        logger.info("모델 로드 건너뜀", extra={"action": "model_disabled"})

    # 수집 데몬은 모델이 있어야 의미가 있다 — 없으면 윈도를 떠도 판정할 수가 없다
    app.state.stream_pump = None
    if settings.notifi_stream_enabled and app.state.model_runtime is not None:
        try:
            from app.stream.pump import StreamPump

            pump = StreamPump(app.state.model_runtime, settings)
            pump.start(app.state.model_runtime.list_device_configs())
            app.state.stream_pump = pump
        except Exception as exc:
            # 수집이 안 떠도 서버는 살아 있어야 한다 — HTTP ingest 경로는 그대로 쓸 수 있다
            logger.error(
                "수집 데몬 기동 실패 — 수집 없이 기동한다",
                extra={"action": "stream_start_failed", "error": str(exc)},
                exc_info=True,
            )

    # 리포트는 I6 조회 → LLM → I3 적재라 모델이 필요 없다 — 모델 로드가 실패해도 스케줄러는 뜬다
    app.state.report_scheduler = None
    if settings.notifi_report_scheduler_enabled:
        try:
            from app.agent.report_scheduler import ReportScheduler

            scheduler = ReportScheduler(settings)
            scheduler.start()
            app.state.report_scheduler = scheduler
        except Exception as exc:
            # 스케줄러가 안 떠도 서버는 살아 있어야 한다 — 수동 리포트 라우트는 그대로 쓸 수 있다
            logger.error(
                "리포트 스케줄러 기동 실패 — 스케줄러 없이 기동한다",
                extra={"action": "report_scheduler_start_failed", "error": str(exc)},
                exc_info=True,
            )

    try:
        yield
    finally:
        if app.state.stream_pump is not None:
            await app.state.stream_pump.stop()
        if app.state.report_scheduler is not None:
            await app.state.report_scheduler.stop()


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
