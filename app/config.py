from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"

    spring_base_url: str = "http://localhost:8080"
    spring_internal_key: str = ""

    # GUARDIAN_NOTIFY 후 EMERGENCY_CALL 진행 전 보호자 확인 대기 시간
    emergency_call_delay_seconds: int = 60

    cartesia_api_key: str = ""
    cartesia_voice_id_ko: str = "ce9ca2b6-2bed-4452-99bb-052e1ec0b534"
    cartesia_voice_id_ja: str = "498e7f37-7fa3-4e2c-b8e2-8b6e9276f956"

    # notifi-ai 모델 런타임. False면 로드를 건너뛴다(에이전트만 개발할 때 CUDA 초기화 회피)
    notifi_model_enabled: bool = True
    # None이면 cuda 우선 자동 선택. artifact_dir도 None이면 패키지 기본 경로
    notifi_model_device: str | None = None
    notifi_artifact_dir: str | None = None
    # 모델 클래스명이 버전을 달고 있어(NotiFiAIv1) 새 모델은 이름이 바뀐다.
    # 설정으로 두면 v2 교체가 환경변수 한 줄이다.
    notifi_model_class: str = "NotiFiAIv1"
    # 패키지 기본값 "runtime/devices"는 cwd 의존이라 절대 경로로 고정한다
    notifi_registry_root: str = str(_REPO_ROOT / "runtime" / "devices")
    # 같은 행동이 이어질 때 NORMAL 이벤트를 다시 보내기까지의 간격
    notifi_normal_interval_seconds: int = 300
    # 추론은 워밍업 후 ~0.21초. 몇 초를 기다린다는 건 이미 이상 신호이므로 포기하고 503을 준다
    notifi_inference_lock_timeout_seconds: float = 3.0
    # 한 추론이 이보다 오래 진행 중이면 멈춘 것으로 보고 health가 503을 반환한다
    notifi_inference_stuck_seconds: float = 60.0
    # 캘리브레이션 NPZ는 트라이얼당 ~0.79MiB — absence 12 + support 16이면 ~22MiB
    notifi_calibration_max_upload_mb: int = 64
    # 캘리브레이션은 support 트라이얼마다 forward를 돌려 수 초가 걸린다.
    # 추론 대기(3초)와 같은 값을 쓰면 캘리브레이션이 자기 차례를 못 잡는다.
    notifi_calibration_lock_timeout_seconds: float = 120.0

    # ── CSI 실시간 수집 데몬 ────────────────────────────────────────────────
    # 기본은 꺼 둔다. 보드가 없는 개발 환경에서 켜지면 시리얼 열기 실패 로그만 쌓인다
    notifi_stream_enabled: bool = False
    # serial(RX 보드 USB) 또는 replay(수집 CSV 재생 — 하드웨어 없이 검증)
    notifi_stream_source: str = "serial"
    notifi_stream_port: str = "COM7"
    notifi_stream_baud: int = 921600
    # 윈도를 뜨는 주기. 짧을수록 감지가 빠르지만 추론 부하가 는다(윈도 자체는 10.13초)
    notifi_stream_stride_seconds: float = 2.0
    # 위험 판정 재신고 억제. Spring은 DANGER마다 에스컬레이션을 새로 만들므로,
    # 이게 없으면 한 번 넘어졌는데 겹치는 윈도 수만큼 119 신고가 걸린다
    notifi_stream_cooldown_seconds: float = 120.0
    notifi_stream_replay_path: str = ""
    notifi_stream_replay_speed: float = 1.0
    notifi_stream_replay_loop: bool = False

    # ── 일일 리포트 스케줄러 ────────────────────────────────────────────────
    # 수집 데몬과 같은 이유로 기본은 꺼 둔다 — 개발 환경에서 켜지면 매일 LLM을 호출한다
    notifi_report_scheduler_enabled: bool = False
    # 아침에 어제 리포트가 도착하게. 대상일은 report_service.default_report_date()(KST 어제)
    notifi_report_hour_kst: int = 8
    # 분까지 두는 건 취향이 아니라 검증 때문이다 — 시 단위만 있으면 발화를 보려고
    # 한 시간을 기다려야 한다
    notifi_report_minute_kst: int = 0


settings = Settings()
