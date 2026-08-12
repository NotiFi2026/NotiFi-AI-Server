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
    # 패키지 기본값 "runtime/devices"는 cwd 의존이라 절대 경로로 고정한다
    notifi_registry_root: str = str(_REPO_ROOT / "runtime" / "devices")
    # 같은 행동이 이어질 때 NORMAL 이벤트를 다시 보내기까지의 간격
    notifi_normal_interval_seconds: int = 300
    # 추론은 워밍업 후 ~0.21초. 몇 초를 기다린다는 건 이미 이상 신호이므로 포기하고 503을 준다
    notifi_inference_lock_timeout_seconds: float = 3.0
    # 한 추론이 이보다 오래 진행 중이면 멈춘 것으로 보고 health가 503을 반환한다
    notifi_inference_stuck_seconds: float = 60.0


settings = Settings()
