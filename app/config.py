from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"

    spring_base_url: str = "http://localhost:8080"
    spring_internal_key: str = ""

    cartesia_api_key: str = ""
    cartesia_voice_id_ko: str = "ce9ca2b6-2bed-4452-99bb-052e1ec0b534"
    cartesia_voice_id_ja: str = "498e7f37-7fa3-4e2c-b8e2-8b6e9276f956"


settings = Settings()
