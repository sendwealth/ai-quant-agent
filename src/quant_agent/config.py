"""统一配置管理 — Pydantic Settings"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """统一配置，支持环境变量覆盖（前缀 QUANT_）"""

    model_config = SettingsConfigDict(env_prefix="QUANT_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── 应用 ──
    app_name: str = "ai-quant-agent"
    debug: bool = False

    # ── 数据源 ──
    tushare_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("QUANT_TUSHARE_TOKEN", "TUSHARE_TOKEN"),
    )
    akshare_timeout: int = 10
    data_cache_ttl: int = 1800  # 秒

    # ── 存储 ──
    data_dir: str = "data/cache"
    parquet_dir: str = "data/parquet"
    # 内置样例数据目录（开箱即用的离线兜底，相对包内路径）
    sample_data_dir: str = "data/samples"

    # ── LLM ──
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"
    openai_base_url: str = "https://api.openai.com/v1"
    zhipu_api_key: Optional[str] = None
    zhipu_model: str = "glm-4"
    # 本地 / 自建 OpenAI 兼容服务（Ollama / LM Studio / vLLM 等）
    llm_base_url: Optional[str] = None
    llm_local_model: Optional[str] = None
    llm_timeout: int = 30
    llm_max_retries: int = 2

    # ── 风控 ──
    initial_capital: float = 100000.0
    max_position_pct: float = 0.20
    max_portfolio_risk: float = 0.80
    default_stop_loss: float = -0.08
    default_take_profit_1: float = 0.10
    default_take_profit_2: float = 0.20
    max_daily_loss_pct: float = -0.03  # 单日最大亏损熔断阈值

    # ── 回测 ──
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001  # A股印花税（仅卖出）
    slippage_bps: float = 1.0

    # ── 并发 ──
    fetch_max_workers: int = 5

    # ── 交易持久化 ──
    # 为 True 时，实盘组合状态由 PaperTradingService 托管并落盘，
    # 进程重启不丢状态。默认 False（纯内存，便于测试/预览）。
    persist_trading: bool = False

    # ── 离线 & 预下载 ──
    offline_mode: bool = False
    preload_stocks: str = "300750,002475,601318,600276,000858,600519"

    # ── 日志 ──
    log_level: str = "INFO"

    # ── 邮件通知 ──
    email_enabled: bool = False
    email_smtp_server: str = "smtp.163.com"
    email_smtp_port: int = 465
    email_sender: str = ""
    email_password: str = ""
    email_recipients: str = ""  # 逗号分隔

    # ── Validators ──

    @field_validator("max_position_pct")
    @classmethod
    def _validate_max_position(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError(f"max_position_pct must be in (0, 1], got {v}")
        return v

    @field_validator("max_portfolio_risk")
    @classmethod
    def _validate_max_portfolio_risk(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError(f"max_portfolio_risk must be in (0, 1], got {v}")
        return v

    @field_validator("default_stop_loss")
    @classmethod
    def _validate_stop_loss(cls, v: float) -> float:
        if v >= 0:
            raise ValueError(f"stop_loss must be negative, got {v}")
        return v

    @field_validator("commission_rate", "stamp_tax_rate")
    @classmethod
    def _validate_non_negative_rate(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"rate must be >= 0, got {v}")
        return v

    @field_validator("max_daily_loss_pct")
    @classmethod
    def _validate_daily_loss(cls, v: float) -> float:
        if v >= 0:
            raise ValueError(f"max_daily_loss_pct must be negative, got {v}")
        return v

    @field_validator("default_take_profit_1", "default_take_profit_2")
    @classmethod
    def _validate_take_profit(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"take_profit must be positive, got {v}")
        return v

    @field_validator("initial_capital")
    @classmethod
    def _validate_initial_capital(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"initial_capital must be positive, got {v}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例"""
    return Settings()
