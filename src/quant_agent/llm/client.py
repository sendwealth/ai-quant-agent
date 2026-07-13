"""LLMClient — LangChain ChatOpenAI 封装，支持 OpenAI / 智谱 / 本地模型

增强（v3.1）:
- 多 provider: OpenAI / 智谱 / Ollama 等 OpenAI 兼容本地服务
- 软降级: soft_fail=True 时无 key 不崩溃，进入「规则增强版」离线模式，
  三大 LLM 能力（情感/报告/风险解读）始终可用
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional, TypeVar, Type

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 智谱 API base URL (OpenAI 兼容接口)
_ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
# 本地模型默认地址 (Ollama / LM Studio 等 OpenAI 兼容服务)
_LOCAL_DEFAULT_BASE_URL = "http://localhost:11434/v1"


class LLMError(Exception):
    """LLM 调用失败"""


class LLMClient:
    """LangChain ChatModel 封装

    自动选择 provider（优先级：OpenAI → 智谱 → 本地模型）:
        openai_api_key   → OpenAI (api.openai.com)
        zhipu_api_key    → 智谱 GLM (open.bigmodel.cn)
        llm_base_url 配置 → 本地/自建 OpenAI 兼容服务（如 Ollama）

    soft_fail=True 时，若没有任何可用 key，不会抛 LLMError，而是进入
    离线降级模式（enabled=False），invoke/structured_output 返回规则模板，
    保证上层三大 LLM 增强功能（情感/报告/风险解读）不中断。
    """

    def __init__(self, settings: Optional[Settings] = None, soft_fail: bool = False):
        self.settings = settings or get_settings()
        self.soft_fail = soft_fail
        self.enabled = False
        self.provider: Optional[str] = None
        self.model = self.settings.openai_model
        self.llm: Optional[ChatOpenAI] = None

        api_key, model, base_url, provider = self._resolve_provider()

        if not api_key:
            if soft_fail:
                logger.info(
                    "LLM 未配置 (无 API key)，进入离线规则增强模式。"
                    "配置 QUANT_OPENAI_API_KEY / QUANT_ZHIPU_API_KEY 或本地模型后可启用。"
                )
                return
            raise LLMError(
                "No API key configured. "
                "Set QUANT_OPENAI_API_KEY or QUANT_ZHIPU_API_KEY in .env"
            )

        self.model = model
        self.provider = provider
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=self.settings.llm_timeout,
            max_retries=self.settings.llm_max_retries,
            temperature=0.3,
        )
        self.enabled = True
        logger.info("LLMClient initialized: provider=%s model=%s", provider, model)

    def _resolve_provider(self) -> tuple[Optional[str], str, str, Optional[str]]:
        """解析 provider 与连接参数。

        Returns:
            (api_key, model, base_url, provider_name)
        """
        s = self.settings

        # 1. OpenAI
        if s.openai_api_key:
            return (
                s.openai_api_key,
                s.openai_model,
                s.openai_base_url,
                "openai",
            )

        # 2. 智谱
        if s.zhipu_api_key:
            return (
                s.zhipu_api_key,
                s.zhipu_model,
                _ZHIPU_BASE_URL,
                "zhipu",
            )

        # 3. 本地 / 自建 OpenAI 兼容服务 (Ollama / LM Studio / vLLM 等)
        #    通过 llm_base_url 指定；需为有效字符串（排除 MagicMock 等占位对象）
        local_base = getattr(s, "llm_base_url", None)
        if isinstance(local_base, str) and local_base.strip():
            local_model = getattr(s, "llm_local_model", None)
            if not isinstance(local_model, str) or not local_model.strip():
                local_model = "local-model"
            return (local_model, local_model, local_base, "local")

        # 4. llm_base_url 未配：若用户把 openai_base_url 指向本地服务也兼容
        openai_base = s.openai_base_url
        if isinstance(openai_base, str) and (
            "localhost" in openai_base or "127.0.0.1" in openai_base
        ):
            local_model = getattr(s, "llm_local_model", None)
            if not isinstance(local_model, str) or not local_model.strip():
                local_model = "local-model"
            return (local_model, local_model, openai_base, "local")

        return (None, self.model, s.openai_base_url, None)

    @property
    def available(self) -> bool:
        """是否处于真实 LLM 模式（非离线降级）"""
        return self.enabled

    # ── 离线降级模板 ──

    def _offline_invoke(self, system: str, user: str) -> str:
        """无 LLM 时的规则增强模板回复。"""
        return (
            "[离线模式] 未配置 LLM API key，已跳过大模型生成。\n"
            "提示：在 .env 中配置 QUANT_OPENAI_API_KEY / QUANT_ZHIPU_API_KEY，"
            "或设置 QUANT_LLM_BASE_URL 指向本地模型（Ollama 等）即可启用增强分析。"
        )

    def invoke(self, system: str, user: str) -> str:
        """基础聊天接口

        Args:
            system: system prompt
            user: user message

        Returns:
            LLM 回复文本

        Raises:
            LLMError: 调用失败
        """
        if not self.enabled:
            return self._offline_invoke(system, user)
        try:
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=user),
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            raise LLMError(f"LLM invoke failed: {e}") from e

    def structured_output(self, system: str, user: str, schema: Type[T]) -> T:
        """结构化输出 — 返回 Pydantic 模型实例

        Args:
            system: system prompt
            user: user message
            schema: Pydantic 模型类 (必须是 BaseModel 子类)

        Returns:
            schema 的实例

        Raises:
            LLMError: 调用或解析失败
        """
        if not self.enabled:
            # 离线降级：返回一个中性结构（信号 HOLD，低信心），不阻断流水线
            try:
                return schema(
                    signal="HOLD",
                    confidence=0.0,
                    sentiment_score=0.0,
                    reasoning="[离线模式] 未配置 LLM，情感分析降级为中性。",
                    key_factors=[],
                )
            except Exception:
                # 非情感 schema 时退化为通用占位
                return schema()
        try:
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=user),
            ]
            structured_llm = self.llm.with_structured_output(schema)
            result = structured_llm.invoke(messages)
            return result
        except Exception as e:
            raise LLMError(f"LLM structured_output failed: {e}") from e


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """获取全局 LLM 客户端单例（严格模式：无 key 时抛 LLMError）"""
    return LLMClient()


@lru_cache(maxsize=1)
def get_llm_client_soft() -> LLMClient:
    """获取全局 LLM 客户端单例（软降级模式：无 key 时不崩溃）。

    供应用运行时使用，保证即便没有 API key，三大 LLM 增强能力也不会中断。
    """
    return LLMClient(soft_fail=True)
