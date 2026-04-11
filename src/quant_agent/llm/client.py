"""LLMClient — LangChain ChatOpenAI 封装，支持 OpenAI / 智谱"""

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


class LLMError(Exception):
    """LLM 调用失败"""


class LLMClient:
    """LangChain ChatModel 封装

    自动选择 provider:
        openai_api_key  → OpenAI (api.openai.com)
        zhipu_api_key   → 智谱 GLM (open.bigmodel.cn)
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

        # 选择 provider
        api_key = self.settings.openai_api_key
        model = self.settings.openai_model
        base_url = self.settings.openai_base_url

        if not api_key and self.settings.zhipu_api_key:
            api_key = self.settings.zhipu_api_key
            model = self.settings.zhipu_model
            base_url = _ZHIPU_BASE_URL

        if not api_key:
            raise LLMError(
                "No API key configured. "
                "Set QUANT_OPENAI_API_KEY or QUANT_ZHIPU_API_KEY in .env"
            )

        self.model = model
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=self.settings.llm_timeout,
            max_retries=self.settings.llm_max_retries,
            temperature=0.3,
        )
        logger.info("LLMClient initialized: model=%s", model)

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
    """获取全局 LLM 客户端单例"""
    return LLMClient()
