"""PlannerAgent -- 自然语言指令解析为结构化执行计划"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel

from ..data.validators import validate_stock_code
from ..llm.client import LLMClient, LLMError
from ..llm.prompts import PLANNER_SYSTEM, PLANNER_USER
from .base import BaseAgent, AgentResult

logger = logging.getLogger(__name__)

# A-share code regex for fallback extraction from free text
_STOCK_CODE_RE = re.compile(r"\b(60|00|30|8\d)\d{4}\b")


class ExecutionPlan(BaseModel):
    """Structured plan parsed from user natural-language input."""

    stock_code: str | None = None
    days: int = 120
    focus_areas: list[str] = ["all"]
    analysis_type: str = "full"  # quick | full | deep
    notes: str = ""


class PlannerAgent(BaseAgent):
    """指令解析 Agent -- 自然语言 -> 结构化执行计划"""

    def __init__(self, llm_client: LLMClient, **kwargs: Any) -> None:
        super().__init__(name="planner", **kwargs)
        self.llm = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_intent(self, user_input: str) -> ExecutionPlan:
        """Parse a natural-language request into an *ExecutionPlan*.

        Strategy:
        1. Try ``LLMClient.structured_output`` with the *ExecutionPlan* schema.
        2. On failure fall back to ``LLMClient.invoke`` + ``json.loads``.
        3. If the LLM could not identify a stock code, attempt a regex
           extraction from *user_input* as a last resort.
        """

        self._log_action("parse_intent_started", extra_fields={"input_len": len(user_input)})

        system = PLANNER_SYSTEM
        user = PLANNER_USER.safe_substitute(user_input=user_input)

        plan = self._try_structured(system, user)
        if plan is None:
            plan = self._try_json_fallback(system, user)
        if plan is None:
            plan = ExecutionPlan()
            self._logger.warning("LLM parsing failed completely; returning default plan")

        # Post-process: try to fill missing stock_code via regex
        if plan.stock_code is None:
            plan = self._fill_stock_code(plan, user_input)

        # Validate stock_code if present
        if plan.stock_code is not None:
            try:
                plan.stock_code = validate_stock_code(plan.stock_code)
            except ValueError as exc:
                self._logger.warning("LLM returned invalid stock_code: %s", exc)
                plan.notes = f"(invalid stock_code ignored: {plan.stock_code}) {plan.notes}"
                plan.stock_code = None

        self._log_action(
            "parse_intent_completed",
            stock_code=plan.stock_code or "",
            extra_fields={
                "days": plan.days,
                "analysis_type": plan.analysis_type,
                "focus_areas": plan.focus_areas,
            },
        )
        return plan

    def analyze(self, stock_code: str, **kwargs: Any) -> AgentResult:
        """Compatibility shim for the *BaseAgent* interface.

        Accepts a *stock_code* and optional ``user_input`` kwarg.  When
        *user_input* is provided the full intent-parsing pipeline runs;
        otherwise a minimal plan is returned for the given stock.
        """

        user_input = kwargs.get("user_input", stock_code)

        try:
            plan = self.parse_intent(user_input)
            # Ensure the returned stock_code matches the one the caller asked
            # for (if the LLM failed to extract it).
            if plan.stock_code is None:
                plan.stock_code = validate_stock_code(stock_code)

            return AgentResult(
                agent_name=self.name,
                stock_code=plan.stock_code,
                signal="HOLD",
                confidence=1.0,
                reasoning=f"Execution plan: {plan.analysis_type} analysis, "
                          f"{plan.days} days, focus={plan.focus_areas}",
                metrics=plan.model_dump(),
                success=True,
            )
        except Exception as exc:
            self._logger.error("PlannerAgent.analyze failed: %s", exc)
            return AgentResult(
                agent_name=self.name,
                stock_code=stock_code,
                signal="HOLD",
                confidence=0.0,
                reasoning=f"Intent parsing failed: {exc}",
                success=False,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_structured(self, system: str, user: str) -> ExecutionPlan | None:
        """Attempt parsing via ``structured_output``."""
        try:
            return self.llm.structured_output(system, user, ExecutionPlan)
        except LLMError as exc:
            self._logger.debug("structured_output failed: %s", exc)
            return None

    def _try_json_fallback(self, system: str, user: str) -> ExecutionPlan | None:
        """Fallback: raw invoke + json.loads."""
        try:
            raw = self.llm.invoke(system, user)
            # Strip markdown code fences if present
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
            data = json.loads(cleaned)
            return ExecutionPlan.model_validate(data)
        except (LLMError, json.JSONDecodeError, Exception) as exc:
            self._logger.debug("JSON fallback failed: %s", exc)
            return None

    @staticmethod
    def _fill_stock_code(plan: ExecutionPlan, user_input: str) -> ExecutionPlan:
        """Try to extract a stock code from *user_input* using regex."""
        match = _STOCK_CODE_RE.search(user_input)
        if match:
            plan.stock_code = match.group(0)
        return plan
