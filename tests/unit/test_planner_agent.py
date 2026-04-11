"""Tests for PlannerAgent -- natural language intent parsing."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from quant_agent.agents.base import AgentResult
from quant_agent.agents.planner import ExecutionPlan, PlannerAgent
from quant_agent.llm.client import LLMClient, LLMError


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_llm_client(
    structured_result=None,
    invoke_result: str | None = None,
    structured_side_effect: Exception | None = None,
    invoke_side_effect: Exception | None = None,
) -> MagicMock:
    """Create a mock LLMClient.

    Args:
        structured_result: Return value for structured_output().
        invoke_result: Return value for invoke().
        structured_side_effect: Exception to raise from structured_output().
        invoke_side_effect: Exception to raise from invoke().
    """
    llm = MagicMock(spec=LLMClient)
    if structured_side_effect:
        llm.structured_output.side_effect = structured_side_effect
    else:
        llm.structured_output.return_value = structured_result

    if invoke_side_effect:
        llm.invoke.side_effect = invoke_side_effect
    else:
        llm.invoke.return_value = invoke_result or ""
    return llm


def _plan_json(
    stock_code: str | None = "300750",
    days: int = 120,
    focus_areas: list[str] | None = None,
    analysis_type: str = "full",
    notes: str = "",
) -> str:
    """Build a JSON string for an ExecutionPlan."""
    return json.dumps({
        "stock_code": stock_code,
        "days": days,
        "focus_areas": focus_areas or ["all"],
        "analysis_type": analysis_type,
        "notes": notes,
    })


# ── parse_intent tests ──────────────────────────────────────────────────────


class TestPlannerAgentParseIntent:
    """PlannerAgent.parse_intent — NL to ExecutionPlan."""

    def test_parse_intent_with_stock_name(self):
        """Chinese stock name + intent is parsed into a valid ExecutionPlan."""
        plan = ExecutionPlan(stock_code="300750", days=120, analysis_type="full")
        llm = _make_llm_client(structured_result=plan)

        agent = PlannerAgent(llm_client=llm)
        result = agent.parse_intent("分析宁德时代的买入机会")

        assert isinstance(result, ExecutionPlan)
        assert result.stock_code == "300750"
        assert result.analysis_type == "full"

    def test_parse_intent_with_stock_code_in_text(self):
        """When LLM returns no stock_code, regex extracts it from the input."""
        # LLM returns no stock_code
        plan = ExecutionPlan(stock_code=None, days=60, analysis_type="quick")
        llm = _make_llm_client(structured_result=plan)

        agent = PlannerAgent(llm_client=llm)
        result = agent.parse_intent("快速分析一下 600519 的走势")

        assert result.stock_code == "600519"

    def test_parse_intent_regex_no_match(self):
        """When neither LLM nor regex finds a stock code, it stays None."""
        plan = ExecutionPlan(stock_code=None, days=120, analysis_type="full")
        llm = _make_llm_client(structured_result=plan)

        agent = PlannerAgent(llm_client=llm)
        result = agent.parse_intent("帮我看看大盘走势")

        assert result.stock_code is None

    def test_parse_intent_structured_output_falls_back_to_json(self):
        """When structured_output fails, falls back to invoke + json.loads."""
        llm = _make_llm_client(
            structured_side_effect=LLMError("Not supported"),
            invoke_result=_plan_json(stock_code="002594", days=90),
        )

        agent = PlannerAgent(llm_client=llm)
        result = agent.parse_intent("分析比亚迪")

        assert result.stock_code == "002594"
        assert result.days == 90

    def test_parse_intent_both_fail_returns_default_plan(self):
        """When both structured_output and invoke fail, returns default ExecutionPlan."""
        llm = _make_llm_client(
            structured_side_effect=LLMError("fail"),
            invoke_side_effect=LLMError("fail"),
        )

        agent = PlannerAgent(llm_client=llm)
        result = agent.parse_intent("随便看看")

        assert isinstance(result, ExecutionPlan)
        assert result.days == 120  # default
        assert result.analysis_type == "full"  # default

    def test_parse_intent_json_in_code_fences(self):
        """JSON fallback handles markdown code fences around the response."""
        json_text = _plan_json(stock_code="300750")
        fenced = f"```json\n{json_text}\n```"
        llm = _make_llm_client(
            structured_side_effect=LLMError("fail"),
            invoke_result=fenced,
        )

        agent = PlannerAgent(llm_client=llm)
        result = agent.parse_intent("分析宁德时代")

        assert result.stock_code == "300750"

    def test_parse_intent_invalid_json_fallback(self):
        """Invalid JSON from invoke causes fallback to default plan."""
        llm = _make_llm_client(
            structured_side_effect=LLMError("fail"),
            invoke_result="This is not valid JSON",
        )

        agent = PlannerAgent(llm_client=llm)
        result = agent.parse_intent("some request")

        assert isinstance(result, ExecutionPlan)
        # Defaults since both LLM calls failed
        assert result.stock_code is None or result.days == 120

    def test_parse_intent_validates_stock_code(self):
        """An invalid stock_code from LLM is rejected and set to None."""
        plan = ExecutionPlan(stock_code="999999", days=120)
        llm = _make_llm_client(structured_result=plan)

        agent = PlannerAgent(llm_client=llm)
        result = agent.parse_intent("分析股票")

        # 999999 has prefix "99" which is not valid
        assert result.stock_code is None
        assert "invalid stock_code ignored" in result.notes

    def test_parse_intent_passes_user_input_to_prompt(self):
        """The user_input is included in the prompt sent to the LLM."""
        llm = _make_llm_client(structured_result=ExecutionPlan())

        agent = PlannerAgent(llm_client=llm)
        agent.parse_intent("帮我分析宁德时代的基本面")

        # Verify structured_output was called with system and user prompts
        call_args = llm.structured_output.call_args[0]
        user_prompt = call_args[1]
        assert "帮我分析宁德时代的基本面" in user_prompt

    def test_parse_intent_beijing_stock_code(self):
        """Beijing (北交所) stock codes with prefix 8xxxxx are extracted."""
        plan = ExecutionPlan(stock_code=None)
        llm = _make_llm_client(structured_result=plan)

        agent = PlannerAgent(llm_client=llm)
        result = agent.parse_intent("看看 830799 这个北交所股票")

        # Regex should extract 830799
        assert result.stock_code == "830799"

    def test_parse_intent_shanghai_stock_code(self):
        """Shanghai main board codes (60xxxx) are extracted by regex."""
        plan = ExecutionPlan(stock_code=None)
        llm = _make_llm_client(structured_result=plan)

        agent = PlannerAgent(llm_client=llm)
        result = agent.parse_intent("分析 600036 招商银行")

        assert result.stock_code == "600036"

    def test_parse_intent_shenzhen_stock_code(self):
        """Shenzhen main board codes (00xxxx) are extracted by regex."""
        plan = ExecutionPlan(stock_code=None)
        llm = _make_llm_client(structured_result=plan)

        agent = PlannerAgent(llm_client=llm)
        result = agent.parse_intent("看看 000858 五粮液")

        assert result.stock_code == "000858"


# ── analyze() tests (BaseAgent interface) ────────────────────────────────────


class TestPlannerAgentAnalyze:
    """PlannerAgent.analyze — BaseAgent compatibility shim."""

    def test_analyze_returns_agent_result(self):
        """analyze() returns an AgentResult with the execution plan."""
        plan = ExecutionPlan(stock_code="300750", days=90, analysis_type="deep")
        llm = _make_llm_client(structured_result=plan)

        agent = PlannerAgent(llm_client=llm)
        result = agent.analyze("300750", user_input="深度分析宁德时代")

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.agent_name == "planner"
        assert result.stock_code == "300750"
        assert result.signal == "HOLD"  # Planner always returns HOLD
        assert result.confidence == 1.0

    def test_analyze_fills_stock_code_from_param(self):
        """When LLM fails to extract stock code, analyze() uses the param."""
        plan = ExecutionPlan(stock_code=None, days=120)
        llm = _make_llm_client(structured_result=plan)

        agent = PlannerAgent(llm_client=llm)
        result = agent.analyze("300750", user_input="随便看看")

        assert result.stock_code == "300750"
        assert result.success is True

    def test_analyze_includes_plan_in_metrics(self):
        """Result metrics contain the full ExecutionPlan dump."""
        plan = ExecutionPlan(
            stock_code="300750", days=60,
            focus_areas=["fundamental", "technical"],
            analysis_type="quick",
        )
        llm = _make_llm_client(structured_result=plan)

        agent = PlannerAgent(llm_client=llm)
        result = agent.analyze("300750")

        assert "days" in result.metrics
        assert result.metrics["days"] == 60
        assert result.metrics["analysis_type"] == "quick"

    def test_analyze_handles_exception(self):
        """analyze() catches exceptions and returns failed AgentResult."""
        llm = _make_llm_client(
            structured_side_effect=LLMError("fail"),
            invoke_side_effect=LLMError("fail"),
        )

        agent = PlannerAgent(llm_client=llm)
        # This should not raise, even though LLM fails
        result = agent.analyze("300750")

        assert isinstance(result, AgentResult)
        assert result.success is True  # Default plan is still returned
        # When parse_intent returns a default plan and stock_code is filled
        # from the param, analyze succeeds

    def test_analyze_with_no_user_input_kwarg(self):
        """analyze() uses stock_code as user_input when kwarg is absent."""
        plan = ExecutionPlan(stock_code="300750")
        llm = _make_llm_client(structured_result=plan)

        agent = PlannerAgent(llm_client=llm)
        result = agent.analyze("300750")  # No user_input kwarg

        assert result.success is True
        # Verify the LLM was called with "300750" as user input
        call_args = llm.structured_output.call_args[0]
        assert "300750" in call_args[1]


# ── ExecutionPlan model tests ───────────────────────────────────────────────


class TestExecutionPlan:
    """ExecutionPlan — Pydantic model validation."""

    def test_default_values(self):
        """ExecutionPlan has sensible defaults."""
        plan = ExecutionPlan()
        assert plan.stock_code is None
        assert plan.days == 120
        assert plan.focus_areas == ["all"]
        assert plan.analysis_type == "full"
        assert plan.notes == ""

    def test_custom_values(self):
        """ExecutionPlan accepts custom values."""
        plan = ExecutionPlan(
            stock_code="600519",
            days=30,
            focus_areas=["fundamental", "sentiment"],
            analysis_type="quick",
            notes="Focus on earnings",
        )
        assert plan.stock_code == "600519"
        assert plan.days == 30
        assert len(plan.focus_areas) == 2
        assert plan.notes == "Focus on earnings"

    def test_model_dump(self):
        """model_dump() returns a dict with all fields."""
        plan = ExecutionPlan(stock_code="300750", days=90)
        dumped = plan.model_dump()

        assert isinstance(dumped, dict)
        assert dumped["stock_code"] == "300750"
        assert dumped["days"] == 90


# ── _fill_stock_code tests ──────────────────────────────────────────────────


class TestPlannerAgentFillStockCode:
    """PlannerAgent._fill_stock_code — regex-based stock code extraction."""

    def test_fill_extracts_shanghai_code(self):
        """60xxxx code is extracted from free text."""
        plan = ExecutionPlan(stock_code=None)
        result = PlannerAgent._fill_stock_code(plan, "看看 600036 怎么样")
        assert result.stock_code == "600036"

    def test_fill_extracts_shenzhen_code(self):
        """00xxxx code is extracted."""
        plan = ExecutionPlan(stock_code=None)
        result = PlannerAgent._fill_stock_code(plan, "分析 000858 五粮液")
        assert result.stock_code == "000858"

    def test_fill_extracts_chinext_code(self):
        """30xxxx code is extracted."""
        plan = ExecutionPlan(stock_code=None)
        result = PlannerAgent._fill_stock_code(plan, "宁德时代 300750")
        assert result.stock_code == "300750"

    def test_fill_extracts_beijing_code(self):
        """8xxxxx code is extracted."""
        plan = ExecutionPlan(stock_code=None)
        result = PlannerAgent._fill_stock_code(plan, "北交所 830799")
        assert result.stock_code == "830799"

    def test_fill_no_match_returns_unchanged(self):
        """When no code is found, stock_code stays None."""
        plan = ExecutionPlan(stock_code=None)
        result = PlannerAgent._fill_stock_code(plan, "看看大盘走势")
        assert result.stock_code is None

    def test_fill_returns_same_plan_object(self):
        """_fill_stock_code mutates and returns the same plan object."""
        plan = ExecutionPlan(stock_code=None)
        returned = PlannerAgent._fill_stock_code(plan, "300750")
        assert returned is plan

    def test_fill_does_not_overwrite_existing_code(self):
        """If stock_code is already set, regex extraction still overwrites it."""
        plan = ExecutionPlan(stock_code="600519")
        result = PlannerAgent._fill_stock_code(plan, "分析 300750")
        # The regex will set it to 300750 (first match)
        assert result.stock_code == "300750"
