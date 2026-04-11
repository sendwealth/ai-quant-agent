# AI Quant Agent 架构重构方案 v3.0

> 作者：资深量化架构工程师 | 日期：2026-04-11
> 基于 2026 年 AI Agent 最佳实践 + 生产级量化系统标准

---

## 一、现状诊断

### 1.1 核心问题

| 维度 | 现状 | 严重度 |
|------|------|--------|
| 数据层 | 财务数据伪造（ROE/P/E 硬编码估算），价格推算错误 | 🔴 致命 |
| 策略层 | 无回测引擎，策略验证为零 | 🔴 致命 |
| 执行层 | 无交易执行，只有信号文件输出 | 🔴 致命 |
| Agent 层 | Agent 间无真正协作，各自独立运行输出 JSON | 🟡 严重 |
| 安全层 | StrategyAgent 直接 exec() LLM 输出 | 🔴 致命 |
| 架构层 | 模块重复（3 套数据获取器、2 套指标库），无依赖注入 | 🟡 严重 |

### 1.2 根因分析

系统按"脚本集合"而非"软件系统"设计。每个分析师是独立脚本拼接而成，
缺乏统一的数据总线、事件驱动、状态管理和可观测性。

---

## 二、设计原则

1. **Data First** — 所有分析必须基于真实数据，禁止伪造/估算核心指标
2. **Test Before Trust** — 任何策略上线前必须通过回测+样本外验证
3. **Agent as Service** — Agent 是有状态的服务，不是无状态的函数
4. **Event-Driven** — 通过事件总线解耦模块，支持实时和批量
5. **Observe Everything** — 全链路可观测，每笔分析可追溯
6. **Fail Safe** — 任何组件故障不影响系统安全（降级 > 崩溃）

---

## 三、目标架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AI Quant Agent v3.0                         │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │  Gateway  │  │  CLI/UI  │  │  Web API │  │  Scheduler/Cron  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘   │
│       │              │              │                  │             │
│       └──────────────┴──────┬───────┴──────────────────┘             │
│                             │                                      │
│                    ┌────────▼────────┐                             │
│                    │   Event Bus     │  ← AsyncIO + Redis Streams  │
│                    │   (事件总线)     │                             │
│                    └────────┬────────┘                             │
│                             │                                      │
│       ┌─────────────────────┼─────────────────────┐                │
│       │                     │                     │                │
│  ┌────▼─────┐        ┌─────▼─────┐        ┌──────▼──────┐         │
│  │ Orchestrator│      │  Data     │        │  Execution  │         │
│  │ (编排器)    │      │  Pipeline │        │  Pipeline   │         │
│  │            │      │  (数据管线) │        │  (执行管线)  │         │
│  └────┬─────┘        └─────┬─────┘        └──────┬──────┘         │
│       │                    │                     │                 │
│  ┌────▼────────────────────┴─────────────────────▼──────┐          │
│  │                    Agent Framework                     │          │
│  │                                                         │          │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │          │
│  │  │ Planner  │ │ Research │ │ Analyst  │ │ Risk     │ │          │
│  │  │ Agent    │ │ Agent    │ │ Agents   │ │ Agent    │ │          │
│  │  │(规划)    │ │(研究)    │ │(分析)    │ │(风控)    │ │          │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ │          │
│  │                                                         │          │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────────────┐   │          │
│  │  │ Strategy │ │ Backtest │ │ Memory (RAG + State) │   │          │
│  │  │ Agent    │ │ Agent    │ │                      │   │          │
│  │  │(策略)    │ │(回测)    │ │ • 短期: Redis        │   │          │
│  │  └──────────┘ └──────────┘ │ • 长期: PostgreSQL   │   │          │
│  │                             │ • 向量: Chroma/PG    │   │          │
│  │                             └──────────────────────┘   │          │
│  └─────────────────────────────────────────────────────────┘          │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Infrastructure Layer                      │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────────────┐ │    │
│  │  │ Data    │ │ Metrics │ │ Config  │ │ Logging/Tracing │ │    │
│  │  │ Sources │ │ (Prom)  │ │ Center  │ │ (OTel/Loki)     │ │    │
│  │  │ (MCP)   │ │         │ │         │ │                  │ │    │
│  │  └─────────┘ └─────────┘ └─────────┘ └──────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 四、分层设计

### 4.1 Data Pipeline（数据管线）— 重建核心

**这是最优先的重构。没有真实数据，其他一切毫无意义。**

```
┌─────────────────────────────────────────────────┐
│                 Data Pipeline                   │
│                                                  │
│  Sources ──▶ Normalizer ──▶ Store ──▶ Serve     │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │ Data Sources (MCP Server)                │   │
│  │                                          │   │
│  │  Level 1 - 行情数据（实时）               │   │
│  │  ├── AkShare (免费，主用)                 │   │
│  │  ├── Tushare Pro (需token，补充)          │   │
│  │  └── 东方财富/新浪 (备用)                 │   │
│  │                                          │   │
│  │  Level 2 - 财务数据（日频更新）           │   │
│  │  ├── Tushare 财务报表接口                │   │
│  │  │   ├── income (利润表)                 │   │
│  │  │   ├── balancesheet (资产负债表)       │   │
│  │  │   └── cashflow (现金流量表)           │   │
│  │  ├── AkShare 财务指标                     │   │
│  │  └── 东方财富财务摘要                     │   │
│  │                                          │   │
│  │  Level 3 - 另类数据（周频）              │   │
│  │  ├── 融资融券余额                        │   │
│  │  ├── 龙虎榜                              │   │
│  │  ├── 大宗交易                            │   │
│  │  └── 机构调研                            │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │ Data Normalizer                          │   │
│  │  • 统一 schema (OHLCV + 指标)            │   │
│  │  • 数据质量校验 (null, 异常值, 停牌)      │   │
│  │  • 前复权/后复权标准化                    │   │
│  │  • 复用因子 (不重复计算)                  │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │ Data Store                               │   │
│  │  • 行情: TimescaleDB (时序) / DuckDB      │   │
│  │  • 财务: PostgreSQL                       │   │
│  │  • 因子: Redis (热) + Parquet (冷)        │   │
│  │  • 向量: pgvector (RAG)                   │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

**关键改变：**

```python
# ❌ 现在：伪造数据
def _estimate_roe(pb_ratio: float) -> float:
    if pb_ratio < 2: return 0.08
    elif pb_ratio < 4: return 0.12
    else: return 0.18

# ✅ 重构：真实数据 + 缓存
class FinancialDataService:
    def get_roe(self, stock_code: str, period: str = "latest") -> float:
        """从 tushare 获取真实 ROE"""
        cached = self.cache.get(f"roe:{stock_code}:{period}")
        if cached and not self._is_stale(cached):
            return cached.value

        raw = self.tushare_api.fetch_income(stock_code)
        roe = self._calc_roe_from_financials(raw)  # 净利润/净资产
        self.cache.set(f"roe:{stock_code}:{period}", roe)
        return roe

    def get_financial_snapshot(self, stock_code: str) -> FinancialSnapshot:
        """一次性获取完整财务快照，避免多次网络请求"""
        return self._batch_fetch(stock_code, fields=[
            "roe", "gross_margin", "net_margin", "debt_ratio",
            "current_ratio", "revenue_growth", "profit_growth",
            "pe_ttm", "pb", "ps_ttm", "ev_ebitda"
        ])
```

### 4.2 Agent Framework — 从"脚本"升级为"Agent"

**2026 年 AI Agent 核心模式：**

```
┌──────────────────────────────────────────────────────┐
│                   Agent Framework                     │
│                                                       │
│  每个 Agent 是一个有状态的服务，遵循统一协议：          │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │  BaseAgent Protocol                            │  │
│  │                                                 │  │
│  │  class BaseAgent:                              │  │
│  │      name: str                                 │  │
│  │      role: AgentRole                           │  │
│  │      tools: list[Tool]      # MCP Tools        │  │
│  │      memory: AgentMemory    # 短期+长期         │  │
│  │      llm: LLMBackend        # 可切换模型        │  │
│  │                                                 │  │
│  │      async def plan(task) -> Plan              │  │
│  │      async def execute(step) -> Result         │  │
│  │      async def reflect(results) -> Insight     │  │
│  │      async def report() -> AgentReport         │  │
│  │                                                 │  │
│  │      # 工具通过 MCP 协议注册                    │  │
│  │      @mcp_tool                                  │  │
│  │      def get_financial_data(...)               │  │
│  │      @mcp_tool                                  │  │
│  │      def calculate_indicator(...)              │  │
│  │      @mcp_tool                                  │  │
│  │      def search_knowledge(...)                 │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
│  Agent 间通信不通过文件，通过 Event Bus：              │
│                                                       │
│  ResearchAgent ──发布──▶ stock.research.completed     │
│       │                     │                        │
│       │              ┌──────▼──────┐                 │
│       │              │ Event Bus   │                 │
│       │              └──────┬──────┘                 │
│       │                     │                        │
│  PlannerAgent ◀──消费──── stock.research.completed    │
│       │                                              │
│       ├──▶ dispatch analyst.task {                    │
│       │      analysts: [fundamental, technical],      │
│       │      stock: "300750",                         │
│       │      context: research_results                │
│       │    }                                         │
│       │                                              │
│       └──▶ AnalystAgents 并行执行                     │
│              │                                       │
│              ▼                                       │
│         analyst.{name}.completed                     │
│              │                                       │
│              ▼                                       │
│         RiskAgent ──▶ risk.assessment.completed       │
│              │                                       │
│              ▼                                       │
│         Decision: BUY/HOLD/SELL + Position Size      │
└──────────────────────────────────────────────────────┘
```

### 4.3 Agent 角色重新定义

**从"独立脚本"变为"协作 Agent 团队"：**

| 角色 | 职责 | 输入 | 输出 | 关键改变 |
|------|------|------|------|----------|
| **PlannerAgent** | 解析用户意图，分解任务，编排执行 | 自然语言指令 | 执行计划 | 新增。当前系统无规划能力 |
| **ResearchAgent** | 深度研究：行业、公司、竞争格局 | 股票代码 | 研究报告 | 升级。从 RAG 检索真实研报+财报 |
| **FundamentalAgent** | 基本面量化分析 | 真实财务数据 | 估值评分 | 重写。数据必须来自真实 API |
| **TechnicalAgent** | 技术面分析 | 真实行情数据 | 技术信号 | 保留框架，修复 bug |
| **SentimentAgent** | 市场情绪分析 | 行情+另类数据 | 情绪分数 | 保留框架，接入真实数据 |
| **StrategyAgent** | 策略生成与验证 | 分析结果 | 策略代码 | 重写。禁止 exec()，用白名单沙箱 |
| **BacktestAgent** | 策略回测验证 | 策略+历史数据 | 回测报告 | 新增。当前完全缺失 |
| **RiskAgent** | 风险评估与仓位管理 | 所有分析结果 | 风险报告 | 重写。实时风控+动态仓位 |
| **ExecutionAgent** | 订单执行（模拟/实盘） | 交易决策 | 执行结果 | 新增。当前无执行层 |

### 4.4 事件驱动架构

```python
# events.py — 事件定义（结构化，类型安全）
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

class EventType(str, Enum):
    # 数据事件
    DATA_QUOTE_UPDATED = "data.quote.updated"
    DATA_FINANCIAL_UPDATED = "data.financial.updated"
    
    # 分析事件
    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"
    
    # 决策事件
    SIGNAL_GENERATED = "signal.generated"
    RISK_CHECK_PASSED = "risk.check.passed"
    RISK_CHECK_FAILED = "risk.check.failed"
    ORDER_SUBMITTED = "order.submitted"
    ORDER_FILLED = "order.filled"
    
    # 系统事件
    SYSTEM_HEARTBEAT = "system.heartbeat"
    SYSTEM_ERROR = "system.error"

@dataclass
class Event:
    type: EventType
    payload: dict
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: str = ""  # 链路追踪
    source: str = ""         # 来源 Agent
    metadata: dict = field(default_factory=dict)

# event_bus.py — 事件总线
class EventBus:
    """基于 Redis Streams 的异步事件总线"""
    
    async def publish(self, event: Event): ...
    async def subscribe(self, event_type: EventType, handler: Callable): ...
    async def publish_and_wait(self, event: Event, timeout: float = 30) -> Event: ...

# 使用示例
class FundamentalAgent(BaseAgent):
    async def on_quote_updated(self, event: Event):
        """当行情更新时，自动触发基本面分析"""
        stock_code = event.payload["stock_code"]
        
        # 检查是否需要更新（避免频繁计算）
        if not self._should_analyze(stock_code):
            return
        
        # 发布分析开始事件
        await self.bus.publish(Event(
            type=EventType.ANALYSIS_STARTED,
            payload={"stock_code": stock_code, "analyst": self.name},
            source=self.name,
            correlation_id=event.correlation_id
        ))
        
        # 执行分析
        result = await self.analyze(stock_code)
        
        # 发布分析完成事件
        await self.bus.publish(Event(
            type=EventType.ANALYSIS_COMPLETED,
            payload={"stock_code": stock_code, "result": result},
            source=self.name,
            correlation_id=event.correlation_id
        ))
```

### 4.5 MCP（Model Context Protocol）工具层

**2026 年标准：所有 Agent 工具通过 MCP 协议暴露，实现可组合、可审计。**

```python
# mcp_tools.py — MCP 工具注册

class QuantMCPServer:
    """量化分析 MCP Server"""
    
    @mcp_tool(
        name="get_stock_price",
        description="获取股票实时/历史价格",
        parameters={
            "stock_code": {"type": "string", "description": "股票代码"},
            "period": {"type": "string", "enum": ["daily", "weekly", "monthly"]},
            "days": {"type": "integer", "default": 250}
        }
    )
    async def get_stock_price(self, stock_code: str, period: str = "daily", days: int = 250):
        df = await self.data_service.get_price(stock_code, period, days)
        return df.to_dict(orient="records")
    
    @mcp_tool(
        name="get_financial_statements",
        description="获取财务报表（利润表、资产负债表、现金流量表）",
        parameters={
            "stock_code": {"type": "string"},
            "statement": {"type": "string", "enum": ["income", "balance", "cashflow"]},
            "periods": {"type": "integer", "default": 4, "description": "最近N期"}
        }
    )
    async def get_financial_statements(self, stock_code: str, statement: str, periods: int = 4):
        return await self.data_service.get_financials(stock_code, statement, periods)
    
    @mcp_tool(
        name="calculate_indicator",
        description="计算技术指标（RSI, MACD, Bollinger, etc.）",
        parameters={
            "data": {"type": "array", "description": "OHLCV 数据"},
            "indicator": {"type": "string", "enum": ["rsi", "macd", "bollinger", "kdj", "atr"]},
            "params": {"type": "object", "description": "指标参数"}
        }
    )
    async def calculate_indicator(self, data: list, indicator: str, params: dict = None):
        return self.indicator_service.calculate(data, indicator, params)
    
    @mcp_tool(
        name="run_backtest",
        description="运行策略回测",
        parameters={
            "strategy": {"type": "object", "description": "策略定义"},
            "stock_code": {"type": "string"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "initial_capital": {"type": "number", "default": 100000}
        }
    )
    async def run_backtest(self, strategy: dict, stock_code: str, ...):
        return await self.backtest_engine.run(...)
    
    @mcp_tool(
        name="search_research",
        description="搜索研报/公告/新闻（RAG）",
        parameters={
            "query": {"type": "string"},
            "stock_code": {"type": "string"},
            "top_k": {"type": "integer", "default": 5}
        }
    )
    async def search_research(self, query: str, stock_code: str = "", top_k: int = 5):
        return await self.rag_service.search(query, stock_code, top_k)
```

### 4.6 回测引擎 — 从零建设

```python
# backtest/engine.py

class BacktestEngine:
    """生产级回测引擎"""
    
    def __init__(self, config: BacktestConfig):
        self.data_service = DataService(config.data)
        self.commission = config.commission  # 手续费模型
        self.slippage = config.slippage      # 滑点模型
    
    async def run(self, strategy: Strategy, universe: list[str],
                  start: str, end: str) -> BacktestResult:
        """
        运行回测
        
        返回：
        - 年化收益率、最大回撤、夏普比率、Sortino 比率
        - Calmar 比率、胜率、盈亏比、最大连续亏损
        - 月度/年度收益分布
        - 与基准对比（相对收益）
        """
        # 1. 加载全量数据（避免前视偏差）
        data = await self.data_service.load(universe, start, end)
        
        # 2. 逐日模拟
        portfolio = Portfolio(initial_capital=strategy.capital)
        signals = []
        
        for date in self._trading_dates(start, end):
            # 截断到当前日期（防止前视偏差）
            available_data = data[:date]
            
            # 生成信号（只看历史数据）
            signal = strategy.generate_signal(available_data, date)
            
            if signal:
                # 计算滑点
                executed_price = self._apply_slippage(
                    signal.price, signal.direction, available_data
                )
                # 执行交易
                portfolio.execute(signal, executed_price, self.commission)
                signals.append(signal)
        
        # 3. 计算绩效
        return self._calculate_metrics(portfolio, signals, data)
    
    def _calculate_metrics(self, portfolio, signals, data) -> BacktestResult:
        """计算完整绩效指标"""
        equity_curve = portfolio.equity_curve
        
        # 基础指标
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        annual_return = self._annualize(total_return, len(equity_curve))
        
        # 风险指标
        daily_returns = pd.Series(equity_curve).pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        max_drawdown = self._calc_max_drawdown(equity_curve)
        
        # 风险调整收益
        sharpe = annual_return / volatility if volatility > 0 else 0
        sortino = annual_return / self._downside_vol(daily_returns)
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 交易统计
        trades = portfolio.closed_trades
        win_rate = len([t for t in trades if t.pnl > 0]) / len(trades) if trades else 0
        avg_win = np.mean([t.pnl for t in trades if t.pnl > 0]) if trades else 0
        avg_loss = np.mean([t.pnl for t in trades if t.pnl <= 0]) if trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # 与基准对比
        benchmark_return = self._calc_benchmark_return(data.benchmark)
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            benchmark_return=benchmark_return,
            alpha=annual_return - benchmark_return,
            equity_curve=equity_curve,
            trades=trades
        )
```

### 4.7 策略安全沙箱

```python
# strategy/sandbox.py — 替代 exec()

class StrategySandbox:
    """
    策略安全沙箱
    
    禁止：exec(), eval(), import os, subprocess, socket...
    允许：pandas, numpy, 指标计算, 信号生成
    """
    
    # 白名单模块
    ALLOWED_MODULES = {
        "pandas": ["DataFrame", "Series"],
        "numpy": ["array", "mean", "std", "where", "sqrt"],
        "indicators": ["rsi", "macd", "sma", "ema", "bollinger_bands", "atr"],
    }
    
    # 禁止操作
    BLOCKED_PATTERNS = [
        r"\bexec\s*\(", r"\beval\s*\(", r"\b__import__\b",
        r"\bos\.", r"\bsubprocess\.", r"\bsocket\.",
        r"\bopen\s*\(", r"\bshutil\.", r"\brequests\.",
    ]
    
    def validate(self, code: str) -> ValidationResult:
        """静态分析策略代码安全性"""
        errors = []
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, code):
                errors.append(f"禁止使用: {pattern}")
        
        # AST 分析
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.ALLOWED_MODULES:
                        errors.append(f"禁止导入: {alias.name}")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
    
    def execute(self, code: str, data: pd.DataFrame) -> pd.Series:
        """在受限环境中执行策略"""
        validation = self.validate(code)
        if not validation.is_valid:
            raise StrategySecurityError(validation.errors)
        
        # 构建受限命名空间
        namespace = {
            "pd": pd,
            "np": np,
            "df": data,
            **self._safe_indicators(),
        }
        
        # 使用 RestrictedPython 或自定义执行器
        exec(compile(code, "<strategy>", "exec"), 
             {"__builtins__": self._restricted_builtins()}, namespace)
        
        if "signals" not in namespace:
            raise StrategyError("策略必须输出 'signals' 变量")
        
        return namespace["signals"]
```

### 4.8 Memory & RAG 层

```python
# memory/agent_memory.py

class AgentMemory:
    """Agent 记忆系统"""
    
    def __init__(self, config):
        # 短期记忆：当前分析会话的状态
        self.working_memory = {}  # Redis，TTL 1h
        
        # 长期记忆：历史分析、市场观察
        self.long_term = VectorStore(config.vector_db)
        
        # 知识库：行业知识、公司信息、研究报告
        self.knowledge_base = RAGService(config.rag)
    
    async def remember(self, key: str, value: Any, ttl: int = 3600):
        """存储到工作记忆"""
        await self.redis.setex(f"wm:{key}", ttl, json.dumps(value))
    
    async def recall(self, key: str) -> Any:
        """从工作记忆读取"""
        data = await self.redis.get(f"wm:{key}")
        return json.loads(data) if data else None
    
    async def store_insight(self, insight: Insight):
        """存储洞察到长期记忆（向量化）"""
        embedding = await self.embed(insight.content)
        await self.long_term.insert(
            id=insight.id,
            embedding=embedding,
            metadata={
                "stock_code": insight.stock_code,
                "agent": insight.agent,
                "timestamp": insight.timestamp,
                "type": insight.type,
            },
            content=insight.content
        )
    
    async def search_similar(self, query: str, stock_code: str = "", 
                              top_k: int = 5) -> list[Insight]:
        """RAG 检索相似历史洞察"""
        embedding = await self.embed(query)
        results = await self.long_term.search(
            embedding, top_k=top_k,
            filter={"stock_code": stock_code} if stock_code else None
        )
        return results
```

### 4.9 可观测性

```python
# observability/tracing.py

class QuantTracer:
    """全链路追踪"""
    
    def trace_analysis(self, stock_code: str, agents: list[str]) -> TraceContext:
        """
        追踪一次完整分析链路
        
        Example trace:
        ├─ data.fetch.price (300750)        120ms  ✅
        ├─ data.fetch.financials (300750)    450ms  ✅
        ├─ agent.research (300750)          2.1s   ✅
        │  ├─ rag.search("宁德时代 Q4")     200ms  ✅
        │  └─ llm.generate(research)        1.8s   ✅
        ├─ agent.fundamental (300750)       800ms  ✅
        │  ├─ data.get_roe()                50ms   ✅ (cached)
        │  └─ calc_valuation()              200ms  ✅
        ├─ agent.technical (300750)         600ms  ✅
        ├─ agent.risk (300750)              300ms  ✅
        └─ decision.generate                100ms  ✅ BUY 85%
        """
        trace_id = str(uuid.uuid4())
        # ... OpenTelemetry span 管理
```

---

## 五、目录结构

```
ai-quant-agent/
├── pyproject.toml
├── docker-compose.yml          # PostgreSQL + Redis + TimescaleDB
├── Dockerfile
│
├── src/
│   └── quant_agent/
│       ├── __init__.py
│       ├── main.py              # 入口：CLI / API / Scheduler
│       ├── config.py            # Pydantic 配置（替代 YAML 散落）
│       │
│       ├── events/              # 事件系统
│       │   ├── __init__.py
│       │   ├── models.py        # Event, EventType 定义
│       │   └── bus.py           # EventBus (Redis Streams)
│       │
│       ├── data/                # 数据管线（核心重构）
│       │   ├── __init__.py
│       │   ├── sources/         # 数据源适配器
│       │   │   ├── base.py      # DataSource 抽象
│       │   │   ├── akshare.py   # AkShare 适配器
│       │   │   ├── tushare.py   # Tushare 适配器
│       │   │   └── eastmoney.py # 东方财富适配器
│       │   ├── normalizer.py    # 数据标准化
│       │   ├── validator.py     # 数据质量校验
│       │   ├── store.py         # 数据存储抽象
│       │   └── service.py       # DataService 统一入口
│       │
│       ├── agents/              # Agent 框架
│       │   ├── __init__.py
│       │   ├── base.py          # BaseAgent 协议
│       │   ├── planner.py       # PlannerAgent
│       │   ├── researcher.py    # ResearchAgent (RAG)
│       │   ├── fundamental.py   # FundamentalAgent
│       │   ├── technical.py     # TechnicalAgent
│       │   ├── sentiment.py     # SentimentAgent
│       │   ├── strategy.py      # StrategyAgent
│       │   ├── backtest.py      # BacktestAgent
│       │   ├── risk.py          # RiskAgent
│       │   └── execution.py     # ExecutionAgent
│       │
│       ├── strategy/            # 策略系统
│       │   ├── __init__.py
│       │   ├── sandbox.py       # 安全沙箱
│       │   ├── indicators.py    # 统一指标库（删除重复）
│       │   ├── templates.py     # 策略模板
│       │   └── optimizer.py     # 参数优化（网格/贝叶斯）
│       │
│       ├── backtest/            # 回测引擎（新建）
│       │   ├── __init__.py
│       │   ├── engine.py        # 回测引擎
│       │   ├── portfolio.py     # 组合模拟
│       │   ├── commission.py    # 手续费模型
│       │   ├── slippage.py      # 滑点模型
│       │   └── metrics.py       # 绩效计算
│       │
│       ├── memory/              # 记忆系统
│       │   ├── __init__.py
│       │   ├── working.py       # 短期记忆 (Redis)
│       │   ├── long_term.py     # 长期记忆 (Vector DB)
│       │   └── rag.py           # RAG 检索
│       │
│       ├── mcp/                 # MCP 工具层
│       │   ├── __init__.py
│       │   ├── server.py        # MCP Server
│       │   └── tools/           # MCP 工具注册
│       │
│       ├── execution/           # 交易执行
│       │   ├── __init__.py
│       │   ├── paper.py         # 模拟交易
│       │   ├── live.py          # 实盘交易（预留）
│       │   └── broker.py        # 券商接口抽象
│       │
│       └── observability/       # 可观测性
│           ├── __init__.py
│           ├── tracing.py       # 链路追踪
│           ├── metrics.py       # 指标采集
│           └── logging.py       # 结构化日志
│
├── tests/                      # 测试（目标覆盖率 >70%）
│   ├── unit/
│   │   ├── test_data_service.py
│   │   ├── test_indicators.py
│   │   ├── test_backtest.py
│   │   ├── test_agents/
│   │   └── test_sandbox.py
│   ├── integration/
│   │   ├── test_data_pipeline.py
│   │   └── test_event_bus.py
│   └── fixtures/
│       └── market_data/        # 固定测试数据
│
├── config/
│   └── default.toml            # 统一配置文件
│
├── scripts/
│   ├── seed_data.py            # 数据初始化
│   ├── run_backtest.py         # 回测入口
│   └── run_analysis.py         # 分析入口
│
└── docs/
    ├── architecture.md
    ├── data_sources.md
    └── api_reference.md
```

---

## 六、技术选型

| 层级 | 技术 | 理由 |
|------|------|------|
| 语言 | Python 3.11+ | 量化生态最成熟 |
| 异步框架 | AsyncIO + aiohttp | 事件驱动基础 |
| 事件总线 | Redis Streams | 轻量、持久化、消费者组 |
| 数据库 | PostgreSQL + TimescaleDB | 关系数据 + 时序数据一体化 |
| 向量数据库 | pgvector | 与 PostgreSQL 同库，减少运维 |
| 缓存 | Redis | 热数据、会话状态 |
| 回测 | 自研（基于 pandas） | 轻量可控，避免 backtrader 复杂度 |
| LLM | OpenAI GPT-4o / 智谱 GLM-4 | Agent 推理 + 研报分析 |
| 嵌入模型 | text-embedding-3-small | RAG 向量化 |
| 配置管理 | Pydantic Settings + TOML | 类型安全、环境变量覆盖 |
| 可观测性 | OpenTelemetry + Prometheus + Loki | 标准化全链路追踪 |
| 容器化 | Docker Compose | 一键部署 |
| 包管理 | uv | 快速、确定性的依赖管理 |
| 测试 | pytest + pytest-asyncio + hypothesis | 含属性测试 |

---

## 七、实施路线图

### Phase 1：数据层重建（2 周）🔴 最高优先级

```
Week 1: 数据源 + 存储
├── 统一 DataService 接口
├── Tushare 财务数据接入（利润表/资产负债表/现金流量表）
├── AkShare 行情数据接入 + 重试/退避
├── 数据校验层（停牌、异常值、缺失值处理）
├── Docker Compose（PostgreSQL + TimescaleDB + Redis）
└── 数据初始化脚本

Week 2: 数据服务 + 指标
├── FinancialDataService（真实 ROE、毛利率、增长率计算）
├── 统一指标库（合并 core/indicators + utils/indicators）
├── 数据缓存层（Redis 热缓存 + Parquet 冷存储）
└── 删除所有伪造数据的代码
```

**验收标准：** `get_financial_snapshot("300750")` 返回真实数据，与东方财富/同花顺对比误差 < 5%。

### Phase 2：回测引擎（1.5 周）

```
Week 3: 回测核心
├── BacktestEngine 逐日模拟
├── 手续费模型（A股：佣金+印花税+过户费）
├── 滑点模型（基于成交量）
├── 绩效指标（Sharpe, Sortino, MaxDD, Calmar, Win Rate）
└── 基准对比

Week 4 前半: 策略沙箱
├── StrategySandbox 安全执行
├── 内置策略模板（MA交叉、RSI反转、MACD动量）
└── 参数优化器（网格搜索）
```

**验收标准：** 能对 300750 运行 MA 交叉策略回测，输出完整绩效报告。

### Phase 3：Agent 框架升级（2 周）

```
Week 4 后半 + Week 5: Agent 基础
├── BaseAgent 协议 + MCP 工具注册
├── EventBus (Redis Streams)
├── PlannerAgent（任务分解 + 编排）
├── 重写 FundamentalAgent（基于真实数据）
├── 重写 TechnicalAgent（修复 bug）
└── 重写 RiskAgent（动态仓位 + 实时风控）

Week 6: 高级 Agent
├── ResearchAgent + RAG（研报/公告检索）
├── StrategyAgent（白名单沙箱，禁止 exec）
├── BacktestAgent（策略自动回测验证）
└── Agent 间协作编排
```

### Phase 4：执行层 + 生产化（1.5 周）

```
Week 7: 执行 + 监控
├── ExecutionAgent + PaperTrading
├── 全链路追踪 (OpenTelemetry)
├── 健康检查 + 告警
├── API 接口（FastAPI）
└── 文档

Week 8: 稳定性
├── 测试覆盖率 > 70%
├── 错误恢复 + 降级策略
├── CI/CD
└── 性能优化
```

---

## 八、关键代码示例

### 8.1 统一入口

```python
# main.py

import asyncio
from quant_agent.config import Settings
from quant_agent.events.bus import EventBus
from quant_agent.data.service import DataService
from quant_agent.agents.planner import PlannerAgent
from quant_agent.observability.tracing import QuantTracer

async def analyze_stock(stock_code: str, mode: str = "full"):
    """分析一只股票的完整流程"""
    settings = Settings()
    
    # 初始化基础设施
    tracer = QuantTracer(settings)
    bus = EventBus(settings.redis)
    data = DataService(settings)
    
    # 初始化 Agent 团队
    planner = PlannerAgent(bus=bus, data=data, memory=...)
    
    with tracer.trace(f"analyze.{stock_code}") as trace:
        # 1. 数据准备
        trace.span("data.prepare")
        await data.ensure_stock_data(stock_code)
        
        # 2. 规划 + 编排
        trace.span("agent.plan")
        plan = await planner.create_plan(f"分析 {stock_code}，给出投资建议")
        
        # 3. 并行执行分析师
        trace.span("agent.execute")
        results = await planner.execute_plan(plan)
        
        # 4. 风控审核
        trace.span("risk.check")
        risk_report = await planner.check_risk(results)
        
        # 5. 生成决策
        decision = await planner.make_decision(results, risk_report)
    
    return decision

if __name__ == "__main__":
    import fire
    fire.Fire({
        "analyze": analyze_stock,
        "backtest": run_backtest,
        "serve": start_api_server,
    })
```

### 8.2 Agent 协作示例

```python
# agents/planner.py

class PlannerAgent(BaseAgent):
    """编排 Agent — 协调其他 Agent 的执行"""
    
    async def create_plan(self, user_request: str) -> Plan:
        """用 LLM 理解意图，分解为可执行的步骤"""
        
        plan = await self.llm.structured_output(
            system="你是一个量化分析规划师。根据用户请求，分解为具体的分析步骤。",
            prompt=f"用户请求: {user_request}",
            output_schema=Plan
        )
        
        return plan
        # Example output:
        # Plan(steps=[
        #   Step(agent="researcher", task="研究宁德时代行业地位和竞争格局"),
        #   Step(agent="fundamental", task="分析宁德时代财务数据"),
        #   Step(agent="technical", task="分析宁德时代技术面"),
        #   Step(agent="sentiment", task="分析宁德时代市场情绪"),
        #   Step(agent="risk", task="综合风险评估", depends_on=["fundamental", "technical", "sentiment"]),
        # ])
    
    async def execute_plan(self, plan: Plan) -> dict:
        """按依赖关系并行执行"""
        
        # 拓扑排序 + 并行执行
        completed = {}
        pending = set(step.id for step in plan.steps)
        
        while pending:
            # 找到所有依赖已满足的步骤
            ready = [
                s for s in plan.steps
                if s.id in pending
                and all(d in completed for d in s.depends_on)
            ]
            
            if not ready:
                raise PlanError("循环依赖或无法满足的依赖")
            
            # 并行执行就绪步骤
            tasks = [self._execute_step(s) for s in ready]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for step, result in zip(ready, results):
                if isinstance(result, Exception):
                    self.logger.error(f"Step {step.id} failed: {result}")
                    completed[step.id] = {"status": "failed", "error": str(result)}
                else:
                    completed[step.id] = result
                    pending.discard(step.id)
            
            # 发布进度事件
            await self.bus.publish(Event(
                type=EventType.ANALYSIS_COMPLETED,
                payload={"completed": len(completed), "pending": len(pending)}
            ))
        
        return completed
    
    async def make_decision(self, results: dict, risk_report: dict) -> Decision:
        """基于所有分析结果做出投资决策"""
        
        # LLM 综合判断
        decision = await self.llm.structured_output(
            system="""你是一个投资决策者。基于多位分析师的报告和风控审核，
            做出最终投资决策。必须给出明确的 BUY/HOLD/SELL 信号和建议仓位。""",
            prompt=f"""
            分析师报告: {json.dumps(results, ensure_ascii=False)}
            风控报告: {json.dumps(risk_report, ensure_ascii=False)}
            """,
            output_schema=Decision
        )
        
        return decision
```

### 8.3 配置管理

```python
# config.py — Pydantic Settings，替代散落的 YAML

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    """统一配置，支持环境变量覆盖"""
    
    # 应用
    app_name: str = "ai-quant-agent"
    debug: bool = False
    
    # 数据源
    tushare_token: Optional[str] = None
    akshare_timeout: int = 10
    data_cache_ttl: int = 1800  # 30 分钟
    
    # 数据库
    postgres_url: str = "postgresql://localhost/quant"
    redis_url: str = "redis://localhost:6379"
    
    # LLM
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"
    zhipu_api_key: Optional[str] = None
    zhipu_model: str = "glm-4"
    
    # 风控
    max_position_pct: float = 0.20
    max_portfolio_risk: float = 0.80
    default_stop_loss: float = -0.08
    default_take_profit: tuple[float, float] = (0.10, 0.20)
    
    # 回测
    commission_rate: float = 0.0003
    stamp_tax: float = 0.001  # A股印花税（卖出）
    slippage_bps: float = 1.0  # 滑点基点
    
    model_config = {"env_prefix": "QUANT_", "env_file": ".env"}
```

---

## 九、风险与降级策略

| 故障场景 | 降级策略 |
|----------|----------|
| Tushare API 不可用 | 切换 AkShare → 本地缓存 → 暂停分析 |
| LLM 不可用 | 规则引擎降级（固定阈值策略） |
| Redis 不可用 | 内存队列 + 本地文件存储 |
| 数据异常（停牌/涨跌停） | 标记异常，跳过当日分析 |
| 回测引擎 OOM | 分批加载 + 流式计算 |

---

## 十、总结

当前系统的根本问题是 **"有 Agent 之名，无系统之实"**。各模块是独立脚本拼凑，
数据层严重造假，缺乏回测验证和真实执行能力。

v3.0 重构的核心思想：

1. **Data First** — 真实数据是一切的基础，禁止伪造
2. **Event-Driven Agent** — Agent 是有状态的服务，通过事件总线协作
3. **MCP Tools** — 所有工具通过 MCP 协议注册，可组合、可审计
4. **Backtest Before Trust** — 任何策略必须通过回测验证
5. **Observe Everything** — 全链路追踪，每笔分析可追溯
6. **Fail Safe** — 任何组件故障安全降级

预计总工时 **7 周**，优先级：数据层 > 回测引擎 > Agent 框架 > 执行层。

---

## 附录：实现状态 (2026-04-11 更新)

> 本文档描述的是目标架构。以下标注各模块的实际实现状态。

### 已完成 (Phase 1)

| 模块 | 状态 | 说明 |
|---|---|---|
| DataSource ABC | ✅ | base.py 定义完整接口 |
| TushareSource | ✅ | 财务报表 + ROE 交叉验证 |
| AkshareSource | ✅ | 行情数据 + 重试退避 |
| BaoStockSource | ✅ | 免费行情，DataSource ABC 合规 |
| DataService 降级链 | ✅ | 多源降级 + 缓存 + 过期检查 |
| 数据验证 | ✅ | normalizer (缺失列抛异常) + validator (非正价格拒绝) |
| 输入验证 | ✅ | stock_code 6 位 A 股代码强制校验 |
| BaseAgent + AgentResult | ✅ | **kwargs 多态接口 |
| FundamentalAgent | ✅ | 真实财务数据评分 |
| TechnicalAgent | ✅ | 技术指标信号生成 |
| RiskAgent | ✅ | 成功结果共识 + 动态仓位 |
| ExecutionAgent | ✅ | 模拟交易 + 止损止盈 + A 股手续费 |
| EventBus | ✅ | 内存同步 pub/sub + 历史记录 |
| 回测引擎 | ✅ | Sharpe/Sortino/Calmar/MaxDD/Alpha/Beta |
| 技术指标库 | ✅ | RSI/MACD/EMA/ATR/ADX/布林带/OBV (向量化) |
| MetricsCollector | ✅ | 计数器/仪表/计时器 |
| 单元测试 | ✅ | 146 passed (含数据管道集成测试) |

### 未实现 (Phase 2-3)

| 模块 | 阶段 | 说明 |
|---|---|---|
| Async EventBus + Redis | Phase 2 | 当前为内存同步 |
| PlannerAgent (LLM) | Phase 2 | 未实现 |
| ResearchAgent (RAG) | Phase 3 | 未实现 |
| SentimentAgent | Phase 3 | 未实现 |
| MCP 工具协议 | Phase 3 | 空 `__init__.py` |
| 记忆系统 | Phase 3 | 空 `__init__.py` |
| 真实交易执行 | Phase 3 | 未实现 |
| 异步数据获取 | Phase 2 | 整个数据层同步 |
| 数据源熔断器 | Phase 2 | 未实现 |
