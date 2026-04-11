# AI Quant Agent v3.0 — Claude Code 开发指南

## 项目概述

A 股量化交易系统，LLM 增强 Orchestrator 驱动多 Agent 协作架构。规则引擎 + LLM 双引擎，支持自然语言交互。

## 文档索引

| 文档 | 说明 |
|---|---|
| `docs/architecture-v3.md` | 架构设计（目标状态） |
| `docs/multi-agent-plan.md` | 多智能体完善计划（已完成） |
| `config/agent_thresholds.yaml` | Agent 评分阈值配置 |
| `.env.example` | 环境变量完整列表 |

## 技术栈

- Python 3.10+, 包管理: uv
- LLM: LangChain + LangGraph (OpenAI / 智谱 GLM 双 provider)
- 数据源: akshare (行情), tushare (财务报表), baostock (免费行情)
- 配置: pydantic-settings + dotenv (AliasChoices) + YAML
- 存储: Parquet (当前) → PostgreSQL/TimescaleDB (后续)
- 测试: pytest + pytest-cov (358 passed)

## 目录结构

```
src/quant_agent/
├── main.py                 # CLI 入口 (--stock / --prompt / --daily-report)
├── orchestrator.py         # Orchestrator 编排器 + AnalysisReport
├── config.py               # Pydantic Settings (QUANT_ 前缀, AliasChoices)
├── portfolio.py            # 统一 Portfolio + CommissionModel + Position + Trade
├── audit.py                # 追加式审计日志 (JSONL)
├── thresholds.py           # Agent 评分阈值加载器 (YAML + 默认值)
├── llm/                    # LLM 层
│   ├── client.py           # LLMClient (LangChain ChatOpenAI, OpenAI/智谱自动切换)
│   ├── prompts.py          # 所有 prompt 模板 (情感/规划/报告/风险)
│   └── report.py           # LLMReportGenerator (综合分析报告生成)
├── data/                   # 数据层
│   ├── sources/base.py     # DataSource ABC + FinancialSnapshot (schema 验证)
│   ├── sources/tushare.py  # Tushare 适配器 (财务报表, Settings 传参)
│   ├── sources/akshare.py  # AkShare 适配器 (行情 + 速率限制)
│   ├── sources/baostock.py # BaoStock 适配器 (免费行情, 上下文管理器)
│   ├── rate_limiter.py     # Token bucket 速率限制器
│   ├── normalizer.py       # 列名标准化 (缺失列抛异常)
│   ├── validator.py        # 数据质量校验 (null/停牌/非正价格)
│   ├── validators.py       # 股票代码输入验证 (沪深创北 6 位代码)
│   ├── store.py            # Parquet 存储 (文件锁 + 原子写入)
│   └── service.py          # DataService 统一入口 (多源降级 + 并发获取)
├── strategy/indicators.py  # 统一指标库 (RSI/MACD/EMA/ATR/ADX/布林带/OBV 向量化)
├── backtest/engine.py      # 回测引擎 (Sharpe/Sortino/Calmar/MaxDD/Alpha/Beta)
├── events/bus.py           # EventBus 事件总线 (保留用于未来异步编排)
├── agents/
│   ├── base.py             # BaseAgent ABC + AgentResult + 结构化日志
│   ├── fundamental.py      # 基本面 Agent (真实财务数据, 可配置阈值)
│   ├── technical.py        # 技术分析 Agent (指标 + 信号, 可配置阈值)
│   ├── sentiment.py        # 情感分析 Agent (LLM 新闻情感 → AgentResult)
│   ├── planner.py          # 指令解析 Agent (自然语言 → ExecutionPlan)
│   ├── risk.py             # 风控 Agent (共识 + 仓位 + LLM 风险解读)
│   └── execution.py        # 执行 Agent (模拟交易 + 止损止盈 + 审计日志)
├── execution/
│   └── paper_trading.py    # PaperTradingService (持久化模拟交易)
├── notification/
│   └── email.py            # EmailNotifier (交易信号/每日报告/异常告警)
├── observability/metrics.py # MetricsCollector + HealthChecker
├── memory/                 # 记忆系统 (待实现)
└── mcp/                    # MCP 工具协议 (待实现)
config/
└── agent_thresholds.yaml   # Agent 评分阈值外部配置
tests/
├── unit/                   # 334 单元测试
│   ├── test_agents.py      # Agent + EventBus 测试
│   ├── test_backtest.py    # 回测引擎 (含已知答案确定性测试)
│   ├── test_data_pipeline.py # 数据管道 (降级/缓存/验证)
│   ├── test_data_service.py # DataService 测试
│   ├── test_execution.py   # ExecutionAgent + AuditLogger 测试
│   ├── test_indicators.py  # 技术指标测试
│   ├── test_email_notification.py # 邮件通知测试
│   ├── test_paper_trading.py # PaperTradingService 测试
│   ├── test_technical_agent.py # TechnicalAgent (100% 覆盖)
│   ├── test_llm_client.py  # LLMClient + 单例测试
│   ├── test_llm_report.py  # LLMReportGenerator 测试
│   ├── test_sentiment_agent.py # SentimentAgent 测试
│   └── test_planner_agent.py  # PlannerAgent 测试
└── test_integration.py     # 24 集成测试 (全链路)
docs/architecture-v3.md     # 架构设计文档
docs/multi-agent-plan.md    # 多智能体完善计划
archive/                    # v2.6 旧代码备份
```

## 核心原则

1. **真实数据** — 所有分析基于真实财务报表，禁止伪造/估算核心指标
2. **LLM 增强** — 规则引擎为基础，LLM 提供情感分析、报告生成、风险解读
3. **结构化日志** — Agent 通过 `_log_action()` 输出结构化日志
4. **Fail Safe** — 数据源故障自动降级，LLM 不可用时跳过增强功能
5. **测试先行** — 新模块必须有单元测试 + 集成测试，LLM 测试全部 mock

## 运行

```bash
uv run python -m quant_agent.main --stock 300750       # 单股分析 (规则引擎)
uv run python -m quant_agent.main --prompt "分析宁德时代的买入机会"  # 自然语言 (需 LLM)
uv run python -m quant_agent.main --multi               # 多股 (并发获取)
uv run python -m quant_agent.main --daily-report         # 批量分析 + 每日邮件报告
uv run python scripts/real_backtest.py                   # 真实数据回测
uv run pytest tests/ -v                                  # 运行全量测试 (358)
uv run pytest tests/unit/ -v                             # 仅单元测试 (334)
uv run pytest tests/test_integration.py -v               # 仅集成测试 (24)
```

## 环境变量

所有配置通过 `QUANT_` 前缀环境变量（.env 文件），完整列表见 `.env.example`：
- `QUANT_TUSHARE_TOKEN` — Tushare API token
- `QUANT_OPENAI_API_KEY` — OpenAI API key (LLM 分析/报告/情感)
- `QUANT_ZHIPU_API_KEY` — 智谱 API key (GLM-4, 与 OpenAI 二选一)
- `QUANT_OPENAI_MODEL` — LLM 模型名称 (默认 gpt-4o)
- `QUANT_DATA_DIR` — 数据存储目录
- `QUANT_FETCH_MAX_WORKERS` — 并发获取线程数 (默认 5)
- `QUANT_EMAIL_ENABLED` — 启用邮件通知 (默认 false)
- `QUANT_EMAIL_SMTP_SERVER` — SMTP 服务器 (默认 smtp.163.com)
- `QUANT_EMAIL_SENDER` / `QUANT_EMAIL_PASSWORD` — 发件邮箱 + 授权码
- `QUANT_EMAIL_RECIPIENTS` — 收件人 (逗号分隔)

## LLM 架构

### 双 Provider 支持

| Provider | 环境变量 | Base URL |
|---|---|---|
| OpenAI | `QUANT_OPENAI_API_KEY` | `https://api.openai.com/v1` |
| 智谱 GLM | `QUANT_ZHIPU_API_KEY` | `https://open.bigmodel.cn/api/coding/paas/v4` |

优先级：`openai_api_key` > `zhipu_api_key`。无 key 时 LLM 功能跳过，规则引擎正常运行。

### 4 个 LLM 使用场景

| 场景 | 模块 | 输入 → 输出 |
|---|---|---|
| 分析报告生成 | `llm/report.py` | AgentResult[] → Markdown 投资分析报告 |
| 情感分析 | `agents/sentiment.py` | 新闻数据 → AgentResult (BUY/SELL/HOLD) |
| 智能指令解析 | `agents/planner.py` | 自然语言 → ExecutionPlan (stock_code, days, focus) |
| 风险解读 | `agents/risk.py` | 风控结果 → 自然语言风险解读 |

### AnalysisReport 扩展字段

- `sentiment_result` — 情感分析 Agent 结果 (参与共识投票)
- `llm_analysis` — LLM 综合分析报告 (Markdown)
- `risk_interpretation` — LLM 风险解读

## 关键约束

- A 股交易时间: 9:30-11:30, 13:00-15:00
- A 股手续费: 佣金万三 + 印花税千一(卖) + 最低5元 (统一 CommissionModel)
- 止损 -8%, 止盈 +10%/+20%, 单只仓位 ≤20% (config 可配置)
- 使用 `uv run python` 而非 `python`
- stock_code 必须通过 `validate_stock_code()` 校验 (60/00/30/8x 前缀, 6 位数字)
- LLM 使用 LangChain ChatOpenAI，structured_output 做结构化解析

## 已完成的改进 (2026-04-11)

### P0 — 核心架构
- Orchestrator 类: 提取 main.py 上帝函数为独立编排器
- EventBus 重构: emit_event → _log_action() 结构化日志
- config → Agent: 风险参数通过 Settings 对象注入 RiskAgent/ExecutionAgent
- tushare.py: 移除残留 os.getenv，统一 Settings 传参

### P1 — 数据层加固
- FinancialSnapshot: 构造时 schema 验证 + validate() 方法
- DataStore: 文件锁 (fcntl.flock) + 原子写入 (temp→os.replace)
- BaoStock: 上下文管理器 (__enter__/__exit__) 管理生命周期
- RateLimiter: Token bucket 速率限制 (Tushare 200/min, AkShare 60/min)

### P2 — 测试 & 质量
- 合并 Portfolio: 统一 portfolio.py，消除回测/执行双实现
- 回测确定性测试: 5 个已知答案测试 (手工计算精确验证)
- TechnicalAgent 测试: 56 个测试，100% 代码覆盖
- 集成测试: 24 个全链路测试 (Orchestrator 端到端)

### P3 — 生产就绪
- PaperTradingService: 持久化模拟交易 (JSON 原子写入，进程重启恢复)
- AuditLogger: 追加式 JSONL 审计日志 (月度归档，线程安全)
- 阈值外部化: config/agent_thresholds.yaml (TechnicalAgent + FundamentalAgent)
- 并发获取: ThreadPoolExecutor 批量数据请求
- 邮件通知: EmailNotifier (交易信号/每日报告/异常告警, SMTP_SSL)

### P4 — LLM 智能化
- LLMClient: LangChain ChatOpenAI 封装，OpenAI/智谱双 provider 自动切换
- SentimentAgent: LLM 新闻情感分析，参与 RiskAgent 共识投票
- PlannerAgent: 自然语言 → ExecutionPlan，--prompt CLI 入口
- LLMReportGenerator: 综合多 Agent 结果生成 Markdown 投资分析报告
- RiskAgent 风险解读: LLM 生成自然语言风险说明和应对建议
- 依赖: langchain + langchain-openai + langgraph

### 测试
- 146 → 358 tests (+212, +145%)
- 新增: LLMClient、SentimentAgent、PlannerAgent、LLMReport、TechnicalAgent 100%覆盖、回测确定性验证、全链路集成、PaperTrading 持久化、审计日志、邮件通知
