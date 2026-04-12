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
- 数据源: efinance (免费行情+财务), tushare (财务报表), akshare (行情+财务), baostock (免费行情)
- 配置: pydantic-settings + dotenv (AliasChoices) + YAML
- 存储: Parquet (当前) → PostgreSQL/TimescaleDB (后续)
- 测试: pytest + pytest-cov (498 passed)

## 目录结构

```
src/quant_agent/
├── main.py                 # CLI 入口 (--stock / --prompt / --screen / --daily-report)
├── orchestrator.py         # Orchestrator 编排器 + AnalysisReport + screen_and_analyze()
├── config.py               # Pydantic Settings (QUANT_ 前缀, AliasChoices, 含 initial_capital 校验)
├── portfolio.py            # 统一 Portfolio + CommissionModel + Position + Trade
├── audit.py                # 追加式审计日志 (JSONL)
├── thresholds.py           # Agent 评分阈值加载器 (YAML + 默认值, 含 screener 配置)
├── llm/                    # LLM 层
│   ├── client.py           # LLMClient (LangChain ChatOpenAI, OpenAI/智谱自动切换)
│   ├── prompts.py          # 所有 prompt 模板 (情感/规划/报告/风险)
│   └── report.py           # LLMReportGenerator (综合分析报告生成)
├── data/                   # 数据层
│   ├── sources/base.py     # DataSource ABC + FinancialSnapshot (schema 验证)
│   ├── sources/tushare.py  # Tushare 适配器 (财务报表, Settings 传参)
│   ├── sources/akshare.py  # AkShare 适配器 (行情 + 财务 + 速率限制)
│   ├── sources/baostock.py # BaoStock 适配器 (免费行情, 重试 + 速率限制)
│   ├── sources/efinance.py # efinance 适配器 (免费行情 + 财务, 东方财富 API)
│   ├── rate_limiter.py     # Token bucket 速率限制器
│   ├── normalizer.py       # 列名标准化 (缺失列抛异常)
│   ├── validator.py        # 数据质量校验 (null/停牌/非正价格) + repair_price_data
│   ├── validators.py       # 股票代码输入验证 (沪深创北 6 位代码)
│   ├── store.py            # Parquet 存储 (文件锁 + 原子写入)
│   └── service.py          # DataService 统一入口 (4源降级 + 财务合并 + 离线 + 修复)
│   ├── store.py            # Parquet 存储 (文件锁 + 原子写入)
│   └── service.py          # DataService 统一入口 (4源降级 + 财务合并 + 离线 + 修复)
├── scripts/                 # 独立脚本
│   └── preload.py          # 预下载数据 (价格 + 财务)
├── strategy/indicators.py  # 统一指标库 (RSI/MACD/EMA/ATR/ADX/布林带/OBV 向量化)
├── backtest/engine.py      # 回测引擎 (Sharpe/Sortino/Calmar/MaxDD/Alpha/Beta)
├── events/bus.py           # EventBus 事件总线 (保留用于未来异步编排)
├── screener/               # 智能选股模块
│   ├── filters.py          # PreFilter 预筛 (ST/退市/低价/低流动性)
│   ├── scorers.py          # 多维评分 (技术~40 + 动量~35 + 流动性~25 + 基本面~10)
│   └── engine.py           # ScreeningEngine 协调器 + StockScore + ScreeningResult
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
├── unit/                   # 472 单元测试
│   ├── test_agents.py      # Agent + EventBus 测试
│   ├── test_backtest.py    # 回测引擎 (含已知答案确定性测试 + Profit Factor 回归)
│   ├── test_config_validation.py # 配置校验 (含 initial_capital 边界)
│   ├── test_data_pipeline.py # 数据管道 (降级/缓存/验证/修复/离线)
│   ├── test_data_service.py # DataService 测试
│   ├── test_efinance_source.py # efinance 数据源 (25测试: ABC/映射/转换/重试/限速)
│   ├── test_execution.py   # ExecutionAgent + AuditLogger 测试
│   ├── test_indicators.py  # 技术指标测试
│   ├── test_email_notification.py # 邮件通知测试
│   ├── test_paper_trading.py # PaperTradingService 测试
│   ├── test_technical_agent.py # TechnicalAgent (100% 覆盖)
│   ├── test_llm_client.py  # LLMClient + 单例测试
│   ├── test_llm_report.py  # LLMReportGenerator 测试
│   ├── test_sentiment_agent.py # SentimentAgent 测试
│   ├── test_planner_agent.py  # PlannerAgent 测试
│   ├── test_rate_limiter.py # RateLimiter Token bucket 测试
│   ├── test_risk_enhanced.py # T+1/日熔断/组合限制 + interpret_risk LLM 路径
│   └── test_screener.py    # Screener 选股 (PreFilter/Scorers/Engine 51测试)
└── test_integration.py     # 26 集成测试 (全链路 + 并发安全)
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
uv run python -m quant_agent.main --screen              # 智能选股 (DEFAULT_POOL ~179只)
uv run python -m quant_agent.main --screen --top 10     # 选股 Top 10
uv run python -m quant_agent.main --screen --screen-analyze --top 5  # 选股+深度分析
uv run python -m quant_agent.main --multi               # 多股 (并发获取)
uv run python -m quant_agent.main --daily-report         # 批量分析 + 每日邮件报告
uv run python -m quant_agent.main --preload --stocks 300750,002475  # 预下载数据
uv run python -m quant_agent.main --stock 300750 --offline # 离线分析 (仅用缓存)
uv run python scripts/real_backtest.py                   # 真实数据回测
uv run pytest tests/ -v                                  # 运行全量测试 (498)
uv run pytest tests/unit/ -v                             # 仅单元测试 (472)
uv run pytest tests/test_integration.py -v               # 仅集成测试 (26)
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
- `QUANT_OFFLINE_MODE` — 离线模式，不发 API 请求 (默认 false)
- `QUANT_PRELOAD_STOCKS` — 预下载股票列表 (逗号分隔)
- `QUANT_DATA_CACHE_TTL` — 缓存有效期秒数 (默认 1800)

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

## 已完成的改进 (2026-04-11, 更新)

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
- 146 → 358 → 474 → 498 tests (+352, +241%)
- 新增: LLMClient、SentimentAgent、PlannerAgent、LLMReport、TechnicalAgent 100%覆盖、回测确定性验证、全链路集成、PaperTrading 持久化、审计日志、邮件通知、efinance 数据源

### P5 — 选股模块
- screener/filters.py: PreFilter 两步预筛 (名称→价格/流动性, 零网络调用)
- screener/scorers.py: 四维评分 (技术~40 + 动量~35 + 流动性~25 + 基本面~10, _value字段排除)
- screener/engine.py: ScreeningEngine 协调器 + DEFAULT_POOL 179只降级池
- orchestrator.py: screen_and_analyze() 选股后深度分析
- main.py: --screen / --top / --full-scan / --screen-analyze CLI 入口
- thresholds.py: _SCREENER_DEFAULTS + agent_thresholds.yaml screener 配置节

### P6 — 测试加固
- RateLimiter 单元测试: 17 测试 (Token bucket 算法 + 并发安全)
- Screener 测试: 51 测试 (PreFilter/Scorers/Engine/Thresholds + _value 排除验证)
- RiskAgent.interpret_risk: 6 测试 (LLM mock, prompt 验证, 无 LLM 返回 None)
- Settings.initial_capital: validator + 6 边界测试 (零/负拒绝)
- Profit Factor 回归: sum(wins)/abs(sum(losses)) 精确验证
- 并发安全回归: 5 线程并发 analyze() 无状态损坏

### P7 — 数据层加固 (4源 + 离线 + 修复)
- EfinanceSource: 免费 efinance 适配器 (东方财富 API, 无 token, 120/min 限速)
- AkShare 财务快照: get_financial_snapshot 接入 (中文列名→FinancialSnapshot)
- AkShare 重试修复: 只重试瞬态错误 (ConnectionError/TimeoutError), 非瞬态立即失败
- BaoStock 加固: 指数退避重试 + 速率限制 (100/min) + login 生命周期修复
- 4 源降级链: Tushare → efinance → AkShare → BaoStock
- 财务多源合并: 遍历所有源 → 合并空字段 → 缓存降级
- repair_price_data: 前向/后向填充 + 线性插值 + 修复-再验证流程
- 预下载脚本: scripts/preload.py (--stocks/--from-file/--price-only/--financial-only)
- 离线模式: --offline 禁止 API 调用，纯读本地 parquet 缓存
- data_cache_ttl 修复: 硬编码 4h → 读取配置 data_cache_ttl (默认 1800s)
- efinance 测试: 25 测试 (ABC 合规/列映射/代码转换/重试策略/速率限制/可用性)
