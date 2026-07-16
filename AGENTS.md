# AI Quant Agent v3.0 — 开发指南

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
- 数据源: tushare (财务报表), efinance (东方财富 API, 无 token), akshare (行情+财务), baostock (免费行情), westock (腾讯自选股, npx CLI, 无 token), sample (确定性合成数据, 离线/测试用)
- 配置: pydantic-settings + dotenv (AliasChoices) + YAML
- 存储: Parquet (当前) → PostgreSQL/TimescaleDB (后续)
- 测试: pytest + pytest-cov (**670 用例, 668 passed / 2 deselected**)
- CLI: Typer (`quant-agent` console 脚本 → `quant_agent.cli:app`)
- Web: 标准库 `http.server` (零额外依赖)

## 目录结构

```
src/quant_agent/
├── cli.py                  # 【真正 CLI 入口】Typer 子命令: analyze / screen / reports / config / preload
├── main.py                 # 【已弃用】旧 argparse 入口, run_pipeline() 包装器 + __main__ 上帝函数; 勿再扩展
├── orchestrator.py         # Orchestrator 编排器 + AnalysisReport + screen_and_analyze() / analyze_batch()
├── config.py               # Pydantic Settings (QUANT_ 前缀, AliasChoices, 含 initial_capital 校验)
├── portfolio.py            # 统一 Portfolio + CommissionModel + Position + Trade
├── audit.py                # 追加式审计日志 (JSONL)
├── thresholds.py           # Agent 评分阈值加载器 (YAML + 默认值, 含 screener 配置)
├── compat.py               # 向后兼容垫片 (report.to_dict 等)
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
│   ├── sources/westock.py  # WeStock 适配器 (腾讯自选股, npx CLI, 免费无 token)
│   ├── sources/sample.py   # 样例数据源 (确定性合成数据, 离线/测试)
│   ├── gate.py             # 数据可信度评估 (evaluate_trust)
│   ├── rate_limiter.py     # Token bucket 速率限制器
│   ├── normalizer.py       # 列名标准化 (缺失列抛异常)
│   ├── validator.py        # 数据质量校验 (null/停牌/非正价格) + repair_price_data
│   ├── validators.py       # 股票代码输入验证 (沪深创北 6 位代码)
│   ├── smoke.py            # 数据冒烟测试 (连通性/可用性探针)
│   ├── samples/            # 内置演示样例行情 (随 wheel 发布)
│   ├── store.py            # Parquet 存储 (文件锁 + 原子写入)
│   └── service.py          # DataService 统一入口 (多源降级 + 财务合并 + 离线 + 修复)
├── strategy/indicators.py  # 统一指标库 (RSI/MACD/EMA/ATR/ADX/布林带/OBV/CCI/随机/威廉 向量化, Wilders 平滑)
├── backtest/               # 回测引擎
│   ├── engine.py           # 回测引擎 (Sharpe/Sortino/Calmar/MaxDD/Alpha/Beta/Profit Factor)
│   └── manifest.py         # 回测可复现性 manifest (uv lock 指纹 + 数据指纹)
├── events/bus.py           # EventBus 事件总线 (保留用于未来异步编排)
├── screener/               # 智能选股模块
│   ├── filters.py          # PreFilter 预筛 (ST/退市/低价/低流动性)
│   ├── scorers.py          # 多维评分 (技术~40 + 动量~35 + 流动性~25 + 基本面~10)
│   ├── engine.py           # ScreeningEngine 协调器 + StockScore + ScreeningResult
│   └── stock_names.py      # 股票代码/名称检索 (search_stocks)
├── agents/
│   ├── base.py             # BaseAgent ABC + AgentResult + 结构化日志
│   ├── fundamental.py      # 基本面 Agent (真实财务数据, 可配置阈值)
│   ├── technical.py        # 技术分析 Agent (指标 + 信号, 可配置阈值)
│   ├── sentiment.py        # 情感分析 Agent (LLM 新闻情感 → AgentResult)
│   ├── planner.py          # 指令解析 Agent (自然语言 → ExecutionPlan)
│   ├── risk.py             # 风控 Agent (共识 + 仓位 + LLM 风险解读)
│   └── execution.py        # 执行 Agent (模拟交易 + 止损止盈 + 审计日志)
├── execution/
│   ├── paper_trading.py    # PaperTradingService (持久化模拟交易)
│   └── store.py            # 模拟交易存储 (JSON 原子写入)
├── notification/
│   └── email.py            # EmailNotifier (交易信号/每日报告/异常告警)
├── observability/          # 可观测性
│   ├── metrics.py          # MetricsCollector + HealthChecker
│   ├── alerting.py         # AlertManager (异常告警)
│   └── health.py           # build_health_report (健康检查聚合)
├── reporting/              # 报告渲染
│   ├── renderer.py         # render_markdown (Markdown 报告)
│   ├── chart.py            # plot_price_chart (价格走势图 PNG)
│   └── history.py          # 报告历史 (list/load/save/latest_for_stock)
├── trading/
│   └── service.py          # TradingService (交易编排入口)
├── web/                    # Web 服务 (标准库 http.server, 零依赖)
│   ├── server.py           # JSON API + 静态前端托管; 业务逻辑抽成 *_core 纯函数便于测试
│   └── static/             # 前端 (index.html / app.js / styles.css)
├── scripts/
│   └── preload.py          # 预下载数据 (价格 + 财务)
config/
└── agent_thresholds.yaml   # Agent 评分阈值外部配置
tests/
├── unit/                   # 单元测试 (约 644 用例)
│   ├── test_agents.py      # Agent + EventBus 测试
│   ├── test_backtest.py    # 回测引擎 (含已知答案确定性测试 + Profit Factor 回归)
│   ├── test_backtest_credibility.py # 回测可复现性 / manifest 测试
│   ├── test_backtest_manifest.py    # manifest 指纹测试
│   ├── test_config_validation.py    # 配置校验 (含 initial_capital 边界)
│   ├── test_data_pipeline.py        # 数据管道 (降级/缓存/验证/修复/离线)
│   ├── test_data_service.py         # DataService 测试
│   ├── test_data_trust.py           # gate.evaluate_trust 测试
│   ├── test_data_smoke.py           # 数据冒烟测试
│   ├── test_efinance_source.py      # efinance 数据源 (ABC/映射/转换/重试/限速)
│   ├── test_westock_source.py       # westock 数据源测试
│   ├── test_execution.py            # ExecutionAgent + AuditLogger 测试
│   ├── test_indicators.py           # 技术指标测试
│   ├── test_indicators_edge_cases.py# 指标边界用例 (恒价/除零等)
│   ├── test_email_notification.py    # 邮件通知测试
│   ├── test_paper_trading.py        # PaperTradingService 测试
│   ├── test_technical_agent.py      # TechnicalAgent (49 测试, 高覆盖)
│   ├── test_llm_client.py           # LLMClient + 单例测试
│   ├── test_llm_report.py           # LLMReportGenerator 测试
│   ├── test_sentiment_agent.py      # SentimentAgent 测试
│   ├── test_planner_agent.py        # PlannerAgent 测试
│   ├── test_rate_limiter.py         # RateLimiter Token bucket 测试
│   ├── test_risk_enhanced.py        # T+1/日熔断/组合限制 + interpret_risk LLM 路径
│   ├── test_screener.py             # Screener 选股 (PreFilter/Scorers/Engine)
│   ├── test_web.py / test_web_e2e.py / test_web_health.py # Web 服务测试
│   ├── test_alerting.py / test_compat.py / test_license_check.py / test_board_lot.py / test_renderer.py # 其他
└── test_integration.py     # 集成测试 (全链路 + 并发安全)
docs/architecture-v3.md     # 架构设计文档
docs/multi-agent-plan.md    # 多智能体完善计划
archive/                    # v2.6 旧代码备份
```

> 注意: `memory/` 与 `mcp/` 为旧文档中的"待实现"项,当前代码中**未实现**,已从结构树移除。

## 核心原则

1. **真实数据** — 所有分析基于真实财务报表，禁止伪造/估算核心指标
2. **LLM 增强** — 规则引擎为基础，LLM 提供情感分析、报告生成、风险解读
3. **结构化日志** — Agent 通过 `_log_action()` 输出结构化日志
4. **Fail Safe** — 数据源故障自动降级，LLM 不可用时跳过增强功能
5. **测试先行** — 新模块必须有单元测试 + 集成测试，LLM 测试全部 mock
6. **降级捕获是设计正确** — `DataService` 与各 `DataSource.fetch_*` 的宽泛 `except` 是有意保留的韧性设计(捕获所有异常才能 fallback 到下一个源),**不要**在无视上下文的情况下收窄为特定异常类;需收窄的是各源内部的"重试"逻辑(只重试瞬态错误),而非降级链本身

## 运行

```bash
# 推荐: Typer CLI (console 脚本 quant-agent, 见 pyproject.toml [project.scripts])
quant-agent analyze 600519                              # 单股分析 (规则引擎, 可加 --chart/--execute)
quant-agent analyze 600519 --prompt "分析买入机会"      # 自然语言 (需 LLM)
quant-agent screen --top 10                             # 智能选股 Top 10
quant-agent screen --top 5 --deep                       # 选股 + 深度分析
quant-agent preload --stocks 300750,002475              # 预下载数据
quant-agent config init                                 # 交互式配置向导 (生成 .env)

# 等价 uv 运行 (CLI 入口为 quant_agent.cli)
uv run quant-agent analyze 600519
uv run python -m quant_agent.cli analyze 600519

# 已弃用 (会发 DeprecationWarning): python -m quant_agent.main
uv run python -m quant_agent.main --stock 300750

# 测试
uv run pytest tests/ -v                                  # 运行全量测试 (670)
uv run pytest tests/unit/ -v                             # 仅单元测试
uv run pytest tests/test_integration.py -v               # 仅集成测试

# Web 服务 (标准库, 零依赖)
uv run python -m quant_agent.web.server
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
- 使用 `uv run` 而非裸 `python`
- stock_code 必须通过 `validate_stock_code()` 校验 (60/00/30/8x 前缀, 6 位数字)
- LLM 使用 LangChain ChatOpenAI，structured_output 做结构化解析

## 已完成的改进 (2026-04-11 基线, 2026-07-16 AGENTS.md 校正)

### P0 — 核心架构
- Orchestrator 类: 提取上帝函数到独立编排器 (`orchestrator.py`)
- EventBus 重构: emit_event → _log_action() 结构化日志
- config → Agent: 风险参数通过 Settings 对象注入 RiskAgent/ExecutionAgent
- tushare.py: 移除残留 os.getenv，统一 Settings 传参
- Typer CLI (`cli.py`) 成为正式入口, 注册 `quant-agent` console 脚本; 旧 `main.py` 标记 `@deprecated`

### P1 — 数据层加固
- FinancialSnapshot: 构造时 schema 验证 + validate() 方法
- DataStore: 文件锁 (fcntl.flock) + 原子写入 (temp→os.replace)
- BaoStock: 上下文管理器 (__enter__/__exit__) 管理生命周期
- RateLimiter: Token bucket 速率限制 (Tushare 200/min, AkShare 60/min)

### P2 — 测试 & 质量
- 合并 Portfolio: 统一 portfolio.py，消除回测/执行双实现
- 回测确定性测试: 已知答案测试 (手工计算精确验证)
- TechnicalAgent 高覆盖测试
- 集成测试: 全链路测试 (Orchestrator 端到端) + 并发安全回归

### P3 — 生产就绪
- PaperTradingService: 持久化模拟交易 (JSON 原子写入，进程重启恢复)
- AuditLogger: 追加式 JSONL 审计日志 (月度归档，线程安全)
- 阈值外部化: config/agent_thresholds.yaml (TechnicalAgent + FundamentalAgent)
- 并发获取: ThreadPoolExecutor 批量数据请求
- 邮件通知: EmailNotifier (交易信号/每日报告/异常告警, SMTP_SSL)
- 可观测性: observability/ (metrics + alerting + health)

### P4 — LLM 智能化
- LLMClient: LangChain ChatOpenAI 封装，OpenAI/智谱双 provider 自动切换
- SentimentAgent: LLM 新闻情感分析，参与 RiskAgent 共识投票
- PlannerAgent: 自然语言 → ExecutionPlan，--prompt CLI 入口
- LLMReportGenerator: 综合多 Agent 结果生成 Markdown 投资分析报告
- RiskAgent 风险解读: LLM 生成自然语言风险说明和应对建议
- 依赖: langchain + langchain-openai + langgraph

### P5 — 选股模块
- screener/filters.py: PreFilter 两步预筛 (名称→价格/流动性, 零网络调用)
- screener/scorers.py: 四维评分 (技术~40 + 动量~35 + 流动性~25 + 基本面~10, _value字段排除)
- screener/engine.py: ScreeningEngine 协调器 + DEFAULT_POOL 降级池
- orchestrator.py: screen_and_analyze() 选股后深度分析
- thresholds.py: _SCREENER_DEFAULTS + agent_thresholds.yaml screener 配置节

### P6 — 测试加固
- RateLimiter 单元测试: Token bucket 算法 + 并发安全
- Screener 测试: PreFilter/Scorers/Engine/Thresholds + _value 排除验证
- RiskAgent.interpret_risk: LLM mock, prompt 验证, 无 LLM 返回 None
- Settings.initial_capital: validator + 边界测试 (零/负拒绝)
- Profit Factor 回归: sum(wins)/abs(sum(losses)) 精确验证
- 并发安全回归: 多线程并发 analyze() 无状态损坏

### P7 — 数据层加固 (多源 + 离线 + 修复)
- EfinanceSource: 免费 efinance 适配器 (东方财富 API, 无 token, 120/min 限速)
- AkShare 财务快照: get_financial_snapshot 接入 (中文列名→FinancialSnapshot)
- AkShare 重试修复: 只重试瞬态错误 (ConnectionError/TimeoutError), 非瞬态立即失败
- BaoStock 加固: 指数退避重试 + 速率限制 (100/min) + login 生命周期修复
- 多源降级链: Tushare → efinance → AkShare → BaoStock (+ sample 离线兜底)
- 财务多源合并: 遍历所有源 → 合并空字段 → 缓存降级
- repair_price_data: 前向/后向填充 + 线性插值 + 修复-再验证流程
- 预下载脚本: scripts/preload.py (--stocks/--from-file/--price-only/--financial-only)
- 离线模式: --offline 禁止 API 调用，纯读本地 parquet 缓存
- data_cache_ttl 修复: 硬编码 4h → 读取配置 data_cache_ttl (默认 1800s)
- 数据可信度: data/gate.py evaluate_trust + 回测 manifest 指纹 (可复现性)

### 2026-07-16 — AGENTS.md 校正
- 测试数更正: 498 → **670** (668 passed / 2 deselected)
- 目录结构补全真实存在的模块: `cli.py`(真正入口)、`web/`、`reporting/`、`trading/`、`observability/`(alerting/health)、`data/gate.py`、`data/smoke.py`、`data/samples/`、`sources/sample.py`、`compat.py`
- 移除不存在的 `memory/`、`mcp/` 待实现项
- 删除 data 目录树中 store.py/service.py 的重复行
- 运行命令改为以 `quant-agent` Typer CLI 为主, 标注 `python -m quant_agent.main` 已弃用
- 新增核心原则: 降级链宽泛 `except` 为有意设计, 不应盲目收窄
- `main.py`: 修复重复 docstring, 在 run_pipeline / __main__ 加 DeprecationWarning
