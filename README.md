# AI Quant Agent v3.0

LLM 增强的多 Agent 协作 A 股量化交易系统。规则引擎 + LLM 双引擎，支持自然语言交互。

## 架构

```
数据源 (Tushare/efinance/AkShare/BaoStock)
    ↓  [RateLimiter]  [并发获取]  [修复-再验证]
DataService (4源降级 + 财务合并 + 缓存 + 校验 + 离线模式 + 文件锁)
    ↓
┌─────────────┬─────────────┬─────────────┐
│ Fundamental  │  Technical  │  Sentiment  │  ← 分析师 Agent
│   Agent      │   Agent     │   Agent     │    (可配置阈值)
└──────┬──────┴──────┬──────┴──────┬──────┘
       └─────────────┼─────────────┘
               Risk Agent          ← 信号共识 + 动态仓位 + LLM 风险解读
                     ↓
             Execution Agent        ← 模拟交易 + 止损止盈 + 审计日志
                     ↓
             BacktestEngine         ← Sharpe/Sortino/MaxDD + 确定性验证

         ┌── LLM 层 (LangChain + OpenAI/智谱 GLM) ──┐
         │  SentimentAgent (新闻情感)                  │
         │  PlannerAgent   (指令解析)                  │
         │  LLMReportGenerator (报告生成)              │
         │  RiskAgent.interpret_risk (风险解读)        │
         └────────────────────────────────────────────┘
```

Agent 通过 **Orchestrator** 编排，使用结构化日志记录，返回标准化的 `AgentResult`。
共享的 `Portfolio` + `CommissionModel` 确保回测与执行佣金计算一致。
支持 **智能选股** (--screen)、**邮件通知** (交易信号、每日报告、异常告警) 和 **自然语言分析** (--prompt)。

## 核心特性

- **LLM 双引擎** — 规则引擎为基础，LangChain ChatOpenAI 提供情感分析、报告生成、风险解读
- **双 Provider** — OpenAI / 智谱 GLM 自动切换，`openai_api_key` 优先，`zhipu_api_key` 备选
- **自然语言交互** — `--prompt "分析宁德时代的买入机会"` 智能解析股票代码和分析范围
- **真实数据** — 4 源降级链 (Tushare/efinance/AkShare/BaoStock)，财务多源合并，FinancialSnapshot schema 验证
- **离线模式** — `--preload` 预下载 + `--offline` 纯缓存分析，无需网络
- **多 Agent 协作** — 基本面 + 技术面 + 情感 → 风控共识 → 执行，Orchestrator 统一编排
- **回测引擎** — Sharpe/Sortino/Calmar/MaxDD/Alpha/Beta 全指标 + 确定性验证
- **风控系统** — 动态仓位、止损止盈、信号共识机制，参数通过 Settings 配置
- **数据安全** — 文件锁 + 原子写入、速率限制、数据修复、输入校验、token 防泄露
- **持久化交易** — PaperTradingService 支持进程重启恢复
- **审计追踪** — 追加式 JSONL 日志，记录每笔交易决策
- **可配置** — Agent 评分阈值外部化 (YAML)，无代码调优
- **高性能** — 并发数据获取 (ThreadPoolExecutor)、向量化指标计算
- **邮件通知** — 交易信号实时推送、每日报告汇总、系统异常告警 (SMTP)

## Quick Start

### Step 1: 安装

```bash
git clone <repo-url> && cd ai-quant-agent
uv sync
```

### Step 2: 配置

```bash
cp .env.example .env
```

编辑 `.env`，**无需任何 token 也能运行**（efinance 免费）：

```bash
# 可选 — 有 Tushare token 可获取更详细的财务报表
QUANT_TUSHARE_TOKEN=你的token    # 注册: https://tushare.pro

# 可选 — 没有 LLM key 也能跑，跳过情感分析和报告生成
QUANT_ZHIPU_API_KEY=xxx         # 智谱 (免费额度: https://open.bigmodel.cn)
# 或
QUANT_OPENAI_API_KEY=sk-xxx     # OpenAI

# 可选 — 收邮件通知
QUANT_EMAIL_ENABLED=true
QUANT_EMAIL_SENDER=xxx@163.com
QUANT_EMAIL_PASSWORD=授权码      # 163邮箱: 设置 → POP3/SMTP → 开启 → 获取授权码
QUANT_EMAIL_RECIPIENTS=xxx@163.com
```

### Step 3: 运行

```bash
# 单股分析
uv run python -m quant_agent.main --stock 300750

# 用中文提问 (需要 LLM key)
uv run python -m quant_agent.main --prompt "贵州茅台现在能买吗"

# 智能选股 (从 ~179 只候选池中筛选 Top 20)
uv run python -m quant_agent.main --screen

# 选股 + 深度分析 Top 5
uv run python -m quant_agent.main --screen --screen-analyze --top 5

# 预下载数据 (用于离线分析)
uv run python -m quant_agent.main --preload --stocks 300750,002475

# 离线分析 (仅使用本地缓存，不发 API 请求)
uv run python -m quant_agent.main --stock 300750 --offline

# 批量分析 4 只股票 + 邮件汇总
uv run python -m quant_agent.main --daily-report
```

### 股票代码格式

| 市场 | 前缀 | 示例 |
|---|---|---|
| 上海主板 | 60xxxx | 600519 (贵州茅台) |
| 深圳主板 | 00xxxx | 000858 (五粮液) |
| 创业板 | 30xxxx | 300750 (宁德时代) |
| 北交所 | 8xxxxx | 830799 |

### 输出解读

```
OK   fundamental: BUY  (85%)   ← 基本面: ROE 22%, PE 18, 低负债
OK   technical:   BUY  (90%)   ← 技术面: RSI 超卖, MACD 金叉
OK   sentiment:   BUY  (75%)   ← 情感面: 政策利好 (需要 LLM key)

Risk: BUY  仓位 16.7%          ← 风控共识: 3/3 投票买入
  止损: 368.00 | 止盈: 440.00 / 480.00

BUY executed: 300750 200股 @ 400.00
  Total equity: 99995.00
```

### 收到邮件的条件

风控信号为 **BUY** 或 **SELL** 时自动发送 HTML 邮件，包含 Agent 投票表和交易详情。
HOLD 不发邮件（避免信息轰炸）。

### 每日定时运行

```bash
# Linux/Mac crontab — 每个交易日 9:15 自动分析
crontab -e
# 添加:
15 9 * * 1-5 cd /path/to/ai-quant-agent && uv run python -m quant_agent.main --daily-report >> /tmp/quant.log 2>&1
```

### 跑测试

```bash
uv run pytest tests/ -v                           # 全量 498 测试
uv run pytest tests/unit/ -v                      # 472 单元测试
uv run pytest tests/test_integration.py -v        # 26 集成测试
uv run pytest tests/unit/test_risk_enhanced.py -v # 风控增强测试 (T+1/熔断/LLM解读)
uv run pytest tests/unit/test_screener.py -v      # 选股模块测试 (51)
uv run pytest tests/unit/test_efinance_source.py -v # efinance 数据源测试 (25)
```

## 项目结构

```
src/quant_agent/
├── main.py                 # CLI 入口 (--stock / --prompt / --daily-report)
├── orchestrator.py         # Orchestrator 编排器 + AnalysisReport
├── config.py               # Pydantic Settings (QUANT_ 前缀)
├── portfolio.py            # 统一 Portfolio + CommissionModel
├── audit.py                # 追加式审计日志 (JSONL)
├── thresholds.py           # Agent 评分阈值加载器
├── llm/                    # LLM 层 (LangChain)
│   ├── client.py           # LLMClient (OpenAI/智谱自动切换)
│   ├── prompts.py          # 所有 prompt 模板 (情感/规划/报告/风险)
│   └── report.py           # LLMReportGenerator (综合报告生成)
├── data/                   # 数据层
│   ├── sources/            # Tushare + efinance + AkShare + BaoStock 适配器
│   ├── rate_limiter.py     # Token bucket 速率限制
│   ├── service.py          # 统一入口 (4源降级 + 财务合并 + 离线)
│   ├── normalizer.py       # 标准化
│   ├── validator.py        # 校验 + repair_price_data
│   └── store.py            # Parquet 存储 (文件锁 + 原子写入)
├── strategy/indicators.py  # 指标库 (RSI/MACD/EMA/ATR/...)
├── backtest/engine.py      # 回测引擎
├── events/bus.py           # 事件总线 (保留用于未来异步)
├── screener/               # 智能选股模块
│   ├── filters.py          # 预筛 (ST/退市/低价/低流动性)
│   ├── scorers.py          # 多维评分 (技术+动量+流动性+基本面)
│   └── engine.py           # ScreeningEngine 协调器
├── agents/                 # Agent 框架
│   ├── base.py             # BaseAgent + AgentResult + 结构化日志
│   ├── fundamental.py      # 基本面 (可配置阈值)
│   ├── technical.py        # 技术面 (可配置阈值)
│   ├── sentiment.py        # 情感分析 (LLM 新闻情感 → AgentResult)
│   ├── planner.py          # 指令解析 (自然语言 → ExecutionPlan)
│   ├── risk.py             # 风控 (共识 + T+1 + 熔断 + 仓位 + LLM 风险解读)
│   └── execution.py        # 执行 (审计日志)
├── execution/
│   └── paper_trading.py    # 持久化模拟交易
├── notification/
│   └── email.py            # 邮件通知 (SMTP)
└── observability/          # 监控
config/
└── agent_thresholds.yaml   # Agent 评分阈值配置
tests/
├── unit/                   # 472 单元测试
└── test_integration.py     # 26 集成测试
```

## 技术栈

| 组件 | 选型 |
|------|------|
| 语言 | Python 3.10+ |
| 包管理 | uv |
| LLM | LangChain (OpenAI / 智谱 GLM 双 Provider) |
| 数据源 | efinance, tushare, akshare, baostock |
| 配置 | pydantic-settings + YAML |
| 存储 | Parquet (文件锁) → PostgreSQL/TimescaleDB |
| 测试 | pytest + pytest-cov (498 tests) |
| 并发 | concurrent.futures.ThreadPoolExecutor |

## LLM 使用场景

| 场景 | 模块 | 输入 → 输出 |
|---|---|---|
| 分析报告生成 | `llm/report.py` | AgentResult[] → Markdown 投资分析报告 |
| 情感分析 | `agents/sentiment.py` | 新闻数据 → AgentResult (BUY/SELL/HOLD) |
| 智能指令解析 | `agents/planner.py` | 自然语言 → ExecutionPlan (stock_code, days, focus) |
| 风险解读 | `agents/risk.py` | 风控结果 → 自然语言风险解读 |

LLM 不可用时自动跳过增强功能，规则引擎正常运行。

## 指标库

RSI, MACD, EMA, SMA, Bollinger Bands, ATR, ADX, CCI, OBV, Williams %R, ROC, Momentum, 成交量分析

## 回测指标

总收益, 年化收益, 最大回撤, Sharpe Ratio, Sortino Ratio, Calmar Ratio, 胜率, 盈亏比, Beta, Alpha

## 风控参数

| 参数 | 默认值 | 环境变量 |
|------|--------|----------|
| 单只仓位上限 | 20% | QUANT_MAX_POSITION_PCT |
| 组合最大风险敞口 | 80% | QUANT_MAX_PORTFOLIO_RISK |
| 止损线 | -8% | QUANT_DEFAULT_STOP_LOSS |
| 止盈线 | +10% / +20% | QUANT_DEFAULT_TAKE_PROFIT_1/2 |
| 日亏损熔断 | -3% | QUANT_MAX_DAILY_LOSS_PCT |
| T+1 限制 | 自动 | 内置 (A 股规则) |
| 佣金 | 万三 (最低5元) | QUANT_COMMISSION_RATE |
| 印花税 | 千一 (仅卖出) | QUANT_STAMP_TAX_RATE |
| 回测整手 | 100 股 | 内置 (A 股规则) |

## 环境变量

完整列表见 `.env.example`，常用配置：

```bash
# 数据源
QUANT_TUSHARE_TOKEN=xxx          # Tushare API token (可选, efinance 免费可用)

# 离线模式 (可选)
QUANT_OFFLINE_MODE=true          # 禁止 API 请求, 仅读缓存
QUANT_PRELOAD_STOCKS=300750,002475,601318  # 预下载股票列表
QUANT_DATA_CACHE_TTL=1800        # 缓存有效期 (秒)

# LLM (二选一，OpenAI 优先)
QUANT_OPENAI_API_KEY=xxx         # OpenAI API key
QUANT_ZHIPU_API_KEY=xxx          # 智谱 API key (GLM-4)
QUANT_OPENAI_MODEL=gpt-4o        # 模型名称 (可选)
QUANT_OPENAI_BASE_URL=https://api.openai.com/v1  # 可覆盖

# 通用
QUANT_DATA_DIR=./data            # 数据目录 (可选)
QUANT_FETCH_MAX_WORKERS=5        # 并发获取线程数 (可选)

# 邮件通知 (可选)
QUANT_EMAIL_ENABLED=true
QUANT_EMAIL_SMTP_SERVER=smtp.163.com
QUANT_EMAIL_SENDER=your@163.com
QUANT_EMAIL_PASSWORD=smtp_auth_code
QUANT_EMAIL_RECIPIENTS=to@example.com
```

## License

MIT

## 免责声明

本项目仅供学习研究，不构成投资建议。股市有风险，投资需谨慎。
