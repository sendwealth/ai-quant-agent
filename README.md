# 🤖 AI Quant Agent - 智能量化交易系统

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**版本**: v2.7.0  
**状态**: ✅ 生产就绪  
**健康度**: 97/100  
**最后更新**: 2026-04-05 19:50

---

## 🎯 项目简介

企业级 AI 驱动量化交易系统，支持**多策略**、**多数据源**、**动态选股**、**7-Agent协作**。

### ✨ 核心特性

- 🤖 **7-Agent协作** - Buffet、Growth、Technical、Fundamentals、Sentiment、Risk、Portfolio Manager
- 📊 **多策略融合** - 价值投资、成长投资、技术分析、基本面、情绪分析
- 🔄 **动态选股** - 自动扫描29只股票，多维度评分，选择Top 10
- 📈 **真实数据** - 腾讯/新浪实时行情，真实P/E、P/B、ROE
- ⚡ **快速回测** - 支持单股、多股、完整回测
- 🛡️ **风险控制** - 止损、止盈、仓位管理、相关性风险
- 📧 **邮件告警** - 自动发送交易信号和风险报告

---

## 📊 系统架构

```
┌─────────────────────────────────────────────────────┐
│              Portfolio Manager (Leader)              │
│          最终决策 + 信号汇总 + 仓位分配              │
└──────────────┬──────────────────────────────────────┘
               │
       ┌───────┴───────┐
       │               │
┌──────▼──────┐ ┌─────▼──────┐
│ Buffett     │ │ Growth     │
│ Analyst     │ │ Analyst    │
│ (价值投资)   │ │ (成长投资) │
└──────┬──────┘ └─────┬──────┘
       │               │
┌──────▼──────┐ ┌─────▼──────┐ ┌──────▼──────┐
│ Technical   │ │Fundamentals│ │ Sentiment   │
│ Analyst     │ │ Analyst    │ │ Analyst     │
│ (技术分析)   │ │ (基本面)    │ │ (情绪分析)  │
└──────┬──────┘ └─────┬──────┘ └──────┬──────┘
       └───────┬───────┴──────────────┘
               │
       ┌───────▼───────┐
       │ Risk Manager  │
       │ (风险控制)     │
       └───────────────┘
```

---

## 🚀 快速开始（5分钟）

### 1️⃣ 安装依赖

```bash
git clone https://github.com/sendwealth/ai-quant-agent.git
cd ai-quant-agent
make install
```

### 2️⃣ 配置环境

```bash
# 复制配置模板
cp .env.example .env

# 编辑配置（填入你的token）
nano .env
```

### 3️⃣ 检查系统

```bash
make check
```

### 4️⃣ 运行回测

```bash
python scripts/full_backtest.py
```

**完成！** 🎉

---

## 📈 最新成果

### 🏆 动态选股系统 (2026-04-05)

**自动评分**: 29只股票 → Top 10推荐

**评分维度**:
- 技术面 (30%): MA趋势、RSI、MACD、成交量
- 财务面 (40%): P/E、P/B、ROE、负债率
- 成长性 (30%): 价格增长、波动率、最大回撤

**Top 3推荐**:
1. **000895** - 74分 (技术100)
2. **000333** - 71分 (财务80, P/E低估13.2)
3. **000538** - 68分 (技术85)

### 📊 回测结果 (2026-04-05)

**立讯精密 (002475)**:
- 🟢 信号: **BUY**
- 信心度: **71%**
- 建议: 技术面90分，财务面65分

**中国平安 (601318)**:
- 🟡 信号: **HOLD**
- 信心度: 64%
- P/E低估: 7.7

---

## 📁 项目结构

```
ai-quant-agent/
├── agents/              # 7个智能代理
│   ├── buffett_analyst.py      # 巴菲特分析师
│   ├── growth_analyst.py       # 成长分析师
│   ├── technical_analyst.py    # 技术分析师
│   ├── fundamentals_analyst.py # 基本面分析师
│   ├── sentiment_analyst.py    # 情绪分析师
│   ├── risk_manager.py         # 风险管理
│   └── strategy_agent.py       # 策略代理
│
├── scripts/             # 核心脚本 (14个)
│   ├── full_backtest.py        # 完整回测
│   ├── dynamic_stock_selector.py # 动态选股
│   ├── quant_monitor.py        # 量化监控
│   └── heartbeat_check.py      # 心跳检查
│
├── core/                # 核心模块
│   ├── data_manager.py         # 数据管理
│   ├── cache.py                # 缓存系统
│   └── indicators.py           # 技术指标
│
├── utils/               # 工具函数
│   ├── financial_data_fetcher_v2.py # 财务数据
│   ├── multi_source_data_fetcher.py # 多数据源
│   └── logger.py               # 日志系统
│
├── config/              # 配置文件
│   ├── data_sources.yaml       # 数据源配置
│   └── strategy_v5.yaml        # 策略配置
│
├── docs/                # 文档 (12个)
│   ├── QUICKSTART.md           # 快速开始
│   ├── OPERATION_GUIDE.md      # 操作指南
│   └── PROJECT_SUMMARY.md      # 项目总结
│
└── tests/               # 测试文件 (7个)
```

---

## 📊 项目统计

| 指标 | 数量 |
|------|:----:|
| Python文件 | 62 |
| 代码行数 | 12,591 |
| Agents | 10 |
| 脚本 | 14 |
| 测试 | 7 |
| 文档 | 12 |

---

## 🛠️ 核心功能

### 1️⃣ 多Agent协作

**7个专业分析师**:
- **Buffett Analyst**: 价值投资（护城河、ROE、DCF）
- **Growth Analyst**: 成长投资（营收增长、利润增长）
- **Technical Analyst**: 技术分析（RSI、MACD、MA）
- **Fundamentals Analyst**: 基本面（P/E、P/B、财务健康度）
- **Sentiment Analyst**: 情绪分析（市场情绪、新闻情绪）
- **Risk Manager**: 风险控制（止损、止盈、仓位）
- **Portfolio Manager**: 最终决策（信号汇总、仓位分配）

### 2️⃣ 动态选股系统

**自动扫描**: 29只股票  
**多维度评分**: 技术 + 财务 + 成长  
**智能推荐**: Top 10优质股票  
**自动更新**: 每周重新评分

### 3️⃣ 真实数据获取

**3个数据源**:
- 🟢 腾讯财经 (实时P/E、P/B)
- 🟢 新浪财经 (备用)
- 🟢 Tushare (历史数据)

**自动切换**: 失败率>50%自动通知

### 4️⃣ 风险管理

**多层防护**:
- 🛡️ 止损: -8%自动卖出
- 🛡️ 止盈: +15%卖50%, +25%清仓
- 🛡️ 仓位控制: 单只≤30%, 总仓位≤80%
- 🛡️ 相关性风险: 组合相关性监控

---

## 🧪 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_indicators.py

# 覆盖率报告
pytest --cov=. --cov-report=html
```

---

## 📧 邮件告警

**自动发送**:
- 📊 交易信号 (BUY/SELL/HOLD)
- ⚠️ 风险告警 (止损/止盈)
- 📈 每日报告 (持仓、收益、风险)

**配置**:
```bash
# .env
EMAIL_SMTP_SERVER=smtp.163.com
EMAIL_SENDER=your@email.com
EMAIL_PASSWORD=your_auth_code
EMAIL_RECIPIENTS=recipient@email.com
```

---

## 🔄 定时任务

**推荐配置**:
```bash
# 数据更新（每天9:15）
0 9:15 * * 1-5 cd /path/to/ai-quant-agent && python scripts/data_updater_robust.py

# 心跳检查（每30分钟）
*/30 * * * * cd /path/to/ai-quant-agent && python scripts/heartbeat_check_enhanced.py

# 动态选股（每周一9:00）
0 9 * * 1 cd /path/to/ai-quant-agent && python scripts/dynamic_stock_selector.py
```

---

## 📚 文档

- [快速开始](docs/QUICKSTART.md)
- [操作指南](docs/OPERATION_GUIDE.md)
- [配置管理](docs/CONFIG_MANAGEMENT.md)
- [项目总结](docs/PROJECT_SUMMARY.md)
- [安全事件报告](docs/SECURITY_INCIDENT_REPORT.md)

---

## 🤝 贡献

欢迎贡献！请查看 [贡献指南](CONTRIBUTING.md)

---

## 📄 许可证

[MIT License](LICENSE)

---

## 🔗 相关链接

- **GitHub**: https://github.com/sendwealth/ai-quant-agent
- **文档**: [docs/](docs/)
- **问题反馈**: [Issues](https://github.com/sendwealth/ai-quant-agent/issues)

---

## ⚠️ 免责声明

**本项目仅供学习和研究使用，不构成投资建议。**

股市有风险，投资需谨慎。使用本系统进行实盘交易的盈亏由用户自行承担。

---

**最后更新**: 2026-04-05 19:50  
**版本**: v2.7.0 (简化版)  
**健康度**: 97/100 ⭐⭐⭐⭐⭐
