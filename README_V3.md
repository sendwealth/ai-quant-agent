# AI量化交易系统 v3.0

> 从-47%失败到稳定盈利的进化之路

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Strategy V5](https://img.shields.io/badge/strategy-V5-green.svg)](config/strategy_v5.yaml)
[![Sharpe 0.712](https://img.shields.io/badge/sharpe-0.712-brightgreen.svg)](docs/V5_VALIDATION_REPORT.md)
[![Return +17.28%](https://img.shields.io/badge/return-+17.28%25-brightgreen.svg)](docs/V5_VALIDATION_REPORT.md)

---

## 🎯 项目简介

基于技术指标的量化交易系统，采用模块化设计，集成风险监控和自动交易。

**当前策略**: V5 Enhanced
- 夏普: **0.712** (+812% vs V4)
- 收益: **+17.28%** (+64.73% vs V4)
- 胜率: **68.4%** (+12.8% vs V4)

**核心特性**:
- ✅ 多股票组合管理
- ✅ 策略信号生成
- ✅ 模拟交易执行
- ✅ 实时风险监控
- ✅ 数据健康检查
- ✅ 心跳系统集成

---

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行分析
```bash
python3 run.py
```

### 3. 查看信号
```bash
cat data/daily_analysis_results.json
```

### 4. 模拟交易
```bash
python3 scripts/paper_trading.py
```

### 5. 风险监控
```bash
python3 scripts/risk_monitor.py
```

---

## 📊 系统架构

```
数据层 (TuShare)
  ↓ 每日更新
策略层 (V5 Enhanced)
  ↓ 信号生成
执行层 (模拟交易)
  ↓ 风险控制
监控层 (心跳系统)
  ↓ 状态通知
```

详细文档: [SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md)

---

## 🛡️ 风控体系

### 三层风控
1. **数据层**: 健康检查、异常检测
2. **策略层**: 参数优化、多维度验证
3. **执行层**: 止损/止盈、仓位限制

### 风险规则
- 止损: -6%
- 止盈1: +10%卖50%
- 止盈2: +20%清仓
- 单只仓位: ≤30%
- 总仓位: ≤80%

---

## 📈 性能对比

### V4 vs V5

| 维度 | V4 | V5 | 改进 |
|------|----|----|------|
| 夏普 | -0.10 | 0.712 | +812% |
| 收益 | -47.45% | +17.28% | +64.73% |
| 胜率 | 55.6% | 68.4% | +12.8% |

详细报告: [V5_VALIDATION_REPORT.md](docs/V5_VALIDATION_REPORT.md)

---

## 📁 项目结构

```
ai-quant-agent/
├── config/              # 配置文件
├── data/                # 数据
├── scripts/             # 核心脚本
│   ├── paper_trading.py
│   ├── risk_monitor.py
│   ├── quant_monitor.py
│   └── heartbeat_check.py
├── examples/            # 示例脚本
├── docs/                # 文档
├── logs/                # 日志
└── run.py               # 快速运行
```

---

## 🎯 进化路线

### Phase 1 (已完成) ✅
- V5策略验证
- 模拟交易系统
- 风险监控系统
- 心跳集成

### Phase 2 (进行中) ⏳
- 小资金实盘测试
- 自动交易执行
- 实时监控面板

### Phase 3 (计划中) 📋
- 集中持仓策略
- 市场环境识别
- 多策略组合

详细计划: [EVOLUTION_PLAN.md](docs/EVOLUTION_PLAN.md)

---

## 📊 监控指标

### 系统指标
- 数据健康: 实时检查
- 系统稳定性: >99%
- 信号准确率: >70%

### 策略指标
- 夏普比率: >0.7
- 收益率: >15%/年
- 胜率: >65%
- 最大回撤: <10%

---

## 🔧 核心脚本

| 脚本 | 用途 | 频率 |
|------|------|------|
| `run.py` | 每日分析 | 每日8:50 |
| `paper_trading.py` | 模拟交易 | 有信号时 |
| `risk_monitor.py` | 风险监控 | 每30分钟 |
| `quant_monitor.py` | 综合监控 | 每30分钟 |
| `heartbeat_check.py` | 心跳检查 | 每30分钟 |

快速指南: [QUICKSTART.md](QUICKSTART.md)

---

## 📞 问题排查

### 数据过期
```bash
python3 scripts/check_data_health.py
python3 examples/fetch_tushare_auto.py
```

### 策略异常
```bash
python3 run.py
cat logs/daily_$(date +%Y%m%d).log
```

### 账户异常
```bash
cat data/paper_trading_state.json
python3 scripts/paper_trading.py
```

---

## 📚 文档

- [快速启动](QUICKSTART.md) - 5分钟上手
- [系统架构](docs/SYSTEM_ARCHITECTURE.md) - 架构设计
- [进化计划](docs/EVOLUTION_PLAN.md) - 路线图
- [验证报告](docs/V5_VALIDATION_REPORT.md) - 回测结果
- [API文档](docs/API.md) - 接口说明

---

## 🎯 目标

### 短期（1个月）
- 月收益率: >3%
- 夏普比率: >0.8
- 最大回撤: <8%

### 中期（3个月）
- 月收益率: >5%
- 夏普比率: >1.0
- 最大回撤: <6%

### 长期（6个月）
- 月收益率: >8%
- 夏普比率: >1.2
- 最大回撤: <5%

---

## 📝 更新日志

### v3.0 (2026-03-13)
- ✅ 完成V5策略验证
- ✅ 部署模拟交易系统
- ✅ 集成风险监控
- ✅ 修复数据健康bug
- ✅ 集成心跳系统

### v2.0 (2026-03-11)
- ✅ V5参数优化
- ✅ 策略对比验证
- ✅ 部署脚本

### v1.0 (2026-03-04)
- ✅ V4策略开发
- ✅ 回测系统
- ✅ 基础框架

---

**系统版本**: v3.0
**更新时间**: 2026-03-13
**维护者**: Nano
**状态**: ✅ 生产就绪
