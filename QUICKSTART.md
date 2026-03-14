# 量化系统快速启动指南

> 5分钟上手 V5 量化策略

---

## ⚡ 快速开始

### 1. 每日分析（1分钟）
```bash
cd ~/clawd/projects/ai-quant-agent
python3 run.py
```

**输出**: `data/daily_analysis_results.json`

---

### 2. 查看信号（30秒）
```bash
cat data/daily_analysis_results.json | python3 -m json.tool
```

**示例输出**:
```json
{
  "signals": {
    "300750": {
      "name": "宁德时代",
      "signal": "BUY",
      "price": 395.5
    }
  }
}
```

---

### 3. 模拟交易（1分钟）
```bash
python3 scripts/paper_trading.py
```

**功能**:
- 自动执行BUY/SELL信号
- 检查止损/止盈
- 保存账户状态

---

### 4. 风险监控（30秒）
```bash
python3 scripts/risk_monitor.py
```

**检查项**:
- 持仓风险（止损/止盈）
- 组合风险（仓位比例）
- 市场风险（情绪判断）

---

### 5. 综合监控（1分钟）
```bash
python3 scripts/quant_monitor.py
```

**输出**: 数据健康 + 策略信号 + 账户状态 + 风险监控

---

## 📊 核心脚本

| 脚本 | 用途 | 频率 |
|------|------|------|
| `run.py` | 每日分析 | 每日8:50 |
| `paper_trading.py` | 模拟交易 | 有信号时 |
| `risk_monitor.py` | 风险监控 | 每30分钟 |
| `quant_monitor.py` | 综合监控 | 每30分钟 |
| `heartbeat_check.py` | 心跳检查 | 每30分钟 |
| `check_data_health.py` | 数据检查 | 每30分钟 |

---

## 🎯 策略参数

### V5策略配置
```yaml
止损: -6%
止盈1: +10% 卖50%
止盈2: +20% 清仓

单只仓位: ≤30%
总仓位: ≤80%
现金比例: ≥20%
```

### 股票池
- 宁德时代 (300750)
- 立讯精密 (002475)
- 中国平安 (601318)
- 恒瑞医药 (600276)

---

## 📈 性能指标

### V5回测结果
| 股票 | 夏普 | 收益 | 胜率 |
|------|------|------|------|
| 宁德时代 | 0.765 | +21.50% | 59.3% |
| 立讯精密 | 0.777 | +23.16% | 75.0% |
| 中国平安 | 0.612 | +14.17% | 71.4% |
| 恒瑞医药 | 0.428 | +10.29% | 67.9% |
| **平均** | **0.712** | **+17.28%** | **68.4%** |

---

## 🛡️ 风控规则

### 止损机制
1. **单笔止损**: -6%自动卖出
2. **日止损**: -3%停止交易
3. **周止损**: -5%暂停策略
4. **月止损**: -10%全面复盘

### 仓位控制
- 单只股票: ≤30%
- 总仓位: ≤80%
- 现金比例: ≥20%

### 熔断机制
- 连续3天亏损 → 降低仓位50%
- 连续5天亏损 → 暂停交易
- 单日亏损-5% → 立即清仓

---

## 🚨 告警规则

### 数据告警
- 数据过期>7天 → 🔴 立即通知
- 数据过期>3天 → 🟡 记录日志
- 数据正常 → 🟢 静默

### 风险告警
- 触发止损 → 🔴 立即通知
- 仓位过高 → 🟡 记录日志
- 无异常 → 🟢 静默

### 账户告警
- 亏损>5% → 🟡 记录日志
- 亏损>10% → 🔴 立即通知
- 盈利正常 → 🟢 静默

---

## 📁 重要文件

### 配置
- `config/strategy_v5.yaml` - V5策略配置

### 数据
- `data/real_*.csv` - 股票历史数据
- `data/daily_analysis_results.json` - 每日分析
- `data/paper_trading_state.json` - 账户状态
- `data/risk_report.json` - 风险报告

### 文档
- `docs/EVOLUTION_PLAN.md` - 进化计划
- `docs/V5_VALIDATION_REPORT.md` - 验证报告
- `docs/SYSTEM_ARCHITECTURE.md` - 系统架构

---

## 🔄 定时任务

```bash
# 查看定时任务
crontab -l | grep quant

# 1. 每日分析（每个交易日 8:50）
50 8 * * 1-5 cd ~/clawd/projects/ai-quant-agent && python3 run.py

# 2. 数据更新（每周日 18:00）
0 18 * * 0 cd ~/clawd/projects/ai-quant-agent && python3 examples/fetch_tushare_auto.py

# 3. 健康检查（每天 9:00）
0 9 * * * cd ~/clawd/projects/ai-quant-agent && python3 utils/health_check.py
```

---

## 💡 使用技巧

### 1. 快速检查系统状态
```bash
python3 scripts/heartbeat_check.py
```

### 2. 查看持仓盈亏
```bash
python3 scripts/risk_monitor.py | grep "盈亏"
```

### 3. 手动更新数据
```bash
python3 examples/fetch_tushare_auto.py
```

### 4. 查看回测结果
```bash
python3 examples/comprehensive_backtest.py
```

---

## 🎯 下一步

### Week 1 (3/13-3/20)
- [ ] 每日监控模拟账户
- [ ] 记录交易日志
- [ ] 观察策略表现

### Week 2 (3/20-3/27)
- [ ] 评估测试结果
- [ ] 调整策略参数
- [ ] 决定是否实盘

### Week 3-4 (3/27-4/10)
- [ ] 优化策略
- [ ] 开发自动交易
- [ ] 准备扩大规模

---

## 📞 问题排查

### 数据过期
```bash
# 检查数据健康
python3 scripts/check_data_health.py

# 手动更新
python3 examples/fetch_tushare_auto.py
```

### 策略信号异常
```bash
# 重新运行分析
python3 run.py

# 查看详细日志
cat logs/daily_$(date +%Y%m%d).log
```

### 账户状态异常
```bash
# 查看账户
cat data/paper_trading_state.json | python3 -m json.tool

# 重置账户（慎用）
rm data/paper_trading_state.json
python3 scripts/paper_trading.py
```

---

**版本**: v3.0
**更新**: 2026-03-13
**状态**: ✅ 生产就绪
