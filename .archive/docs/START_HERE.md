# 🚀 快速开始

**3步开始自动化模拟交易**

---

## 第1步：运行自动交易机器人 (5秒)

```bash
cd /Users/rowan/clawd/projects/ai-quant-agent
python3 examples/auto_trading_bot.py
```

**就这样！** 系统会自动：
- ✅ 检查4只股票的MA信号
- ✅ 自动买入/卖出
- ✅ 自动止损止盈
- ✅ 保存交易记录

---

## 第2步：设置每日自动运行 (1分钟)

```bash
# 编辑crontab
crontab -e

# 添加这行（每天15:30自动运行）
30 15 * * 1-5 cd /Users/rowan/clawd/projects/ai-quant-agent && python3 examples/auto_trading_bot.py >> logs/auto_trading.log 2>&1

# 保存退出
```

**现在每天收盘后自动运行，你什么都不用管！**

---

## 第3步：查看结果 (随时)

```bash
# 查看持仓和交易
cat data/auto_portfolio.json

# 查看日志
tail -20 logs/auto_trading.log
```

---

## 📊 V4配置速查

| 股票 | 权重 | MA | 初始买入 |
|------|------|----|---------| 
| 宁德时代 | 45% | 10/35 | 13,500元 |
| 立讯精密 | 30% | 10/35 | 9,000元 |
| 中国平安 | 15% | 8/25 | 4,500元 |
| 恒瑞医药 | 10% | 8/30 | 3,000元 |

---

## 📚 完整文档

- **README.md** - 项目概览
- **docs/FINAL_CONCLUSION.md** - 最终结论
- **docs/AUTO_TRADING_GUIDE.md** - 自动化交易完整指南
- **docs/QUICK_REFERENCE_CARD.md** - 每日参考卡

---

## 🎯 预期

**合理预期** (3-6个月):
- 夏普: 0.40-0.60
- 收益: 5-12%
- 胜率: 60-68%

**不是**:
- ❌ 夏普0.714（回测值）
- ❌ 收益19.6%（回测值）

---

## ⚠️ 重要

- 这是**模拟盘**，不是实盘
- 置信度48%，需要验证
- 可能亏损
- 3个月后评估

---

## 🆘 需要帮助？

### 常用命令

```bash
# 运行交易机器人
python3 examples/auto_trading_bot.py

# 查看持仓
cat data/auto_portfolio.json

# 更新数据
python3 examples/fetch_tushare_auto.py

# 查看日志
tail -f logs/auto_trading.log
```

### 遇到问题？

查看完整文档：
- `docs/AUTO_TRADING_GUIDE.md`
- `docs/FINAL_CONCLUSION.md`

---

**现在就开始！** 🚀

```bash
python3 examples/auto_trading_bot.py
```
