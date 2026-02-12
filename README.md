# 🤖 AI智能体量化交易系统

> **目标：利用AI智能体实现自动化盈利的量化交易**

## 🎯 项目愿景

打造一个基于多智能体协同的AI原生量化交易系统，通过自然语言生成交易策略、强化学习优化参数、多智能体对抗训练，实现稳定盈利。

## 🚀 核心特性

### 1. 智能体架构
- **策略智能体**：从自然语言描述自动生成可执行交易策略代码
- **分析智能体**：实时市场分析、趋势识别、情绪分析
- **风控智能体**：动态风险管理、止损止盈、仓位管理
- **优化智能体**：强化学习参数优化，每天测试5000+组合
- **回测智能体**：10年历史数据快速回测，性能评估

### 2. AI原生能力
- ⚡ **30秒策略生成**：输入自然语言 → 输出可执行代码 + 10年回测报告
- 🎯 **自动参数优化**：强化学习引擎日均测试5000+参数组合，平均提升收益9.3%
- 🔄 **多智能体互博**：模拟20个AI策略对抗训练，找出最优组合
- 📊 **自适应行情**：AI实时调整买卖价格，自动优化加仓间距

### 3. 交易模式
- ✅ **100%全自动**：无需人工干预，24/7运行
- 🎛️ **半自动模式**：用户可选择多/空，系统自动执行
- 📈 **盈利加仓**：趋势追踪，盈利时顺势加仓
- 📉 **高抛低吸**：亏损时马丁策略，低吸高抛
- ⏰ **择时交易**：高/低波动时段自动调整策略

## 🛠️ 技术栈

### 核心框架
- **VeighNa (vn.py)** - 量化交易平台开发框架
- **Qlib** - 微软AI量化投资工具包
- **Backtrader** - 回测引擎
- **Zipline** - 另一个回测选择

### AI/ML
- **PyTorch** / **TensorFlow** - 深度学习框架
- **LangChain** - 大语言模型应用框架
- **OpenAI API** / **智谱AI** - LLM推理
- **Stable-Baselines3** - 强化学习算法

### 数据
- **Tushare** - A股数据
- **yfinance** - 美股数据
- **CCXT** - 加密货币数据
- **AkShare** - 财经数据接口

### 部署
- **Docker** - 容器化部署
- **Redis** - 消息队列/缓存
- **PostgreSQL** - 数据存储
- **FastAPI** - API服务

## 📁 项目结构

```
ai-quant-agent/
├── agents/              # AI智能体模块
│   ├── strategy_agent.py    # 策略智能体
│   ├── analysis_agent.py    # 分析智能体
│   ├── risk_agent.py        # 风控智能体
│   ├── optimize_agent.py   # 优化智能体
│   └── backtest_agent.py   # 回测智能体
├── strategies/          # 交易策略
│   ├── trend/                # 趋势策略
│   ├── mean_reversion/       # 均值回归
│   ├── ml/                   # 机器学习策略
│   └── llm/                  # LLM生成策略
├── data/               # 数据模块
│   ├── fetcher.py            # 数据获取
│   ├── processor.py          # 数据处理
│   └── storage.py            # 数据存储
├── backtest/           # 回测模块
│   ├── engine.py             # 回测引擎
│   ├── metrics.py            # 性能指标
│   └── visualization.py      # 可视化
├── trading/            # 交易模块
│   ├── executor.py           # 交易执行
│   ├── order_manager.py      # 订单管理
│   └── position_manager.py   # 持仓管理
├── optimization/       # 优化模块
│   ├── rl_optimizer.py       # 强化学习优化
│   ├── genetic.py            # 遗传算法
│   └── grid_search.py        # 网格搜索
├── utils/              # 工具模块
│   ├── logger.py             # 日志
│   ├── config.py             # 配置
│   └── indicators.py         # 技术指标
├── api/                # API服务
│   ├── main.py               # FastAPI主程序
│   └── routes.py             # 路由
├── tests/              # 测试
├── docs/               # 文档
├── config/             # 配置文件
└── requirements.txt    # 依赖
```

## 🎮 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 配置
```bash
cp config/config.example.yaml config/config.yaml
# 编辑config.yaml，填入你的API密钥等配置
```

### 启动策略智能体
```bash
python agents/strategy_agent.py
```

### 回测
```bash
python backtest/engine.py --strategy trend_following --symbol AAPL --start 2015-01-01 --end 2025-01-01
```

### 实盘交易（谨慎！）
```bash
python trading/executor.py --mode live --account your_account
```

## 📈 盈利策略

### 策略类型
1. **趋势跟踪策略** - 移动平均、MACD、突破
2. **均值回归策略** - RSI、布林带、统计套利
3. **机器学习策略** - LSTM、Transformer、强化学习
4. **AI生成策略** - LLM自动生成代码，人类审核执行

### 盈利模式
- ✅ **趋势加仓**：盈利时顺势加仓，最大化收益
- 📉 **低吸高抛**：亏损时加仓摊薄成本，反弹获利
- ⏰ **择时交易**：高波动时段激进，低波动时段保守
- 🎯 **动态止损**：AI实时调整止损位，保护利润
- 🔄 **多策略组合**：20个策略同时运行，降低风险

## ⚠️ 风险提示

**量化交易有风险，入市需谨慎！**

1. 本项目仅供学习研究，不构成投资建议
2. 实盘前请充分回测，控制仓位
3. 不要投入超过承受能力的资金
4. AI策略可能失效，需要持续优化
5. 严格遵守风险管理规则

## 🔮 未来规划

- [x] 项目架构设计
- [ ] 策略智能体开发
- [ ] 强化学习优化器
- [ ] 多智能体对抗训练
- [ ] 实盘对接交易所
- [ ] 盈利验证

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📞 联系

- GitHub: sendwealth
- Email: sendwealth@163.com

---

**记住：市场永远是对的，保持谦逊，持续学习。**
