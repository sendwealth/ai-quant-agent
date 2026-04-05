# AI 量化交易系统 - 完整指南

> 🚀 企业级量化交易系统，支持多数据源、智能重试、数据验证和实时告警

**最后更新**: 2026-03-25 21:40
**版本**: v2.0.0
**状态**: ✅ 生产就绪

---

## 📋 目录

- [快速开始](#快速开始)
- [系统架构](#系统架构)
- [核心功能](#核心功能)
- [配置指南](#配置指南)
- [使用指南](#使用指南)
- [运维指南](#运维指南)
- [常见问题](#常见问题)

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
nano .env  # 填入你的配置
```

### 2. 配置数据源

```bash
# 编辑配置文件
nano config/data_sources.yaml
```

### 3. 启动系统

```bash
# 更新数据
python3 scripts/data_updater_robust.py

# 检查数据健康
python3 scripts/heartbeat_check_enhanced.py
```

### 4. 验证系统

```bash
# 测试邮件告警
python3 scripts/test_email_alert.py

# 检查配置
python3 scripts/config_manager.py check
```

---

## 🏗️ 系统架构

```
ai-quant-agent/
├── config/                    # 配置文件
│   ├── data_sources.yaml      # 数据源配置
│   ├── settings.py           # 系统设置
│   └── backups/              # 配置备份
├── scripts/                   # 核心脚本
│   ├── data_updater_robust.py     # 数据更新（主）
│   ├── heartbeat_check_enhanced.py # 心跳检查
│   ├── config_manager.py          # 配置管理
│   └── test_email_alert.py        # 邮件测试
├── agents/                    # 7-Agent 协作系统
├── core/                      # 核心模块
├── data/                      # 数据存储
│   ├── 300750_宁德时代.csv
│   ├── 002475_立讯精密.csv
│   └── ...
├── docs/                      # 文档
│   ├── README.md             # 本文档（主入口）
│   ├── OPERATION_GUIDE.md    # 操作指南
│   ├── DATA_UPDATER_GUIDE.md # 数据更新详解
│   └── CONFIG_MANAGEMENT.md  # 配置管理详解
├── .env                       # 环境变量（敏感信息）
└── requirements.txt           # 依赖列表
```

---

## 🎯 核心功能

### 1. 容错数据更新 ⭐⭐⭐⭐⭐

**特性**：
- ✅ 3个数据源（AkShare + Tushare + 新浪财经）
- ✅ 自动切换（AkShare → Tushare → 新浪财经）
- ✅ 指数退避重试（5秒 → 10秒 → 20秒）
- ✅ 8项数据验证（去重、缺失值、异常值等）
- ✅ 数据质量评分（0-100分）

**使用方法**：
```bash
# 更新所有股票
python3 scripts/data_updater_robust.py

# 输出示例
{
  "timestamp": "2026-03-25T21:31:29",
  "total": 4,
  "success": 4,
  "failed": 0,
  "details": [
    {
      "code": "300750",
      "name": "宁德时代",
      "status": "success",
      "rows": 484,
      "latest_date": "2026-03-25",
      "validation_issues": ["数据质量评分: 95/100"]
    }
  ]
}
```

### 2. 7-Agent 协作系统 ⭐⭐⭐⭐⭐

**团队构成**：
1. Portfolio Manager - 决策者
2. Buffett Analyst - 价值投资
3. Growth Analyst - 成长投资
4. Technical Analyst - 技术分析
5. Fundamentals Analyst - 基本面
6. Sentiment Analyst - 情绪分析
7. Risk Manager - 风险控制

**使用方法**：
```bash
# 运行分析师
python3 agents/buffett_analyst.py --stock 300750

# 汇总信号
python3 agents/risk_manager.py
```

### 3. 智能告警系统 ⭐⭐⭐⭐

**告警规则**：
- 🔴 失败率 > 50% → 邮件告警
- 🟡 数据过期 > 1天 → 自动更新
- 🟢 数据质量 < 60分 → 警告

**使用方法**：
```bash
# 测试邮件
python3 scripts/test_email_alert.py

# 检查系统健康
python3 scripts/heartbeat_check_enhanced.py
```

### 4. 配置管理 ⭐⭐⭐⭐

**功能**：
- ✅ 环境变量管理（.env）
- ✅ YAML 配置文件
- ✅ 自动备份机制
- ✅ 配置健康检查

**使用方法**：
```bash
# 备份配置
python3 scripts/config_manager.py backup

# 恢复配置
python3 scripts/config_manager.py restore

# 检查配置
python3 scripts/config_manager.py check
```

---

## ⚙️ 配置指南

### 1. 环境变量配置

**创建 .env 文件**：
```bash
# TuShare API Token（免费）
TUSHARE_TOKEN=YOUR_TUSHARE_TOKEN

# Email Alert Configuration
EMAIL_SMTP_SERVER=smtp.163.com
EMAIL_SMTP_PORT=465
EMAIL_SENDER=your_email@163.com
EMAIL_PASSWORD=your_auth_code  # 授权码，不是密码！
EMAIL_RECIPIENTS=alert@163.com
```

### 2. 数据源配置

**编辑 config/data_sources.yaml**：
```yaml
# 数据源优先级
akshare:
  enabled: true
  priority: 1  # 主要数据源

tushare:
  enabled: true
  priority: 2  # 备用数据源
  token: "${TUSHARE_TOKEN}"

sina_finance:
  enabled: true
  priority: 3  # 第三备用

# 重试配置（指数退避）
retry:
  max_attempts: 3
  delay_seconds: 5
  exponential_backoff: true

# 监控股票
monitored_stocks:
  - code: "300750"
    name: "宁德时代"
  - code: "002475"
    name: "立讯精密"
```

### 3. 邮件告警配置

**获取163邮箱授权码**：
1. 登录163邮箱
2. 设置 → POP3/SMTP/IMAP
3. 开启"IMAP/SMTP服务"
4. 获取"授权码"

**配置**：
```yaml
email_alert:
  enabled: true
  smtp_server: "smtp.163.com"
  smtp_port: 465
  sender: "${EMAIL_SENDER}"
  password: "${EMAIL_PASSWORD}"
  recipients:
    - "${EMAIL_RECIPIENTS}"
```

---

## 📖 使用指南

### 日常操作

#### 1. 更新数据（每天）

```bash
# 手动更新
python3 scripts/data_updater_robust.py

# 或添加到 crontab（每天18:30自动更新）
30 18 * * * cd ~/clawd/projects/ai-quant-agent && python3 scripts/data_updater_robust.py >> logs/data_update.log 2>&1
```

#### 2. 检查系统健康（每小时）

```bash
# 手动检查
python3 scripts/heartbeat_check_enhanced.py

# 或添加到 crontab（每小时检查）
0 * * * * cd ~/clawd/projects/ai-quant-agent && python3 scripts/heartbeat_check_enhanced.py >> logs/heartbeat.log 2>&1
```

#### 3. 备份配置（每周）

```bash
# 手动备份
python3 scripts/config_manager.py backup

# 或添加到 crontab（每周一备份）
0 0 * * 1 cd ~/clawd/projects/ai-quant-agent && python3 scripts/config_manager.py backup
```

### 定时任务配置

```bash
# 编辑 crontab
crontab -e

# 添加以下任务
# 数据更新（每天18:30）
30 18 * * * cd ~/clawd/projects/ai-quant-agent && python3 scripts/data_updater_robust.py >> logs/data_update.log 2>&1

# 心跳检查（每小时）
0 * * * * cd ~/clawd/projects/ai-quant-agent && python3 scripts/heartbeat_check_enhanced.py >> logs/heartbeat.log 2>&1

# 配置备份（每周一）
0 0 * * 1 cd ~/clawd/projects/ai-quant-agent && python3 scripts/config_manager.py backup
```

---

## 🔧 运维指南

### 日志查看

```bash
# 数据更新日志
tail -f logs/data_update.log

# 心跳检查日志
tail -f logs/heartbeat.log

# 应用日志
tail -f logs/app.log
```

### 数据健康检查

```bash
# 快速检查
python3 scripts/check_data_health.py

# 输出示例
{
  "status": "ok",
  "message": "数据新鲜（0天前）",
  "latest_date": "2026-03-25"
}
```

### 配置管理

```bash
# 检查配置健康度
python3 scripts/config_manager.py check

# 列出备份
python3 scripts/config_manager.py list

# 恢复配置
python3 scripts/config_manager.py restore
```

### 性能监控

```bash
# 查看系统资源
htop

# 查看磁盘使用
du -sh data/ logs/

# 查看数据量
wc -l data/*.csv
```

---

## ❓ 常见问题

### Q1: 数据更新失败怎么办？

**症状**：
```
❌ 宁德时代(300750) 更新失败
```

**解决方案**：
1. 检查网络连接
2. 检查数据源配置（Tushare token）
3. 查看详细日志
4. 系统会自动切换到备用数据源

```bash
# 查看日志
tail -100 logs/data_update.log

# 手动测试
python3 scripts/data_updater_robust.py
```

### Q2: 邮件告警发送失败？

**症状**：
```
❌ 邮件发送失败: SMTPAuthenticationError
```

**解决方案**：
1. 确认使用的是**授权码**，不是密码
2. 检查 SMTP 服务器地址
3. 检查邮箱是否开启 SMTP 服务

```bash
# 测试邮件配置
python3 scripts/test_email_alert.py
```

### Q3: 数据质量评分低？

**症状**：
```
"数据质量评分: 60/100"
```

**原因**：
- 数据缺失值过多
- 发现异常值
- 数据缺口较大

**解决方案**：
1. 查看详细验证报告
2. 检查数据源
3. 手动补充数据

### Q4: 如何添加新的监控股票？

**步骤**：
1. 编辑配置文件
   ```bash
   nano config/data_sources.yaml
   ```

2. 添加股票
   ```yaml
   monitored_stocks:
     - code: "000001"
       name: "平安银行"
   ```

3. 更新数据
   ```bash
   python3 scripts/data_updater_robust.py
   ```

### Q5: 配置文件丢失怎么办？

**解决方案**：
```bash
# 恢复最新备份
python3 scripts/config_manager.py restore

# 查看所有备份
python3 scripts/config_manager.py list
```

---

## 📚 文档索引

- [操作指南](OPERATION_GUIDE.md) - 详细的操作步骤
- [数据更新详解](DATA_UPDATER_GUIDE.md) - 数据更新系统详解
- [配置管理详解](CONFIG_MANAGEMENT.md) - 配置管理系统
- [全面回测报告](COMPREHENSIVE_BACKTEST_REPORT.md) - 策略回测结果
- [最优策略](FINAL_OPTIMAL_STRATEGY.md) - 最优投资策略
- [30%年化方案](PLAN_30_PERCENT.md) - 年化30%实施方案

---

## 📊 系统状态

**当前版本**: v2.0.0
**最后更新**: 2026-03-25 21:40
**系统状态**: ✅ 生产就绪

**核心指标**:
- 数据源: 3个（AkShare + Tushare + 新浪财经）
- 监控股票: 4只
- 数据质量: 95/100
- 成功率: 100%

**待办事项**:
- [ ] 添加更多数据源（如：东方财富）
- [ ] 优化数据验证规则
- [ ] 添加机器学习模型
- [ ] 实现自动化交易

---

## 🤝 贡献指南

欢迎贡献代码和建议！

1. Fork 项目
2. 创建特性分支
3. 提交改动
4. 推送到分支
5. 创建 Pull Request

---

## 📄 许可证

MIT License

---

**维护者**: Nano (AI Assistant)
**联系方式**: sendwealth@163.com
**GitHub**: https://github.com/your-repo/ai-quant-agent

---

_最后更新: 2026-03-25 21:40_
