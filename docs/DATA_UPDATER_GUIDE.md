# 容错数据更新系统使用指南

## 功能特性

1. **多数据源切换** - AkShare（主要）→ Tushare（备用）
2. **智能重试机制** - 失败后自动重试3次
3. **邮件告警** - 数据过期或更新失败自动通知
4. **YAML配置** - 灵活配置数据源、股票、告警阈值

---

## 快速开始

### 1. 基本使用（AkShare免费）

```bash
# 更新所有监控股票数据
python3 scripts/data_updater_robust.py
```

**输出示例**:
```json
{
  "timestamp": "2026-03-25T20:25:35",
  "total": 4,
  "success": 4,
  "failed": 0,
  "details": [...]
}
```

### 2. 配置备用数据源（Tushare）

**获取 Tushare Token**:
1. 访问 https://tushare.pro/register
2. 注册账户
3. 获取 API Token

**配置方法**:
```yaml
# config/data_sources.yaml

tushare:
  enabled: true
  priority: 2
  token: "你的token"  # 填入你的 Tushare token
  timeout: 30
```

### 3. 配置邮件告警

**获取邮箱授权码**（以163邮箱为例）:
1. 登录163邮箱
2. 设置 → POP3/SMTP/IMAP
3. 开启"IMAP/SMTP服务"
4. 获取"授权码"

**配置方法**:
```yaml
# config/data_sources.yaml

email_alert:
  enabled: true
  smtp_server: "smtp.163.com"
  smtp_port: 465
  sender: "your_email@163.com"
  password: "你的授权码"  # 注意：是授权码，不是密码
  recipients:
    - "alert@163.com"
```

---

## 配置文件详解

### config/data_sources.yaml

```yaml
# 数据源配置
akshare:
  enabled: true
  priority: 1  # 优先级，数字越小越优先
  timeout: 30

tushare:
  enabled: false  # 启用需要配置 token
  priority: 2
  token: ""  # 填入你的 token
  timeout: 30

# 邮件告警配置
email_alert:
  enabled: false  # 启用需要配置授权码
  smtp_server: "smtp.163.com"
  smtp_port: 465
  sender: ""
  password: ""  # 授权码
  recipients: []

# 重试配置
retry:
  max_attempts: 3  # 每个数据源的最大重试次数
  delay_seconds: 5  # 重试间隔
  exponential_backoff: false  # 是否使用指数退避

# 监控股票列表
monitored_stocks:
  - code: "300750"
    name: "宁德时代"
  - code: "002475"
    name: "立讯精密"
  - code: "601318"
    name: "中国平安"
  - code: "600276"
    name: "恒瑞医药"

# 告警阈值
alert_thresholds:
  data_stale_days: 1  # 数据过期超过多少天发送告警
  failure_rate: 0.5  # 失败率超过多少发送严重告警
```

---

## 集成到定时任务

### 方式1: Cron Job

```bash
# 编辑 crontab
crontab -e

# 添加定时任务（每天18:30更新数据）
30 18 * * * cd /path/to/ai-quant-agent && python3 scripts/data_updater_robust.py >> logs/data_update.log 2>&1
```

### 方式2: OpenClaw Heartbeat

编辑 `HEARTBEAT.md`:
```markdown
## 6️⃣ 量化数据更新
- **脚本**: `~/clawd/projects/ai-quant-agent/scripts/data_updater_robust.py`
- **动作**:
  - 运行容错数据更新
  - 自动重试失败的数据源
  - 发送邮件告警（如果配置）
- **频率**: 每次心跳检查（约30分钟）
- **通知规则**:
  - 🔴 失败率>50% → 立即通知
  - 🟡 数据过期>1天 → 发送告警
  - 🟢 全部成功 → 静默
```

---

## 常见问题

### Q1: AkShare 连接失败怎么办？

**原因**: AkShare 依赖东方财富等第三方数据源，网络不稳定

**解决方案**:
1. **启用 Tushare 备用源**（推荐）
   ```yaml
   tushare:
     enabled: true
     token: "你的token"
   ```

2. **增加重试次数**
   ```yaml
   retry:
     max_attempts: 5
     delay_seconds: 10
   ```

### Q2: 邮件告警发送失败？

**检查清单**:
- [ ] `enabled: true`
- [ ] 使用**授权码**而不是密码
- [ ] SMTP 服务器地址正确
- [ ] 端口正确（163: 465）

### Q3: 数据源都不可用？

**临时方案**:
```bash
# 手动下载历史数据
python3 examples/fetch_tushare_auto.py
```

**长期方案**:
- 配置多个备用数据源
- 使用本地数据库缓存

---

## 性能指标

| 指标 | 值 |
|------|-----|
| 单次更新时间 | ~5秒（4只股票） |
| 重试延迟 | 5秒 |
| 最大重试次数 | 3次 |
| 告警延迟 | <1分钟 |

---

## 监控与日志

### 日志位置
```
logs/app.log  # 应用日志
logs/data_update.log  # 更新日志（cron）
```

### 检查数据健康度
```bash
python3 scripts/check_data_health.py
```

**输出**:
```json
{
  "status": "ok",
  "message": "数据新鲜（0天前）",
  "latest_date": "2026-03-25"
}
```

---

## 更新记录

- **2026-03-25**: 创建容错数据更新系统
  - 多数据源支持（AkShare + Tushare）
  - 3次重试机制
  - 邮件告警
  - YAML配置

---

## 下一步优化

- [ ] 添加更多数据源（如：新浪财经、东方财富）
- [ ] 实现指数退避重试
- [ ] 添加数据验证（去重、缺失值检查）
- [ ] 支持增量更新（只更新最新日期）
- [ ] 添加 Webhook 告警（Slack、钉钉）
