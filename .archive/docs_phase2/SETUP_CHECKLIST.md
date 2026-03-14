# ✅ 模拟盘设置检查清单

**检查时间**: 2026-03-04 20:42

---

## ✅ 已完成的设置

### 1. 核心系统 ✅
- [x] 自动交易机器人已创建 (`auto_trading_bot.py`)
- [x] 首次运行成功
- [x] 持仓文件已创建 (`data/auto_portfolio.json`)
- [x] 交易规则已配置（V4最优配置）

### 2. 监控工具 ✅
- [x] 状态监控工具 (`status_monitor.py`)
- [x] 每日运行脚本 (`daily_run.sh`)
- [x] 清理脚本 (`cleanup.sh`, `cleanup_repository.sh`)

### 3. 文档系统 ✅
- [x] 快速开始指南 (`START_HERE.md`)
- [x] 自动化交易指南 (`docs/AUTO_TRADING_GUIDE.md`)
- [x] 快速参考卡 (`docs/QUICK_REFERENCE_CARD.md`)
- [x] 模拟盘启动文档 (`SIMULATION_STARTED.md`)

### 4. 数据系统 ✅
- [x] 股票数据已加载（4只股票，1213天）
- [x] TuShare Token已配置
- [x] 数据获取脚本已就绪 (`fetch_tushare_auto.py`)

---

## ⚠️ 还需要手动设置的（重要！）

### 1. 每日自动运行 ❌ 未设置

**这是最重要的！** 需要设置cron任务，让系统每天自动运行。

#### 设置步骤：

```bash
# 1. 打开终端，编辑crontab
crontab -e

# 2. 添加以下行（每天15:30自动运行）
30 15 * * 1-5 cd /Users/rowan/clawd/projects/ai-quant-agent && ./daily_run.sh >> logs/auto_trading.log 2>&1

# 3. 保存退出（vim: 按 ESC，输入 :wq，回车）

# 4. 验证是否添加成功
crontab -l
```

**说明**:
- `30 15 * * 1-5` = 每周一到周五，15:30运行
- `./daily_run.sh` = 运行每日脚本
- `>> logs/auto_trading.log 2>&1` = 保存日志

---

### 2. 每周更新数据 ⚠️ 建议设置

虽然不是必须，但建议每周更新一次股票数据。

#### 设置步骤：

```bash
# 1. 编辑crontab
crontab -e

# 2. 添加以下行（每周日18:00更新数据）
0 18 * * 0 cd /Users/rowan/clawd/projects/ai-quant-agent && python3 examples/fetch_tushare_auto.py >> logs/data_update.log 2>&1

# 3. 保存退出
```

**说明**:
- `0 18 * * 0` = 每周日18:00运行
- 更新最新的股票数据

---

### 3. 通知提醒 ⚠️ 可选

如果你想在有交易时收到通知，可以设置。

#### 方式1: 邮件通知

在 `daily_run.sh` 中添加：
```bash
# 在文件末尾添加
if [ -f "data/auto_portfolio.json" ]; then
    # 检查是否有新交易（对比昨天的持仓）
    # 如果有，发送邮件
    echo "有新交易！" | mail -s "模拟盘交易提醒" your@email.com
fi
```

#### 方式2: 微信通知（需要企业微信）

可以配置企业微信机器人，在有交易时推送消息。

---

## 📋 完整检查清单

### 必须设置（核心）

- [x] ✅ 自动交易机器人已运行
- [x] ✅ 持仓文件已创建
- [x] ✅ V4配置已加载
- [ ] ❌ **每日自动运行（cron）** - **需要手动设置**

### 建议设置（优化）

- [ ] ⚠️ 每周更新数据（cron）
- [ ] ⚠️ 通知提醒（邮件/微信）
- [ ] ⚠️ 定期检查日志

### 可选设置（高级）

- [ ] 📊 数据备份
- [ ] 📈 性能监控
- [ ] 🔔 异常告警

---

## 🚀 立即行动（3分钟）

### 第1步：设置每日自动运行（必须）

```bash
# 1. 打开终端
# 2. 复制粘贴以下命令
crontab -e

# 3. 在打开的编辑器中，添加这一行
30 15 * * 1-5 cd /Users/rowan/clawd/projects/ai-quant-agent && ./daily_run.sh >> logs/auto_trading.log 2>&1

# 4. 保存退出
# vim: 按 ESC，输入 :wq，回车
# nano: 按 Ctrl+X，输入 Y，回车

# 5. 验证
crontab -l
```

### 第2步：设置每周更新数据（建议）

```bash
# 在同一个crontab中添加
0 18 * * 0 cd /Users/rowan/clawd/projects/ai-quant-agent && python3 examples/fetch_tushare_auto.py >> logs/data_update.log 2>&1
```

### 第3步：测试

```bash
# 手动运行一次，确保一切正常
cd /Users/rowan/clawd/projects/ai-quant-agent
./daily_run.sh

# 查看日志
tail -20 logs/auto_trading.log
```

---

## 📊 设置完成后的状态

### 每天自动发生：

```
15:30 - 系统自动运行
  ↓
检查4只股票的MA信号
  ↓
如果有买入信号 → 自动买入
如果有卖出信号 → 自动卖出
  ↓
保存交易记录
  ↓
保存日志到 logs/auto_trading.log
```

### 每周自动发生：

```
周日 18:00 - 自动更新股票数据
  ↓
从TuShare获取最新数据
  ↓
更新 data/real_*.csv 文件
  ↓
保存日志到 logs/data_update.log
```

---

## 🎯 验证设置

### 检查cron是否设置成功

```bash
# 查看当前的cron任务
crontab -l

# 应该看到：
# 30 15 * * 1-5 cd /Users/rowan/clawd/projects/ai-quant-agent && ./daily_run.sh >> logs/auto_trading.log 2>&1
# 0 18 * * 0 cd /Users/rowan/clawd/projects/ai-quant-agent && python3 examples/fetch_tushare_auto.py >> logs/data_update.log 2>&1
```

### 检查文件权限

```bash
# 确保脚本有执行权限
ls -la /Users/rowan/clawd/projects/ai-quant-agent/daily_run.sh
# 应该显示: -rwxr-xr-x (有x执行权限)

# 如果没有，运行：
chmod +x /Users/rowan/clawd/projects/ai-quant-agent/daily_run.sh
```

---

## 📞 常见问题

### Q: 如果我不想用cron怎么办？

A: 可以每天手动运行：
```bash
python3 examples/auto_trading_bot.py
```

### Q: cron不运行怎么办？

A: 检查：
1. 路径是否正确（使用绝对路径）
2. 脚本是否有执行权限
3. 查看系统日志：`tail -f /var/log/system.log`

### Q: 如何知道系统是否运行？

A: 查看日志：
```bash
tail -f logs/auto_trading.log
```

### Q: 如何停止自动运行？

A: 编辑crontab删除相关行：
```bash
crontab -e
# 删除对应的行
# 保存退出
```

---

## ✅ 最终确认

### 必须确认的3件事：

1. **自动交易机器人已运行** ✅
   ```bash
   python3 examples/status_monitor.py
   # 应该显示当前状态
   ```

2. **每日自动运行已设置** ❓ **需要你手动设置**
   ```bash
   crontab -l
   # 应该看到每日运行的任务
   ```

3. **知道如何查看状态** ✅
   ```bash
   # 查看状态
   python3 examples/status_monitor.py

   # 查看持仓
   cat data/auto_portfolio.json

   # 查看日志
   tail -f logs/auto_trading.log
   ```

---

## 🎉 总结

### 已完成（自动）：
- ✅ 自动交易机器人
- ✅ V4最优配置
- ✅ 监控工具
- ✅ 文档系统

### 需要你手动设置（2分钟）：
- ❌ **每日自动运行（cron）** ← **最重要！**
- ⚠️ 每周更新数据（可选）

### 设置完成后：
- 🤖 系统每天自动运行
- 📊 自动检测信号
- 💰 自动交易
- 📝 自动记录
- ⏰ 3个月后查看结果

---

**现在就去设置cron吧！只需2分钟！** 🚀

```bash
crontab -e
# 添加: 30 15 * * 1-5 cd /Users/rowan/clawd/projects/ai-quant-agent && ./daily_run.sh >> logs/auto_trading.log 2>&1
```
