# 快速开始指南

> 5分钟快速上手 AI 量化交易系统

---

## 📋 准备工作

### 必需
- Python 3.9+
- pip
- 1GB 可用磁盘空间

### 可选
- 163邮箱（用于告警通知）
- Tushare账户（免费，获取 token）

---

## 🚀 5步快速开始

### 步骤 1: 下载项目（30秒）

```bash
git clone https://github.com/your-repo/ai-quant-agent.git
cd ai-quant-agent
```

### 步骤 2: 安装依赖（1分钟）

```bash
make install
```

或手动安装：
```bash
pip install -r requirements.txt
```

### 步骤 3: 配置环境（2分钟）

```bash
# 复制配置模板
cp .env.example .env

# 编辑配置
nano .env
```

**最小配置**（必需）:
```bash
# TuShare Token（免费）
TUSHARE_TOKEN=33649d8db312befd2e253d93e9bd2860e9c5e819864c8a2078b3869b
```

**完整配置**（推荐）:
```bash
# TuShare Token
TUSHARE_TOKEN=33649d8db312befd2e253d93e9bd2860e9c5e819864c8a2078b3869b

# 邮件配置（用于告警）
EMAIL_SMTP_SERVER=smtp.163.com
EMAIL_SMTP_PORT=465
EMAIL_SENDER=your_email@163.com
EMAIL_PASSWORD=your_auth_code
EMAIL_RECIPIENTS=alert@163.com
```

### 步骤 4: 检查系统（30秒）

```bash
make check
```

**预期输出**:
```
✅ 环境变量: .env 文件配置完整
✅ 配置文件: 配置文件存在
✅ 数据文件: 数据文件完整 (4 只股票)
✅ 依赖包: 依赖包完整
✅ 系统状态良好，可以正常运行
```

### 步骤 5: 更新数据（1分钟）

```bash
make update
```

**预期输出**:
```json
{
  "total": 4,
  "success": 4,
  "failed": 0
}
```

---

## ✅ 验证安装

### 检查数据

```bash
ls -lh data/
```

**预期输出**:
```
-rw-r--r-- 1 user user 157K Mar 25 22:00 300750_宁德时代.csv
-rw-r--r-- 1 user user 315K Mar 25 22:00 002475_立讯精密.csv
-rw-r--r-- 1 user user 158K Mar 25 22:00 601318_中国平安.csv
-rw-r--r-- 1 user user 158K Mar 25 22:00 600276_恒瑞医药.csv
```

### 查看数据

```bash
head -5 data/300750_宁德时代.csv
```

### 测试邮件（可选）

```bash
make email
```

---

## 🎯 下一步

### 1. 查看数据

```bash
# 查看最新数据
tail -5 data/300750_宁德时代.csv

# 统计数据量
wc -l data/*.csv
```

### 2. 运行分析

```bash
# 运行分析师
python3 agents/buffett_analyst.py --stock 300750

# 汇总信号
python3 agents/risk_manager.py
```

### 3. 设置定时任务

```bash
# 查看定时任务配置
make cron

# 或手动编辑
crontab -e
```

### 4. 查看文档

```bash
# 查看所有命令
make help

# 查看脚本索引
cat docs/SCRIPT_INDEX.md

# 查看完整文档
cat docs/README.md
```

---

## 🔧 常用操作

### 每日操作

```bash
# 更新数据
make update

# 检查系统
make check

# 查看报告
make report
```

### 每周操作

```bash
# 备份配置
make backup

# 查看日志
make logs
```

### 故障排查

```bash
# 检查系统
make check

# 查看日志
tail -50 logs/data_update.log

# 测试邮件
make email
```

---

## ❓ 常见问题

### Q1: make: command not found

**解决**:
```bash
# macOS
xcode-select --install

# Linux
sudo apt-get install make
```

### Q2: pip install 失败

**解决**:
```bash
# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q3: TUSHARE_TOKEN 无效

**解决**:
1. 访问 https://tushare.pro/register
2. 注册账户（免费）
3. 获取 API Token
4. 更新 .env 文件

### Q4: 数据更新失败

**解决**:
1. 检查网络连接
2. 检查 TUSHARE_TOKEN 配置
3. 查看日志: `make logs`
4. 系统会自动切换到备用数据源

---

## 📚 学习资源

### 文档
- [系统总览](docs/README.md) - 完整介绍
- [脚本索引](docs/SCRIPT_INDEX.md) - 所有脚本
- [操作指南](docs/OPERATION_GUIDE.md) - 详细步骤

### 示例
- [数据更新示例](examples/data_update_example.py)
- [策略回测示例](examples/backtest_example.py)
- [Agent 协作示例](examples/agent_collab_example.py)

---

## 🎉 完成！

恭喜！你已经成功部署了 AI 量化交易系统。

**下一步建议**:
- [ ] 阅读完整文档: `docs/README.md`
- [ ] 了解所有脚本: `docs/SCRIPT_INDEX.md`
- [ ] 设置定时任务: `make cron`
- [ ] 配置邮件告警: `make email`
- [ ] 运行7-Agent系统: `python3 agents/risk_manager.py`

**需要帮助？**
- 查看日志: `make logs`
- 检查系统: `make check`
- 查看文档: `docs/`

---

_快速开始指南 v2.0.0 | 2026-03-25_
