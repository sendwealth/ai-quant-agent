# 配置管理指南

## 🚨 问题：为什么配置会丢失？

### 常见原因

1. **创建新脚本时覆盖配置**
   - 新脚本创建新配置文件，没有迁移旧配置
   - 解决：使用配置管理器迁移

2. **敏感信息未保存**
   - 密码、token 等敏感信息不在 git 中
   - 解决：使用 `.env` 文件

3. **没有备份机制**
   - 配置修改后没有备份
   - 解决：定期备份配置

---

## ✅ 解决方案：避免配置丢失

### 方案1：使用 .env 文件（推荐）⭐

**优点**：
- ✅ 敏感信息隔离
- ✅ 不会被 git 追踪
- ✅ 支持环境变量替换

**配置步骤**：

1. **创建 .env 文件**
   ```bash
   cd ~/clawd/projects/ai-quant-agent
   nano .env
   ```

2. **添加配置**
   ```bash
   # TuShare API Token
   TUSHARE_TOKEN=your_token_here

   # Email Configuration
   EMAIL_SMTP_SERVER=smtp.163.com
   EMAIL_SMTP_PORT=465
   EMAIL_SENDER=your_email@163.com
   EMAIL_PASSWORD=your_auth_code  # 授权码，不是密码！
   EMAIL_RECIPIENTS=recipient@163.com
   ```

3. **配置 YAML 使用环境变量**
   ```yaml
   tushare:
     enabled: true
     token: "${TUSHARE_TOKEN}"

   email_alert:
     enabled: true
     password: "${EMAIL_PASSWORD}"
   ```

---

### 方案2：配置备份机制

**备份配置**：
```bash
cd ~/clawd/projects/ai-quant-agent
python3 scripts/config_manager.py backup
```

**恢复配置**：
```bash
# 恢复最新备份
python3 scripts/config_manager.py restore

# 恢复指定备份
python3 scripts/config_manager.py restore --file config/backups/data_sources.yaml.20260325.bak
```

**列出备份**：
```bash
python3 scripts/config_manager.py list
```

**检查配置健康度**：
```bash
python3 scripts/config_manager.py check
```

---

### 方案3：配置模板管理

**创建配置模板**（不含敏感信息）：
```yaml
# config/data_sources.yaml.template
tushare:
  enabled: true
  token: "${TUSHARE_TOKEN}"  # 从环境变量读取

email_alert:
  enabled: true
  password: "${EMAIL_PASSWORD}"  # 从环境变量读取
```

**首次配置**：
```bash
# 1. 复制模板
cp config/data_sources.yaml.template config/data_sources.yaml

# 2. 配置 .env
nano .env

# 3. 检查配置
python3 scripts/config_manager.py check
```

---

## 📋 最佳实践

### 1. 敏感信息管理

**✅ 正确做法**：
- 使用 `.env` 文件存储密码、token
- `.env` 文件在 `.gitignore` 中
- YAML 使用 `${ENV_VAR}` 引用环境变量

**❌ 错误做法**：
- 在 YAML 中明文存储密码
- 在代码中硬编码密码
- 将 `.env` 文件提交到 git

### 2. 配置文件结构

```
config/
├── data_sources.yaml          # 主配置文件
├── data_sources.yaml.template # 模板文件（可提交git）
├── strategy_v4.yaml           # 策略配置
└── backups/                   # 备份目录
    ├── data_sources.yaml.20260325_210000.bak
    └── data_sources.yaml.20260324_180000.bak
```

### 3. 定期备份

**手动备份**：
```bash
python3 scripts/config_manager.py backup
```

**自动备份**（添加到 crontab）：
```bash
# 每天备份一次配置
0 0 * * * cd ~/clawd/projects/ai-quant-agent && python3 scripts/config_manager.py backup
```

### 4. 配置迁移

当创建新脚本或更新配置时：

```bash
# 1. 备份旧配置
python3 scripts/config_manager.py backup

# 2. 创建新配置
# ... 创建新脚本 ...

# 3. 迁移配置（保留旧值）
python3 scripts/config_manager.py migrate
```

---

## 🔧 配置检查清单

在部署新脚本前，检查以下项：

- [ ] `.env` 文件存在且包含所有必需的环境变量
- [ ] YAML 配置使用 `${ENV_VAR}` 引用环境变量
- [ ] 敏感信息不在 git 中
- [ ] 已创建配置备份
- [ ] 运行 `config_manager.py check` 检查健康度

---

## 📊 当前配置状态

### 已配置
- ✅ TUSHARE_TOKEN（在 `.env` 中）

### 待配置
- [ ] EMAIL_PASSWORD（邮件告警授权码）

### 配置文件
```
.env                        # 环境变量（不追踪）
config/data_sources.yaml    # 主配置文件
config/backups/             # 备份目录
```

---

## 🚀 快速开始

### 首次配置

1. **配置环境变量**
   ```bash
   cd ~/clawd/projects/ai-quant-agent
   nano .env
   ```

   添加：
   ```bash
   TUSHARE_TOKEN=your_token
   EMAIL_PASSWORD=your_auth_code
   ```

2. **检查配置**
   ```bash
   python3 scripts/config_manager.py check
   ```

3. **备份配置**
   ```bash
   python3 scripts/config_manager.py backup
   ```

### 启用邮件告警

1. **配置 .env**
   ```bash
   EMAIL_PASSWORD=your_163_auth_code
   ```

2. **启用邮件**
   ```yaml
   email_alert:
     enabled: true
   ```

3. **测试发送**
   ```bash
   python3 scripts/data_updater_robust.py
   ```

---

## 📝 相关文件

- `scripts/config_manager.py` - 配置管理器
- `scripts/data_updater_robust.py` - 容错数据更新（支持环境变量）
- `config/data_sources.yaml` - 主配置文件
- `.env` - 环境变量文件
- `.gitignore` - 忽略敏感文件

---

*创建时间: 2026-03-25*
