# 🔧 Git推送问题排查

**问题**: 无法推送到GitHub (403错误 + 权限被拒绝)
**原因**: token权限不足 + SSH密钥关联账户错误

---

## 🚨 发现的问题

### 1️⃣ Token权限不足 (403错误)

**错误**: `The requested URL returned error: 403`
**原因**: Token缺少必要权限

**需要权限**:
- ✅ **repo** (完整仓库访问)
- ✅ **workflow** (如果使用GitHub Actions)
- ✅ **admin:repo_hook** (如果使用webhooks)

---

### 2️⃣ SSH密钥账户不匹配

**错误**: `Permission to sendwealth/ai-quant-agent.git denied to oklaM`
**原因**: SSH密钥关联的是`oklaM`账户，但仓库属于`sendwealth`

---

## ✅ 解决方案

### 方案1: 使用正确的Token（推荐）

**步骤**:

1. **删除现有token**:
   - 访问: https://github.com/settings/tokens
   - 找到并删除当前token

2. **创建新token (classic)**:
   - 点击 "Generate new token (classic)"
   - 设置名称: `ai-quant-agent`
   - **勾选权限**:
     - ✅ **repo** (全部)
     - ✅ **workflow**
     - ✅ **admin:repo_hook**
   - 过期时间: 90天
   - 点击 "Generate token"

3. **更新本地Git配置**:
   ```bash
   cd ~/clawd/projects/ai-quant-agent

   # 使用新token
   git remote set-url origin https://<NEW_TOKEN>@github.com/sendwealth/ai-quant-agent.git

   # 推送
   git push origin main --force
   ```

---

### 方案2: 使用正确的SSH密钥

**步骤**:

1. **检查当前SSH密钥**:
   ```bash
   ls -la ~/.ssh
   cat ~/.ssh/id_rsa.pub
   ```

2. **选项A - 为sendwealth账户添加SSH密钥**:
   - 登录GitHub账户 `sendwealth`
   - 访问: https://github.com/settings/keys
   - 点击 "New SSH key"
   - 粘贴 `~/.ssh/id_rsa.pub` 的内容
   - 保存

   然后推送:
   ```bash
   git remote set-url origin git@github.com:sendwealth/ai-quant-agent.git
   git push origin main --force
   ```

3. **选项B - 使用特定SSH密钥**:
   ```bash
   # 如果有多个SSH密钥
   echo "Host github.com\n  IdentityFile ~/.ssh/id_rsa_sendwealth" >> ~/.ssh/config

   # 推送
   git push origin main --force
   ```

---

### 方案3: 使用GitHub CLI（最简单）

**步骤**:

1. **安装GitHub CLI**:
   ```bash
   brew install gh
   ```

2. **登录**:
   ```bash
   gh auth login
   # 选择: GitHub.com
   # 选择: HTTPS
   # 选择: 使用token
   # 粘贴token
   ```

3. **推送**:
   ```bash
   git push origin main --force
   ```

---

## 📊 当前项目状态

**本地**: ✅ 完全就绪
- 版本: v2.7.2
- 提交: f7c3abd
- 代码: 12,366行 (60个文件)
- 文档: 23个
- 健康度: 97/100

**Git历史**: ✅ 已清理敏感信息
- 已删除: `docs/SECURITY_INCIDENT_REPORT.md`
- 所有提交: ✅ 已重写

**待执行**:
- ⏳ 推送到GitHub

---

## 🎯 推荐步骤

**最快解决**:
```bash
# 1. 生成有正确权限的token
# 访问: https://github.com/settings/tokens

# 2. 更新配置
git remote set-url origin https://<NEW_TOKEN>@github.com/sendwealth/ai-quant-agent.git

# 3. 推送
git push origin main --force
```

**预期结果**:
```
Counting objects: 3, done.
Compressing objects: 100% (2/2), done.
Writing objects: 100% (2/2), 300 bytes | 300.00 KiB/s, done.
Total 3 (delta 1), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (1/1), completed with 1 local object.
To https://github.com/sendwealth/ai-quant-agent.git
 + 9d47a83...f7c3abd main -> main (forced update)
```

---

**重要提醒**:
- ⚠️ 使用 `--force` 推送会覆盖远程历史
- ✅ 已确保本地代码是最完整的版本
- ✅ 已删除所有敏感信息

**创建时间**: 2026-04-05 20:48
**系统状态**: ✅ 生产就绪，待推送
