# 🔴 Token账户不匹配问题

**错误**: `Permission to sendwealth/ai-quant-agent.git denied to oklaM`

**根本原因**: Token关联的账户与仓库所有者不匹配

---

## 🔍 问题分析

**当前状态**:
- **Token关联账户**: `oklaM`
- **仓库所有者**: `sendwealth`
- **权限**: ❌ `oklaM`没有权限推送到`sendwealth`的仓库

---

## ✅ 解决方案 (3选1)

### 方案1: 使用sendwealth账户的Token ⭐推荐

**步骤**:

1. **登录sendwealth账户**:
   - 访问: https://github.com/login
   - 使用`sendwealth`账户登录

2. **生成新token**:
   - 访问: https://github.com/settings/tokens
   - 点击 "Generate new token (classic)"
   - **重要**: 确认右上角显示的是 `sendwealth`账户
   - 勾选权限:
     - ✅ repo (完整仓库访问)
     - ✅ workflow
   - 点击 "Generate token"
   - **复制token** (只显示一次)

3. **更新本地配置**:
   ```bash
   cd ~/clawd/projects/ai-quant-agent
   git remote set-url origin https://<SENDWEALTH_TOKEN>@github.com/sendwealth/ai-quant-agent.git
   git push origin --all --force
   ```

---

### 方案2: 将oklaM添加为协作者

**步骤**:

1. **登录sendwealth账户**

2. **添加协作者**:
   - 访问: https://github.com/sendwealth/ai-quant-agent/settings/access
   - 点击 "Add people"
   - 搜索并添加 `oklaM`
   - 设置权限: **Write** 或 **Admin**

3. **oklaM接受邀请**:
   - 访问: https://github.com/oklaM?tab=invitations
   - 接受邀请

4. **使用oklaM的token推送**:
   ```bash
   cd ~/clawd/projects/ai-quant-agent
   git push origin --all --force
   ```

---

### 方案3: 转移仓库所有权

**步骤**:

1. **登录sendwealth账户**

2. **转移仓库**:
   - 访问: https://github.com/sendwealth/ai-quant-agent/settings
   - 滚动到 "Danger Zone" → "Transfer"
   - 转移到 `oklaM` 账户

3. **更新远程URL**:
   ```bash
   cd ~/clawd/projects/ai-quant-agent
   git remote set-url origin https://github.com/oklaM/ai-quant-agent.git
   git push -u origin main --force
   ```

**⚠️ 注意**: 转移仓库会改变仓库URL，可能影响其他协作者

---

## 📊 推荐方案对比

| 方案 | 优点 | 缺点 | 推荐度 |
|------|------|------|:------:|
| **方案1** - sendwealth token | ✅ 最简单<br>✅ 不改变仓库 | ❌ 需要sendwealth账户访问 | ⭐⭐⭐⭐⭐ |
| **方案2** - 添加协作者 | ✅ 保留两个账户<br>✅ 协作方便 | ❌ 需要两步操作 | ⭐⭐⭐⭐ |
| **方案3** - 转移所有权 | ✅ 完全控制 | ❌ 改变仓库URL<br>❌ 影响其他用户 | ⭐⭐ |

---

## 🎯 立即可执行的步骤

**最快方案 (方案1)**:

```bash
# 1. 登录 sendwealth 账户
# 2. 访问: https://github.com/settings/tokens
# 3. 生成新token (确认账户是sendwealth)
# 4. 复制token

# 5. 更新配置
cd ~/clawd/projects/ai-quant-agent
git remote set-url origin https://<SENDWEALTH_TOKEN>@github.com/sendwealth/ai-quant-agent.git

# 6. 推送
git push origin --all --force
```

**预期结果**:
```
Counting objects: 3, done.
Compressing objects: 100% (2/2), done.
Writing objects: 100% (2/2), 300 bytes | 300.00 KiB/s, done.
To https://github.com/sendwealth/ai-quant-agent.git
 + 9d47a83...f7c3abd main -> main (forced update)
```

---

## 📝 重要提示

**关键点**:
1. ⚠️ **必须使用sendwealth账户生成token**
2. ✅ 生成token时，确认右上角显示的是`sendwealth`
3. ✅ 勾选`repo`和`workflow`权限
4. ✅ Token只显示一次，立即复制保存

---

**创建时间**: 2026-04-05 20:51
**问题**: 账户不匹配 (oklaM vs sendwealth)
**解决方案**: 使用正确的账户生成token
