# 脚本索引

> 📚 AI 量化交易系统 - 脚本使用指南

**最后更新**: 2026-03-25 21:45

---

## 🚀 快速启动

```bash
# 查看帮助
./scripts/quick_start.sh help

# 检查系统
./scripts/quick_start.sh check

# 更新数据
./scripts/quick_start.sh update

# 测试邮件
./scripts/quick_start.sh email

# 备份配置
./scripts/quick_start.sh backup

# 生成报告
./scripts/quick_start.sh report
```

---

## 📋 脚本清单

### 核心脚本

#### 1. data_updater_robust.py
**功能**: 容错数据更新器（增强版）

**特性**:
- ✅ 3个数据源（AkShare + Tushare + 新浪财经）
- ✅ 指数退避重试（5秒 → 10秒 → 20秒）
- ✅ 8项数据验证（去重、缺失值、异常值等）
- ✅ 数据质量评分（0-100分）
- ✅ 邮件告警

**使用**:
```bash
python3 scripts/data_updater_robust.py
```

**输出**:
```json
{
  "timestamp": "2026-03-25T21:31:29",
  "total": 4,
  "success": 4,
  "failed": 0,
  "details": [...]
}
```

---

#### 2. heartbeat_check_enhanced.py
**功能**: 增强版心跳检查

**特性**:
- ✅ 数据健康检查
- ✅ 数据过期自动触发更新
- ✅ 账户状态检查
- ✅ 风险告警

**使用**:
```bash
python3 scripts/heartbeat_check_enhanced.py
```

**输出**:
```
✅ 量化系统正常
📊 系统状态:
  数据: 2026-03-25
  账户: ¥100,000.00 (+0.00%)
```

---

#### 3. config_manager.py
**功能**: 配置管理器

**特性**:
- ✅ 配置备份
- ✅ 配置恢复
- ✅ 健康检查
- ✅ 备份列表

**使用**:
```bash
# 备份配置
python3 scripts/config_manager.py backup

# 恢复配置
python3 scripts/config_manager.py restore

# 检查健康
python3 scripts/config_manager.py check

# 列出备份
python3 scripts/config_manager.py list
```

---

#### 4. test_email_alert.py
**功能**: 邮件告警测试

**使用**:
```bash
python3 scripts/test_email_alert.py
```

**输出**:
```
✅ 测试邮件发送成功！
   收件人: sendwealth@163.com
   请检查收件箱（或垃圾邮件文件夹）
```

---

#### 5. system_check.py
**功能**: 系统健康检查

**检查项**:
- ✅ 环境变量
- ✅ 配置文件
- ✅ 数据文件
- ✅ 数据新鲜度
- ✅ 依赖包
- ✅ 目录结构

**使用**:
```bash
python3 scripts/system_check.py
```

**输出**:
```
============================================================
🔍 量化系统健康检查报告
============================================================
✅ 环境变量: .env 文件配置完整
✅ 配置文件: 配置文件存在
✅ 数据文件: 数据文件完整 (4 只股票)
✅ 数据新鲜度: 数据新鲜（0 天前）
✅ 依赖包: 依赖包完整
✅ 目录结构: 目录结构完整
============================================================
✅ 系统状态良好，可以正常运行
============================================================
```

---

#### 6. quick_start.sh
**功能**: 快速启动脚本（一键运行）

**命令**:
- `check` - 检查系统状态
- `update` - 更新数据
- `email` - 测试邮件告警
- `backup` - 备份配置
- `report` - 生成报告
- `help` - 显示帮助

**使用**:
```bash
./scripts/quick_start.sh check
```

---

### 工具脚本

#### check_data_health.py
**功能**: 数据健康检查

**使用**:
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

#### use_config.py
**功能**: 配置使用示例

**使用**:
```bash
python3 scripts/use_config.py
```

---

## 📊 脚本对比

| 脚本 | 功能 | 状态 | 推荐 |
|------|------|------|------|
| data_updater_robust.py | 数据更新（增强版） | ✅ 主力 | ⭐⭐⭐⭐⭐ |
| heartbeat_check_enhanced.py | 心跳检查（增强版） | ✅ 主力 | ⭐⭐⭐⭐⭐ |
| config_manager.py | 配置管理 | ✅ 工具 | ⭐⭐⭐⭐ |
| test_email_alert.py | 邮件测试 | ✅ 工具 | ⭐⭐⭐⭐ |
| system_check.py | 系统检查 | ✅ 工具 | ⭐⭐⭐⭐⭐ |
| quick_start.sh | 快速启动 | ✅ 入口 | ⭐⭐⭐⭐⭐ |

---

## 🎯 使用场景

### 场景1: 日常运维

```bash
# 1. 检查系统
./scripts/quick_start.sh check

# 2. 更新数据
./scripts/quick_start.sh update

# 3. 查看报告
./scripts/quick_start.sh report
```

### 场景2: 首次部署

```bash
# 1. 检查系统
python3 scripts/system_check.py

# 2. 配置环境变量
nano .env

# 3. 配置数据源
nano config/data_sources.yaml

# 4. 测试邮件
python3 scripts/test_email_alert.py

# 5. 更新数据
python3 scripts/data_updater_robust.py
```

### 场景3: 问题排查

```bash
# 1. 系统检查
python3 scripts/system_check.py

# 2. 数据健康检查
python3 scripts/check_data_health.py

# 3. 查看日志
tail -100 logs/data_update.log
```

### 场景4: 定时任务

```bash
# 每天更新数据
0 18 * * * cd ~/clawd/projects/ai-quant-agent && ./scripts/quick_start.sh update >> logs/cron.log 2>&1

# 每周备份配置
0 0 * * 0 cd ~/clawd/projects/ai-quant-agent && ./scripts/quick_start.sh backup >> logs/cron.log 2>&1

# 每月生成报告
0 0 1 * * cd ~/clawd/projects/ai-quant-agent && ./scripts/quick_start.sh report >> logs/cron.log 2>&1
```

---

## 🔧 开发脚本

### 脚本模板

```python
#!/usr/bin/env python3
"""
脚本名称 - 脚本功能描述

特性：
1. 功能1
2. 功能2
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入依赖
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """主函数"""
    logger.info("开始执行...")
    # 你的代码
    logger.info("执行完成")


if __name__ == '__main__':
    exit(main())
```

### 最佳实践

1. **使用 logger 而非 print**
2. **添加项目根目录到 sys.path**
3. **使用 argparse 处理命令行参数**
4. **返回合适的状态码（0=成功，1=失败）**
5. **添加文档字符串**
6. **捕获异常并记录日志**

---

## 📝 脚本开发清单

创建新脚本时，确保：

- [ ] 添加到 `scripts/` 目录
- [ ] 使用 logger 记录日志
- [ ] 添加 PROJECT_ROOT 到 sys.path
- [ ] 添加文档字符串
- [ ] 返回正确的状态码
- [ ] 更新本索引文档
- [ ] 添加使用示例

---

## 📚 相关文档

- [主文档](README.md)
- [操作指南](OPERATION_GUIDE.md)
- [数据更新详解](DATA_UPDATER_GUIDE.md)
- [配置管理详解](CONFIG_MANAGEMENT.md)

---

_最后更新: 2026-03-25 21:45_
