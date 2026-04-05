# Contributing to AI Quant Agent

感谢你对 AI 量化交易系统的关注！我们欢迎所有形式的贡献。

## 🤔 如何贡献

### 报告 Bug

1. 检查 [Issues](https://github.com/your-repo/ai-quant-agent/issues) 确保问题未被报告
2. 创建新 Issue，包含：
   - 清晰的标题和描述
   - 复现步骤
   - 期望行为
   - 实际行为
   - 系统环境（Python版本、操作系统等）

### 提出新功能

1. 在 Issues 中创建 "Feature Request"
2. 描述功能需求和使用场景
3. 等待维护者反馈

### 提交代码

#### 1. Fork & Clone

```bash
# Fork 项目到你的账号
# 然后克隆
git clone https://github.com/your-username/ai-quant-agent.git
cd ai-quant-agent

# 添加上游仓库
git remote add upstream https://github.com/your-repo/ai-quant-agent.git
```

#### 2. 创建分支

```bash
# 创建新分支
git checkout -b feature/your-feature-name
```

#### 3. 开发环境设置

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -e ".[dev]"

# 复制配置
cp .env.example .env
```

#### 4. 代码规范

我们使用以下工具保持代码质量：

```bash
# 格式化代码
black .

# 检查代码风格
flake8 .

# 类型检查
mypy .
```

#### 5. 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_agents.py

# 生成覆盖率报告
pytest --cov=. --cov-report=html
```

#### 6. 提交

```bash
# 查看更改
git status
git diff

# 添加更改
git add .

# 提交（使用清晰的提交信息）
git commit -m "feat: add new feature description"
# 或
git commit -m "fix: fix bug description"
# 或
git commit -m "docs: update documentation"
```

#### 7. 推送 & Pull Request

```bash
# 推送到你的 fork
git push origin feature/your-feature-name

# 在 GitHub 上创建 Pull Request
```

### Pull Request 指南

1. **标题**: 清晰简洁
2. **描述**: 
   - 改了什么
   - 为什么改
   - 如何测试
3. **关联 Issue**: 使用 "Fixes #123" 或 "Closes #456"
4. **测试**: 确保所有测试通过
5. **文档**: 更新相关文档

## 📝 代码规范

### Python 代码风格

- 遵循 [PEP 8](https://pep8.org/)
- 使用 [Black](https://github.com/psf/black) 格式化
- 最大行长度: 100 字符
- 使用类型提示 (Type Hints)

### 示例代码

```python
from typing import Dict, List, Optional
import pandas as pd


def analyze_stock(
    stock_code: str,
    data: pd.DataFrame,
    config: Optional[Dict] = None
) -> Dict[str, float]:
    """
    分析股票数据
    
    Args:
        stock_code: 股票代码
        data: 股票历史数据
        config: 配置参数
    
    Returns:
        分析结果字典，包含信号和信心度
    
    Raises:
        ValueError: 如果数据为空
    """
    if data.empty:
        raise ValueError("Data cannot be empty")
    
    # 分析逻辑
    result = {
        "signal": "BUY",
        "confidence": 0.75
    }
    
    return result
```

### 文档字符串

使用 Google 风格的文档字符串：

```python
def function(arg1: str, arg2: int) -> bool:
    """函数的简短描述.
    
    更详细的描述（可选）.
    
    Args:
        arg1: 参数1的说明
        arg2: 参数2的说明
    
    Returns:
        返回值的说明
    
    Raises:
        ValueError: 异常说明
    
    Example:
        >>> function("test", 42)
        True
    """
    return True
```

## 🧪 测试

### 单元测试

```python
import pytest
from agents.technical_analyst import TechnicalAnalyst


def test_calculate_rsi():
    """测试 RSI 计算"""
    analyst = TechnicalAnalyst("300750")
    rsi = analyst.calculate_rsi([1, 2, 3, 4, 5])
    assert 0 <= rsi <= 100
```

### 运行测试

```bash
# 所有测试
pytest

# 特定文件
pytest tests/test_agents.py

# 特定函数
pytest tests/test_agents.py::test_calculate_rsi

# 带覆盖率
pytest --cov=. --cov-report=html
```

## 📚 文档

### 文档结构

```
docs/
├── README.md           # 项目概览
├── QUICKSTART.md       # 快速开始
├── OPERATION_GUIDE.md  # 操作指南
├── API.md             # API 文档
└── CHANGELOG.md       # 更新日志
```

### 更新文档

- 修改功能时，更新相关文档
- 添加新功能时，创建新文档
- 修复 bug 时，在 CHANGELOG 中记录

## 🔒 安全

### 敏感信息

**永远不要提交**:
- `.env` 文件
- API 密钥
- 数据库密码
- 个人信息

### 报告安全漏洞

如果发现安全漏洞，请**不要**公开创建 Issue。  
请发送邮件至: security@example.com

## 📋 开发流程

1. **Issue**: 创建或认领 Issue
2. **分支**: 创建特性分支
3. **开发**: 编写代码和测试
4. **测试**: 确保测试通过
5. **文档**: 更新文档
6. **提交**: 提交 Pull Request
7. **审查**: 等待代码审查
8. **合并**: 合并到主分支

## 🏆 贡献者

感谢所有贡献者！查看 [CONTRIBUTORS.md](CONTRIBUTORS.md) 了解详情。

## 📞 联系方式

- **Issues**: GitHub Issues
- **讨论**: GitHub Discussions
- **邮件**: ai-quant-agent@example.com

---

再次感谢你的贡献！🎉
