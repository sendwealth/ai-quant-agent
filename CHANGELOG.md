# 更新日志

本文档记录项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [2.7.0] - 2026-04-05

### 🎯 项目简化

**重大优化**: 删除55个冗余文件，代码量减少42%

#### 删除文件 🗑️
- ❌ 13个邮件脚本 (basic_email.py, simple_email.py等)
- ❌ 1个冗余配置 (config.py)
- ❌ 12个重复报告 (BACKTEST_REPORT_FIXED.md等)
- ❌ 7个根目录报告 (simulation_report.md等)
- ❌ 22个冗余脚本 (auto_fix_p0.py等)

#### 简化成果 📊
- Python文件: 100 → 62 (-38%)
- 文档文件: 53 → 21 (-60%)
- 代码行数: 19,250 → 12,591 (-35%)

#### 保留核心 🎯
- ✅ 10个Agents (完整功能)
- ✅ 14个核心脚本 (回测、选股、监控)
- ✅ 7个测试文件
- ✅ 12个核心文档

### 🔐 安全修复

**紧急修复**: .env文件泄露

- ✅ Git历史清理 (16个提交)
- ✅ 远程仓库强制更新
- ✅ Pre-commit hook创建

**待办**: 重新生成所有凭证

---

## [2.6.0] - 2026-04-05

### 新增 ✨

#### 开源最佳实践
- **LICENSE**: MIT许可证
- **CONTRIBUTING.md**: 贡献指南（中文详细版）
- **CODE_OF_CONDUCT.md**: 行为准则
- **pyproject.toml**: Python项目配置（PEP 517）
- **requirements-dev.txt**: 开发依赖

#### 动态选股系统
- **scripts/dynamic_stock_selector.py**: 自动扫描29只股票
- 多维度评分（技术+财务+成长）
- Top 10推荐（自动更新配置）

#### 财务数据修复
- **utils/financial_data_fetcher_v2.py**: 腾讯/新浪实时行情
- 真实P/E、P/B、ROE数据
- 修复P/E=0, P/B=0问题

### 修复 🐛

#### Agents修复 (4个)
- ✅ Technical Analyst: 修复损坏代码
- ✅ Fundamentals Analyst: 修复损坏代码
- ✅ Sentiment Analyst: 修复损坏代码
- ✅ Growth Analyst: 实现完整功能

#### 代码质量修复 (2个)
- ✅ risk_agent.py: 实现相关性风险计算
- ✅ technical_analyst.py: 修复裸露except

### 优化 🚀

- Agent可用性: 16.7% → 100%
- 数据质量: 模拟数据 → 真实数据
- 健康度: 94/100 → 97/100

---

## [2.0.0] - 2026-03-25

### 新增 ✨

#### 核心功能
- **多数据源支持**: AkShare、Tushare、新浪财经
- **指数退避重试**: 智能重试机制
- **数据验证系统**: 8项数据验证
- **邮件告警系统**: 自动通知

#### 工具脚本
- system_check.py
- quick_start.sh
- config_manager.py
- test_email_alert.py

#### 文档系统
- docs/README.md
- docs/SCRIPT_INDEX.md
- docs/CONFIG_MANAGEMENT.md
- docs/DATA_UPDATER_GUIDE.md

### 优化 🚀

- 数据更新速度提升 20%
- 数据验证效率提升 50%
- 内存使用降低 30%

---

## [1.0.0] - 2026-03-04

### 新增 ✨

- 初始版本发布
- 基础量化策略
- 数据获取功能
- 简单回测系统

---

_格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)_
_EOF
