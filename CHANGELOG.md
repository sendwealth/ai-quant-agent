# 更新日志 (Changelog)

本文件遵循 [Keep a Changelog](https://keepachangelog.com/) 约定，版本号遵循
[语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 安全与开源成熟度加固（P0–P1）
- **P0-01/02**：修复集成测试配置契约回归，用真实 `Settings` 替换 `MagicMock` fixture，
  防止配置字段被静默吞掉。
- **P0-03**：引入 `pytest-socket` 网络隔离——单测/集成测试默认禁用 socket；
  真实数据源连通性测试归入 `tests/smoke/`（`@pytest.mark.smoke`），默认不运行。
- **P0-04**：消除测试中真实未验证 HTTPS 请求；CI 增加 `gitleaks` 密钥扫描与
  `pip-audit` 依赖漏洞扫描。
- **P0-05**：对齐包元数据——将 `baostock` 声明为可选依赖 extra，新增 CLI `--version`，
  README 顶部增加研究/模拟用途风险提示。
- **P0-06 / P1-01**：新增 `Makefile`（`make check` 组合 lint/type/test/build）、
  Ruff 配置与 `pre-commit` 钩子。
- **P1-02/03**：新增 GitHub Actions CI（Python 3.10/3.11/3.12 矩阵 + lint + 类型 +
  测试 + 构建）。
- **P1-04/05**：测试覆盖率门槛（≥70%），Dependabot、CodeQL、pip-audit、gitleaks 持续扫描。

### 协作契约与治理（P2）
- 新增 `CONTRIBUTING.md` / `CODE_OF_CONDUCT.md` / `SECURITY.md` / `SUPPORT.md` / `CHANGELOG.md`。
- 新增 Issue 模板（Bug / Feature）、PR 模板（含安全自查清单）。
- `SECURITY.md` 明确私有漏洞披露流程与现有安全设计（网络隔离 / TLS / 密钥管理 / 交易安全）。

### 数据谱系与运行模式（P3 / P4）
- **P3 数据谱系**：新增 `DataProvenance` 数据类，`FinancialSnapshot` 携带来源/获取时间/
  可信度；`DataService` 在各获取路径（缓存/实时源/合并/样例/离线）记录谱系，
  `AnalysisReport.data_lineage` 汇总并在 Markdown 报告末尾展示「数据来源」表。
- **P4 运行模式**：`Settings.run_mode`（`research`/`backtest`/`paper`/`live`），
  默认 `research`；`live` 在本开源版本未实现，设置后启动失败（fail-safe）。
- **P4 Web 加固**：`run_web` 非 loopback 绑定打印安全告警，启用端口复用。

### 发布治理与运营（P4 收尾）
- **P4.1 分支保护 / DCO**：`CONTRIBUTING.md` 明确主分支保护规则（CI 全绿、≥1 审查、
  禁直推、DCO 签署）；`.github/PULL_REQUEST_TEMPLATE.md` 增加安全自查清单。
- **P4.2 发布流程**：新增 `.github/workflows/release.yml`——tag 触发，先 TestPyPI
  验证安装再发 PyPI，生成 `SHA256SUMS` 与 release notes；`make release-verify`
  固化发布前兼容性与许可证自检。
- **P4.3 版本兼容**：新增 `quant_agent/compat` 模块与 `uv run python -m quant_agent.compat`
  CLI，校验 Python ≥3.10 与关键依赖版本区间。
- **P4.4 许可证清单**：新增 `scripts/license_check.py`，列出运行时依赖许可证并标记
  需复核的 copyleft / 商业许可，CI 可用于阻断。
- **P4.5 季度审查**：新增 `docs/quarterly-review.md` 模板（依赖/数据源/许可证/路线图/
  安全基线），书面记录主分支保护规则。

## [3.1.0] - 2026-04-11
- 多 Agent 协作架构（基本面 / 技术 / 情感 / 规划 / 风控 / 执行）。
- LLM 增强：情感分析、指令解析、报告生成、风险解读（OpenAI / 智谱双 provider）。
- 智能选股模块（预筛 + 四维评分 + 降级池）。
- 4 源数据降级（Tushare / efinance / AkShare / BaoStock）+ 离线模式 + 数据修复。
- PaperTrading 持久化模拟交易 + 追加式审计日志 + 邮件通知。

---

[Unreleased]: https://github.com/<your-org>/ai-quant-agent/compare/v3.1.0...HEAD
[3.1.0]: https://github.com/<your-org>/ai-quant-agent/releases/tag/v3.1.0
