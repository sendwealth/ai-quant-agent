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

### 可信度硬约束（建议 #2–#5 落地）
- **#2 数据可信门禁**：新增 `data/gate.py`（`evaluate_trust` / `DataTrustError`）。
  `sample`/`low` 数据被**硬阻断**进入交易与回测决策（`trading`/`backtest`），只读用途
  强制报告水印 + Web 红色警示横幅；`web/server.py` 暴露 `data_warning` 标志。
- **#2 fail closed（缺谱系默认拒绝）**：`evaluate_trust` 对决策用途（``trading``/
  ``backtest``）在**无数据谱系**时由「默认放行」改为 **fail closed**——缺谱系即禁止
  执行（`allowed=False`），仅当显式传入 ``research_mode=True`` 时豁免（放行并标红，
  不构成决策依据）。``TradingService.execute`` / ``BacktestEngine.run`` /
  ``WalkForwardValidator.run`` / ``Orchestrator.analyze`` 均新增 ``research_mode``
  透传参数；``Orchestrator`` 在数据执行前先汇总 ``data_lineage``，使门禁可校验谱系。
  既有的引擎/回测单测与探索脚本统一标注 ``research_mode=True``。
- **#3 数据源健康评分 + 告警**：`observability/health.py` 新增 `compute_data_health_score`
  （单源 0–100 + 整体聚合 + 失败源列表）；`observability/alerting.py` 新增
  `smoke_source_failure_rule`（真实失败源 critical）与 `data_health_score_rule`
  （健康分过低 warning），并注册为默认规则；`data/smoke.py` 报告附带 `data_health_score`；
  CI 工作流把冒烟结果渲染为 Step Summary 并暴露 `degraded`/`failed_sources` 输出（不再只
  上传空 artifact），脚本新增 `--out` 写出 `smoke-report.json`。
- **#4 回测可信度**：`backtest/engine.py` 记录复权方式（`adjust`）与 point-in-time 校验
  问题（`point_in_time_issues`）；新增 `backtest/walk_forward.py`——无泄漏的
  `walk_forward_splits`、样本内/外验证器 `WalkForwardValidator`、`validate_point_in_time`
  （信号越界/NaN/日期乱序检测）。滑点与佣金此前已在引擎内。
- **#3 walk-forward 逐日无前视（防前视泄漏增强）**：`WalkForwardValidator.run` 默认
  ``strict=True``，样本外信号改为**逐日生成**——每个测试 bar 仅将 ``train + test[:i+1]``
  喂给策略函数，结构性杜绝策略内部引用未来行导致的前视泄漏（旧行为 ``strict=False``
  仍保留但默认关闭）。`WalkForwardFold` 新增 ``oos_signals`` 字段便于审计；策略返回长度
  与窗口不一致时严格模式报错。`oos_signals` 以 NaN→持有(0) 并沿用上一有效信号。
- **#5 实盘就绪 scaffold**：新增 `execution/broker.py`（刻意与模拟执行器解耦）——券商适配
  `BrokerAdapter`、幂等下单 `IdempotentBroker`（`make_idempotency_key` 去重）、回报对账
  `OrderReconciler`、市场状态约束 `MarketCalendar` 与涨跌停判定 `price_within_limit`。
  明确标注为**模板/非生产**，需资质团队接入真实券商后方可启用。

## [3.1.0] - 2026-04-11
- 多 Agent 协作架构（基本面 / 技术 / 情感 / 规划 / 风控 / 执行）。
- LLM 增强：情感分析、指令解析、报告生成、风险解读（OpenAI / 智谱双 provider）。
- 智能选股模块（预筛 + 四维评分 + 降级池）。
- 4 源数据降级（Tushare / efinance / AkShare / BaoStock）+ 离线模式 + 数据修复。
- PaperTrading 持久化模拟交易 + 追加式审计日志 + 邮件通知。

---

[Unreleased]: https://github.com/<your-org>/ai-quant-agent/compare/v3.1.0...HEAD
[3.1.0]: https://github.com/<your-org>/ai-quant-agent/releases/tag/v3.1.0
