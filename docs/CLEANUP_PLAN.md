# 📋 项目简化计划

**当前状态**:
- Python文件: 100个
- 文档文件: 53个
- 总代码行数: 19,250行
- 冗余文件: ~40个

**目标**:
- Python文件: 50个 (-50%)
- 文档文件: 10个 (-81%)
- 代码行数: ~10,000行 (-48%)

---

## 🗑️ 冗余文件清单

### 1️⃣ 根目录邮件脚本 (11个 → 删除)

**冗余邮件脚本**:
- basic_email.py
- simple_email.py
- manual_email.py
- minimal_email.py
- minimal_test.py
- pure_ascii_email.py
- quick_email_report.py
- send_final_report.py
- send_final_report_simple.py
- send_final_report_v2.py
- send_test_email.py
- simple_ascii_test.py
- utf8_email_report.py

**保留**: 无（使用 scripts/send_daily_report.py）

---

### 2️⃣ 根目录冗余配置 (1个 → 删除)

**冗余配置**:
- config.py（使用 config/settings.py）

---

### 3️⃣ docs目录重复报告 (20+个 → 5个)

**删除**:
- BACKTEST_REPORT_FIXED.md
- COMPLETE_TEST_REPORT.md
- COMPREHENSIVE_BACKTEST_REPORT.md
- DEEP_inspection_report.md
- DEeper_inspection_report.md
- FINAL_INSPECTION_SUMMARY.md
- FINAL_OPTIMAL_STRATEGY.md
- FULL_BACKTEST_REPORT.md
- OPEN_SOURCE_BEST_PRACTICES_REPORT.md
- PROJECT_HEALTH_CHECK_REPORT.md
- SCRIPT_INDEX.md
- TUSHARE_INTEGRATION_REPORT.md

**保留**:
- README.md (项目总览)
- QUICKSTART.md (快速开始)
- OPERATION_GUIDE.md (操作指南)
- CONFIG_MANAGEMENT.md (配置管理)
- SECURITY_INCIDENT_REPORT.md (安全事件)

---

### 4️⃣ 根目录重复报告 (5个 → 删除)

**删除**:
- current_status_summary.md
- final_simulation_report.md
- latest_project_status_report.md
- latest_simulation_report.md
- operation_summary.md
- operation_summary_final.md
- simulation_report.md

---

### 5️⃣ scripts目录冗余脚本 (30个 → 15个)

**删除**:
- auto_fix_p0.py（一次性修复）
- backtest_visualization.py（未使用）
- check_data_health.py（合并到 quant_monitor.py）
- clawteam_launcher.py（未使用）
- comprehensive_backtest.py（重复）
- config_manager.py（使用 core/config_loader.py）
- dynamic_rotation_strategy.py（未使用）
- end_to_end_test.py（测试完成）
- optimize_v2.py（旧版本）
- optimized_strategy_v3.py（未使用）
- phase5_test.py（测试完成）
- quality_stock_screener.py（使用 v2）
- quick_backtest_fixed.py（临时文件）
- realtime_monitor.py（使用 quant_monitor.py）
- remove_all_mock_data.py（一次性清理）
- replace_prints.py（一次性修复）
- send_report.py（重复）
- send_simulation_report.py（重复）
- simple_timing_strategy.py（未使用）
- smart_selection_strategy_v5.py（未使用）
- system_optimizer.py（一次性优化）
- test_email_alert.py（测试完成）
- use_config.py（示例文件）

**保留**:
- backtest_7agent.py（核心回测）
- data_updater_robust.py（数据更新）
- dynamic_stock_selector.py（动态选股）
- full_backtest.py（完整回测）
- get_real_price.py（价格获取）
- heartbeat_check.py（心跳检查）
- heartbeat_check_enhanced.py（增强检查）
- multi_stock_backtest.py（多股票回测）
- quality_stock_screener_v2.py（股票筛选）
- quant_monitor.py（量化监控）
- quickstart.py（快速开始）
- risk_monitor.py（风险监控）
- send_daily_report.py（发送报告）
- system_check.py（系统检查）

---

### 6️⃣ .archive目录 (保留)

**已归档**: 3个文件（保留）

---

### 7️⃣ data/reports目录 (清理)

**删除**: 旧报告文件
**保留**: 最新报告

---

## 📊 清理统计

| 类别 | 删除 | 保留 | 减少 |
|------|:----:|:----:|:----:|
| Python文件 | 40 | 50 | -40% |
| 文档文件 | 15 | 5 | -75% |
| 代码行数 | 8,000 | 11,250 | -42% |

---

## 🎯 最终结构

```
ai-quant-agent/
├── agents/              # 10个Agents
├── core/                # 核心模块
├── utils/               # 工具函数
├── config/              # 配置文件
├── scripts/             # 15个核心脚本
├── tests/               # 测试文件
├── docs/                # 5个核心文档
├── data/                # 数据目录
└── README.md            # 项目说明
```

**总计**:
- Python文件: 50个
- 文档文件: 10个
- 代码行数: ~11,000行

---

_计划创建时间: 2026-04-05 19:48_
