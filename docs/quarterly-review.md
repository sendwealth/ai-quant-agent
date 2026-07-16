# 季度维护审查（Quarterly Review）

> 对应开源成熟度要求 §7「每个季度检查依赖、数据源可用性、许可证、路线图和安全基线」。
> 维护者轮值执行，结果记录在 `docs/quarterly-review-YYYY-Qn.md`（本文件为模板）。

## 1. 依赖审查

- [ ] `uv pip compile` 重新解析，对比 `uv.lock` 是否有漂移
- [ ] `uv run python scripts/license_check.py` 许可证清单无新增高风险许可
- [ ] Dependabot / pip-audit 高危漏洞均已处理或记录风险接受
- [ ] Python 3.10 / 3.11 / 3.12 在 CI 矩阵均通过

## 2. 数据源可用性

- [ ] Tushare / efinance / AkShare / BaoStock 四个源的冒烟测试
      （`make smoke` 或 `.github/workflows/smoke-test.yml`）均通过
- [ ] 各源速率限制与字段映射是否需要更新（上游 API 变更）
- [ ] 降级链顺序与样例/离线兜底仍可用

## 3. 许可证与合规

- [ ] `LICENSE` 仍为 MIT，无新增代码引入不兼容许可
- [ ] 样例数据与第三方资源（图标/数据）再分发许可已确认
- [ ] 所有 API key 仍仅来自环境变量 / secret store

## 4. 路线图

- [ ] 里程碑 M1/M2/M3 进度更新
- [ ] 本季度新增的 good first issue 与社区反馈是否处理
- [ ] 下一季度重点（回测现实约束、实盘适配器、文档补齐等）

## 5. 安全基线

- [ ] `gitleaks` / `pip-audit` / CodeQL 仍在本仓库 CI 运行且无阻断项
- [ ] 分支保护规则（见下「主分支保护」）未被放宽
- [ ] 安全披露邮箱（`SECURITY.md`）仍有效
- [ ] Web 默认绑定 loopback、远程模式需认证的策略未退化

## 6. 结论

- 整体健康度：✅ / ⚠️ / ❌
- 需跟进事项（负责人 / 截止日期）：

| 事项 | 负责人 | 截止 |
|------|--------|------|
|      |        |      |

---

## 主分支保护规则（Branch Protection）

本仓库 `main` 分支受保护，合并需满足：

1. **CI 必须通过**：`quality` + `type` + `test`（3.10/3.11/3.12 矩阵）+ `build`
   以及 `security`（pip-audit + gitleaks）全部绿。
2. **至少 1 次代码审查**（required review）。
3. **禁止直接 push**：仅允许经 PR 合并；强制线性历史（squash merge）。
4. **DCO / CLA**：提交需签署（`git commit -s`）或经 CLA 检查；CI 中校验。
5. **分支最新**：PR 必须基于最新 `main`（required status checks 生效）。

> 上述规则在 GitHub 仓库 Settings → Branches 配置；本段为可核查的书面记录。
