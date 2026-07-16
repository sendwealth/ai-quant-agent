# 贡献指南 (Contributing)

感谢你考虑为 **AI Quant Agent** 做贡献！本文件说明如何本地搭建、开发与提交变更。

## 行为准则

参与本项目的所有贡献者均须遵守 [行为准则](CODE_OF_CONDUCT.md)。

## 开发环境

本项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖与虚拟环境。

```bash
# 1. 克隆仓库
git clone https://github.com/<your-org>/ai-quant-agent.git
cd ai-quant-agent

# 2. 安装依赖（含 baostock 可选源）
uv sync --extra baostock

# 3. 安装 pre-commit 钩子（提交前自动 lint/格式化）
uv run pre-commit install
```

## 本地验证

提交前请运行完整门禁（等价于 CI）：

```bash
make check      # format + lint + type + test + build
```

也可单独运行：

```bash
make lint       # ruff check
make type       # mypy
make test       # pytest（默认禁用网络，仅 smoke 标记需联网）
make smoke      # 真实数据源连通性测试（需网络）
```

## 代码规范

- 风格与质量由 **Ruff** 统一管控（行宽 100，双引号）。
- 类型注解使用 `from __future__ import annotations`，核心模块通过 `mypy`。
- 提交信息遵循 [Conventional Commits](https://www.conventionalcommits.org/)：
  `feat:` / `fix:` / `refactor:` / `docs:` / `test:` / `chore:`。

## 测试要求

- 新增功能 **必须** 包含单元测试；涉及多模块协作的变更建议补充集成测试。
- **单元测试不得访问网络**（CI 默认 `--disable-socket`）。需要真实数据源的测试归入
  `tests/smoke/` 并打上 `@pytest.mark.smoke`，这些测试默认不运行。
- LLM 相关测试 **必须 mock**，禁止在测试中对真实 API 发起请求。
- 所有数据必须基于真实来源；测试严禁伪造核心财务/行情数据。

## Pull Request

1. Fork 并创建特性分支（`feat/xxx` 或 `fix/xxx`）。
2. 确保 `make check` 全绿。
3. 在 PR 中说明：动机、变更内容、测试方式、是否影响默认行为。
4. 关联相关 Issue（如 `Closes #123`）。

## 发布流程

项目遵循语义化版本：破坏性变更升主版本、新增功能升次版本、修复升补丁版本。

1. 在 `CHANGELOG.md` 的 `Unreleased` 段汇总本次变更；发布时改为对应版本号与日期。
2. 确保 `make check`（`format/lint/type/test/build`）与 `make release-verify`
   （兼容性自检 + 许可证清单）全绿。
3. 推送带 `v` 前缀的 tag（如 `v3.2.0`）；`.github/workflows/release.yml` 会：
   - 重新验证门禁；
   - 先发布到 **TestPyPI** 并尝试验证安装；
   - 再发布到 **PyPI**，生成 `SHA256SUMS` 与 GitHub Release notes。
4. 手动触发可在 Actions → Release 中选择「仅 TestPyPI 验证」。

## 主分支保护

`main` 分支受保护，合并需满足：CI 全绿、至少 1 次审查、禁止直接 push、
提交需经 DCO 签署（`git commit -s`）或 CLA 检查。详见
[季度审查文档](docs/quarterly-review.md#主分支保护规则branch-protection)。

## 签署（DCO）

本仓库要求 **Developer Certificate of Origin (DCO)**。每次提交请使用：

```bash
git commit -s -m "feat: 你的变更"
```

提交的 `Signed-off-by` 行即表示你同意 DCO 条款。CI 会校验签名。

## 安全相关

**请勿在公开 Issue 中报告安全漏洞。** 请按 [SECURITY.md](SECURITY.md) 的私下进行披露。
