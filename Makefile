# AI Quant Agent — 发布前检查 / 开发工作流
# 用法 (需先 `uv sync`):
#   make check      完整门禁：format + lint + type + test + build
#   make lint       代码风格 / 质量 (ruff)
#   make type       静态类型检查 (mypy)
#   make test       单元测试 + 集成测试（默认禁用网络）
#   make smoke      真实数据源连通性测试（需联网，默认不跑）
#   make build      构建 sdist/wheel
#   make format     自动格式化 (ruff format)
#   make fix        ruff 自动修复

PYTHON ?= uv run python
PIP ?= uv run

.PHONY: help check lint type test smoke build format fix clean release-verify

help:
	@echo "Targets: check lint type test smoke build format fix clean"

# 默认门禁：格式 + lint + 类型检查 + 测试 + 构建。类型检查为必过项。
check: format lint type test build
	@echo "\n✅ 全部门禁通过 (format/lint/type/test/build)"

lint:
	$(PYTHON) -m ruff check src tests scripts

type:
	$(PYTHON) -m mypy src/quant_agent

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) -m pytest -q -m smoke

build:
	uv build

format:
	$(PYTHON) -m ruff format src tests

fix:
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

clean:
	rm -rf build dist *.egg-info .mypy_cache .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

# P4.2/P4.3/P4.4 发布前额外校验：版本兼容 + 许可证清单
release-verify:
	$(PYTHON) -m quant_agent.compat
	$(PYTHON) scripts/license_check.py
