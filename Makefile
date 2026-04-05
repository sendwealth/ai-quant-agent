.PHONY: help check update email backup report clean install test

# 默认目标
.DEFAULT_GOAL := help

# 帮助信息
help: ## 显示帮助信息
	@echo "🚀 AI 量化交易系统 - Make 命令"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""

# 系统检查
check: ## 检查系统状态
	@echo "🔍 系统健康检查..."
	@python3 scripts/system_check.py

# 更新数据
update: ## 更新股票数据
	@echo "📊 更新股票数据..."
	@python3 scripts/data_updater_robust.py

# 心跳检查
heartbeat hb: ## 心跳检查
	@echo "💓 心跳检查..."
	@python3 scripts/heartbeat_check_enhanced.py

# 测试邮件
email: ## 测试邮件告警
	@echo "📧 测试邮件告警..."
	@python3 scripts/test_email_alert.py

# 备份配置
backup: ## 备份配置文件
	@echo "💾 备份配置文件..."
	@python3 scripts/config_manager.py backup

# 恢复配置
restore: ## 恢复配置文件
	@echo "🔄 恢复配置文件..."
	@python3 scripts/config_manager.py restore

# 生成报告
report: ## 生成系统报告
	@echo "📈 生成系统报告..."
	@./scripts/quick_start.sh report

# 安装依赖
install: ## 安装依赖包
	@echo "📦 安装依赖包..."
	@pip install -r requirements.txt

# 运行测试
test: ## 运行测试
	@echo "🧪 运行测试..."
	@python3 -m pytest tests/ -v

# 代码格式化
format: ## 格式化代码
	@echo "✨ 格式化代码..."
	@black scripts/ agents/ core/ utils/
	@isort scripts/ agents/ core/ utils/

# 清理临时文件
clean: ## 清理临时文件
	@echo "🧹 清理临时文件..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type f -name "*.log" -delete
	@rm -rf .pytest_cache .coverage htmlcov/

# 完整流程
all: check update report ## 完整流程（检查 + 更新 + 报告）

# 开发环境设置
setup: install ## 设置开发环境
	@echo "✅ 开发环境设置完成"
	@echo ""
	@echo "下一步:"
	@echo "  1. 配置环境变量: nano .env"
	@echo "  2. 配置数据源: nano config/data_sources.yaml"
	@echo "  3. 测试邮件: make email"
	@echo "  4. 更新数据: make update"

# 查看日志
logs: ## 查看最新日志
	@tail -50 logs/data_update.log

# 监控模式
monitor: ## 实时监控（每小时检查）
	@echo "🔍 开始实时监控（按 Ctrl+C 停止）..."
	@while true; do \
		./scripts/quick_start.sh check; \
		echo ""; \
		echo "下次检查: 1小时后"; \
		sleep 3600; \
	done

# 定时任务设置
cron: ## 设置定时任务
	@echo "⏰ 设置定时任务..."
	@echo ""
	@echo "添加以下内容到 crontab:"
	@echo ""
	@echo "# 数据更新（每天18:30）"
	@echo "30 18 * * * cd $(PWD) && make update >> logs/cron.log 2>&1"
	@echo ""
	@echo "# 心跳检查（每小时）"
	@echo "0 * * * * cd $(PWD) && make heartbeat >> logs/cron.log 2>&1"
	@echo ""
	@echo "# 配置备份（每周一）"
	@echo "0 0 * * 1 cd $(PWD) && make backup >> logs/cron.log 2>&1"
	@echo ""
	@echo "运行 'crontab -e' 编辑定时任务"

# 显示版本
version: ## 显示版本信息
	@echo "AI 量化交易系统 v2.0.0"
	@echo "最后更新: 2026-03-25"

# 快速开始
quickstart: ## 快速开始指南
	@echo "🚀 快速开始指南"
	@echo ""
	@echo "1. 安装依赖:"
	@echo "   make install"
	@echo ""
	@echo "2. 配置环境:"
	@echo "   cp .env.example .env"
	@echo "   nano .env"
	@echo ""
	@echo "3. 检查系统:"
	@echo "   make check"
	@echo ""
	@echo "4. 更新数据:"
	@echo "   make update"
	@echo ""
	@echo "5. 测试邮件:"
	@echo "   make email"
	@echo ""
	@echo "完成！查看 Makefile 了解更多命令。"
