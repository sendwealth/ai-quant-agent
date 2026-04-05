#!/bin/bash
# 快速启动脚本 - 一键运行常用操作

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# 显示帮助
show_help() {
    cat << EOF
🚀 AI 量化交易系统 - 快速启动脚本

用法: ./scripts/quick_start.sh [命令]

命令:
  check     检查系统状态
  update    更新数据
  email     测试邮件告警
  backup    备份配置
  report    生成报告
  help      显示帮助

示例:
  ./scripts/quick_start.sh check    # 检查系统
  ./scripts/quick_start.sh update   # 更新数据
  ./scripts/quick_start.sh email    # 测试邮件

EOF
}

# 检查系统
check_system() {
    print_header "🔍 系统健康检查"
    python3 scripts/system_check.py
}

# 更新数据
update_data() {
    print_header "📊 更新股票数据"
    python3 scripts/data_updater_robust.py
    print_success "数据更新完成"
}

# 测试邮件
test_email() {
    print_header "📧 测试邮件告警"
    python3 scripts/test_email_alert.py
}

# 备份配置
backup_config() {
    print_header "💾 备份配置文件"
    python3 scripts/config_manager.py backup
    print_success "配置已备份"
}

# 生成报告
generate_report() {
    print_header "📈 生成系统报告"

    echo -e "\n${YELLOW}📊 数据状态${NC}"
    python3 scripts/check_data_health.py

    echo -e "\n${YELLOW}📁 文件统计${NC}"
    echo "数据文件: $(ls data/*.csv 2>/dev/null | wc -l) 个"
    echo "脚本文件: $(ls scripts/*.py 2>/dev/null | wc -l) 个"
    echo "文档文件: $(ls docs/*.md 2>/dev/null | wc -l) 个"

    echo -e "\n${YELLOW}💾 磁盘使用${NC}"
    du -sh data/ scripts/ docs/ config/ 2>/dev/null

    print_success "报告生成完成"
}

# 心跳检查
heartbeat_check() {
    print_header "💓 心跳检查"
    python3 scripts/heartbeat_check_enhanced.py
}

# 主函数
main() {
    case "${1:-help}" in
        check)
            check_system
            ;;
        update)
            update_data
            ;;
        email)
            test_email
            ;;
        backup)
            backup_config
            ;;
        report)
            generate_report
            ;;
        heartbeat|hb)
            heartbeat_check
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知命令: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 运行
main "$@"
