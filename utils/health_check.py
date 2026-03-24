#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统健康检查
System Health Check

全面检查系统状态
"""

from utils.logger import get_logger

logger = get_logger(__name__)

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


class SystemHealthCheck:
    """系统健康检查"""

    def __init__(self):
        self.results = {"timestamp": datetime.now().isoformat(), "checks": {}, "status": "unknown"}

    def check_python_version(self):
        """检查Python版本"""
        version = sys.version_info

        if version.major >= 3 and version.minor >= 9:
            self.results["checks"]["python_version"] = {
                "status": "ok",
                "version": f"{version.major}.{version.minor}.{version.micro}",
            }
        else:
            self.results["checks"]["python_version"] = {
                "status": "error",
                "version": f"{version.major}.{version.minor}.{version.micro}",
                "message": "需要Python 3.9+",
            }

    def check_dependencies(self):
        """检查依赖"""
        required = ["pandas", "numpy", "yaml", "loguru"]
        missing = []

        for package in required:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)

        if not missing:
            self.results["checks"]["dependencies"] = {"status": "ok", "message": "所有依赖已安装"}
        else:
            self.results["checks"]["dependencies"] = {
                "status": "error",
                "message": f'缺少依赖: {", ".join(missing)}',
            }

    def check_data_files(self):
        """检查数据文件"""
        data_dir = Path(__file__).parent.parent / "data"

        required_files = ["auto_portfolio.json", "monitor_state.json"]

        missing = []
        for file in required_files:
            if not (data_dir / file).exists():
                missing.append(file)

        if not missing:
            self.results["checks"]["data_files"] = {"status": "ok", "message": "数据文件完整"}
        else:
            self.results["checks"]["data_files"] = {
                "status": "warning",
                "message": f'缺少文件: {", ".join(missing)}',
            }

    def check_data_freshness(self):
        """检查数据新鲜度"""
        from datetime import datetime, timedelta

        import pandas as pd

        data_dir = Path(__file__).parent.parent / "data"
        stock_files = sorted(data_dir.glob("real_*.csv"))  # 添加排序

        if not stock_files:
            self.results["checks"]["data_freshness"] = {
                "status": "error",
                "message": "未找到股票数据文件",
            }
            return

        # 检查第一个文件的最新日期
        try:
            df = pd.read_csv(stock_files[0])
            if "datetime" in df.columns or "trade_date" in df.columns:
                date_col = "datetime" if "datetime" in df.columns else "trade_date"

                # 智能解析日期（支持整数和字符串格式）
                sample = df[date_col].iloc[0]
                sample_dtype = str(df[date_col].dtype)

                if "int" in sample_dtype or (isinstance(sample, str) and sample.isdigit()):
                    df[date_col] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d")
                else:
                    df[date_col] = pd.to_datetime(df[date_col])

                latest_date = df[date_col].max()
                # 安全转换（直接解析日期字符串，避免nanosecond问题）
                latest_str = (
                    str(latest_date.date()) if hasattr(latest_date, "date") else str(latest_date)
                )
                latest_pydatetime = datetime.strptime(latest_str.split()[0], "%Y-%m-%d")
                age_days = (datetime.now() - latest_pydatetime).days

                if age_days > 7:
                    self.results["checks"]["data_freshness"] = {
                        "status": "error",
                        "message": f"数据过期{age_days}天（>7天）",
                        "latest_date": str(latest_date.date()),
                    }
                elif age_days > 3:
                    self.results["checks"]["data_freshness"] = {
                        "status": "warning",
                        "message": f"数据{age_days}天前更新",
                        "latest_date": str(latest_date.date()),
                    }
                else:
                    self.results["checks"]["data_freshness"] = {
                        "status": "ok",
                        "message": f"数据新鲜（{age_days}天前）",
                        "latest_date": str(latest_date.date()),
                    }
            else:
                self.results["checks"]["data_freshness"] = {
                    "status": "warning",
                    "message": "无法解析日期列",
                }
        except Exception as e:
            self.results["checks"]["data_freshness"] = {
                "status": "error",
                "message": f"检查失败: {str(e)[:50]}",
            }

    def check_config(self):
        """检查配置"""
        config_file = Path(__file__).parent.parent / "config" / "strategy_v4.yaml"

        if config_file.exists():
            self.results["checks"]["config"] = {"status": "ok", "message": "配置文件存在"}
        else:
            self.results["checks"]["config"] = {"status": "error", "message": "配置文件不存在"}

    def check_cron_jobs(self):
        """检查定时任务"""
        try:
            result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)

            if result.returncode == 0:
                cron_content = result.stdout

                if "run.py" in cron_content:
                    self.results["checks"]["cron"] = {"status": "ok", "message": "定时任务已配置"}
                else:
                    self.results["checks"]["cron"] = {
                        "status": "warning",
                        "message": "未找到run.py定时任务",
                    }
            else:
                self.results["checks"]["cron"] = {"status": "warning", "message": "未设置定时任务"}
        except Exception as e:
            self.results["checks"]["cron"] = {"status": "error", "message": f"检查失败: {e}"}

    def check_tests(self):
        """检查测试"""
        try:
            result = subprocess.run(
                ["pytest", "tests/", "-v", "--tb=no", "-q"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # 解析输出
            output = result.stdout

            if "passed" in output:
                self.results["checks"]["tests"] = {"status": "ok", "message": "测试通过"}
            else:
                self.results["checks"]["tests"] = {"status": "warning", "message": "部分测试失败"}
        except Exception as e:
            self.results["checks"]["tests"] = {"status": "error", "message": f"测试失败: {e}"}

    def run_all_checks(self):
        """运行所有检查"""
        print("\n" + "=" * 50)
        logger.info("🏥 系统健康检查")
        print("=" * 50 + "\n")

        # 执行检查
        self.check_python_version()
        self.check_dependencies()
        self.check_data_files()
        self.check_data_freshness()  # 新增
        self.check_config()
        self.check_cron_jobs()
        self.check_tests()

        # 计算总体状态
        statuses = [c["status"] for c in self.results["checks"].values()]

        if all(s == "ok" for s in statuses):
            self.results["status"] = "healthy"
        elif any(s == "error" for s in statuses):
            self.results["status"] = "unhealthy"
        else:
            self.results["status"] = "warning"

        # 打印结果
        for check_name, check_result in self.results["checks"].items():
            status_emoji = {"ok": "✅", "warning": "⚠️", "error": "❌"}

            emoji = status_emoji.get(check_result["status"], "❓")
            message = check_result.get("message", "No message")
            logger.info("{emoji} {check_name}: {message}")

        print("\n" + "=" * 50)
        logger.info("总体状态: {self.results['status'].upper()}")
        print("=" * 50 + "\n")

        return self.results


def main():
    """主函数"""
    checker = SystemHealthCheck()
    results = checker.run_all_checks()

    # 保存结果
    output_file = Path(__file__).parent.parent / "logs" / "health_check.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("结果已保存: {output_file}")

    return 0 if results["status"] == "healthy" else 1


if __name__ == "__main__":
    sys.exit(main())
