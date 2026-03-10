"""
系统可靠性增强模块
==================
错误处理、日志记录、数据验证、备份机制
"""
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import traceback
import sys
import pandas as pd

class SystemReliability:
    """系统可靠性管理"""

    def __init__(self):
        self.log_dir = Path('logs')
        self.backup_dir = Path('backups')
        self.data_dir = Path('data')

        # 创建必要目录
        self.log_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)

        # 配置日志
        self.setup_logging()

        # 错误计数器
        self.error_count = 0
        self.max_errors = 5

    def setup_logging(self):
        """配置详细日志系统"""
        log_file = self.log_dir / f'system_{datetime.now().strftime("%Y%m%d")}.log'

        # 创建logger
        self.logger = logging.getLogger('AutoTrading')
        self.logger.setLevel(logging.DEBUG)

        # 清除现有handlers
        self.logger.handlers.clear()

        # 文件handler
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)

        # 控制台handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.logger.info("="*70)
        self.logger.info("系统可靠性模块启动")
        self.logger.info("="*70)

    def validate_data(self, data: pd.DataFrame, stock_code: str) -> bool:
        """验证股票数据完整性"""
        try:
            # 检查必要列
            required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                self.logger.error(f"{stock_code}: 缺少列 {missing}")
                return False

            # 检查数据量
            if len(data) < 100:
                self.logger.warning(f"{stock_code}: 数据量不足 ({len(data)}行)")
                return False

            # 检查空值
            null_count = data[required_cols].isnull().sum().sum()
            if null_count > 0:
                self.logger.warning(f"{stock_code}: 发现{null_count}个空值")

            # 检查价格合理性
            if (data['close'] <= 0).any():
                self.logger.error(f"{stock_code}: 发现无效价格（≤0）")
                return False

            self.logger.info(f"{stock_code}: 数据验证通过 ✅")
            return True

        except Exception as e:
            self.logger.error(f"{stock_code}: 数据验证失败 - {e}")
            return False

    def backup_portfolio(self):
        """备份持仓文件"""
        try:
            portfolio_file = self.data_dir / 'auto_portfolio.json'
            if not portfolio_file.exists():
                return

            # 创建备份文件名（带时间戳）
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.backup_dir / f'portfolio_{timestamp}.json'

            # 复制文件
            shutil.copy2(portfolio_file, backup_file)

            # 清理旧备份（保留最近30天）
            self.cleanup_old_backups(days=30)

            self.logger.info(f"持仓已备份: {backup_file}")

        except Exception as e:
            self.logger.error(f"备份失败: {e}")

    def cleanup_old_backups(self, days: int = 30):
        """清理旧备份文件"""
        try:
            cutoff = datetime.now() - timedelta(days=days)
            count = 0

            for backup_file in self.backup_dir.glob('portfolio_*.json'):
                # 从文件名提取时间戳
                try:
                    timestamp_str = backup_file.stem.split('_')[1] + '_' + backup_file.stem.split('_')[2]
                    file_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')

                    if file_time < cutoff:
                        backup_file.unlink()
                        count += 1
                except:
                    continue

            if count > 0:
                self.logger.info(f"清理了{count}个旧备份文件")

        except Exception as e:
            self.logger.error(f"清理备份失败: {e}")

    def safe_execute(self, func, *args, **kwargs) -> Optional[Any]:
        """安全执行函数，带错误处理"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"执行错误 ({self.error_count}/{self.max_errors}): {e}")
            self.logger.error(traceback.format_exc())

            # 保存错误信息
            self.save_error(e, func.__name__)

            # 如果错误次数过多，发送告警
            if self.error_count >= self.max_errors:
                self.send_alert(f"系统错误次数过多 ({self.error_count}次)")

            return None

    def save_error(self, error: Exception, function_name: str):
        """保存错误信息"""
        try:
            error_file = self.log_dir / 'errors.json'

            errors = []
            if error_file.exists():
                with open(error_file, 'r', encoding='utf-8') as f:
                    errors = json.load(f)

            errors.append({
                'time': datetime.now().isoformat(),
                'function': function_name,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': traceback.format_exc()
            })

            # 只保留最近100个错误
            errors = errors[-100:]

            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(errors, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.logger.error(f"保存错误信息失败: {e}")

    def send_alert(self, message: str):
        """发送告警（可扩展为邮件/微信/钉钉等）"""
        self.logger.warning(f"🚨 系统告警: {message}")

        # 保存告警记录
        try:
            alert_file = self.log_dir / 'alerts.json'

            alerts = []
            if alert_file.exists():
                with open(alert_file, 'r', encoding='utf-8') as f:
                    alerts = json.load(f)

            alerts.append({
                'time': datetime.now().isoformat(),
                'message': message
            })

            # 只保留最近50个告警
            alerts = alerts[-50:]

            with open(alert_file, 'w', encoding='utf-8') as f:
                json.dump(alerts, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.logger.error(f"保存告警失败: {e}")

    def health_check(self) -> Dict[str, Any]:
        """系统健康检查"""
        health = {
            'time': datetime.now().isoformat(),
            'status': 'healthy',
            'checks': {}
        }

        try:
            # 1. 检查数据文件
            data_files = list(self.data_dir.glob('real_*.csv'))
            health['checks']['data_files'] = {
                'status': 'ok' if len(data_files) >= 4 else 'warning',
                'count': len(data_files)
            }

            # 2. 检查持仓文件
            portfolio_file = self.data_dir / 'auto_portfolio.json'
            health['checks']['portfolio'] = {
                'status': 'ok' if portfolio_file.exists() else 'error',
                'exists': portfolio_file.exists()
            }

            # 3. 检查日志目录
            health['checks']['logs'] = {
                'status': 'ok' if self.log_dir.exists() else 'error',
                'exists': self.log_dir.exists()
            }

            # 4. 检查备份目录
            backup_count = len(list(self.backup_dir.glob('*.json')))
            health['checks']['backups'] = {
                'status': 'ok' if backup_count > 0 else 'warning',
                'count': backup_count
            }

            # 5. 检查错误计数
            health['checks']['errors'] = {
                'status': 'ok' if self.error_count < self.max_errors else 'error',
                'count': self.error_count,
                'max': self.max_errors
            }

            # 计算总体状态
            statuses = [check['status'] for check in health['checks'].values()]
            if 'error' in statuses:
                health['status'] = 'error'
            elif 'warning' in statuses:
                health['status'] = 'warning'

        except Exception as e:
            health['status'] = 'error'
            health['error'] = str(e)

        return health

    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        try:
            # 读取持仓
            portfolio_file = self.data_dir / 'auto_portfolio.json'
            if portfolio_file.exists():
                with open(portfolio_file, 'r', encoding='utf-8') as f:
                    portfolio = json.load(f)
                trades_count = len(portfolio.get('trades', []))
                positions_count = len(portfolio.get('positions', {}))
            else:
                trades_count = 0
                positions_count = 0

            # 备份统计
            backup_count = len(list(self.backup_dir.glob('*.json')))

            # 错误统计
            error_file = self.log_dir / 'errors.json'
            if error_file.exists():
                with open(error_file, 'r', encoding='utf-8') as f:
                    errors = json.load(f)
                errors_count = len(errors)
            else:
                errors_count = 0

            return {
                'trades_count': trades_count,
                'positions_count': positions_count,
                'backup_count': backup_count,
                'errors_count': errors_count,
                'current_errors': self.error_count
            }

        except Exception as e:
            self.logger.error(f"获取统计信息失败: {e}")
            return {}
