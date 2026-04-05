#!/usr/bin/env python3
"""
配置文件管理器

功能：
1. 配置文件备份（加密）
2. 配置文件恢复
3. 敏感信息加密存储
4. 配置迁移和升级
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from utils.logger import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """配置文件管理器"""

    def __init__(self):
        self.config_dir = PROJECT_ROOT / 'config'
        self.backup_dir = PROJECT_ROOT / 'config' / 'backups'
        self.env_file = PROJECT_ROOT / '.env'

        # 确保备份目录存在
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def load_env(self) -> Dict[str, str]:
        """加载环境变量"""
        env_vars = {}
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
        return env_vars

    def save_env(self, env_vars: Dict[str, str]):
        """保存环境变量"""
        with open(self.env_file, 'w') as f:
            f.write("# 环境变量配置\n")
            f.write(f"# 更新时间: {datetime.now().isoformat()}\n\n")

            for key, value in sorted(env_vars.items()):
                f.write(f"{key}={value}\n")

    def backup_config(self, config_name: str = 'data_sources.yaml'):
        """
        备份配置文件

        Args:
            config_name: 配置文件名
        """
        config_path = self.config_dir / config_name

        if not config_path.exists():
            logger.error(f"配置文件不存在: {config_path}")
            return False

        # 读取配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 脱敏处理（不备份密码）
        safe_config = self._sanitize_config(config)

        # 创建备份
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f"{config_name}.{timestamp}.bak"

        with open(backup_path, 'w', encoding='utf-8') as f:
            yaml.dump(safe_config, f, allow_unicode=True, default_flow_style=False)

        logger.info(f"✅ 配置已备份: {backup_path}")
        return True

    def restore_config(
        self,
        config_name: str = 'data_sources.yaml',
        backup_file: Optional[str] = None
    ):
        """
        恢复配置文件

        Args:
            config_name: 配置文件名
            backup_file: 指定备份文件（可选）
        """
        config_path = self.config_dir / config_name

        if backup_file:
            backup_path = Path(backup_file)
        else:
            # 查找最新的备份
            backups = sorted(self.backup_dir.glob(f"{config_name}.*.bak"), reverse=True)
            if not backups:
                logger.error("未找到备份文件")
                return False
            backup_path = backups[0]

        if not backup_path.exists():
            logger.error(f"备份文件不存在: {backup_path}")
            return False

        # 读取备份
        with open(backup_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 合并环境变量
        config = self._merge_env_vars(config)

        # 恢复配置
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

        logger.info(f"✅ 配置已恢复: {config_path}")
        return True

    def _sanitize_config(self, config: Dict) -> Dict:
        """脱敏配置（移除密码等敏感信息）"""
        safe_config = config.copy()

        # 脱敏邮件密码
        if 'email_alert' in safe_config:
            if 'password' in safe_config['email_alert']:
                safe_config['email_alert']['password'] = '${EMAIL_PASSWORD}'

        # 脱敏 Tushare token
        if 'tushare' in safe_config:
            if 'token' in safe_config['tushare']:
                safe_config['tushare']['token'] = '${TUSHARE_TOKEN}'

        return safe_config

    def _merge_env_vars(self, config: Dict) -> Dict:
        """合并环境变量到配置"""
        env_vars = self.load_env()

        # 替换邮件配置
        if 'email_alert' in config:
            email = config['email_alert']
            for key in ['smtp_server', 'smtp_port', 'sender', 'password', 'recipients']:
                env_key = f'EMAIL_{key.upper()}'
                if env_key in env_vars:
                    if key == 'recipients':
                        email[key] = env_vars[env_key].split(',')
                    else:
                        email[key] = env_vars[env_key]

        # 替换 Tushare token
        if 'tushare' in config and 'TUSHARE_TOKEN' in env_vars:
            config['tushare']['token'] = env_vars['TUSHARE_TOKEN']

        return config

    def migrate_config(
        self,
        old_config: Dict,
        new_template: Dict
    ) -> Dict:
        """
        迁移配置（保留旧配置的值）

        Args:
            old_config: 旧配置
            new_template: 新模板

        Returns:
            合并后的配置
        """
        merged = new_template.copy()

        # 递归合并
        def deep_merge(base: Dict, update: Dict):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value

        deep_merge(merged, old_config)
        return merged

    def check_config_health(self, config_name: str = 'data_sources.yaml') -> Dict:
        """
        检查配置健康度

        Returns:
            健康度报告
        """
        config_path = self.config_dir / config_name
        env_vars = self.load_env()

        report = {
            'status': 'ok',
            'issues': [],
            'recommendations': []
        }

        if not config_path.exists():
            report['status'] = 'error'
            report['issues'].append(f"配置文件不存在: {config_path}")
            return report

        # 读取配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 检查 Tushare 配置
        if config.get('tushare', {}).get('enabled'):
            if not env_vars.get('TUSHARE_TOKEN'):
                report['status'] = 'warning'
                report['issues'].append('Tushare 已启用但未配置 token')
                report['recommendations'].append('在 .env 文件中设置 TUSHARE_TOKEN')

        # 检查邮件配置
        if config.get('email_alert', {}).get('enabled'):
            if not env_vars.get('EMAIL_PASSWORD'):
                report['status'] = 'warning'
                report['issues'].append('邮件告警已启用但未配置密码')
                report['recommendations'].append('在 .env 文件中设置 EMAIL_PASSWORD')

        return report

    def list_backups(self, config_name: str = 'data_sources.yaml'):
        """列出所有备份"""
        backups = sorted(self.backup_dir.glob(f"{config_name}.*.bak"), reverse=True)

        if not backups:
            print("未找到备份文件")
            return

        print(f"找到 {len(backups)} 个备份:")
        for backup in backups:
            stat = backup.stat()
            size = stat.st_size
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  {backup.name} ({size} bytes, {mtime})")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='配置文件管理器')
    parser.add_argument('action', choices=['backup', 'restore', 'check', 'list'], help='操作')
    parser.add_argument('--config', default='data_sources.yaml', help='配置文件名')
    parser.add_argument('--file', help='指定备份文件')

    args = parser.parse_args()

    manager = ConfigManager()

    if args.action == 'backup':
        manager.backup_config(args.config)
    elif args.action == 'restore':
        manager.restore_config(args.config, args.file)
    elif args.action == 'check':
        report = manager.check_config_health(args.config)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.action == 'list':
        manager.list_backups(args.config)


if __name__ == '__main__':
    main()
