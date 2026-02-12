"""
配置管理模块
加载和管理系统配置
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from loguru import logger


class Config:
    """配置管理类"""

    def __init__(self, config_path: str = None):
        """
        初始化配置

        Args:
            config_path: 配置文件路径，默认为 config/config.yaml
        """
        if config_path is None:
            # 默认配置路径
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        """加载配置文件"""
        if not self.config_path.exists():
            # 如果配置文件不存在，尝试使用示例配置
            example_path = self.config_path.parent / "config.example.yaml"
            if example_path.exists():
                logger.warning(f"配置文件不存在，使用示例配置: {self.config_path}")
                logger.warning(f"请复制示例配置并修改: cp {example_path} {self.config_path}")
                self.config_path = example_path
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
            else:
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        else:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)

        # 加载环境变量覆盖
        self._load_env_overrides()

    def _load_env_overrides(self):
        """从环境变量加载配置覆盖"""
        # LLM API密钥
        if os.getenv('OPENAI_API_KEY'):
            self.config['llm']['openai']['api_key'] = os.getenv('OPENAI_API_KEY')
        if os.getenv('ZHIPUAI_API_KEY'):
            self.config['llm']['zhipuai']['api_key'] = os.getenv('ZHIPUAI_API_KEY')

        # Tushare token
        if os.getenv('TUSHARE_TOKEN'):
            self.config['data']['tushare']['token'] = os.getenv('TUSHARE_TOKEN')

        # 数据库密码
        if os.getenv('DB_PASSWORD'):
            self.config['database']['postgresql']['password'] = os.getenv('DB_PASSWORD')

    def get(self, *keys, default=None) -> Any:
        """
        获取配置值

        Args:
            *keys: 配置键的路径，如 'llm', 'openai', 'api_key'
            default: 默认值

        Returns:
            配置值
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value

    def set(self, *keys, value: Any):
        """
        设置配置值

        Args:
            *keys: 配置键的路径
            value: 要设置的值
        """
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value

    def save(self):
        """保存配置到文件"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"配置已保存到: {self.config_path}")

    def __repr__(self) -> str:
        return f"Config(path={self.config_path})"


# 全局配置实例
_config: Config = None


def get_config() -> Config:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config():
    """重新加载配置"""
    global _config
    _config = Config()
    logger.info("配置已重新加载")
