#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件
Configuration File
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """配置类"""
    
    # TuShare API Token
    TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN', '')
    
    # 项目路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    BACKUPS_DIR = os.path.join(BASE_DIR, 'backups')
    
    # 交易参数
    INITIAL_CAPITAL = 100000  # 初始资金10万
    
    # 风险参数
    MAX_POSITION_PCT = 0.15  # 单股最大仓位15%
    MAX_DRAWDOWN = 0.10      # 最大回撤10%
    STOP_LOSS = 0.05         # 止损5%
    
    @classmethod
    def ensure_directories(cls):
        """确保目录存在"""
        for directory in [cls.DATA_DIR, cls.LOGS_DIR, cls.BACKUPS_DIR]:
            os.makedirs(directory, exist_ok=True)
