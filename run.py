#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速运行脚本
Quick Run Script

一键运行交易引擎
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from trading.engine import TradingEngine
from loguru import logger


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='量化交易引擎')
    parser.add_argument('--auto-trade', action='store_true', help='自动交易')
    parser.add_argument('--config', type=str, help='配置文件路径')
    
    args = parser.parse_args()
    
    # 创建交易引擎
    engine = TradingEngine(config_path=args.config)
    
    # 运行
    results = engine.run(auto_trade=args.auto_trade)
    
    # 保存结果
    import json
    output_path = Path(__file__).parent / 'data' / 'daily_analysis_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.success(f"✅ 分析完成! 结果已保存: {output_path}")


if __name__ == '__main__':
    main()
