#!/usr/bin/env python3
"""
使用配置管理替换硬编码
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def update_data_manager():
    """更新 data_manager.py 使用配置"""
    
    file_path = PROJECT_ROOT / "core" / "data_manager.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 添加配置导入
    if 'from config.settings import Settings' not in content:
        # 在 logger 导入后添加
        content = re.sub(
            r'(from utils\.logger import get_logger\nlogger = get_logger\(__name__\))',
            r'\1\n\nfrom config.settings import Settings',
            content
        )
    
    # 替换硬编码的 Redis 配置
    content = re.sub(
        r"self\.redis_client = redis\.Redis\(host='localhost', port=6379, db=0\)",
        r"self.redis_client = redis.Redis(\n                host=Settings.REDIS_HOST,\n                port=Settings.REDIS_PORT,\n                db=Settings.REDIS_DB\n            )",
        content
    )
    
    # 替换硬编码的缓存目录
    content = re.sub(
        r'self\.file_cache_dir = PROJECT_ROOT / "data" / "cache"',
        r'self.file_cache_dir = Path(Settings.FILE_CACHE_DIR)',
        content
    )
    
    # 替换硬编码的缓存时间
    content = re.sub(
        r'max_age_minutes: int = 30\)',
        r'max_age_minutes: int = Settings.CACHE_MAX_AGE_MINUTES)',
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("   ✅ 更新 core/data_manager.py")

def update_analysts():
    """更新分析师脚本使用配置"""
    
    analysts = [
        "agents/buffett_analyst.py",
        "agents/growth_analyst.py",
        "agents/technical_analyst.py",
        "agents/fundamentals_analyst.py",
        "agents/sentiment_analyst.py",
        "agents/risk_manager.py",
    ]
    
    for analyst_path in analysts:
        file_path = PROJECT_ROOT / analyst_path
        
        if not file_path.exists():
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 添加配置导入（如果还没有）
        if 'from config.settings import Settings' not in content:
            content = re.sub(
                r'(from utils\.logger import get_logger\nlogger = get_logger\(__name__\))',
                r'\1\n\nfrom config.settings import Settings',
                content
            )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"   ✅ 更新 {analyst_path}")

def main():
    """主函数"""
    print("======================================================================")
    print("⚙️  使用配置管理替换硬编码")
    print("======================================================================")
    print()
    
    print("1️⃣ 更新核心模块...")
    update_data_manager()
    print()
    
    print("2️⃣ 更新分析师脚本...")
    update_analysts()
    print()
    
    print("======================================================================")
    print("✅ 配置管理优化完成")
    print("======================================================================")

if __name__ == "__main__":
    main()
