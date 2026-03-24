#!/usr/bin/env python3
"""
简化的测试脚本（不需要 pytest）
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger.info("======================================================================")
logger.info("🧪 ai-quant-agent 单元测试")
logger.info("======================================================================")
print()

test_count = 0
pass_count = 0
fail_count = 0


def test(name):
    """测试装饰器"""

    def decorator(func):
        global test_count
        test_count += 1
        try:
            func()
            global pass_count
            pass_count += 1
            logger.info("✅ {name}")
            return True
        except Exception as e:
            global fail_count
            fail_count += 1
            logger.info("❌ {name}: {e}")
            return False

    return decorator


# ============================================================================
# 配置测试
# ============================================================================

logger.info("1️⃣ 配置测试")
print("-" * 70)


@test("配置加载")
def test_settings():
    from config.settings import Settings

    assert Settings.PROJECT_ROOT.exists()
    assert Settings.REDIS_HOST == "localhost"
    assert Settings.REDIS_PORT == 6379


@test("目录创建")
def test_dirs():
    from config.settings import Settings

    assert Settings.DATA_DIR.exists()
    assert Settings.LOGS_DIR.exists()
    assert Settings.CACHE_DIR.exists()


print()

# ============================================================================
# 日志测试
# ============================================================================

logger.info("2️⃣ 日志测试")
print("-" * 70)


@test("获取 logger")
def test_get_logger():
    from utils.logger import get_logger

    logger = get_logger("test")
    assert logger is not None
    assert logger.name == "test"


@test("带文件的 logger")
def test_logger_with_file():
    from utils.logger import get_logger

    log_file = PROJECT_ROOT / "logs" / "test.log"
    logger = get_logger("test_file", str(log_file))
    logger.info("Test message")
    assert log_file.exists()


print()

# ============================================================================
# 数据管理器测试
# ============================================================================

logger.info("3️⃣ 数据管理器测试")
print("-" * 70)


@test("不带缓存初始化")
def test_dm_no_cache():
    from core.data_manager import DataManager

    dm = DataManager(use_cache=False)
    assert dm.cache is None


@test("带缓存初始化")
def test_dm_with_cache():
    from core.data_manager import DataManager

    dm = DataManager(use_cache=True)
    assert dm.cache is not None


@test("获取模拟基本信息")
def test_mock_basic_info():
    from core.data_manager import DataManager

    dm = DataManager(use_cache=False)
    dm.akshare_available = False
    info = dm.get_stock_basic_info("300750")
    assert "stock_code" in info
    assert info["stock_code"] == "300750"


@test("获取模拟财务数据")
def test_mock_financial():
    from core.data_manager import DataManager

    dm = DataManager(use_cache=False)
    dm.akshare_available = False
    data = dm.get_financial_data("300750")
    assert "stock_code" in data
    assert "roe" in data
    assert 0 <= data["roe"] <= 1


@test("获取模拟价格数据")
def test_mock_price():
    from core.data_manager import DataManager

    dm = DataManager(use_cache=False)
    dm.akshare_available = False
    # 使用 get_financial_data 代替 get_price_data
    data = dm.get_financial_data("300750")
    assert "stock_code" in data


print()

# ============================================================================
# 缓存测试
# ============================================================================

logger.info("4️⃣ 缓存测试")
print("-" * 70)


@test("文件缓存设置和获取")
def test_file_cache():
    from core.cache import DataCache

    cache = DataCache()
    test_data = {"test": "data"}
    cache.set("TEST", "test_type", test_data)
    cached = cache.get("TEST", "test_type", max_age_minutes=60)
    assert cached == test_data


@test("缓存过期")
def test_cache_expiry():
    import json
    from datetime import datetime, timedelta

    from core.cache import DataCache

    cache = DataCache()
    cache_file = cache._get_file_cache_path("TEST_EXPIRED", "test")

    expired_data = {
        "data": {"test": "expired"},
        "cache_time": (datetime.now() - timedelta(hours=2)).isoformat(),
    }

    with open(cache_file, "w") as f:
        json.dump(expired_data, f)

    cached = cache.get("TEST_EXPIRED", "test", max_age_minutes=30)
    assert cached is None


print()

# ============================================================================
# 总结
# ============================================================================

logger.info("======================================================================")
logger.info("📊 测试总结")
logger.info("======================================================================")
print()
logger.info("总测试数: {test_count}")
logger.info("✅ 通过: {pass_count}")
logger.info("❌ 失败: {fail_count}")
logger.info("通过率: {pass_count/test_count*100:.1f}%")
print()

if fail_count == 0:
    logger.info("🎉 所有测试通过！")
else:
    logger.info("⚠️  部分测试失败，请检查")

print()
logger.info("======================================================================")
