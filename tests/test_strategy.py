#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略测试
Strategy Tests
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_strategy import MAStrategy, StrategyFactory
from core.config_loader import ConfigLoader


class TestMAStrategy:
    """MA策略测试"""
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        np.random.seed(42)
        
        # 生成100天的模拟数据
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # 价格趋势: 上升趋势
        trend = np.linspace(100, 120, 100)
        noise = np.random.randn(100) * 2
        
        close = trend + noise
        high = close + np.random.rand(100) * 3
        low = close - np.random.rand(100) * 3
        open_price = close + np.random.randn(100) * 1
        volume = np.random.randint(1000, 10000, 100)
        
        return pd.DataFrame({
            'datetime': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }).set_index('datetime')
    
    @pytest.fixture
    def strategy_config(self):
        """策略配置"""
        return {
            'name': 'TestMA',
            'weight': 0.25,
            'params': {
                'ma_fast': 10,
                'ma_slow': 30
            },
            'risk': {
                'stop_loss': -0.08,
                'take_profit_1': 0.10,
                'take_profit_2': 0.20
            }
        }
    
    def test_strategy_creation(self, strategy_config):
        """测试策略创建"""
        strategy = MAStrategy(strategy_config)
        
        assert strategy.name == 'TestMA'
        assert strategy.params['ma_fast'] == 10
        assert strategy.params['ma_slow'] == 30
    
    def test_generate_signals(self, sample_data, strategy_config):
        """测试信号生成"""
        strategy = MAStrategy(strategy_config)
        signals = strategy.generate_signals(sample_data)
        
        # 信号应该与数据长度相同
        assert len(signals) == len(sample_data)
        
        # 信号应该只包含 -1, 0, 1
        unique_signals = signals.unique()
        assert all(s in [-1, 0, 1] for s in unique_signals)
    
    def test_calculate_position_size(self, strategy_config):
        """测试仓位计算"""
        strategy = MAStrategy(strategy_config)
        
        capital = 100000
        price = 100
        
        shares = strategy.calculate_position_size(capital, price)
        
        # 权重0.25，资金10万，价格100
        # 应该买入: 100000 * 0.25 / 100 = 250股
        assert shares == 250
    
    def test_validate_data(self, sample_data, strategy_config):
        """测试数据验证"""
        strategy = MAStrategy(strategy_config)
        
        # 完整数据应该通过验证
        assert strategy.validate_data(sample_data)
        
        # 缺少列的数据应该失败
        incomplete_data = sample_data.drop('volume', axis=1)
        assert not strategy.validate_data(incomplete_data)
        
        # 空数据应该失败
        empty_data = pd.DataFrame()
        assert not strategy.validate_data(empty_data)
    
    def test_check_risk_stop_loss(self, strategy_config):
        """测试止损检查"""
        strategy = MAStrategy(strategy_config)
        
        position = {
            'entry_price': 100,
            'shares': 100
        }
        
        # 价格下跌9% (超过止损-8%)
        current_price = 91
        should_close, reason = strategy.check_risk(position, current_price)
        
        assert should_close == True
        assert '止损' in reason
    
    def test_check_risk_take_profit_1(self, strategy_config):
        """测试止盈1检查"""
        strategy = MAStrategy(strategy_config)
        
        position = {
            'entry_price': 100,
            'shares': 100
        }
        
        # 价格上涨11% (超过止盈1 +10%)
        current_price = 111
        should_close, reason = strategy.check_risk(position, current_price)
        
        assert should_close == True
        assert '止盈1' in reason
    
    def test_check_risk_take_profit_2(self, strategy_config):
        """测试止盈2检查"""
        strategy = MAStrategy(strategy_config)
        
        position = {
            'entry_price': 100,
            'shares': 100
        }
        
        # 价格上涨21% (超过止盈2 +20%)
        current_price = 121
        should_close, reason = strategy.check_risk(position, current_price)
        
        assert should_close == True
        assert '止盈2' in reason
    
    def test_check_risk_normal(self, strategy_config):
        """测试正常情况 (无需平仓)"""
        strategy = MAStrategy(strategy_config)
        
        position = {
            'entry_price': 100,
            'shares': 100
        }
        
        # 价格上涨5% (正常范围)
        current_price = 105
        should_close, reason = strategy.check_risk(position, current_price)
        
        assert should_close == False
        assert reason == ''
    
    def test_get_strategy_info(self, strategy_config):
        """测试获取策略信息"""
        strategy = MAStrategy(strategy_config)
        info = strategy.get_strategy_info()
        
        assert info['name'] == 'TestMA'
        assert 'params' in info
        assert 'config' in info


class TestStrategyFactory:
    """策略工厂测试"""
    
    def test_create_ma_strategy(self):
        """测试创建MA策略"""
        config = {
            'type': 'ma',
            'name': 'TestMA',
            'weight': 0.25,
            'params': {
                'ma_fast': 10,
                'ma_slow': 30
            }
        }
        
        strategy = StrategyFactory.create(config)
        
        assert isinstance(strategy, MAStrategy)
        assert strategy.name == 'TestMA'
    
    def test_create_unsupported_strategy(self):
        """测试创建不支持的策略"""
        config = {
            'type': 'unsupported',
            'name': 'TestStrategy'
        }
        
        with pytest.raises(ValueError):
            StrategyFactory.create(config)


class TestConfigLoader:
    """配置加载器测试"""
    
    @pytest.fixture
    def config_loader(self):
        """创建配置加载器"""
        config_path = Path(__file__).parent.parent / 'config' / 'strategy_v4.yaml'
        
        if config_path.exists():
            return ConfigLoader(str(config_path))
        else:
            pytest.skip("配置文件不存在")
    
    def test_load_config(self, config_loader):
        """测试加载配置"""
        assert config_loader.config is not None
        assert 'strategy' in config_loader.config
        assert 'stocks' in config_loader.config
    
    def test_get_strategy_config(self, config_loader):
        """测试获取策略配置"""
        strategy_config = config_loader.get_strategy_config()
        
        assert 'name' in strategy_config
        assert 'version' in strategy_config
    
    def test_get_stock_config(self, config_loader):
        """测试获取股票配置"""
        stock_config = config_loader.get_stock_config('300750')
        
        assert stock_config is not None
        assert stock_config['name'] == '宁德时代'
        assert stock_config['weight'] == 0.45
    
    def test_get_all_stocks(self, config_loader):
        """测试获取所有股票"""
        stocks = config_loader.get_all_stocks(enabled_only=True)
        
        assert len(stocks) > 0
        assert all(s.get('enabled', True) for s in stocks)
    
    def test_get_risk_params(self, config_loader):
        """测试获取风险参数"""
        risk_params = config_loader.get_risk_params()
        
        assert 'stop_loss' in risk_params
        assert 'take_profit_1' in risk_params
        assert 'take_profit_2' in risk_params
    
    def test_get_method(self, config_loader):
        """测试get方法"""
        # 获取嵌套配置
        name = config_loader.get('strategy', 'name')
        assert name == 'V4 Enhanced'
        
        # 获取不存在的配置
        value = config_loader.get('strategy', 'nonexistent', default='default')
        assert value == 'default'


# ========== 运行测试 ==========

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
