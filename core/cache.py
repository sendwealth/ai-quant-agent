#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据缓存模块
Data Cache Module

提供高效的数据缓存机制，减少重复计算和I/O
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib
import pickle
import json
from datetime import datetime, timedelta
from loguru import logger


class DataCache:
    """
    数据缓存器
    
    功能:
    - 缓存计算结果
    - 缓存加载的数据
    - 自动过期清理
    """
    
    def __init__(self, cache_dir: str = 'cache', expire_hours: int = 24):
        """
        初始化缓存器
        
        Args:
            cache_dir: 缓存目录
            expire_hours: 过期时间(小时)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.expire_hours = expire_hours
        
        # 缓存统计
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        logger.info(f"缓存器初始化: {self.cache_dir}")
    
    def _hash_key(self, key: str) -> str:
        """生成缓存键的hash"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        hash_key = self._hash_key(key)
        return self.cache_dir / f"{hash_key}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存
        
        Args:
            key: 缓存键
        
        Returns:
            缓存数据，如果不存在或过期则返回None
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            self.stats['misses'] += 1
            return None
        
        try:
            # 检查是否过期
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if datetime.now() - mtime > timedelta(hours=self.expire_hours):
                # 过期，删除
                cache_path.unlink()
                self.stats['evictions'] += 1
                self.stats['misses'] += 1
                return None
            
            # 加载缓存
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            self.stats['hits'] += 1
            logger.debug(f"缓存命中: {key}")
            return data
            
        except Exception as e:
            logger.warning(f"缓存读取失败: {e}")
            self.stats['misses'] += 1
            return None
    
    def set(self, key: str, data: Any):
        """
        设置缓存
        
        Args:
            key: 缓存键
            data: 缓存数据
        """
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.debug(f"缓存保存: {key}")
            
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    def delete(self, key: str):
        """删除缓存"""
        cache_path = self._get_cache_path(key)
        
        if cache_path.exists():
            cache_path.unlink()
            logger.debug(f"缓存删除: {key}")
    
    def clear(self):
        """清空所有缓存"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        
        logger.info("缓存已清空")
    
    def cleanup_expired(self):
        """清理过期缓存"""
        count = 0
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            
            if datetime.now() - mtime > timedelta(hours=self.expire_hours):
                cache_file.unlink()
                count += 1
        
        if count > 0:
            logger.info(f"清理过期缓存: {count}个")
    
    def get_stats(self) -> dict:
        """获取缓存统计"""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total if total > 0 else 0
        
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'evictions': self.stats['evictions'],
            'hit_rate': hit_rate,
            'total_requests': total
        }
    
    def get_size(self) -> dict:
        """获取缓存大小"""
        total_size = 0
        file_count = 0
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            total_size += cache_file.stat().st_size
            file_count += 1
        
        return {
            'file_count': file_count,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024)
        }


class IndicatorCache:
    """
    技术指标缓存
    
    专门用于缓存计算的技术指标
    """
    
    def __init__(self, cache: DataCache):
        """
        初始化指标缓存
        
        Args:
            cache: DataCache实例
        """
        self.cache = cache
    
    def _make_key(self, code: str, indicator: str, params: dict) -> str:
        """生成缓存键"""
        params_str = json.dumps(params, sort_keys=True)
        return f"{code}_{indicator}_{params_str}"
    
    def get_indicator(self, 
                     code: str, 
                     indicator: str, 
                     params: dict) -> Optional[pd.Series]:
        """
        获取指标缓存
        
        Args:
            code: 股票代码
            indicator: 指标名称
            params: 指标参数
        
        Returns:
            指标数据
        """
        key = self._make_key(code, indicator, params)
        return self.cache.get(key)
    
    def set_indicator(self,
                     code: str,
                     indicator: str,
                     params: dict,
                     data: pd.Series):
        """
        设置指标缓存
        
        Args:
            code: 股票代码
            indicator: 指标名称
            params: 指标参数
            data: 指标数据
        """
        key = self._make_key(code, indicator, params)
        self.cache.set(key, data)


# ========== 全局缓存实例 ==========
_global_cache = None
_global_indicator_cache = None


def get_cache() -> DataCache:
    """获取全局缓存实例"""
    global _global_cache
    
    if _global_cache is None:
        cache_dir = Path(__file__).parent.parent / 'cache'
        _global_cache = DataCache(str(cache_dir))
    
    return _global_cache


def get_indicator_cache() -> IndicatorCache:
    """获取全局指标缓存实例"""
    global _global_indicator_cache
    
    if _global_indicator_cache is None:
        _global_indicator_cache = IndicatorCache(get_cache())
    
    return _global_indicator_cache


# ========== 导出 __all__ = [
    'DataCache',
    'IndicatorCache',
    'get_cache',
    'get_indicator_cache',
]
