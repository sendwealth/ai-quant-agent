"""
A股数据获取模块
使用AkShare和Tushare获取A股数据
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime, timedelta

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("⚠️  AkShare未安装，使用模拟数据")

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    print("⚠️  Tushare未安装，使用模拟数据")


class AStockDataFetcher:
    """A股数据获取器"""

    def __init__(self, tushare_token: str = None):
        """
        初始化

        Args:
            tushare_token: Tushare API Token
        """
        self.tushare_token = tushare_token

        if TUSHARE_AVAILABLE and tushare_token:
            try:
                ts.set_token(tushare_token)
                self.ts_pro = ts.pro_api()
                print("✓ Tushare API 已连接")
            except Exception as e:
                print(f"⚠️  Tushare连接失败: {e}")
                self.ts_pro = None
        else:
            self.ts_pro = None

        if AKSHARE_AVAILABLE:
            print("✓ AkShare 已就绪")

    def fetch_stock_daily(self,
                          symbol: str,
                          start_date: str = None,
                          end_date: str = None,
                          source: str = 'akshare') -> pd.DataFrame:
        """
        获取A股日线数据

        Args:
            symbol: 股票代码（如 '000001', '600000'）
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            source: 数据源 ('akshare' 或 'tushare')

        Returns:
            DataFrame (包含open, high, low, close, volume)
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d')

        print(f"获取A股数据: {symbol} ({start_date} -> {end_date})")

        try:
            if source == 'akshare' and AKSHARE_AVAILABLE:
                return self._fetch_akshare(symbol, start_date, end_date)
            elif source == 'tushare' and self.ts_pro:
                return self._fetch_tushare(symbol, start_date, end_date)
            else:
                return self._generate_mock_data(symbol, start_date, end_date)

        except Exception as e:
            print(f"⚠️  数据获取失败: {e}")
            print("使用模拟数据...")
            return self._generate_mock_data(symbol, start_date, end_date)

    def _fetch_akshare(self,
                        symbol: str,
                        start_date: str,
                        end_date: str) -> pd.DataFrame:
        """使用AkShare获取数据"""
        try:
            # AkShare stock_zh_a_hist接口
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )

            # 标准化列名
            df = df.rename(columns={
                '日期': 'datetime',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume'
            })

            # 保留需要的列
            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

            # 数据清洗
            df = df.dropna()
            df = df.sort_values('datetime').reset_index(drop=True)

            # 确保数值类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            print(f"✓ 使用AkShare获取成功: {len(df)}条记录")
            return df

        except Exception as e:
            print(f"⚠️  AkShare获取失败: {e}")
            raise

    def _fetch_tushare(self,
                        symbol: str,
                        start_date: str,
                        end_date: str) -> pd.DataFrame:
        """使用Tushare获取数据"""
        try:
            # Tushare daily接口
            df = self.ts_pro.daily(
                ts_code=symbol,
                start_date=start_date,
                end_date=end_date
            )

            # 标准化列名
            df = df.rename(columns={
                'trade_date': 'datetime',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume'
            })

            # 日期格式转换
            df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d')

            # 保留需要的列并排序
            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('datetime').reset_index(drop=True)

            # 数据清洗
            df = df.dropna()
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            print(f"✓ 使用Tushare获取成功: {len(df)}条记录")
            return df

        except Exception as e:
            print(f"⚠️  Tushare获取失败: {e}")
            raise

    def _generate_mock_data(self,
                             symbol: str,
                             start_date: str,
                             end_date: str) -> pd.DataFrame:
        """生成模拟数据"""
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        dates = pd.date_range(start_dt, end_dt, freq='D')
        # 过滤周末
        dates = [d for d in dates if d.weekday() < 5]

        np.random.seed(hash(symbol) % 10000)
        base_price = np.random.uniform(5, 50)

        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * (1 + np.cumsum(returns))

        df = pd.DataFrame({
            'datetime': dates,
            'open': prices * (1 + np.random.randn(len(dates)) * 0.005),
            'high': prices * (1 + np.abs(np.random.randn(len(dates))) * 0.01),
            'low': prices * (1 - np.abs(np.random.randn(len(dates))) * 0.01),
            'close': prices,
            'volume': np.random.randint(100000, 10000000, len(dates))
        })

        print(f"✓ 生成模拟数据: {len(df)}条记录")
        return df

    def get_stock_list(self, market: str = '沪深') -> pd.DataFrame:
        """
        获取股票列表

        Args:
            market: 市场 ('沪深', '沪', '深')

        Returns:
            股票列表DataFrame
        """
        if AKSHARE_AVAILABLE:
            try:
                df = ak.stock_zh_a_spot_em()
                print(f"✓ 获取股票列表: {len(df)}只")
                return df
            except Exception as e:
                print(f"⚠️  获取股票列表失败: {e}")

        return pd.DataFrame()

    def get_hot_stocks(self, n: int = 20) -> List[str]:
        """
        获取热门股票

        Args:
            n: 返回数量

        Returns:
            股票代码列表
        """
        if AKSHARE_AVAILABLE:
            try:
                # 获取涨跌幅榜
                df = ak.stock_zh_a_spot_em()
                # 按涨跌幅排序
                df_sorted = df.sort_values('涨跌幅', ascending=False)
                # 取前n个
                hot_stocks = df_sorted['代码'].head(n).tolist()
                print(f"✓ 获取热门股票: {len(hot_stocks)}只")
                return hot_stocks
            except Exception as e:
                print(f"⚠️  获取热门股票失败: {e}")

        # 默认热门股票
        return ['000001', '000002', '600000', '600036', '601318',
                '600519', '000858', '002594', '600276', '601012']

    def fetch_multiple_stocks(self,
                                symbols: List[str],
                                start_date: str = None,
                                end_date: str = None,
                                source: str = 'akshare') -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            source: 数据源

        Returns:
            字典 {symbol: DataFrame}
        """
        data_dict = {}

        for symbol in symbols:
            try:
                df = self.fetch_stock_daily(symbol, start_date, end_date, source)
                data_dict[symbol] = df
                print(f"✓ {symbol}: {len(df)}条记录")
            except Exception as e:
                print(f"✗ {symbol}: 获取失败 ({e})")

        return data_dict


def get_popular_astocks() -> List[str]:
    """
    获取热门A股列表

    Returns:
        股票代码列表
    """
    return [
        '000001',  # 平安银行
        '000002',  # 万科A
        '000858',  # 五粮液
        '002594',  # 比亚迪
        '600000',  # 浦发银行
        '600036',  # 招商银行
        '600519',  # 贵州茅台
        '600276',  # 恒瑞医药
        '601318',  # 中国平安
        '601012',  # 隆基绿能
        '600900',  # 长江电力
        '000725',  # 京东方A
        '002415',  # 海康威视
        '601888',  # 中国中免
        '600887',  # 伊利股份
    ]


if __name__ == "__main__":
    # 测试A股数据获取
    print("="*70)
    print("A股数据获取测试")
    print("="*70)

    fetcher = AStockDataFetcher()

    # 测试获取单只股票
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

    print(f"\n测试1: 获取贵州茅台 (600519)")
    df = fetcher.fetch_stock_daily('600519', start_date, end_date, source='akshare')

    if df is not None and len(df) > 0:
        print(f"\n数据预览:")
        print(df.head())
        print(f"\n数据统计:")
        print(df.describe())
        print(f"\n数据行数: {len(df)}")

    # 测试获取热门股票
    print(f"\n\n测试2: 获取热门股票")
    hot_stocks = get_popular_astocks()[:5]
    print(f"热门股票: {hot_stocks}")

    data_dict = fetcher.fetch_multiple_stocks(
        hot_stocks,
        start_date,
        end_date,
        source='akshare'
    )

    print(f"\n成功获取: {len(data_dict)}只股票")
