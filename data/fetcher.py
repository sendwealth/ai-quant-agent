"""
数据获取模块
从多个数据源获取历史和实时数据
"""

import yfinance as yf
import pandas as pd
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from loguru import logger


class DataFetcher:
    """数据获取器 - 支持多数据源"""

    def __init__(self, default_source: str = 'yfinance'):
        """
        初始化数据获取器

        Args:
            default_source: 默认数据源
        """
        self.default_source = default_source
        logger.info(f"数据获取器初始化，默认数据源: {default_source}")

    def fetch_stock_data(self,
                         symbol: str,
                         start_date: str,
                         end_date: str = None,
                         interval: str = '1d',
                         source: str = None) -> pd.DataFrame:
        """
        获取股票历史数据

        Args:
            symbol: 股票代码（如 'AAPL', 'SPY'）
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)，默认为今天
            interval: 数据间隔 ('1d', '1h', '5m')
            source: 数据源，默认使用default_source

        Returns:
            包含OHLCV数据的DataFrame
        """
        source = source or self.default_source

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"获取{source}数据: {symbol}, {start_date} -> {end_date}")

        try:
            if source == 'yfinance':
                df = self._fetch_yfinance(symbol, start_date, end_date, interval)
            else:
                raise ValueError(f"不支持的数据源: {source}")

            # 数据清洗
            df = self._clean_data(df)

            logger.info(f"成功获取{len(df)}条数据")
            return df

        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            raise

    def _fetch_yfinance(self,
                        symbol: str,
                        start_date: str,
                        end_date: str,
                        interval: str) -> pd.DataFrame:
        """使用yfinance获取数据"""
        ticker = yf.Ticker(symbol)

        # 获取历史数据
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,  # 自动调整价格（拆股、分红）
            prepost=False,  # 不包含盘前盘后
            actions=False  # 不包含分红、拆股等操作
        )

        # 标准化列名
        df.columns = [col.lower() for col in df.columns]

        # 重置索引
        df = df.reset_index()

        # 日期列重命名
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'datetime'})
        elif 'datetime' not in df.columns:
            df = df.rename(columns={df.columns[0]: 'datetime'})

        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据

        Args:
            df: 原始数据

        Returns:
            清洗后的数据
        """
        # 检查必要的列
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"缺少必要的列: {col}")

        # 删除缺失值
        df = df.dropna(subset=required_columns)

        # 确保数值类型
        for col in required_columns + ['volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 删除任何还有缺失值的行
        df = df.dropna()

        # 按日期排序
        if 'datetime' in df.columns:
            df = df.sort_values('datetime')

        # 重置索引
        df = df.reset_index(drop=True)

        return df

    def fetch_multiple_symbols(self,
                                symbols: List[str],
                                start_date: str,
                                end_date: str = None,
                                interval: str = '1d',
                                source: str = None) -> Dict[str, pd.DataFrame]:
        """
        批量获取多个股票的数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            interval: 数据间隔
            source: 数据源

        Returns:
            字典 {symbol: DataFrame}
        """
        source = source or self.default_source
        data_dict = {}

        for symbol in symbols:
            try:
                df = self.fetch_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    source=source
                )
                data_dict[symbol] = df
                logger.info(f"成功获取 {symbol} 数据: {len(df)}条")
            except Exception as e:
                logger.error(f"获取 {symbol} 失败: {e}")

        return data_dict

    def get_latest_price(self, symbol: str, source: str = None) -> float:
        """
        获取最新价格

        Args:
            symbol: 股票代码
            source: 数据源

        Returns:
            最新价格
        """
        source = source or self.default_source

        try:
            # 获取最近1天的数据
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')

            df = self.fetch_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval='1d',
                source=source
            )

            if len(df) > 0:
                latest_price = df['close'].iloc[-1]
                logger.info(f"{symbol} 最新价格: {latest_price:.2f}")
                return latest_price
            else:
                raise ValueError("没有获取到数据")

        except Exception as e:
            logger.error(f"获取最新价格失败: {e}")
            raise

    def get_available_intervals(self) -> List[str]:
        """
        获取支持的时间间隔

        Returns:
            支持的间隔列表
        """
        return ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']


# 便捷函数
def get_stock_data(symbol: str,
                   start_date: str,
                   end_date: str = None,
                   interval: str = '1d') -> pd.DataFrame:
    """
    便捷函数：获取股票数据

    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        interval: 数据间隔

    Returns:
        DataFrame
    """
    fetcher = DataFetcher()
    return fetcher.fetch_stock_data(symbol, start_date, end_date, interval)


if __name__ == "__main__":
    # 测试数据获取
    fetcher = DataFetcher()

    print("="*60)
    print("测试数据获取")
    print("="*60)

    # 获取苹果公司（AAPL）最近一年的数据
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    print(f"\n获取AAPL数据 ({start_date} -> {end_date})...")
    df = fetcher.fetch_stock_data('AAPL', start_date, end_date)

    print(f"\n数据预览:")
    print(df.head())
    print(f"\n数据统计:")
    print(df.describe())

    print(f"\n数据列: {list(df.columns)}")
    print(f"数据行数: {len(df)}")

    # 获取最新价格
    print(f"\n获取最新价格...")
    latest_price = fetcher.get_latest_price('AAPL')
    print(f"AAPL 最新价格: ${latest_price:.2f}")

    # 批量获取多个股票
    print(f"\n批量获取数据...")
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    data_dict = fetcher.fetch_multiple_symbols(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )

    print(f"\n成功获取 {len(data_dict)} 只股票的数据")
    for symbol, data in data_dict.items():
        print(f"  {symbol}: {len(data)}条记录")
