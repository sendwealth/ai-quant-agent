#!/usr/bin/env python3
"""
容错数据更新脚本（增强版）

特性：
1. 多数据源支持（AkShare、Tushare、新浪财经）
2. 指数退避重试机制
3. 数据验证（去重、缺失值检查）
4. 邮件告警
5. 环境变量配置
"""

#!/usr/bin/env python3
"""
容错数据更新脚本（增强版）

特性：
1. 多数据源支持（AkShare、Tushare、新浪财经）
2. 指数退避重试机制
3. 数据验证（去重、缺失值检查）
4. 邮件告警
5. 环境变量配置
"""

import json
import math
import os
import smtplib
import sys
import time
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 添加项目根目录到路径（必须在其他导入之前）
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


# 加载 .env 文件
def load_env():
    """加载环境变量"""
    env_path = PROJECT_ROOT / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    # 处理注释
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    if '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()


# 加载配置
def load_config():
    """加载数据源配置（支持环境变量替换）"""
    config_path = PROJECT_ROOT / 'config' / 'data_sources.yaml'
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

            # 递归替换环境变量
            def replace_env_vars(obj):
                if isinstance(obj, dict):
                    return {k: replace_env_vars(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [replace_env_vars(item) for item in obj]
                elif isinstance(obj, str):
                    if obj.startswith('${') and obj.endswith('}'):
                        env_var = obj[2:-1]
                        return os.environ.get(env_var, '')
                    return obj
                else:
                    return obj

            config = replace_env_vars(config)
            return config
    return {}


class DataValidator:
    """数据验证器"""

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, stock_code: str) -> Tuple[bool, pd.DataFrame, List[str]]:
        """
        验证数据完整性

        Args:
            df: 原始数据
            stock_code: 股票代码

        Returns:
            (是否有效, 清洗后的数据, 问题列表)
        """
        issues = []

        if df is None or df.empty:
            return False, df, ["数据为空"]

        # 1. 去重
        original_rows = len(df)
        df = df.drop_duplicates(subset=['date'], keep='last')
        if len(df) < original_rows:
            duplicates = original_rows - len(df)
            issues.append(f"去除 {duplicates} 条重复数据")
            logger.info(f"{stock_code}: 去除 {duplicates} 条重复数据")

        # 2. 检查必需列
        required_columns = ['date', 'open', 'close', 'high', 'low', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, df, [f"缺少必需列: {missing_columns}"]

        # 3. 检查缺失值
        missing_stats = {}
        for col in required_columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_pct = missing_count / len(df) * 100
                missing_stats[col] = missing_pct
                issues.append(f"{col} 列缺失 {missing_count} 条 ({missing_pct:.1f}%)")

        # 如果缺失值超过 20%，认为数据无效
        if missing_stats and max(missing_stats.values()) > 20:
            return False, df, [f"缺失值过多: {missing_stats}"]

        # 4. 填充缺失值（使用前向填充）
        if missing_stats:
            df[required_columns] = df[required_columns].fillna(method='ffill')
            df[required_columns] = df[required_columns].fillna(method='bfill')
            issues.append("已填充缺失值（前向+后向填充）")

        # 5. 检查异常值
        # 价格不能为负数
        price_columns = ['open', 'close', 'high', 'low']
        for col in price_columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                issues.append(f"{col} 列有 {negative_count} 个负值（异常）")
                df = df[df[col] >= 0]

        # 成交量不能为负数
        negative_volume = (df['volume'] < 0).sum()
        if negative_volume > 0:
            issues.append(f"volume 列有 {negative_volume} 个负值（异常）")
            df = df[df['volume'] >= 0]

        # 6. 检查价格逻辑
        # high >= low, high >= open, high >= close, low <= open, low <= close
        invalid_high = (df['high'] < df['low']).sum()
        if invalid_high > 0:
            issues.append(f"{invalid_high} 条数据 high < low（异常）")
            df = df[df['high'] >= df['low']]

        # 7. 检查日期连续性
        df = df.sort_values('date')
        df['date'] = pd.to_datetime(df['date'])
        date_gaps = df['date'].diff().dt.days
        large_gaps = (date_gaps > 7).sum()  # 超过7天的缺口
        if large_gaps > 0:
            issues.append(f"发现 {large_gaps} 处数据缺口（>7天）")

        # 8. 检查数据新鲜度
        latest_date = df['date'].max()
        age_days = (datetime.now() - latest_date.to_pydatetime()).days
        if age_days > 7:
            issues.append(f"数据已过期 {age_days} 天")

        # 9. 数据质量评分
        quality_score = 100
        if issues:
            # 每个问题扣5分
            quality_score = max(0, 100 - len(issues) * 5)

        issues.append(f"数据质量评分: {quality_score}/100")

        return len(df) > 0, df, issues


class DataSource:
    """数据源基类"""

    def __init__(self, name: str):
        self.name = name
        self.available = False
        self._check_availability()

    def _check_availability(self):
        """检查数据源是否可用"""
        raise NotImplementedError

    def fetch_stock_data(self, stock_code: str, days: int = 365) -> Optional[pd.DataFrame]:
        """获取股票数据"""
        raise NotImplementedError


class AkShareSource(DataSource):
    """AkShare 数据源（主要）"""

    def __init__(self):
        super().__init__("AkShare")

    def _check_availability(self):
        try:
            import akshare as ak
            self.ak = ak
            # 测试连接
            ak.stock_zh_a_hist(symbol='300750', period='daily', adjust='hfq')
            self.available = True
            logger.info("AkShare 数据源可用")
        except Exception as e:
            self.available = False
            logger.warning(f"AkShare 数据源不可用: {e}")

    def fetch_stock_data(self, stock_code: str, days: int = 365) -> Optional[pd.DataFrame]:
        if not self.available:
            return None

        try:
            df = self.ak.stock_zh_a_hist(
                symbol=stock_code,
                period='daily',
                adjust='hfq',
                start_date=(datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')
            )

            if df.empty:
                logger.warning(f"AkShare: {stock_code} 数据为空")
                return None

            # 标准化列名
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_change',
                '涨跌额': 'change',
                '换手率': 'turnover'
            })

            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            logger.info(f"AkShare: {stock_code} 获取成功 ({len(df)} 条)")
            return df

        except Exception as e:
            logger.error(f"AkShare: {stock_code} 获取失败: {e}")
            return None


class TushareSource(DataSource):
    """Tushare 数据源（备用）"""

    def __init__(self, token: str = None):
        self.token = token or ""
        super().__init__("Tushare")

    def _check_availability(self):
        if not self.token:
            self.available = False
            logger.warning("Tushare 未配置 token")
            return

        try:
            import tushare as ts
            ts.set_token(self.token)
            self.pro = ts.pro_api()
            # 测试连接
            self.pro.daily(ts_code='300750.SZ', limit=1)
            self.available = True
            logger.info("Tushare 数据源可用")
        except Exception as e:
            self.available = False
            logger.warning(f"Tushare 数据源不可用: {e}")

    def fetch_stock_data(self, stock_code: str, days: int = 365) -> Optional[pd.DataFrame]:
        if not self.available:
            return None

        try:
            # 转换代码格式（300750 → 300750.SZ）
            ts_code = f"{stock_code}.SZ" if stock_code.startswith(('0', '3')) else f"{stock_code}.SH"

            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')

            df = self.pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )

            if df.empty:
                logger.warning(f"Tushare: {stock_code} 数据为空")
                return None

            # 标准化列名
            df = df.rename(columns={
                'trade_date': 'date',
                'vol': 'volume',
                'pct_chg': 'pct_change'
            })

            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            logger.info(f"Tushare: {stock_code} 获取成功 ({len(df)} 条)")
            return df

        except Exception as e:
            logger.error(f"Tushare: {stock_code} 获取失败: {e}")
            return None


class SinaFinanceSource(DataSource):
    """新浪财经数据源（备用）"""

    def __init__(self):
        super().__init__("SinaFinance")

    def _check_availability(self):
        try:
            import requests
            self.requests = requests
            # 测试连接
            test_url = f"http://hq.sinajs.cn/list=sh600000"
            response = requests.get(test_url, timeout=5)
            if response.status_code == 200:
                self.available = True
                logger.info("新浪财经数据源可用")
            else:
                self.available = False
                logger.warning(f"新浪财经数据源不可用: HTTP {response.status_code}")
        except Exception as e:
            self.available = False
            logger.warning(f"新浪财经数据源不可用: {e}")

    def fetch_stock_data(self, stock_code: str, days: int = 365) -> Optional[pd.DataFrame]:
        if not self.available:
            return None

        try:
            # 新浪财经历史数据接口
            # 转换代码格式（300750 → sz300750, 600276 → sh600276）
            market = 'sz' if stock_code.startswith(('0', '3')) else 'sh'
            full_code = f"{market}{stock_code}"

            # 新浪财经历史数据URL
            # 注意：新浪接口可能随时变化，这里使用一个相对稳定的接口
            url = f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
            
            params = {
                'symbol': full_code,
                'scale': '240',  # 日线
                'ma': 'no',
                'datalen': days * 2  # 获取足够多的数据
            }

            response = self.requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                logger.warning(f"新浪财经: {stock_code} HTTP {response.status_code}")
                return None

            # 解析数据
            # 新浪返回的是JSON数组格式
            data = response.json()

            if not data or not isinstance(data, list):
                logger.warning(f"新浪财经: {stock_code} 数据格式错误")
                return None

            # 转换为DataFrame
            df = pd.DataFrame(data)

            if df.empty:
                logger.warning(f"新浪财经: {stock_code} 数据为空")
                return None

            # 标准化列名
            column_mapping = {
                'day': 'date',
                'open': 'open',
                'close': 'close',
                'high': 'high',
                'low': 'low',
                'volume': 'volume'
            }

            df = df.rename(columns=column_mapping)

            # 确保数值类型
            for col in ['open', 'close', 'high', 'low', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            logger.info(f"新浪财经: {stock_code} 获取成功 ({len(df)} 条)")
            return df

        except Exception as e:
            logger.error(f"新浪财经: {stock_code} 获取失败: {e}")
            return None


class EmailAlerter:
    """邮件告警器"""

    def __init__(
        self,
        enabled: bool = False,
        smtp_server: str = "smtp.163.com",
        smtp_port: int = 465,
        sender: str = "",
        password: str = "",
        recipients: List[str] = None
    ):
        self.enabled = enabled
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender = sender
        self.password = password
        self.recipients = recipients or []

    def send_alert(self, subject: str, body: str):
        """发送告警邮件"""
        if not self.enabled:
            logger.debug("邮件告警未启用，跳过发送")
            return False

        if not self.password:
            logger.warning("邮件告警未配置密码，跳过发送")
            return False

        if not self.recipients:
            logger.warning("邮件告警未配置收件人，跳过发送")
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.sender, self.password)
                server.send_message(msg)

            logger.info(f"邮件告警已发送: {subject}")
            return True
        except Exception as e:
            logger.error(f"邮件发送失败: {e}")
            return False


class RobustDataUpdater:
    """容错数据更新器（增强版）"""

    def __init__(self):
        # 先加载环境变量
        load_env()

        self.data_dir = Settings.DATA_DIR
        self.config = load_config()
        self.validator = DataValidator()

        # 邮件告警配置
        email_config = self.config.get('email_alert', {})
        self.alerter = EmailAlerter(
            enabled=email_config.get('enabled', False),
            smtp_server=email_config.get('smtp_server', 'smtp.163.com'),
            smtp_port=email_config.get('smtp_port', 465),
            sender=email_config.get('sender', ''),
            password=email_config.get('password', ''),
            recipients=email_config.get('recipients', [])
        )

        # 数据源（按优先级排序）
        self.sources = self._init_sources()

        # 监控股票
        self.monitored_stocks = self._load_monitored_stocks()

        # 重试配置（支持指数退避）
        retry_config = self.config.get('retry', {})
        self.max_retries = retry_config.get('max_attempts', 3)
        self.base_delay = retry_config.get('delay_seconds', 5)
        self.exponential_backoff = retry_config.get('exponential_backoff', False)

    def _init_sources(self) -> List[DataSource]:
        """初始化数据源"""
        sources = []

        # AkShare
        akshare_config = self.config.get('akshare', {})
        if akshare_config.get('enabled', True):
            sources.append(AkShareSource())

        # Tushare
        tushare_config = self.config.get('tushare', {})
        if tushare_config.get('enabled', False):
            token = tushare_config.get('token', '')
            if token:
                sources.append(TushareSource(token))

        # 新浪财经
        sina_config = self.config.get('sina_finance', {})
        if sina_config.get('enabled', True):
            sources.append(SinaFinanceSource())

        # 按优先级排序
        sources.sort(key=lambda s: self.config.get(s.name.lower(), {}).get('priority', 99))

        return sources

    def _load_monitored_stocks(self) -> Dict[str, str]:
        """加载监控股票列表"""
        stocks_config = self.config.get('monitored_stocks', [])
        if stocks_config:
            return {s['code']: s['name'] for s in stocks_config}

        # 默认股票
        return {
            '300750': '宁德时代',
            '002475': '立讯精密',
            '601318': '中国平安',
            '600276': '恒瑞医药'
        }

    def _calculate_backoff_delay(self, retry_count: int) -> float:
        """
        计算退避延迟时间（指数退避）

        Args:
            retry_count: 重试次数（从0开始）

        Returns:
            延迟秒数
        """
        if not self.exponential_backoff:
            return self.base_delay

        # 指数退避: base_delay * 2^retry_count
        # 最大延迟 60 秒
        delay = min(self.base_delay * (2 ** retry_count), 60)

        # 添加随机抖动（±10%），避免多个客户端同时重试
        jitter = delay * 0.1 * (2 * np.random.random() - 1)
        delay = max(1, delay + jitter)

        return delay

    def fetch_with_retry(
        self,
        stock_code: str,
        stock_name: str,
        retry_count: int = 0,
        source_index: int = 0
    ) -> Tuple[bool, Optional[pd.DataFrame], List[str]]:
        """
        带重试的数据获取（支持指数退避）

        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            retry_count: 当前重试次数
            source_index: 当前数据源索引

        Returns:
            (成功标志, 数据DataFrame, 验证问题列表)
        """
        # 检查是否有可用的数据源
        available_sources = [(i, s) for i, s in enumerate(self.sources) if s.available]
        if not available_sources:
            logger.error(f"{stock_name}({stock_code}) 没有可用的数据源")
            return False, None, ["没有可用的数据源"]

        # 尝试所有可用数据源
        for idx, source in available_sources:
            if idx < source_index:
                continue  # 跳过已经尝试过的数据源

            df = source.fetch_stock_data(stock_code)
            if df is not None and not df.empty:
                # 数据验证
                valid, cleaned_df, issues = self.validator.validate_dataframe(df, stock_code)

                if valid:
                    return True, cleaned_df, issues
                else:
                    logger.warning(f"{source.name}: {stock_code} 数据验证失败: {issues}")

        # 所有数据源都失败，检查是否需要重试
        if retry_count < self.max_retries:
            # 计算退避延迟
            delay = self._calculate_backoff_delay(retry_count)
            logger.warning(
                f"{stock_name}({stock_code}) 所有数据源失败，"
                f"{delay:.1f}秒后重试 ({retry_count + 1}/{self.max_retries})"
            )
            time.sleep(delay)
            return self.fetch_with_retry(stock_code, stock_name, retry_count + 1, 0)

        # 重试耗尽
        logger.error(f"{stock_name}({stock_code}) 所有数据源失败，重试耗尽")
        return False, None, ["重试耗尽"]

    def update_all_stocks(self) -> Dict:
        """
        更新所有监控股票数据

        Returns:
            更新结果统计
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'total': len(self.monitored_stocks),
            'success': 0,
            'failed': 0,
            'details': []
        }

        for code, name in self.monitored_stocks.items():
            success, df, issues = self.fetch_with_retry(code, name)

            if success and df is not None:
                # 保存数据
                csv_path = self.data_dir / f"{code}_{name}.csv"
                df.to_csv(csv_path, index=False)

                # 同时保存为 real_{code}.csv 格式（兼容现有代码）
                real_path = self.data_dir / f"real_{code}.csv"
                df.to_csv(real_path, index=False)

                result['success'] += 1
                result['details'].append({
                    'code': code,
                    'name': name,
                    'status': 'success',
                    'rows': len(df),
                    'latest_date': str(df['date'].iloc[-1].date()),
                    'validation_issues': issues if issues else None
                })
                logger.info(f"✅ {name}({code}) 更新成功")
            else:
                result['failed'] += 1
                result['details'].append({
                    'code': code,
                    'name': name,
                    'status': 'failed',
                    'issues': issues
                })
                logger.error(f"❌ {name}({code}) 更新失败")

        # 检查数据健康度，必要时发送告警
        self._check_and_alert(result)

        return result

    def _check_and_alert(self, update_result: Dict):
        """检查数据健康度并发送告警"""

        # 计算失败率
        failure_rate = update_result['failed'] / update_result['total'] if update_result['total'] > 0 else 0

        # 检查数据新鲜度
        stale_stocks = []
        thresholds = self.config.get('alert_thresholds', {})
        stale_days = thresholds.get('data_stale_days', 1)

        for detail in update_result['details']:
            if detail['status'] == 'success':
                try:
                    latest = datetime.strptime(detail['latest_date'], '%Y-%m-%d')
                    age_days = (datetime.now() - latest).days
                    if age_days > stale_days:
                        stale_stocks.append(f"{detail['name']}({detail['code']}): 过期{age_days}天")
                except:
                    pass

        # 发送告警
        failure_threshold = thresholds.get('failure_rate', 0.5)
        if failure_rate > failure_threshold:
            # 失败率过高，发送严重告警
            self.alerter.send_alert(
                subject="【严重】量化系统数据更新失败率过高",
                body=f"""
数据更新失败率: {failure_rate:.1%}
成功: {update_result['success']}
失败: {update_result['failed']}

详情:
{json.dumps(update_result['details'], ensure_ascii=False, indent=2)}

时间: {update_result['timestamp']}
                """
            )

        elif stale_stocks:
            # 数据过期告警
            self.alerter.send_alert(
                subject="【警告】量化系统数据过期",
                body=f"""
以下股票数据已过期:

{chr(10).join(stale_stocks)}

请手动检查数据更新。

时间: {datetime.now().isoformat()}
                """
            )

    def run(self):
        """执行更新（入口函数）"""
        logger.info("=" * 60)
        logger.info("开始容错数据更新（增强版）")
        logger.info(f"监控股票: {len(self.monitored_stocks)} 只")
        logger.info(f"可用数据源: {[s.name for s in self.sources if s.available]}")
        logger.info(f"重试策略: 指数退避={'启用' if self.exponential_backoff else '禁用'}")
        logger.info("=" * 60)

        result = self.update_all_stocks()

        logger.info("=" * 60)
        logger.info(f"更新完成: 成功 {result['success']}/{result['total']}")
        logger.info("=" * 60)

        return result


def main():
    """主函数"""
    updater = RobustDataUpdater()
    result = updater.run()

    # 输出 JSON 结果
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 返回状态码
    return 0 if result['failed'] == 0 else 1


if __name__ == '__main__':
    exit(main())
