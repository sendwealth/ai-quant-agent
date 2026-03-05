"""
API连接测试脚本
测试Tushare和智谱AI API连接
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import yaml

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_tushare_api():
    """
    测试1: Tushare API连接
    """
    print("\n" + "="*70)
    print("测试1: Tushare API 连接")
    print("="*70)

    # 读取配置
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("❌ 配置文件不存在: config/config.yaml")
        return False

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    tushare_token = config['data']['tushare']['token']

    if tushare_token == "YOUR_TUSHARE_TOKEN":
        print("❌ Tushare Token 未配置")
        return False

    print(f"\nToken: {tushare_token[:20]}...")

    try:
        import tushare as ts
        print("✓ Tushare 库已安装")
    except ImportError:
        print("❌ Tushare 库未安装")
        print("   安装命令: pip install tushare")
        return False

    try:
        ts.set_token(tushare_token)
        pro = ts.pro_api()
        print("✓ Tushare API 连接成功")

        # 测试获取股票列表
        print(f"\n测试获取股票列表...")
        df_stock_basic = pro.stock_basic(exchange='', list_status='L')

        if df_stock_basic is not None and len(df_stock_basic) > 0:
            print(f"✓ 获取股票列表成功: {len(df_stock_basic)}只股票")
            print(f"   前5只:")
            print(df_stock_basic.head())
        else:
            print("⚠️  股票列表为空")
            return False

        # 测试获取日线数据
        print(f"\n测试获取日线数据...")
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')

        df_daily = pro.daily(
            ts_code='600519.SH',  # 贵州茅台
            start_date=start_date,
            end_date=end_date
        )

        if df_daily is not None and len(df_daily) > 0:
            print(f"✓ 获取日线数据成功: {len(df_daily)}条记录")
            print(f"   时间范围: {df_daily['trade_date'].min()} ~ {df_daily['trade_date'].max()}")
            print(f"   最新价格: ¥{df_daily['close'].iloc[-1]:.2f}")
            print(f"   最新成交量: {df_daily['vol'].iloc[-1]/10000:.1f}万股")
            print(f"\n   数据预览:")
            print(df_daily.head())
        else:
            print("⚠️  日线数据为空")
            return False

        print(f"\n✅ Tushare API 测试通过!")
        return True

    except Exception as e:
        print(f"❌ Tushare API 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_zhipuai_api():
    """
    测试2: 智谱AI API连接
    """
    print("\n" + "="*70)
    print("测试2: 智谱AI API 连接")
    print("="*70)

    # 读取配置
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("❌ 配置文件不存在: config/config.yaml")
        return False

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    api_key = config['llm']['zhipuai']['api_key']

    if api_key == "YOUR_ZHIPUAI_API_KEY":
        print("❌ 智谱AI API Key 未配置")
        return False

    print(f"\nAPI Key: {api_key[:20]}...")

    try:
        from zhipuai import ZhipuAI
        print("✓ zhipuai 库已安装")
    except ImportError:
        print("❌ zhipuai 库未安装")
        print("   安装命令: pip install zhipuai")
        return False

    try:
        client = ZhipuAI(api_key=api_key)
        print("✓ 智谱AI 客户端初始化成功")

        # 测试简单对话
        print(f"\n测试简单对话...")
        response = client.chat.completions.create(
            model=config['llm']['zhipuai']['model'],
            messages=[
                {"role": "user", "content": "请用一句话介绍什么是量化交易？"}
            ],
            temperature=0.7,
        )

        if response and response.choices:
            result = response.choices[0].message.content
            print(f"✓ API调用成功")
            print(f"\n模型回复:")
            print(f"   {result}")
        else:
            print("⚠️  API返回为空")
            return False

        # 测试交易策略生成
        print(f"\n\n测试交易策略生成...")
        strategy_prompt = """
        请生成一个简单的移动平均线交叉交易策略，要求：
        1. 使用5日和20日均线
        2. 当5日线上穿20日线时买入
        3. 当5日线下穿20日线时卖出
        4. 给出清晰的交易规则
        """

        response = client.chat.completions.create(
            model=config['llm']['zhipuai']['model'],
            messages=[
                {"role": "user", "content": strategy_prompt}
            ],
            temperature=0.7,
        )

        if response and response.choices:
            result = response.choices[0].message.content
            print(f"✓ 策略生成成功")
            print(f"\n生成的策略:")
            print("   " + result.replace('\n', '\n   '))
        else:
            print("⚠️  策略生成失败")

        print(f"\n✅ 智谱AI API 测试通过!")
        return True

    except Exception as e:
        print(f"❌ 智谱AI API 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_simple_backtest():
    """
    测试3: 运行简单回测
    """
    print("\n" + "="*70)
    print("测试3: 运行示例回测")
    print("="*70)

    try:
        from data.astock_fetcher import AStockDataFetcher
        from utils.indicators import sma
        import pandas as pd
        import numpy as np

        print("\n✓ 导入模块成功")

        # 获取A股数据
        print(f"\n获取A股数据...")
        fetcher = AStockDataFetcher()

        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

        df = fetcher.fetch_stock_daily('600519', start_date, end_date, source='akshare')

        if df is None or len(df) == 0:
            print("❌ 数据获取失败")
            return False

        print(f"✓ 数据获取成功: {len(df)}条记录")

        # 生成信号：简单均线策略
        print(f"\n生成交易信号...")
        df['sma_short'] = sma(df['close'], 5)
        df['sma_long'] = sma(df['close'], 20)

        df['signal'] = 0
        df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1  # 买入信号
        df.loc[df['sma_short'] < df['sma_long'], 'signal'] = -1  # 卖出信号

        # 计算收益率
        print(f"\n计算收益率...")
        df['returns'] = df['close'].pct_change()

        # 简单回测
        print(f"\n运行回测...")

        initial_capital = 100000
        capital = initial_capital
        position = 0
        trades = []
        equity_curve = []

        for i in range(1, len(df)):
            if pd.isna(df['signal'].iloc[i]):
                continue

            current_price = df['close'].iloc[i]
            signal = df['signal'].iloc[i]

            # 执行交易
            if signal == 1 and position == 0:  # 买入
                position = capital / current_price
                trades.append({
                    'date': df['datetime'].iloc[i],
                    'action': 'BUY',
                    'price': current_price,
                    'shares': position
                })
                print(f"   {df['datetime'].iloc[i]}: 买入 {position:.2f}股 @ ¥{current_price:.2f}")

            elif signal == -1 and position > 0:  # 卖出
                capital = position * current_price
                trades.append({
                    'date': df['datetime'].iloc[i],
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position
                })
                print(f"   {df['datetime'].iloc[i]}: 卖出 {position:.2f}股 @ ¥{current_price:.2f}")
                position = 0

            # 计算权益
            if position > 0:
                equity = position * current_price
            else:
                equity = capital
            equity_curve.append(equity)

        # 最终平仓
        if position > 0:
            final_price = df['close'].iloc[-1]
            capital = position * final_price
            trades.append({
                'date': df['datetime'].iloc[-1],
                'action': 'SELL',
                'price': final_price,
                'shares': position
            })
            print(f"   {df['datetime'].iloc[-1]}: 最终平仓 {position:.2f}股 @ ¥{final_price:.2f}")

        # 计算回测指标
        total_return = (capital - initial_capital) / initial_capital
        annual_return = (1 + total_return) ** (365 / len(df)) - 1
        equity_values = pd.Series(equity_curve)
        daily_returns = equity_values.pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

        # 计算最大回撤
        cummax = equity_values.cummax()
        drawdown = (equity_values - cummax) / cummax
        max_drawdown = drawdown.min()

        print(f"\n{'='*70}")
        print("回测结果")
        print(f"{'='*70}")
        print(f"\n初始资金: ¥{initial_capital:,.2f}")
        print(f"最终资金: ¥{capital:,.2f}")
        print(f"\n总收益率: {total_return*100:+.2f}%")
        print(f"年化收益: {annual_return*100:+.2f}%")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"最大回撤: {max_drawdown*100:.2f}%")
        print(f"交易次数: {len(trades)}")

        # 买入持有对比
        buy_hold_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
        print(f"\n买入持有收益: {buy_hold_return*100:+.2f}%")

        if total_return > buy_hold_return:
            print(f"✓ 策略跑赢买入持有!")
        else:
            print(f"⚠️  策略不如买入持有")

        print(f"\n✅ 回测测试完成!")
        return True

    except Exception as e:
        print(f"❌ 回测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    主函数
    """
    print("\n" + "="*70)
    print("AI智能体量化交易系统 - API连接测试")
    print("="*70)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试内容:")
    print(f"  1. Tushare API 连接")
    print(f"  2. 智谱AI API 连接")
    print(f"  3. 运行示例回测")

    results = {}

    # 测试1: Tushare API
    results['tushare_api'] = test_tushare_api()

    # 测试2: 智谱AI API
    results['zhipuai_api'] = test_zhipuai_api()

    # 测试3: 回测
    results['backtest'] = run_simple_backtest()

    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)

    print(f"\n测试结果:")
    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status} {test_name}")

    print(f"\n总体评分: {passed}/{total} ({passed/total*100:.0f}%)")

    if passed == total:
        print(f"\n🎉 所有测试通过! 系统已准备就绪!")
        print(f"\n下一步:")
        print(f"  1. 运行完整验证: python3 examples/verify_astock.py")
        print(f"  2. 开始策略开发")
        print(f"  3. 运行参数优化")
        print(f"  4. 启动模拟交易")
    else:
        print(f"\n⚠️  部分测试未通过，请检查失败项")

    print("\n" + "="*70)

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
