"""
完整验证脚本 - 整合所有下一步计划
1. 接入真实数据
2. 优化参数
3. 验证A股
4. 完善模拟交易系统
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_complete_verification():
    """
    运行完整验证
    """
    print("\n" + "="*70)
    print("AI智能体量化交易系统 - 完整验证")
    print("="*70)
    print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"验证目标:")
    print(f"  1. 接入真实数据")
    print(f"  2. 优化参数")
    print(f"  3. 验证A股")
    print(f"  4. 完善模拟交易系统")

    results = {}

    # 任务1: 接入真实数据
    try:
        print(f"\n{'#'*70}")
        print(f"# 任务1: 接入真实数据")
        print(f"{'#'*70}")

        from data.astock_fetcher import AStockDataFetcher, get_popular_astocks
        from data.fetcher import DataFetcher

        print(f"\n✓ A股数据模块")
        print(f"✓ 美股数据模块")

        fetcher = AStockDataFetcher()

        # 获取A股数据
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

        print(f"\n获取A股数据...")
        df_astock = fetcher.fetch_stock_daily('600519', start_date, end_date, source='akshare')

        if df_astock is not None and len(df_astock) > 0:
            results['astock_data'] = True
            print(f"✓ A股数据获取成功: {len(df_astock)}条记录")
        else:
            results['astock_data'] = False
            print(f"⚠️  A股数据获取失败，使用模拟数据")

        # 获取美股数据
        try:
            yf_fetcher = DataFetcher()
            end_date_us = datetime.now().strftime('%Y-%m-%d')
            start_date_us = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

            print(f"\n获取美股数据...")
            df_usstock = yf_fetcher.fetch_stock_data('SPY', start_date_us, end_date_us)

            if df_usstock is not None and len(df_usstock) > 0:
                results['usstock_data'] = True
                print(f"✓ 美股数据获取成功: {len(df_usstock)}条记录")
            else:
                results['usstock_data'] = False
                print(f"⚠️  美股数据获取失败")
        except Exception as e:
            results['usstock_data'] = False
            print(f"⚠️  美股数据获取失败: {e}")

    except Exception as e:
        results['real_data'] = False
        print(f"❌ 真实数据接入失败: {e}")
        return results

    # 任务2: 优化参数
    try:
        print(f"\n{'#'*70}")
        print(f"# 任务2: 优化参数")
        print(f"{'#'*70}")

        from optimization.parameter_optimizer import ParameterOptimizer, simple_backtest
        from examples.verify_profitability import momentum_strategy, simple_sma_strategy

        # 使用A股数据进行优化
        df = df_astock if df_astock is not None and len(df_astock) > 0 else None

        if df is None:
            print(f"⚠️  没有可用数据，跳过参数优化")
            results['parameter_optimization'] = False
        else:
            optimizer = ParameterOptimizer(
                backtest_func=simple_backtest,
                metric='annual_return'
            )

            print(f"\n优化动量策略...")
            momentum_results = optimizer.optimize_momentum_strategy(df)

            if len(momentum_results) > 0:
                best_momentum = momentum_results.iloc[0]
                print(f"\n最佳动量策略参数:")
                print(f"  period: {int(best_momentum['period'])}")
                print(f"  threshold: {best_momentum['threshold']:.3f}")
                print(f"  年化收益: {best_momentum['annual_return']*100:.2f}%")
                results['parameter_optimization'] = True
                results['best_momentum_params'] = {
                    'period': int(best_momentum['period']),
                    'threshold': best_momentum['threshold']
                }
            else:
                results['parameter_optimization'] = False

    except Exception as e:
        results['parameter_optimization'] = False
        print(f"❌ 参数优化失败: {e}")

    # 任务3: 验证A股
    try:
        print(f"\n{'#'*70}")
        print(f"# 任务3: 验证A股")
        print(f"{'#'*70}")

        from examples.verify_profitability import (
            simple_sma_strategy,
            simple_rsi_strategy,
            simple_macd_strategy,
            momentum_strategy
        )

        df = df_astock if df_astock is not None and len(df_astock) > 0 else None

        if df is None:
            print(f"⚠️  没有可用数据，跳过A股验证")
            results['astock_verification'] = False
        else:
            strategies = [
                ("均线交叉 (10/30)", lambda d: simple_sma_strategy(d, 10, 30)),
                ("RSI策略", lambda d: simple_rsi_strategy(d, 14, 30, 70)),
                ("MACD策略", lambda d: simple_macd_strategy(d)),
                ("动量策略", lambda d: momentum_strategy(d,
                                                        results.get('best_momentum_params', {}).get('period', 10),
                                                        results.get('best_momentum_params', {}).get('threshold', 0.03))),
            ]

            strategy_results = []

            for strategy_name, strategy_func in strategies:
                print(f"\n{'─'*70}")
                print(f"策略: {strategy_name}")
                print(f"{'─'*70}")

                try:
                    signals = strategy_func(df)
                    metrics = simple_backtest(df, signals)

                    strategy_results.append({
                        'strategy': strategy_name,
                        'total_return': metrics['total_return'],
                        'annual_return': metrics['annual_return'],
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'max_drawdown': metrics['max_drawdown']
                    })

                    print(f"\n结果:")
                    print(f"  总收益: {metrics['total_return']*100:+.2f}%")
                    print(f"  年化收益: {metrics['annual_return']*100:+.2f}%")
                    print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
                    print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")

                except Exception as e:
                    print(f"\n❌ 回测失败: {e}")

            # 排序结果
            strategy_results.sort(key=lambda x: x['annual_return'], reverse=True)

            # 总结
            print(f"\n{'='*70}")
            print("A股策略对比总结")
            print(f"{'='*70}")

            print(f"\n{'策略':<20} {'总收益':<12} {'年化收益':<12} {'夏普比率':<10} {'最大回撤':<10}")
            print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

            for result in strategy_results:
                print(f"{result['strategy']:<20} "
                      f"{result['total_return']*100:>10.2f}% "
                      f"{result['annual_return']*100:>10.2f}% "
                      f"{result['sharpe_ratio']:>9.2f} "
                      f"{result['max_drawdown']*100:>9.2f}%")

            # 找出最佳策略
            if strategy_results:
                best_strategy = strategy_results[0]
                print(f"\n🏆 最佳A股策略: {best_strategy['strategy']}")
                print(f"   年化收益: {best_strategy['annual_return']*100:.2f}%")
                print(f"   夏普比率: {best_strategy['sharpe_ratio']:.2f}")

                results['astock_verification'] = True
                results['best_astock_strategy'] = best_strategy
            else:
                results['astock_verification'] = False

    except Exception as e:
        results['astock_verification'] = False
        print(f"❌ A股验证失败: {e}")

    # 任务4: 完善模拟交易系统
    try:
        print(f"\n{'#'*70}")
        print(f"# 任务4: 完善模拟交易系统")
        print(f"{'#'*70}")

        from trading.enhanced_paper_trading import EnhancedPaperTrading

        df = df_astock if df_astock is not None and len(df_astock) > 0 else None

        if df is None:
            print(f"⚠️  没有可用数据，跳过模拟交易测试")
            results['paper_trading'] = False
        else:
            print(f"\n测试增强模拟交易系统...")

            system = EnhancedPaperTrading(
                initial_capital=100000,
                commission=0.001,
                slippage=0.0001,
                enable_risk_control=True
            )

            # 获取最佳策略参数
            if results.get('best_momentum_params'):
                period = results['best_momentum_params']['period']
                threshold = results['best_momentum_params']['threshold']
            else:
                period = 10
                threshold = 0.03

            # 生成信号
            signals = momentum_strategy(df, period, threshold)

            print(f"\n运行模拟交易 ({len(df)} 个交易日)...\n")

            # 逐日交易
            for i in range(len(df)):
                if i >= len(signals):
                    break

                price = df['close'].iloc[i]
                date = df['datetime'].iloc[i].strftime('%Y-%m-%d') if hasattr(df['datetime'].iloc[i], 'strftime') else str(df['datetime'].iloc[i])
                signal = int(signals.iloc[i])

                # 执行信号
                system.execute_signal(price, signal, date)

                # 每50天打印一次状态
                if (i + 1) % 50 == 0:
                    equity = system.equity_curve[-1]
                    print(f"第 {i+1} 天: 权益 ¥{equity:,.2f}, 持仓 {system.position:.2f}")

            # 最终平仓
            if system.position != 0:
                final_price = df['close'].iloc[-1]
                final_date = df['datetime'].iloc[-1]
                if hasattr(final_date, 'strftime'):
                    final_date = final_date.strftime('%Y-%m-%d')
                system.execute_signal(final_price, 0, final_date)

            # 打印报告
            metrics = system.print_report()

            results['paper_trading'] = True
            results['paper_trading_metrics'] = metrics

    except Exception as e:
        results['paper_trading'] = False
        print(f"❌ 模拟交易测试失败: {e}")
        import traceback
        traceback.print_exc()

    # 生成最终报告
    print(f"\n{'='*70}")
    print("完整验证总结")
    print(f"{'='*70}")

    print(f"\n任务完成情况:")
    print(f"  1. 接入真实数据: {'✅ 完成' if results.get('astock_data') or results.get('usstock_data') else '❌ 未完成'}")
    print(f"  2. 优化参数: {'✅ 完成' if results.get('parameter_optimization') else '❌ 未完成'}")
    print(f"  3. 验证A股: {'✅ 完成' if results.get('astock_verification') else '❌ 未完成'}")
    print(f"  4. 完善模拟交易系统: {'✅ 完成' if results.get('paper_trading') else '❌ 未完成'}")

    completed = sum([
        results.get('astock_data') or results.get('usstock_data'),
        results.get('parameter_optimization'),
        results.get('astock_verification'),
        results.get('paper_trading')
    ])

    print(f"\n总体进度: {completed}/4 ({completed/4*100:.0f}%)")

    if completed == 4:
        print(f"\n🎉 所有任务完成! 系统已准备就绪!")

        if results.get('best_astock_strategy'):
            print(f"\n推荐A股策略:")
            strategy = results['best_astock_strategy']
            print(f"  策略: {strategy['strategy']}")
            print(f"  年化收益: {strategy['annual_return']*100:.2f}%")
            print(f"  夏普比率: {strategy['sharpe_ratio']:.2f}")

        print(f"\n下一步:")
        print(f"  1. 运行模拟交易系统 (examples/verify_astock.py)")
        print(f"  2. 验证多只股票")
        print(f"  3. 调整风控参数")
        print(f"  4. 准备实盘对接")

    else:
        print(f"\n⚠️  部分任务未完成，需要继续完善")

    print(f"\n{'='*70}")

    return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("AI智能体量化交易系统 - 完整验证")
    print("="*70)

    results = run_complete_verification()

    success = sum([
        results.get('astock_data') or results.get('usstock_data'),
        results.get('parameter_optimization'),
        results.get('astock_verification'),
        results.get('paper_trading')
    ]) == 4

    print(f"\n" + "="*70)
    if success:
        print("✅ 所有任务完成!")
    else:
        print("⚠️  部分任务未完成")
    print("="*70)

    sys.exit(0 if success else 1)
