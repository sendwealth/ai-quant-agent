#!/usr/bin/env python3
"""
模拟盘状态监控
快速查看当前持仓和收益
"""
import json
from pathlib import Path
from datetime import datetime

def main():
    """显示模拟盘状态"""
    print("="*70)
    print("🤖 模拟盘状态监控")
    print("="*70)
    print(f"查询时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 读取持仓
    filepath = Path('data/auto_portfolio.json')
    if not filepath.exists():
        print("❌ 未找到持仓文件")
        print("请先运行: python3 examples/auto_trading_bot.py")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 基本信息
    print("📊 基本信息")
    print("-" * 70)
    print(f"初始资金: {data.get('initial_capital', 100000):,.2f} 元")
    print(f"当前现金: {data.get('cash', 0):,.2f} 元")
    print(f"最后更新: {data.get('update_time', 'N/A')}")
    print()

    # 持仓信息
    positions = data.get('positions', {})
    if not positions:
        print("📦 持仓状态: 空仓")
        print("系统正在等待买入信号...")
    else:
        print("📦 持仓详情")
        print("-" * 70)
        print(f"{'股票':<10} {'持仓':<8} {'成本':<10} {'现价':<10} {'市值':<12} {'盈亏':<15}")
        print("-" * 70)

        total_market_value = 0
        for code, pos in positions.items():
            market_value = pos['shares'] * pos.get('current_price', pos['buy_price'])
            profit = market_value - pos['cost']
            profit_pct = profit / pos['cost'] * 100
            total_market_value += market_value

            status = "✅" if profit > 0 else "❌"
            print(f"{pos['name']:<10} {pos['shares']:<8} {pos['buy_price']:<10.2f} "
                  f"{pos.get('current_price', pos['buy_price']):<10.2f} "
                  f"{market_value:<12.2f} {status} {profit:+.2f} ({profit_pct:+.1f}%)")

        print("-" * 70)
        print(f"持仓市值: {total_market_value:,.2f} 元")
        print()

    # 总资产
    cash = data.get('cash', 0)
    total = cash + total_market_value if positions else cash
    initial = data.get('initial_capital', 100000)
    total_return = (total - initial) / initial * 100

    print("💰 资产汇总")
    print("-" * 70)
    print(f"总资产: {total:,.2f} 元")
    print(f"总收益: {total - initial:+,.2f} 元 ({total_return:+.2f}%)")
    print(f"仓位: {(total - cash) / total * 100:.1f}%")
    print("="*70)

    # 交易记录
    trades = data.get('trades', [])
    if trades:
        print()
        print(f"📝 最近交易记录 (共{len(trades)}笔)")
        print("-" * 70)
        for trade in trades[-5:]:  # 显示最近5笔
            action = trade['action']
            time = trade['time'][:10]  # 只显示日期
            name = trade['name']
            shares = trade['shares']
            price = trade['price']
            amount = trade['amount']

            if action == 'BUY':
                print(f"✅ {time} 买入 {name} {shares}股 @{price:.2f} = {amount:.2f}元")
            else:
                profit = trade.get('profit', 0)
                profit_pct = trade.get('profit_pct', 0)
                reason = trade.get('reason', '')
                print(f"🔴 {time} 卖出 {name} {shares}股 @{price:.2f} = {amount:.2f}元 "
                      f"(盈亏:{profit:+.2f}/{profit_pct:+.1f}% 原因:{reason})")

    print()
    print("💡 提示:")
    print("  - 每天收盘后自动运行")
    print("  - 查看日志: tail -f logs/auto_trading.log")
    print("  - 手动运行: python3 examples/auto_trading_bot.py")

if __name__ == '__main__':
    main()
