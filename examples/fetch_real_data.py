#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速数据获取脚本
Quick Data Fetch Script

获取真实股票数据用于分析
"""

import tushare as ts
import pandas as pd
import json
import os
from datetime import datetime, timedelta

# 设置token
ts.set_token('33649d8db312befd2e253d93e9bd2860e9c5e819864c8a2078b3869b')
pro = ts.pro_api()

def fetch_stock_data(ts_code, days=120):
    """获取单只股票数据"""
    try:
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')

        # 获取日线数据
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

        if df.empty:
            print(f"  {ts_code}: 无数据")
            return None

        # 获取基本面数据
        try:
            basic = pro.daily_basic(ts_code=ts_code, start_date=end_date, end_date=end_date)
            if not basic.empty:
                latest_basic = basic.iloc[0].to_dict()
            else:
                latest_basic = {}
        except:
            latest_basic = {}

        # 保存数据
        data = {
            'code': ts_code,
            'name': ts_code,  # 后续补充名称
            'price_data': df.to_dict('records'),
            'basic_info': latest_basic,
            'update_time': datetime.now().isoformat()
        }

        print(f"  {ts_code}: ✅ {len(df)}条数据")
        return data

    except Exception as e:
        print(f"  {ts_code}: ❌ {e}")
        return None

def fetch_all_stocks():
    """获取所有股票池数据"""
    print(f"\n{'='*70}")
    print(f"获取真实股票数据 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}\n")

    # 加载股票池
    with open('data/stock_pool_extended.json', 'r', encoding='utf-8') as f:
        pool_data = json.load(f)

    all_stocks = []
    for category, data in pool_data['categories'].items():
        all_stocks.extend(data['stocks'])

    print(f"股票池: {len(all_stocks)} 只\n")

    # 获取数据
    results = {}
    success_count = 0

    for stock in all_stocks:
        code = stock['code']
        # 转换代码格式 (000001 -> 000001.SZ)
        if code.startswith('6'):
            ts_code = f"{code}.SH"
        else:
            ts_code = f"{code}.SZ"

        data = fetch_stock_data(ts_code)
        if data:
            data['name'] = stock['name']
            data['sector'] = stock.get('sector', 'N/A')
            results[code] = data
            success_count += 1

    print(f"\n成功获取: {success_count}/{len(all_stocks)} 只\n")

    # 保存结果
    output_file = 'data/real_stock_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 数据已保存到 {output_file}\n")

    return results

def fetch_market_data():
    """获取市场数据"""
    print(f"\n{'='*70}")
    print("获取市场数据")
    print(f"{'='*70}\n")

    try:
        end_date = datetime.now().strftime('%Y%m%d')

        # 主要指数
        indices = ['000001.SH', '399001.SZ', '399006.SZ', '000300.SH']
        index_data = {}

        print("主要指数:")
        for index_code in indices:
            try:
                df = pro.index_daily(ts_code=index_code, start_date=end_date, end_date=end_date)
                if not df.empty:
                    latest = df.iloc[0]
                    index_data[index_code] = {
                        'close': float(latest['close']),
                        'pct_chg': float(latest['pct_chg']),
                        'volume': float(latest['vol']),
                        'amount': float(latest['amount'])
                    }
                    print(f"  {index_code}: {latest['close']:.2f} ({latest['pct_chg']:+.2f}%)")
            except Exception as e:
                print(f"  {index_code}: ❌ {e}")

        # 北向资金
        print("\n北向资金:")
        try:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            money_flow = pro.moneyflow_hsgt(start_date=start_date, end_date=end_date)

            if not money_flow.empty:
                latest = money_flow.iloc[-1]
                north_flow = {
                    'today': float(latest['north_money']),
                    'date': latest['trade_date']
                }
                print(f"  今日: {latest['north_money']:+.2f}亿")
        except Exception as e:
            print(f"  ❌ {e}")
            north_flow = {}

        # 保存
        market_data = {
            'indices': index_data,
            'north_flow': north_flow,
            'update_time': datetime.now().isoformat()
        }

        output_file = 'data/real_market_data.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(market_data, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 市场数据已保存到 {output_file}\n")

        return market_data

    except Exception as e:
        print(f"❌ 获取市场数据失败: {e}\n")
        return None

def main():
    """主函数"""
    print("\n🚀 开始获取真实数据...\n")

    # 1. 获取股票数据
    stock_data = fetch_all_stocks()

    # 2. 获取市场数据
    market_data = fetch_market_data()

    print(f"\n{'='*70}")
    print("✅ 真实数据获取完成！")
    print(f"{'='*70}\n")

    # 统计
    if stock_data:
        print(f"📊 数据统计:")
        print(f"  股票数量: {len(stock_data)}")
        print(f"  数据天数: 120天")
        print(f"  数据文件: data/real_stock_data.json")
        print(f"  市场数据: data/real_market_data.json")
        print()

if __name__ == '__main__':
    main()
