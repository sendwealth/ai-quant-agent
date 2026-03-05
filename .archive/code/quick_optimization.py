"""
快速数据获取和参数优化脚本
===========================
目标：
1. 获取10只真实A股数据
2. 参数网格搜索优化
3. 找到最优参数组合
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

# 尝试导入akshare
try:
    import akshare as ak
    HAS_AKSHARE = True
    print("✅ akshare已安装")
except ImportError:
    HAS_AKSHARE = False
    print("❌ akshare未安装，请运行: pip install akshare")

def get_real_stock_data(stock_code, name, start_date='20210101', end_date='20240101'):
    """获取真实股票数据"""
    if not HAS_AKSHARE:
        print(f"  ❌ {name}: akshare未安装")
        return None
    
    try:
        print(f"  🔄 获取{name}({stock_code})数据...")
        
        # 使用akshare获取数据
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"  # 前复权
        )
        
        if df is None or len(df) < 100:
            print(f"  ❌ {name}: 数据不足")
            return None
        
        # 重命名列
        df = df.rename(columns={
            '日期': 'datetime',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume'
        })
        
        # 保存
        filename = f"real_{stock_code}.csv"
        df.to_csv(f'data/{filename}', index=False)
        
        print(f"  ✅ {name}: {len(df)}天数据已保存")
        return df
        
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return None

def fetch_real_stocks():
    """获取多只真实股票数据"""
    print("\n" + "="*70)
    print("获取真实A股数据")
    print("="*70)
    
    # 股票列表（代码, 名称）
    stocks = [
        ('600519', '茅台'),
        ('000858', '五粮液'),
        ('002594', '比亚迪'),
        ('300750', '宁德时代'),
        ('601318', '中国平安'),
        ('000001', '平安银行'),
        ('600036', '招商银行'),
        ('601166', '兴业银行'),
        ('000333', '美的集团'),
        ('600276', '恒瑞医药'),
    ]
    
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    success = 0
    for code, name in stocks:
        df = get_real_stock_data(code, name)
        if df is not None:
            success += 1
    
    print(f"\n成功获取: {success}/{len(stocks)}只股票")
    
    if success < 5:
        print("\n⚠️ 获取数据失败，可能的原因：")
        print("1. akshare未安装: pip install akshare")
        print("2. 网络连接问题")
        print("3. 股票代码错误")
    
    return success

def grid_search_optimization(df, param_ranges):
    """网格搜索参数优化"""
    from advanced_strategy_v2 import advanced_backtest
    
    print("\n" + "="*70)
    print("参数网格搜索优化")
    print("="*70)
    
    results = []
    total = (len(param_ranges['ma_fast']) * 
             len(param_ranges['ma_slow']) * 
             len(param_ranges['atr_stop']) *
             len(param_ranges['position']))
    
    count = 0
    best_sharpe = -999
    best_params = None
    
    for ma_fast in param_ranges['ma_fast']:
        for ma_slow in param_ranges['ma_slow']:
            if ma_fast >= ma_slow:
                continue
                
            for atr_stop in param_ranges['atr_stop']:
                for position in param_ranges['position']:
                    count += 1
                    
                    params = {
                        'ma_fast': ma_fast,
                        'ma_slow': ma_slow,
                        'atr_stop_mult': atr_stop,
                        'atr_trail_mult': atr_stop * 0.8,
                        'position_size': position,
                        'use_market_filter': True,
                        'use_rsi_filter': True,
                        'take_profit_1': 0.10,
                        'take_profit_2': 0.20,
                        'partial_exit_1': 0.5,
                        'partial_exit_2': 0.5,
                    }
                    
                    try:
                        r = advanced_backtest(df, params)
                        
                        results.append({
                            'params': params,
                            'sharpe': r['sharpe'],
                            'return': r['return'],
                            'win_rate': r['win_rate'],
                            'max_dd': r['max_dd']
                        })
                        
                        if r['sharpe'] > best_sharpe:
                            best_sharpe = r['sharpe']
                            best_params = params.copy()
                            best_result = r.copy()
                        
                        if count % 10 == 0:
                            print(f"  进度: {count}/{total} | 最佳夏普: {best_sharpe:.2f}")
                    
                    except Exception as e:
                        pass
    
    print(f"\n✅ 完成! 测试了{len(results)}组参数")
    print(f"\n【最佳参数】")
    print(f"  MA周期: {best_params['ma_fast']}/{best_params['ma_slow']}")
    print(f"  ATR止损: {best_params['atr_stop_mult']}x")
    print(f"  仓位: {best_params['position_size']*100:.0f}%")
    print(f"\n【最佳结果】")
    print(f"  夏普: {best_result['sharpe']:.2f}")
    print(f"  收益: {best_result['return']*100:+.2f}%")
    print(f"  胜率: {best_result['win_rate']*100:.1f}%")
    print(f"  回撤: {best_result['max_dd']*100:.1f}%")
    
    # 保存结果
    output = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': len(results),
        'best_params': best_params,
        'best_result': {
            'sharpe': best_result['sharpe'],
            'return': best_result['return'],
            'win_rate': best_result['win_rate'],
            'max_dd': best_result['max_dd']
        },
        'all_results': results[:100]  # 只保存前100个
    }
    
    with open('data/optimization_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n结果已保存: data/optimization_results.json")
    
    return best_params, best_result

def main():
    """主流程"""
    print("\n" + "="*70)
    print("快速优化流程")
    print("="*70)
    
    # 步骤1: 获取真实数据
    print("\n【步骤1】获取真实数据")
    success = fetch_real_stocks()
    
    if success < 5:
        print("\n⚠️ 真实数据不足，使用模拟数据演示优化")
        # 使用模拟数据
        df = pd.read_csv('data/sim_BULL001.csv')
        if '日期' in df.columns:
            df = df.rename(columns={
                '日期': 'datetime', '开盘': 'open', '最高': 'high',
                '最低': 'low', '收盘': 'close', '成交量': 'volume'
            })
    else:
        # 使用第一只真实股票
        import glob
        files = glob.glob('data/real_*.csv')
        if files:
            df = pd.read_csv(files[0])
        else:
            print("❌ 没有找到真实数据文件")
            return
    
    # 步骤2: 参数优化
    print("\n【步骤2】参数网格搜索")
    
    param_ranges = {
        'ma_fast': [5, 8, 10, 12, 15],
        'ma_slow': [20, 25, 30, 35, 40],
        'atr_stop': [2.0, 2.5, 3.0, 3.5, 4.0],
        'position': [0.20, 0.25, 0.30, 0.35, 0.40]
    }
    
    best_params, best_result = grid_search_optimization(df, param_ranges)
    
    # 步骤3: 生成报告
    print("\n" + "="*70)
    print("优化完成!")
    print("="*70)
    print("\n【下一步】")
    print("1. 在所有真实股票上测试最优参数")
    print("2. 对比优化前后表现")
    print("3. 如果达标，准备模拟盘测试")
    print("4. 如果不达标，继续调整参数范围")

if __name__ == '__main__':
    main()
