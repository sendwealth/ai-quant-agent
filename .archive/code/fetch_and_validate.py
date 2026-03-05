"""
稳健数据获取脚本
================
功能：
1. 重试机制（网络失败自动重试）
2. 延迟请求（避免被限流）
3. 进度保存（支持断点续传）
4. 多股票获取
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import time

# 尝试导入akshare
try:
    import akshare as ak
    HAS_AKSHARE = True
    print("✅ akshare已安装")
except ImportError:
    HAS_AKSHARE = False
    print("❌ akshare未安装")
    print("请运行: pip install akshare")

def get_stock_data_with_retry(stock_code, name, max_retries=3, delay=2):
    """带重试机制的数据获取"""
    if not HAS_AKSHARE:
        return None
    
    for attempt in range(max_retries):
        try:
            print(f"  🔄 [{attempt+1}/{max_retries}] 获取{name}({stock_code})...")
            
            # 添加延迟，避免请求过快
            if attempt > 0:
                wait_time = delay * (attempt + 1)
                print(f"     等待{wait_time}秒...")
                time.sleep(wait_time)
            
            # 获取数据（3年历史）
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=1095)).strftime('%Y%m%d')
            
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )
            
            if df is None or len(df) < 100:
                print(f"  ⚠️ {name}: 数据不足({len(df) if df is not None else 0}天)")
                return None
            
            # 重命名列
            df = df.rename(columns={
                '日期': 'datetime',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_change',
                '涨跌额': 'change',
                '换手率': 'turnover'
            })
            
            # 保存
            data_dir = Path('data')
            data_dir.mkdir(exist_ok=True)
            
            filename = f"real_{stock_code}.csv"
            filepath = data_dir / filename
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            # 统计信息
            days = len(df)
            start_price = df['close'].iloc[0]
            end_price = df['close'].iloc[-1]
            total_return = (end_price / start_price - 1) * 100
            
            print(f"  ✅ {name}: {days}天 | {start_price:.2f}→{end_price:.2f} ({total_return:+.1f}%)")
            
            return {
                'code': stock_code,
                'name': name,
                'days': days,
                'start_date': str(df['datetime'].iloc[0]),
                'end_date': str(df['datetime'].iloc[-1]),
                'start_price': float(start_price),
                'end_price': float(end_price),
                'total_return': float(total_return)
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ [{attempt+1}/{max_retries}] {name}: {error_msg[:100]}")
            
            if attempt < max_retries - 1:
                print(f"     将在{delay}秒后重试...")
                time.sleep(delay)
            else:
                print(f"  ❌ {name}: 重试{max_retries}次后失败")
                return None
    
    return None

def fetch_multiple_stocks():
    """批量获取股票数据"""
    print("\n" + "="*70)
    print("批量获取A股数据")
    print("="*70)
    
    # 股票列表（代码, 名称, 行业）
    stocks = [
        # 白酒
        ('600519', '茅台', '白酒'),
        ('000858', '五粮液', '白酒'),
        ('000568', '泸州老窖', '白酒'),
        
        # 新能源
        ('002594', '比亚迪', '新能源汽车'),
        ('300750', '宁德时代', '锂电池'),
        ('601012', '隆基绿能', '光伏'),
        
        # 银行
        ('601318', '中国平安', '保险'),
        ('000001', '平安银行', '银行'),
        ('600036', '招商银行', '银行'),
        ('601166', '兴业银行', '银行'),
        
        # 家电
        ('000333', '美的集团', '家电'),
        ('000651', '格力电器', '家电'),
        
        # 医药
        ('600276', '恒瑞医药', '医药'),
        ('000538', '云南白药', '医药'),
        ('300760', '迈瑞医疗', '医疗器械'),
        
        # 科技
        ('002415', '海康威视', '安防'),
        ('600588', '用友网络', '软件'),
        ('002230', '科大讯飞', 'AI'),
        
        # 消费
        ('600887', '伊利股份', '乳业'),
        ('000895', '双汇发展', '食品'),
    ]
    
    # 加载已有进度
    progress_file = Path('data/fetch_progress.json')
    completed = {}
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                completed = json.load(f)
            print(f"\n发现已有进度: {len(completed)}只股票")
        except:
            pass
    
    results = []
    success = 0
    failed = 0
    
    for i, (code, name, industry) in enumerate(stocks, 1):
        print(f"\n[{i}/{len(stocks)}] {name}({code}) - {industry}")
        
        # 检查是否已完成
        if code in completed:
            print(f"  ⏭️ 已存在，跳过")
            results.append(completed[code])
            success += 1
            continue
        
        # 获取数据
        result = get_stock_data_with_retry(code, name, max_retries=3, delay=3)
        
        if result:
            result['industry'] = industry
            results.append(result)
            completed[code] = result
            success += 1
            
            # 保存进度
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(completed, f, ensure_ascii=False, indent=2)
        else:
            failed += 1
        
        # 添加延迟，避免请求过快
        if i < len(stocks):
            print(f"\n  ⏳ 等待2秒...")
            time.sleep(2)
    
    # 汇总报告
    print("\n" + "="*70)
    print("数据获取汇总")
    print("="*70)
    print(f"\n总计: {len(stocks)}只股票")
    print(f"成功: {success}只 ✅")
    print(f"失败: {failed}只 ❌")
    
    if results:
        # 按行业分组统计
        print("\n【成功列表】")
        industries = {}
        for r in results:
            ind = r.get('industry', '其他')
            if ind not in industries:
                industries[ind] = []
            industries[ind].append(r)
        
        for ind, stocks_list in industries.items():
            print(f"\n{ind} ({len(stocks_list)}只):")
            for s in stocks_list:
                print(f"  - {s['name']}({s['code']}): {s['days']}天, {s['total_return']:+.1f}%")
        
        # 保存汇总
        summary = {
            'fetch_time': datetime.now().isoformat(),
            'total': len(stocks),
            'success': success,
            'failed': failed,
            'stocks': results
        }
        
        with open('data/fetch_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n汇总已保存: data/fetch_summary.json")
    
    return success

def test_optimized_params_on_new_data():
    """在新数据上测试优化后的参数"""
    print("\n" + "="*70)
    print("在新数据上验证优化参数")
    print("="*70)
    
    # 导入必要的函数
    import sys
    sys.path.insert(0, 'examples')
    
    from simple_optimization import advanced_backtest
    
    # 加载优化参数（从之前的结果）
    try:
        with open('data/optimization_results.json', 'r') as f:
            opt_results = json.load(f)
        best_params = opt_results['best_params']
        print("\n✅ 加载优化参数成功")
        print(f"  MA: {best_params['ma_fast']}/{best_params['ma_slow']}")
        print(f"  ATR止损: {best_params['atr_stop_mult']}x")
    except:
        print("\n⚠️ 未找到优化参数，使用默认参数")
        best_params = {
            'ma_fast': 10,
            'ma_slow': 30,
            'atr_stop_mult': 2.5,
            'atr_trail_mult': 2.0,
            'use_dynamic_position': True,
            'use_macd': True,
            'use_rsi': True,
            'take_profit_1': 0.10,
            'take_profit_2': 0.20,
            'partial_exit_1': 0.5,
            'partial_exit_2': 0.5
        }
    
    # 查找所有真实数据文件
    data_dir = Path('data')
    real_files = list(data_dir.glob('real_*.csv'))
    
    if not real_files:
        print("\n❌ 没有找到真实数据文件")
        return
    
    print(f"\n找到{len(real_files)}只股票数据")
    print("\n开始测试...\n")
    
    results = []
    
    for filepath in real_files:
        stock_code = filepath.stem.replace('real_', '')
        
        # 加载股票名称
        try:
            with open('data/fetch_summary.json', 'r') as f:
                summary = json.load(f)
                stock_info = next((s for s in summary['stocks'] if s['code'] == stock_code), None)
                stock_name = stock_info['name'] if stock_info else stock_code
        except:
            stock_name = stock_code
        
        print(f"测试 {stock_name}({stock_code})...")
        
        # 加载数据
        df = pd.read_csv(filepath)
        
        # 标准化列名
        if 'datetime' not in df.columns and '日期' in df.columns:
            df = df.rename(columns={
                '日期': 'datetime', '开盘': 'open', '最高': 'high',
                '最低': 'low', '收盘': 'close', '成交量': 'volume'
            })
        
        if 'close' not in df.columns or len(df) < 60:
            print(f"  ❌ 数据不足")
            continue
        
        # 计算买入持有收益
        bh_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        
        # 运行策略
        try:
            r = advanced_backtest(df, best_params)
            
            result = {
                'code': stock_code,
                'name': stock_name,
                'strategy_return': r['return'] * 100,
                'bh_return': bh_return,
                'excess_return': r['return'] * 100 - bh_return,
                'sharpe': r['sharpe'],
                'win_rate': r['win_rate'] * 100,
                'max_dd': r['max_dd'] * 100,
                'trades': r['trades']
            }
            
            results.append(result)
            
            # 显示结果
            status = "✅" if r['sharpe'] >= 0.5 and r['win_rate'] >= 0.6 else "⚠️"
            print(f"  {status} 策略{r['return']*100:+.2f}% | 买入{bh_return:+.1f}% | 夏普{r['sharpe']:.2f} | 胜率{r['win_rate']*100:.1f}%")
            
        except Exception as e:
            print(f"  ❌ 测试失败: {str(e)[:100]}")
    
    # 汇总统计
    if results:
        print("\n" + "="*70)
        print("验证结果汇总")
        print("="*70)
        
        df_results = pd.DataFrame(results)
        
        print(f"\n【整体表现】")
        print(f"  平均收益: {df_results['strategy_return'].mean():+.2f}%")
        print(f"  平均夏普: {df_results['sharpe'].mean():.3f}")
        print(f"  平均胜率: {df_results['win_rate'].mean():.1f}%")
        print(f"  平均回撤: {df_results['max_dd'].mean():.1f}%")
        
        print(f"\n【对比买入持有】")
        print(f"  策略平均: {df_results['strategy_return'].mean():+.2f}%")
        print(f"  买入持有: {df_results['bh_return'].mean():+.2f}%")
        print(f"  超额收益: {df_results['excess_return'].mean():+.2f}%")
        
        # 实盘条件检查
        print(f"\n【实盘条件】")
        avg_sharpe = df_results['sharpe'].mean()
        avg_return = df_results['strategy_return'].mean()
        avg_win_rate = df_results['win_rate'].mean()
        
        print(f"  {'✅' if avg_sharpe >= 0.5 else '❌'} 夏普≥0.5: {avg_sharpe:.3f}")
        print(f"  {'✅' if avg_win_rate >= 60 else '❌'} 胜率≥60%: {avg_win_rate:.1f}%")
        print(f"  {'✅' if avg_return >= 5 else '❌'} 收益≥5%: {avg_return:+.2f}%")
        
        # 达标股票统计
        qualified = df_results[
            (df_results['sharpe'] >= 0.5) & 
            (df_results['win_rate'] >= 60)
        ]
        
        print(f"\n【达标股票】{len(qualified)}/{len(results)}只")
        if len(qualified) > 0:
            print("\n推荐实盘股票:")
            for _, row in qualified.sort_values('sharpe', ascending=False).iterrows():
                print(f"  ✅ {row['name']}({row['code']}): 夏普{row['sharpe']:.2f}, 收益{row['strategy_return']:+.2f}%, 胜率{row['win_rate']:.1f}%")
        
        # 保存验证结果
        validation_report = {
            'test_time': datetime.now().isoformat(),
            'total_stocks': len(results),
            'qualified_stocks': len(qualified),
            'avg_sharpe': float(avg_sharpe),
            'avg_return': float(avg_return),
            'avg_win_rate': float(avg_win_rate),
            'avg_max_dd': float(df_results['max_dd'].mean()),
            'excess_return': float(df_results['excess_return'].mean()),
            'results': results
        }
        
        with open('data/validation_results.json', 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n验证结果已保存: data/validation_results.json")

def main():
    """主流程"""
    print("="*70)
    print("数据获取与验证流程")
    print("="*70)
    
    # 步骤1: 获取数据
    success = fetch_multiple_stocks()
    
    if success < 3:
        print("\n⚠️ 获取数据不足，验证中止")
        print("\n可能的原因:")
        print("1. 网络连接问题")
        print("2. akshare API限制")
        print("3. 股票代码错误")
        print("\n建议:")
        print("- 检查网络连接")
        print("- 稍后重试")
        print("- 使用已有数据继续验证")
    
    # 步骤2: 验证优化参数
    test_optimized_params_on_new_data()
    
    print("\n" + "="*70)
    print("流程完成")
    print("="*70)

if __name__ == '__main__':
    main()
