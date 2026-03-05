"""
改进版数据获取 - 更稳定的获取方式
====================================
优化：
1. 更长的延迟（5-10秒）
2. 更多的重试（5次）
3. 断点续传
4. 错误日志记录
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import time
import random

try:
    import akshare as ak
    HAS_AKSHARE = True
    print("✅ akshare已安装")
except ImportError:
    HAS_AKSHARE = False
    print("❌ 请安装: pip install akshare")

def get_stock_data_robust(stock_code, name, max_retries=5, base_delay=5):
    """更稳健的数据获取"""
    if not HAS_AKSHARE:
        return None
    
    for attempt in range(max_retries):
        try:
            # 随机延迟，避免固定频率
            delay = base_delay + random.randint(1, 5)
            if attempt > 0:
                delay *= (attempt + 1)
            
            print(f"  🔄 [{attempt+1}/{max_retries}] {name}({stock_code}) - 等待{delay}秒...")
            time.sleep(delay)
            
            # 获取数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=1095)).strftime('%Y%m%d')
            
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            
            if df is None or len(df) < 100:
                print(f"  ⚠️ {name}: 数据不足({len(df) if df is not None else 0}天)")
                return None
            
            # 标准化列名
            df = df.rename(columns={
                '日期': 'datetime', '开盘': 'open', '最高': 'high',
                '最低': 'low', '收盘': 'close', '成交量': 'volume',
                '成交额': 'amount', '振幅': 'amplitude',
                '涨跌幅': 'pct_change', '涨跌额': 'change', '换手率': 'turnover'
            })
            
            # 保存
            data_dir = Path('data')
            data_dir.mkdir(exist_ok=True)
            filepath = data_dir / f"real_{stock_code}.csv"
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            # 统计
            days = len(df)
            total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
            
            print(f"  ✅ {name}: {days}天, {total_return:+.1f}%")
            
            return {
                'code': stock_code,
                'name': name,
                'days': days,
                'total_return': total_return
            }
            
        except Exception as e:
            error_msg = str(e)[:150]
            print(f"  ❌ [{attempt+1}/{max_retries}] {name}: {error_msg}")
            
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 10
                print(f"     {wait}秒后重试...")
                time.sleep(wait)
            else:
                # 记录失败
                with open('data/fetch_errors.log', 'a') as f:
                    f.write(f"{datetime.now()} | {stock_code} | {name} | {error_msg}\n")
                return None
    
    return None

def batch_fetch_stocks():
    """批量获取股票数据"""
    print("\n" + "="*70)
    print("批量获取A股数据 - 改进版")
    print("="*70)
    
    # 股票列表
    stocks = [
        # 白酒
        ('600519', '茅台'),
        ('000858', '五粮液'),
        ('000568', '泸州老窖'),
        ('000596', '古井贡酒'),
        
        # 新能源
        ('002594', '比亚迪'),
        ('300750', '宁德时代'),
        ('601012', '隆基绿能'),
        ('002129', 'TCL中环'),
        
        # 银行
        ('601318', '中国平安'),
        ('601398', '工商银行'),
        ('600036', '招商银行'),
        ('601166', '兴业银行'),
        
        # 医药
        ('600276', '恒瑞医药'),
        ('000538', '云南白药'),
        ('300760', '迈瑞医疗'),
        ('002007', '华兰生物'),
        
        # 科技
        ('002415', '海康威视'),
        ('002230', '科大讯飞'),
        ('600588', '用友网络'),
        ('002036', '联创电子'),
    ]
    
    # 加载进度
    progress_file = Path('data/fetch_progress.json')
    completed = {}
    if progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                completed = json.load(f)
            print(f"\n发现已有进度: {len(completed)}只股票\n")
        except:
            pass
    
    results = []
    success = 0
    
    for i, (code, name) in enumerate(stocks, 1):
        print(f"[{i}/{len(stocks)}] {name}({code})")
        
        # 跳过已完成的
        if code in completed:
            print(f"  ⏭️ 已存在，跳过")
            results.append(completed[code])
            success += 1
            continue
        
        # 获取数据
        result = get_stock_data_robust(code, name, max_retries=5, base_delay=8)
        
        if result:
            results.append(result)
            completed[code] = result
            success += 1
            
            # 保存进度
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(completed, f, ensure_ascii=False, indent=2)
        
        # 批次间延迟
        if i < len(stocks) and code not in completed:
            batch_delay = random.randint(3, 8)
            print(f"\n  ⏳ 批次延迟{batch_delay}秒...\n")
            time.sleep(batch_delay)
    
    # 汇总
    print("\n" + "="*70)
    print(f"获取完成: {success}/{len(stocks)}只股票")
    print("="*70)
    
    if results:
        # 按收益排序
        results.sort(key=lambda x: x['total_return'], reverse=True)
        
        print("\n【成功列表】")
        for r in results:
            print(f"  ✅ {r['name']}({r['code']}): {r['days']}天, {r['total_return']:+.1f}%")
        
        # 保存汇总
        summary = {
            'fetch_time': datetime.now().isoformat(),
            'total': len(stocks),
            'success': success,
            'stocks': results
        }
        
        with open('data/fetch_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n汇总已保存: data/fetch_summary.json")
    
    return success

if __name__ == '__main__':
    batch_fetch_stocks()
