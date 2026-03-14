"""
TuShare数据获取脚本 - 自动版
===================
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import time

# TuShare配置
TUSHARE_TOKEN = '33649d8db312befd2e253d93e9bd2860e9c5e819864c8a2078b3869b'

try:
    import tushare as ts
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    HAS_TUSHARE = True
    print("✅ TuShare已配置")
except Exception as e:
    HAS_TUSHARE = False
    print(f"❌ TuShare配置失败: {e}")
    exit(1)

def get_stock_data(stock_code, name):
    """获取单只股票数据"""
    try:
        print(f"  🔄 {name}({stock_code})...", end=' ')
        
        # 格式化代码
        ts_code = stock_code + ('.SH' if stock_code.startswith('6') else '.SZ')
        
        # 获取数据
        df = pro.daily(
            ts_code=ts_code,
            start_date='20210301',
            end_date=datetime.now().strftime('%Y%m%d')
        )
        
        if df is None or len(df) < 100:
            print(f"❌ 数据不足({len(df) if df is not None else 0}天)")
            return None
        
        # 排序
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 标准化列名
        df = df.rename(columns={
            'trade_date': 'datetime',
            'vol': 'volume'
        })
        
        # 保存
        df[['datetime', 'open', 'high', 'low', 'close', 'volume']].to_csv(
            f'data/real_{stock_code}.csv', index=False
        )
        
        # 统计
        total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        print(f"✅ {len(df)}天, {total_return:+.1f}%")
        
        return {'code': stock_code, 'name': name, 'days': len(df), 'return': total_return}
        
    except Exception as e:
        print(f"❌ {str(e)[:50]}")
        return None

def main():
    """主函数"""
    print("\n" + "="*70)
    print("TuShare数据获取")
    print("="*70)
    
    # 股票列表
    stocks = [
        ('600519', '茅台'), ('000858', '五粮液'), ('000568', '泸州老窖'),
        ('002594', '比亚迪'), ('300750', '宁德时代'), ('601012', '隆基绿能'),
        ('601318', '中国平安'), ('600036', '招商银行'), ('601166', '兴业银行'),
        ('600276', '恒瑞医药'), ('000538', '云南白药'), ('300760', '迈瑞医疗'),
        ('002415', '海康威视'), ('002230', '科大讯飞'), ('000333', '美的集团'),
        ('000651', '格力电器'), ('600887', '伊利股份'), ('000895', '双汇发展'),
        ('601398', '工商银行'), ('000001', '平安银行')
    ]
    
    # 加载进度
    progress_file = Path('data/tushare_progress.json')
    completed = {}
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            completed = json.load(f)
        print(f"已有进度: {len(completed)}只\n")
    
    results = []
    success = 0
    
    for i, (code, name) in enumerate(stocks, 1):
        print(f"[{i}/{len(stocks)}] ", end='')
        
        if code in completed:
            # 检查旧数据是否有return字段
            old_data = completed[code]
            if 'return' not in old_data:
                print(f"🔄 {name} 数据格式旧，重新获取...")
                result = get_stock_data(code, name)
                if result:
                    results.append(result)
                    completed[code] = result
                    success += 1
                continue
            print(f"⏭️ {name} 已存在")
            results.append(old_data)
            success += 1
            continue
        
        result = get_stock_data(code, name)
        
        if result:
            results.append(result)
            completed[code] = result
            success += 1
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(completed, f, ensure_ascii=False, indent=2)
        
        time.sleep(0.3)
    
    # 汇总
    print("\n" + "="*70)
    print(f"完成: {success}/{len(stocks)}只")
    print("="*70)
    
    if results:
        # 安全排序：确保所有结果都有return字段
        for r in results:
            if 'return' not in r:
                r['return'] = 0
        results.sort(key=lambda x: x['return'], reverse=True)
        print("\n【按收益排序】")
        for r in results:
            print(f"  {r['name']}: {r['days']}天, {r['return']:+.1f}%")
        
        # 保存汇总
        with open('data/tushare_summary.json', 'w', encoding='utf-8') as f:
            json.dump({
                'time': datetime.now().isoformat(),
                'success': success,
                'stocks': results
            }, f, ensure_ascii=False, indent=2)
        
        # 数据验证
        print("\n🔍 验证数据新鲜度...")
        try:
            import pandas as pd
            from datetime import datetime
            
            # 检查第一只股票
            test_file = f"data/real_{stocks[0][0]}.csv"
            df = pd.read_csv(test_file)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                latest = df['datetime'].max()
                age = (datetime.now() - latest.to_pydatetime()).days
                
                if age > 7:
                    print(f"❌ 警告: 数据已过期{age}天!")
                elif age > 3:
                    print(f"⚠️  注意: 数据{age}天前更新")
                else:
                    print(f"✅ 数据新鲜（{age}天前）")
        except Exception as e:
            print(f"⚠️  验证失败: {e}")

if __name__ == '__main__':
    Path('data').mkdir(exist_ok=True)
    main()
