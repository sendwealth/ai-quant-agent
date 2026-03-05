"""
TuShare数据获取脚本
===================
使用TuShare获取A股历史数据（更稳定）
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import time

# TuShare配置
TUSHARE_TOKEN = '33649d8db312befd2e253d93e9bd2860e9c5e819864c8a2078b3869b'

try:
    import tushare as ts
    # 设置token
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    HAS_TUSHARE = True
    print("✅ TuShare已配置")
except ImportError:
    HAS_TUSHARE = False
    print("❌ 请安装: pip install tushare")
except Exception as e:
    HAS_TUSHARE = False
    print(f"❌ TuShare配置失败: {e}")

def get_stock_data_tushare(stock_code, name, start_date='20210301'):
    """使用TuShare获取股票数据"""
    if not HAS_TUSHARE:
        return None
    
    try:
        print(f"  🔄 获取{name}({stock_code})...")
        
        # TuShare代码格式：600519.SH, 000858.SZ
        ts_code = stock_code
        if stock_code.startswith('6'):
            ts_code += '.SH'
        else:
            ts_code += '.SZ'
        
        # 获取数据
        end_date = datetime.now().strftime('%Y%m%d')
        
        df = pro.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or len(df) < 100:
            print(f"  ⚠️ {name}: 数据不足({len(df) if df is not None else 0}天)")
            return None
        
        # TuShare返回的数据是倒序的，需要正序
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 标准化列名
        df = df.rename(columns={
            'trade_date': 'datetime',
            'vol': 'volume',
            'pct_chg': 'pct_change'
        })
        
        # 只保留需要的列
        columns_needed = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        df = df[columns_needed]
        
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
        print(f"  ❌ {name}: {str(e)[:100]}")
        return None

def batch_fetch_with_tushare():
    """使用TuShare批量获取"""
    print("\n" + "="*70)
    print("TuShare数据获取")
    print("="*70)
    
    # 股票列表
    stocks = [
        # 白酒
        ('600519', '茅台'),
        ('000858', '五粮液'),
        ('000568', '泸州老窖'),
        ('000596', '古井贡酒'),
        ('002304', '洋河股份'),
        
        # 新能源
        ('002594', '比亚迪'),
        ('300750', '宁德时代'),
        ('601012', '隆基绿能'),
        ('002129', 'TCL中环'),
        ('600438', '通威股份'),
        
        # 银行金融
        ('601318', '中国平安'),
        ('601398', '工商银行'),
        ('600036', '招商银行'),
        ('601166', '兴业银行'),
        ('000001', '平安银行'),
        
        # 医药
        ('600276', '恒瑞医药'),
        ('000538', '云南白药'),
        ('300760', '迈瑞医疗'),
        ('002007', '华兰生物'),
        ('000661', '长春高新'),
        
        # 科技
        ('002415', '海康威视'),
        ('002230', '科大讯飞'),
        ('600588', '用友网络'),
        ('000725', '京东方A'),
        ('002475', '立讯精密'),
        
        # 消费
        ('000333', '美的集团'),
        ('000651', '格力电器'),
        ('600887', '伊利股份'),
        ('000895', '双汇发展'),
        ('002714', '牧原股份'),
    ]
    
    # 加载进度
    progress_file = Path('data/tushare_progress.json')
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
    failed = 0
    
    for i, (code, name) in enumerate(stocks, 1):
        print(f"[{i}/{len(stocks)}] {name}({code})")
        
        # 跳过已完成的
        if code in completed:
            print(f"  ⏭️ 已存在，跳过")
            results.append(completed[code])
            success += 1
            continue
        
        # 获取数据
        result = get_stock_data_tushare(code, name)
        
        if result:
            results.append(result)
            completed[code] = result
            success += 1
            
            # 保存进度
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(completed, f, ensure_ascii=False, indent=2)
        else:
            failed += 1
        
        # 延迟（TuShare免费版有频率限制）
        if i < len(stocks):
            time.sleep(0.5)  # 500ms延迟
    
    # 汇总
    print("\n" + "="*70)
    print("获取完成")
    print("="*70)
    print(f"\n总计: {len(stocks)}只")
    print(f"成功: {success}只 ✅")
    print(f"失败: {failed}只 ❌")
    
    if results:
        # 按收益排序
        results.sort(key=lambda x: x['total_return'], reverse=True)
        
        print("\n【成功列表】")
        for r in results:
            print(f"  ✅ {r['name']}({r['code']}): {r['days']}天, {r['total_return']:+.1f}%")
        
        # 保存汇总
        summary = {
            'fetch_time': datetime.now().isoformat(),
            'source': 'tushare',
            'total': len(stocks),
            'success': success,
            'failed': failed,
            'stocks': results
        }
        
        with open('data/tushare_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n汇总已保存: data/tushare_summary.json")
    
    return success

def test_tushare_connection():
    """测试TuShare连接"""
    print("\n测试TuShare连接...")
    
    try:
        # 测试获取股票列表
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name')
        
        if df is not None and len(df) > 0:
            print(f"✅ 连接成功！可用股票数: {len(df)}")
            print(f"\n示例股票:")
            print(df.head(10).to_string(index=False))
            return True
        else:
            print("❌ 连接失败")
            return False
            
    except Exception as e:
        print(f"❌ 连接错误: {e}")
        return False

if __name__ == '__main__':
    # 先测试连接
    if test_tushare_connection():
        print("\n" + "="*70)
        input("按回车键开始获取数据...")
        batch_fetch_with_tushare()
    else:
        print("\n请检查:")
        print("1. TuShare token是否正确")
        print("2. 网络连接是否正常")
        print("3. 是否已安装tushare: pip install tushare")
