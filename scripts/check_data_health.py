#!/usr/bin/env python3
"""
数据健康监控脚本
用于心跳系统检查数据新鲜度
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

def check_data_health():
    """检查数据健康状态"""
    data_dir = Path(__file__).parent.parent / 'data'
    
    # 优先检查监控的股票（而不是第一个文件）
    monitored_stocks = ['300750', '002475', '601318', '600276']
    stock_files = []
    for code in monitored_stocks:
        f = data_dir / f'real_{code}.csv'
        if f.exists():
            stock_files.append(f)
    
    # 如果监控的股票没有数据，再检查其他文件
    if not stock_files:
        stock_files = sorted(data_dir.glob('real_*.csv'))
    
    if not stock_files:
        return {
            'status': 'error',
            'message': '未找到股票数据文件',
            'action': '运行 python3 examples/fetch_tushare_auto.py'
        }
    
    # 检查最新数据日期
    try:
        csv_file = stock_files[0]
        # 尝试读取，检查是否有headers
        df_test = pd.read_csv(csv_file, nrows=1)
        if 'datetime' in df_test.columns or 'trade_date' in df_test.columns:
            df = pd.read_csv(csv_file)
            date_col = 'datetime' if 'datetime' in df.columns else 'trade_date'
        else:
            # 无header的CSV，第一列是日期
            df = pd.read_csv(csv_file, header=None, names=['date', 'open', 'high', 'low', 'close', 'volume'])
            date_col = 'date'
        
        # 智能解析日期（支持整数和字符串格式）
        sample = df[date_col].iloc[0]
        sample_dtype = str(df[date_col].dtype)
        
        if 'int' in sample_dtype or (isinstance(sample, str) and sample.isdigit()):
            df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d')
        else:
            df[date_col] = pd.to_datetime(df[date_col])
        
        latest_date = df[date_col].max()
        
        # 安全转换（直接解析日期字符串，避免nanosecond问题）
        latest_str = str(latest_date.date()) if hasattr(latest_date, 'date') else str(latest_date)
        latest_pydatetime = datetime.strptime(latest_str.split()[0], '%Y-%m-%d')
        age_days = (datetime.now() - latest_pydatetime).days
        
        if age_days > 7:
            return {
                'status': 'critical',
                'message': f'数据已过期{age_days}天',
                'latest_date': str(latest_date.date()),
                'action': '立即运行 python3 examples/fetch_tushare_auto.py'
            }
        elif age_days > 3:
            return {
                'status': 'warning',
                'message': f'数据{age_days}天更新',
                'latest_date': str(latest_date.date()),
                'action': '考虑更新数据'
            }
        else:
            return {
                'status': 'ok',
                'message': f'数据新鲜（{age_days}天前）',
                'latest_date': str(latest_date.date())
            }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'检查失败: {str(e)}',
            'action': '检查数据文件格式'
        }

if __name__ == '__main__':
    result = check_data_health()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 返回状态码
    exit({
        'ok': 0,
        'warning': 1,
        'error': 2,
        'critical': 3
    }.get(result['status'], 2))
