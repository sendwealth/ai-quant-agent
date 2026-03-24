#!/usr/bin/env python3
"""
获取真实股价 - 多数据源尝试
"""
import sys
import requests
import json

def get_price_from_eastmoney(stock_code: str) -> float:
    """从东方财富获取股价"""
    try:
        # 判断市场
        market = "SZ" if stock_code.startswith(("0", "3")) else "SH"
        secid = f"{market}.{stock_code}"
        
        url = "https://push2.eastmoney.com/api/qt/stock/get"
        params = {
            "secid": secid,
            "fields": "f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f57,f58,f60,f170,f171"
        }
        
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        if data and "data" in data and data["data"]:
            price = data["data"]["f43"] / 100  # 价格需要除以100
            print(f"✅ 东方财富: {stock_code} = {price}")
            return price
    except Exception as e:
        print(f"❌ 东方财富失败: {e}")
    return None

def get_price_from_tencent(stock_code: str) -> float:
    """从腾讯获取股价"""
    try:
        market = "sz" if stock_code.startswith(("0", "3")) else "sh"
        url = f"https://web.sqt.gtimg.cn/q={market}{stock_code}"
        
        response = requests.get(url, timeout=5)
        text = response.text
        
        # 解析格式: v_sz300750="51~宁德时代~300750~..."
        if '~' in text:
            parts = text.split('~')
            if len(parts) > 3:
                price = float(parts[3])
                print(f"✅ 腾讯: {stock_code} = {price}")
                return price
    except Exception as e:
        print(f"❌ 腾讯失败: {e}")
    return None

def get_price_from_sina(stock_code: str) -> float:
    """从新浪获取股价"""
    try:
        market = "sz" if stock_code.startswith(("0", "3")) else "sh"
        url = f"https://hq.sinajs.cn/list={market}{stock_code}"
        
        headers = {
            "Referer": "https://finance.sina.com.cn"
        }
        response = requests.get(url, headers=headers, timeout=5)
        text = response.text
        
        # 解析格式: var hq_str_sz300750="宁德时代,395.500,..."
        if '=' in text and ',' in text:
            data_str = text.split('"')[1]
            parts = data_str.split(',')
            if len(parts) > 3:
                price = float(parts[3])
                print(f"✅ 新浪: {stock_code} = {price}")
                return price
    except Exception as e:
        print(f"❌ 新浪失败: {e}")
    return None

def get_real_price(stock_code: str) -> float:
    """尝试多个数据源获取股价"""
    print(f"\n🔍 获取 {stock_code} 真实股价...")
    
    # 尝试多个数据源
    for fetcher in [get_price_from_eastmoney, get_price_from_tencent, get_price_from_sina]:
        price = fetcher(stock_code)
        if price and price > 0:
            return price
    
    print(f"❌ 所有数据源失败")
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 get_real_price.py <股票代码> [股票代码2] ...")
        sys.exit(1)
    
    prices = {}
    for code in sys.argv[1:]:
        price = get_real_price(code)
        if price:
            prices[code] = price
    
    print("\n📊 真实股价:")
    print(json.dumps(prices, indent=2, ensure_ascii=False))
