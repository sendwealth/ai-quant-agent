#!/usr/bin/env python3
"""
动态选股系统 - 自动发现优质股票

功能：
1. 扫描所有可用股票
2. 多维度评分（财务、技术、成长、情绪）
3. 筛选优质股票
4. 自动更新监控列表
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np

# 设置logger
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DynamicStockSelector:
    """动态选股系统"""
    
    def __init__(self):
        self.data_dir = PROJECT_ROOT / "data"
        self.config_file = PROJECT_ROOT / "config" / "data_sources.yaml"
        self.stocks_data = {}  # 存储所有股票数据
        
    def scan_available_stocks(self) -> List[str]:
        """扫描所有可用股票"""
        logger.info("\n📊 扫描可用股票...")
        
        stock_files = list(self.data_dir.glob("real_*.csv"))
        stock_codes = []
        
        for file in stock_files:
            code = file.stem.replace("real_", "")
            if code.isdigit() and len(code) == 6:
                stock_codes.append(code)
        
        logger.info(f"✅ 找到 {len(stock_codes)} 只股票")
        return sorted(stock_codes)
    
    def load_stock_data(self, stock_code: str) -> pd.DataFrame:
        """加载股票历史数据"""
        file_path = self.data_dir / f"real_{stock_code}.csv"
        
        if not file_path.exists():
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            
            # 标准化列名
            if 'datetime' in df.columns:
                df['date'] = pd.to_datetime(df['datetime'])
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            df = df.sort_values('date').reset_index(drop=True)
            return df
            
        except Exception as e:
            logger.warning(f"加载 {stock_code} 数据失败: {e}")
            return pd.DataFrame()
    
    def calculate_technical_score(self, df: pd.DataFrame) -> Dict:
        """计算技术面得分"""
        if df.empty or len(df) < 50:
            return {"score": 0, "reasons": ["数据不足"]}
        
        try:
            close = df['close']
            
            # 1. 趋势得分（MA50 vs MA200）
            ma_50 = close.rolling(50).mean().iloc[-1]
            ma_200 = close.rolling(200).mean().iloc[-1] if len(df) >= 200 else ma_50
            
            trend_score = 0
            if ma_50 > ma_200:
                trend_score = 30  # 上升趋势
            elif ma_50 < ma_200 * 0.95:
                trend_score = 0  # 下降趋势
            else:
                trend_score = 15  # 横盘
            
            # 2. RSI 得分
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            rsi_score = 0
            if 30 < rsi < 70:
                rsi_score = 30  # 中性，好
            elif rsi < 30:
                rsi_score = 40  # 超卖，买入机会
            elif rsi > 80:
                rsi_score = 0  # 超买，风险
            
            # 3. MACD 得分
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            
            macd_score = 20 if histogram.iloc[-1] > 0 else 0
            
            # 4. 成交量得分
            volume = df['volume']
            avg_volume = volume.rolling(20).mean().iloc[-1]
            current_volume = volume.iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            volume_score = 20 if 0.8 < volume_ratio < 2.0 else 10
            
            # 总分
            total_score = trend_score + rsi_score + macd_score + volume_score
            
            reasons = []
            if trend_score == 30:
                reasons.append("上升趋势")
            elif trend_score == 0:
                reasons.append("下降趋势")
            
            if rsi < 30:
                reasons.append(f"RSI超卖({rsi:.0f})")
            elif rsi > 70:
                reasons.append(f"RSI超买({rsi:.0f})")
            
            if histogram.iloc[-1] > 0:
                reasons.append("MACD金叉")
            
            return {
                "score": min(100, total_score),
                "rsi": float(rsi),
                "macd": "金叉" if histogram.iloc[-1] > 0 else "死叉",
                "trend": "上升" if trend_score == 30 else "下降" if trend_score == 0 else "横盘",
                "reasons": reasons
            }
            
        except Exception as e:
            logger.warning(f"计算技术指标失败: {e}")
            return {"score": 0, "reasons": ["计算失败"]}
    
    def calculate_financial_score(self, stock_code: str) -> Dict:
        """计算财务面得分"""
        try:
            # 使用财务数据获取器
            from utils.financial_data_fetcher_v2 import FinancialDataFetcherV2
            
            fetcher = FinancialDataFetcherV2()
            data = fetcher.get_real_time_indicators(stock_code)
            
            score = 0
            reasons = []
            
            # 1. P/E 得分（市盈率）
            pe = data.get('pe_ratio', 25)
            if pe < 15:
                score += 30
                reasons.append(f"P/E低估({pe:.1f})")
            elif pe < 25:
                score += 20
                reasons.append(f"P/E合理({pe:.1f})")
            elif pe < 40:
                score += 10
            else:
                score += 0
                reasons.append(f"P/E偏高({pe:.1f})")
            
            # 2. P/B 得分（市净率）
            pb = data.get('pb_ratio', 3)
            if pb < 2:
                score += 20
                reasons.append(f"P/B低估({pb:.1f})")
            elif pb < 4:
                score += 15
            elif pb < 6:
                score += 10
            else:
                score += 0
                reasons.append(f"P/B偏高({pb:.1f})")
            
            # 3. ROE 得分
            roe = data.get('roe', 0.1)
            if roe > 0.15:
                score += 30
                reasons.append(f"ROE优秀({roe*100:.1f}%)")
            elif roe > 0.10:
                score += 20
                reasons.append(f"ROE良好({roe*100:.1f}%)")
            else:
                score += 0
            
            # 4. 负债率得分
            de_ratio = data.get('de_ratio', 0.5)
            if de_ratio < 0.4:
                score += 20
                reasons.append(f"负债率低({de_ratio*100:.1f}%)")
            elif de_ratio < 0.6:
                score += 15
            else:
                score += 5
                reasons.append(f"负债率较高({de_ratio*100:.1f}%)")
            
            return {
                "score": min(100, score),
                "pe": pe,
                "pb": pb,
                "roe": roe,
                "de_ratio": de_ratio,
                "reasons": reasons
            }
            
        except Exception as e:
            logger.warning(f"获取财务数据失败: {e}")
            return {"score": 0, "reasons": ["财务数据获取失败"]}
    
    def calculate_growth_score(self, df: pd.DataFrame) -> Dict:
        """计算成长性得分"""
        if df.empty or len(df) < 100:
            return {"score": 0, "reasons": ["数据不足"]}
        
        try:
            close = df['close']
            
            # 1. 价格增长（过去1年）
            if len(df) >= 250:
                price_1y_ago = close.iloc[-250]
                price_now = close.iloc[-1]
                price_growth = (price_now - price_1y_ago) / price_1y_ago
            else:
                price_growth = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
            
            # 2. 波动率（越低越好）
            returns = close.pct_change()
            volatility = returns.std() * np.sqrt(252)  # 年化波动率
            
            # 3. 最大回撤
            cummax = close.cummax()
            drawdown = (close - cummax) / cummax
            max_drawdown = drawdown.min()
            
            # 计算得分
            score = 0
            reasons = []
            
            # 增长得分
            if price_growth > 0.3:
                score += 40
                reasons.append(f"涨幅优秀(+{price_growth*100:.1f}%)")
            elif price_growth > 0.15:
                score += 30
                reasons.append(f"涨幅良好(+{price_growth*100:.1f}%)")
            elif price_growth > 0:
                score += 20
            else:
                score += 0
                reasons.append(f"涨幅为负({price_growth*100:.1f}%)")
            
            # 波动率得分（越低越好）
            if volatility < 0.25:
                score += 30
                reasons.append(f"波动率低({volatility*100:.1f}%)")
            elif volatility < 0.35:
                score += 20
            else:
                score += 10
                reasons.append(f"波动率较高({volatility*100:.1f}%)")
            
            # 回撤得分（越小越好）
            if max_drawdown > -0.2:
                score += 30
                reasons.append(f"回撤较小({max_drawdown*100:.1f}%)")
            elif max_drawdown > -0.35:
                score += 20
            else:
                score += 10
                reasons.append(f"回撤较大({max_drawdown*100:.1f}%)")
            
            return {
                "score": min(100, score),
                "price_growth": price_growth,
                "volatility": volatility,
                "max_drawdown": max_drawdown,
                "reasons": reasons
            }
            
        except Exception as e:
            logger.warning(f"计算成长性失败: {e}")
            return {"score": 0, "reasons": ["计算失败"]}
    
    def calculate_overall_score(self, technical: Dict, financial: Dict, growth: Dict) -> Dict:
        """计算综合得分"""
        # 权重
        weights = {
            "technical": 0.3,
            "financial": 0.4,
            "growth": 0.3
        }
        
        weighted_score = (
            technical.get("score", 0) * weights["technical"] +
            financial.get("score", 0) * weights["financial"] +
            growth.get("score", 0) * weights["growth"]
        )
        
        # 汇总所有原因
        all_reasons = []
        all_reasons.extend(technical.get("reasons", []))
        all_reasons.extend(financial.get("reasons", []))
        all_reasons.extend(growth.get("reasons", []))
        
        return {
            "score": round(weighted_score),
            "technical_score": technical.get("score", 0),
            "financial_score": financial.get("score", 0),
            "growth_score": growth.get("score", 0),
            "reasons": all_reasons
        }
    
    def evaluate_all_stocks(self, max_stocks: int = 20) -> List[Dict]:
        """评估所有股票"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 开始评估所有股票")
        logger.info(f"{'='*80}\n")
        
        stock_codes = self.scan_available_stocks()
        results = []
        
        for i, code in enumerate(stock_codes[:max_stocks], 1):
            logger.info(f"\n[{i+1}/{len(stock_codes[:max_stocks])}] 分析 {code}...")
            
            # 加载数据
            df = self.load_stock_data(code)
            if df.empty:
                logger.warning(f"  ❌ 无数据")
                continue
            
            # 计算各维度得分
            technical = self.calculate_technical_score(df)
            financial = self.calculate_financial_score(code)
            growth = self.calculate_growth_score(df)
            
            # 计算综合得分
            overall = self.calculate_overall_score(technical, financial, growth)
            
            # 获取股票名称（简化版）
            stock_name = f"股票{code}"
            
            result = {
                "code": code,
                "name": stock_name,
                "overall_score": overall["score"],
                "technical_score": overall["technical_score"],
                "financial_score": overall["financial_score"],
                "growth_score": overall["growth_score"],
                "reasons": overall["reasons"],
                "details": {
                    "technical": technical,
                    "financial": financial,
                    "growth": growth
                }
            }
            
            results.append(result)
            
            # 显示得分
            score_emoji = "🟢" if overall["score"] >= 70 else "🟡" if overall["score"] >= 50 else "🔴"
            logger.info(f"  {score_emoji} 综合得分: {overall['score']}/100")
            logger.info(f"     技术面: {technical.get('score', 0)}  财务面: {financial.get('score', 0)}  成长性: {growth.get('score', 0)}")
            
            if overall["score"] >= 60:
                logger.info(f"     亮点: {', '.join(overall['reasons'][:3])}")
        
        # 按综合得分排序
        results.sort(key=lambda x: x["overall_score"], reverse=True)
        
        return results
    
    def select_top_stocks(self, results: List[Dict], top_n: int = 10) -> List[Dict]:
        """选择得分最高的股票"""
        return results[:top_n]
    
    def update_config(self, top_stocks: List[Dict]):
        """更新配置文件中的监控列表"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 更新监控列表
            config['monitored_stocks'] = [
                {"code": stock["code"], "name": stock["name"]}
                for stock in top_stocks
            ]
            
            # 保存配置
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True)
            
            logger.info(f"\n✅ 已更新配置文件: {len(top_stocks)} 只股票")
            
        except Exception as e:
            logger.error(f"更新配置文件失败: {e}")
    
    def save_report(self, results: List[Dict], top_stocks: List[Dict]):
        """保存选股报告"""
        output_dir = PROJECT_ROOT / "data" / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"dynamic_selection_{timestamp}.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_analyzed": len(results),
            "top_stocks": top_stocks,
            "all_results": results,
            "selection_criteria": {
                "min_overall_score": 60,
                "weights": {
                    "technical": 0.3,
                    "financial": 0.4,
                    "growth": 0.3
                }
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 报告已保存: {output_file}")
        
        return output_file


def main():
    """主函数"""
    selector = DynamicStockSelector()
    
    # 1. 评估所有股票
    results = selector.evaluate_all_stocks(max_stocks=29)
    
    if not results:
        logger.error("❌ 无可用股票数据")
        return
    
    # 2. 选择top股票
    top_stocks = selector.select_top_stocks(results, top_n=10)
    
    # 3. 显示结果
    print(f"\n{'='*80}")
    print(f"🎯 动态选股结果 - Top {len(top_stocks)}")
    print(f"{'='*80}\n")
    
    for i, stock in enumerate(top_stocks, 1):
        score_emoji = "🟢" if stock["overall_score"] >= 70 else "🟡" if stock["overall_score"] >= 50 else "🔴"
        print(f"{i}. {score_emoji} {stock['name']} ({stock['code']}) - {stock['overall_score']}/100")
        print(f"   技术面:{stock['technical_score']}  财务面:{stock['financial_score']}  成长性:{stock['growth_score']}")
        if stock['reasons']:
            print(f"   亮点: {', '.join(stock['reasons'][:3])}")
        print()
    
    # 4. 更新配置
    selector.update_config(top_stocks[:4])  # 只更新前4只到监控列表
    
    # 5. 保存报告
    selector.save_report(results, top_stocks)
    
    print(f"\n{'='*80}")
    print(f"✅ 动态选股完成!")
    print(f"   - 分析股票: {len(results)} 只")
    print(f"   - 推荐股票: {len(top_stocks)} 只")
    print(f"   - 更新监控: {min(4, len(top_stocks))} 只")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
