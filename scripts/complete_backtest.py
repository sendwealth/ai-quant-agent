#!/usr/bin/env python3
"""
完整回测脚本 - 自动运行agents + 回测
"""
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime

class CompleteBacktest:
    def __init__(self):
        self.signals_dir = Path("data/signals")
        self.reports_dir = Path("data/reports")
        self.agents = ["buffett", "technical", "fundamentals", "growth", "sentiment"]
        
    def load_top_stocks(self, top_n=4):
        """从动态选股结果加载Top N股票"""
        report_files = sorted(self.reports_dir.glob("dynamic_selection_*.json"), reverse=True)
        
        if not report_files:
            print("❌ 无动态选股结果，请先运行: python3 scripts/dynamic_stock_selector.py")
            return []
        
        with open(report_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        
        top_stocks = data.get("top_stocks", [])[:top_n]
        print(f"✅ 动态选股Top {len(top_stocks)}: {[s['code'] for s in top_stocks]}")
        return top_stocks
    
    def run_agent(self, agent_name, stock_code):
        """运行单个agent分析股票"""
        import subprocess
        
        agent_script = f"agents/{agent_name}_analyst.py"
        if not Path(agent_script).exists():
            print(f"⚠️  Agent脚本不存在: {agent_script}")
            return False
        
        cmd = ["python3", agent_script, "--stock", stock_code]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return True
            else:
                print(f"⚠️  {agent_name} 分析失败: {result.stderr[:100]}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⚠️  {agent_name} 超时")
            return False
        except Exception as e:
            print(f"⚠️  {agent_name} 异常: {e}")
            return False
    
    def run_all_agents_for_stock(self, stock_code, stock_name):
        """为单只股票运行所有agents"""
        print(f"\n  📊 运行Agents分析...")
        
        success_count = 0
        for agent in self.agents:
            print(f"    • {agent}...", end=" ", flush=True)
            if self.run_agent(agent, stock_code):
                print("✅")
                success_count += 1
            else:
                print("❌")
        
        print(f"  ✅ 成功: {success_count}/{len(self.agents)}")
        return success_count == len(self.agents)
    
    def load_signals(self, stock_code):
        """加载股票的所有agent信号"""
        signals = {}
        
        for agent in self.agents:
            signal_file = self.signals_dir / f"{agent}_{stock_code}.json"
            
            if signal_file.exists():
                try:
                    with open(signal_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        signals[agent] = {
                            "signal": data.get("signal", "HOLD"),
                            "confidence": data.get("confidence", 0.5),
                            "reasoning": data.get("reasoning", ""),
                        }
                except Exception as e:
                    print(f"⚠️  加载 {agent} 信号失败: {e}")
        
        return signals
    
    def calculate_consensus(self, signals):
        """计算共识信号"""
        if not signals:
            return "HOLD", 0.5, "无信号"
        
        buy_count = sum(1 for s in signals.values() if s["signal"] == "BUY")
        sell_count = sum(1 for s in signals.values() if s["signal"] == "SELL")
        hold_count = sum(1 for s in signals.values() if s["signal"] == "HOLD")
        
        avg_confidence = sum(s["confidence"] for s in signals.values()) / len(signals)
        total = len(signals)
        
        if buy_count >= total * 0.6:
            return "BUY", avg_confidence, f"强烈买入（{buy_count}/{total} agents看涨）"
        elif sell_count >= total * 0.6:
            return "SELL", avg_confidence, f"强烈卖出（{sell_count}/{total} agents看跌）"
        elif buy_count > sell_count:
            return "BUY", avg_confidence, f"买入（{buy_count}/{total} agents看涨）"
        elif sell_count > buy_count:
            return "SELL", avg_confidence, f"卖出（{sell_count}/{total} agents看跌）"
        else:
            return "HOLD", avg_confidence, "持有（多空分歧）"
    
    def backtest_stock(self, stock):
        """完整回测单只股票"""
        code = stock['code']
        name = stock['name']
        score = stock['overall_score']
        
        print(f"\n{'='*60}")
        print(f"🔍 完整分析 {name} ({code}) - 选股得分: {score}分")
        print(f"{'='*60}")
        
        # 1. 运行所有agents
        if not self.run_all_agents_for_stock(code, name):
            print(f"⚠️  Agents分析不完整")
        
        # 2. 加载信号
        signals = self.load_signals(code)
        
        if not signals:
            print(f"❌ 无信号数据")
            return None
        
        # 3. 显示信号
        print(f"\n  📊 Agent信号汇总:")
        print(f"  {'Agent':<15} {'信号':<12} {'信心度':<8} {'理由'}")
        print(f"  {'-'*80}")
        
        for agent, sig in signals.items():
            signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(sig["signal"], "⚪")
            print(f"  {agent:<15} {signal_emoji} {sig['signal']:<8} {int(sig['confidence']*100):>3}%    {sig['reasoning'][:50]}")
        
        # 4. 计算共识
        consensus_signal, confidence, reasoning = self.calculate_consensus(signals)
        
        print(f"\n  {'='*60}")
        print(f"  🎯 共识信号")
        print(f"  {'='*60}")
        signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(consensus_signal, "⚪")
        print(f"  信号: {signal_emoji} {consensus_signal}")
        print(f"  信心度: {int(confidence*100)}%")
        print(f"  理由: {reasoning}")
        
        return {
            "code": code,
            "name": name,
            "selection_score": score,
            "signals": signals,
            "consensus": {
                "signal": consensus_signal,
                "confidence": confidence,
                "reasoning": reasoning,
            }
        }
    
    def run_complete_backtest(self, top_n=4):
        """运行完整回测流程"""
        print(f"\n{'='*80}")
        print(f"📊 完整回测报告 - Top {top_n}股票")
        print(f"{'='*80}")
        
        # 1. 加载Top N股票
        top_stocks = self.load_top_stocks(top_n)
        
        if not top_stocks:
            print("❌ 无股票数据")
            return
        
        # 2. 完整回测每只股票
        all_results = []
        for stock in top_stocks:
            result = self.backtest_stock(stock)
            if result:
                all_results.append(result)
        
        # 3. 汇总报告
        print(f"\n{'='*80}")
        print(f"📊 汇总报告")
        print(f"{'='*80}\n")
        
        print(f"✅ 成功分析: {len(all_results)}/{len(top_stocks)} 只股票\n")
        
        # 统计信号
        buy_count = sum(1 for r in all_results if r.get('consensus', {}).get('signal') == 'BUY')
        hold_count = sum(1 for r in all_results if r.get('consensus', {}).get('signal') == 'HOLD')
        sell_count = sum(1 for r in all_results if r.get('consensus', {}).get('signal') == 'SELL')
        
        print(f"信号分布:")
        print(f"  🟢 BUY:  {buy_count} 只")
        print(f"  🟡 HOLD: {hold_count} 只")
        print(f"  🔴 SELL: {sell_count} 只")
        
        if buy_count > 0:
            print(f"\n💡 建议买入:")
            for r in all_results:
                if r.get('consensus', {}).get('signal') == 'BUY':
                    code = r['code']
                    name = r['name']
                    confidence = r['consensus'].get('confidence', 0)
                    score = r.get('selection_score', 0)
                    print(f"  • {name} ({code}): {int(confidence*100)}% 信心度 | 选股得分: {score}分")
        
        # 4. 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"complete_backtest_{timestamp}.json"
        
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "total_stocks": len(top_stocks),
                "analyzed_stocks": len(all_results),
                "results": all_results,
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 报告已保存: {report_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=4, help="回测Top N股票")
    args = parser.parse_args()
    
    backtest = CompleteBacktest()
    backtest.run_complete_backtest(args.top)
