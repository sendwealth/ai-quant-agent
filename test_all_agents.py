#!/usr/bin/env python3
"""
测试所有修复后的Analyst
"""

import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

def test_agent(agent_name, stock_code="300750"):
    """测试单个analyst"""
    script_path = PROJECT_ROOT / "agents" / f"{agent_name}_analyst.py"
    
    if not script_path.exists():
        print(f"❌ {agent_name}_analyst.py 不存在")
        return False
    
    print(f"\n{'='*60}")
    print(f"Testing {agent_name.upper()} Analyst...")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--stock", stock_code],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print(f"✅ {agent_name.upper()} Analyst: 成功")
            
            # 提取信号
            for line in result.stdout.split('\n'):
                if '信号:' in line or '信心度:' in line or '理由:' in line:
                    print(f"  {line.strip()}")
            return True
        else:
            print(f"❌ {agent_name.upper()} Analyst: 失败 (exit code {result.returncode})")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {agent_name.upper()} Analyst: 超时")
        return False
    except Exception as e:
        print(f"❌ {agent_name.upper()} Analyst: 异常 - {e}")
        return False

def main():
    print("\n" + "="*60)
    print("量化交易系统 - Agent 测试")
    print("="*60)
    
    agents = ["technical", "fundamentals", "growth", "sentiment", "buffett"]
    stock_code = "300750"
    
    results = {}
    for agent in agents:
        results[agent] = test_agent(agent, stock_code)
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    success_count = sum(1 for r in results.values() if r)
    print(f"成功: {success_count}/{len(agents)}")
    
    if success_count == len(agents):
        print("✅ 所有Agent测试通过！")
    else:
        print("⚠️ 部分Agent测试失败")
        failed = [k for k, v in results.items() if not v]
        if failed:
            print(f"失败: {', '.join(failed)}")
    
    return 0 if success_count == len(agents) else 1

if __name__ == "__main__":
    sys.exit(main())
