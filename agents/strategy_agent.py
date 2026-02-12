"""
策略智能体
从自然语言描述自动生成可执行交易策略代码
"""

import json
import re
from typing import Dict, List, Optional, Any
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatZhipuAI
from loguru import logger

from utils.config import get_config
from utils.indicators import *


class StrategyAgent:
    """策略智能体 - 从自然语言生成交易策略"""

    def __init__(self):
        """初始化策略智能体"""
        self.config = get_config()
        self._init_llm()

    def _init_llm(self):
        """初始化大语言模型"""
        provider = self.config.get('llm', 'provider', default='openai')

        if provider == 'openai':
            api_key = self.config.get('llm', 'openai', 'api_key')
            model = self.config.get('llm', 'openai', 'model', default='gpt-4-turbo-preview')
            self.llm = ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=0.7
            )
            logger.info(f"使用OpenAI LLM: {model}")

        elif provider == 'zhipuai':
            api_key = self.config.get('llm', 'zhipuai', 'api_key')
            model = self.config.get('llm', 'zhipuai', 'model', default='glm-4-turbo')
            self.llm = ChatZhipuAI(
                api_key=api_key,
                model=model,
                temperature=0.7
            )
            logger.info(f"使用智谱AI LLM: {model}")

        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")

    def generate_strategy(self, description: str, market_type: str = 'stock') -> Dict[str, Any]:
        """
        从自然语言描述生成交易策略

        Args:
            description: 策略描述（自然语言）
            market_type: 市场类型（stock/crypto/forex）

        Returns:
            策略代码和元数据
        """
        logger.info(f"生成策略: {description}")

        # 构建提示词
        prompt_template = """
你是一个专业的量化交易策略开发专家。根据用户的自然语言描述，生成可执行的Python交易策略代码。

用户描述: {description}
市场类型: {market_type}

要求:
1. 生成完整的Python策略代码，使用pandas计算技术指标
2. 代码必须包含generate_signals函数，输入是DataFrame(df)，输出是信号series
3. 信号值: 1=买入, -1=卖出, 0=持有
4. 包含必要的参数和注释
5. 确保代码正确且可执行
6. 使用提供的indicators模块中的函数

返回格式(JSON):
{{
    "strategy_name": "策略名称",
    "description": "策略描述",
    "parameters": {{"param1": "value1"}},
    "code": "完整的Python代码",
    "entry_conditions": ["开仓条件1"],
    "exit_conditions": ["平仓条件1"]
}}
"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["description", "market_type"]
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            # 生成策略
            response = chain.run(description=description, market_type=market_type)

            # 解析JSON响应
            strategy_dict = self._parse_llm_response(response)

            logger.info(f"成功生成策略: {strategy_dict['strategy_name']}")

            return strategy_dict

        except Exception as e:
            logger.error(f"生成策略失败: {e}")
            return None

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应为JSON"""
        # 尝试提取JSON
        json_match = re.search(r'\{[\s\S]*\}', response)

        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                # 如果JSON解析失败，构建基本结构
                return {
                    "strategy_name": "Generated Strategy",
                    "description": response,
                    "parameters": {},
                    "code": self._extract_code_from_response(response),
                    "entry_conditions": [],
                    "exit_conditions": []
                }
        else:
            return {
                "strategy_name": "Generated Strategy",
                "description": response,
                "parameters": {},
                "code": self._extract_code_from_response(response),
                "entry_conditions": [],
                "exit_conditions": []
            }

    def _extract_code_from_response(self, response: str) -> str:
        """从响应中提取Python代码"""
        # 查找代码块
        code_block = re.search(r'```python\n([\s\S]*?)```', response)

        if code_block:
            return code_block.group(1)
        else:
            # 如果没有代码块，返回整个响应
            return response

    def optimize_parameters(self, strategy_code: str,
                           df: pd.DataFrame,
                           target_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        优化策略参数

        Args:
            strategy_code: 策略代码
            df: 历史数据
            target_metric: 优化目标指标

        Returns:
            优化后的参数和性能
        """
        logger.info("开始参数优化...")

        # 这里可以集成强化学习优化器
        # 暂时返回空结果
        return {
            "parameters": {},
            "performance": {},
            "message": "参数优化功能待开发"
        }

    def analyze_strategy(self, strategy_code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析策略性能

        Args:
            strategy_code: 策略代码
            df: 历史数据

        Returns:
            策略性能指标
        """
        logger.info("分析策略性能...")

        try:
            # 执行策略代码
            namespace = {}
            exec(strategy_code, namespace)

            # 生成信号
            if 'generate_signals' in namespace:
                signals = namespace['generate_signals'](df)

                # 计算基本指标
                returns = df['close'].pct_change()
                strategy_returns = signals.shift(1) * returns

                total_return = (1 + strategy_returns).prod() - 1
                annual_return = (1 + total_return) ** (252 / len(df)) - 1
                volatility = strategy_returns.std() * np.sqrt(252)
                sharpe_ratio = annual_return / volatility if volatility > 0 else 0

                # 最大回撤
                cumulative = (1 + strategy_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()

                return {
                    "total_return": total_return,
                    "annual_return": annual_return,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "win_rate": (strategy_returns > 0).mean(),
                    "message": "策略分析完成"
                }

            else:
                return {
                    "message": "策略代码中未找到generate_signals函数"
                }

        except Exception as e:
            logger.error(f"策略分析失败: {e}")
            return {
                "message": f"策略分析失败: {str(e)}"
            }


# 示例策略生成器
def generate_example_strategies() -> List[str]:
    """
    生成示例策略描述
    用于测试策略智能体
    """
    examples = [
        "当20日均线向上突破60日均线时买入，当20日均线向下突破60日均线时卖出",
        "当RSI低于30时买入，当RSI高于70时卖出",
        "当MACD柱状图从负转正时买入，当MACD柱状图从正转负时卖出",
        "当价格突破布林带上轨时买入，当价格跌破布林带下轨时卖出",
        "当ADX大于25且+DI大于-DI时买入，当ADX大于25且-DI大于+DI时卖出",
        "随机指标%K低于20时买入，%K高于80时卖出",
        "当威廉指标低于-80时买入，威廉指标高于-20时卖出"
    ]

    return examples


if __name__ == "__main__":
    # 测试策略智能体
    agent = StrategyAgent()

    # 生成示例策略
    examples = generate_example_strategies()

    for i, example in enumerate(examples[:2], 1):
        print(f"\n{'='*60}")
        print(f"示例 {i}: {example}")
        print('='*60)

        strategy = agent.generate_strategy(example, market_type='stock')

        if strategy:
            print(f"\n策略名称: {strategy['strategy_name']}")
            print(f"描述: {strategy['description']}")
            print(f"\n参数: {strategy['parameters']}")
            print(f"\n代码:\n{strategy['code'][:500]}...")
            print(f"\n开仓条件: {strategy['entry_conditions']}")
            print(f"平仓条件: {strategy['exit_conditions']}")
