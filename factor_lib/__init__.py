# -*- coding: utf-8 -*-
"""
因子库模块
================================================================================
按因子类别分组：
- A类: 趋势与动量因子
- B类: 均值回归因子
- C类: 波动率与风险因子
- D类: 微观结构与流动性因子
- E类: 统计分布因子
- F类: 宏观因子
- G类: 行业因子
- H类: 事件因子

使用示例:
    from factor_lib.momentum import MomentumFactors
    from factor_lib.reversal import ReversalFactors
    from factor_lib.volatility import VolatilityFactors
    from factor_lib.liquidity import LiquidityFactors
    from factor_lib.distribution import DistributionFactors
================================================================================
"""

from .momentum import MomentumFactors
from .reversal import ReversalFactors
from .volatility import VolatilityFactors
from .liquidity import LiquidityFactors
from .distribution import DistributionFactors

__all__ = [
    'MomentumFactors',
    'ReversalFactors',
    'VolatilityFactors',
    'LiquidityFactors',
    'DistributionFactors',
]
