# -*- coding: utf-8 -*-
"""因子相关性分析脚本"""

import pandas as pd
import numpy as np
from tools.data_loader import load_price_data
from factor_lib.momentum import MomentumFactors

# 加载数据
close_df, high_df, low_df, open_df, vol_df, amount_df = load_price_data()

# 计算因子
a01 = MomentumFactors.A01_vol_scaled_mom(close_df, period=20)
a07 = MomentumFactors.A07_simple_momentum(close_df, period=20)
a02 = MomentumFactors.A02_efficiency_ratio(close_df, period=20)
a05 = MomentumFactors.A05_rsi(close_df, period=14)

# 选取同一截面计算相关性
date_idx = 100  # 取第100个交易日
a01_vals = a01.iloc[date_idx].dropna()
a07_vals = a07.iloc[date_idx].dropna()
a02_vals = a02.iloc[date_idx].dropna()
a05_vals = a05.iloc[date_idx].dropna()

# 取交集
common = a01_vals.index.intersection(a07_vals.index).intersection(a02_vals.index).intersection(a05_vals.index)

# 计算相关性矩阵
data = pd.DataFrame({
    'A01_波动率缩放': a01_vals[common],
    'A07_简单动量': a07_vals[common],
    'A02_路径效率': a02_vals[common],
    'A05_RSI': a05_vals[common]
})
corr = data.corr()

print('='*50)
print('因子横截面相关性矩阵')
print('='*50)
print(corr.round(4))
print()
print(f'A01 vs A07 相关系数: {corr.loc["A01_波动率缩放", "A07_简单动量"]:.4f}')
print(f'A01 vs A02 相关系数: {corr.loc["A01_波动率缩放", "A02_路径效率"]:.4f}')
print(f'A01 vs A05 相关系数: {corr.loc["A01_波动率缩放", "A05_RSI"]:.4f}')
