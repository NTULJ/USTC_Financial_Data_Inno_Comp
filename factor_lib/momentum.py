# -*- coding: utf-8 -*-
"""
================================================================================
A类因子：趋势与动量因子
================================================================================

包含因子：
    A01 波动率缩放动量 (Volatility-Scaled Momentum)
    A02 路径效率 (Efficiency Ratio)
    A03 时序动量加速器 (Momentum Accelerator)
    A04 MACD柱(标准化) (Normalized MACD)
    A05 RSI相对强弱 (Relative Strength Index)
    A06 最高价突破 (Donchian Breakout)

研报来源：
    - 华泰证券《CTA策略系列：动量优化》
    - 光大证券《效率系数》
    - 天风证券《高频因子：加速度》
    - 广发证券《技术指标的量化择时》
    - 海通证券《RSI的多因子应用》
    - 申万宏源《突破策略在ETF中的应用》

使用方法:
    from factor_lib.momentum import MomentumFactors
    
    # 创建因子计算器实例
    mf = MomentumFactors()
    
    # 计算单个因子
    factor = mf.A01_vol_scaled_mom(close_df)
    
    # 或使用静态方法
    factor = MomentumFactors.A01_vol_scaled_mom(close_df)
================================================================================
"""

import pandas as pd
import numpy as np


class MomentumFactors:
    """趋势与动量因子计算器"""
    
    @staticmethod
    def A01_vol_scaled_mom(close_df, period=20):
        """
        A01 波动率缩放动量 (Volatility-Scaled Momentum)
        ================================================================================
        公式: Ret_period / Std(Ret, period)
        
        参数:
            close_df: 收盘价矩阵 (index=date, columns=sec)
            period: 动量计算周期，默认20
        
        返回:
            DataFrame: 波动率缩放动量因子值
        
        说明:
            - Ret_period: 过去period日累计收益率
            - Std(Ret, period): 过去period日收益率标准差
            - 消除波动率差异带来的规模偏差
        """
        # 计算日收益率
        daily_returns = close_df.pct_change()
        
        # 计算period日累计收益率
        ret_period = daily_returns.rolling(window=period).sum()
        
        # 计算period日收益率标准差
        std_period = daily_returns.rolling(window=period).std()
        
        # 波动率缩放动量 = 累计收益 / 收益波动率
        factor = ret_period / std_period.replace(0, np.nan)
        
        return factor
    
    @staticmethod
    def A02_efficiency_ratio(close_df, period=20):
        """
        A02 路径效率 (Efficiency Ratio, ER)
        ================================================================================
        公式: Net_Chg / Sum(Abs_Chg)
        
        参数:
            close_df: 收盘价矩阵
            period: 计算周期，默认20
        
        返回:
            DataFrame: 路径效率因子值
        
        说明:
            - Net_Chg: 期末价格 - 期初价格 (净变化)
            - Sum(Abs_Chg): 期间每日价格变化绝对值之和
            - ER接近1表示强趋势，接近0表示震荡
        """
        # 计算period日净变化
        net_change = close_df - close_df.shift(period)
        
        # 计算period日每日变化绝对值之和
        daily_changes = close_df.diff().abs()
        sum_abs_changes = daily_changes.rolling(window=period).sum()
        
        # 路径效率 = 净变化 / 绝对变化之和
        factor = net_change / sum_abs_changes.replace(0, np.nan)
        
        return factor
    
    @staticmethod
    def A03_momentum_accelerator(close_df, short_period=10, long_period=20):
        """
        A03 时序动量加速器 (Momentum Accelerator)
        ================================================================================
        公式: Slope(Close, short_period) - Slope(Close, long_period)
        
        参数:
            close_df: 收盘价矩阵
            short_period: 短期周期，默认10
            long_period: 长期周期，默认20
        
        返回:
            DataFrame: 动量加速器因子值
        
        说明:
            - 短期斜率反映近期动量
            - 长期斜率反映中长期趋势
            - 两者差值表示动量变化（加速度）
            - 正值表示动量增强，负值表示动量减弱
        """
        def calc_slope(series, window):
            """计算滚动窗口内的价格斜率"""
            slopes = pd.Series(index=series.index, dtype=float)
            for i in range(window - 1, len(series)):
                y = series.iloc[i - window + 1:i + 1].values
                x = np.arange(window)
                slope = np.polyfit(x, y, 1)[0]
                slopes.iloc[i] = slope
            return slopes
        
        # 计算短期和长期斜率
        slope_short = close_df.apply(lambda x: calc_slope(x, short_period))
        slope_long = close_df.apply(lambda x: calc_slope(x, long_period))
        
        # 动量加速器 = 短期斜率 - 长期斜率
        factor = slope_short - slope_long
        
        return factor
    
    @staticmethod
    def A04_normalized_macd(close_df, fast=12, slow=26):
        """
        A04 MACD柱(标准化) (Normalized MACD Histogram)
        ================================================================================
        公式: (EMA_fast - EMA_slow) / Close
        
        参数:
            close_df: 收盘价矩阵
            fast: 快速EMA周期,默认12
            slow: 慢速EMA周期,默认26
        
        返回:
            DataFrame: 标准化MACD因子值
        
        说明:
            - EMA_fast: 快速指数移动平均
            - EMA_slow: 慢速指数移动平均
            - 标准化处理消除价格规模差异
        """
        # 计算EMA
        ema_fast = close_df.ewm(span=fast, adjust=False).mean()
        ema_slow = close_df.ewm(span=slow, adjust=False).mean()
        
        # MACD柱状图
        macd = ema_fast - ema_slow
        
        # 标准化：除以收盘价
        factor = macd / close_df.replace(0, np.nan)
        
        return factor
    
    @staticmethod
    def A05_rsi(close_df, period=14):
        """
        A05 RSI相对强弱 (Relative Strength Index)
        ================================================================================
        公式: RSI(Close, period)
        
        参数:
            close_df: 收盘价矩阵
            period: RSI计算周期，默认14
        
        返回:
            DataFrame: RSI因子值
        
        说明:
            - RSI = 100 - (100 / (1 + RS))
            - RS = 平均涨幅 / 平均跌幅
            - 0-100波动，>70超买，<30超卖
        """
        # 计算价格变化
        delta = close_df.diff()
        
        # 分离上涨和下跌
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        
        # 计算平均涨幅和跌幅（使用指数移动平均）
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        # 计算RS和RSI
        rs = avg_gain / avg_loss.replace(0, np.nan)
        factor = 100 - (100 / (1 + rs))
        
        return factor
    
    @staticmethod
    def A06_donchian_breakout(close_df, high_df, period=60):
        """
        A06 最高价突破 (Donchian Breakout)
        ================================================================================
        公式: Close / Max(High, period)
        
        参数:
            close_df: 收盘价矩阵
            high_df: 最高价矩阵
            period: 回看周期，默认60
        
        返回:
            DataFrame: 最高价突破因子值
        
        说明:
            - Max(High, period): 过去period日最高价
            - 比率接近1表示接近高点
            - Donchian通道突破策略
        """
        # 计算过去period日最高价
        max_high = high_df.rolling(window=period).max()
        
        # 最高价突破 = 当前收盘价 / period日最高价
        factor = close_df / max_high.replace(0, np.nan)
        
        return factor
    
    @classmethod
    def get_all_factors(cls, close_df, high_df=None, **kwargs):
        """
        一次性计算所有A类因子
        
        参数:
            close_df: 收盘价矩阵
            high_df: 最高价矩阵（用于A06）
            **kwargs: 其他可选参数
        
        返回:
            dict: 因子名称到因子DataFrame的字典
        """
        factors = {}
        
        # A01 波动率缩放动量
        factors['A01_波动率缩放动量'] = cls.A01_vol_scaled_mom(close_df, period=kwargs.get('period', 20))
        
        # A02 路径效率
        factors['A02_路径效率'] = cls.A02_efficiency_ratio(close_df, period=kwargs.get('period', 20))
        
        # A03 时序动量加速器
        factors['A03_时序动量加速器'] = cls.A03_momentum_accelerator(
            close_df, 
            short_period=kwargs.get('short_period', 10),
            long_period=kwargs.get('long_period', 20)
        )
        
        # A04 MACD标准化
        factors['A04_MACD标准化'] = cls.A04_normalized_macd(
            close_df,
            fast=kwargs.get('fast', 12),
            slow=kwargs.get('slow', 26)
        )
        
        # A05 RSI
        factors['A05_RSI'] = cls.A05_rsi(close_df, period=kwargs.get('rsi_period', 14))
        
        # A06 最高价突破（需要high数据）
        if high_df is not None:
            factors['A06_最高价突破'] = cls.A06_donchian_breakout(close_df, high_df, period=kwargs.get('period', 60))
        
        return factors
