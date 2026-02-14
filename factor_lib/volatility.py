# -*- coding: utf-8 -*-
"""
================================================================================
C类因子: 波动率与风险因子
================================================================================

包含因子：
    C01 下行波动占比 (Downside Volatility Ratio)
    C02 极差波动率 (Parkinson Volatility)
    C03 Alpha#023 (GICS Alpha)
    C04 ATR占比 (ATR Ratio)

研报来源：
    - 兴业证券《剔除下行风险》
    - 长江证券《高频波动率低频化》
    - 国泰君安《Alpha191因子》
    - 华泰证券《波动率择时》

使用方法:
    from factor_lib.volatility import VolatilityFactors
    
    factor = VolatilityFactors.C01_downside_volatility(close_df)
================================================================================
"""

import pandas as pd
import numpy as np


class VolatilityFactors:
    """波动率与风险因子计算器"""
    
    @staticmethod
    def C01_downside_volatility(close_df, period=20):
        """
        C01 下行波动占比 (Downside Volatility Ratio)
        ================================================================================
        公式: Std(Ret_Down) / Std(Ret_Total)
        
        参数:
            close_df: 收盘价矩阵 (index=date, columns=sec)
            period: 计算周期，默认20
        
        返回:
            DataFrame: 下行波动占比因子值
        
        说明:
            - Ret_Down: 负收益（取负收益或0）
            - Ret_Total: 全部收益
            - 衡量下行风险与总风险的比值
            - 值越大表示下行风险越高
            - 因子取反：值越小（下行风险低）对应高因子值
        """
        # 计算日收益率
        ret = close_df.pct_change()
        
        # 下行收益：负收益保留，其余置为0
        ret_down = ret.copy()
        ret_down[ret_down > 0] = 0
        
        # 上行收益（用于对称处理）
        ret_up = ret.copy()
        ret_up[ret_up < 0] = 0
        
        # 计算下行波动率
        std_down = ret_down.rolling(window=period).std()
        
        # 计算总波动率
        std_total = ret.rolling(window=period).std()
        
        # 下行波动占比
        factor = std_down / std_total.replace(0, np.nan)
        
        return factor
    
    @staticmethod
    def C02_parkinson_volatility(high_df, low_df, period=20):
        """
        C02 极差波动率 (Parkinson Volatility)
        ================================================================================
        公式: (1 / (4*ln(2))) * (ln(High / Low))^2 的平方根
        简化形式: sqrt(Mean(ln(High/Low)^2) / (4*ln(2)))
        
        参数:
            high_df: 最高价矩阵 (index=date, columns=sec)
            low_df: 最低价矩阵 (index=date, columns=sec)
            period: 计算周期，默认20
        
        返回:
            DataFrame: 极差波动率因子值
        
        说明:
            - Parkinson估计量利用高低价差估计波动率
            - 比传统标准差更有效利用价格范围信息
            - 对极端波动更敏感
            - 值越大表示波动越高
        """
        # 计算高低价比率的对数
        hl_ratio = np.log(high_df / low_df.replace(0, np.nan))
        
        # 计算平方的均值
        hl_sq_mean = (hl_ratio ** 2).rolling(window=period).mean()
        
        # Parkinson波动率估计量
        # sqrt(mean(ln(High/Low)^2) / (4*ln(2)))
        factor = np.sqrt(hl_sq_mean / (4 * np.log(2)))
        
        return factor
    
    @staticmethod
    def C03_alpha023_gtja(close_df, period=20):
        """
        C03 Alpha#023 (GICS Alpha / Conditional Volatility)
        ================================================================================
        公式: Std( If(Close > Delay(Close), Close, 0) )
        
        参数:
            close_df: 收盘价矩阵
            period: 计算周期，默认20
        
        返回:
            DataFrame: Alpha023因子值
        
        说明:
            - 只考虑上涨日的收益波动
            - 条件波动率：上涨时的波动率
            - 反映市场上涨时的稳定性
            - 值越大表示上涨时波动越大
        """
        # 昨日收盘价
        prev_close = close_df.shift(1)
        
        # 如果今日收盘 > 昨日收盘，取今日收盘价；否则取0
        up_mask = close_df > prev_close
        conditional_price = close_df.where(up_mask, 0)
        
        # 计算上涨日的波动率
        factor = conditional_price.rolling(window=period).std()
        
        return factor
    
    @staticmethod
    def C04_atr_ratio(close_df, high_df, low_df, period=14):
        """
        C04 ATR占比 (Average True Range Ratio)
        ================================================================================
        公式: ATR(period) / Close
        
        参数:
            close_df: 收盘价矩阵
            high_df: 最高价矩阵
            low_df: 最低价矩阵
            period: ATR计算周期，默认14
        
        返回:
            DataFrame: ATR占比因子值
        
        说明:
            - ATR (Average True Range): 平均真实波幅
            - True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
            - 衡量价格的绝对波动幅度
            - 值越大表示波动越高
        """
        # 昨日收盘价
        prev_close = close_df.shift(1)
        
        # 计算三种真实波幅
        tr1 = high_df - low_df  # 今日高低点
        tr2 = (high_df - prev_close).abs()  # 今日高点与昨日收盘
        tr3 = (low_df - prev_close).abs()   # 今日低点与昨日收盘
        
        # True Range = max(TR1, TR2, TR3)
        # 使用fillna确保NaN不会传播
        true_range = tr1.fillna(0)
        true_range = true_range.where(tr1 >= tr2.fillna(0), tr2)
        true_range = true_range.where(true_range >= tr3.fillna(0), tr3)
        
        # 计算ATR
        atr = true_range.rolling(window=period, min_periods=1).mean()
        
        # ATR占比 = ATR / Close
        # 避免除零
        factor = atr / close_df.replace(0, np.nan)
        
        # 处理无穷值
        factor = factor.replace([np.inf, -np.inf], np.nan)
        
        return factor
    
    @classmethod
    def get_all_factors(cls, close_df, high_df=None, low_df=None, **kwargs):
        """
        一次性计算所有C类因子
        
        参数:
            close_df: 收盘价矩阵
            high_df: 最高价矩阵（用于C02, C04）
            low_df: 最低价矩阵（用于C02, C04）
            **kwargs: 其他可选参数
        
        返回:
            dict: 因子名称到因子DataFrame的字典
        """
        factors = {}
        
        # C01 下行波动占比
        factors['C01_下行波动占比'] = cls.C01_downside_volatility(
            close_df, period=kwargs.get('period', 20))
        
        # C02 极差波动率 - 需要High、Low数据
        if high_df is not None and low_df is not None:
            factors['C02_极差波动率'] = cls.C02_parkinson_volatility(
                high_df, low_df, period=kwargs.get('period', 20))
        
        # C03 Alpha023
        factors['C03_Alpha023'] = cls.C03_alpha023_gtja(
            close_df, period=kwargs.get('period', 20))
        
        # C04 ATR占比 - 需要OHLC数据
        if high_df is not None and low_df is not None:
            factors['C04_ATR占比'] = cls.C04_atr_ratio(
                close_df, high_df, low_df, period=kwargs.get('atr_period', 14))
        
        return factors
