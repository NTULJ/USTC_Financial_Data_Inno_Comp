# -*- coding: utf-8 -*-
"""
================================================================================
D类因子：微观结构与流动性因子
================================================================================

包含因子：
    D01 Amihud变化率 (Amihud Illiquidity Ratio)
    D02 日内动量 (Intraday Momentum)
    D03 高低价差比 (High-Low Range Ratio)
    D04 开盘跳空 (Overnight Gap)

研报来源：
    - 广发证券《流动性因子实证》
    - 中信建投《收盘效应》
    - 东方证券《隔夜信息含量》

使用方法:
    from factor_lib.liquidity import LiquidityFactors
    
    factor = LiquidityFactors.D01_amihud_change(close_df, amount_df)
================================================================================
"""

import pandas as pd
import numpy as np


class LiquidityFactors:
    """微观结构与流动性因子计算器"""
    
    @staticmethod
    def D01_amihud_change(close_df, amount_df, period=20):
        """
        D01 Amihud变化率 (Amihud Illiquidity Ratio Change)
        ================================================================================
        公式: Delta(Abs(Ret) / Amount)
        
        参数:
            close_df: 收盘价矩阵 (index=date, columns=sec)
            amount_df: 成交额矩阵 (index=date, columns=sec)
            period: 变化率计算周期，默认20
        
        返回:
            DataFrame: Amihud变化率因子值
        
        说明:
            - Amihud = |Ret| / Amount (非流动性指标)
            - Delta(Amihud): Amihud的变化率
            - 衡量流动性变化程度
            - 值越大表示流动性恶化越严重
        """
        # 计算日收益率
        ret = close_df.pct_change()
        
        # 计算Amihud非流动性指标: |Ret| / Amount
        # 成交额需要缩放（原始单位可能很大）
        amihud = np.abs(ret) / (amount_df / 1e8)  # 归一化处理
        
        # 计算Amihud的变化率
        amihud_change = amihud.diff(period)
        
        return amihud_change
    
    @staticmethod
    def D02_intraday_momentum(close_df, open_df, high_df, low_df):
        """
        D02 日内动量 (Intraday Momentum)
        ================================================================================
        公式: (Close - Open) / (High - Low)
        
        参数:
            close_df: 收盘价矩阵
            open_df: 开盘价矩阵
            high_df: 最高价矩阵
            low_df: 最低价矩阵
        
        返回:
            DataFrame: 日内动量因子值
        
        说明:
            - 衡量日内价格变动方向和幅度
            - 正值：收盘高于开盘（上涨）
            - 负值：收盘低于开盘（下跌）
            - 分母为日内振幅，归一化处理
        """
        # 计算日内动量
        numerator = close_df - open_df
        denominator = high_df - low_df
        
        # 避免除零
        factor = numerator / denominator.replace(0, np.nan)
        
        return factor
    
    @staticmethod
    def D03_high_low_range(close_df, high_df, low_df, vol_df, period=20):
        """
        D03 高低价差比 (High-Low Range Ratio)
        ================================================================================
        公式: Sum(High-Low, period) / Sum(Vol, period)
        
        参数:
            close_df: 收盘价矩阵
            high_df: 最高价矩阵
            low_df: 最低价矩阵
            vol_df: 成交量矩阵
            period: 计算周期，默认20
        
        返回:
            DataFrame: 高低价差比因子值
        
        说明:
            - 分子：累计日内波动幅度
            - 分母：累计成交量
            - 衡量每单位成交量带来的价格波动
            - 值越大表示市场微观结构越不稳定
        """
        # 计算日内波动
        daily_range = high_df - low_df
        
        # 计算累计值
        sum_range = daily_range.rolling(window=period).sum()
        sum_vol = vol_df.rolling(window=period).sum()
        
        # 计算比值
        factor = sum_range / sum_vol.replace(0, np.nan)
        
        return factor
    
    @staticmethod
    def D04_overnight_gap(close_df, open_df):
        """
        D04 开盘跳空 (Overnight Gap)
        ================================================================================
        公式: (Open - Delay(Close)) / Delay(Close)
        
        参数:
            close_df: 收盘价矩阵
            open_df: 开盘价矩阵
        
        返回:
            DataFrame: 开盘跳空因子值
        
        说明:
            - 衡量隔夜信息含量
            - 正值：跳空高开（利好）
            - 负值：跳空低开（利空）
            - 反映隔夜消息对市场的影响
        """
        # 昨日收盘价
        prev_close = close_df.shift(1)
        
        # 开盘跳空
        gap = (open_df - prev_close) / prev_close.replace(0, np.nan)
        
        return gap
    
    @classmethod
    def get_all_factors(cls, close_df, high_df=None, low_df=None, 
                        open_df=None, vol_df=None, amount_df=None, **kwargs):
        """
        一次性计算所有D类因子
        
        参数:
            close_df: 收盘价矩阵
            high_df: 最高价矩阵（用于D02, D03）
            low_df: 最低价矩阵（用于D02, D03）
            open_df: 开盘价矩阵（用于D02, D04）
            vol_df: 成交量矩阵（用于D03）
            amount_df: 成交额矩阵（用于D01）
            **kwargs: 其他可选参数
        
        返回:
            dict: 因子名称到因子DataFrame的字典
        """
        factors = {}
        
        # D01 Amihud变化率 - 需要成交额数据
        if amount_df is not None:
            factors['D01_Amihud变化率'] = cls.D01_amihud_change(
                close_df, amount_df, period=kwargs.get('period', 20))
        
        # D02 日内动量 - 需要OHLC数据
        if high_df is not None and low_df is not None and open_df is not None:
            factors['D02_日内动量'] = cls.D02_intraday_momentum(
                close_df, open_df, high_df, low_df)
        
        # D03 高低价差比 - 需要High、Low、Vol数据
        if high_df is not None and low_df is not None and vol_df is not None:
            factors['D03_高低价差比'] = cls.D03_high_low_range(
                close_df, high_df, low_df, vol_df, 
                period=kwargs.get('period', 20))
        
        # D04 开盘跳空 - 需要Open数据
        if open_df is not None:
            factors['D04_开盘跳空'] = cls.D04_overnight_gap(close_df, open_df)
        
        return factors
