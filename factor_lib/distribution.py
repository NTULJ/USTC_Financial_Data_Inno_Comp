# -*- coding: utf-8 -*-
"""
================================================================================
E类因子：统计分布因子
================================================================================

包含因子：
    E01 偏度 (Skewness)
    E02 峰度 (Kurtosis)

研报来源：
    - 申万宏源《高阶矩因子》

使用方法:
    from factor_lib.distribution import DistributionFactors
    
    factor = DistributionFactors.E01_skewness(close_df)
================================================================================
"""

import pandas as pd
import numpy as np


class DistributionFactors:
    """统计分布因子计算器"""
    
    @staticmethod
    def E01_skewness(close_df, period=20):
        """
        E01 偏度 (Skewness)
        ================================================================================
        公式: -1 * Skew(Ret, period)
        
        参数:
            close_df: 收盘价矩阵 (index=date, columns=sec)
            period: 计算周期，默认20
        
        返回:
            DataFrame: 偏度因子值
        
        说明:
            - Skew(Ret): 收益率分布的偏度
            - 正偏度：右尾长（更多极端正收益）
            - 负偏度：左尾长（更多极端负收益）
            - 乘以-1：让正偏度（好收益）对应正因子值
            - 反映收益分布的不对称性
        """
        # 计算日收益率
        ret = close_df.pct_change()
        
        # 计算滚动偏度
        skew = ret.rolling(window=period).skew()
        
        # 乘以-1进行反转
        factor = -1 * skew
        
        return factor
    
    @staticmethod
    def E02_kurtosis(close_df, period=20):
        """
        E02 峰度 (Kurtosis)
        ================================================================================
        公式: -1 * Kurt(Ret, period)
        
        参数:
            close_df: 收盘价矩阵 (index=date, columns=sec)
            period: 计算周期，默认20
        
        返回:
            DataFrame: 峰度因子值
        
        说明:
            - Kurt(Ret): 收益率分布的峰度（超额峰度）
            - 高峰度：尖峰肥尾（更多极端收益）
            - 低峰度：扁平分布
            - 乘以-1：让低峰度（稳定）对应正因子值
            - 反映收益分布的尾部厚度
        """
        # 计算日收益率
        ret = close_df.pct_change()
        
        # 计算滚动峰度（Fisher=True表示超额峰度）
        kurt = ret.rolling(window=period).kurt()
        
        # 乘以-1进行反转
        factor = -1 * kurt
        
        return factor
    
    @classmethod
    def get_all_factors(cls, close_df, **kwargs):
        """
        一次性计算所有E类因子
        
        参数:
            close_df: 收盘价矩阵
            **kwargs: 其他可选参数
        
        返回:
            dict: 因子名称到因子DataFrame的字典
        """
        factors = {}
        
        # E01 偏度
        factors['E01_偏度'] = cls.E01_skewness(
            close_df, period=kwargs.get('period', 20))
        
        # E02 峰度
        factors['E02_峰度'] = cls.E02_kurtosis(
            close_df, period=kwargs.get('period', 20))
        
        return factors
