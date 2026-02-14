# -*- coding: utf-8 -*-
"""
================================================================================
B类因子：反转与均值回归因子
================================================================================

包含因子：
    B01 Alpha#006 - 量价相关性
    B02 Alpha#012 - 成交量变化与价格变化
    B03 Alpha#001 (GTJA191) - 标准化价格偏离
    B04 Alpha#013 - 量价协方差
    B05 KD指标(K值) - 随机指标
    B06 Alpha#041 - 加权价格偏离

研报来源：
    - 招商证券《Alpha101》系列
    - WorldQuant Alpha公式
    - 国泰君安《基于短周期价量特征》
    - 海通证券《KDJ的量化改进》

使用方法:
    from factor_lib.reversal import ReversalFactors
    
    factor = ReversalFactors.B01_alpha006(close_df, vol_df)
================================================================================
"""

import pandas as pd
import numpy as np


class ReversalFactors:
    """反转与均值回归因子计算器"""
    
    @staticmethod
    def B01_alpha006(close_df, vol_df, period=20):
        """
        B01 Alpha#006 - 量价相关性
        ================================================================================
        公式: -1 * Ts_Rank(Corr(Close, Vol, period))
        
        参数:
            close_df: 收盘价矩阵
            vol_df: 成交量矩阵
            period: 相关性计算周期，默认20
        
        返回:
            DataFrame: 因子值
        
        说明:
            - Ts_Rank: 时间序列排名，值越大表示近期排名越高
            - Corr(Close, Vol): 价格与成交量的相关系数
            - 负号反转：相关性越高，因子值越低
            - 捕捉量价背离的反转机会
        """
        # 计算收盘价与成交量的滚动相关系数
        corr = close_df.rolling(window=period).corr(vol_df)
        
        # 计算时间序列排名 (Ts_Rank)
        # 对每个日期，计算过去period天的排名百分位
        ts_rank = corr.rolling(window=period).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan,
            raw=False
        )
        
        # 乘以-1进行反转
        factor = -1 * ts_rank
        
        return factor
    
    @staticmethod
    def B02_alpha012(close_df, vol_df):
        """
        B02 Alpha#012 - 成交量变化与价格变化
        ================================================================================
        公式: Sign(Delta(Vol)) * -Delta(Close)
        
        参数:
            close_df: 收盘价矩阵
            vol_df: 成交量矩阵
        
        返回:
            DataFrame: 因子值
        
        说明:
            - Delta(Vol): 成交量变化 (今日-昨日)
            - Delta(Close): 价格变化 (今日-昨日)
            - Sign(Delta(Vol)): 成交量变化方向 (+1/-1)
            - 成交量放大且价格下跌 → 正向因子值
            - 成交量放大且价格上涨 → 负向因子值
            - 捕捉量价背离的反转信号
        """
        # 计算成交量变化
        delta_vol = vol_df.diff()
        
        # 计算价格变化
        delta_close = close_df.diff()
        
        # Sign(Delta(Vol)) * -Delta(Close)
        factor = np.sign(delta_vol) * (-delta_close)
        
        return factor
    
    @staticmethod
    def B03_alpha001_gtja(close_df, period=20):
        """
        B03 Alpha#001 (GTJA191) - 标准化价格偏离
        ================================================================================
        公式: (Close - MA_period) / Std(Close, period)
        
        参数:
            close_df: 收盘价矩阵
            period: 均线和标准差周期，默认20
        
        返回:
            DataFrame: 因子值
        
        说明:
            - MA_period: period日简单移动平均
            - Std(Close, period): period日价格标准差
            - 标准化处理消除波动率差异
            - 衡量当前价格相对均值的偏离程度
        """
        # 计算周期内均值
        ma = close_df.rolling(window=period).mean()
        
        # 计算周期内标准差
        std = close_df.rolling(window=period).std()
        
        # 标准化价格偏离
        factor = (close_df - ma) / std.replace(0, np.nan)
        
        return factor
    
    @staticmethod
    def B04_alpha013(close_df, vol_df, period=20):
        """
        B04 Alpha#013 - 量价协方差
        ================================================================================
        公式: Rank(Cov(Close, Vol))
        
        参数:
            close_df: 收盘价矩阵
            vol_df: 成交量矩阵
            period: 协方差计算周期，默认20
        
        返回:
            DataFrame: 因子值
        
        说明:
            - Cov(Close, Vol): 价格与成交量的协方差
            - Rank: 跨截面排名
            - 捕捉量价同向变动的趋势强度
        """
        # 计算滚动协方差
        cov = close_df.rolling(window=period).cov(vol_df)
        
        # 计算排名（跨截面）
        factor = cov.rank(axis=1, pct=True)
        
        return factor
    
    @staticmethod
    def B05_kd_stochastic(close_df, high_df, low_df, n=14, m=3):
        """
        B05 KD指标(K值) - 随机指标
        ================================================================================
        公式: (Close - Low_n) / (High_n - Low_n)
        
        其中：
            - Low_n: n日内最低价
            - High_n: n日内最高价
        
        参数:
            close_df: 收盘价矩阵
            high_df: 最高价矩阵
            low_df: 最低价矩阵
            n: 随机指标周期，默认14
            m: 平滑周期，默认3
        
        返回:
            DataFrame: K值
        
        说明:
            - K值 = (收盘价 - n日最低价) / (n日最高价 - n日最低价) * 100
            - 0-100波动，>80超买，<20超卖
            - 经典超买超卖指标
        """
        # 计算n日内最低价和最高价
        low_n = low_df.rolling(window=n).min()
        high_n = high_df.rolling(window=n).max()
        
        # 计算K值
        k_value = (close_df - low_n) / (high_n - low_n).replace(0, np.nan) * 100
        
        # 可选：平滑处理（K值的D值）
        # k_smooth = k_value.rolling(window=m).mean()
        
        return k_value
    
    @staticmethod
    def B06_vwap_deviation(close_df, vol_df, period=20):
        """
        B06 Alpha#041 - 加权价格偏离 (VWAP Deviation)
        ================================================================================
        公式: Close / VWAP - 1
        
        参数:
            close_df: 收盘价矩阵
            vol_df: 成交量矩阵
            period: VWAP计算周期，默认20
        
        返回:
            DataFrame: 因子值
        
        说明:
            - VWAP: 成交量加权平均价 = Sum(Price*Vol) / Sum(Vol)
            - 衡量收盘价相对日内平均成本的偏离
            - 正值表示收盘高于平均成本，可能有卖压
        """
        # 计算成交量加权平均价 (VWAP)
        # 方法：滚动计算
        price_vol = close_df * vol_df
        sum_price_vol = price_vol.rolling(window=period).sum()
        sum_vol = vol_df.rolling(window=period).sum()
        
        vwap = sum_price_vol / sum_vol.replace(0, np.nan)
        
        # 计算偏离度
        factor = (close_df / vwap.replace(0, np.nan)) - 1
        
        return factor
    
    @classmethod
    def get_all_factors(cls, close_df, high_df=None, low_df=None, vol_df=None, **kwargs):
        """
        一次性计算所有B类因子
        
        参数:
            close_df: 收盘价矩阵
            high_df: 最高价矩阵（用于B05）
            low_df: 最低价矩阵（用于B05）
            vol_df: 成交量矩阵（用于B01, B04, B06）
            **kwargs: 其他可选参数
        
        返回:
            dict: 因子名称到因子DataFrame的字典
        """
        factors = {}
        
        # B01 Alpha#006 - 需要成交量数据
        if vol_df is not None:
            factors['B01_Alpha006量价相关'] = cls.B01_alpha006(
                close_df, vol_df, period=kwargs.get('period', 20))
        
        # B02 Alpha#012 - 需要成交量数据
        if vol_df is not None:
            factors['B02_Alpha012量价变化'] = cls.B02_alpha012(close_df, vol_df)
        
        # B03 Alpha#001 (GTJA191)
        factors['B03_标准化价格偏离'] = cls.B03_alpha001_gtja(
            close_df, period=kwargs.get('period', 20))
        
        # B04 Alpha#013 - 需要成交量数据
        if vol_df is not None:
            factors['B04_Alpha013量价协方差'] = cls.B04_alpha013(
                close_df, vol_df, period=kwargs.get('period', 20))
        
        # B05 KD指标 - 需要high和low数据
        if high_df is not None and low_df is not None:
            factors['B05_KD指标'] = cls.B05_kd_stochastic(
                close_df, high_df, low_df, 
                n=kwargs.get('n', 14), 
                m=kwargs.get('m', 3))
        
        # B06 VWAP偏离 - 需要成交量数据
        if vol_df is not None:
            factors['B06_VWAP偏离'] = cls.B06_vwap_deviation(
                close_df, vol_df, period=kwargs.get('period', 20))
        
        return factors
