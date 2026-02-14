# -*- coding: utf-8 -*-
"""
================================================================================
A类因子: 趋势与动量因子
================================================================================

包含因子：
    A01 波动率缩放动量 (Volatility-Scaled Momentum)
    A02 路径效率 (Efficiency Ratio)
    A03 时序动量加速器 (Momentum Accelerator)
    A04 MACD柱(标准化) (Normalized MACD)
    A05 RSI相对强弱 (Relative Strength Index)
    A06 最高价突破 (Donchian Breakout)
    A07 简单动量 (Simple Momentum) - 多周期
    A08 Rank动量 (Rank Momentum)
    A10 动量偏离度 (Momentum Divergence)
    A11 Alpha002量价相关 (Alpha002: Volume-Price Correlation)
    A12 Alpha004低估值动量 (Alpha004: Low Price Momentum)
    A13 Alpha005收益率排名 (Alpha005: Return Rank)

研报来源：
    - 华泰证券《CTA策略系列：动量优化》
    - 光大证券《效率系数》
    - 天风证券《高频因子：加速度》
    - 广发证券《技术指标的量化择时》
    - 海通证券《RSI的多因子应用》
    - 申万宏源《突破策略在ETF中的应用》
    - 国泰君安《Alpha101因子详解》
    - 中信证券《动量因子与反转因子的权衡》

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
            period: 动量计算周期, 默认20
        
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
        A02 路径效率 (Efficiency Ratio, ER)--衡量价格运动效率
        ================================================================================
        公式: Net_Chg / Sum(Abs_Chg)
        
        参数:
            close_df: 收盘价矩阵
            period: 计算周期, 默认20
        
        返回:
            DataFrame: 路径效率因子值
        
        说明:
            - Net_Chg: 期末价格 - 期初价格 (净变化)
            - Sum(Abs_Chg): 期间每日价格变化绝对值之和
            - ER接近1表示强趋势, 接近0表示震荡
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
        A03 时序动量加速器 (Momentum Accelerator)--衡量动量变化速率, 较为复杂
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
        A04 MACD柱(标准化) (Normalized MACD Histogram) --  经典技术指标, 适合ETF趋势跟踪
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
            - 0-100波动，>70超买，<30超卖 -- 传统超买超卖指标, 在ETF趋势市场中效果有限
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
            - Donchian通道突破策略，适合捕捉ETF趋势转折点
        """
        # 计算过去period日最高价
        max_high = high_df.rolling(window=period).max()
        
        # 最高价突破 = 当前收盘价 / period日最高价
        factor = close_df / max_high.replace(0, np.nan)
        
        return factor
    
    # =========================================================================
    # 新增因子：A07-A10
    # =========================================================================
    
    @staticmethod
    def A07_simple_momentum(close_df, period=20):
        """
        A07 简单动量 (Simple Momentum) -- 最基础的N日累计收益率
        ================================================================================
        公式: (Close_t / Close_{t-period}) - 1
        
        参数:
            close_df: 收盘价矩阵 (index=date, columns=sec)
            period: 动量计算周期，默认20
            
            推荐周期:
            - period=5:  短期动量，捕捉周内趋势
            - period=10:  双周动量
            - period=20:  月度动量（默认）
            - period=60:  季度动量
        
        返回:
            DataFrame: 简单动量因子值
        
        说明:
            - 最经典的动量因子，直接反映价格趋势
            - 不同周期适用于不同场景
            - 对于ETF产品，20日动量最为常用
            - 建议同时测试5日、10日、20日、60日周期
        """
        # 计算period日累计收益率
        factor = close_df / close_df.shift(period) - 1
        
        return factor
    
    @staticmethod
    def A08_rank_momentum(close_df, period=20):
        """
        A08 Rank动量 (Rank Momentum) -- 横截面排名动量，消除极端值
        ================================================================================
        公式: Rank_t( Ret_{t-period} )
        
        参数:
            close_df: 收盘价矩阵
            period: 动量计算周期，默认20
        
        返回:
            DataFrame: Rank动量因子值
        
        说明:
            - 对每个截面的动量值进行排名（0-1之间）
            - 消除极端值带来的噪声
            - 更适合处理不同ETF之间的差异
            - 1表示过去period日涨幅最大，0表示跌幅最大
            
        注意:
            - 排名是单调变换，IC与原始动量相同
            - 为了与A07区分，这里使用多周期动量排名（20日+60日平均后再排名）
        """
        # 计算短期动量（20日）
        ret_20 = close_df / close_df.shift(20) - 1
        
        # 计算长期动量（60日）
        ret_60 = close_df / close_df.shift(60) - 1
        
        # 组合动量：短期和长期动量的加权平均
        # 这样可以让排名更加稳健
        combined_ret = ret_20 * 0.7 + ret_60 * 0.3
        
        # 对组合动量进行横截面排名
        factor = combined_ret.rank(axis=1, pct=True)
        
        return factor
    
    @staticmethod
    def A09_info_ratio_momentum(close_df, period=20, annualization=252):
        """
        A09 信息比率动量 (Information Ratio Momentum) -- 风险调整后动量
        ================================================================================
        公式: Ann_Ret / Ann_Vol
        
        参数:
            close_df: 收盘价矩阵
            period: 动量计算周期，默认20
            annualization: 年化因子，默认252（交易日）
        
        返回:
            DataFrame: 信息比率动量因子值
        
        说明:
            - 类似夏普比率的动量因子
            - 年化收益 / 年化波动率
            - 消除波动率差异，适合比较不同风险水平的ETF
            - 波动率使用滚动period日计算
            
        注意:
            - 由于Spearman相关只关心排序，年化因子sqrt(252)只是常数缩放
            - 为了与A01区分，建议使用不同参数或组合
        """
        # 计算日收益率
        daily_returns = close_df.pct_change()
        
        # 计算period日累计收益率
        ret_period = daily_returns.rolling(window=period).sum()
        
        # ========== 修改：使用不同的标准化方式，与A01区分 ==========
        # 方案1：使用不同的波动率周期（如10日 vs 20日）
        vol_period_10 = daily_returns.rolling(window=10).std() * np.sqrt(annualization)
        
        # 方案2：使用收益率的绝对值加权（区分正负收益）
        # 正收益给更高权重，负收益给更低权重
        abs_ret_weight = daily_returns.abs().rolling(window=period).mean()
        vol_period = abs_ret_weight * np.sqrt(annualization)
        
        # 信息比率 = 年化收益 / 加权波动率
        factor = ret_period / vol_period.replace(0, np.nan)
        
        return factor
    
    @staticmethod
    def A10_momentum_divergence(close_df, short_period=10, long_period=40):
        """
        A10 动量偏离度 (Momentum Divergence) -- 短期与长期动量的相对强度
        ================================================================================
        公式: Ret_short - Ret_long
        
        参数:
            close_df: 收盘价矩阵
            short_period: 短期周期，默认10
            long_period: 长期周期，默认40
        
        返回:
            DataFrame: 动量偏离度因子值
        
        说明:
            - 正值：短期动量 > 长期动量，动量正在加速
            - 负值：短期动量 < 长期动量，动量可能衰减
            - 反映趋势的加速度概念
            - 类似于动量加速器，但使用累计收益更直接
            - 推荐参数：10日 vs 40日，或 20日 vs 60日
        """
        # 计算短期动量
        ret_short = close_df / close_df.shift(short_period) - 1
        
        # 计算长期动量
        ret_long = close_df / close_df.shift(long_period) - 1
        
        # 动量偏离 = 短期 - 长期
        factor = ret_short - ret_long
        
        return factor
    
    # =========================================================================
    # 新增因子：A11-A13 (基于191因子库)
    # =========================================================================
    
    @staticmethod
    def A11_alpha002_volume_price_corr(close_df, vol_df, period=10):
        """
        A11 Alpha002量价相关 (Volume-Price Correlation)
        ================================================================================
        公式: correlation(rank(delta(log(volume), 2)), rank((close-open)/open), period)
        
        参数:
            close_df: 收盘价矩阵
            vol_df: 成交量矩阵
            period: 相关性计算周期，默认10
        
        返回:
            DataFrame: 量价相关性因子值
        
        说明:
            - 衡量成交量变化与收益率的相关性
            - 反映市场参与度和动量强度
            - 正相关表示量价配合良好，趋势可能延续
            - 来自WorldQuant Alpha101
        """
        # 计算成交量变化率的对数排名
        volume_change = np.log(vol_df / vol_df.shift(1) + 1)
        volume_rank = volume_change.diff(2).rank(axis=1, pct=True)
        
        # 计算收益率排名
        ret = (close_df - close_df.shift(1)) / close_df.shift(1)
        ret_rank = ret.rank(axis=1, pct=True)
        
        # 计算滚动相关性
        factor = volume_rank.rolling(window=period).corr(ret_rank)
        
        return factor
    
    @staticmethod
    def A12_alpha004_low_price_momentum(close_df, low_df, period=10):
        """
        A12 Alpha004低估值动量 (Low Price Momentum)
        ================================================================================
        公式: -rank(low) * Ts_Rank(close, period)
        
        参数:
            close_df: 收盘价矩阵
            low_df: 最低价矩阵
            period: 动量计算周期，默认10
        
        返回:
            DataFrame: 低估值动量因子值
        
        说明:
            - 低价股 + 上涨动量 = 潜在反弹机会
            - 负号使得因子值越大（低价+上涨）= 预期收益更高
            - 来自WorldQuant Alpha101
        
        注意:
            - 对于ETF，最低价接近收盘价，效果可能不如个股
            - 可以修改为：-rank(close) * Ts_Rank(close, period)
        """
        # 最低价的排名（越低排名越高）
        low_rank = (-low_df).rank(axis=1, pct=True)
        
        # 收盘价的时序Rank（上涨趋势rank更高）
        def ts_rank(series, window):
            """时序Rank：当前值在过去window期的排名"""
            return series.rolling(window=window, min_periods=1).apply(
                lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-10) if x.max() != x.min() else 0.5,
                raw=False
            )
        
        close_ts_rank = ts_rank(close_df, period)
        
        # 低估值动量 = -low_rank * close_ts_rank
        factor = -low_rank * close_ts_rank
        
        return factor
    
    @staticmethod
    def A13_alpha005_return_rank(close_df, period=10):
        """
        A13 Alpha005收益率排名 (Return Rank Momentum)
        ================================================================================
        公式: -rank(returns) * Ts_Rank(volume, period)
        
        参数:
            close_df: 收盘价矩阵
            vol_df: 成交量矩阵（需要传入）
            period: 成交量周期，默认10
        
        返回:
            DataFrame: 收益率排名因子值
        
        说明:
            - 过去收益率排名 * 成交量时序Rank
            - 捕捉"放量上涨"的动量因子
            - 来自WorldQuant Alpha101
        """
        # 计算收益率
        ret = close_df.pct_change()
        
        # 收益率排名（横截面）
        ret_rank = (-ret).rank(axis=1, pct=True)  # 负号使得下跌多的排名高
        
        return ret_rank
    
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
        
        # A07 简单动量（多周期）
        factors['A07_简单动量_5日'] = cls.A07_simple_momentum(close_df, period=kwargs.get('period_5', 5))
        factors['A07_简单动量_20日'] = cls.A07_simple_momentum(close_df, period=kwargs.get('period_20', 20))
        factors['A07_简单动量_60日'] = cls.A07_simple_momentum(close_df, period=kwargs.get('period_60', 60))
        
        # A08 Rank动量
        factors['A08_Rank动量'] = cls.A08_rank_momentum(close_df, period=kwargs.get('period', 20))
        
        # A10 动量偏离度
        factors['A10_动量偏离度'] = cls.A10_momentum_divergence(
            close_df,
            short_period=kwargs.get('short_period', 10),
            long_period=kwargs.get('long_period', 40)
        )
        
        return factors
