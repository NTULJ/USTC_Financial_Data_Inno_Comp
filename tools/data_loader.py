# -*- coding: utf-8 -*-
"""
================================================================================
数据加载工具
================================================================================
功能：
    - 加载ETF价格数据
    - 数据清洗和预处理
    - 输出标准格式的数据矩阵

输出格式：
    close_df: 收盘价矩阵 (index=date, columns=sec)
    high_df: 最高价矩阵
    low_df: 最低价矩阵
    open_df: 开盘价矩阵
    
使用方法:
    from tools.data_loader import load_price_data
    
    close_df, high_df, low_df, open_df = load_price_data()
================================================================================
"""

import pandas as pd
import numpy as np


def load_price_data(data_path='data/附件2_ETF日频量价数据.csv'):
    """
    加载ETF价格数据并转换为宽格式矩阵
    
    参数:
        data_path: CSV文件路径，默认 'data/附件2_ETF日频量价数据.csv'
    
    返回:
        tuple: (close_df, high_df, low_df, open_df, vol_df, amount_df)
            - close_df: 收盘价矩阵 (index=date, columns=sec)
            - high_df: 最高价矩阵
            - low_df: 最低价矩阵
            - open_df: 开盘价矩阵
            - vol_df: 成交量矩阵
            - amount_df: 成交额矩阵
    """
    print("=" * 60)
    print("【数据加载】正在加载ETF日频量价数据...")
    print("=" * 60)
    
    # 读取原始CSV数据
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # 转换为宽格式（日期为行，ETF为列）
    close_df = df.pivot(index='date', columns='sec', values='close')
    high_df = df.pivot(index='date', columns='sec', values='high')
    low_df = df.pivot(index='date', columns='sec', values='low')
    open_df = df.pivot(index='date', columns='sec', values='open')
    vol_df = df.pivot(index='date', columns='sec', values='volume')
    amount_df = df.pivot(index='date', columns='sec', values='amount')
    
    # 数据清洗：去除全是NaN的ETF列
    valid_cols = close_df.notna().sum() > 0
    close_df = close_df.loc[:, valid_cols]
    high_df = high_df.loc[:, valid_cols]
    low_df = low_df.loc[:, valid_cols]
    open_df = open_df.loc[:, valid_cols]
    vol_df = vol_df.loc[:, valid_cols]
    amount_df = amount_df.loc[:, valid_cols]
    
    print(f"✓ 数据加载完成")
    print(f"  - 日期范围: {close_df.index[0].date()} 到 {close_df.index[-1].date()}")
    print(f"  - 交易日数量: {len(close_df)}")
    print(f"  - ETF数量: {len(close_df.columns)}")
    print(f"  - ETF代码: {list(close_df.columns)}")
    
    return close_df, high_df, low_df, open_df, vol_df, amount_df


def load_macro_data(data_path='data/附件3_高频经济指标.csv'):
    """
    加载宏观数据
    
    参数:
        data_path: CSV文件路径
    
    返回:
        DataFrame: 宏观指标矩阵
    """
    print("=" * 60)
    print("【数据加载】正在加载宏观数据...")
    print("=" * 60)
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    print(f"✓ 宏观数据加载完成")
    print(f"  - 日期范围: {df.index[0].date()} 到 {df.index[-1].date()}")
    print(f"  - 指标数量: {len(df.columns)}")
    print(f"  - 指标列表: {list(df.columns)}")
    
    return df


def prepare_factor_inputs(close_df, high_df=None, **kwargs):
    """
    预计算因子计算所需的常用输入数据
    
    参数:
        close_df: 收盘价矩阵
        high_df: 最高价矩阵（可选）
        **kwargs: 其他参数
    
    返回:
        dict: 预计算的数据字典
    """
    inputs = {}
    
    # 日收益率
    inputs['ret'] = close_df.pct_change()
    
    # 对数收益率
    inputs['log_ret'] = np.log(close_df / close_df.shift(1))
    
    # 成交量变化率
    # inputs['vol_change'] = ...
    
    # 可以继续添加其他预计算数据...
    
    return inputs


if __name__ == "__main__":
    # 测试数据加载
    close_df, high_df, low_df, open_df = load_price_data()
    print(f"\n收盘价矩阵形状: {close_df.shape}")
    print(close_df.head())
