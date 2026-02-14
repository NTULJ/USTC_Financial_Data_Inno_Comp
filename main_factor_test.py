#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
因子测试主程序 -- 【主程序】统一运行入口
================================================================================

功能：
    - 集中管理因子配置
    - 批量计算和测试因子

因子类别：
    A类: 趋势与动量因子
    B类: 均值回归因子
    C类: 波动率因子
    D类: 流动性因子
    E类: 情绪因子
    F类: 宏观因子
    G类: 行业因子
    H类: 事件因子

使用方法:
    python main_factor_test.py
    
    或在Jupyter中:
    %run main_factor_test.py
    
输出：
    - 各因子IC分析结果
    - 汇总排名表格
================================================================================
"""

# 导入因子库
from factor_lib.momentum import MomentumFactors
from factor_lib.reversal import ReversalFactors
from factor_lib.liquidity import LiquidityFactors
from factor_lib.distribution import DistributionFactors

# 导入工具箱
from tools.data_loader import load_price_data
from tools.ic_test import run_single_factor_test, run_all_factor_tests, print_summary_table


def main():
    """
    主函数：因子计算和测试入口
    """
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*   ETF因子测试系统                                            *")
    print("*   架构: 逻辑与运行分离                                        *")
    print("*   类别: A类(趋势动量) + B类(反转均值回归) + D类(流动性) + E类(统计分布)    *")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    
    # =========================================================================
    # 第一步：数据加载（只加载一次，节省时间）
    # =========================================================================
    close_df, high_df, low_df, open_df, vol_df, amount_df = load_price_data()
    
    # =========================================================================
    # 第二步：配置要测试的因子（配置中心）
    # =========================================================================
    # 格式：(因子代号, 因子函数, 需要的参数)
    
    factor_config = [
        # =========================================================================
        # A类：趋势与动量因子
        # =========================================================================
        ('A01_波动率缩放动量', 
         MomentumFactors.A01_vol_scaled_mom, 
         {'close_df': close_df, 'period': 20}),
        
        ('A02_路径效率', 
         MomentumFactors.A02_efficiency_ratio, 
         {'close_df': close_df, 'period': 20}),
        
        ('A03_时序动量加速器', 
         MomentumFactors.A03_momentum_accelerator, 
         {'close_df': close_df, 'short_period': 10, 'long_period': 20}),
        
        ('A04_MACD标准化', 
         MomentumFactors.A04_normalized_macd, 
         {'close_df': close_df, 'fast': 12, 'slow': 26}),
        
        ('A05_RSI', 
         MomentumFactors.A05_rsi, 
         {'close_df': close_df, 'period': 14}),
        
        ('A06_最高价突破', 
         MomentumFactors.A06_donchian_breakout, 
         {'close_df': close_df, 'high_df': high_df, 'period': 60}),
        
        # =========================================================================
        # B类：反转与均值回归因子
        # =========================================================================
        ('B01_Alpha006量价相关', 
         ReversalFactors.B01_alpha006, 
         {'close_df': close_df, 'vol_df': vol_df, 'period': 20}),
        
        ('B02_Alpha012量价变化', 
         ReversalFactors.B02_alpha012, 
         {'close_df': close_df, 'vol_df': vol_df}),
        
        ('B03_标准化价格偏离', 
         ReversalFactors.B03_alpha001_gtja, 
         {'close_df': close_df, 'period': 20}),
        
        ('B04_Alpha013量价协方差', 
         ReversalFactors.B04_alpha013, 
         {'close_df': close_df, 'vol_df': vol_df, 'period': 20}),
        
        ('B05_KD指标', 
         ReversalFactors.B05_kd_stochastic, 
         {'close_df': close_df, 'high_df': high_df, 'low_df': low_df, 'n': 14}),
        
        ('B06_VWAP偏离', 
         ReversalFactors.B06_vwap_deviation, 
         {'close_df': close_df, 'vol_df': vol_df, 'period': 20}),
        
        # =========================================================================
        # D类：微观结构与流动性因子
        # =========================================================================
        ('D01_Amihud变化率', 
         LiquidityFactors.D01_amihud_change, 
         {'close_df': close_df, 'amount_df': amount_df, 'period': 20}),
        
        ('D02_日内动量', 
         LiquidityFactors.D02_intraday_momentum, 
         {'close_df': close_df, 'open_df': open_df, 'high_df': high_df, 'low_df': low_df}),
        
        ('D03_高低价差比', 
         LiquidityFactors.D03_high_low_range, 
         {'close_df': close_df, 'high_df': high_df, 'low_df': low_df, 'vol_df': vol_df, 'period': 20}),
        
        ('D04_开盘跳空', 
         LiquidityFactors.D04_overnight_gap, 
         {'close_df': close_df, 'open_df': open_df}),
        
        # =========================================================================
        # E类：统计分布因子
        # =========================================================================
        ('E01_偏度', 
         DistributionFactors.E01_skewness, 
         {'close_df': close_df, 'period': 20}),
        
        ('E02_峰度', 
         DistributionFactors.E02_kurtosis, 
         {'close_df': close_df, 'period': 20}),
        
        # =========================================================================
        # 后续可以在这里无限添加 C, F, G, H 类因子
        # =========================================================================
    ]
    
    # =========================================================================
    # 第四步：循环计算因子
    # =========================================================================
    print(f"\n即将测试 {len(factor_config)} 个因子...")
    
    all_factors = {}
    all_results = {}
    
    for name, func, params in factor_config:
        print(f"\n>>> 正在计算: {name}")
        
        try:
            # 动态调用函数计算因子
            factor_value = func(**params)
            all_factors[name] = factor_value
            
            # 运行IC测试
            result = run_single_factor_test(factor_value, close_df, name)
            all_results[name] = result
            
        except Exception as e:
            print(f"❌ {name} 计算失败: {e}")
            import traceback
            traceback.print_exc()
    
    # =========================================================================
    # 第五步：打印汇总结果
    # =========================================================================
    summary_df = print_summary_table(all_results)
    
    # =========================================================================
    # 第六步：保存结果（可选）
    # =========================================================================
    # 保存因子值供后续策略使用
    # pd.to_pickle(all_factors, 'all_factors.pkl')
    
    # 保存测试结果
    # summary_df.to_csv('factor_ic_summary.csv', encoding='utf-8-sig')
    
    print("\n" + "=" * 70)
    print("【完成】因子测试全部完成！")
    print("=" * 70)
    
    return all_factors, all_results


def demo_single_factor():
    """
    演示：单独测试某个因子
    """
    # 加载数据
    close_df, high_df, low_df, open_df = load_price_data()
    
    # 计算单个因子
    factor = MomentumFactors.A01_vol_scaled_mom(close_df, period=20)
    
    # 运行测试
    result = run_single_factor_test(factor, close_df, 'A01_波动率缩放动量')
    
    return result


if __name__ == "__main__":
    # 运行主程序
    factors, results = main()
