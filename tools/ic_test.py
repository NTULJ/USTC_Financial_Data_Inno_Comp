# -*- coding: utf-8 -*-
"""
================================================================================
IC测试工具
================================================================================
功能：
    - 计算未来收益率
    - 计算IC值和统计指标
    - 单因子测试
    - 批量因子测试
    - 结果汇总和可视化

使用方法:
    from tools.ic_test import run_single_factor_test, run_all_factor_tests
    
    # 单因子测试
    result = run_single_factor_test(factor_df, close_df, 'A01_波动率动量')
    
    # 批量测试
    results = run_all_factor_tests(factors_dict, close_df)
================================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats


def calculate_forward_returns(price_df, period=5):
    """
    计算未来N日收益率
    
    参数:
        price_df: 价格矩阵 (index=date, columns=sec)
        period: 预测期(默认5日)--预测未来几天的收益 (周度调仓则 period=5)
    
    返回:
        DataFrame: 未来收益率矩阵
    """
    # shift(-period) 将未来价格移到当前行
    forward_ret = price_df.shift(-period) / price_df - 1
    return forward_ret

def calculate_spearman_correlation(x, y):
    """
    手动计算斯皮尔曼相关系数(不依赖scipy)
    
    参数:
        x, y: 两个Series
    
    返回:
        float: 相关系数
    """
    # 计算秩次
    def rank_data(series):
        return series.rank(method='average')
    
    x_rank = rank_data(x)
    y_rank = rank_data(y)
    
    # 皮尔逊相关系数
    n = len(x_rank)
    if n < 2:
        return np.nan
    
    x_mean = x_rank.mean()
    y_mean = y_rank.mean()
    
    numerator = ((x_rank - x_mean) * (y_rank - y_mean)).sum()
    denominator = np.sqrt(((x_rank - x_mean)**2).sum() * ((y_rank - y_mean)**2).sum())
    
    if denominator == 0:
        return np.nan
    
    return numerator / denominator

def calculate_factor_ic(factor_df, forward_ret_df):
    """
    计算因子的IC序列(使用 scipy.stats.spearmanr)
    ================================================================================
    
    参数:
        factor_df: 因子值矩阵 (index=date, columns=sec)
        forward_ret_df: 未来收益率矩阵 (index=date, columns=sec)
    
    返回:
        DataFrame: IC序列 (index=date, columns=['ic', 'daily_p_value'])
    
    优化点:
        1. 使用 scipy 权威库计算相关系数和 p-value
        2. 增加对每日样本量的判断（样本太少算出来的 IC 没意义）
    """
    ic_data = []
    
    # 找到两个DataFrame索引的交集（确保日期对齐）
    common_index = factor_df.index.intersection(forward_ret_df.index)
    
    for date in common_index:
        # 1. 截取当日截面数据
        f_day = factor_df.loc[date]
        r_day = forward_ret_df.loc[date]
        
        # 2. 数据对齐与清洗 (非常关键！)
        # 只有当 因子 和 收益 都有值时，才参与计算
        valid_mask = ~(np.isnan(f_day) | np.isnan(r_day))
        f_valid = f_day[valid_mask]
        r_valid = r_day[valid_mask]
        
        # 3. 样本量检查
        # 比如少于 10 只 ETF，计算出来的相关性偶然性太大，建议跳过
        if len(f_valid) < 10: 
            continue
            
        # 4. 调用 scipy 计算 Spearman 相关系数
        # result.correlation 是 IC
        # result.pvalue 是 当日排名的随机性概率 (H0: 两者不相关)
        try:
            res = stats.spearmanr(f_valid, r_valid)
            ic = res.correlation
            p_val = res.pvalue
            
            # 极少数情况 spearmanr 可能返回 nan，做一层保护
            if not np.isnan(ic):
                ic_data.append({
                    'date': date, 
                    'ic': ic,
                    'daily_p_value': p_val  # 记录每日的显著性
                })
        except Exception as e:
            # 捕获可能的数学错误
            continue
            
    # 转换为 DataFrame
    if not ic_data:
        return pd.DataFrame()
        
    return pd.DataFrame(ic_data).set_index('date')

def calculate_factor_metrics(ic_df):
    """
    计算因子 IC 的综合统计指标（引入 T 检验）
    ================================================================================
    
    参数:
        ic_df: IC序列 DataFrame (index=date, columns=['ic'])
    
    返回:
        dict: 包含以下键值的字典
            - ic_mean: IC均值 (越高越好)
            - ic_std: IC波动 (越低越好)
            - ic_ir: 信息比率 (绝对值 > 0.5 优秀)
            - ic_positive_ratio: 胜率
            - ic_t_stat: T统计量 (通常绝对值 > 2 说明显著)
            - p_value: 长期显著性P值 (通常要求 < 0.05)
            - n_samples: 参与计算的期数
    
    优化点:
        1. 使用单样本 T 检验 (One-sample t-test) 计算真实的 t-stat 和 p-value
        2. 修正了 p-value 的含义：这里检验的是"IC均值是否显著不为0"
    """
    # 提取 IC 序列
    if ic_df.empty or 'ic' not in ic_df.columns:
        return {}
        
    ic_series = ic_df['ic']
    
    # 1. 基础指标
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    
    # IR (Information Ratio)
    # 注意：如果 std 为 0，IR 会报错，需处理
    if ic_std == 0:
        ic_ir = 0
    else:
        ic_ir = ic_mean / ic_std
        
    # IC 正比率 (胜率)
    ic_positive_ratio = (ic_series > 0).sum() / len(ic_series)
    
    # 2. 统计显著性检验 (T-Test)
    # H0 (原假设): IC 的均值等于 0 (因子无效)
    # H1 (备择假设): IC 的均值不等于 0 (因子有效)
    # 使用 scipy.stats.ttest_1samp
    if len(ic_series) > 1:
        t_stat, p_value = stats.ttest_1samp(ic_series, 0, nan_policy='omit')
    else:
        t_stat, p_value = np.nan, np.nan
        
    return {
        'ic_mean': ic_mean,                 # IC 均值 (越高越好)
        'ic_std': ic_std,                   # IC 波动 (越低越好)
        'ic_ir': ic_ir,                     # 稳健性 (绝对值 > 0.5 优秀)
        'ic_positive_ratio': ic_positive_ratio, # 胜率
        'ic_t_stat': t_stat,                # T 统计量 (通常绝对值 > 2 说明显著)
        'p_value': p_value,                 # 长期显著性 P 值 (通常要求 < 0.05)
        'n_samples': len(ic_series)         # 参与计算的期数
    }
    

def run_single_factor_test(factor_df, price_df, factor_name, period=5):
    """
    运行单因子测试
    
    参数:
        factor_df: 因子值矩阵
        price_df: 价格矩阵
        factor_name: 因子名称（用于显示）
        period: 预测期(默认5日)--预测未来几天的收益 (周度调仓则 period=5)
    
    返回:
        dict: 包含IC分析和因子数据的字典
    """
    print(f"\n{'='*60}")
    print(f"【单因子测试】{factor_name}")
    print(f"{'='*60}")
    
    # 计算未来收益率
    forward_ret = calculate_forward_returns(price_df, period)
    
    # 计算IC序列
    ic_df = calculate_factor_ic(factor_df, forward_ret)
    
    # 计算统计指标
    metrics = calculate_factor_metrics(ic_df)
    
    # 处理空结果的情况
    if not metrics or 'ic_mean' not in metrics:
        print(f"\n❌ 因子 {factor_name} IC计算失败：数据不足或计算异常")
        return {
            'factor_name': factor_name,
            'factor_df': factor_df,
            'ic_df': ic_df,
            'metrics': {},
            'forward_ret': forward_ret
        }
    
    # 打印结果
    print(f"\nIC分析结果:")
    print(f"  - IC均值: {metrics['ic_mean']:.4f} (标准: >0.03 为有效)")
    print(f"  - IC标准差: {metrics['ic_std']:.4f}")
    print(f"  - ICIR: {metrics['ic_ir']:.4f} (标准: >0.5 为优秀)")
    print(f"  - IC正比率: {metrics['ic_positive_ratio']*100:.1f}% (标准: >50%)")
    print(f"  - t统计量: {metrics['ic_t_stat']:.4f}")
    print(f"  - p值: {metrics['p_value']:.4f}")
    
    # 评估因子有效性--
    effectiveness = ""
    if metrics['ic_mean'] > 0.03:
        effectiveness += "✓ 因子有效 "
    else:
        effectiveness += "✗ 因子未达有效标准 "
    
    if metrics['ic_ir'] > 0.5:
        effectiveness += "✓ ICIR优秀"
    elif metrics['ic_ir'] > 0.3:
        effectiveness += "△ ICIR一般"
    else:
        effectiveness += "✗ ICIR较差"
    
    print(f"  - 有效性评估: {effectiveness}")
    
    return {
        'factor_name': factor_name,
        'factor_df': factor_df,
        'ic_df': ic_df,
        'metrics': metrics,
        'forward_ret': forward_ret
    }


def run_all_factor_tests(factors, close_df, period=5):
    """
    运行所有因子的测试
    
    参数:
        factors: 因子字典 {因子名称: 因子DataFrame}
        close_df: 收盘价矩阵
        period: 预测期
    
    返回:
        dict: 所有因子的测试结果
    """
    print("\n" + "=" * 60)
    print("【批量测试】开始运行所有因子测试...")
    print("=" * 60)
    
    results = {}
    
    for factor_name, factor_df in factors.items():
        result = run_single_factor_test(factor_df, close_df, factor_name, period)
        results[factor_name] = result
    
    return results


def print_summary_table(results):
    """
    打印汇总结果表格
    
    参数:
        results: 所有因子测试结果
    """
    print("\n" + "=" * 60)
    print("【汇总结果】因子IC表现排名")
    print("=" * 60)
    
    # 创建汇总表格
    summary_data = []
    for factor_name, result in results.items():
        metrics = result['metrics']
        # 跳过计算失败的因子
        if not metrics or 'ic_mean' not in metrics:
            print(f"⚠️ 跳过因子: {factor_name} (数据不足)")
            continue
        summary_data.append({
            '因子名称': factor_name,
            'IC均值': metrics.get('ic_mean', np.nan),
            'IC标准差': metrics.get('ic_std', np.nan),
            'ICIR': metrics.get('ic_ir', np.nan),
            '正比率': metrics.get('ic_positive_ratio', np.nan),
            'p值': metrics.get('p_value', np.nan)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('ICIR', ascending=False)
    
    # 格式化输出
    print(f"\n{'因子名称':<20} {'IC均值':>10} {'IC标准差':>10} {'ICIR':>10} {'正比率':>10} {'p值':>10}")
    print("-" * 70)
    
    for _, row in summary_df.iterrows():
        print(f"{row['因子名称']:<20} {row['IC均值']:>10.4f} {row['IC标准差']:>10.4f} "
              f"{row['ICIR']:>10.4f} {row['正比率']:>10.1%} {row['p值']:>10.4f}")
    
    print("\n评价标准:")
    print("  - IC均值 > 0.03: 因子有效")
    print("  - ICIR > 0.5: ICIR优秀 | 0.3-0.5: 一般 | <0.3: 较差")
    print("  - 正比率 > 50%: 方向稳定")
    print("  - p值 < 0.05: 统计显著")
    
    return summary_df
