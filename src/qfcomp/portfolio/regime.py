# -*- coding: utf-8 -*-
"""
宏观 Regime 仓位调节模块
========================
职责：
  利用宏观因子 F01–F06 判定市场环境（Regime），动态调节整体仓位水平。
  
  宏观因子的特征是"时间序列信号"——同一天所有 ETF 的值相同，
  因此不能用于截面选股，但非常适合做仓位择时（Position Sizing）。

设计逻辑：
  1. 每个宏观因子映射为一个仓位调节信号 ∈ [0, 1]
     - F01 (VIX恐慌): 高恐慌 → 减仓；对应 sigmoid(-F01)
     - F02 (信用利差): 利差走扩 → 减仓；对应 sigmoid(-F02)
     - F03 (期限利差): 曲线陡峭 → 经济扩张 → 加仓；对应 sigmoid(F03)
     - F04 (通胀剪刀差): PPI > CPI → 企业利润承压 → 减仓；对应 sigmoid(-F04)
     - F05 (大小盘溢价): 方向性不明确，不用于仓位；权重 = 0
     - F06 (中美利差): 中国利率高 → 资金流入 → 加仓；对应 sigmoid(F06)
  2. 对各信号取加权平均，得到总仓位系数 position_scale ∈ [min_pos, 1.0]
  3. 将 position_scale 应用到调仓计划的权重上：w_new = w_old * scale（余量为现金）

防前视偏差：
  - 宏观因子在 factors.library 中使用 expanding z-score 标准化
  - 信号平滑仅使用历史窗口，不使用未来观测
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------

# 各宏观因子的仓位调节方向与权重
# direction: +1 表示因子值越高越乐观（加仓），-1 表示越高越悲观（减仓）
# weight: 该因子在仓位决策中的权重（0 表示不使用）
MACRO_SIGNAL_CONFIG = {
    "F01": {"direction": -1, "weight": 0.30},  # VIX 高 → 减仓
    "F02": {"direction": -1, "weight": 0.25},  # 信用利差走扩 → 减仓
    "F03": {"direction": +1, "weight": 0.15},  # 期限利差陡 → 扩张 → 加仓
    "F04": {"direction": -1, "weight": 0.10},  # PPI-CPI 通胀剪刀差大 → 减仓
    "F05": {"direction":  0, "weight": 0.00},  # 大小盘溢价 → 不用于仓位
    "F06": {"direction": +1, "weight": 0.20},  # 中美利差高 → 资金流入 → 加仓
}

# 仓位调节范围
MIN_POSITION_SCALE = 0.3    # 最低仓位（极端恐慌时仍保留 30%）
MAX_POSITION_SCALE = 1.0    # 最高仓位（满仓）

# Sigmoid 平滑参数
SIGMOID_K = 1.5             # 控制 sigmoid 陡峭度（越大越敏感）


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray, k: float = None) -> np.ndarray:
    """
    Sigmoid 映射：将 z-score 映射到 (0, 1)
    k 控制陡峭度：k=1 为标准 sigmoid，k 越大越敏感
    """
    k = k or SIGMOID_K
    return 1.0 / (1.0 + np.exp(-k * x))


def calc_macro_signals(macro_df: pd.DataFrame,
                       config: dict = None,
                       smooth_window: int = 5) -> pd.DataFrame:
    """
    将宏观因子转换为仓位信号。

    Parameters
    ----------
    macro_df : pd.DataFrame
        宏观因子矩阵 (index=date, columns=[F01..F06])，值为 z-score
    config : dict
        各因子的方向和权重配置
    smooth_window : int
        信号平滑窗口（移动平均天数），避免日频噪声导致频繁调仓

    Returns
    -------
    pd.DataFrame
        columns=['signal_F01', ..., 'signal_F06', 'composite_signal']
        每行为当日的仓位调节信号 ∈ (0, 1)
    """
    config = config or MACRO_SIGNAL_CONFIG
    result = pd.DataFrame(index=macro_df.index)

    for factor_name, cfg in config.items():
        if factor_name not in macro_df.columns:
            continue
        direction = cfg["direction"]
        if direction == 0:
            continue

        raw = macro_df[factor_name].copy()
        # 平滑处理：减少日频噪声
        smoothed = raw.rolling(smooth_window, min_periods=1).mean()
        # 按方向映射：direction * z-score → sigmoid
        signal = _sigmoid(direction * smoothed)
        result[f"signal_{factor_name}"] = signal

    return result


def calc_position_scale(macro_df: pd.DataFrame,
                        config: dict = None,
                        smooth_window: int = 5,
                        min_scale: float = None,
                        max_scale: float = None) -> pd.Series:
    """
    计算每日仓位缩放系数 ∈ [min_scale, max_scale]

    Parameters
    ----------
    macro_df : pd.DataFrame
        宏观因子矩阵 (index=date, columns=[F01..F06])
    config : dict
        各因子的方向和权重配置
    smooth_window : int
        信号平滑窗口
    min_scale : float
        最低仓位比例
    max_scale : float
        最高仓位比例

    Returns
    -------
    pd.Series
        index=date, values=仓位缩放系数
    """
    config = config or MACRO_SIGNAL_CONFIG
    min_scale = min_scale if min_scale is not None else MIN_POSITION_SCALE
    max_scale = max_scale if max_scale is not None else MAX_POSITION_SCALE

    signals = calc_macro_signals(macro_df, config, smooth_window)

    if signals.empty:
        return pd.Series(max_scale, index=macro_df.index, name="position_scale")

    # 加权平均
    weights = []
    factor_cols = []
    for factor_name, cfg in config.items():
        col = f"signal_{factor_name}"
        if col in signals.columns and cfg["weight"] > 0:
            weights.append(cfg["weight"])
            factor_cols.append(col)

    if not factor_cols:
        return pd.Series(max_scale, index=macro_df.index, name="position_scale")

    w = np.array(weights)
    w = w / w.sum()  # 归一化

    # 加权合成信号 ∈ (0, 1)
    composite = signals[factor_cols].values @ w
    composite = pd.Series(composite, index=signals.index, name="composite_signal")

    # 线性映射到 [min_scale, max_scale]
    position_scale = min_scale + (max_scale - min_scale) * composite
    position_scale.name = "position_scale"

    return position_scale


def apply_position_scale(schedule: dict,
                         position_scale: pd.Series) -> dict:
    """
    将仓位缩放系数应用到调仓计划上。

    权重乘以 scale 后，总仓位 = scale（剩余部分为现金），
    单个 ETF 的相对权重不变。

    Parameters
    ----------
    schedule : dict
        {date: pd.Series(weights)}，权重和为 1.0
    position_scale : pd.Series
        每日仓位缩放系数

    Returns
    -------
    dict
        调节后的调仓计划，权重和 = position_scale[date]
    """
    adjusted = {}
    for dt, weights in schedule.items():
        if dt in position_scale.index:
            scale = position_scale.loc[dt]
        else:
            # 找最近的已知日期（向前查找，避免前视偏差）
            valid_dates = position_scale.index[position_scale.index <= dt]
            if len(valid_dates) > 0:
                scale = position_scale.loc[valid_dates[-1]]
            else:
                scale = MAX_POSITION_SCALE  # 无数据时满仓

        # 确保 scale 在合理范围内
        scale = np.clip(scale, MIN_POSITION_SCALE, MAX_POSITION_SCALE)
        adjusted[dt] = weights * scale

    return adjusted


def summarize_regime(position_scale: pd.Series,
                     start_date: str = None) -> dict:
    """
    输出 Regime 状态统计摘要

    Returns
    -------
    dict
        包含均值、中位数、最小值、最大值、各区间占比
    """
    if start_date:
        ps = position_scale.loc[position_scale.index >= pd.Timestamp(start_date)]
    else:
        ps = position_scale

    return {
        "mean": ps.mean(),
        "median": ps.median(),
        "min": ps.min(),
        "max": ps.max(),
        "pct_below_50": (ps < 0.5).mean(),
        "pct_50_80": ((ps >= 0.5) & (ps < 0.8)).mean(),
        "pct_above_80": (ps >= 0.8).mean(),
    }


__all__ = [
    "calc_position_scale",
    "apply_position_scale",
    "summarize_regime",
    "calc_macro_signals",
    "MACRO_SIGNAL_CONFIG",
]
