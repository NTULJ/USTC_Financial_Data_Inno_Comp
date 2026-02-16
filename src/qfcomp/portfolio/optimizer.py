# -*- coding: utf-8 -*-
"""
组合优化模块
============
提供两类权重生成器：
  1) 最小方差权重
  2) 风险平价权重
均满足：
  - 无空仓约束：w >= 0
  - 单资产权重上限：max_weight
  - 权重和为 1
  - 至少持有 min_holdings 只资产
"""
import numpy as np
import pandas as pd
from scipy import optimize

from qfcomp.config import COV_LOOKBACK, SHRINKAGE_FACTOR, MAX_SINGLE_WEIGHT, MIN_HOLDINGS


# ---------------------------------------------------------------------------
# 协方差矩阵处理
# ---------------------------------------------------------------------------

def shrink_cov(cov: pd.DataFrame, shrinkage: float = None) -> pd.DataFrame:
    """对协方差矩阵做简单收缩（向对角线收缩），提升数值稳定性。"""
    shrinkage = SHRINKAGE_FACTOR if shrinkage is None else shrinkage
    cov_np = cov.to_numpy()
    diag = np.diag(np.diag(cov_np))
    shrunk = shrinkage * diag + (1 - shrinkage) * cov_np
    return pd.DataFrame(shrunk, index=cov.index, columns=cov.columns)


# ---------------------------------------------------------------------------
# 约束与后处理
# ---------------------------------------------------------------------------

def _post_process_weights(w: np.ndarray,
                          assets: list,
                          max_weight: float,
                          min_holdings: int) -> pd.Series:
    """
    - 去负数，按上限截断
    - 确保至少持有 min_holdings 只资产
    - 归一化到和为 1
    """
    n = len(w)
    if n == 0:
        return pd.Series(dtype=float)
    if max_weight * n < 1 - 1e-12:
        raise ValueError(f"不可行约束: n={n}, max_weight={max_weight}, 无法满足 sum(w)=1")

    w = np.maximum(w, 0.0)
    if w.sum() <= 0:
        w = np.ones_like(w)

    # 若持仓数不足，先在原权重最大的 min_holdings 只资产上分配初始权重
    if (w > 1e-10).sum() < min_holdings:
        k = min(min_holdings, n)
        idx = np.argsort(w)[-k:]
        base = np.zeros_like(w)
        base[idx] = 1.0 / k
        w = base

    w = _project_to_capped_simplex(w, max_weight=max_weight)

    # 数值稳健性检查
    if np.max(w) > max_weight + 1e-8:
        raise ValueError("权重后处理失败：存在超过 max_weight 的资产权重")
    if abs(w.sum() - 1.0) > 1e-8:
        raise ValueError("权重后处理失败：权重和不为 1")
    if (w > 1e-10).sum() < min_holdings:
        raise ValueError("权重后处理失败：持仓数不足 min_holdings")

    return pd.Series(w, index=assets)


def _project_to_capped_simplex(w: np.ndarray,
                               max_weight: float,
                               tol: float = 1e-12,
                               max_iter: int = 100) -> np.ndarray:
    """
    投影到约束集合：
      - w_i >= 0
      - w_i <= max_weight
      - sum(w) = 1
    """
    w = np.maximum(w.astype(float), 0.0)
    if w.sum() <= tol:
        w = np.ones_like(w, dtype=float)
    w = w / w.sum()

    for _ in range(max_iter):
        over = w > (max_weight + tol)
        if not over.any():
            break

        w[over] = max_weight
        free = ~over
        remain = 1.0 - w[over].sum()
        if remain <= 0:
            # 极端数值场景，退化为均匀后继续迭代
            w = np.ones_like(w, dtype=float) / len(w)
            continue

        free_sum = w[free].sum()
        if free_sum <= tol:
            w[free] = remain / max(1, free.sum())
        else:
            w[free] = w[free] / free_sum * remain

    # 最终数值修正
    w = np.clip(w, 0.0, max_weight)
    shortfall = 1.0 - w.sum()
    if abs(shortfall) > 1e-10:
        free = w < (max_weight - tol)
        if free.any():
            free_sum = w[free].sum()
            if free_sum <= tol:
                w[free] += shortfall / free.sum()
            else:
                w[free] += shortfall * (w[free] / free_sum)
        else:
            w = w / w.sum()
            w = np.minimum(w, max_weight)
            w = w / w.sum()

    return w


# ---------------------------------------------------------------------------
# 最小方差
# ---------------------------------------------------------------------------

def min_variance_weights(cov: pd.DataFrame,
                         max_weight: float = None,
                         min_holdings: int = None) -> pd.Series:
    """解最小方差优化：min w^T C w, s.t. sum w = 1, 0<=w<=max_weight"""
    max_weight = MAX_SINGLE_WEIGHT if max_weight is None else max_weight
    min_holdings = MIN_HOLDINGS if min_holdings is None else min_holdings

    n = len(cov)
    assets = list(cov.index)
    cov_np = cov.to_numpy()

    def obj(w):
        return w @ cov_np @ w

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
    bounds = [(0.0, max_weight) for _ in range(n)]
    x0 = np.ones(n) / n

    res = optimize.minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        # 失败时退化为等权并截断上限
        w = x0
    else:
        w = res.x

    return _post_process_weights(w, assets, max_weight, min_holdings)


# ---------------------------------------------------------------------------
# 风险平价
# ---------------------------------------------------------------------------

def risk_parity_weights(cov: pd.DataFrame,
                        max_weight: float = None,
                        min_holdings: int = None) -> pd.Series:
    """
    风险平价：最小化各资产风险贡献的相对离散程度。
    使用对数壁障 + 相对偏差目标，确保数值稳定。
    """
    max_weight = MAX_SINGLE_WEIGHT if max_weight is None else max_weight
    min_holdings = MIN_HOLDINGS if min_holdings is None else min_holdings

    n = len(cov)
    assets = list(cov.index)
    cov_np = cov.to_numpy()

    def obj(w):
        # 风险贡献 RC_i = w_i * (Σ w)_i
        cw = cov_np @ w
        rc = w * cw
        avg_rc = rc.mean()
        if avg_rc < 1e-20:
            return 0.0
        # 使用相对偏差（scale-invariant），避免极小绝对值导致的数值问题
        return ((rc / avg_rc - 1.0) ** 2).sum()

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
    bounds = [(0.0, max_weight) for _ in range(n)]
    x0 = np.ones(n) / n

    res = optimize.minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        w = x0
    else:
        w = res.x

    return _post_process_weights(w, assets, max_weight, min_holdings)


# ---------------------------------------------------------------------------
# 辅助：根据历史收益矩阵计算协方差
# ---------------------------------------------------------------------------

def compute_cov_from_returns(ret_window: pd.DataFrame,
                             shrinkage: float = None) -> pd.DataFrame:
    """从收益窗口计算协方差并收缩"""
    cov = ret_window.cov()
    return shrink_cov(cov, shrinkage)


__all__ = [
    "compute_cov_from_returns",
    "min_variance_weights",
    "risk_parity_weights",
]
