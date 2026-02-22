# -*- coding: utf-8 -*-
"""
CVaR/RP 混合优化器贝叶斯调参（固定训练/测试日期切分）
===================================================
目标：
  1) 在给定训练集区间做参数搜索（Bayesian TPE）
  2) 固定参数后在给定测试集区间做 OOS 验证
  3) 输出调参明细、最优参数、训练/测试对比与可视化

说明：
  - 不使用 walk-forward 多折逻辑，按固定日期切分。
  - 候选策略使用 hybrid_cvar_rp（CVaR 与风险平价凸组合）。
  - max_weight 固定 35%，不参与搜索。
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from qfcomp.analysis.cvar_tuning_plots import generate_cvar_bayes_plots
from qfcomp.config import (
    BACKTEST_START,
    COV_LOOKBACK,
    FORWARD_RETURN_PERIODS,
    MIN_HOLDINGS,
    OUTPUT_DIR,
    TOP_N,
)
from qfcomp.data_loader.loader import load_all
from qfcomp.factors.calc import compute_factors, prepare_factor_matrices
from qfcomp.factors.combine import combine_factors, export_composite_factor
from qfcomp.factors.testing import test_all_factors, select_effective_factors_from_ic
from qfcomp.portfolio.regime import apply_position_scale, calc_position_scale

# 约束口径（与项目里现有评估保持一致）
MDD_WORSE_TOL = 0.005
ANN_RETURN_DROP_TOL = 0.01
FULL_SAMPLE_SHARPE_IMPROVE = 0.02
ABS_SPLIT_PERIODS = [
    ("2021_2022", "2021-01-04", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024_2025", "2024-01-01", "2025-12-31"),
]

# 用户要求：max_weight 固定 35%，不参与优化
FIXED_MAX_WEIGHT = 0.35


def _require_optuna():
    try:
        import optuna  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "缺少依赖 optuna，无法运行贝叶斯调参。请先安装: `pip install optuna`。"
        ) from exc
    return optuna


def _build_signals_fresh(
    run_dir: Path,
    close_matrix: pd.DataFrame,
    aligned: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    print("=" * 60)
    print("Step 1: 因子计算与合成（fresh）")
    panel_wide, macro_df = compute_factors(aligned)
    processed = prepare_factor_matrices(panel_wide, method="rank")
    _, ic_series_dict = test_all_factors(processed, close_matrix)

    fwd_shift = FORWARD_RETURN_PERIODS[0]
    shifted_ic = {name: ic.shift(fwd_shift) for name, ic in ic_series_dict.items()}
    train_cutoff = pd.Timestamp(BACKTEST_START) - pd.Timedelta(days=1)
    effective, prestart_summary = select_effective_factors_from_ic(
        shifted_ic, cutoff=str(train_cutoff.date())
    )
    prestart_summary.to_csv(run_dir / "单因子测试结果.csv", index=False)
    pd.DataFrame({"因子": effective}).to_csv(
        run_dir / "有效因子.csv", index=False, encoding="utf-8-sig"
    )

    composite = combine_factors(processed, ic_series_dict, effective)
    export_composite_factor(composite, output_path=str(run_dir / "合成因子序列.csv"))

    position_scale = calc_position_scale(macro_df, smooth_window=5)
    ps_df = position_scale.to_frame("position_scale")
    ps_df.index.name = "date"
    ps_df.to_csv(run_dir / "宏观仓位系数.csv")
    return composite, position_scale


def _load_cached_signals(run_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    comp_path = run_dir / "合成因子序列.csv"
    ps_path = run_dir / "宏观仓位系数.csv"
    if not comp_path.exists():
        raise FileNotFoundError(f"未找到缓存合成因子文件: {comp_path}")
    if not ps_path.exists():
        raise FileNotFoundError(f"未找到缓存仓位系数文件: {ps_path}")

    comp_long = pd.read_csv(comp_path, parse_dates=["date"])
    composite = comp_long.pivot(index="date", columns="sec", values="composite_score").sort_index()

    ps_df = pd.read_csv(ps_path, parse_dates=["date"])
    if "position_scale" not in ps_df.columns:
        raise ValueError(f"{ps_path} 缺少列 `position_scale`")
    position_scale = ps_df.set_index("date")["position_scale"].sort_index()
    return composite, position_scale


def _tail_metrics(nav: pd.Series, q: float = 0.95) -> Dict[str, float]:
    ret = nav.pct_change().dropna()
    if ret.empty:
        return {"var95_loss": np.nan, "es95_loss": np.nan}
    loss = -ret
    var_q = loss.quantile(q)
    tail = loss[loss >= var_q]
    es_q = tail.mean() if len(tail) > 0 else np.nan
    return {"var95_loss": float(var_q), "es95_loss": float(es_q)}


def _split_sharpes_abs(nav: pd.Series, calc_performance_fn) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label, start, end in ABS_SPLIT_PERIODS:
        sub = nav.loc[(nav.index >= pd.Timestamp(start)) & (nav.index <= pd.Timestamp(end))]
        if len(sub) < 3:
            out[f"sharpe_{label}"] = np.nan
            continue
        out[f"sharpe_{label}"] = float(calc_performance_fn(sub)["sharpe"])
    return out


def _split_sharpe_ge_count_local(
    nav: pd.Series,
    rp_nav: pd.Series,
    calc_performance_fn,
    n_splits: int = 3,
) -> tuple[int, int]:
    nav = nav.dropna()
    rp_nav = rp_nav.dropna()
    if nav.empty or rp_nav.empty:
        return 0, 0

    idx = nav.index.intersection(rp_nav.index)
    if len(idx) < max(30, n_splits * 6):
        return 0, 0

    nav = nav.reindex(idx)
    rp_nav = rp_nav.reindex(idx)
    parts = np.array_split(np.arange(len(idx)), n_splits)
    ge_cnt = 0
    valid = 0
    for p in parts:
        if len(p) < 3:
            continue
        sn = float(calc_performance_fn(nav.iloc[p])["sharpe"])
        sr = float(calc_performance_fn(rp_nav.iloc[p])["sharpe"])
        if np.isfinite(sn) and np.isfinite(sr):
            valid += 1
            if sn >= sr:
                ge_cnt += 1
    return ge_cnt, valid


def _schedule_structure_metrics(
    raw_schedule: Dict[pd.Timestamp, pd.Series],
    adjusted_schedule: Dict[pd.Timestamp, pd.Series],
    max_weight: float,
) -> Dict[str, float]:
    if not raw_schedule:
        return {
            "avg_holdings": np.nan,
            "avg_hhi": np.nan,
            "avg_cap_hit_ratio": np.nan,
            "avg_turnover": np.nan,
        }

    holdings: List[float] = []
    hhi_vals: List[float] = []
    cap_hit_vals: List[float] = []
    for _, w in sorted(raw_schedule.items(), key=lambda kv: kv[0]):
        w_pos = w[w > 1e-10]
        holdings.append(float(len(w_pos)))
        if len(w_pos) == 0:
            continue
        w_norm = w_pos / w_pos.sum()
        hhi_vals.append(float((w_norm ** 2).sum()))
        cap_hit_vals.append(float((w_pos >= max_weight - 1e-8).mean()))

    turnover_vals: List[float] = []
    prev = None
    for _, w in sorted(adjusted_schedule.items(), key=lambda kv: kv[0]):
        cur = w.sort_index()
        if prev is not None:
            idx = prev.index.union(cur.index)
            turn = 0.5 * (cur.reindex(idx, fill_value=0.0) - prev.reindex(idx, fill_value=0.0)).abs().sum()
            turnover_vals.append(float(turn))
        prev = cur

    return {
        "avg_holdings": float(np.mean(holdings)) if holdings else np.nan,
        "avg_hhi": float(np.mean(hhi_vals)) if hhi_vals else np.nan,
        "avg_cap_hit_ratio": float(np.mean(cap_hit_vals)) if cap_hit_vals else np.nan,
        "avg_turnover": float(np.mean(turnover_vals)) if turnover_vals else np.nan,
    }


def _validate_schedule(
    schedule: Dict[pd.Timestamp, pd.Series],
    max_weight: float,
    min_holdings: int,
    tol: float = 1e-8,
) -> tuple[bool, int]:
    violations = 0
    for _, w in schedule.items():
        if abs(float(w.sum()) - 1.0) > tol:
            violations += 1
        if (w < -tol).any():
            violations += 1
        if (w > max_weight + tol).any():
            violations += 1
        if int((w > 1e-10).sum()) < min_holdings:
            violations += 1
    return violations == 0, violations


def _safe_float(v: object, default: float = float("nan")) -> float:
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return default


def _first_trading_on_or_after(index: pd.DatetimeIndex, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    pos = index.searchsorted(dt, side="left")
    if pos >= len(index):
        return None
    return index[pos]


def _last_trading_on_or_before(index: pd.DatetimeIndex, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    pos = index.searchsorted(dt, side="right") - 1
    if pos < 0:
        return None
    return index[pos]


def _filter_schedule_by_date(
    schedule: Dict[pd.Timestamp, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[pd.Timestamp, pd.Series]:
    return {dt: w for dt, w in schedule.items() if start <= dt <= end}


def _run_nav_for_period(
    close_matrix: pd.DataFrame,
    schedule: Dict[pd.Timestamp, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
    strategy_name: str,
    run_backtests_fn,
    extract_nav_fn,
) -> Optional[pd.Series]:
    period_schedule = _filter_schedule_by_date(schedule, start=start, end=end)
    if not period_schedule:
        return None

    px = close_matrix.loc[(close_matrix.index >= start) & (close_matrix.index <= end)]
    if px.empty:
        return None

    try:
        res = run_backtests_fn(px, {strategy_name: period_schedule})
        nav = extract_nav_fn(res, start_date=str(start.date())).get(strategy_name)
    except Exception:  # noqa: BLE001
        return None
    if nav is None or len(nav) < 3:
        return None
    nav = nav.loc[nav.index <= end]
    return nav if len(nav) >= 3 else None


def _nav_row(strategy: str, nav: pd.Series, calc_performance_fn) -> Dict[str, float | str]:
    return {
        "strategy": strategy,
        **calc_performance_fn(nav),
        **_tail_metrics(nav),
        **_split_sharpes_abs(nav, calc_performance_fn),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CVaR/RP 混合优化器贝叶斯调参（固定日期切分）")
    parser.add_argument("--top-n", type=int, default=TOP_N, help="每期选股数量 Top N（当 low/high 未指定时使用）")
    parser.add_argument("--top-n-low", type=int, default=3, help="TopN 搜索下界")
    parser.add_argument("--top-n-high", type=int, default=7, help="TopN 搜索上界")
    parser.add_argument("--top-n-step", type=int, default=1, help="TopN 搜索步长")
    parser.add_argument("--n-trials", type=int, default=120, help="贝叶斯 trial 数")
    parser.add_argument("--n-startup-trials", type=int, default=24, help="TPE 随机预热 trial 数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--timeout", type=int, default=0, help="调参超时秒数，0 表示不限制")

    parser.add_argument("--alpha-low", type=float, default=0.90)
    parser.add_argument("--alpha-high", type=float, default=0.985)
    parser.add_argument("--window-low", type=int, default=60)
    parser.add_argument("--window-high", type=int, default=220)
    parser.add_argument("--window-step", type=int, default=5)
    parser.add_argument("--lambda-low", type=float, default=1e-3)
    parser.add_argument("--lambda-high", type=float, default=3e-2)
    parser.add_argument("--beta-low", type=float, default=0.0, help="hybrid beta 下界（0=纯CVaR）")
    parser.add_argument("--beta-high", type=float, default=1.0, help="hybrid beta 上界（1=纯RP）")
    parser.add_argument("--beta-step", type=float, default=0.05, help="hybrid beta 步长")

    parser.add_argument("--train-start", type=str, default="2019-11-01", help="训练集起始日")
    parser.add_argument("--train-end", type=str, default="2020-12-31", help="训练集结束日")
    parser.add_argument("--test-start", type=str, default="2021-01-04", help="测试集起始日")
    parser.add_argument("--test-end", type=str, default="2025-10-30", help="测试集结束日（默认=最后可用交易日）")
    parser.add_argument("--min-train-days", type=int, default=220, help="训练集最少交易日")
    parser.add_argument("--local-splits", type=int, default=2, help="训练集鲁棒性检查分段数")

    parser.add_argument(
        "--reuse-run-dir",
        type=str,
        default="",
        help="可选：复用某次 run_main 输出目录（读取 合成因子序列.csv 与 宏观仓位系数.csv）",
    )
    parser.add_argument("--export-nav", action="store_true", help="导出训练/测试净值 CSV")
    parser.add_argument("--export-plots", action="store_true", help="导出调参可视化图")
    parser.add_argument(
        "--export-backtest-plots",
        action="store_true",
        help="导出测试集净值/回撤图（RP vs best_candidate）",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    optuna = _require_optuna()

    from qfcomp.backtest.engine import (
        build_optimized_schedule,
        calc_performance,
        export_backtest_plots,
        export_nav,
        extract_nav,
        run_backtests,
    )

    if args.alpha_low <= 0 or args.alpha_high >= 1 or args.alpha_low >= args.alpha_high:
        raise ValueError("alpha 搜索区间必须满足 0<low<high<1。")
    if args.window_low < 20 or args.window_high < args.window_low:
        raise ValueError("window 搜索区间非法。")
    if args.lambda_low <= 0 or args.lambda_high <= args.lambda_low:
        raise ValueError("turnover_lambda 搜索区间必须满足 0<low<high。")
    if not (0.0 <= args.beta_low <= args.beta_high <= 1.0):
        raise ValueError("beta 搜索区间必须满足 0<=low<=high<=1。")
    if args.beta_step <= 0:
        raise ValueError("beta_step 必须 > 0。")
    if args.n_trials <= 0:
        raise ValueError("n_trials 必须 > 0。")
    if args.min_train_days <= 0:
        raise ValueError("min_train_days 必须 > 0。")
    if args.local_splits <= 0:
        raise ValueError("local_splits 必须 > 0。")
    if args.top_n_low <= 0 or args.top_n_high < args.top_n_low or args.top_n_step <= 0:
        raise ValueError("TopN 搜索区间非法。")
    if args.top_n_low < MIN_HOLDINGS:
        raise ValueError(f"top_n_low 不能小于最小持仓数 MIN_HOLDINGS={MIN_HOLDINGS}。")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(OUTPUT_DIR) / f"cvar_hybrid_bayes_split_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {run_dir}")

    print("=" * 60)
    print("Step 0: 数据加载")
    data = load_all()
    close_matrix = data["close_matrix"].sort_index()

    reuse_run_dir = Path(args.reuse_run_dir).expanduser() if args.reuse_run_dir else None
    if reuse_run_dir is not None:
        print("=" * 60)
        print(f"Step 1: 复用信号缓存: {reuse_run_dir}")
        composite, position_scale = _load_cached_signals(reuse_run_dir)
    else:
        composite, position_scale = _build_signals_fresh(
            run_dir=run_dir,
            close_matrix=close_matrix,
            aligned=data["aligned"],
        )
    composite = composite.sort_index()
    position_scale = position_scale.sort_index()

    # 日期切分（映射到交易日）
    train_start_raw = pd.Timestamp(args.train_start)
    train_end_raw = pd.Timestamp(args.train_end)
    test_start_raw = pd.Timestamp(args.test_start)
    test_end_raw = pd.Timestamp(args.test_end) if args.test_end else pd.Timestamp(close_matrix.index.max())

    train_start = _first_trading_on_or_after(close_matrix.index, train_start_raw)
    train_end = _last_trading_on_or_before(close_matrix.index, train_end_raw)
    test_start = _first_trading_on_or_after(close_matrix.index, test_start_raw)
    test_end = _last_trading_on_or_before(close_matrix.index, test_end_raw)
    if any(x is None for x in [train_start, train_end, test_start, test_end]):
        raise ValueError("训练/测试日期不在有效交易日范围内，请检查输入。")

    train_start = pd.Timestamp(train_start)
    train_end = pd.Timestamp(train_end)
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)

    if not (train_start <= train_end < test_start <= test_end):
        raise ValueError(
            "日期切分非法，需满足 train_start <= train_end < test_start <= test_end。"
        )
    train_days = int(((close_matrix.index >= train_start) & (close_matrix.index <= train_end)).sum())
    if train_days < int(args.min_train_days):
        raise ValueError(
            f"训练集交易日不足: {train_days} < {int(args.min_train_days)}。"
        )

    split_df = pd.DataFrame(
        [
            {
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "train_days": train_days,
                "top_n_low": int(args.top_n_low),
                "top_n_high": int(args.top_n_high),
                "top_n_step": int(args.top_n_step),
                "fixed_max_weight": FIXED_MAX_WEIGHT,
                "min_holdings": MIN_HOLDINGS,
                "rp_cov_window": COV_LOOKBACK,
            }
        ]
    )
    split_path = run_dir / "CVAR贝叶斯_split_config.csv"
    split_df.to_csv(split_path, index=False)
    print("=" * 60)
    print(
        f"Step 2: 固定切分完成 "
        f"train[{train_start.date()}~{train_end.date()}], "
        f"test[{test_start.date()}~{test_end.date()}], train_days={train_days}"
    )

    rp_train_cache: Dict[int, Dict[str, object]] = {}

    def _get_rp_train_ref(top_n: int) -> Dict[str, object]:
        top_n = int(top_n)
        if top_n in rp_train_cache:
            return rp_train_cache[top_n]
        rp_train_raw_all = build_optimized_schedule(
            composite.loc[:train_end],
            close_matrix.loc[:train_end],
            optimizer="risk_parity",
            top_n=top_n,
            max_weight=FIXED_MAX_WEIGHT,
            min_holdings=MIN_HOLDINGS,
            cov_window=COV_LOOKBACK,
            rebal_start=train_start,
        )
        rp_train_raw = _filter_schedule_by_date(rp_train_raw_all, start=train_start, end=train_end)
        rp_train_adj = apply_position_scale(rp_train_raw, position_scale)
        rp_train_nav = _run_nav_for_period(
            close_matrix=close_matrix,
            schedule=rp_train_adj,
            start=train_start,
            end=train_end,
            strategy_name=f"rp_train_top{top_n}",
            run_backtests_fn=run_backtests,
            extract_nav_fn=extract_nav,
        )
        if rp_train_nav is None:
            raise ValueError(f"训练期 RP 净值为空，top_n={top_n} 无法调参。")
        rp_train_perf = calc_performance(rp_train_nav)
        ref = {
            "rp_nav": rp_train_nav,
            "mdd_limit": abs(float(rp_train_perf["max_drawdown"])) + MDD_WORSE_TOL,
            "ann_ret_limit": float(rp_train_perf["annual_return"]) - ANN_RETURN_DROP_TOL,
            "sharpe_target": float(rp_train_perf["sharpe"]) + FULL_SAMPLE_SHARPE_IMPROVE,
        }
        rp_train_cache[top_n] = ref
        return ref

    trial_rows: List[Dict[str, object]] = []

    def evaluate_candidate(
        trial_no: int,
        top_n: int,
        cvar_alpha: float,
        cov_window: int,
        turnover_lambda: float,
        hybrid_beta: float,
    ) -> Dict[str, object]:
        rec: Dict[str, object] = {
            "trial_number": int(trial_no),
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "top_n": int(top_n),
            "cvar_alpha": float(cvar_alpha),
            "cov_window": int(cov_window),
            "max_weight": FIXED_MAX_WEIGHT,
            "turnover_lambda": float(turnover_lambda),
            "hybrid_beta": float(hybrid_beta),
            "status": "ok",
            "error": "",
        }
        try:
            raw_all = build_optimized_schedule(
                composite.loc[:train_end],
                close_matrix.loc[:train_end],
                optimizer="hybrid_cvar_rp",
                top_n=top_n,
                max_weight=FIXED_MAX_WEIGHT,
                min_holdings=MIN_HOLDINGS,
                cov_window=int(cov_window),
                cvar_alpha=float(cvar_alpha),
                turnover_lambda=float(turnover_lambda),
                hybrid_beta=float(hybrid_beta),
                rebal_start=train_start,
            )
            raw_train = _filter_schedule_by_date(raw_all, start=train_start, end=train_end)
            adj_train = apply_position_scale(raw_train, position_scale)
            feasible_all, feasible_violations = _validate_schedule(
                raw_train,
                max_weight=FIXED_MAX_WEIGHT,
                min_holdings=MIN_HOLDINGS,
            )
            struct = _schedule_structure_metrics(raw_train, adj_train, max_weight=FIXED_MAX_WEIGHT)
            rec.update(
                {
                    "rebal_days": int(len(raw_train)),
                    "feasible_all_dates": bool(feasible_all),
                    "feasible_violations": int(feasible_violations),
                    **struct,
                }
            )
            if len(raw_train) == 0:
                rec["status"] = "empty_schedule"
                rec["objective"] = -1e6
                return rec

            train_nav = _run_nav_for_period(
                close_matrix=close_matrix,
                schedule=adj_train,
                start=train_start,
                end=train_end,
                strategy_name=f"cand_train_t{trial_no}",
                run_backtests_fn=run_backtests,
                extract_nav_fn=extract_nav,
            )
            if train_nav is None:
                rec["status"] = "empty_nav"
                rec["objective"] = -1e6
                return rec

            perf = calc_performance(train_nav)
            tail = _tail_metrics(train_nav)
            split_abs = _split_sharpes_abs(train_nav, calc_performance)
            rec.update({k: float(v) for k, v in perf.items()})
            rec.update({k: float(v) for k, v in tail.items()})
            rec.update({k: float(v) for k, v in split_abs.items()})

            rp_ref = _get_rp_train_ref(top_n)
            rp_train_nav = rp_ref["rp_nav"]
            rp_train_mdd_limit = float(rp_ref["mdd_limit"])
            rp_train_ann_ret_limit = float(rp_ref["ann_ret_limit"])
            rp_train_sharpe_target = float(rp_ref["sharpe_target"])

            hard_mdd_ok = abs(float(rec["max_drawdown"])) <= rp_train_mdd_limit
            hard_return_ok = float(rec["annual_return"]) >= rp_train_ann_ret_limit
            local_split_ge_cnt, local_split_valid = _split_sharpe_ge_count_local(
                nav=train_nav,
                rp_nav=rp_train_nav,
                calc_performance_fn=calc_performance,
                n_splits=int(args.local_splits),
            )
            local_split_rule_ok = (local_split_valid >= 2) and (local_split_ge_cnt >= 2)
            full_sample_sharpe_improve_ok = float(rec["sharpe"]) >= rp_train_sharpe_target
            robust_ok = local_split_rule_ok and full_sample_sharpe_improve_ok
            pass_all_rules = hard_mdd_ok and hard_return_ok and robust_ok
            rec.update(
                {
                    "hard_mdd_ok": bool(hard_mdd_ok),
                    "hard_return_ok": bool(hard_return_ok),
                    "local_split_ge_rp_count": int(local_split_ge_cnt),
                    "local_split_valid_segments": int(local_split_valid),
                    "local_split_rule_ok": bool(local_split_rule_ok),
                    "full_sample_sharpe_improve_ok": bool(full_sample_sharpe_improve_ok),
                    "robust_ok": bool(robust_ok),
                    "pass_all_rules": bool(pass_all_rules),
                }
            )

            obj = float(rec["sharpe"])
            if not feasible_all:
                obj -= 2.0 + 0.1 * float(feasible_violations)
            if not hard_mdd_ok:
                obj -= 5.0 + 100.0 * max(0.0, abs(float(rec["max_drawdown"])) - rp_train_mdd_limit)
            if not hard_return_ok:
                obj -= 5.0 + 100.0 * max(0.0, rp_train_ann_ret_limit - float(rec["annual_return"]))
            if local_split_valid < 2:
                obj -= 1.0
            elif local_split_ge_cnt < 2:
                obj -= 1.5 * float(2 - local_split_ge_cnt)
            if not full_sample_sharpe_improve_ok:
                obj -= 0.5
            avg_turnover = _safe_float(rec.get("avg_turnover"), default=np.nan)
            if np.isfinite(avg_turnover) and avg_turnover > 0.55:
                obj -= 0.4 * float(avg_turnover - 0.55)
            rec["objective"] = float(obj)
            return rec
        except Exception as exc:  # noqa: BLE001
            rec["status"] = "exception"
            rec["error"] = str(exc)
            rec["objective"] = -1e6
            return rec

    sampler = optuna.samplers.TPESampler(
        seed=int(args.seed),
        n_startup_trials=max(1, int(args.n_startup_trials)),
        multivariate=True,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial) -> float:
        top_n = trial.suggest_int(
            "top_n",
            int(args.top_n_low),
            int(args.top_n_high),
            step=int(args.top_n_step),
        )
        cvar_alpha = trial.suggest_float("cvar_alpha", args.alpha_low, args.alpha_high)
        cov_window = trial.suggest_int(
            "cov_window",
            int(args.window_low),
            int(args.window_high),
            step=max(1, int(args.window_step)),
        )
        turnover_lambda = trial.suggest_float(
            "turnover_lambda",
            float(args.lambda_low),
            float(args.lambda_high),
            log=True,
        )
        hybrid_beta = trial.suggest_float(
            "hybrid_beta",
            float(args.beta_low),
            float(args.beta_high),
            step=float(args.beta_step),
        )
        rec = evaluate_candidate(
            trial_no=trial.number,
            top_n=top_n,
            cvar_alpha=cvar_alpha,
            cov_window=cov_window,
            turnover_lambda=turnover_lambda,
            hybrid_beta=hybrid_beta,
        )
        trial_rows.append(rec)
        return float(rec["objective"])

    print("=" * 60)
    print(
        f"Step 3: 启动贝叶斯调参（optimizer=hybrid_cvar_rp, "
        f"fixed_max_weight={FIXED_MAX_WEIGHT:.2f}, "
        f"top_n_range=[{int(args.top_n_low)},{int(args.top_n_high)}], "
        f"n_trials={int(args.n_trials)}）"
    )
    study.optimize(
        objective,
        n_trials=int(args.n_trials),
        timeout=(None if int(args.timeout) <= 0 else int(args.timeout)),
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    if not trial_rows:
        raise ValueError("未产出任何 trial 结果。")

    trials_df = pd.DataFrame(trial_rows).sort_values(
        ["objective", "trial_number"], ascending=[False, True]
    ).reset_index(drop=True)
    trials_path = run_dir / "CVAR贝叶斯_trials.csv"
    trials_df.to_csv(trials_path, index=False)

    pass_df = trials_df[trials_df["pass_all_rules"].fillna(False).astype(bool)].copy()
    pass_path = run_dir / "CVAR贝叶斯_通过规则.csv"
    pass_df.to_csv(pass_path, index=False)

    top20 = trials_df.sort_values(["objective", "trial_number"], ascending=[False, True]).head(20)
    top20_path = run_dir / "CVAR贝叶斯_top20.csv"
    top20.to_csv(top20_path, index=False)

    if not pass_df.empty:
        best_row = pass_df.sort_values(
            ["objective", "sharpe", "calmar", "es95_loss", "avg_turnover"],
            ascending=[False, False, False, True, True],
        ).iloc[0]
        selected_from = "pass_all_rules"
    else:
        best_row = trials_df.iloc[0]
        selected_from = "objective_fallback"

    best_params = {
        "top_n": int(best_row["top_n"]),
        "cvar_alpha": float(best_row["cvar_alpha"]),
        "cov_window": int(best_row["cov_window"]),
        "max_weight": FIXED_MAX_WEIGHT,
        "turnover_lambda": float(best_row["turnover_lambda"]),
        "hybrid_beta": float(best_row["hybrid_beta"]),
    }

    rp_train_ref_best = _get_rp_train_ref(int(best_params["top_n"]))
    rp_train_nav = rp_train_ref_best["rp_nav"]

    # 用最优参数重跑训练集（用于输出对比与可选导出）
    cand_train_raw_all = build_optimized_schedule(
        composite.loc[:train_end],
        close_matrix.loc[:train_end],
        optimizer="hybrid_cvar_rp",
        top_n=best_params["top_n"],
        max_weight=FIXED_MAX_WEIGHT,
        min_holdings=MIN_HOLDINGS,
        cov_window=best_params["cov_window"],
        cvar_alpha=best_params["cvar_alpha"],
        turnover_lambda=best_params["turnover_lambda"],
        hybrid_beta=best_params["hybrid_beta"],
        rebal_start=train_start,
    )
    cand_train_raw = _filter_schedule_by_date(cand_train_raw_all, start=train_start, end=train_end)
    cand_train_adj = apply_position_scale(cand_train_raw, position_scale)
    cand_train_nav = _run_nav_for_period(
        close_matrix=close_matrix,
        schedule=cand_train_adj,
        start=train_start,
        end=train_end,
        strategy_name="cand_train_best",
        run_backtests_fn=run_backtests,
        extract_nav_fn=extract_nav,
    )
    if cand_train_nav is None:
        raise ValueError("最优参数在训练集的净值为空。")

    # 测试集 OOS：candidate vs RP
    cand_test_raw_all = build_optimized_schedule(
        composite.loc[:test_end],
        close_matrix.loc[:test_end],
        optimizer="hybrid_cvar_rp",
        top_n=best_params["top_n"],
        max_weight=FIXED_MAX_WEIGHT,
        min_holdings=MIN_HOLDINGS,
        cov_window=best_params["cov_window"],
        cvar_alpha=best_params["cvar_alpha"],
        turnover_lambda=best_params["turnover_lambda"],
        hybrid_beta=best_params["hybrid_beta"],
        rebal_start=test_start,
    )
    cand_test_raw = _filter_schedule_by_date(cand_test_raw_all, start=test_start, end=test_end)
    cand_test_adj = apply_position_scale(cand_test_raw, position_scale)
    cand_test_nav = _run_nav_for_period(
        close_matrix=close_matrix,
        schedule=cand_test_adj,
        start=test_start,
        end=test_end,
        strategy_name="cand_test_best",
        run_backtests_fn=run_backtests,
        extract_nav_fn=extract_nav,
    )
    if cand_test_nav is None:
        raise ValueError("最优参数在测试集的净值为空。")

    rp_test_raw_all = build_optimized_schedule(
        composite.loc[:test_end],
        close_matrix.loc[:test_end],
        optimizer="risk_parity",
        top_n=best_params["top_n"],
        max_weight=FIXED_MAX_WEIGHT,
        min_holdings=MIN_HOLDINGS,
        cov_window=COV_LOOKBACK,
        rebal_start=test_start,
    )
    rp_test_raw = _filter_schedule_by_date(rp_test_raw_all, start=test_start, end=test_end)
    rp_test_adj = apply_position_scale(rp_test_raw, position_scale)
    rp_test_nav = _run_nav_for_period(
        close_matrix=close_matrix,
        schedule=rp_test_adj,
        start=test_start,
        end=test_end,
        strategy_name="rp_test",
        run_backtests_fn=run_backtests,
        extract_nav_fn=extract_nav,
    )
    if rp_test_nav is None:
        raise ValueError("测试集 RP 净值为空。")

    train_rows = []
    if rp_train_nav is not None:
        train_rows.append(_nav_row("RP_baseline_train", rp_train_nav, calc_performance))
    train_rows.append(_nav_row("best_candidate_train", cand_train_nav, calc_performance))
    train_compare_df = pd.DataFrame(train_rows)
    train_compare_path = run_dir / "CVAR贝叶斯_训练集对比.csv"
    train_compare_df.to_csv(train_compare_path, index=False)

    test_compare_df = pd.DataFrame(
        [
            _nav_row("RP_baseline_test", rp_test_nav, calc_performance),
            _nav_row("best_candidate_test", cand_test_nav, calc_performance),
        ]
    )
    test_compare_path = run_dir / "CVAR贝叶斯_测试集对比.csv"
    test_compare_df.to_csv(test_compare_path, index=False)

    oos_sharpe_excess = float(test_compare_df.iloc[1]["sharpe"] - test_compare_df.iloc[0]["sharpe"])
    summary_df = pd.DataFrame(
        [
            {
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "selected_from": selected_from,
                **best_params,
                "train_best_objective": _safe_float(best_row.get("objective")),
                "train_best_sharpe": _safe_float(best_row.get("sharpe")),
                "train_pass_all_rules": bool(best_row.get("pass_all_rules", False)),
                "test_rp_sharpe": float(test_compare_df.iloc[0]["sharpe"]),
                "test_cand_sharpe": float(test_compare_df.iloc[1]["sharpe"]),
                "test_sharpe_excess": oos_sharpe_excess,
            }
        ]
    )
    summary_path = run_dir / "CVAR贝叶斯_摘要.csv"
    summary_df.to_csv(summary_path, index=False)

    best_params_payload = {
        "mode": "fixed_split_bayes_hybrid_cvar_rp",
        "train_start": str(train_start.date()),
        "train_end": str(train_end.date()),
        "test_start": str(test_start.date()),
        "test_end": str(test_end.date()),
        "fixed_max_weight": FIXED_MAX_WEIGHT,
        "optimizer": "hybrid_cvar_rp",
        "best_params": best_params,
        "selected_from": selected_from,
        "train_best_objective": _safe_float(best_row.get("objective")),
        "train_best_sharpe": _safe_float(best_row.get("sharpe")),
        "test_sharpe_excess_vs_rp": oos_sharpe_excess,
    }
    best_path = run_dir / "CVAR贝叶斯_best_params.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best_params_payload, f, ensure_ascii=False, indent=2)

    if args.export_nav:
        nav_map = {
            "best_candidate_train": cand_train_nav,
            "RP_baseline_test": rp_test_nav,
            "best_candidate_test": cand_test_nav,
        }
        if rp_train_nav is not None:
            nav_map["RP_baseline_train"] = rp_train_nav
        export_nav(nav_map, output_dir=run_dir)

    plot_paths: List[str] = []
    if args.export_plots:
        rp_train_row = train_compare_df[train_compare_df["strategy"] == "RP_baseline_train"]
        ref_row = rp_train_row.iloc[0] if not rp_train_row.empty else train_compare_df.iloc[0]
        rp_plot_ref = {k: float(v) for k, v in ref_row.items() if k != "strategy"}
        plot_paths = generate_cvar_bayes_plots(
            trials_df=trials_df,
            rp_ref=rp_plot_ref,
            output_dir=run_dir / "plots",
            top_k=10,
        )

    bt_plot_paths = {}
    if args.export_backtest_plots:
        bt_plot_paths = export_backtest_plots(
            {"RP_baseline": rp_test_nav, "best_candidate": cand_test_nav},
            output_dir=run_dir,
        )

    print("=" * 60)
    print("固定切分贝叶斯调参摘要")
    print(f"  训练区间: {train_start.date()} ~ {train_end.date()}")
    print(f"  测试区间: {test_start.date()} ~ {test_end.date()}")
    print(f"  trial 数: {len(trials_df)}")
    print(f"  通过全部规则数: {int(pass_df.shape[0])}")
    print(f"  最优来源: {selected_from}")
    print(f"  测试集 Sharpe 超额 (cand - RP): {oos_sharpe_excess:.4f}")
    print(f"  Split 配置: {split_path}")
    print(f"  Trials 明细: {trials_path}")
    print(f"  Top20: {top20_path}")
    print(f"  训练集对比: {train_compare_path}")
    print(f"  测试集对比: {test_compare_path}")
    print(f"  摘要: {summary_path}")
    print(f"  最优参数: {best_path}")
    if args.export_plots:
        print(f"  调参图数量: {len(plot_paths)}")
        print(f"  调参图目录: {run_dir / 'plots'}")
    if bt_plot_paths:
        print(f"  回测净值图: {bt_plot_paths.get('nav', '')}")
        print(f"  回测回撤图: {bt_plot_paths.get('drawdown', '')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
