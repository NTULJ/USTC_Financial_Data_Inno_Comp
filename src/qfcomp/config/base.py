# -*- coding: utf-8 -*-
"""
全局配置参数
============
集中管理策略参数、文件路径、因子列表等，方便调参和复现。
"""
from pathlib import Path

# ========== 路径配置 ==========
# src/qfcomp/config/base.py -> 项目根目录为上三级
PROJECT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 原始数据文件
PRICE_FILE = DATA_DIR / "附件2 ETF日频量价数据（开盘、收盘、高、低、成交量、成交额）.csv"
MACRO_FILE = DATA_DIR / "附件3 高频经济指标（信用利差、期限利差、汇率等）.csv"
PRODUCT_POOL_FILE = DATA_DIR / "附件1 28只非债券ETF产品池.xlsx"

# ========== 回测参数 ==========
BACKTEST_START = "2021-01-04"          # 回测起始日（赛题硬性要求）
RISK_FREE_RATE = 0.0                   # 无风险利率（赛题要求 0%）
TRANSACTION_COST = 2.5 / 10000         # 交易手续费 万分之2.5
REBALANCE_FREQ = "W"                   # 调仓频率：周度
TOP_N = 5                              # 每期选取 Top N 只 ETF（≥3）
MAX_SINGLE_WEIGHT = 0.35               # 单只 ETF 权重上限 35%
MIN_HOLDINGS = 3                       # 最少持仓数量

# ========== 因子配置 ==========
# 截面因子（用于 ETF 排序）
PANEL_FACTORS = [
    "A01", "A02", "A03", "A04", "A05", "A06",
    "B01", "B02", "B03", "B04", "B05", "B06",
    "C01", "C02", "C03", "C04",
    "D01", "D02", "D03", "D04",
    "E01", "E02",
    "G01", "G02", "G03",
    "H01",
    "T01",
]

# 附加的辅助因子（收益/波动率，用于因子构造但不直接纳入合成）
AUX_FACTORS = ["ret_1", "ret_5", "ret_20", "vol_5", "vol_20"]

# 宏观因子（用于 Regime 分析，不参与截面排序）
MACRO_FACTORS = ["F01", "F02", "F03", "F04", "F05", "F06"]

# ========== 因子测试参数 ==========
FORWARD_RETURN_PERIODS = [5]           # 前瞻收益期限(交易日)：5日≈一周持有期
IC_ROLLING_WINDOW = 60                 # 滚动 IC 计算窗口（交易日）
IC_MIN_OBS = 10                        # 截面最少有效观测数

# ========== 因子合成参数 ==========
COMBINE_METHOD = "icir"                # 合成方法：ic / icir / equal
COMBINE_ROLLING_WINDOW = 60            # 合成权重的滚动窗口

# ========== 组合优化参数 ==========
COV_LOOKBACK = 60                      # 协方差矩阵估计窗口
SHRINKAGE_FACTOR = 0.5                 # Ledoit-Wolf 收缩系数（简化版）
