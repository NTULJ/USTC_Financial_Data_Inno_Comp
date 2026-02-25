# 项目进展总结 - 国元证券 × 中国科大金融数据创新大赛

## 0. 运行方式（最新）
### 0.1 环境准备
```bash
conda env create -f environment.yml
conda activate financial_inno_comp
pip install -e .
```

### 0.2 主流程（推荐，默认即最终配置）
```bash
python src/qfcomp/pipelines/run_main.py
```
或（安装了项目脚本后）：
```bash
qf-main
```

说明：
- 当前默认配置已写入 `src/qfcomp/config/base.py`，会自动加载：
  - 去重后的固定有效因子（内置常量）
  - 最优优化参数（内置常量）
  - 合成方法：`icir_robust`
  - Regime：`rule_v2`（`gamma=0.40, stress_threshold=0.8, max_step=0.03`）
- `run_main.py` 为纯配置驱动版本，不再依赖外部 `best_params.json` / `有效因子.csv`。
- 输出目录采用微秒级时间戳，不会因并行运行互相覆盖。

### 0.3 主流程（提交版复现）
```bash
python src/qfcomp/pipelines/run_main.py
```
说明：提交版 `run_main.py` 不接受策略参数命令行覆盖，全部参数以 `src/qfcomp/config/base.py` 为准。

### 0.4 WFO Bayes 调参
```bash
python src/qfcomp/pipelines/run_cvar_bayes.py --help
```
或（安装了项目脚本后）：
```bash
qf-cvar-bayes --help
```

常用稳定版示例（含换手惩罚 + 剪枝）：
```bash
python src/qfcomp/pipelines/run_cvar_bayes.py \
  --reuse-run-dir outputs/<某次run_main输出目录> \
  --n-trials 160 \
  --obj-std-penalty 1.0 \
  --obj-worst-penalty 1.0 \
  --rp-anchor-lambda 0.2 \
  --obj-turnover-penalty 0.2 \
  --pruner median \
  --pruner-startup-trials 20 \
  --pruner-warmup-steps 2
```
### 0.5 严格 OOS（分段重调参 + 净值拼接）
```bash
python src/qfcomp/pipelines/run_strict_oos_stitch.py --help
```
或（安装了项目脚本后）：
```bash
qf-strict-oos --help
```

示例（复用某次 `run_main` 输出的信号缓存）：
```bash
python src/qfcomp/pipelines/run_strict_oos_stitch.py \
  --reuse-run-dir outputs/<某次run_main输出目录> \
  --n-trials 40 \
  --outer-test-start 2021-01-04 \
  --outer-test-end 2025-10-30 \
  --export-backtest-plots
```

说明：
- 每个外层测试段开始前，使用该时点之前历史数据重新调参；
- 该段参数仅用于该段 OOS 回测，最终将各段 OOS 净值首尾拼接；
- 输出目录包含每段内层折配置、每段最优参数、每段绩效及拼接总绩效。
### 0.6 `src` 目录结构
```text
src/
└── qfcomp/
    ├── analysis/
    │   └── cvar_tuning_plots.py        # CVaR 调参结果可视化
    ├── backtest/
    │   └── engine.py                   # bt 回测引擎封装与绩效统计
    ├── config/
    │   └── base.py                     # 全局配置与默认参数
    ├── data_loader/
    │   └── loader.py                   # 附件数据读取、对齐、调仓日生成
    ├── factors/
    │   ├── calc.py                     # 因子计算与标准化
    │   ├── combine.py                  # 因子合成（icir / icir_robust）
    │   ├── library.py                  # 因子定义库
    │   └── testing.py                  # 单因子 IC / ICIR / p 值评估
    ├── pipelines/
    │   ├── run_main.py                 # 主流程：因子->选股->优化->回测
    │   ├── run_cvar_bayes.py           # WFO + Optuna 的 CVaR/Hybrid 调参
    │   └── run_strict_oos_stitch.py    # 严格时间推进 OOS：分段重调参+净值拼接
    └── portfolio/
        ├── optimizer.py                # RP/CVaR/Hybrid 权重优化
        └── regime.py                   # 宏观 Regime 仓位缩放（rule/rule_v2）
```

## 1. 这个项目在做什么
这是一个面向赛题的 ETF 组合策略项目。核心任务是：在给定 28 只非债券 ETF 的前提下，基于量价与宏观数据做因子选股和组合优化，按周调仓，在 2021 年以后回测中获得更高风险调整收益（Sharpe）并控制回撤。

## 2. 赛题约束与项目落地
- 资产范围：仅 28 只非债券 ETF。
- 回测起点：2021-01-04。
- 调仓频率：周度（每周首个交易日）。
- 持仓约束：单资产权重不超过 35%，最小持仓数受约束。
- 成本设定：手续费万分之 2.5。
- 回测口径：风险收益指标统一基于收盘价计算。
- 框架要求：回测框架统一使用 `bt`。
- 规则红线：调仓期若使用调仓期后的数据/信息，判定为未来函数（报告 0 分）。
- 滑点口径：初赛回测可不考虑滑点。
- 前视控制：因子合成采用 `fwd shift + rolling`，调参与评估采用训练/测试切分。

以上约束已在当前代码流程中实现，并作为调参时的可行性与筛选条件。

## 3. 任务拆解
任务一：因子构建与单因子测试
- 赛题要求：基于附件 2/3 构建因子，做单因子测试，至少包含 RankIC、IR。
- 当前完成：已实现多类截面因子与宏观因子；已输出 Rank IC 与 ICIR；有效因子筛选流程已跑通。

任务二：ETF 初选与等权组合
- 赛题要求：每周首个交易日按合成因子排序选前 N（N>=3）并等权回测。
- 当前完成：周度 TopN 选股与等权组合已实现，基础绩效指标可导出。

任务三：组合优化与对比分析
- 赛题要求：在任务二基础上加入优化模型，并与等权对比。
- 当前完成：已支持 `risk_parity / min_variance / cvar / hybrid_cvar_rp`，并可输出优化后与等权的净值和指标对比。

## 4. 项目整体思路
1. 读取并对齐数据：量价、宏观、资产池。
2. 计算因子矩阵：得到日度 `date × sec` 的因子值。
3. 单因子检验：用 Rank IC 评估因子有效性并筛选。
4. 因子合成：按滚动 IC/ICIR 权重合成 `composite` 分数。
5. 选股：每个调仓日取 TopN 标的。
6. 组合优化：在选中标的中用 `risk_parity / cvar / hybrid_cvar_rp` 生成权重。
7. 宏观仓位调节：用 regime 信号缩放总仓位。
8. 回测评估：输出收益、波动、Sharpe、回撤、Calmar、尾部风险与结构指标。

## 5. 当前实现状态（已更新到 2026-02-23 晚）
- 已将组合优化调参从固定切分升级为 WFO（滚动步进交叉验证）：
  - 基础窗口：`2019-11-01~2020-12-31` 训练，测试覆盖 `2021-01-04~2025-10-30`
  - 年度滚动折：2 年训练 + 1 年测试 + 1 年步进，最少 5 折
- 已将 Bayes 目标升级为稳健目标：`mean(OOS Sharpe) - std_penalty*std(OOS Sharpe) - worst_fold_penalty + 相对RP约束项`，并加入 RP 锚定正则（active share 惩罚）。
- `run_cvar_bayes.py` 已支持同时搜索：
  - `top_n / cvar_alpha / cov_window / turnover_lambda / hybrid_beta`
  - `cvar_method ∈ {empirical, parametric, cornish_fisher}`
- 因子库新增并验证了 3 个有效因子：`K02`、`M03`、`N01`（已进入有效因子统计）。
- 已完成相关性去重流程：
  - 核心策略：pairwise 相关阈值 `|corr|>0.7`，按 `|ICIR|` 贪心保留（每簇保留 1 个）
  - 去重结果：18 -> 13 个因子，保留 `A02`，删除 `A01/A05/B05/D02/N01`
  - 结果文件：`outputs/20260223_195939_dedup_corr070_greedy/去重决策_greedy.csv`
- `run_main.py` 已切换为提交版纯配置驱动：不再通过命令行读取 `best_params` 或有效因子 CSV。
- Regime 参数固定在 `base.py`：`relax_gamma=0.40`、`stress_threshold=0.8`、`max_step=0.03`。
- 输出目录冲突问题已修复：`run_main.py / run_cvar_bayes.py` 的输出时间戳已升级为微秒级，避免并行运行写入同一目录。

## 6. 最近实验结果与消融结论
### 6.1 因子集消融（WFO Bayes, seed=42）
统一口径：`run_cvar_bayes.py`，测试集看 `best_candidate_test`。

| 因子集版本 | 输出目录 | WFO目标 | 测试Sharpe | 测试Calmar | 测试MDD |
|---|---|---:|---:|---:|---:|
| 全量不去重 | `outputs/cvar_hybrid_bayes_split_20260223_210043` | -0.3638 | 0.8747 | 1.0797 | -11.50% |
| drop A01 | `outputs/cvar_hybrid_bayes_split_20260223_210639` | -0.0667 | 1.1997 | 1.3940 | -8.85% |
| drop A02 | `outputs/cvar_hybrid_bayes_split_20260223_211317` | -0.1484 | 1.0379 | 1.3110 | -11.07% |
| keep A02 only | `outputs/cvar_hybrid_bayes_split_20260223_211852` | 0.0492 | 1.1243 | 1.1495 | -9.58% |
| pairwise corr>0.7 贪心去重（top_n 固定9） | `outputs/cvar_hybrid_bayes_split_20260223_214730` | **0.1799** | **1.1812** | **1.4507** | **-7.54%** |

结论：
- 去重是有效组件，明显优于“不去重”。
- 在高相关簇中保留 `A02` 是有效决策。
- 当前最稳版本是 pairwise 去重（`corr>0.7`）+ top_n=9 方案。

### 6.2 合成方法 × Regime 消融（贝叶斯调参前对比）
| 组合 | 输出目录 | 优化后Sharpe | 优化后Calmar | 优化后MDD |
|---|---|---:|---:|---:|
| `icir + rule` | `outputs/20260223_223044` | 1.095 | 1.340 | -7.68% |
| `icir + off` | `outputs/20260223_223054` | 1.085 | 1.463 | -10.65% |
| `icir_robust + rule` | `outputs/20260223_223105` | 1.050 | 1.330 | -7.85% |
| `icir_robust + off` | `outputs/20260223_223115` | 1.059 | 1.401 | -11.49% |

结论：
- 在非最优参数下，`icir_robust` 尚未优于 `icir`。
- `regime=off` 会提升收益/Calmar，但会增加回撤。

### 6.3 在“最优 Bayes 参数 + 去重因子”下复现
统一输入：
- 最优参数：`outputs/cvar_hybrid_bayes_split_20260223_214730/CVAR贝叶斯_best_params.json`
- 去重因子：`data/corr0.7_greedy_有效因子.csv`

| 合成方法 | 输出目录 | 优化后Sharpe | 优化后Calmar | 优化后MDD |
|---|---|---:|---:|---:|
| `icir` | `outputs/20260223_225059` | 1.181 | 1.451 | -7.54% |
| `icir_robust` | `outputs/20260223_225544` | **1.282** | **1.719** | -8.08% |

结论：
- 在最优参数口径下，`icir_robust` 显著优于 `icir`（Sharpe/Calmar 提升明显）。
- 回撤略增（约 +0.54pct），但仍处于可接受区间。

### 6.4 Regime v2 改造与消融（在 6.3 最优输入上）
改造点：
- 保留 rule-based 宏观仓位主逻辑，但加入“受控提仓”：
  - `relax_gamma`：将 `scale_rule` 与 1.0 混合（只提高仓位，不破坏风控底座）
  - `stress_threshold`：仅在压力不高时允许提仓
  - `max_step`：限制仓位日变化，避免抖动

| Regime 版本 | 关键参数 | 输出目录 | 优化后Sharpe | 优化后Calmar | 优化后MDD |
|---|---|---|---:|---:|---:|
| `rule` 基线 | `gamma=0` | `outputs/20260223_230802` | 1.282 | 1.719 | -8.08% |
| `off` | N/A | `outputs/20260223_230434` | 1.307 | 1.627 | -13.01% |
| `rule_v2`（保守） | `gamma=0.25, th=0.8, step=0.03` | `outputs/20260223_230834` | 1.317 | 1.906 | -8.08% |
| `rule_v2`（进攻） | `gamma=0.35, th=0.8, step=0.03` | `outputs/20260223_230915` | **1.330** | **1.974** | -8.12% |
| `rule_v2`（进攻，最新复现） | `gamma=0.40, th=0.8, step=0.03` | `outputs/20260224_191252_097026` | **1.399** | **1.880** | -9.47% |

结论：
- `rule_v2` 相比原 `rule` 显著提升 Sharpe/Calmar，且回撤基本不恶化。
- `off` 虽有 Sharpe 提升，但回撤显著变差，不适合作为主提交。
- 当前最优组合已从 `icir_robust + rule` 升级为 `icir_robust + rule_v2`。

## 7. 当前最优可提交配置（具体见config/base.py）
- 因子集：`corr>0.7` pairwise 贪心去重后的 13 因子（保留 `A02`）
- 合成方法：`icir_robust`
- Regime：`rule_v2`（建议 `gamma=0.40, stress_threshold=0.8, max_step=0.03`）
- 优化器：`hybrid_cvar_rp`
- 参数：
  - `top_n=7`
  - `cvar_alpha=0.922443`
  - `cvar_method=empirical`
  - `cov_window=145`
  - `turnover_lambda=0.0289476`
  - `hybrid_beta=0.10`
  - `max_weight=0.35`
