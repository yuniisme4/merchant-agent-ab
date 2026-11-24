"""用双重差分(DID)估计处理效应，并与仿真时植入的真值对照。

核心思路（"前后差分法"，和经典 2x2 DID 代数等价，但更直观）:
    对每个商户，分别算出 pre 期和 post 期的指标均值，
    diff_i = post_mean_i - pre_mean_i
再用 diff_i 对"是否在实验组"做回归：
    diff_i = a + b * is_treatment_i + e_i
b 就是 DID 估计出来的处理效应。

本脚本还做了三件进阶的事:
1. 回归调整 (CUPED 思路): 在上面的回归里加入 pre 期均值作为协变量，
   用干预前的信息解释掉一部分方差，让估计更精确（标准误更小）。
2. ITT vs CACE: ITT 把实验组里没采纳方案的商户也算进"实验组效应"里，
   数值会被稀释；CACE 用 Wald 估计 (ITT / 采纳率) 还原"真正采纳者"的效应，
   更接近仿真时设定的真值。
3. 安慰剂检验: 把 pre 期自己拆成两半，假装"前半 vs 后半"做一次 DID，
   这段时间根本没有任何干预，估计出来的效应应该接近 0——
   如果不接近 0，说明方法本身有问题（比如分流不平衡、估计量有偏）。

用法:
    python experiment/analyze_did.py --panel data/simulated/panel.csv \
        --merchants data/simulated/merchants.csv
"""
import argparse
import math

import numpy as np
import pandas as pd


def ols(y, X):
    """最小二乘回归，返回系数、标准误、95%置信区间、p值。
    只用 numpy，不依赖 scipy / statsmodels，方便初学者环境直接运行。
    """
    X = np.column_stack([np.ones(len(X)), X]) if X.ndim == 1 else \
        np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = (resid @ resid) / (n - k)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    # 标准正态分布的双侧 p 值，用标准库 math.erf 实现，不依赖 scipy
    pval = np.array([2 * (1 - 0.5 * (1 + math.erf(abs(zi) / 2 ** 0.5))) for zi in z])
    ci_lo = beta - 1.96 * se
    ci_hi = beta + 1.96 * se
    return beta, se, ci_lo, ci_hi, pval


def period_summary(panel, period):
    """把面板数据汇总成每个商户在某个period(pre/post)的均值。"""
    sub = panel[panel.period == period]
    g = sub.groupby("merchant_id")
    out = pd.DataFrame({
        "transactions": g["transactions"].mean(),
        "redemptions": g["redemptions"].sum() / g["transactions"].sum().replace(0, np.nan),
        "activity_score": g["activity_score"].mean(),
    })
    return out


def run_did(diff, is_treat, pre_baseline=None, label=""):
    """跑一次 DID 回归，pre_baseline 不为空时做回归调整（CUPED 思路）。"""
    mask = diff.notna() & is_treat.notna()
    if pre_baseline is not None:
        mask &= pre_baseline.notna()
        X = np.column_stack([is_treat[mask].values, pre_baseline[mask].values])
    else:
        X = is_treat[mask].values
    beta, se, lo, hi, p = ols(diff[mask].values, X)
    # beta[0] 是截距，beta[1] 才是我们关心的"是否实验组"的系数（即处理效应）
    print(f"  [{label}] 估计效应 = {beta[1]:.4f}  "
          f"95%CI=[{lo[1]:.4f}, {hi[1]:.4f}]  p={p[1]:.4f}  (n={mask.sum()})")
    return beta[1], se[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/simulated/panel.csv")
    ap.add_argument("--merchants", default="data/simulated/merchants.csv")
    ap.add_argument("--effect-size", type=float, default=0.15,
                    help="仿真时设定的交易量相对提升（用于计算真值，需与simulate时一致）")
    ap.add_argument("--coupon-effect", type=float, default=0.10,
                    help="仿真时设定的核销率相对提升（需与simulate时一致）")
    args = ap.parse_args()

    panel = pd.read_csv(args.panel)
    merchants = pd.read_csv(args.merchants).set_index("merchant_id")
    is_treat = (merchants["group"] == "treatment").astype(int)
    adoption_rate_in_treat = merchants.loc[merchants.group == "treatment", "adopted"].mean()
    adopters = merchants[merchants["adopted"] == 1]
    # 仿真真值：采纳者每月交易量提升 = effect_size * 其基线月均交易量
    true_trans = args.effect_size * adopters["baseline_rate"].mean()
    true_redeem = args.coupon_effect * adopters["redeem_prob"].mean()

    pre = period_summary(panel, "pre")
    post = period_summary(panel, "post")
    diff = post - pre  # 每个商户: post均值 - pre均值

    print("=" * 60)
    print("1) 交易量 (transactions)")
    print("-" * 60)
    itt_t, _ = run_did(diff["transactions"], is_treat, label="简单DID (ITT)")
    itt_t_adj, _ = run_did(diff["transactions"], is_treat, pre["transactions"],
                            label="回归调整DID (ITT, 含pre期协变量)")
    cace_t = itt_t_adj / adoption_rate_in_treat
    print(f"  [CACE，采纳者效应] = ITT / 采纳率({adoption_rate_in_treat:.2f}) = {cace_t:.4f}")
    print(f"  [真值对照] 采纳者效应真值 ≈ {true_trans:.4f}，"
          f"对应 ITT 真值 ≈ {true_trans * adoption_rate_in_treat:.4f}")

    print()
    print("2) 核销率 (redemptions / transactions)")
    print("-" * 60)
    itt_r, _ = run_did(diff["redemptions"], is_treat, label="简单DID (ITT)")
    itt_r_adj, _ = run_did(diff["redemptions"], is_treat, pre["redemptions"],
                            label="回归调整DID (ITT, 含pre期协变量)")
    cace_r = itt_r_adj / adoption_rate_in_treat
    print(f"  [CACE，采纳者效应] = ITT / 采纳率({adoption_rate_in_treat:.2f}) = {cace_r:.4f}")
    print(f"  [真值对照] 采纳者效应真值 ≈ {true_redeem:.4f}，"
          f"对应 ITT 真值 ≈ {true_redeem * adoption_rate_in_treat:.4f}")

    print()
    print("3) 活跃度分数 (activity_score，无公开数据支撑，仅作占位指标)")
    print("-" * 60)
    run_did(diff["activity_score"], is_treat, label="简单DID (ITT)")

    print()
    print("4) 安慰剂检验：把 pre 期拆成两半，假装做一次 DID（真实效应应为0）")
    print("-" * 60)
    months = sorted(panel[panel.period == "pre"]["month"].unique())
    if len(months) < 2:
        print("  pre 期月份太少，无法拆分，跳过安慰剂检验。")
    else:
        mid = len(months) // 2
        fake_pre_months = months[:mid] if mid > 0 else months[:1]
        fake_post_months = months[mid:] if mid > 0 else months[1:]
        sub = panel[panel.period == "pre"]
        g1 = sub[sub.month.isin(fake_pre_months)].groupby("merchant_id")["transactions"].mean()
        g2 = sub[sub.month.isin(fake_post_months)].groupby("merchant_id")["transactions"].mean()
        fake_diff = g2 - g1
        run_did(fake_diff, is_treat, label="安慰剂DID (应接近0且不显著)")

    print()
    print("=" * 60)
    print("解读提示:")
    print("- ITT 把实验组里没采纳方案的商户也算进去，数值会比真实效应小。")
    print("- CACE 试图还原真正采纳方案的商户的效应，应更接近仿真设定的真值。")
    print("- 回归调整(含pre期协变量)通常比简单DID的置信区间更窄，估计更精确。")
    print("- 安慰剂检验的效应若明显偏离0或p值很小，说明分流或方法可能有问题，")
    print("  需要回头检查随机分流的平衡性（上一步 simulate 脚本打印的 SMD）。")


if __name__ == "__main__":
    main()
