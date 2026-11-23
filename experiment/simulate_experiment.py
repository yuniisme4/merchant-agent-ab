"""仿真商家 A/B 实验数据生成器。

做三件事:
1. 生成一批虚拟商家，交易量分布尽量贴近 calibrate_tianchi.py 估计出的真实分布
   （长尾：少数商家占大头）。
2. 按"基线交易量分层"做分层随机分流（实验组/对照组），这样两组在干预前就是
   同质、平衡的——这正是 A/B 实验"随机"的意义。
3. 在实验期(post)对实验组中"采纳"了 Agent 方案的商家植入一个已知的处理效应，
   其余商家（对照组、以及实验组里没采纳的商家）保持原有基线行为不变。

植入的效应是"已知的"，所以后面不管用什么统计方法（DID、回归调整...）去估计，
都可以对照这个真值，检验方法本身靠不靠谱。这是本仓库"先验证方法、再谈真实收益"
思路的核心一步。

用法:
    python experiment/simulate.py --calibration data/calibration/calibration_params.json \
        --n-merchants 1000 --pre-months 3 --post-months 3 \
        --effect-size 0.15 --adoption-rate 0.6 --out data/simulated

输出 (out 目录):
    merchants.csv   商户级信息：分层、分组、是否采纳
    panel.csv       商户 x 月份 面板数据：交易量、核销量、活跃度分数
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---- 校准参数的默认值（取自对天池数据的估计结果，读不到校准文件时用这些兜底）----
DEFAULTS = {
    "mean_monthly_total_trans": 92.33 / 6,   # 原始统计是 6 个月的总量，这里换算成月均
    "nb_size_mom": 0.0123,                    # 负二项 size 参数，越小长尾越重
    "median_redeem_rate_merchant": 0.125,
    "day_of_week_profile": None,              # 本脚本按月仿真，暂不使用周内规律
}


def load_calibration(path):
    params = dict(DEFAULTS)
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        md = raw.get("merchant_distribution", {})
        if "mean" in md:
            params["mean_monthly_total_trans"] = md["mean"] / 6
        if md.get("nb_size_mom"):
            params["nb_size_mom"] = md["nb_size_mom"]
        if "median_redeem_rate_merchant" in raw:
            params["median_redeem_rate_merchant"] = raw["median_redeem_rate_merchant"]
    return params


def make_merchants(n, params, rng):
    """生成商户级基线：每个商户有自己的月均交易量水平(rate_i)。

    用 Gamma 分布做"商户异质性"，Gamma 的 shape 就是负二项的 size 参数：
    shape 越小，分布越扁平、长尾越重——和真实数据"前10%商户占大头"的特征一致。
    """
    r = params["nb_size_mom"]
    mean = params["mean_monthly_total_trans"]
    rate_i = rng.gamma(shape=r, scale=mean / r, size=n)
    redeem_prob_i = np.clip(
        rng.normal(params["median_redeem_rate_merchant"], 0.03, size=n), 0.01, 0.6
    )
    merchants = pd.DataFrame({
        "merchant_id": [f"M{i:05d}" for i in range(n)],
        "baseline_rate": rate_i,
        "redeem_prob": redeem_prob_i,
    })
    # 用基线交易量分层（四分位），后面在层内做随机分流——这是"同质商家筛选+分层随机"
    merchants["stratum"] = pd.qcut(merchants["baseline_rate"], 4,
                                    labels=["Q1_低", "Q2", "Q3", "Q4_高"])
    return merchants


def assign_groups(merchants, treat_ratio, adoption_rate, rng):
    """层内随机分流 + 实验组内随机确定谁真正采纳了方案。"""
    groups = pd.Series(index=merchants.index, dtype=object)
    for s, idx in merchants.groupby("stratum", observed=True).groups.items():
        idx = np.array(idx)
        rng.shuffle(idx)
        cut = int(len(idx) * treat_ratio)
        groups.loc[idx[:cut]] = "treatment"
        groups.loc[idx[cut:]] = "control"
    merchants = merchants.copy()
    merchants["group"] = groups
    is_t = merchants["group"] == "treatment"
    adopted = np.zeros(len(merchants), dtype=int)
    adopted[is_t.values] = (rng.random(is_t.sum()) < adoption_rate).astype(int)
    merchants["adopted"] = adopted
    return merchants


def check_balance(merchants):
    """A/A 式平衡检验：比较两组基线交易量的标准化均值差 SMD。
    经验上 |SMD| < 0.1 认为基本平衡；这里只是提醒，不会中断流程。
    """
    t = merchants.loc[merchants.group == "treatment", "baseline_rate"]
    c = merchants.loc[merchants.group == "control", "baseline_rate"]
    pooled_sd = np.sqrt((t.var() + c.var()) / 2)
    smd = (t.mean() - c.mean()) / pooled_sd if pooled_sd > 0 else 0.0
    print(f"[平衡检验] 实验组 vs 对照组 基线交易量 SMD = {smd:.4f} "
          f"({'OK，基本平衡' if abs(smd) < 0.1 else '偏大，建议检查分层/随机种子'})")
    return smd


def make_panel(merchants, pre_months, post_months, effect_size, coupon_effect, rng):
    """生成月度面板：每个商户每个月的交易量、核销量、活跃度分数。

    - pre 期：所有商户都按自己的基线行为随机波动（真正的"干预前"）。
    - post 期：只有"实验组 且 实际采纳"的商户，交易量和核销率按设定的效应提升；
      其余商户（含实验组里没采纳的）维持基线不变——这让我们同时能算：
        ITT（意向处理效应，把没采纳的也算进实验组） 和
        CACE / 采纳者效应（只看真正采纳的商户）。
    """
    rows = []
    n_months = pre_months + post_months
    for _, m in merchants.iterrows():
        for month in range(n_months):
            period = "pre" if month < pre_months else "post"
            lift = effect_size if (period == "post" and m.adopted == 1) else 0.0
            rate = m.baseline_rate * (1 + lift)
            trans = rng.poisson(max(rate, 0.01))
            redeem_prob = m.redeem_prob
            if period == "post" and m.adopted == 1:
                redeem_prob = np.clip(redeem_prob * (1 + coupon_effect), 0, 0.95)
            redemptions = rng.binomial(trans, redeem_prob) if trans > 0 else 0
            # 活跃度分数：没有真实数据支撑，这里简单设成"采纳月份有加分 + 噪声"，
            # 仅作为一个独立于交易量的过程指标占位，请结合实际业务重新设计。
            activity = rng.normal(1.0 + (0.3 if (period == "post" and m.adopted == 1) else 0), 0.2)
            rows.append([m.merchant_id, m.stratum, m.group, m.adopted,
                        period, month, trans, redemptions, max(activity, 0)])
    return pd.DataFrame(rows, columns=[
        "merchant_id", "stratum", "group", "adopted", "period", "month",
        "transactions", "redemptions", "activity_score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default=None, help="calibrate_tianchi.py 输出的 json 路径（可选）")
    ap.add_argument("--n-merchants", type=int, default=1000)
    ap.add_argument("--treat-ratio", type=float, default=0.5, help="实验组占比")
    ap.add_argument("--adoption-rate", type=float, default=0.6, help="实验组里真正采纳方案的比例")
    ap.add_argument("--pre-months", type=int, default=3)
    ap.add_argument("--post-months", type=int, default=3)
    ap.add_argument("--effect-size", type=float, default=0.15, help="采纳商家交易量相对提升幅度，0.15表示提升15%%")
    ap.add_argument("--coupon-effect", type=float, default=0.10, help="采纳商家核销率相对提升幅度")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/simulated")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    params = load_calibration(args.calibration)
    print("[使用的基线参数]", json.dumps(params, ensure_ascii=False, indent=2))

    merchants = make_merchants(args.n_merchants, params, rng)
    merchants = assign_groups(merchants, args.treat_ratio, args.adoption_rate, rng)
    check_balance(merchants)

    panel = make_panel(merchants, args.pre_months, args.post_months,
                       args.effect_size, args.coupon_effect, rng)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merchants.to_csv(out / "merchants.csv", index=False)
    panel.to_csv(out / "panel.csv", index=False)

    print(f"\n已生成 {len(merchants)} 个虚拟商家，{len(panel)} 行月度面板数据。")
    print(f"真实植入的处理效应（仅采纳者，post期）：交易量 +{args.effect_size*100:.0f}%，"
          f"核销率 +{args.coupon_effect*100:.0f}%")
    print(f"文件已保存到: {out}/merchants.csv, {out}/panel.csv")


if __name__ == "__main__":
    main()
