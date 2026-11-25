"""带"经营问题真值"的商家漏斗仿真器（为诊断模块准备）。

和 simulate_experiment.py 的区别：
    那个只有"交易量"一个指标，适合做 A/B 实验评估；
    这个给每个商家生成完整的经营漏斗和辅助指标，并且**悄悄植入一种经营问题**，
    我们自己知道答案，所以后面写诊断规则时，可以对照真值评估诊断得准不准。

每个商家每月的数据（面板）:
    exposure(曝光) -> clicks(点击) -> orders(下单) -> transactions(到店核销交易)
    refunds(退款数)、rating(评分)、coupon_depth(券力度，折扣深度)、active_days(商家当月活跃天数)

植入的经营问题（约 35% 的商家有，其余健康）:
    traffic      流量不足    曝光逐月下滑，其他环节正常
    conversion   转化低      点击->下单转化率下滑，券力度正常
    coupon       券力度弱    点击->下单转化率下滑，同时券力度变浅
    service      履约/评分差 下单->核销比例下滑，评分下降、退款上升
    inactive     商家不活跃  活跃天数骤降，曝光也被平台降权，轻度下滑
问题从某个月(onset)开始，用 3 个月逐步恶化到终值（严重程度 severity 越小越严重），
所以数据里存在**真实的下滑过程**——这样识别模块里的"下滑趋势"规则也能被检验。

还有一类重要的商家：**健康但体量小**（自然低动销，没有任何经营问题）。
好的诊断应该能分辨出"这家不是出了问题，只是本来就小"，而不是硬找问题。

用法:
    python experiment/simulate_funnel.py --calibration data/calibration/calibration_params.json \
        --n-merchants 3000 --months 8 --out data/simulated_funnel

输出:
    panel.csv      商户 x 月份 面板
    merchants.csv  真值：问题类型、起始月、严重程度、当前真实水平(baseline_rate)等
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from simulate_experiment import load_calibration  # 复用校准参数读取

PROBLEM_TYPES = ["traffic", "conversion", "coupon", "service", "inactive"]
PROBLEM_PROBS = [0.25, 0.20, 0.20, 0.20, 0.15]
RAMP_MONTHS = 3  # 问题用几个月恶化到终值


def simulate(n, months, params, p_problem, rng):
    M, T = n, months
    r = params["nb_size_mom"]
    mean = params["mean_monthly_total_trans"]
    t0 = rng.gamma(shape=r, scale=mean / r, size=M)  # 健康时的月均到店交易量

    # 各环节转化率（商家间有差异）
    ctr = rng.beta(3.2, 36.8, M)      # 曝光->点击，均值约 8%
    cvr = rng.beta(4.8, 35.2, M)      # 点击->下单，均值约 12%
    ver = rng.beta(30, 10, M)         # 下单->核销，均值约 75%
    exposure0 = t0 / (ctr * cvr * ver)  # 反推曝光量，使健康时的期望交易量等于 t0
    depth0 = rng.beta(3, 17, M)       # 券力度（折扣深度），均值约 15%
    rating0 = np.clip(rng.normal(4.5, 0.25, M), 3.5, 5.0)
    act0 = rng.beta(6, 4, M)          # 日均活跃概率，均值约 60%

    # 植入问题
    has_problem = rng.random(M) < p_problem
    ptype = np.where(has_problem, rng.choice(PROBLEM_TYPES, M, p=PROBLEM_PROBS), "none")
    onset = rng.integers(2, max(T - 2, 3), M)       # 问题起始月（从第2月起，保证有"正常期"可对比）
    severity = rng.uniform(0.25, 0.6, M)            # 终值倍数，越小越严重
    t_idx = np.arange(T)[None, :]
    ramp = np.clip((t_idx - onset[:, None] + 1) / RAMP_MONTHS, 0, 1)
    factor = np.where(has_problem[:, None], 1 - (1 - severity[:, None]) * ramp, 1.0)  # (M,T)

    def when(kind):  # 该商家在该类问题下才生效的因子，否则为 1
        return np.where((ptype == kind)[:, None], factor, 1.0)

    f_exp = when("traffic") * (0.5 + 0.5 * when("inactive"))  # 不活跃会被轻度降权
    f_cvr = when("conversion") * when("coupon")
    f_ver = when("service")
    f_depth = when("coupon")
    f_act = when("inactive")
    svc_drop = 1 - when("service")  # 0 -> 无问题，越大越严重

    # 逐月生成观测数据
    exposure = rng.poisson(exposure0[:, None] * f_exp)
    clicks = rng.binomial(exposure, np.clip(ctr[:, None], 0, 1))
    orders = rng.binomial(clicks, np.clip(cvr[:, None] * f_cvr, 0, 1))
    trans = rng.binomial(orders, np.clip(ver[:, None] * f_ver, 0, 1))
    refunds = rng.binomial(orders, np.clip(0.03 + 0.12 * svc_drop, 0, 1))
    rating = np.clip(rating0[:, None] + rng.normal(0, 0.1, (M, T)) - 1.2 * svc_drop, 1, 5)
    depth = np.clip(depth0[:, None] * f_depth + rng.normal(0, 0.01, (M, T)), 0, 0.6)
    active_days = rng.binomial(30, np.clip(act0[:, None] * f_act, 0, 1))

    ids = np.array([f"M{i:05d}" for i in range(M)])
    panel = pd.DataFrame({
        "merchant_id": np.repeat(ids, T), "month": np.tile(np.arange(T), M),
        "period": "pre",  # 与 low_sale.py 兼容：全部视为"已知历史"
        "exposure": exposure.ravel(), "clicks": clicks.ravel(), "orders": orders.ravel(),
        "transactions": trans.ravel(), "refunds": refunds.ravel(),
        "rating": rating.ravel().round(3), "coupon_depth": depth.ravel().round(4),
        "active_days": active_days.ravel(),
    })
    # 当前(最后一个月)的真实期望交易量 = 评估"真低动销"用的真值
    cur_rate = (exposure0 * f_exp[:, -1] * ctr * cvr * f_cvr[:, -1] * ver * f_ver[:, -1])
    truth = pd.DataFrame({
        "merchant_id": ids, "problem_type": ptype,
        "onset_month": np.where(has_problem, onset, -1),
        "severity": np.where(has_problem, severity.round(3), np.nan),
        "initial_rate": t0, "baseline_rate": cur_rate,  # baseline_rate=当前真实水平，供 low_sale.py 评估
    })
    return panel, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--n-merchants", type=int, default=3000)
    ap.add_argument("--months", type=int, default=8)
    ap.add_argument("--p-problem", type=float, default=0.35, help="有经营问题的商家占比")
    ap.add_argument("--dispersion-floor", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/simulated_funnel")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    params = load_calibration(args.calibration, args.dispersion_floor)
    panel, truth = simulate(args.n_merchants, args.months, params, args.p_problem, rng)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out / "panel.csv", index=False)
    truth.to_csv(out / "merchants.csv", index=False)

    print(f"\n已生成 {args.n_merchants} 个商家 x {args.months} 个月 = {len(panel)} 行面板数据")
    print("\n[植入的问题类型分布（真值）]")
    print(truth["problem_type"].value_counts().to_string())
    print(f"\n文件已保存到: {out}/panel.csv, {out}/merchants.csv")


if __name__ == "__main__":
    main()
