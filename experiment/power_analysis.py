"""功效分析：回答"要多少商家、跑几个月，这个实验才测得出效果？"

做法（蒙特卡洛仿真）:
    对每一组 (商家数, 实验期月数) 的组合，重复仿真很多次，每次都完整走一遍
    "生成商家 -> 筛选 -> 分层随机分流 -> 植入效应 -> 回归调整DID估计"，
    统计 p<0.05 的比例：
      - 有真实效应时，这个比例叫"功效/检出率"(power)，一般希望 >= 0.8；
      - 零效应时，这个比例叫"假阳性率"，应该接近 0.05，明显偏高说明方法有问题。

指标: 交易量(transactions) 和 核销率(redemptions/transactions)，均为 ITT 估计
（回归调整 DID，含 pre 期协变量，与 analyze_did.py 中一致）。

用法:
    python experiment/power_analysis.py --calibration data/calibration/calibration_params.json
可选: --reps 200（每组重复次数，越大越准但越慢）、--effect-size、--coupon-effect 等

注意: 表里的"商家数"是筛选低基线商户之前的数量，实际入组人数约为其 75%。
"""
import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

# 复用前面两个脚本里的函数（三个文件要放在同一个 experiment 文件夹里）
from simulate_experiment import (load_calibration, make_merchants,
                                 filter_homogeneous_merchants, add_strata,
                                 assign_groups)
from analyze_did import ols


def itt_pvalue(diff, is_treat, pre):
    """回归调整 DID 的 ITT 效应 p 值。样本太少或矩阵奇异时返回 nan。"""
    mask = np.isfinite(diff) & np.isfinite(pre)
    if mask.sum() < 20:
        return np.nan
    X = np.column_stack([is_treat[mask], pre[mask]])
    try:
        _, _, _, _, p = ols(diff[mask], X)
    except np.linalg.LinAlgError:
        return np.nan
    return p[1]


def simulate_once(params, n, pre_m, post_m, effect, coupon_effect,
                  adoption_rate, min_q, rng):
    """向量化的一次完整仿真，返回 (交易量p值, 核销率p值)。"""
    with contextlib.redirect_stdout(io.StringIO()):  # 屏蔽被复用函数里的 print
        m = make_merchants(n, params, rng)
        m = filter_homogeneous_merchants(m, min_q)
        m = add_strata(m)
        m = assign_groups(m, 0.5, adoption_rate, rng)

    base = m["baseline_rate"].to_numpy()
    rp = m["redeem_prob"].to_numpy()
    adopted = m["adopted"].to_numpy().astype(bool)
    is_treat = (m["group"] == "treatment").to_numpy().astype(float)
    T = pre_m + post_m

    lift = np.zeros((len(m), T))
    lift[adopted, pre_m:] = effect
    trans = rng.poisson(np.maximum(base[:, None] * (1 + lift), 0.01))

    prob = np.tile(rp[:, None], (1, T))
    prob[adopted, pre_m:] = np.clip(rp[adopted, None] * (1 + coupon_effect), 0, 0.95)
    red = rng.binomial(trans, prob)

    # 交易量：每个商户 post 均值 - pre 均值
    pre_tr, post_tr = trans[:, :pre_m].mean(1), trans[:, pre_m:].mean(1)
    p_tr = itt_pvalue(post_tr - pre_tr, is_treat, pre_tr)

    # 核销率：分母为 0 的商户记为缺失（与 analyze_did.py 一致）
    with np.errstate(divide="ignore", invalid="ignore"):
        pre_rr = red[:, :pre_m].sum(1) / trans[:, :pre_m].sum(1)
        post_rr = red[:, pre_m:].sum(1) / trans[:, pre_m:].sum(1)
    p_rr = itt_pvalue(post_rr - pre_rr, is_treat, pre_rr)
    return p_tr, p_rr


def rate(pvals, alpha=0.05):
    pv = np.asarray(pvals, dtype=float)
    pv = pv[np.isfinite(pv)]
    return float((pv < alpha).mean()) if len(pv) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--reps", type=int, default=200, help="每组重复仿真次数")
    ap.add_argument("--n-list", type=int, nargs="+", default=[500, 1500, 4000],
                    help="商家数（筛选前）的候选值")
    ap.add_argument("--post-list", type=int, nargs="+", default=[1, 3, 6],
                    help="实验期月数的候选值")
    ap.add_argument("--pre-months", type=int, default=3)
    ap.add_argument("--effect-size", type=float, default=0.15)
    ap.add_argument("--coupon-effect", type=float, default=0.10)
    ap.add_argument("--adoption-rate", type=float, default=0.6)
    ap.add_argument("--min-baseline-quantile", type=float, default=0.25)
    ap.add_argument("--dispersion-floor", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data/simulated")
    args = ap.parse_args()

    params = load_calibration(args.calibration, args.dispersion_floor)
    rows = []
    total = len(args.n_list) * len(args.post_list)
    done = 0
    for n in args.n_list:
        for post_m in args.post_list:
            res = {}
            for sid, scenario in enumerate(["effect", "null"]):
                eff = args.effect_size if scenario == "effect" else 0.0
                cpe = args.coupon_effect if scenario == "effect" else 0.0
                p_tr, p_rr = [], []
                for r in range(args.reps):
                    rng = np.random.default_rng([args.seed, n, post_m, sid, r])
                    a, b = simulate_once(params, n, args.pre_months, post_m, eff, cpe,
                                         args.adoption_rate, args.min_baseline_quantile, rng)
                    p_tr.append(a)
                    p_rr.append(b)
                res[scenario] = (rate(p_tr), rate(p_rr))
            rows.append({
                "商家数(筛选前)": n, "实验期月数": post_m,
                "交易量_功效": res["effect"][0], "核销率_功效": res["effect"][1],
                "交易量_假阳性率": res["null"][0], "核销率_假阳性率": res["null"][1],
            })
            done += 1
            print(f"进度 {done}/{total}: n={n}, post={post_m} 个月 完成", flush=True)

    df = pd.DataFrame(rows).round(3)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "power_results.csv", index=False, encoding="utf-8-sig")

    se = (0.05 * 0.95 / args.reps) ** 0.5
    print("\n" + "=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)
    print("怎么看:")
    print("- 功效：有真实效应时检出的比例，一般希望 >= 0.8。")
    print(f"- 假阳性率：零效应时误报的比例，应接近 0.05"
          f"（重复 {args.reps} 次，随机波动约 ±{2*se:.3f} 属正常）。")
    print(f"结果已保存到 {out / 'power_results.csv'}")


if __name__ == "__main__":
    main()
