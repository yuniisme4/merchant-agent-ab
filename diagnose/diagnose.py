"""门店经营问题诊断（Agent 的第 2 步：诊断）。

思路：让每个商家"和自己的历史比"，看经营漏斗的哪一个环节掉下去了。
    早期窗口（默认前 2 个月，认为是问题出现之前） vs 近期窗口（默认最近 3 个月）

对比的信号（比值 = 近期 / 早期）:
    exp_ratio    月均曝光              <- 流量
    act_ratio    商家月均活跃天数        <- 商家是否还在经营
    cvr_ratio    点击->下单转化率        <- 页面/券的吸引力
    depth_ratio  券力度(折扣深度)        <- 用来区分"转化低"和"券力度弱"
    ver_ratio    下单->核销比例          <- 履约
    rating_drop  评分下降幅度            <- 服务口碑

判定规则（每个信号都折算成"强度 = 下滑幅度 / 阈值下滑幅度"，>=1 算触发，取最强者）:
    商家不活跃   活跃天数骤降（同时它会拖累曝光，所以触发时不再单独判流量问题）
    流量不足     曝光下滑，且商家仍然活跃
    转化低       转化率下滑，券力度没怎么变
    券力度弱     转化率下滑，且券力度同步变浅
    履约/评分差  评分明显下降，或下单->核销比例下滑
    无明显问题   没有任何信号触发（常见于"本来就小"的健康商家）
    数据不足     样本量太小，漏斗比率不可靠，也没有其他信号

为什么只做规则而不上机器学习：规则透明、可解释，每个结论都带证据；
而且我们有仿真真值，可以直接评估并调阈值。真实业务里没有真值，
阈值需要结合业务经验和小规模人工复核来定。

用法:
    python diagnose/diagnose.py --panel data/simulated_funnel/panel.csv \
        --merchants data/simulated_funnel/merchants.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 复用识别模块（low_sale.py 在同级的 detect 文件夹）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "detect"))
from low_sale import build_features, classify  # noqa: E402

LABELS = {"traffic": "流量不足", "conversion": "转化低", "coupon": "券力度弱",
          "service": "履约/评分差", "inactive": "商家不活跃",
          "none": "无明显问题", "insufficient": "数据不足"}


def window_stats(panel, months):
    """汇总一个时间窗口内各环节指标（用合计值算比率，样本小的商家更稳）。"""
    sub = panel[panel["month"].isin(months)]
    g = sub.groupby("merchant_id")
    s = g[["exposure", "clicks", "orders", "transactions", "refunds"]].sum()
    out = pd.DataFrame(index=s.index)
    out["exposure"] = s["exposure"] / len(months)
    out["ctr"] = s["clicks"] / s["exposure"].replace(0, np.nan)
    out["cvr"] = s["orders"] / s["clicks"].replace(0, np.nan)
    out["ver"] = s["transactions"] / s["orders"].replace(0, np.nan)
    out["refund_rate"] = s["refunds"] / s["orders"].replace(0, np.nan)
    out["rating"] = g["rating"].mean()
    out["depth"] = g["coupon_depth"].mean()
    out["active"] = g["active_days"].mean()
    out["n_exposure"], out["n_clicks"], out["n_orders"] = s["exposure"], s["clicks"], s["orders"]
    # 原始合计量：显著性检验要用
    out["sum_exposure"], out["sum_clicks"] = s["exposure"], s["clicks"]
    out["sum_orders"], out["sum_trans"] = s["orders"], s["transactions"]
    out["sum_active"] = g["active_days"].sum()
    return out


def prop_z(xE, nE, xR, nR):
    """两个比例（早期 vs 近期）的 z 检验统计量；负得越多，说明近期比例越显著地低于早期。"""
    pE, pR = xE / nE, xR / nR
    p = (xE + xR) / (nE + nR)
    se = np.sqrt(p * (1 - p) * (1 / nE + 1 / nR))
    return ((pR - pE) / se).replace([np.inf, -np.inf], np.nan).fillna(0)


def pois_z(sumE, mE, sumR, mR):
    """月均计数（如曝光）的泊松 z 检验：早期 mE 个月合计 sumE，近期 mR 个月合计 sumR。"""
    rE, rR = sumE / mE, sumR / mR
    se = np.sqrt(sumE / mE ** 2 + sumR / mR ** 2)
    return ((rR - rE) / se).replace([np.inf, -np.inf], np.nan).fillna(0)


def pct0(x):
    """格式化百分数，缺失时显示"无数据"。"""
    return f"{x * 100:.0f}%" if np.isfinite(x) else "无数据"


def strength(ratio, th):
    """比值 -> 信号强度。ratio=th 时强度为 1（刚好触发），缺失视为 0（没证据）。"""
    return ((1 - ratio) / (1 - th)).fillna(0).clip(lower=0)


def diagnose(panel, n_base, n_recent, th, min_exposure, min_clicks, min_orders, z_crit):
    months = sorted(panel["month"].unique())
    base_m, recent_m = months[:n_base], months[-n_recent:]
    E, R = window_stats(panel, base_m), window_stats(panel, recent_m)

    d = pd.DataFrame(index=E.index)
    d["exp_ratio"] = R["exposure"] / E["exposure"].replace(0, np.nan)
    d["act_ratio"] = R["active"] / E["active"].replace(0, np.nan)
    d["cvr_ratio"] = R["cvr"] / E["cvr"].replace(0, np.nan)
    d["ver_ratio"] = R["ver"] / E["ver"].replace(0, np.nan)
    d["depth_ratio"] = R["depth"] / E["depth"].replace(0, np.nan)
    d["rating_drop"] = E["rating"] - R["rating"]

    # 样本量足够才相信漏斗比率；活跃天数、评分不依赖样本量，始终可用
    ok_exp = E["n_exposure"] >= min_exposure
    ok_cvr = E["n_clicks"] >= min_clicks
    ok_ver = E["n_orders"] >= min_orders

    # 显著性检验：比值跌破阈值还不够，还要"不是随机波动"才触发。
    # 小商家的比率天然抖动很大，不检验的话会大量误报。
    nB, nR = len(base_m), len(recent_m)
    z_exp = pois_z(E["sum_exposure"], nB, R["sum_exposure"], nR)
    z_act = prop_z(E["sum_active"], 30 * nB, R["sum_active"], 30 * nR)
    z_cvr = prop_z(E["sum_orders"], E["sum_clicks"], R["sum_orders"], R["sum_clicks"])
    z_ver = prop_z(E["sum_trans"], E["sum_orders"], R["sum_trans"], R["sum_orders"])
    sig = lambda z: z <= -z_crit
    d["z_exp"], d["z_act"], d["z_cvr"], d["z_ver"] = z_exp.round(2), z_act.round(2), z_cvr.round(2), z_ver.round(2)

    s_inactive = strength(d["act_ratio"], th["act"]).where(sig(z_act), 0)
    s_traffic = strength(d["exp_ratio"], th["exp"]).where(ok_exp & sig(z_exp), 0)
    s_traffic = s_traffic.where(s_inactive < 1, 0)  # 不活跃已解释曝光下滑
    s_cvr = strength(d["cvr_ratio"], th["cvr"]).where(ok_cvr & sig(z_cvr), 0)
    shallow = d["depth_ratio"] < th["depth"]
    s_coupon = s_cvr.where(shallow, 0)
    s_conv = s_cvr.where(~shallow, 0)
    s_service = np.maximum((d["rating_drop"] / th["rating"]).fillna(0).clip(lower=0),
                           strength(d["ver_ratio"], th["ver"]).where(ok_ver & sig(z_ver), 0))

    scores = pd.DataFrame({"inactive": s_inactive, "traffic": s_traffic, "conversion": s_conv,
                           "coupon": s_coupon, "service": s_service})
    best = scores.idxmax(axis=1)
    triggered = scores.max(axis=1) >= 1
    insufficient = ~(ok_exp | ok_cvr | ok_ver)
    d["diagnosis"] = np.where(triggered, best, np.where(insufficient, "insufficient", "none"))
    for k in scores.columns:
        d[f"score_{k}"] = scores[k].round(2)
    d["evidence"] = [evidence(r, E.loc[i], R.loc[i]) for i, r in d.iterrows()]
    d["diagnosis_cn"] = d["diagnosis"].map(LABELS)
    return d


def evidence(r, e, rc):
    """生成中文证据，后面方案模块和大模型会直接引用。"""
    dg = r["diagnosis"]
    pct = lambda x: f"{(1 - x) * 100:.0f}%"
    if dg == "traffic":
        return (f"月均曝光较早期下降{pct(r['exp_ratio'])}（{e['exposure']:.0f}→{rc['exposure']:.0f}），"
                f"商家活跃天数基本正常（{e['active']:.0f}→{rc['active']:.0f}天），其他环节无明显异常")
    if dg == "inactive":
        return (f"商家月均活跃天数较早期下降{pct(r['act_ratio'])}（{e['active']:.0f}→{rc['active']:.0f}天），"
                f"曝光随之下降{pct(r['exp_ratio']) if r['exp_ratio'] < 1 else '0%'}")
    if dg in ("conversion", "coupon"):
        s = f"点击→下单转化率较早期下降{pct(r['cvr_ratio'])}（{e['cvr']*100:.1f}%→{rc['cvr']*100:.1f}%）"
        if dg == "coupon":
            return s + f"，同时券力度变浅{pct(r['depth_ratio'])}（{e['depth']*100:.0f}%→{rc['depth']*100:.0f}%）"
        return s + f"，但券力度基本不变（{e['depth']*100:.0f}%→{rc['depth']*100:.0f}%）"
    if dg == "service":
        return (f"评分下降{r['rating_drop']:.2f}分（{e['rating']:.2f}→{rc['rating']:.2f}），"
                f"下单→核销比例{pct0(e['ver'])}→{pct0(rc['ver'])}，"
                f"退款率{pct0(e['refund_rate'])}→{pct0(rc['refund_rate'])}")
    if dg == "insufficient":
        return "漏斗样本量太小，比率不可靠，且活跃度、评分无异常，暂无法判断"
    return "各环节与自身历史相比无明显异常，更像是体量本身较小，而非经营出了问题"


def evaluate(diag, truth, subset_name):
    t = truth.set_index("merchant_id").reindex(diag.index)["problem_type"]
    pred = diag["diagnosis"].replace({"insufficient": "none"})  # "数据不足"也算"没发现问题"
    acc = float((pred == t).mean())
    print(f"\n[{subset_name}] 商家数={len(diag)}  诊断准确率={acc:.3f}")
    cm = pd.crosstab(t.rename("真值"), diag["diagnosis"].rename("诊断"))
    print(cm.to_string())
    prob = t != "none"
    if prob.sum():
        print(f"  有问题的商家里：查出任意问题的比例(召回)={float((pred[prob] != 'none').mean()):.3f}，"
              f"类型判对的比例={float((pred[prob] == t[prob]).mean()):.3f}")
    if (~prob).sum():
        print(f"  没问题的商家里：被误判为有问题的比例={float((pred[~prob] != 'none').mean()):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/simulated_funnel/panel.csv")
    ap.add_argument("--merchants", default=None, help="提供真值文件则输出评估")
    ap.add_argument("--base-months", type=int, default=2)
    ap.add_argument("--recent-months", type=int, default=3)
    ap.add_argument("--window", type=int, default=3, help="低动销识别窗口")
    ap.add_argument("--quantile", type=float, default=0.25)
    ap.add_argument("--th-exp", type=float, default=0.7)
    ap.add_argument("--th-act", type=float, default=0.7)
    ap.add_argument("--th-cvr", type=float, default=0.75)
    ap.add_argument("--th-ver", type=float, default=0.85)
    ap.add_argument("--th-depth", type=float, default=0.8)
    ap.add_argument("--th-rating", type=float, default=0.3)
    ap.add_argument("--min-exposure", type=int, default=100)
    ap.add_argument("--min-clicks", type=int, default=50)
    ap.add_argument("--min-orders", type=int, default=20)
    ap.add_argument("--z-crit", type=float, default=2.5,
                    help="显著性门槛：z 统计量需低于 -该值 才算真下滑（越大越保守）")
    ap.add_argument("--out", default="data/diagnose")
    args = ap.parse_args()

    panel = pd.read_csv(args.panel)
    th = {"exp": args.th_exp, "act": args.th_act, "cvr": args.th_cvr, "ver": args.th_ver,
          "depth": args.th_depth, "rating": args.th_rating}
    diag = diagnose(panel, args.base_months, args.recent_months, th,
                    args.min_exposure, args.min_clicks, args.min_orders, args.z_crit)

    low = classify(build_features(panel, args.window), args.quantile, 0.1)
    diag["low_sale_level"] = low["level"]
    diag["is_low_sale"] = low["is_low_sale"]
    diag["low_sale_reason"] = low["reason"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    diag.reset_index().to_csv(out / "diagnosis.csv", index=False, encoding="utf-8-sig")

    flagged = diag[diag["is_low_sale"]]
    print(f"共 {len(diag)} 个商家，其中识别为低动销 {len(flagged)} 个")
    print("\n[低动销商家的诊断结果分布]")
    print(flagged["diagnosis_cn"].value_counts().to_string())
    print("\n[诊断示例（每类各取 1 个）]")
    for k in ["traffic", "inactive", "conversion", "coupon", "service", "none"]:
        ex = flagged[flagged["diagnosis"] == k].head(1)
        if len(ex):
            print(f"- {ex.index[0]}【{LABELS[k]}】{ex['evidence'].iloc[0]}")

    if args.merchants:
        truth = pd.read_csv(args.merchants)
        evaluate(flagged, truth, "评估 A：被识别为低动销的商家（真实使用场景）")
        evaluate(diag, truth, "评估 B：全部商家（单看诊断规则本身的能力）")
    print(f"\n结果已保存到 {out}/diagnosis.csv")


if __name__ == "__main__":
    main()
