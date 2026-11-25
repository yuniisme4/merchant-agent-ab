"""端到端流水线：双通道识别 -> 诊断 -> 方案 -> 价值评估。

为什么要"双通道"识别：
    通道 A「水平低」  ：最近交易量在同侪里排后 25%。抓的是"本来就小"的商家，
                        其中大多没有经营问题，修好了也没多少增量。
    通道 B「显著下滑」：最近交易量相比自己早期显著下降（比例跌破阈值 + 显著性检验）。
                        抓的是"原本不小、后来掉下来"的商家，可挽回的交易量主要在这里。
    两个通道取并集作为候选，再统一做诊断和方案。

评估口径（仿真里有真值，所以能算）:
    真实可挽回量 = 问题商家"健康时的月均交易量 - 当前真实月均交易量"
    - 各通道抓到了多少真实可挽回量？
    - 运营人力有限（只能服务 K 家）时，按优先级取前 K 家，能覆盖多少真实可挽回量？
      对比"只用通道 A"与"双通道"。
    - 抓到的可挽回量里，有多少拿到的是"对症"的方案？

用法:
    python agent/pipeline.py --panel data/simulated_funnel/panel.csv \
        --merchants data/simulated_funnel/merchants.csv --budget 300
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for sub in ["detect", "diagnose", "agent"]:
    sys.path.insert(0, str(ROOT / sub))
import diagnose as dg  # noqa: E402
import recommend as rc  # noqa: E402
from low_sale import build_features, classify  # noqa: E402


def decline_channel(panel, n_base, n_recent, th_decline, z_crit, min_gain):
    """通道 B：相比自己早期，交易量显著下滑。"""
    months = sorted(panel["month"].unique())
    base_m, rec_m = months[:n_base], months[-n_recent:]
    s_b = panel[panel["month"].isin(base_m)].groupby("merchant_id")["transactions"].sum()
    s_r = panel[panel["month"].isin(rec_m)].groupby("merchant_id")["transactions"].sum()
    z = dg.pois_z(s_b, n_base, s_r, n_recent)
    rate_b, rate_r = s_b / n_base, s_r / n_recent
    ratio = rate_r / rate_b.replace(0, np.nan)
    gain = rate_b - rate_r
    flag = (ratio < th_decline) & (z <= -z_crit) & (gain >= min_gain)
    return pd.DataFrame({"decline_ratio": ratio.round(3), "decline_z": z.round(2),
                         "is_declining": flag.fillna(False)})


def topk_share(df, score, k, gain):
    """按分数取前 k 家，返回其覆盖的真实可挽回量占总量的比例。"""
    top = score.sort_values(ascending=False).head(k).index
    return float(gain.reindex(top).fillna(0).sum() / gain.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/simulated_funnel/panel.csv")
    ap.add_argument("--merchants", default=None, help="真值文件（提供则输出评估）")
    ap.add_argument("--base-months", type=int, default=2)
    ap.add_argument("--recent-months", type=int, default=3)
    ap.add_argument("--quantile", type=float, default=0.25, help="通道A：低位分位线")
    ap.add_argument("--th-decline", type=float, default=0.7, help="通道B：近期/早期 低于该比值")
    ap.add_argument("--min-gain", type=float, default=1.0, help="通道B：至少少了多少笔/月才算")
    ap.add_argument("--z-crit", type=float, default=2.5)
    ap.add_argument("--budget", type=int, default=300, help="运营人力：最多能服务几家")
    ap.add_argument("--out", default="data/pipeline")
    args = ap.parse_args()

    panel = pd.read_csv(args.panel)
    nb, nr = args.base_months, args.recent_months

    # --- 1. 识别：两个通道 ---
    low = classify(build_features(panel, nr), args.quantile, 0.1)
    dec = decline_channel(panel, nb, nr, args.th_decline, args.z_crit, args.min_gain)
    ch = pd.DataFrame({"A": low["is_low_sale"], "B": dec["is_declining"]}).fillna(False)
    ch["channel"] = np.select([ch.A & ch.B, ch.A, ch.B], ["两者", "仅水平低", "仅显著下滑"], "未识别")

    # --- 2. 诊断（对全体商家算一遍，候选里取用）---
    th = {"exp": 0.7, "act": 0.7, "cvr": 0.75, "ver": 0.85, "depth": 0.8, "rating": 0.3}
    diag = dg.diagnose(panel, nb, nr, th, 100, 50, 20, args.z_crit)

    # --- 3. 方案（只给候选商家）---
    ctx = rc.merchant_context(panel, nb, nr)
    df = diag.join(ctx[["recoverable"]]).join(ch[["channel"]])
    df["confidence"] = df.apply(rc.confidence, axis=1)
    df["priority"] = rc.assign_priority(df)
    df["score"] = df["recoverable"] * df["confidence"].map({"高": 1.0, "中": 0.7, "低": 0.2})
    cand = df[df["channel"] != "未识别"]
    plans = [rc.build_plan(mid, row, ctx.loc[mid]) for mid, row in cand.iterrows()]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    flat = pd.DataFrame([{
        "merchant_id": p["merchant_id"], "channel": cand.loc[p["merchant_id"], "channel"],
        "diagnosis_cn": p["diagnosis_cn"], "confidence": p["confidence"], "priority": p["priority"],
        "score": round(float(cand.loc[p["merchant_id"], "score"]), 2),
        "recoverable_trans_per_month": p["recoverable_trans_per_month"],
        "strategy_ids": ",".join(a["id"] for a in p["actions"]), "plan_text": rc.plan_text(p)}
        for p in plans]).sort_values("score", ascending=False)
    flat.to_csv(out / "plans.csv", index=False, encoding="utf-8-sig")
    with open(out / "plans_top20.md", "w", encoding="utf-8") as f:
        f.write("# 优先级最高的 20 份运营方案（仿真数据，双通道识别）\n\n")
        for mid in flat[flat["diagnosis_cn"] != "数据不足"].head(20)["merchant_id"]:
            p = next(x for x in plans if x["merchant_id"] == mid)
            f.write(f"渠道：{cand.loc[mid, 'channel']}\n```\n{rc.plan_text(p)}\n```\n\n")

    print(f"共 {len(panel['merchant_id'].unique())} 个商家，候选 {len(cand)} 个")
    print("\n[候选商家来源]")
    print(cand["channel"].value_counts().to_string())
    print("\n[候选商家的诊断结果]")
    print(cand["diagnosis_cn"].value_counts().to_string())

    if args.merchants:
        truth = pd.read_csv(args.merchants).set_index("merchant_id")
        t = truth["problem_type"].reindex(df.index)
        gain = (truth["initial_rate"] - truth["baseline_rate"]).clip(lower=0).where(t != "none", 0)
        gain = gain.reindex(df.index)
        has = t != "none"

        print("\n" + "=" * 64)
        print("评估（仿真真值）：各通道抓到了多少真实可挽回量？")
        rows = []
        for name in ["仅水平低", "仅显著下滑", "两者", "未识别"]:
            m = df["channel"] == name
            rows.append({"来源": name, "商家数": int(m.sum()), "其中真有问题": int((m & has).sum()),
                         "真实可挽回月交易量": round(float(gain[m].sum()), 1),
                         "占全部可挽回量": round(float(gain[m].sum() / gain.sum()), 3)})
        print(pd.DataFrame(rows).to_string(index=False))
        cov = float(gain[df["channel"] != "未识别"].sum() / gain.sum())
        a_only = float(gain[ch["A"].reindex(df.index)].sum() / gain.sum())
        print(f"\n只用通道A（水平低）覆盖 {a_only:.1%} 的可挽回量；双通道覆盖 {cov:.1%}")

        print(f"\n运营人力有限（只能服务 {args.budget} 家）时，按优先级取前 {args.budget} 家，覆盖的可挽回量：")
        in_a = ch["A"].reindex(df.index)
        share_a = topk_share(df, df["score"].where(in_a, -1), args.budget, gain)
        share_all = topk_share(df, df["score"].where(df["channel"] != "未识别", -1), args.budget, gain)
        print(f"  只用通道A：{share_a:.1%}    双通道：{share_all:.1%}")

        picked = df[df["channel"] != "未识别"]
        pred = picked["diagnosis"].replace({"insufficient": "none"})
        tp = t.reindex(picked.index)
        ok = (pred == tp) & (tp != "none")
        g = gain.reindex(picked.index)
        print(f"\n抓到的可挽回量里，拿到\"对症方案\"的占比：{float(g[ok].sum() / max(g.sum(), 1e-9)):.1%}")
        print(f"  （其余是漏诊/数据不足 {float(g[(~ok) & (pred == 'none')].sum() / max(g.sum(), 1e-9)):.1%}，"
              f"错诊 {float(g[(~ok) & (pred != 'none')].sum() / max(g.sum(), 1e-9)):.1%}）")
        pd.DataFrame(rows).to_csv(out / "channel_eval.csv", index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到 {out}/plans.csv, plans_top20.md")


if __name__ == "__main__":
    main()
