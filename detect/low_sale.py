"""低动销门店识别（Agent 的第 1 步：识别）。

这一步不需要大模型，用透明的规则 + 统计量就能做，好处是可解释、可复核:
    1. 近期表现: 最近 window 个月的月均交易量
    2. 同侪对比: 在所有商家里排在什么位置（低于 quantile 分位线算"低位"）
    3. 趋势:     最近 window 个月交易量是在下滑还是上升（用线性斜率）
    4. 严重程度: 持续零交易=重度；低位且下滑=中度；仅低位=轻度；其余正常

重要：识别只使用 pre 期（"决策时已经知道的历史数据"），不碰实验期数据，
也不使用仿真里的 baseline_rate / stratum 这些"上帝视角"字段，否则就是数据泄漏。

每个被识别的商家会附带一句中文"识别依据"，后面诊断模块和大模型会直接用它。

仿真里我们知道每个商家的真实基线水平，所以可以评估识别得准不准
（精确率/召回率），并看看"观察窗口越长是否越准"。真实业务里没有这个真值，
这个评估是仿真的好处。

用法:
    python detect/low_sale.py --panel data/simulated_all/panel.csv \
        --merchants data/simulated_all/merchants.csv --window 3 --quantile 0.25
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_features(panel, window):
    """汇总每个商家在最近 window 个月（仅 pre 期）的表现特征。"""
    pre = panel[panel["period"] == "pre"] if "period" in panel.columns else panel
    months = sorted(pre["month"].unique())
    use = months[-window:]
    sub = pre[pre["month"].isin(use)]
    wide = sub.pivot_table(index="merchant_id", columns="month",
                           values="transactions", aggfunc="sum", fill_value=0)[use]

    feat = pd.DataFrame(index=wide.index)
    feat["recent_mean"] = wide.mean(axis=1)
    feat["zero_months"] = (wide == 0).sum(axis=1)
    feat["n_months"] = len(use)
    if len(use) >= 3:
        # 线性斜率：每月交易量平均变化多少笔（用中心化的月份序号做最小二乘）
        x = np.arange(len(use)) - (len(use) - 1) / 2
        feat["slope"] = (wide.values * x).sum(axis=1) / (x ** 2).sum()
    else:
        feat["slope"] = np.nan  # 月份太少，趋势不可靠，不判断
    feat["rel_slope"] = feat["slope"] / (feat["recent_mean"] + 1)  # +1 避免除以0
    return feat


def classify(feat, quantile, decline_thr):
    """按规则给出严重程度，并生成中文识别依据。"""
    feat = feat.copy()
    thr = feat["recent_mean"].quantile(quantile)
    n = len(feat)
    feat["better_than"] = (feat["recent_mean"].rank(method="min") - 1) / n  # 超过了多少比例的同侪

    low = feat["recent_mean"] <= thr
    dead = feat["zero_months"] == feat["n_months"]
    declining = (feat["rel_slope"] < -decline_thr).fillna(False)

    feat["is_low_sale"] = low
    feat["level"] = np.select(
        [dead, low & declining, low],
        ["重度:持续零交易", "中度:低位且下滑", "轻度:低位"],
        default="正常",
    )

    def reason(r):
        if r["level"] == "正常":
            return ""
        s = (f"近{int(r['n_months'])}个月月均交易{r['recent_mean']:.1f}笔，"
             f"仅超过{r['better_than']*100:.0f}%的同侪")
        if r["zero_months"] > 0:
            s += f"，其中{int(r['zero_months'])}个月零交易"
        if r["level"].startswith("中度"):
            s += f"，且呈下滑趋势（月均变化{r['slope']:+.2f}笔）"
        return s

    feat["reason"] = feat.apply(reason, axis=1)
    return feat


def evaluate(feat, merchants, quantile):
    """用仿真真值评估：真实基线处于后 quantile 的商家，才算"真低动销"。"""
    m = merchants.set_index("merchant_id").reindex(feat.index)
    truth = m["baseline_rate"] <= m["baseline_rate"].quantile(quantile)
    pred = feat["is_low_sale"]
    tp = int((truth & pred).sum())
    precision = tp / max(int(pred.sum()), 1)
    recall = tp / max(int(truth.sum()), 1)
    f1 = 2 * precision * recall / (precision + recall) if tp > 0 else 0.0
    return {"识别出的商家数": int(pred.sum()), "真低动销商家数": int(truth.sum()),
            "精确率": round(precision, 3), "召回率": round(recall, 3), "F1": round(f1, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/simulated_all/panel.csv")
    ap.add_argument("--merchants", default=None, help="可选，提供后会用仿真真值评估识别效果")
    ap.add_argument("--window", type=int, default=3, help="观察窗口（最近几个月）")
    ap.add_argument("--quantile", type=float, default=0.25, help="低于该分位线算低位")
    ap.add_argument("--decline-thr", type=float, default=0.1, help="相对斜率低于 -该值 算下滑")
    ap.add_argument("--out", default="data/detect")
    args = ap.parse_args()

    panel = pd.read_csv(args.panel)
    feat = classify(build_features(panel, args.window), args.quantile, args.decline_thr)

    print(f"共 {len(feat)} 个商家，窗口={args.window} 个月，低位分位线={args.quantile}")
    print("\n[严重程度分布]")
    print(feat["level"].value_counts().to_string())
    print("\n[识别依据示例（每类各取 1 个）]")
    for level in ["重度:持续零交易", "中度:低位且下滑", "轻度:低位"]:
        ex = feat[feat["level"] == level].head(1)
        if len(ex):
            print(f"- {ex.index[0]}（{level}）: {ex['reason'].iloc[0]}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    feat.reset_index().to_csv(out / "low_sale_flags.csv", index=False, encoding="utf-8-sig")

    if args.merchants:
        merchants = pd.read_csv(args.merchants)
        n_pre = panel[panel["period"] == "pre"]["month"].nunique()
        rows = []
        for w in range(1, n_pre + 1):
            f = classify(build_features(panel, w), args.quantile, args.decline_thr)
            rows.append({"窗口(月)": w, **evaluate(f, merchants, args.quantile)})
        ev = pd.DataFrame(rows)
        ev.to_csv(out / "detect_eval.csv", index=False, encoding="utf-8-sig")
        print("\n[用仿真真值评估：观察窗口越长，识别越准吗？]")
        print(ev.to_string(index=False))
        print("说明：真低动销 = 仿真里基线交易量处于后 "
              f"{args.quantile*100:.0f}% 的商家（真实业务里没有这个真值）。")

    print(f"\n结果已保存到 {out}/low_sale_flags.csv")


if __name__ == "__main__":
    main()
