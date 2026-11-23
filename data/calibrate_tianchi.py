"""从天池 O2O 线下表估计商户级基线参数，用于校准仿真器。

用法:
    python calibrate_tianchi.py --input data/raw/ccf_offline_stage1_train.csv \
                                --out data/calibration

输出 (out 目录):
    merchant_summary.csv     商户级汇总（领券、核销、普通交易、核销率等）
    monthly_matrix.csv       商户 x 月份 的交易量矩阵
    conversion_by_type.csv   按券类型（折扣/满减）的 15 天核销率
    conversion_by_rate.csv   按折扣率分箱的 15 天核销率
    conversion_by_distance.csv 按距离的 15 天核销率
    calibration_params.json  仿真器可直接读取的汇总参数
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["User_id", "Merchant_id", "Coupon_id", "Discount_rate",
        "Distance", "Date_received", "Date"]
WINDOW = 15  # 核销窗口（天）


def load(path):
    # 原始文件用字符串 "null" 表示缺失
    df = pd.read_csv(path, dtype=str, keep_default_na=False,
                     na_values=["null", ""])
    missing = set(COLS) - set(df.columns)
    if missing:
        raise ValueError(f"缺少字段: {missing}")
    for c in ["Date_received", "Date"]:
        s = df[c].str.replace(r"\.0$", "", regex=True)
        df[c] = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    df["received"] = df["Coupon_id"].notna()
    df["consumed"] = df["Date"].notna()
    df["redeemed"] = df["received"] & df["consumed"]       # 用券消费
    df["normal"] = (~df["received"]) & df["consumed"]       # 普通消费
    lag = (df["Date"] - df["Date_received"]).dt.days
    df["redeemed_15d"] = df["redeemed"] & (lag <= WINDOW)
    return df


def parse_discount(s):
    """返回 (是否满减, 折扣率, 满减门槛)。'150:20' -> (1, 0.867, 150)。"""
    if ":" in s:
        a, b = (float(v) for v in s.split(":"))
        return 1, 1 - b / a, a
    return 0, float(s), 0.0


def merchant_summary(df):
    g = df.groupby("Merchant_id")
    m = pd.DataFrame({
        "coupons_received": g["received"].sum(),
        "redeemed": g["redeemed"].sum(),
        "normal_trans": g["normal"].sum(),
    })
    m["total_trans"] = m["redeemed"] + m["normal_trans"]
    m["redeem_rate"] = m["redeemed"] / m["coupons_received"].replace(0, np.nan)
    m["redeem_share"] = m["redeemed"] / m["total_trans"].replace(0, np.nan)
    m["active_days"] = (df[df["consumed"]].groupby("Merchant_id")["Date"]
                        .nunique().reindex(m.index).fillna(0).astype(int))
    return m


def distribution_params(m):
    x = m["total_trans"]
    mean, var = x.mean(), x.var()
    top10 = x.sort_values(ascending=False).head(max(1, int(len(x) * 0.1)))
    return {
        "n_merchants": int(len(x)),
        "total_trans_quantiles": {str(q): float(x.quantile(q))
                                  for q in [.1, .25, .5, .75, .9, .99]},
        "top10pct_share": float(top10.sum() / x.sum()),
        "mean": float(mean),
        "var": float(var),
        # 负二项 size 参数（矩估计）；var<=mean 时无过度离散
        "nb_size_mom": float(mean ** 2 / (var - mean)) if var > mean else None,
        "zero_share": float((x == 0).mean()),
    }


def time_params(df):
    tx = df[df["consumed"]].copy()
    dow = tx.groupby(tx["Date"].dt.dayofweek).size()
    dow = (dow / dow.mean()).reindex(range(7)).fillna(0)  # 均值归一化
    tx["month"] = tx["Date"].dt.to_period("M")
    mm = tx.groupby(["Merchant_id", "month"]).size().unstack(fill_value=0)
    # 低动销持续性：本月处于后 25% 的商户，下月仍处于后 25% 的比例
    pers = []
    for a, b in zip(mm.columns[:-1], mm.columns[1:]):
        la = mm[a] <= mm[a].quantile(.25)
        lb = mm[b] <= mm[b].quantile(.25)
        if la.sum() > 0:
            pers.append(float(lb[la].mean()))
    return dow, mm, pers


def conversion_tables(df):
    rc = df[df["received"]].copy()
    # 剔除窗口未走完的领券记录（右删失）
    end = max(df["Date"].max(), df["Date_received"].max())
    rc = rc[rc["Date_received"] <= end - pd.Timedelta(days=WINDOW)]
    uniq = rc["Discount_rate"].dropna().unique()
    tbl = pd.DataFrame([parse_discount(s) for s in uniq], index=uniq,
                       columns=["is_manjian", "rate", "threshold"])
    rc = rc.join(tbl, on="Discount_rate")
    by_type = rc.groupby("is_manjian")["redeemed_15d"].agg(["mean", "size"])
    bins = [0, .5, .7, .8, .9, .95, 1.0]
    by_rate = (rc.groupby(pd.cut(rc["rate"], bins), observed=True)
               ["redeemed_15d"].agg(["mean", "size"]))
    by_dist = rc.groupby("Distance")["redeemed_15d"].agg(["mean", "size"])
    overall = float(rc["redeemed_15d"].mean())
    return by_type, by_rate, by_dist, overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="data/calibration")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load(args.input)
    m = merchant_summary(df)
    dow, mm, pers = time_params(df)
    by_type, by_rate, by_dist, overall = conversion_tables(df)

    m.to_csv(out / "merchant_summary.csv")
    mm.to_csv(out / "monthly_matrix.csv")
    by_type.to_csv(out / "conversion_by_type.csv")
    by_rate.to_csv(out / "conversion_by_rate.csv")
    by_dist.to_csv(out / "conversion_by_distance.csv")

    params = {
        "merchant_distribution": distribution_params(m),
        "day_of_week_profile": [float(v) for v in dow],
        "low_sale_persistence_by_month_pair": pers,
        "coupon_redeem_rate_15d_overall": overall,
        "median_redeem_rate_merchant": float(m["redeem_rate"].median()),
    }
    (out / "calibration_params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(params, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
