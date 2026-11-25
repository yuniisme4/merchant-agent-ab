"""个性化运营方案生成（Agent 的第 3 步：方案）。

输入：诊断结果（diagnose.py 的输出）+ 商家面板数据
输出：每个低动销商家一份方案：问题是什么、依据、优先级、该做哪几件事、暂时别做什么、看什么指标。

设计原则（为什么这样做）:
1. 策略库是人工维护的"白名单"。方案里的每一条动作都来自库，带编号(如 CPN-01)，
   不会让模型凭空发明动作。好处：可审核、可统计"哪类策略被采纳、效果如何"，
   后面 A/B 实验里"实验组采纳了哪条策略"就靠这个编号。
2. 个性化来自数据：动作文字里嵌入该商家自己的数字（当前券力度 vs 同侪中位数、
   评分下降多少、活跃天数等），而不是千篇一律的模板。
3. 带"暂不建议"：有些动作在特定问题下会适得其反（比如评分很差时加大推广，
   只会放大差评），这类约束写进方案，是区别于"只会推荐活动"的关键。
4. 带优先级：按"可挽回的月交易量 x 置信度"排序，运营人力有限时先服务最值得的商家。
5. 证据不足时不编造：诊断为"数据不足"就建议先补数据，诊断为"无明显问题"就不建议投入大量资源。
6. 本脚本不调用任何大模型，断网也能跑。大模型只适合做最后一步"把结构化方案改写成口语化建议"，
   所以脚本可以导出提示词文件（--export-prompts），提示词里明确要求"不得新增事实、
   不得改动数字和策略编号"，让模型只润色、不做决策。

用法:
    python agent/recommend.py --panel data/simulated_funnel/panel.csv \
        --diagnosis data/diagnose/diagnosis.csv --merchants data/simulated_funnel/merchants.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "diagnose"))
sys.path.insert(0, str(ROOT / "detect"))
from diagnose import window_stats, LABELS  # noqa: E402

# ---------------------------------------------------------------------------
# 策略库：可以直接改这里的文字、增删策略。action 里的 {xxx} 会被替换成该商家的真实数据。
# ---------------------------------------------------------------------------
STRATEGIES = {
    "TRF-01": dict(name="完善门店基础信息", owner="商家", difficulty="低", metric="曝光量", weeks=2,
                   action="补全门店头图、品类标签、营业时间和商圈标签（近期月均曝光约{exposure_now:.0f}次，早期约{exposure_early:.0f}次）"),
    "TRF-02": dict(name="参加平台流量活动并小额试投", owner="商家+平台运营", difficulty="中", metric="曝光量、到店交易", weeks=4,
                   action="报名平台近期流量扶持活动，设置每日推广预算上限，先小额试跑2周再决定是否加码"),
    "TRF-03": dict(name="上线引流款套餐", owner="商家", difficulty="中", metric="曝光点击率、到店交易", weeks=4,
                   action="上线1个低价引流款套餐，用来带动搜索点击，再引导到高毛利套餐"),
    "CVR-01": dict(name="优化商品详情页", owner="商家", difficulty="低", metric="点击→下单转化率", weeks=2,
                   action="更换主图、补充套餐内容和使用规则说明，让顾客一眼看懂买什么（近期点击→下单转化率明显低于自己早期水平）"),
    "CVR-02": dict(name="设置组合套餐", owner="商家", difficulty="中", metric="点击→下单转化率、客单价", weeks=4,
                   action="把热销单品组合成2人餐/家庭餐等套餐，降低顾客的选择成本"),
    "CPN-01": dict(name="券力度调回同侪水平", owner="商家", difficulty="低", metric="点击→下单转化率、核销量", weeks=4,
                   action="当前券力度约{depth_now:.0f}%，同侪中位数约{peer_depth:.0f}%（早期自己约{depth_early:.0f}%）；"
                          "建议先调到{target_depth:.0f}%左右小范围测试，观察转化变化后再调整"),
    "CPN-02": dict(name="设置满减门槛与限时券", owner="商家", difficulty="中", metric="核销量、客单价", weeks=4,
                   action="用满减门槛拉高客单价，再配合限时券制造紧迫感，避免单纯降价"),
    "SVC-01": dict(name="排查差评与退款原因", owner="商家", difficulty="低", metric="评分、退款率", weeks=2,
                   action="评分较早期下降{rating_drop:.2f}分（现{rating_now:.2f}），近期退款率约{refund_now:.1f}%；"
                          "先逐条梳理近期差评和退款原因，归类（出品/等待/服务/与描述不符）"),
    "SVC-02": dict(name="整改履约流程", owner="商家", difficulty="中", metric="下单→核销比例、评分", weeks=4,
                   action="针对排查出的主要原因整改（备货、出餐/服务时长、核销流程培训），并设定可检查的目标"),
    "SVC-03": dict(name="主动回复差评并补救", owner="商家", difficulty="低", metric="评分", weeks=2,
                   action="对近期差评逐条回复，并联系顾客提供补救方案，挽回口碑"),
    "ACT-01": dict(name="设置每日经营提醒与基础任务", owner="平台运营", difficulty="低", metric="商家活跃天数", weeks=2,
                   action="近期月均活跃约{act_now:.0f}天（早期约{act_early:.0f}天）；推送每日经营提醒，并设置简单任务（登录、回复评价、更新菜单）"),
    "ACT-02": dict(name="客户经理回访唤醒", owner="平台运营", difficulty="中", metric="商家活跃天数、曝光量", weeks=2,
                   action="由客户经理电话/上门回访，了解商家是否遇到经营困难，并协助恢复日常运营"),
    "GEN-01": dict(name="低成本试探：新客首单小额券", owner="商家", difficulty="低", metric="到店交易", weeks=4,
                   action="设置新客首单小额券，试运行4周，观察到店交易是否有自然增长"),
    "GEN-02": dict(name="补充数据后再诊断", owner="平台运营", difficulty="低", metric="数据完整度", weeks=4,
                   action="数据量太少，当前无法可靠判断问题所在；建议积累2-4周数据后重新诊断，期间先完善基础信息"),
}

# 每类诊断 -> 推荐策略（按优先顺序）与"暂不建议"提醒
PLAYBOOK = {
    "traffic":      dict(strategies=["TRF-01", "TRF-02", "TRF-03"],
                         avoid="暂不建议大幅降价或加大券力度：转化环节目前并无异常，问题在流量入口"),
    "conversion":   dict(strategies=["CVR-01", "CVR-02"],
                         avoid="暂不建议加大推广预算：问题出在转化而不是流量，买来的流量也会流失"),
    "coupon":       dict(strategies=["CPN-01", "CPN-02", "CVR-01"],
                         avoid="暂不建议一次性把券力度提得过高：先在同侪中位水平附近小范围试验，避免亏损"),
    "service":      dict(strategies=["SVC-01", "SVC-02", "SVC-03"],
                         avoid="评分和履约改善前，暂不建议加大推广或加大券力度：更多流量只会放大差评影响"),
    "inactive":     dict(strategies=["ACT-01", "ACT-02"],
                         avoid="商家重新活跃前，暂不建议投放付费推广：没人维护，流量也接不住"),
    "none":         dict(strategies=["GEN-01"],
                         avoid="暂不建议投入大量运营资源：未发现明显经营问题，更像是体量本身较小"),
    "insufficient": dict(strategies=["GEN-02", "TRF-01"],
                         avoid="证据不足时不要下结论，也不建议大额投入"),
}
MAX_ACTIONS = 3
PROBLEMS = ["traffic", "conversion", "coupon", "service", "inactive"]


def merchant_context(panel, n_base, n_recent):
    """每个商家的"个性化数字"：用于填充策略文字，并估算可挽回的交易量。"""
    months = sorted(panel["month"].unique())
    base_m, rec_m = months[:n_base], months[-n_recent:]
    E, R = window_stats(panel, base_m), window_stats(panel, rec_m)
    tE = panel[panel["month"].isin(base_m)].groupby("merchant_id")["transactions"].mean()
    tR = panel[panel["month"].isin(rec_m)].groupby("merchant_id")["transactions"].mean()
    ctx = pd.DataFrame({
        "trans_early": tE, "trans_now": tR,
        "exposure_early": E["exposure"], "exposure_now": R["exposure"],
        "depth_early": E["depth"] * 100, "depth_now": R["depth"] * 100,
        "rating_now": R["rating"], "rating_drop": E["rating"] - R["rating"],
        "refund_now": R["refund_rate"] * 100,
        "act_early": E["active"], "act_now": R["active"],
    })
    ctx["peer_depth"] = R["depth"].median() * 100  # 同侪中位数
    ctx["target_depth"] = np.minimum(ctx["peer_depth"], ctx["depth_now"] + 5)  # 每次最多上调5个百分点
    ctx["recoverable"] = (ctx["trans_early"] - ctx["trans_now"]).clip(lower=0)  # 相比早期少掉的月交易量
    return ctx.fillna(0)


def confidence(row):
    dg = row["diagnosis"]
    if dg in ("none", "insufficient"):
        return "低"
    return "高" if row[f"score_{dg}"] >= 2 else "中"


def assign_priority(df):
    """只在"确诊有问题"的商家里排序：可挽回交易量 x 置信度，前30%高、中间40%中、其余低。"""
    prio = pd.Series("低", index=df.index)
    prob = df["diagnosis"].isin(PROBLEMS)
    score = df["recoverable"] * df["confidence"].map({"高": 1.0, "中": 0.7, "低": 0.2})
    if prob.sum():
        rank = score[prob].rank(pct=True, ascending=False)
        prio[prob] = np.where(rank <= 0.3, "高", np.where(rank <= 0.7, "中", "低"))
    return prio


def build_plan(mid, row, ctx_row):
    dg = row["diagnosis"]
    book = PLAYBOOK[dg]
    c = {k: float(v) for k, v in ctx_row.items()}
    actions = []
    for sid in book["strategies"][:MAX_ACTIONS]:
        s = STRATEGIES[sid]
        actions.append({"id": sid, "name": s["name"], "text": s["action"].format(**c),
                        "owner": s["owner"], "difficulty": s["difficulty"],
                        "metric": s["metric"], "weeks": s["weeks"]})
    return {
        "merchant_id": mid, "diagnosis": dg, "diagnosis_cn": LABELS[dg],
        "confidence": row["confidence"], "priority": row["priority"],
        "recoverable_trans_per_month": round(c["recoverable"], 1),
        "evidence": row["evidence"], "actions": actions, "avoid": book["avoid"],
    }


def plan_text(p):
    lines = [f"【{p['merchant_id']}】诊断：{p['diagnosis_cn']}（置信度{p['confidence']}，优先级{p['priority']}）",
             f"依据：{p['evidence']}"]
    if p["recoverable_trans_per_month"] > 0 and p["diagnosis"] in PROBLEMS:
        lines.append(f"可挽回：相比早期，月均少了约{p['recoverable_trans_per_month']}笔到店交易")
    lines.append("建议动作（按顺序）：")
    for i, a in enumerate(p["actions"], 1):
        lines.append(f"  {i}. [{a['id']}] {a['name']}：{a['text']}（{a['owner']}，难度{a['difficulty']}，"
                     f"观察{a['metric']}，约{a['weeks']}周）")
    lines.append(f"暂不建议：{p['avoid']}")
    return "\n".join(lines)


def llm_prompt(p):
    """导出给大模型的提示词：只让它润色，不让它做决策。"""
    facts = {k: p[k] for k in ["diagnosis_cn", "confidence", "evidence", "actions", "avoid"]}
    return ("你是商家运营顾问。请把下面的结构化运营方案，改写成给店主看的口语化建议，不超过200字。\n"
            "硬性要求：1) 不得新增任何事实或数字；2) 不得修改已有数字和策略编号；"
            "3) 不得增加或删减建议动作；4) 必须保留\"暂不建议\"的提醒；"
            "5) 若置信度为低，要明确说明判断依据不足。\n\n"
            f"结构化方案：\n{json.dumps(facts, ensure_ascii=False, indent=2)}")


def business_eval(plans, truth, ctx):
    """用仿真真值评估：可挽回的交易量，有多少落在了"对症"的方案里？"""
    t = truth.set_index("merchant_id")["problem_type"].reindex(plans.index)
    has = t != "none"
    outcome = np.where(plans["diagnosis"] == t, "对症",
              np.where(plans["diagnosis"].isin(["none", "insufficient"]), "漏诊(无针对性方案)", "错诊(方案不对症)"))
    d = pd.DataFrame({"结果": outcome, "可挽回": ctx["recoverable"].reindex(plans.index)})[has]
    s = d.groupby("结果").agg(商家数=("可挽回", "size"), 可挽回月交易量合计=("可挽回", "sum")).round(1)
    s["占比"] = (s["可挽回月交易量合计"] / s["可挽回月交易量合计"].sum()).round(3)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/simulated_funnel/panel.csv")
    ap.add_argument("--diagnosis", default="data/diagnose/diagnosis.csv")
    ap.add_argument("--merchants", default=None, help="可选：真值文件，用于评估方案覆盖度")
    ap.add_argument("--base-months", type=int, default=2)
    ap.add_argument("--recent-months", type=int, default=3)
    ap.add_argument("--all", action="store_true", help="为所有商家生成方案（默认只给被识别为低动销的）")
    ap.add_argument("--top", type=int, default=3, help="屏幕上展示几个示例方案")
    ap.add_argument("--export-prompts", type=int, default=0, help="导出前N个商家的大模型提示词")
    ap.add_argument("--out", default="data/plans")
    args = ap.parse_args()

    panel = pd.read_csv(args.panel)
    diag = pd.read_csv(args.diagnosis).set_index("merchant_id")
    ctx = merchant_context(panel, args.base_months, args.recent_months)
    if not args.all:
        diag = diag[diag["is_low_sale"]]

    df = diag.join(ctx[["recoverable"]])
    df["confidence"] = df.apply(confidence, axis=1)
    df["priority"] = assign_priority(df)

    plans = [build_plan(mid, row, ctx.loc[mid]) for mid, row in df.iterrows()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    flat = pd.DataFrame([{
        "merchant_id": p["merchant_id"], "diagnosis": p["diagnosis"], "diagnosis_cn": p["diagnosis_cn"],
        "confidence": p["confidence"], "priority": p["priority"],
        "recoverable_trans_per_month": p["recoverable_trans_per_month"],
        "strategy_ids": ",".join(a["id"] for a in p["actions"]), "plan_text": plan_text(p)} for p in plans])
    flat.to_csv(out / "plans.csv", index=False, encoding="utf-8-sig")
    with open(out / "plans.jsonl", "w", encoding="utf-8") as f:
        for p in plans:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    order = {"高": 0, "中": 1, "低": 2}
    top = sorted((p for p in plans if p["diagnosis"] in PROBLEMS),
                 key=lambda p: (order[p["priority"]], -p["recoverable_trans_per_month"]))
    with open(out / "plans_top20.md", "w", encoding="utf-8") as f:
        f.write("# 优先级最高的 20 份运营方案（仿真数据）\n\n")
        for p in top[:20]:
            f.write("```\n" + plan_text(p) + "\n```\n\n")
    if args.export_prompts:
        with open(out / "llm_prompts.jsonl", "w", encoding="utf-8") as f:
            for p in top[:args.export_prompts]:
                f.write(json.dumps({"merchant_id": p["merchant_id"], "prompt": llm_prompt(p)},
                                   ensure_ascii=False) + "\n")

    print(f"共生成 {len(plans)} 份方案")
    print("\n[诊断类型 x 优先级]")
    print(pd.crosstab(flat["diagnosis_cn"], flat["priority"]).to_string())
    print(f"\n[优先级最高的 {args.top} 份方案示例]")
    for p in top[:args.top]:
        print("\n" + plan_text(p))
    if args.merchants:
        truth = pd.read_csv(args.merchants)
        print("\n[用仿真真值评估：低动销商家中\"真有问题\"的那部分，可挽回的交易量被怎样的方案覆盖]")
        print(business_eval(flat.set_index("merchant_id"), truth, ctx).to_string())
    print(f"\n结果已保存到 {out}/plans.csv, plans.jsonl, plans_top20.md")


if __name__ == "__main__":
    main()
