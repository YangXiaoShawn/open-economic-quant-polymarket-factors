"""生成阶段性报告插图 → report_figs/*.png"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).parent
LOGS = ROOT / "logs"
OUT = ROOT / "report_figs"
OUT.mkdir(exist_ok=True)

DARK = "#0d1117"; FG = "#e6edf3"; MUT = "#8b949e"; GRID = "#30363d"

def style(ax):
    ax.set_facecolor("#161b22")
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUT, labelsize=9)
    ax.xaxis.label.set_color(MUT); ax.yaxis.label.set_color(MUT)
    ax.title.set_color(FG)
    ax.grid(color=GRID, alpha=0.4, linewidth=0.6)

def save(fig, name):
    fig.patch.set_facecolor(DARK)
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print("saved", name)

# ── 图1：方法论漏斗 — PM-MOM IC 随假象修正逐级收敛 ───────────────────
stages = [
    "① clip截断\n+pct收益", "② 去截断+Δp\n+可交易过滤", "③ +skip-day\n(端点噪声)",
    "④ +结算收益\n入样", "⑤ +入场可执行\n(最终)",
]
raw_ic = [-0.440, -0.340, -0.105, -0.096, -0.054]
pn_ic  = [np.nan, np.nan, -0.076, -0.071, -0.032]
x = np.arange(len(stages))
fig, ax = plt.subplots(figsize=(8.2, 3.8))
ax.bar(x - 0.18, raw_ic, 0.36, color="#f85149", label="原始 Δp IC")
ax.bar(x + 0.18, pn_ic, 0.36, color="#d29922", label="价格中性 IC")
for i, v in enumerate(raw_ic):
    ax.text(i - 0.18, v - 0.012, f"{v:.3f}", ha="center", color=FG, fontsize=8)
for i, v in enumerate(pn_ic):
    if np.isfinite(v):
        ax.text(i + 0.18, v - 0.012, f"{v:.3f}", ha="center", color=FG, fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=8.5)
ax.set_title("PM-MOM 反转 IC：每修正一个机制性假象，幅度收敛一级（最终仍显著，t=−3.4）")
ax.axhline(0, color=MUT, linewidth=0.8)
ax.legend(facecolor="#161b22", labelcolor=FG, edgecolor=GRID, fontsize=9)
style(ax)
save(fig, "fig1_artifact_funnel.png")

# ── 图2：七因子终版 IC（原始 vs 价格中性，样本外）────────────────────
fs = pd.read_csv(LOGS / "expanded_factor_summary.csv")
fs = fs.sort_values("eval_ic_pn")
y = np.arange(len(fs))
fig, ax = plt.subplots(figsize=(8.2, 4.0))
ax.barh(y + 0.18, fs["eval_ic"], 0.36, color="#58a6ff", label="原始 Δp IC（样本外）")
ax.barh(y - 0.18, fs["eval_ic_pn"], 0.36, color="#3fb950", label="价格中性 IC（样本外）")
for i, (_, r) in enumerate(fs.iterrows()):
    ax.text(r["eval_ic_pn"] + (0.004 if r["eval_ic_pn"] >= 0 else -0.004),
            i - 0.18, f"t={r['eval_tstat_pn']:.1f}", va="center",
            ha="left" if r["eval_ic_pn"] >= 0 else "right", color=MUT, fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(fs["factor"], fontsize=9, color=FG)
ax.axvline(0, color=MUT, linewidth=0.8)
ax.set_title("七因子样本外 IC（4000 市场 × 574 天，后半段；价格中性 = 对 [1,p,p²] 残差）")
ax.legend(facecolor="#161b22", labelcolor=FG, edgecolor=GRID, fontsize=9, loc="lower right")
style(ax)
save(fig, "fig2_factor_ic.png")

# ── 图3：因子长期多空累计 Δp 价差 ────────────────────────────────────
lt = json.loads((LOGS / "factor_longterm.json").read_text(encoding="utf-8"))
palette = {"PM-MOM": "#f85149", "PM-VOL": "#d29922", "PM-LIQ": "#3fb950",
           "PM-EXTR": "#58a6ff", "PM-SPREAD": "#bc8cff", "PM-DRIFT": "#8b949e",
           "PM-TTR": "#39c5cf"}
fig, ax = plt.subplots(figsize=(8.6, 4.2))
for f, d in lt["factors"].items():
    dates = pd.to_datetime(d["dates"])
    lbl = f + ("（反向）" if d["orient"] < 0 else "")
    ax.plot(dates, d["cum_dp_pp"], color=palette.get(f, MUT), linewidth=1.5, label=lbl)
split = pd.Timestamp(lt["split_date"])
ax.axvline(split, color=FG, linestyle="--", linewidth=1)
ax.text(split, ax.get_ylim()[1] * 0.95, " ← 方向估计 | 样本外 → ", color=FG, fontsize=8.5)
ax.set_title("因子五分位多空累计 Δp 价差（百分点，5 天调仓，107 期）")
ax.set_ylabel("累计 Δp（pp）")
ax.legend(facecolor="#161b22", labelcolor=FG, edgecolor=GRID, fontsize=8, ncol=2)
style(ax)
save(fig, "fig3_longterm_ls.png")

# ── 图4：分类别价格中性 IC（MOM 反转的结构）──────────────────────────
cat = pd.read_csv(LOGS / "expanded_category_ic.csv")
mom = cat[cat["factor"] == "PM-MOM"].sort_values("mean_ic")
fig, ax = plt.subplots(figsize=(7.2, 3.4))
colors = ["#3fb950" if abs(t) >= 2 else "#6e7681" for t in mom["tstat"]]
ax.barh(mom["category"], -mom["mean_ic"], color=colors)
for i, (_, r) in enumerate(mom.iterrows()):
    ax.text(-r["mean_ic"] + 0.002, i, f"t={abs(r['tstat']):.1f}", va="center", color=MUT, fontsize=8.5)
ax.set_title("PM-MOM 反转强度（按类别，价格中性 |IC|；绿色 = |t|≥2 显著）")
ax.set_xlabel("|IC|（反转方向）")
ax.tick_params(axis="y", labelcolor=FG)
style(ax)
save(fig, "fig4_mom_by_category.png")

# ── 图5：组合净值（组合套利 50% + 做空赢家 50%）──────────────────────
bt = pickle.loads((ROOT / "cache" / "backtest_result.pkl").read_bytes())
def ser(key):
    pts = bt[key]["equity"]
    return pd.Series({pd.Timestamp(p["date"]): p["value"] for p in pts}).sort_index()
fig, ax = plt.subplots(figsize=(8.6, 4.0))
ax.plot(ser("combined"), color="#58a6ff", linewidth=2, label="合并组合")
ax.plot(ser("short_winners"), color="#f85149", linewidth=1.3, linestyle="--", label="做空赢家 (50%)")
ax.plot(ser("portfolio_arb"), color="#bc8cff", linewidth=1.3, linestyle="--", label="组合套利 (50%)")
m = bt["combined"]["metrics"]
ax.set_title(f"组合净值（本地数据 574 天）：+{m['total_return_pct']:.0f}%，"
             f"Sharpe {m['sharpe_ratio']:.2f}，最大回撤 {m['max_drawdown_pct']:.1f}%")
ax.set_ylabel("净值（USD）")
ax.legend(facecolor="#161b22", labelcolor=FG, edgecolor=GRID, fontsize=9)
style(ax)
save(fig, "fig5_portfolio_equity.png")

# ── 图6：做空赢家每期净收益分布（核心价格带）─────────────────────────
pr = pd.read_csv(LOGS / "momrev_period_returns.csv")
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.hist(pr["short_net"] * 100, bins=30, color="#f85149", alpha=0.85, edgecolor=DARK)
mu = pr["short_net"].mean() * 100
ax.axvline(mu, color=FG, linestyle="--", linewidth=1.2)
ax.text(mu + 1, ax.get_ylim()[1] * 0.9, f"均值 +{mu:.1f}%/5天", color=FG, fontsize=9)
ax.axvline(0, color=MUT, linewidth=0.8)
ax.set_title("做空赢家空头腿：每期净收益分布（0.15–0.85 核心带，真实点差成本，107 期）")
ax.set_xlabel("净收益（%/5 天）")
style(ax)
save(fig, "fig6_sw_distribution.png")

print("All figures done.")
