"""All report figures. Rules: fixed categorical slot per issue (never re-coloured by rank), one y-axis per
panel (series of different units are z-scored or put in stacked panels), hairline grids."""
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.dates as mdates
from .config import PROC, RES, START, END, SPLIT, ELECTION_DAY
from .viz import setup, save, SLOTS, INK2, MUTED, GRID, DEM, REP, DIV
from .stats_utils import logit

setup()


def issue_colors():
    iss = pd.read_csv(RES / "issue_ranking.csv").issue.tolist()
    return {k: SLOTS[i] for i, k in enumerate(iss)}


def _dates(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))


def fig_markets():
    m = pd.read_csv(PROC / "markets.csv", index_col=0, parse_dates=True).loc[str(START):str(END)]
    big = pd.read_csv(RES / "largest_market_moves.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for col, lab, c in [("poly_DSenate_DHouse", "Polymarket: D Senate + D House", SLOTS[0]),
                        ("kalshi_sweep", "Kalshi: Democrats sweep", SLOTS[1]),
                        ("kalshi_house_d", "Kalshi: D win House", SLOTS[2]),
                        ("kalshi_senate_d", "Kalshi: D win Senate", SLOTS[6])]:
        ax.plot(m.index, m[col] * 100, color=c, lw=1.8 if "sweep" in col or "DHouse" in col else 1.2, label=lab)
    # labels verified against the headline corpus on/around each date (see results/largest_market_moves.csv)
    events = {"2025-11-05": "Nov-4 off-year sweep\n+ CA Prop 50", "2025-11-10": "Shutdown deal:\nDem defections",
              "2025-12-03": "TN-7 special", "2026-09-16": "Mideast escalation /\noil-shock risk"}
    key = []
    for i, (d, lab) in enumerate(events.items(), 1):
        d = pd.Timestamp(d); ax.axvline(d, color=MUTED, lw=0.7, zorder=0)
        ax.text(d + pd.Timedelta(days=4 if i == 2 else 0), 101, str(i), fontsize=8, color=INK2, ha="center", va="bottom", fontweight="bold")
        key.append(f"{i}  {lab.replace(chr(10), ' ')}")
    ax.text(0.99, 0.03, "\n".join(key), transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)
    ax.axvline(pd.Timestamp(SPLIT), color=INK2, lw=0.8)
    ax.text(pd.Timestamp(SPLIT), 1.5, " in-sample | out-of-sample", fontsize=7, color=INK2, va="bottom", ha="center")
    ax.set_ylim(0, 100); ax.set_ylabel("Implied probability (%)"); _dates(ax)
    ax.set_title("Event-contract probabilities for the 2026 midterms")
    ax.legend(loc="upper left", ncol=4, bbox_to_anchor=(0, -0.09)); ax.set_ylim(0, 108); save(fig, "01_markets")


def fig_salience():
    R = pd.read_csv(RES / "issue_ranking_all.csv").head(10)
    v = pd.read_csv(RES / "kalshi_validation.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.75})
    y = np.arange(len(R))[::-1]; h = 0.26
    for j, (col, lab) in enumerate([("share_news", "News headlines"), ("share_social_engagement", "Social (engagement-weighted)"),
                                    ("salience_recent60", "Last 60 days (both)")]):
        a.barh(y + (1 - j) * h, R[col] * 100, height=h - 0.03, color=SLOTS[[0, 1, 6][j]], label=lab)
    a.set_yticks(y); a.set_yticklabels(R.issue); a.set_xlabel("Share of issue-relevant discourse (%)")
    a.set_title("What the discourse is about"); a.legend(loc="lower right")
    v = v.sort_values("last")
    b.barh(v.iloc[:, 0], v["last"] * 100, color=SLOTS[0], height=0.6)
    for yy, val in enumerate(v["last"] * 100):
        b.text(val + 1, yy, f"{val:.0f}%", va="center", fontsize=8, color=INK2)
    b.set_xlim(0, 105); b.set_xlabel("Kalshi price (%)")
    b.set_title("Market: top issue in 2026 exit poll"); save(fig, "02_issue_salience")


def fig_salience_time():
    col = issue_colors()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    for ax, ch in zip(axes, ["news", "social"]):
        s = pd.read_csv(RES / f"salience_{ch}.csv", index_col=0, parse_dates=True)
        s = s[[k for k in col if k in s]].resample("W-SUN").mean()
        s = s.div(s.sum(1), axis=0) * 100
        ax.stackplot(s.index, s.T.values, colors=[col[k] for k in s], labels=s.columns, edgecolor="#fcfcfb", linewidth=0.6)
        ax.set_ylim(0, 100); ax.set_ylabel("% of issue discourse"); ax.set_title(f"Issue attention over time — {ch}")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8); _dates(axes[1]); save(fig, "03_salience_time")


def fig_indices():
    col = issue_colors(); I = pd.read_csv(RES / "sentiment_index.csv", index_col=0, parse_dates=True)
    m = pd.read_csv(PROC / "markets.csv", index_col=0, parse_dates=True)
    L = logit(m.poly_DSenate_DHouse).reindex(I.index).ffill()
    Lz = (L - L.mean()) / L.std()
    keys = [k for k in col] + ["MSI"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.2), sharex=True, sharey=True)
    for ax, k in zip(axes.flat, keys):
        ax.plot(Lz.index, Lz, color=MUTED, lw=1.0, label="P(D sweep), logit z")
        ax.plot(I.index, I[k], color=col.get(k, SLOTS[0]) if k != "MSI" else "#0b0b0b", lw=1.5, label="Sentiment z")
        ax.axhline(0, color=GRID, lw=0.8); ax.set_title(k, fontsize=9)
    axes[0, 0].legend(loc="lower left", fontsize=7); _dates(axes[-1, 0])
    for ax in axes[-1]: ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Topic sentiment indices vs. Democratic-sweep probability (both standardised)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(); save(fig, "04_sentiment_indices")


def fig_corr():
    R = pd.read_csv(RES / "sentiment_market_corr.csv").iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.6)); y = np.arange(len(R))
    ax.hlines(y, R.ci_lo, R.ci_hi, color=SLOTS[0], lw=2, alpha=0.5)
    ax.plot(R.daily_dcorr, y, "o", color=SLOTS[0], ms=8, label="Daily Δ correlation (95% block-bootstrap CI)")
    ax.plot(R.weekly_dcorr, y, "D", color=SLOTS[1], ms=7, label="Weekly Δ correlation")
    ax.axvline(0, color=INK2, lw=0.8); ax.set_yticks(y); ax.set_yticklabels(R["index"])
    for yy, q in zip(y, R.q_bh):
        if q < 0.10: ax.text(ax.get_xlim()[1], yy, " q<0.10" if q >= 0.05 else " q<0.05", va="center", fontsize=7, color=INK2)
    ax.set_xlabel("corr(Δ sentiment index, Δ logit P(D sweep))")
    ax.set_title("Sentiment changes vs. sweep-probability changes"); ax.legend(loc="lower left", fontsize=7)
    save(fig, "05_corr_summary")


def fig_ccf():
    C = pd.read_csv(RES / "ccf.csv", index_col=0)
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("div", [DIV[0], DIV[1], DIV[2]])
    v = np.nanmax(np.abs(C.values))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(C.T.values, aspect="auto", cmap=cm, vmin=-v, vmax=v)
    ax.set_yticks(range(C.shape[1])); ax.set_yticklabels(C.columns, fontsize=8)
    ax.set_xticks(range(0, len(C), 2)); ax.set_xticklabels(C.index[::2])
    ax.axvline(list(C.index).index(0), color=INK2, lw=0.8)
    ax.set_xlabel("lag L (days): corr(ΔI(t−L), ΔlogitP(t));  L>0 ⇒ sentiment leads market")
    ax.set_title("Lead–lag structure"); ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.025, label="correlation")
    save(fig, "06_ccf")


def fig_rolling():
    r = pd.read_csv(RES / "rolling_corr.csv", index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(r.index, r.r90, color=SLOTS[0]); ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title("Rolling 90-day correlation: Δ MSI vs Δ logit P(D sweep)"); ax.set_ylabel("correlation"); _dates(ax)
    save(fig, "07_rolling_corr")


def fig_betas():
    P = pd.read_csv(RES / "political_betas.csv").sort_values("t_total_full")
    fig, ax = plt.subplots(figsize=(9, 5.6)); y = np.arange(len(P))
    ax.barh(y, P.t_total_full, color=[DEM if s > 0 else REP for s in P["sign"]], height=0.65)
    ax.axvline(0, color=INK2, lw=0.8)
    for x in (-1.96, 1.96): ax.axvline(x, color=MUTED, lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels([f"{t}  ({g})" for t, g in zip(P.ticker, P.group)], fontsize=8)
    ax.set_xlabel("HAC t-stat of total sweep beta  (AR on Δ logit P, contemporaneous + 1-day lag)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=DEM, label="Ex-ante: Dem-sweep beneficiary"), Patch(color=REP, label="Ex-ante: Dem-sweep loser")],
              loc="lower right")
    ax.set_title("Which stocks actually trade on the sweep probability?"); save(fig, "08_political_betas")


def fig_basket():
    B = pd.read_csv(RES / "basket_returns.csv", index_col=0, parse_dates=True)
    pm = pd.read_csv(PROC / "markets_1600et.csv", index_col=0, parse_dates=True).poly_sweep_1600.reindex(B.index)
    I = pd.read_csv(RES / "sentiment_index.csv", index_col=0, parse_dates=True).MSI.reindex(B.index)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.4), sharex=True, gridspec_kw={"height_ratios": [1.4, 1, 1]})
    for col, lab, c in [("LS", "Thesis L/S (beneficiaries − losers)", "#0b0b0b"), ("long", "Long leg: Dem-sweep beneficiaries", DEM),
                        ("short", "Short leg: Dem-sweep losers", REP), ("LS_data_driven", "Data-driven L/S (formed in-sample)", SLOTS[3])]:
        axes[0].plot(B.index, ((1 + B[col].fillna(0)).cumprod() - 1) * 100, color=c, lw=1.8 if col == "LS" else 1.2, label=lab)
    axes[0].axhline(0, color=INK2, lw=0.8); axes[0].set_ylabel("Cumulative market-adj. return (%)")
    axes[0].set_title("Mid-term basket, sweep probability and sentiment"); axes[0].legend(fontsize=7, loc="upper left")
    axes[1].plot(pm.index, pm * 100, color=SLOTS[0]); axes[1].set_ylabel("P(D sweep) %, 16:00 ET")
    axes[2].plot(I.index, I, color=SLOTS[6]); axes[2].axhline(0, color=GRID); axes[2].set_ylabel("MSI (z)")
    for ax in axes: ax.axvline(pd.Timestamp(SPLIT), color=MUTED, lw=0.8)
    _dates(axes[-1]); save(fig, "09_basket")


def fig_event():
    E = pd.read_csv(RES / "event_study.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(9, 3.6)); x = np.arange(len(E))
    ax.bar(x, E.LS_AR * 100, color=[DEM if d > 0 else REP for d in E.dL], width=0.6)
    ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(x)
    ax.set_xticklabels([f"{d:%d %b %y}\nΔp {p:+.1f}pp" for d, p in zip(E.date, E.dp_pp)], fontsize=7)
    ax.set_ylabel("L/S basket AR (%)"); ax.set_title("Basket return on the 10 largest sweep-probability shocks")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=DEM, label="P(D sweep) up"), Patch(color=REP, label="P(D sweep) down")], fontsize=7)
    save(fig, "10_event_study")


def fig_topics():
    T = pd.read_csv(RES / "topic_model_comparison.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    labels = [f"{m}\nK={k}" for m, k in zip(T.model, T.K)]
    cols = [SLOTS[{"LDA": 1, "NMF": 2, "BERTopic": 0}[m]] for m in T.model]
    for ax, met, ttl in zip(axes, ["c_npmi", "diversity", "stability_jaccard"], ["NPMI coherence", "Topic diversity", "Seed stability (Jaccard)"]):
        ax.bar(range(len(T)), T[met].fillna(0), color=cols, width=0.65); ax.set_xticks(range(len(T)))
        ax.set_xticklabels(labels, fontsize=6.5); ax.set_title(ttl, fontsize=9)
    fig.tight_layout(); save(fig, "11_topic_model_comparison")


def main():
    for f in (fig_markets, fig_salience, fig_salience_time, fig_indices, fig_corr, fig_ccf, fig_rolling,
              fig_betas, fig_basket, fig_event, fig_topics):
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            print("FAILED", f.__name__, repr(e))


if __name__ == "__main__":
    main()
