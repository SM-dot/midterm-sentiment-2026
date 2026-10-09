"""Sentiment <-> prediction-market analysis.

H1 (referendum hypothesis): worsening tone of issue discourse under a Republican trifecta raises the
    probability of a Democratic sweep  ->  corr(dI_k, dlogit p) < 0.
H2 (efficient markets): if event-contract prices are weak-form efficient, sentiment should co-move with
    them contemporaneously but should NOT forecast future price changes out of sample.
"""
import json, numpy as np, pandas as pd
from .config import PROC, RES, START, END
from .stats_utils import (logit, efficiency_battery, stationarity, hac_ols, boot_corr_ci, ccf, granger, bh, oos_r2)

TARGETS = {"poly_DSenate_DHouse": "Polymarket D sweep", "kalshi_sweep": "Kalshi D sweep",
           "kalshi_house_d": "Kalshi D House", "kalshi_senate_d": "Kalshi D Senate"}


def load():
    m = pd.read_csv(PROC / "markets.csv", index_col=0, parse_dates=True)
    days = pd.date_range(START, END, freq="D")
    m = m.reindex(days).ffill(limit=2)
    L = logit(m[list(TARGETS)])
    Is = pd.read_csv(RES / "sentiment_index.csv", index_col=0, parse_dates=True).reindex(days)        # smoothed: levels
    I = pd.read_csv(RES / "sentiment_index_daily.csv", index_col=0, parse_dates=True).reindex(days)  # unsmoothed: changes
    return m, L, I, Is


def main():
    m, L, I, Is = load()
    out = {}
    # ---- 1. market data quality + efficiency ----
    dL = L.diff()
    q = {"corr_daily_dlogit_poly_kalshi": dL.poly_DSenate_DHouse.corr(dL.kalshi_sweep),
         "corr_weekly_dlogit_poly_kalshi": L.resample("W-SUN").last().diff().corr().loc["poly_DSenate_DHouse", "kalshi_sweep"],
         "corr_level_poly_kalshi": m.poly_DSenate_DHouse.corr(m.kalshi_sweep),
         "mean_abs_gap_pp": 100 * (m.poly_DSenate_DHouse - m.kalshi_sweep).abs().mean(),
         "implied_from_marginals_corr": m.kalshi_sweep.corr(m.kalshi_house_d * m.kalshi_senate_d)}
    json.dump(q, open(RES / "market_quality.json", "w"), indent=2)
    eff = pd.DataFrame([efficiency_battery(dL[c], TARGETS[c]) for c in TARGETS])
    half = dL.index[len(dL) // 2]
    eff = pd.concat([eff, pd.DataFrame([efficiency_battery(dL.poly_DSenate_DHouse[:half], "Polymarket sweep, 1st half"),
                                        efficiency_battery(dL.poly_DSenate_DHouse[half:], "Polymarket sweep, 2nd half")])])
    eff.to_csv(RES / "market_efficiency.csv", index=False)
    st = pd.DataFrame([stationarity(L.poly_DSenate_DHouse, "logit p (Polymarket sweep)")] +
                      [stationarity(Is[c], c) for c in Is])
    st.to_csv(RES / "stationarity.csv", index=False)

    # ---- 2. contemporaneous association: levels (descriptive) vs changes (inferential) ----
    y = L.poly_DSenate_DHouse
    rows = []
    for k in I:
        x = I[k]
        lvl = pd.concat([Is[k], y], axis=1).dropna().corr().iloc[0, 1]
        dx, dy = x.diff(), y.diff()
        r = pd.concat([dx, dy], axis=1).dropna().corr().iloc[0, 1]
        mod = hac_ols(dy, dx.rename("dI"))
        lo, hi = boot_corr_ci(dx, dy)
        w = pd.concat([x.resample("W-SUN").mean(), y.resample("W-SUN").last()], axis=1).diff().dropna()
        rw = w.corr().iloc[0, 1]
        lag, gp_sm = granger(dy, dx); _, gp_ms = granger(dx, dy)
        rows.append({"index": k, "level_corr": lvl, "daily_dcorr": r, "ci_lo": lo, "ci_hi": hi,
                     "beta_hac": mod.params["dI"], "t_hac": mod.tvalues["dI"], "p_hac": mod.pvalues["dI"],
                     "weekly_dcorr": rw, "n_weeks": len(w), "granger_lag": lag,
                     "granger_p_sent_to_mkt": gp_sm, "granger_p_mkt_to_sent": gp_ms})
    R = pd.DataFrame(rows)
    R["q_bh"] = bh(R.p_hac); R["q_bh_granger"] = bh(R.granger_p_sent_to_mkt)
    R.to_csv(RES / "sentiment_market_corr.csv", index=False)

    # chamber decomposition + venue robustness: daily dcorr against every target
    dec = pd.DataFrame({TARGETS[t]: {k: pd.concat([I[k].diff(), L[t].diff()], axis=1).dropna().corr().iloc[0, 1]
                                     for k in I} for t in TARGETS})
    dec.to_csv(RES / "sentiment_corr_by_contract.csv")

    # lead-lag
    C = pd.DataFrame({k: ccf(I[k].diff(), y.diff()) for k in I}); C.index.name = "lag_days_sentiment_leads"
    C.to_csv(RES / "ccf.csv")

    # rolling 90d correlation of changes (composite)
    roll = pd.concat([I.MSI.diff(), y.diff()], axis=1).dropna()
    roll.columns = ["dMSI", "dL"]
    roll["r90"] = roll.dMSI.rolling(90, min_periods=60).corr(roll.dL)
    roll.to_csv(RES / "rolling_corr.csv")

    # ---- 3. out-of-sample predictability (H2) ----
    dy = y.diff()
    top = R.reindex(R.daily_dcorr.abs().sort_values(ascending=False).index)["index"]
    top = [k for k in top if k not in ("MSI", "Overall tone")][:4]
    oos = []
    for name, X in [("MSI", I[["MSI"]]), ("Overall tone", I[["Overall tone"]]), ("Top-4 issues", I[top]),
                    ("All issues", I.drop(columns=["MSI", "Overall tone"]))]:
        r2, p = oos_r2(dy, X.diff(), start=120)
        oos.append({"predictors": name, "oos_r2": r2, "clark_west_p": p})
    ar = oos_r2(dy, dy.to_frame("lag"), start=120)
    oos.append({"predictors": "AR(3) own lags (benchmark)", "oos_r2": ar[0], "clark_west_p": ar[1]})
    pd.DataFrame(oos).to_csv(RES / "oos_predictability.csv", index=False)

    # ---- 4. biggest market moves (for narrative annotation) ----
    big = dy.abs().sort_values(ascending=False).head(12).index
    pd.DataFrame({"date": big, "dlogit": dy[big].values, "p": m.poly_DSenate_DHouse[big].values,
                  "p_prev": m.poly_DSenate_DHouse.shift(1)[big].values}).sort_values("date").to_csv(
        RES / "largest_market_moves.csv", index=False)
    print(eff.round(3).to_string()); print(R.round(3).to_string()); print(pd.DataFrame(oos).round(4))


if __name__ == "__main__":
    main()
