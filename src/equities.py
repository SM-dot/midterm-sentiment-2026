"""Mid-term basket: market-adjusted performance, political ("sweep") betas, and link to sentiment.

Market adjustment: AR_it = r_it - beta_i * r_SPY,t with Blume-adjusted betas estimated on 2024-07-01..2025-07-18,
strictly BEFORE the study window (no look-ahead, and betas are not contaminated by the political shock we
are trying to measure).
Political beta:   AR_it = a_i + g0_i dL_t + g1_i dL_{t-1} + e_it,   dL = change in logit P(D sweep) sampled at
                  16:00 ET (same instant as the equity close), HAC(5) errors.
"""
import json, numpy as np, pandas as pd
import statsmodels.api as sm
from .config import PROC, RES, BASKET, MARKET, BETA_START, BETA_END, START, END, SPLIT
from .stats_utils import logit, hac_ols, ccf, granger, boot_corr_ci

ANN = 252


def load():
    px = pd.read_csv(PROC / "prices.csv", index_col=0, parse_dates=True)
    r = px.pct_change(fill_method=None)
    pm = pd.read_csv(PROC / "markets_1600et.csv", index_col=0, parse_dates=True)
    return px, r, pm


def betas(r):
    w = r.loc[str(BETA_START):str(BETA_END)]
    out = {}
    for t in BASKET:
        d = w[[t, MARKET]].dropna()
        b = sm.OLS(d[t], sm.add_constant(d[MARKET])).fit().params[MARKET]
        out[t] = 0.67 * b + 0.33
    return pd.Series(out)


def perf(x):
    x = x.dropna(); eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
    down = np.sqrt((np.minimum(x, 0) ** 2).mean()) * np.sqrt(ANN)
    ann = (1 + x).prod() ** (ANN / len(x)) - 1; vol = x.std() * np.sqrt(ANN)
    return {"ann_return": ann, "ann_vol": vol, "sharpe": x.mean() / x.std() * np.sqrt(ANN),
            "sortino": x.mean() * ANN / down, "max_dd": dd.min(), "calmar": ann / -dd.min() if dd.min() < 0 else np.nan,
            "t_mean": x.mean() / (x.std() / np.sqrt(len(x))), "n": len(x)}


def political_beta(ar, dL):
    X = pd.concat([dL.rename("dL"), dL.shift(1).rename("dL_lag")], axis=1)
    m = hac_ols(ar, X, lags=5)
    tot = m.params["dL"] + m.params["dL_lag"]
    V = m.cov_params().loc[["dL", "dL_lag"], ["dL", "dL_lag"]].values.sum()
    return {"g0": m.params["dL"], "t0": m.tvalues["dL"], "g_total": tot, "t_total": tot / np.sqrt(V), "r2": m.rsquared}


def main():
    px, r, pm = load()
    b = betas(r)
    S = r.loc[str(START):str(END)]
    AR = S[list(BASKET)].sub(S[MARKET].values[:, None] * b.values[None, :], axis=0)
    L = logit(pm.poly_sweep_1600).reindex(S.index)  # value at the 16:00 ET close of each trading day
    dL = L.diff()
    sign = pd.Series({k: v[0] for k, v in BASKET.items()})
    meta = pd.DataFrame({k: {"sign": v[0], "group": v[1], "thesis": v[2]} for k, v in BASKET.items()}).T

    # ---- per-stock political betas: full, in-sample, out-of-sample ----
    rows = []
    for t in BASKET:
        full = political_beta(AR[t], dL)
        is_ = political_beta(AR[t][:str(SPLIT)], dL[:str(SPLIT)])
        oos = political_beta(AR[t][str(SPLIT):], dL[str(SPLIT):])
        rows.append({"ticker": t, **meta.loc[t].to_dict(), "beta_mkt": b[t],
                     **{f"{k}_full": v for k, v in full.items()},
                     "g_total_is": is_["g_total"], "t_total_is": is_["t_total"],
                     "g_total_oos": oos["g_total"], "t_total_oos": oos["t_total"],
                     "car_window": (1 + AR[t].dropna()).prod() - 1})
    PB = pd.DataFrame(rows); PB.to_csv(RES / "political_betas.csv", index=False)

    # ---- baskets ----
    long_, short_ = [t for t in BASKET if sign[t] > 0], [t for t in BASKET if sign[t] < 0]
    MB = pd.DataFrame({"long": AR[long_].mean(1), "short": AR[short_].mean(1)})
    MB["LS"] = MB.long - MB.short
    is_rank = PB.set_index("ticker").g_total_is.sort_values()
    dlong, dshort = list(is_rank.index[-5:]), list(is_rank.index[:5])
    MB["LS_data_driven"] = AR[dlong].mean(1) - AR[dshort].mean(1)
    raw_ls = S[long_].mean(1) - S[short_].mean(1)
    MB["LS_raw"] = raw_ls
    MB.index.name = "date"; MB.to_csv(RES / "basket_returns.csv")
    json.dump({"thesis_long": long_, "thesis_short": short_, "data_long": dlong, "data_short": dshort,
               "sign_agreement_is": float((np.sign(is_rank) == sign[is_rank.index]).mean()),
               "sign_agreement_full": float((np.sign(PB.set_index("ticker").g_total_full) == sign).mean())},
              open(RES / "basket_composition.json", "w"), indent=2)

    bb = []
    for col in ["LS", "long", "short", "LS_data_driven"]:
        for lab, sl in [("full", slice(None)), ("in-sample", slice(None, str(SPLIT))), ("out-of-sample", slice(str(SPLIT), None))]:
            pb_ = political_beta(MB[col][sl], dL[sl])
            corr = pd.concat([MB[col][sl], dL[sl]], axis=1).dropna().corr().iloc[0, 1]
            bb.append({"basket": col, "sample": lab, "corr_dL": corr, **pb_, **perf(MB[col][sl])})
    pd.DataFrame(bb).to_csv(RES / "basket_political_beta.csv", index=False)

    # ---- event study on the largest probability shocks ----
    ev = dL.abs().dropna().sort_values(ascending=False).head(10).index
    nonev = MB.LS.drop(ev).dropna()
    es = pd.DataFrame({"date": ev, "dL": dL[ev].values, "dp_pp": 100 * pm.poly_sweep_1600.reindex(S.index).diff()[ev].values,
                       "LS_AR": MB.LS[ev].values, "signed_AR": np.sign(dL[ev].values) * MB.LS[ev].values,
                       "z_vs_nonevent": (MB.LS[ev].values - nonev.mean()) / nonev.std()}).sort_values("date")
    es.to_csv(RES / "event_study.csv", index=False)
    from scipy import stats
    es_t = stats.ttest_1samp(es.signed_AR, 0)
    # ---- sentiment <-> basket ----
    Is = pd.read_csv(RES / "sentiment_index.csv", index_col=0, parse_dates=True)
    I = pd.read_csv(RES / "sentiment_index_daily.csv", index_col=0, parse_dates=True)
    # The calendar-day index for UTC day t closes at 20:00 ET, after the 16:00 equity close. To keep the regressor
    # inside the return interval we use the PREVIOUS calendar day's index: change between t-1 and the prior trading
    # day's t-1 (weekend sentiment accumulates into Monday's change).
    Ilag = I.shift(1)
    dI = Ilag.reindex(S.index).diff()
    rows = []
    for k in I:
        x = dI[k]
        m1 = hac_ols(MB.LS, x.rename("dI"), lags=5)
        m2 = hac_ols(MB.LS, pd.concat([x.rename("dI"), dL.rename("dL")], axis=1), lags=5)
        w = pd.concat([MB.LS.resample("W-FRI").sum(), Ilag[k].resample("W-FRI").mean().diff()], axis=1).dropna()
        rows.append({"index": k, "daily_corr": pd.concat([x, MB.LS], axis=1).dropna().corr().iloc[0, 1],
                     "beta_dI": m1.params["dI"], "t_dI": m1.tvalues["dI"],
                     "beta_dI_ctrl_dL": m2.params["dI"], "t_dI_ctrl_dL": m2.tvalues["dI"], "t_dL_ctrl": m2.tvalues["dL"],
                     "weekly_corr": w.corr().iloc[0, 1],
                     "level_corr_CAR": pd.concat([Is[k].reindex(S.index), (1 + MB.LS.fillna(0)).cumprod()], axis=1).dropna().corr().iloc[0, 1],
                     "granger_p_sent_to_basket": granger(MB.LS, x)[1]})
    SB = pd.DataFrame(rows); SB.to_csv(RES / "sentiment_basket_corr.csv", index=False)
    pd.DataFrame({"MSI": ccf(dI.MSI, MB.LS, 10), "dL": ccf(dL, MB.LS, 10)}).to_csv(RES / "ccf_basket.csv")
    mbci = boot_corr_ci(dL, MB.LS, block=5)

    # ---- factor-model robustness (Fama-French 5 + momentum, data through last available month) ----
    fac = {}
    try:
        ff = pd.read_csv(PROC / "ff_factors.csv", index_col=0, parse_dates=True)
        df = pd.concat([MB.LS_raw, ff, dL.rename("dL")], axis=1).dropna()
        Xf = df[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]]
        m0 = sm.OLS(df.LS_raw, sm.add_constant(Xf)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        m1 = sm.OLS(df.LS_raw, sm.add_constant(pd.concat([Xf, df.dL], axis=1))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        fac = {"n": int(m0.nobs), "end": str(df.index.max().date()),
               "alpha_ann_ff6": m0.params["const"] * ANN, "t_alpha_ff6": m0.tvalues["const"], "r2_ff6": m0.rsquared,
               "loadings_ff6": m0.params.drop("const").round(3).to_dict(), "t_loadings_ff6": m0.tvalues.drop("const").round(2).to_dict(),
               "gamma_dL_ff6": m1.params["dL"], "t_gamma_dL_ff6": m1.tvalues["dL"], "r2_ff6_plus_dL": m1.rsquared}
    except FileNotFoundError:
        pass

    # ---- exploratory sentiment-timing overlay (sign learned in-sample, applied out-of-sample) ----
    sig = Is.MSI.diff(5).shift(2).reindex(S.index)  # MSI of day t-2 is complete at 20:00 ET t-2 -> position set at close t-1
    is_sign = np.sign(pd.concat([sig, MB.LS], axis=1)[:str(SPLIT)].dropna().corr().iloc[0, 1])
    strat = (np.sign(sig) * is_sign * MB.LS)[str(SPLIT):]
    psig = dL.shift(1)
    p_sign = np.sign(pd.concat([psig, MB.LS], axis=1)[:str(SPLIT)].dropna().corr().iloc[0, 1])
    pstrat = (np.sign(psig) * p_sign * MB.LS)[str(SPLIT):]
    timing = {"msi_timing_oos": perf(strat), "is_sign_msi": is_sign,
              "prob_momentum_oos": perf(pstrat), "buy_and_hold_LS_oos": perf(MB.LS[str(SPLIT):]),
              "info_ratio_msi_vs_bh": (strat - MB.LS[str(SPLIT):]).mean() / (strat - MB.LS[str(SPLIT):]).std() * np.sqrt(ANN)}
    json.dump({"factor_model": fac, "timing": timing, "event_study_t": es_t.statistic, "event_study_p": es_t.pvalue,
               "corr_dL_LS_ci": list(mbci)}, open(RES / "basket_extras.json", "w"), indent=2, default=float)
    print(PB[["ticker", "sign", "g_total_full", "t_total_full", "g_total_is", "g_total_oos"]].round(3).to_string())
    print(pd.DataFrame(bb).round(3).to_string()); print(SB.round(3).to_string()); print(json.dumps(fac, indent=1, default=float))


if __name__ == "__main__":
    main()
