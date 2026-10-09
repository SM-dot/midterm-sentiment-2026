"""Headline numbers + power analysis for the equity test (results/summary_numbers.json)."""
import json, numpy as np, pandas as pd
from scipy import stats
from .config import RES, PROC, START, END
from .stats_utils import logit


def main():
    pm = pd.read_csv(PROC / "markets_1600et.csv", index_col=0, parse_dates=True).poly_sweep_1600.loc[str(START):str(END)].dropna()
    B = pd.read_csv(RES / "basket_political_beta.csv")
    ls = B[(B.basket == "LS") & (B["sample"] == "full")].iloc[0]
    se = ls.g_total / ls.t_total
    tot = float(logit(pm.iloc[-1]) - logit(pm.iloc[0]))
    P = pd.read_csv(RES / "political_betas.csv"); agree = int((np.sign(P.g_total_full) == P["sign"]).sum())
    R = pd.read_csv(RES / "sentiment_market_corr.csv")
    out = {"p_start": float(pm.iloc[0]), "p_end": float(pm.iloc[-1]), "dlogit_total": tot,
           "LS_gamma_se": float(se),
           "MDE_gamma_80pct_power": float(2.8 * se),            # (z_0.975 + z_0.80) * SE
           "MDE_cum_AR_over_window": float(2.8 * se * tot),
           "sign_agree": agree, "binom_p": float(stats.binomtest(agree, len(P), 0.5, alternative="greater").pvalue),
           "n_sig_5pct": int((P.t_total_full.abs() > 1.96).sum()),
           "n_weeks": int(R.n_weeks.max()), "weekly_corr_se": float(1 / np.sqrt(R.n_weeks.max()))}
    json.dump(out, open(RES / "summary_numbers.json", "w"), indent=2); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
