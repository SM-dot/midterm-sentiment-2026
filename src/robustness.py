"""Robustness grid for the headline sentiment<->market result. Every row is corr(dI_k, dlogit P) on daily data."""
import numpy as np, pandas as pd
from .config import PROC, RES, START, END
from .stats_utils import logit
from . import index as ix


def dcorr(x, y):
    return pd.concat([x.diff(), y.diff()], axis=1).dropna().corr().iloc[0, 1]


def main():
    days = pd.date_range(START, END, freq="D")
    m = pd.read_csv(PROC / "markets.csv", index_col=0, parse_dates=True).reindex(days).ffill(limit=2)
    Lp, Lk = logit(m.poly_DSenate_DHouse), logit(m.kalshi_sweep)
    base = pd.read_csv(RES / "sentiment_index_daily.csv", index_col=0, parse_dates=True)
    vad = pd.read_csv(RES / "sentiment_index_daily_vader.csv", index_col=0, parse_dates=True)
    zc = pd.read_csv(RES / "sentiment_z_by_channel.csv", index_col=0, parse_dates=True)
    big = Lp.diff().abs().sort_values(ascending=False).head(10).index
    Lp_x = Lp.copy(); 
    rows = []
    for k in base.columns:
        r = {"index": k, "baseline (RoBERTa, unsmoothed, Polymarket)": dcorr(base[k], Lp),
             "VADER sentiment": dcorr(vad[k], Lp) if k in vad else np.nan,
             "Kalshi sweep target": dcorr(base[k], Lk),
             "news channel only": dcorr(zc[f"news|{k}"], Lp) if f"news|{k}" in zc else np.nan,
             "social channel only": dcorr(zc[f"social|{k}"], Lp) if f"social|{k}" in zc else np.nan}
        d = pd.concat([base[k].diff(), Lp.diff()], axis=1).drop(index=big, errors="ignore").dropna()
        r["excl. 10 largest market moves"] = d.corr().iloc[0, 1]
        r["level corr RoBERTa vs VADER index"] = pd.concat([base[k], vad.get(k)], axis=1).dropna().corr().iloc[0, 1] if k in vad else np.nan
        rows.append(r)
    # smoothing half-life sensitivity (re-build indices in memory)
    d = ix.load_docs("sent"); issues = pd.read_csv(RES / "issue_ranking.csv").issue.tolist()
    for hl in (1, 7):
        ix.HALF_LIFE = hl
        _, _, idx = ix.sentiment_panel(d, issues)
        for r in rows:
            if r["index"] in idx:
                r[f"half-life {hl}d"] = dcorr(idx[r["index"]], Lp)
    ix.HALF_LIFE = 3
    R = pd.DataFrame(rows).set_index("index"); R.to_csv(RES / "robustness.csv"); print(R.round(3).to_string())


if __name__ == "__main__":
    main()
