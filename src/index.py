"""Topic-sentiment and topic-salience indices (daily, calendar time).

For channel c in {news, social}, issue k, day t (docs i in the cell, weights w_i, scores s_i):

    raw_kct   = sum_i w_i s_i / sum_i w_i
    shrunk    = (sum_i w_i s_i + n0 * mu_kc,t-1) / (sum_i w_i + n0)      # empirical-Bayes shrinkage toward the
                                                                          # EXPANDING (past-only) mean -> thin days
                                                                          # are pulled to the prior, no look-ahead
    smooth    = EWMA(shrunk, half-life = 3 days)
    z_kct     = (smooth - mean_kc) / sd_kc                               # per-channel standardisation removes the
                                                                          # level gap between left-leaning Bluesky
                                                                          # and mainstream headlines
    I_kt      = mean_c z_kct                                              # issue sentiment index
    MSI_t     = sum_k omega_k I_kt,  omega_k = share of voice in formation period (ex-ante, fixed)

Salience (attention) index: share_kct = weighted docs on issue k / weighted docs on any issue, 7-day mean.
"""
import numpy as np, pandas as pd
from .config import PROC, RES, START, END, SPLIT

N0, HALF_LIFE, MIN_DAYS = 20.0, 3, 21
NON_ISSUES = {"Other / non-issue", "Non-US / off-topic", "Horse race & campaign meta"}


def load_docs(score="sent"):
    c = pd.read_parquet(PROC / "corpus.parquet")
    t = pd.read_parquet(PROC / "doc_topics.parquet")
    s = pd.read_parquet(PROC / "sentiment.parquet")
    a = pd.read_parquet(PROC / "doc_issue.parquet")
    d = c.merge(t, on="doc_id").merge(s, on="doc_id").merge(a, on="doc_id")
    d["score"] = d[score]
    return d


def _cell_stats(d):
    d = d.assign(ws=d.weight * d.score)
    g = d.groupby(["channel", "issue", "date"]).agg(sw=("weight", "sum"), sws=("ws", "sum"), n=("score", "size"))
    return g.reset_index()


def sentiment_panel(d, issues, smooth=True):
    days = pd.date_range(START, END, freq="D")
    g = _cell_stats(d[d.issue.isin(issues)])
    out = {}
    for (ch, k), x in g.groupby(["channel", "issue"]):
        x = x.set_index("date").reindex(days).fillna({"sw": 0, "sws": 0, "n": 0})
        cum_sw, cum_sws = x.sw.cumsum().shift(1), x.sws.cumsum().shift(1)
        mu = (cum_sws / cum_sw).bfill().fillna(0)            # expanding mean, past-only
        shrunk = (x.sws + N0 * mu) / (x.sw + N0)
        sm = shrunk.ewm(halflife=HALF_LIFE).mean() if smooth else shrunk
        sm[x.n.cumsum() < 30] = np.nan                       # burn-in: need some history first
        out[(ch, k)] = sm
    raw = pd.DataFrame(out)
    z = (raw - raw.mean()) / raw.std()
    idx = z.T.groupby(level=1).mean().T                      # average the channels -> issue index
    return raw, z, idx


def salience_panel(d):
    days = pd.date_range(START, END, freq="D")
    g = d.groupby(["channel", "date", "issue"]).weight.sum().unstack().reindex(columns=None)
    out = {}
    for ch, x in g.groupby(level=0):
        x = x.droplevel(0).reindex(days).fillna(0)
        x = x.drop(columns=[k for k in x if k in NON_ISSUES], errors="ignore")
        sh = x.div(x.sum(1), axis=0).rolling(7, min_periods=3).mean()
        out[ch] = sh
    return out


def overall_tone(d, smooth=True):
    days = pd.date_range(START, END, freq="D")
    x = d[~d.issue.isin({"Non-US / off-topic"})]
    out = {}
    for ch, y in x.groupby("channel"):
        s = (y.weight * y.score).groupby(y.date).sum() / y.weight.groupby(y.date).sum()
        s = s.reindex(days)
        out[ch] = s.ewm(halflife=HALF_LIFE).mean() if smooth else s
    o = pd.DataFrame(out); return ((o - o.mean()) / o.std()).mean(1)


def main(score="sent", tag=""):
    d = load_docs(score)
    issues_df = pd.read_csv(RES / "issue_ranking.csv")
    issues = issues_df.issue.tolist()
    raw, z, idx = sentiment_panel(d, issues)
    _, _, idx_u = sentiment_panel(d, issues, smooth=False)   # unsmoothed: used for all daily-change inference
    form = d[(d.date < pd.Timestamp(SPLIT)) & d.issue.isin(issues)]
    omega = form.groupby("issue").weight.sum(); omega /= omega.sum()
    idx["MSI"] = (idx[omega.index] * omega).sum(1, min_count=len(omega) // 2) / \
                 idx[omega.index].notna().mul(omega).sum(1)
    idx["Overall tone"] = overall_tone(d)
    idx_u["MSI"] = (idx_u[omega.index] * omega).sum(1, min_count=len(omega) // 2) / idx_u[omega.index].notna().mul(omega).sum(1)
    idx_u["Overall tone"] = overall_tone(d, smooth=False)
    idx.index.name = idx_u.index.name = "date"
    idx.to_csv(RES / f"sentiment_index{tag}.csv"); idx_u.to_csv(RES / f"sentiment_index_daily{tag}.csv")
    z.columns = [f"{a}|{b}" for a, b in z.columns]; z.index.name = "date"; z.to_csv(RES / f"sentiment_z_by_channel{tag}.csv")
    raw.columns = [f"{a}|{b}" for a, b in raw.columns]; raw.index.name = "date"; raw.to_csv(RES / f"sentiment_raw_by_channel{tag}.csv")
    if not tag:
        sal = salience_panel(d)
        for ch, s in sal.items():
            s.index.name = "date"; s.to_csv(RES / f"salience_{ch}.csv")
        omega.rename("omega").to_csv(RES / "msi_weights.csv")
    print(idx.describe().T[["count", "mean", "std"]])
    return idx


if __name__ == "__main__":
    main()
