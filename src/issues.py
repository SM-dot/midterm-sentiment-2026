"""Micro-topics -> voter-issue taxonomy -> issue salience ranking (+ external validation).

Inputs : results/topic_info.csv (BERTopic output), results/topic_issue_map.csv (annotated map)
Outputs: data/processed/doc_issue.parquet, results/issue_ranking.csv, results/taxonomy_agreement.json
"""
import json, numpy as np, pandas as pd
from scipy import stats
from .config import PROC, RES, END
from .topics import embed, ISSUE_ANCHORS
ISSUE_ANCHORS_KEYS = list(ISSUE_ANCHORS)

TAU = 0.50           # cosine threshold for re-assigning HDBSCAN outliers to the nearest mapped topic
NON_ISSUE = {"Other / non-issue", "Non-US / off-topic", "Horse race & campaign meta"}
MAX_ISSUES = 8       # indices are built for at most 8 issues (one categorical colour each, never cycled)
KALSHI_MAP = {"Economy and cost of living": "Economy & cost of living", "Democracy": "Democracy & rule of law",
              "Iran war and foreign policy": "Foreign policy & Iran war", "Health care": "Health care",
              "Immigration": "Immigration & ICE", "Crime": "Crime & public safety",
              "Abortion": "Abortion & reproductive rights"}


def cohen_kappa(a, b):
    a, b = pd.Series(a).values, pd.Series(b).values; labs = sorted(set(a) | set(b))
    po = np.mean(a == b); pe = sum(np.mean(a == l) * np.mean(b == l) for l in labs)
    return (po - pe) / (1 - pe)


def main():
    c = pd.read_parquet(PROC / "corpus.parquet")
    t = pd.read_parquet(PROC / "doc_topics.parquet")
    c = c.merge(t, on="doc_id")
    info = pd.read_csv(RES / "topic_info.csv")
    mp = pd.read_csv(RES / "topic_issue_map.csv")
    info = info.merge(mp[["topic", "issue"]], on="topic", how="left")
    info["issue"] = info.issue.fillna("Other / non-issue")
    t2i = dict(zip(info.topic, info.issue))

    # outliers -> nearest topic centroid if similar enough
    cent = np.load(PROC / "topic_centroids.npy")
    tids = info.topic.values
    e = embed(c)
    out = np.where(c.topic.values == -1)[0]
    sims = e[out] @ cent.T
    best = sims.argmax(1); ok = sims.max(1) >= TAU
    topic_full = c.topic.values.copy(); topic_full[out[ok]] = tids[best[ok]]
    c["issue"] = [t2i.get(x, "Unassigned") if x != -1 else "Unassigned" for x in topic_full]
    c["reassigned"] = (c.topic.values == -1) & (c.issue != "Unassigned")
    c[["doc_id", "issue"]].to_parquet(PROC / "doc_issue.parquet")

    sub = info[info.issue.isin(set(ISSUE_ANCHORS_KEYS))]
    agree = {
        "kappa_on_substantive_topics": float(cohen_kappa(sub.issue, sub.auto_issue)),
        "agreement_on_substantive_docs": float(np.average(sub.issue == sub.auto_issue, weights=sub.n)),
        "n_topics": int(len(info)),
        "outlier_share_raw": float(np.mean(c.topic.values == -1)),
        "outlier_share_after_reassign": float(np.mean(c.issue == "Unassigned")),
        "kappa_llm_vs_anchor_topics": float(cohen_kappa(info.issue, info.auto_issue)),
        "agreement_llm_vs_anchor_docs": float(np.average(info.issue == info.auto_issue, weights=info.n)),
    }

    # ---- salience ranking ----
    x = c[~c.issue.isin(NON_ISSUE | {"Unassigned"})]
    recent = x[x.date > pd.Timestamp(END) - pd.Timedelta(days=60)]
    def share(df, w=None):
        s = df.groupby("issue")[w].sum() if w else df.groupby("issue").size()
        return s / s.sum()
    R = pd.DataFrame({
        "share_news": share(x[x.channel == "news"]),
        "share_social": share(x[x.channel == "social"]),
        "share_social_engagement": share(x[x.channel == "social"], "weight"),
        "share_recent60_news": share(recent[recent.channel == "news"]),
        "share_recent60_social": share(recent[recent.channel == "social"], "weight"),
    }).fillna(0)
    xc = x[x.topic != -1]  # robustness: only documents HDBSCAN itself clustered
    R["share_clustered_only"] = share(xc[xc.channel == "news"]).add(share(xc[xc.channel == "social"], "weight"), fill_value=0) / 2
    R = R.fillna(0)
    R["salience_score"] = R[["share_news", "share_social_engagement"]].mean(1)
    R["salience_recent60"] = R[["share_recent60_news", "share_recent60_social"]].mean(1)
    R["n_docs"] = x.groupby("issue").size()
    R = R.sort_values("salience_score", ascending=False)
    R.index.name = "issue"

    # external validation: Kalshi "most important issue in the 2026 exit poll" market
    k = pd.read_csv(PROC / "kalshi_exitpoll_issue.csv")
    k["issue_tax"] = k.issue.map(KALSHI_MAP)
    v = k.dropna(subset=["issue_tax"]).set_index("issue_tax").join(R[["salience_score", "salience_recent60"]]).fillna(0)
    agree["spearman_salience_vs_kalshi"] = float(stats.spearmanr(v["last"], v.salience_score)[0])
    agree["spearman_recent60_vs_kalshi"] = float(stats.spearmanr(v["last"], v.salience_recent60)[0])
    agree["kalshi_top_issue"] = k.sort_values("last").issue.iloc[-1]
    agree["corr_news_vs_social_share"] = float(R.share_news.corr(R.share_social_engagement, method="spearman"))
    v.to_csv(RES / "kalshi_validation.csv")
    R.to_csv(RES / "issue_ranking_all.csv")
    R.head(MAX_ISSUES).to_csv(RES / "issue_ranking.csv")
    json.dump(agree, open(RES / "taxonomy_agreement.json", "w"), indent=2)
    print(R.round(3).to_string()); print(json.dumps(agree, indent=1))


if __name__ == "__main__":
    main()
