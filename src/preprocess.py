"""Raw JSONL -> one tidy corpus (data/processed/corpus.parquet).

Columns: doc_id, date, channel {news, social}, text, weight, source
* social weight = 1 + log1p(likes + reposts): engagement is revealed attention, log-damped so that a single
  viral post cannot dominate a day.
* exact and near-exact duplicates (normalised text) are removed per channel; this matters because the same
  post is often returned by several retrieval queries, and wire stories are syndicated across outlets.
"""
import re, numpy as np, pandas as pd
from .config import RAW, PROC, START, END

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@[\w\.\-]+")
WS = re.compile(r"\s+")


def clean(t):
    t = URL.sub(" ", t or "")
    t = MENTION.sub("@user", t)
    return WS.sub(" ", t).strip()


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower())[:200]


def load_social():
    d = pd.read_json(RAW / "bluesky.jsonl", lines=True, dtype={"id": str})
    d = d.drop_duplicates("id")
    d["text"] = d["text"].map(clean)
    d = d[d.text.str.len() >= 40]
    d["date"] = pd.to_datetime(d["window"])
    d["weight"] = 1 + np.log1p(d.likes.fillna(0) + d.reposts.fillna(0))
    d["k"] = d.text.map(norm)
    d = d.sort_values("likes", ascending=False).drop_duplicates("k")
    return pd.DataFrame({"doc_id": d.id, "date": d.date, "channel": "social", "text": d.text,
                         "weight": d.weight, "source": "bluesky", "likes": d.likes, "query": d["query"]})


def load_news():
    d = pd.read_json(RAW / "news.jsonl", lines=True, dtype={"id": str})
    d["date"] = pd.to_datetime(d.published, utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    d["text"] = d.title.map(clean)
    d = d[(d.text.str.len() >= 25) & d.date.notna()]
    d["k"] = d.text.map(norm)
    d = d.sort_values("date").drop_duplicates("k")
    return pd.DataFrame({"doc_id": d.id, "date": d.date, "channel": "news", "text": d.text,
                         "weight": 1.0, "source": d.source, "likes": np.nan, "query": d["query"]})


def main():
    c = pd.concat([load_social(), load_news()], ignore_index=True)
    c = c[(c.date >= pd.Timestamp(START)) & (c.date <= pd.Timestamp(END))].reset_index(drop=True)
    c.to_parquet(PROC / "corpus.parquet")
    print(c.groupby("channel").agg(n=("text", "size"), days=("date", "nunique"),
                                   first=("date", "min"), last=("date", "max")))
    return c


if __name__ == "__main__":
    main()
