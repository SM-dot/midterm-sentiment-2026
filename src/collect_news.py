"""News headlines -> data/raw/news.jsonl

Google News RSS supports `after:`/`before:` operators, which gives a free, reproducible, *historical*
news API that aggregates thousands of outlets (AP, Reuters, Politico, Fox, CNN, local papers...).
We pull issue-agnostic queries in 3-day windows. GDELT's DOC API is used separately (gdelt_tone) as a
cross-check on tone, because its free tier only covers a rolling ~3-month window.
"""
import json, time, datetime as dt
import feedparser
from tqdm import tqdm
from .config import RAW, START, END, NEWS_QUERIES, NEWS_WINDOW_DAYS
from .http import get

URL = "https://news.google.com/rss/search"
OUT = RAW / "news.jsonl"
DONE = RAW / "news_done.txt"


def main(pause=1.2):
    done = set(DONE.read_text().splitlines()) if DONE.exists() else set()
    wins, d = [], START
    while d <= END:
        wins.append((d, min(d + dt.timedelta(NEWS_WINDOW_DAYS), END + dt.timedelta(1))))
        d += dt.timedelta(NEWS_WINDOW_DAYS)
    jobs = [(q, a, b) for q in NEWS_QUERIES for a, b in wins if f"{q}|{a}" not in done]
    with open(OUT, "a") as fo, open(DONE, "a") as fd:
        for q, a, b in tqdm(jobs, desc="news"):
            r = get(URL, params={"q": f"{q} after:{a} before:{b}", "hl": "en-US", "gl": "US", "ceid": "US:en"})
            if r.status_code == 200:
                for e in feedparser.parse(r.content).entries:
                    src = e.get("source", {}).get("title", "")
                    title = e.title
                    if src and title.endswith(" - " + src):
                        title = title[: -len(src) - 3]
                    fo.write(json.dumps({"id": e.get("id", e.link), "published": e.get("published"),
                                         "title": title, "source": src, "query": q, "window": a.isoformat()}) + "\n")
                fd.write(f"{q}|{a}\n"); fd.flush()
            time.sleep(pause)


if __name__ == "__main__":
    main()
