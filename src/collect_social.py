"""Bluesky (AT Protocol) public search -> data/raw/bluesky.jsonl

Design choices
* Daily windows x issue-agnostic queries, sort=top (engagement-ranked). Unauthenticated search exposes one
  page (<=100 posts) per (query, window), so temporal coverage is obtained by slicing time finely.
* Author DIDs are SHA-256 hashed at ingestion; no handles or raw identifiers are stored.
* Resumable: (query, day) pairs already on disk are skipped.
"""
import json, hashlib, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .config import RAW, START, END, SOCIAL_QUERIES
from .http import get

URL = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
OUT = RAW / "bluesky.jsonl"
DONE = RAW / "bluesky_done.txt"


def _fetch(q, day):
    since = f"{day.isoformat()}T00:00:00Z"
    until = f"{(day + dt.timedelta(days=1)).isoformat()}T00:00:00Z"
    r = get(URL, params={"q": q, "since": since, "until": until, "limit": 100, "sort": "top", "lang": "en"})
    if r.status_code != 200:
        return q, day, []
    rows = []
    for p in r.json().get("posts", []):
        rec = p.get("record", {})
        rows.append({
            "id": p["uri"], "created": rec.get("createdAt"), "text": rec.get("text", ""),
            "likes": p.get("likeCount", 0), "reposts": p.get("repostCount", 0), "replies": p.get("replyCount", 0),
            "author": hashlib.sha256(p["author"]["did"].encode()).hexdigest()[:16], "query": q, "window": day.isoformat(),
        })
    return q, day, rows


def main(workers=4):
    done = set(DONE.read_text().split()) if DONE.exists() else set()
    days = [START + dt.timedelta(d) for d in range((END - START).days + 1)]
    jobs = [(q, d) for q in SOCIAL_QUERIES for d in days if f"{q}|{d}" not in done]
    with ThreadPoolExecutor(workers) as ex, open(OUT, "a") as fo, open(DONE, "a") as fd:
        futs = [ex.submit(_fetch, q, d) for q, d in jobs]
        for f in tqdm(as_completed(futs), total=len(futs), desc="bluesky"):
            q, d, rows = f.result()
            for r in rows:
                fo.write(json.dumps(r) + "\n")
            fd.write(f"{q}|{d}\n")


if __name__ == "__main__":
    main()
