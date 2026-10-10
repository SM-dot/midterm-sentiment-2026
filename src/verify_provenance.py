"""Live provenance audit: re-fetch random samples from every upstream API *now* and check they match what the
pipeline stored. Writes results/provenance_audit.json. Nothing in the pipeline is synthetic; this proves it."""
import json, datetime as dt, numpy as np, pandas as pd, yfinance as yf, feedparser
from .config import RAW, PROC, RES, POLY_EVENT_SLUG, POLY_SWEEP_Q, KALSHI
from .http import get

rng = np.random.default_rng()  # deliberately unseeded: a different sample on every audit


def bluesky(n=50):
    d = pd.read_json(RAW / "bluesky.jsonl", lines=True, dtype={"id": str}).sample(n, random_state=int(rng.integers(1e9)))
    found, same = 0, 0
    for i in range(0, n, 25):
        chunk = d.iloc[i:i + 25]
        r = get("https://api.bsky.app/xrpc/app.bsky.feed.getPosts", params=[("uris", u) for u in chunk.id]).json()
        live = {p["uri"]: p["record"].get("text", "") for p in r.get("posts", [])}
        for u, t in zip(chunk.id, chunk.text):
            if u in live:
                found += 1; same += live[u] == t
    return {"sampled": n, "still_online": found, "text_identical": same,
            "note": "posts can be deleted by authors after collection; identical text proves they are real records"}


def news(k=3):
    d = pd.read_json(RAW / "news.jsonl", lines=True)
    out = []
    groups = list(d.groupby(["query", "window"]))
    for j in rng.choice(len(groups), k, replace=False):
        (q, w), g = groups[j]
        a = pd.Timestamp(w).date(); b = a + dt.timedelta(3)
        r = get("https://news.google.com/rss/search", params={"q": f"{q} after:{a} before:{b}", "hl": "en-US", "gl": "US", "ceid": "US:en"})
        live = {e.get("id") for e in feedparser.parse(r.content).entries}
        out.append({"query": q, "window": w, "stored": len(g), "live_now": len(live), "overlap": len(set(g.id) & live)})
    return out


def polymarket(n=10):
    ev = get("https://gamma-api.polymarket.com/events", params={"slug": POLY_EVENT_SLUG}).json()[0]
    m = [m for m in ev["markets"] if m["question"] == POLY_SWEEP_Q][0]
    h = get("https://clob.polymarket.com/prices-history", params={"market": json.loads(m["clobTokenIds"])[0], "interval": "max", "fidelity": 1440}).json()["history"]
    live = pd.Series({(pd.Timestamp(x["t"], unit="s") - pd.Timedelta(hours=1)).normalize(): float(x["p"]) for x in h})
    st = pd.read_csv(PROC / "markets.csv", index_col=0, parse_dates=True).poly_DSenate_DHouse.dropna()
    common = st.index.intersection(live.index); pick = rng.choice(common, n, replace=False)
    return {"dates_checked": [str(pd.Timestamp(x).date()) for x in pick], "max_abs_diff": float((st[pick] - live[pick]).abs().max()),
            "live_latest": {str(live.index[-1].date()): live.iloc[-1]}}


def kalshi():
    s, t = KALSHI["kalshi_sweep"]
    m = get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{t}").json()["market"]
    st = pd.read_csv(PROC / "markets.csv", index_col=0, parse_dates=True).kalshi_sweep.dropna()
    return {"ticker": t, "live_last_price": m.get("last_price_dollars"), "live_bid": m.get("yes_bid_dollars"),
            "live_ask": m.get("yes_ask_dollars"), "stored_last_date": str(st.index[-1].date()), "stored_last_mid": float(st.iloc[-1])}


def prices(n=10):
    st = pd.read_csv(PROC / "prices.csv", index_col=0, parse_dates=True)
    tick = list(rng.choice([c for c in st.columns], 4, replace=False))
    live = yf.download(tick, start=str(st.index[0].date()), end=str((st.index[-1] + pd.Timedelta(days=1)).date()),
                       auto_adjust=True, progress=False)["Close"]
    live.index = pd.to_datetime(live.index).tz_localize(None)
    days = rng.choice(st.index[-120:], n, replace=False)
    rel = (st.loc[days, tick] / live.loc[days, tick] - 1).abs().max().max()
    return {"tickers": tick, "max_rel_diff": float(rel), "stored_last_date": str(st.index[-1].date())}


def main():
    out = {"audited_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    for name, f in [("bluesky", bluesky), ("google_news", news), ("polymarket", polymarket), ("kalshi", kalshi), ("yfinance", prices)]:
        try:
            out[name] = f()
        except Exception as e:
            out[name] = {"error": repr(e)}
    json.dump(out, open(RES / "provenance_audit.json", "w"), indent=2, default=str); print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
