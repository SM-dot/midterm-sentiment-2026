"""Prediction-market histories (Polymarket CLOB + Kalshi candlesticks) -> data/processed/markets.csv"""
import json, time, datetime as dt
import pandas as pd
from .config import PROC, RAW, POLY_EVENT_SLUG, KALSHI, KALSHI_ISSUE_SERIES, START
from .http import get


def polymarket():
    ev = get("https://gamma-api.polymarket.com/events", params={"slug": POLY_EVENT_SLUG}).json()[0]
    out = {}
    for m in ev["markets"]:
        tok = json.loads(m["clobTokenIds"])[0]  # YES token
        h = get("https://clob.polymarket.com/prices-history",
                params={"market": tok, "interval": "max", "fidelity": 1440}).json()["history"]
        # daily points are stamped ~00:00 UTC, i.e. they are the CLOSE of the previous UTC day -> label that day
        s = pd.Series({(pd.Timestamp(x["t"], unit="s") - pd.Timedelta(hours=1)).normalize(): float(x["p"]) for x in h})
        label = m["question"].replace("2026 Balance of Power: ", "")
        out["poly_" + label.replace(", ", "_").replace(" ", "")] = s.groupby(level=0).last()
    return pd.DataFrame(out)


def kalshi_series(series, ticker, start=START):
    end = int(time.time()); st = int(pd.Timestamp(start - dt.timedelta(30)).timestamp())
    r = get(f"https://api.elections.kalshi.com/trade-api/v2/series/{series}/markets/{ticker}/candlesticks",
            params={"start_ts": st, "end_ts": end, "period_interval": 1440}).json()
    rows = {}
    for c in r.get("candlesticks", []):
        # end-of-day MID quote (robust to bid-ask bounce in last-trade prints); last trade only as fallback
        b, a = c["yes_bid"].get("close_dollars"), c["yes_ask"].get("close_dollars")
        px = (float(b) + float(a)) / 2 if b and a and float(a) > 0 else (c["price"].get("close_dollars") or c["price"].get("previous_dollars"))
        if px is not None:
            # candle closes at end_period_ts (UTC midnight-ish ET); label by the trading day it covers
            rows[(pd.Timestamp(c["end_period_ts"], unit="s") - pd.Timedelta(hours=6)).normalize()] = float(px)
    return pd.Series(rows)


def kalshi_issue_snapshot():
    d = get("https://api.elections.kalshi.com/trade-api/v2/markets",
            params={"series_ticker": KALSHI_ISSUE_SERIES, "limit": 50}).json()["markets"]
    rows = [{"issue": m["yes_sub_title"], "ticker": m["ticker"],
             "last": float(m.get("last_price_dollars") or 0), "volume": float(m.get("volume_fp") or 0)} for m in d]
    df = pd.DataFrame(rows).sort_values("last", ascending=False)
    hist = {}
    for m in d:  # full history of the exit-poll issue market for the validation chart
        s = kalshi_series(KALSHI_ISSUE_SERIES, m["ticker"], start=dt.date(2026, 9, 1))
        hist[m["yes_sub_title"]] = s
    pd.DataFrame(hist).sort_index().to_csv(PROC / "kalshi_issue_history.csv")
    return df


def poly_hourly(question):
    ev = get("https://gamma-api.polymarket.com/events", params={"slug": POLY_EVENT_SLUG}).json()[0]
    m = [m for m in ev["markets"] if m["question"] == question][0]
    tok = json.loads(m["clobTokenIds"])[0]
    st, end, rows = int(pd.Timestamp(START).timestamp()), int(time.time()), {}
    while st < end:
        en = st + 7 * 86400
        for x in get("https://clob.polymarket.com/prices-history",
                     params={"market": tok, "startTs": st, "endTs": en, "fidelity": 60}).json()["history"]:
            if st <= x["t"] < en:  # the endpoint appends the live quote; drop it
                rows[x["t"]] = float(x["p"])
        st = en
    return pd.Series(rows).sort_index()


def kalshi_hourly(series, ticker):
    st, end, rows = int(pd.Timestamp(START - dt.timedelta(30)).timestamp()), int(time.time()), {}
    while st < end:
        en = st + 7 * 86400
        for c in get(f"https://api.elections.kalshi.com/trade-api/v2/series/{series}/markets/{ticker}/candlesticks",
                     params={"start_ts": st, "end_ts": en, "period_interval": 60}).json().get("candlesticks", []):
            b, a = c["yes_bid"].get("close_dollars"), c["yes_ask"].get("close_dollars")
            if b and a and float(a) > 0:
                rows[c["end_period_ts"]] = (float(b) + float(a)) / 2  # mid-quote: robust to stale last-trade
        st = en
    return pd.Series(rows).sort_index()


def at_close(s, hour=16):
    """Last observation at or before 16:00 America/New_York for each NYSE date (equity-aligned probability)."""
    idx = pd.to_datetime(s.index, unit="s", utc=True).tz_convert("America/New_York")
    s = pd.Series(s.values, idx)
    days = pd.date_range(s.index.min().normalize(), s.index.max().normalize(), freq="D", tz="America/New_York")
    out = {d.tz_localize(None): s[: d + pd.Timedelta(hours=hour)].iloc[-1] for d in days
           if len(s[: d + pd.Timedelta(hours=hour)])}
    return pd.Series(out)


def main():
    from .config import POLY_SWEEP_Q
    h = pd.DataFrame({"poly_sweep_1600": at_close(poly_hourly(POLY_SWEEP_Q)),
                      "kalshi_sweep_1600": at_close(kalshi_hourly(*KALSHI["kalshi_sweep"]))})
    h.index.name = "date"; h.to_csv(PROC / "markets_1600et.csv"); print(h.dropna().tail(3))
    p = polymarket()
    k = pd.DataFrame({name: kalshi_series(*v) for name, v in KALSHI.items()})
    df = p.join(k, how="outer").sort_index()
    df.index.name = "date"
    df.to_csv(PROC / "markets.csv")
    kalshi_issue_snapshot().to_csv(PROC / "kalshi_exitpoll_issue.csv", index=False)
    print(df.tail()); print(df.describe().T[["count", "min", "max"]])


if __name__ == "__main__":
    main()
