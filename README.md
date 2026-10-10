# midterm-sentiment-2026

**Topic modeling → issue sentiment → event-contract probabilities → a mid-term equity basket, for the 2026 US midterms.**

📄 **Full write-up: [REPORT.md](REPORT.md)**: design, results, conclusions and limitations.

## The three questions and the short answers

| Question | Answer |
|---|---|
| What issues do voters care about? | **Economy & cost of living** (the Kalshi exit-poll market prices it at 92% to be the top issue). The *discourse*, however, is dominated by democracy / rule of law (33%), redistricting (20%), Epstein / corruption (11%) and the Iran war (8%). The salience ranking matches the market's ordering (ρ = 0.85) but badly understates the economy's level. |
| Does issue sentiment track P(Democratic sweep) on Kalshi / Polymarket? | **Only contemporaneously and weakly** (Δ-correlation ≈ 0.10 daily, 0.20 weekly; nothing survives FDR correction), mostly through the **House** contract. There is no lead and no out-of-sample predictability (OOS R² ≈ 0.4%). The sign is positive, which rejects the referendum hypothesis. |
| Do mid-term-exposed equities co-move with sentiment or the sweep odds? | **No detectable relationship in this cycle.** The thesis long/short basket worked in-sample (t = 1.6), reversed out-of-sample (t = −0.7), and has no factor-adjusted political loading or sentiment link. The test can only rule out exposures above about 19% cumulative abnormal return. |

## Repository layout

```
run_all.py                 end-to-end pipeline (resumable, step-wise)
src/
  config.py                every date, query, contract and ticker (incl. the ex-ante basket + theses)
  http.py                  shared HTTP client with exponential back-off
  collect_social.py        Bluesky public search, daily windows, hashed authors
  collect_news.py          Google News RSS with after:/before: windows
  collect_markets.py       Polymarket CLOB + Kalshi candles; daily (re-labelled) and 16:00-ET-aligned hourly
  collect_equities.py      yfinance prices + Kenneth French FF5/momentum factors
  preprocess.py            cleaning, de-duplication, engagement weights
  topics.py                BERTopic (MiniLM-UMAP-HDBSCAN-cTF-IDF), LDA/NMF baselines, coherence, seed stability
  issues.py                micro-topic → issue taxonomy, outlier re-assignment, salience ranking, Kalshi validation
  sentiment.py             twitter-RoBERTa sentiment + VADER robustness
  index.py                 EB-shrunk, channel-standardised issue indices, MSI composite, salience indices
  stats_utils.py           efficiency battery, ADF/KPSS, HAC, stationary bootstrap, CCF, Granger, OOS R² + Clark-West, BH-FDR
  market_analysis.py       sentiment ↔ prediction-market tests
  equities.py              market-adjusted returns, sweep betas, baskets (IS/OOS), event study, FF6, overlay appraisal
  robustness.py            VADER / Kalshi / channel / half-life / outlier-day robustness grid
  summary.py               headline numbers + power analysis
  figures.py, viz.py       all figures (fixed colour slots, no dual axes)
  report_tables.py         renders results/tables.md: every number quoted in REPORT.md
results/                   all tables (CSV/JSON); topic_issue_map.csv is the annotated taxonomy
figures/                   11 figures used in the report
tests/                     unit tests for the statistical utilities
```

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py                     # ~1.5 h on an Apple-silicon laptop (collection ~25 min, embeddings + sentiment ~45 min on MPS)
python run_all.py --skip-collect      # analysis only
pytest -q tests
```

* **Topic ids are frozen.** The default run re-uses `data/processed/doc_topics.parquet`, because `results/topic_issue_map.csv` (the LLM-annotated taxonomy) is keyed on those ids. `--refit-topics` refits BERTopic and stops so the map can be re-annotated.
* Live APIs keep moving: re-collecting after 8 Oct 2026 will extend the market series (the analysis window is fixed in `config.py`).
* For CPU-only machines, change `device="mps"` in `topics.py` / `sentiment.py` to `"cpu"` (or `"cuda"`).

## Data provenance: everything is live API data, nothing is simulated

Every input is pulled from a public API at run time; there is **no mock, synthetic or hand-entered data** anywhere
in the pipeline. (The only use of randomness is retry jitter in `src/http.py` and the unit tests' own test series.)

| Source | Endpoint |
|---|---|
| Bluesky | `https://api.bsky.app/xrpc/app.bsky.feed.searchPosts` |
| News | `https://news.google.com/rss/search` (`after:` / `before:` windows) |
| Polymarket | `https://gamma-api.polymarket.com/events`, `https://clob.polymarket.com/prices-history` |
| Kalshi | `https://api.elections.kalshi.com/trade-api/v2/.../candlesticks` |
| Equities | Yahoo Finance via `yfinance`; Fama–French factors from Kenneth French's data library |

* `config.END` is **dynamic**: the last complete day in New York time, so every run extends the data to yesterday
  (the collectors are incremental). Set `MIDTERM_END=2026-10-08` to replicate REPORT.md exactly.
* **Live audit:** `python -m src.verify_provenance` re-fetches a fresh random sample from every API and compares it
  with what was stored. The result is in [`results/provenance_audit.json`](results/provenance_audit.json). Latest run: 50/50 Bluesky posts
  still online with identical text, 158/159 Google News headlines returned again live, Polymarket and yfinance
  values identical (max difference 0), and Kalshi's live quote consistent with the stored series.

## Data ethics

Only public posts and headlines are used. Author identifiers are SHA-256 hashed at ingestion, and raw text and derived embeddings are git-ignored rather than redistributed. Prediction-market and price data come from public endpoints.
