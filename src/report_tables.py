"""Render every table quoted in REPORT.md from results/ (so no number is hand-transcribed)."""
import json, pandas as pd
from .config import RES

def md(df, fmt=".3f"): return df.to_markdown(index=False, floatfmt=fmt)

def main():
    out = []
    t = pd.read_csv(RES / "topic_model_comparison.csv"); out += ["### topic_models", md(t)]
    out += ["### bertopic_stability", json.dumps(json.load(open(RES / "bertopic_stability.json")))]
    r = pd.read_csv(RES / "issue_ranking_all.csv")[["issue", "share_news", "share_social_engagement", "salience_score", "salience_recent60", "share_clustered_only", "n_docs"]]
    for c in r.columns[1:-1]: r[c] = (100 * r[c]).round(1)
    out += ["### issue_ranking", md(r, ".1f")]
    out += ["### taxonomy", json.dumps(json.load(open(RES / "taxonomy_agreement.json")), indent=1)]
    out += ["### kalshi_validation", md(pd.read_csv(RES / "kalshi_validation.csv"))]
    e = pd.read_csv(RES / "market_efficiency.csv"); out += ["### efficiency", md(e)]
    out += ["### market_quality", json.dumps(json.load(open(RES / "market_quality.json")), indent=1)]
    s = pd.read_csv(RES / "stationarity.csv"); out += ["### stationarity", md(s)]
    c = pd.read_csv(RES / "sentiment_market_corr.csv")[["index", "level_corr", "daily_dcorr", "ci_lo", "ci_hi", "t_hac", "p_hac", "q_bh", "weekly_dcorr", "granger_p_sent_to_mkt", "q_bh_granger", "granger_p_mkt_to_sent"]]
    out += ["### sent_market", md(c)]
    out += ["### by_contract", pd.read_csv(RES / "sentiment_corr_by_contract.csv").to_markdown(index=False, floatfmt=".3f")]
    out += ["### oos", md(pd.read_csv(RES / "oos_predictability.csv"), ".4f")]
    out += ["### robustness", pd.read_csv(RES / "robustness.csv").to_markdown(index=False, floatfmt=".3f")]
    p = pd.read_csv(RES / "political_betas.csv")[["ticker", "group", "sign", "beta_mkt", "g_total_full", "t_total_full", "t_total_is", "t_total_oos", "car_window"]]
    out += ["### political_betas", md(p)]
    b = pd.read_csv(RES / "basket_political_beta.csv")[["basket", "sample", "corr_dL", "g_total", "t_total", "ann_return", "ann_vol", "sharpe", "max_dd", "t_mean", "n"]]
    out += ["### basket", md(b)]
    out += ["### event", md(pd.read_csv(RES / "event_study.csv"))]
    out += ["### sent_basket", md(pd.read_csv(RES / "sentiment_basket_corr.csv"))]
    out += ["### extras", json.dumps(json.load(open(RES / "basket_extras.json")), indent=1)]
    out += ["### summary", json.dumps(json.load(open(RES / "summary_numbers.json")), indent=1)]
    out += ["### composition", json.dumps(json.load(open(RES / "basket_composition.json")))]
    out += ["### msi_weights", pd.read_csv(RES / "msi_weights.csv").to_markdown(index=False, floatfmt=".3f")]
    (RES / "tables.md").write_text("\n\n".join(out)); print("wrote results/tables.md")

if __name__ == "__main__":
    main()
