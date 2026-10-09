"""End-to-end pipeline.

    python run_all.py                    # everything, re-using the saved topic assignment
    python run_all.py --skip-collect     # analysis only, on data already in data/
    python run_all.py --from market_analysis
    python run_all.py --refit-topics     # refit BERTopic (topic ids change -> re-annotate results/topic_issue_map.csv)

The topic->issue map (results/topic_issue_map.csv) is the one LLM-in-the-loop artefact. It is keyed on the
topic ids in data/processed/doc_topics.parquet, so the default run never refits the topic model.
"""
import argparse, importlib, time, sys
from src.config import PROC

STEPS = ["collect_markets", "collect_equities", "collect_social", "collect_news", "preprocess", "topics", "issues",
         "sentiment", "index", "market_analysis", "equities", "robustness", "summary", "figures", "report_tables"]
COLLECT = {"collect_markets", "collect_equities", "collect_social", "collect_news"}


def run(step, refit):
    m = importlib.import_module(f"src.{step}")
    if step == "collect_equities":
        m.prices(); m.french()
    elif step == "topics":
        if refit or not (PROC / "doc_topics.parquet").exists():
            m.main(); print("!! topics refit: re-annotate results/topic_issue_map.csv before continuing"); sys.exit(0)
        m.run_baselines(); m.bertopic_stability()
    elif step == "index":
        m.main(); m.main(score="vader", tag="_vader")
    else:
        m.main()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-collect", action="store_true"); ap.add_argument("--from", dest="frm", choices=STEPS)
    ap.add_argument("--refit-topics", action="store_true")
    a = ap.parse_args(); started = a.frm is None
    for s in STEPS:
        started = started or s == a.frm
        if not started or (a.skip_collect and s in COLLECT):
            continue
        t = time.time(); run(s, a.refit_topics); print(f"[{s}] done in {time.time() - t:.0f}s", flush=True)
