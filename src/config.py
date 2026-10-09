"""Central configuration. Every date, query and ticker used anywhere in the pipeline lives here."""
from pathlib import Path
import datetime as dt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
RES = ROOT / "results"
for p in (RAW, PROC, FIG, RES):
    p.mkdir(parents=True, exist_ok=True)

# Study window: first day the Polymarket balance-of-power book traded -> last full day before the run.
START = dt.date(2025, 7, 19)
END = dt.date(2026, 10, 8)
ELECTION_DAY = dt.date(2026, 11, 3)
# Equity beta-estimation window (strictly before the study window to avoid look-ahead in "market adjustment").
BETA_START = dt.date(2024, 7, 1)
BETA_END = dt.date(2025, 7, 18)
# Split used for in-sample basket formation vs out-of-sample evaluation.
SPLIT = dt.date(2026, 3, 1)

SEED = 7

# Retrieval queries are deliberately issue-agnostic (elections / parties / voters), so that the issue
# taxonomy is *discovered* by the topic model rather than imposed by the analyst.
SOCIAL_QUERIES = ["midterms", "election", "Congress", "Democrats", "Republicans", "voters"]
NEWS_QUERIES = ["midterms", "midterm elections", "voters", "Congress Democrats Republicans", "election poll"]
NEWS_WINDOW_DAYS = 3

# Prediction-market contracts (YES = Democratic sweep of both chambers unless noted).
POLY_EVENT_SLUG = "balance-of-power-2026-midterms"
POLY_SWEEP_Q = "2026 Balance of Power: D Senate, D House"
KALSHI = {
    "kalshi_sweep": ("KXBALANCEPOWERCOMBO", "KXBALANCEPOWERCOMBO-27FEB-DD"),
    "kalshi_house_d": ("CONTROLH", "CONTROLH-2026-D"),
    "kalshi_senate_d": ("CONTROLS", "CONTROLS-2026-D"),
}
KALSHI_ISSUE_SERIES = "KXEXITPOLLISSUE"

# Ex-ante "mid-term basket". Sign = expected sign of the stock's reaction to a HIGHER probability of a
# Democratic sweep. Each name carries a one-line policy thesis; see REPORT.md section 5.1.
BASKET = {
    # ---- expected beneficiaries of a Democratic sweep (+1) ----
    "FSLR": (+1, "Clean energy", "Utility-scale solar; IRA credit preservation / slower repeal"),
    "ENPH": (+1, "Clean energy", "Residential solar; 25D credit and net-metering politics"),
    "RUN":  (+1, "Clean energy", "Residential solar financing; most credit-sensitive name"),
    "NEE":  (+1, "Clean energy", "Largest renewable developer"),
    "PLUG": (+1, "Clean energy", "45V hydrogen credit"),
    "CNC":  (+1, "Medicaid / ACA", "Largest Medicaid + ACA-exchange insurer"),
    "MOH":  (+1, "Medicaid / ACA", "Pure-play Medicaid MCO"),
    "OSCR": (+1, "Medicaid / ACA", "ACA-exchange pure play; enhanced-subsidy sensitivity"),
    "HCA":  (+1, "Hospitals", "Uncompensated-care exposure to Medicaid/ACA coverage"),
    "THC":  (+1, "Hospitals", "Higher Medicaid payer mix"),
    # ---- expected losers of a Democratic sweep (-1) ----
    "GEO":  (-1, "Detention", "ICE detention contracts; appropriations + oversight risk"),
    "CXW":  (-1, "Detention", "ICE detention contracts"),
    "DJT":  (-1, "Political brand", "Trump Media; direct political-fortune proxy"),
    "COIN": (-1, "Crypto", "Market-structure legislation / SEC posture"),
    "HOOD": (-1, "Crypto", "Retail crypto + deregulation beneficiary"),
    "RGR":  (-1, "Firearms", "Classic partisan-regulation exposure"),
    "SWBI": (-1, "Firearms", "Classic partisan-regulation exposure"),
    "XOM":  (-1, "Fossil energy", "Federal leasing / permitting posture"),
    "BTU":  (-1, "Fossil energy", "Coal; EPA rule-making exposure"),
    "LMT":  (-1, "Defense", "Reconciliation-funded defense plus-ups at risk"),
}
MARKET = "SPY"
SECTOR_ETFS = ["XLE", "XLV", "XLF", "XLI", "XLK", "XLU", "ICLN"]
