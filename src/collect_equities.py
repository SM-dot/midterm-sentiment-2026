"""Daily adjusted closes for the basket, the market and sector ETFs (yfinance) + Kenneth French factors."""
import io, zipfile
import pandas as pd, yfinance as yf
from .config import PROC, BASKET, MARKET, SECTOR_ETFS, BETA_START, END
from .http import get

FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_CSV.zip"


def prices():
    tick = list(BASKET) + [MARKET] + SECTOR_ETFS
    px = yf.download(tick, start=str(BETA_START), end=str(END + pd.Timedelta(days=1)),
                     auto_adjust=True, progress=False)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px.to_csv(PROC / "prices.csv")
    print(px.notna().sum().sort_values().head(8))


def french():
    out = []
    for name in ("F-F_Research_Data_5_Factors_2x3_daily", "F-F_Momentum_Factor_daily"):
        try:
            z = zipfile.ZipFile(io.BytesIO(get(FF_URL.format(name)).content))
            raw = z.read(z.namelist()[0]).decode("latin1").splitlines()
            hdr = next(i for i, l in enumerate(raw) if l.strip().startswith(",") or l.lower().startswith(",mkt") or "Mom" in l and l.startswith(","))
            df = pd.read_csv(io.StringIO("\n".join(raw[hdr:])), index_col=0)
            df = df[pd.to_numeric(df.index, errors="coerce").notna()]
            df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m%d")
            df.columns = [c.strip() for c in df.columns]
            out.append(df.astype(float) / 100)
        except Exception as e:  # factor data is a robustness input only
            print("French data unavailable:", e)
    if len(out) == 2:
        ff = out[0].join(out[1], how="inner")
        ff.to_csv(PROC / "ff_factors.csv"); print("FF factors through", ff.index.max().date())


if __name__ == "__main__":
    prices(); french()
