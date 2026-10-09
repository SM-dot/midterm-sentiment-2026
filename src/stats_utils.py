"""Time-series statistics used throughout: weak-form efficiency battery, stationarity, HAC regression,
block-bootstrap correlation CIs, lead-lag CCF, Granger, out-of-sample R^2 with Clark-West, BH-FDR."""
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats


def logit(p, eps=0.005):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))


# ---------- weak-form efficiency (Aloosh, Choi & Ouzan 2023 battery) ----------
def bartels(x):
    x = pd.Series(x).dropna().values; n = len(x); r = stats.rankdata(x)
    rvn = np.sum(np.diff(r) ** 2) / np.sum((r - r.mean()) ** 2)
    var = 4 * (n - 2) * (5 * n ** 2 - 2 * n - 9) / (5 * n * (n + 1) * (n - 1) ** 2)
    z = (rvn - 2) / np.sqrt(var); return rvn, 2 * stats.norm.sf(abs(z))


def runs_test(x):
    x = pd.Series(x).dropna(); x = x[x != x.median()]; s = (x > x.median()).astype(int).values
    n1, n2 = s.sum(), len(s) - s.sum(); runs = 1 + np.sum(s[1:] != s[:-1])
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z = (runs - mu) / np.sqrt(var); return z, 2 * stats.norm.sf(abs(z))


def efficiency_battery(dx, name):
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from arch.unitroot import VarianceRatio
    dx = pd.Series(dx).dropna()
    lb = acorr_ljungbox(dx, lags=[5, 10], return_df=True)
    vr = {q: VarianceRatio(dx.cumsum().values, lags=q, robust=True) for q in (2, 5, 10)}
    return {"series": name, "n": len(dx), "AC(1)": dx.autocorr(1),
            "LB(5) p": lb.lb_pvalue.iloc[0], "LB(10) p": lb.lb_pvalue.iloc[1],
            "VR(2)": vr[2].vr, "VR(2) p": vr[2].pvalue, "VR(5)": vr[5].vr, "VR(5) p": vr[5].pvalue,
            "VR(10) p": vr[10].pvalue, "Runs p": runs_test(dx)[1], "Bartels p": bartels(dx)[1]}


def stationarity(x, name):
    from statsmodels.tsa.stattools import adfuller, kpss
    import warnings
    x = pd.Series(x).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = kpss(x, regression="c", nlags="auto")[1]
    return {"series": name, "ADF p (level)": adfuller(x, autolag="AIC")[1], "KPSS p (level)": k,
            "ADF p (diff)": adfuller(x.diff().dropna(), autolag="AIC")[1]}


# ---------- correlation with inference ----------
def hac_ols(y, X, lags=7):
    df = pd.concat([y, X], axis=1).dropna()
    m = sm.OLS(df.iloc[:, 0], sm.add_constant(df.iloc[:, 1:])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return m


def boot_corr_ci(x, y, block=7, reps=2000, seed=7):
    from arch.bootstrap import StationaryBootstrap
    df = pd.concat([x, y], axis=1).dropna().values
    bs = StationaryBootstrap(block, df, seed=seed)
    r = [np.corrcoef(d[0][0][:, 0], d[0][0][:, 1])[0, 1] for d in bs.bootstrap(reps)]
    return np.percentile(r, [2.5, 97.5])


def ccf(x, y, max_lag=14):
    """corr(x_{t-L}, y_t): positive L means x LEADS y by L periods."""
    return pd.Series({L: pd.concat([x.shift(L), y], axis=1).dropna().corr().iloc[0, 1]
                      for L in range(-max_lag, max_lag + 1)})


def granger(y, x, maxlag=7):
    """p-value that x Granger-causes y, lag order by AIC in a bivariate VAR."""
    from statsmodels.tsa.api import VAR
    df = pd.concat([y, x], axis=1).dropna(); df.columns = ["y", "x"]
    p = max(1, VAR(df).select_order(maxlag).aic)
    res = VAR(df).fit(p)
    return p, res.test_causality("y", ["x"], kind="f").pvalue


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = p[o] * n / (np.arange(n) + 1); q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1); return out


def oos_r2(y, X, start, lags=(0, 1, 2), ridge=1.0):
    """Expanding-window one-step-ahead forecast of y_{t+1} from X_t..X_{t-2} vs. the prevailing mean.
    Returns (OOS R^2, Clark-West one-sided p)."""
    Z = pd.concat({f"{c}_l{l}": X[c].shift(l) for c in X for l in lags}, axis=1)
    df = pd.concat([y.shift(-1).rename("y"), Z], axis=1).dropna()
    e0, e1, f0s, f1s = [], [], [], []
    for t in range(start, len(df)):
        tr, te = df.iloc[:t], df.iloc[t]
        A = tr.drop(columns="y").values; mu_a, sd_a = A.mean(0), A.std(0) + 1e-9
        A = (A - mu_a) / sd_a; b = np.linalg.solve(A.T @ A + ridge * len(A) * np.eye(A.shape[1]), A.T @ (tr.y.values - tr.y.mean()))
        f1 = tr.y.mean() + ((te.drop("y").values - mu_a) / sd_a) @ b; f0 = tr.y.mean()
        e0.append(te.y - f0); e1.append(te.y - f1); f0s.append(f0); f1s.append(f1)
    e0, e1, f0s, f1s = map(np.array, (e0, e1, f0s, f1s))
    r2 = 1 - np.sum(e1 ** 2) / np.sum(e0 ** 2)
    adj = e0 ** 2 - (e1 ** 2 - (f0s - f1s) ** 2)
    t = adj.mean() / (adj.std(ddof=1) / np.sqrt(len(adj)))
    return r2, stats.norm.sf(t)
