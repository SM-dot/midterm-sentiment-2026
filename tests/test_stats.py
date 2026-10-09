import numpy as np, pandas as pd
from src.stats_utils import bartels, runs_test, logit, bh, ccf, efficiency_battery
from src.issues import cohen_kappa


def test_randomness_tests_accept_iid():
    x = np.random.default_rng(0).standard_normal(2000)
    assert bartels(x)[1] > 0.01 and runs_test(x)[1] > 0.01


def test_randomness_tests_reject_ar1():
    rng = np.random.default_rng(1); e = rng.standard_normal(2000); x = np.zeros(2000)
    for t in range(1, 2000): x[t] = 0.6 * x[t - 1] + e[t]
    assert bartels(x)[1] < 1e-6 and runs_test(x)[1] < 1e-6
    assert efficiency_battery(pd.Series(x), "ar")["LB(5) p"] < 1e-6


def test_logit_bounds():
    assert np.isfinite(logit(np.array([0.0, 1.0]))).all()


def test_ccf_sign_convention():
    rng = np.random.default_rng(2); x = pd.Series(rng.standard_normal(500)); y = x.shift(3) + 0.1 * rng.standard_normal(500)
    assert ccf(x, y, 5).idxmax() == 3  # x leads y by 3 -> peak at positive lag


def test_bh_monotone():
    q = bh([0.01, 0.04, 0.03, 0.5]); assert (q >= np.array([0.01, 0.04, 0.03, 0.5])).all()


def test_kappa_perfect():
    assert abs(cohen_kappa(list("aabbc"), list("aabbc")) - 1) < 1e-12
