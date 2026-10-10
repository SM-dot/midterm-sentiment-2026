### topic_models

| model    |   K |   c_npmi |   c_v |   diversity |   stability_jaccard |   stability_sd |
|:---------|----:|---------:|------:|------------:|--------------------:|---------------:|
| LDA      |  10 |    0.018 | 0.432 |       0.650 |               0.279 |          0.023 |
| NMF      |  10 |    0.099 | 0.587 |       0.910 |               0.818 |          0.105 |
| LDA      |  20 |    0.006 | 0.418 |       0.665 |               0.253 |          0.004 |
| NMF      |  20 |    0.076 | 0.544 |       0.805 |               0.756 |          0.066 |
| LDA      |  30 |   -0.005 | 0.415 |       0.667 |               0.229 |          0.005 |
| NMF      |  30 |    0.048 | 0.508 |       0.783 |               0.710 |          0.037 |
| BERTopic |  30 |    0.053 | 0.569 |       0.989 |             nan     |        nan     |

### bertopic_stability

{"ari_mean": 0.9211738502030291, "jaccard_mean": 0.6671278081096308, "jaccard_sd": 0.006585554619150447, "n_topics": [101, 106, 98]}

### issue_ranking

| issue                          |   share_news |   share_social_engagement |   salience_score |   salience_recent60 |   share_clustered_only |   n_docs |
|:-------------------------------|-------------:|--------------------------:|-----------------:|--------------------:|-----------------------:|---------:|
| Democracy & rule of law        |         30.5 |                      36.2 |             33.4 |                39.8 |                   27.9 |    43153 |
| Redistricting & gerrymandering |         30.4 |                       9.1 |             19.7 |                12.0 |                   20.6 |    13634 |
| Epstein files & corruption     |          6.1 |                      15.0 |             10.5 |                11.3 |                   11.3 |    15647 |
| Foreign policy & Iran war      |          7.8 |                       8.6 |              8.2 |                 9.0 |                    9.1 |     9839 |
| Economy & cost of living       |          5.5 |                       4.3 |              4.9 |                 7.5 |                    4.6 |     5356 |
| Health care                    |          3.8 |                       5.3 |              4.5 |                 2.8 |                    5.6 |     5740 |
| Immigration & ICE              |          3.8 |                       5.1 |              4.5 |                 2.4 |                    5.5 |     5507 |
| Government shutdown & budget   |          4.7 |                       2.2 |              3.4 |                 0.5 |                    4.1 |     2754 |
| Civil rights & social issues   |          1.2 |                       4.2 |              2.7 |                 4.6 |                    3.1 |     4573 |
| Guns & political violence      |          0.7 |                       2.7 |              1.7 |                 0.8 |                    1.4 |     2787 |
| Tech, AI & data centers        |          2.0 |                       1.0 |              1.5 |                 3.8 |                    1.8 |     1338 |
| Climate & environment          |          0.9 |                       1.7 |              1.3 |                 2.2 |                    1.5 |     1844 |
| Social Security & safety net   |          0.5 |                       1.6 |              1.1 |                 1.0 |                    1.1 |     1683 |
| Tariffs & trade                |          0.9 |                       1.0 |              0.9 |                 1.1 |                    1.1 |     1189 |
| Abortion & reproductive rights |          0.6 |                       0.7 |              0.7 |                 0.7 |                    0.8 |      773 |
| Education & science            |          0.4 |                       0.8 |              0.6 |                 0.5 |                    0.5 |      880 |
| Crime & public safety          |          0.2 |                       0.5 |              0.3 |                 0.2 |                    0.3 |      494 |

### taxonomy

{
 "kappa_on_substantive_topics": 0.5850974638440578,
 "agreement_on_substantive_docs": 0.7593779996160491,
 "n_topics": 260,
 "outlier_share_raw": 0.36003194219393253,
 "outlier_share_after_reassign": 0.04971197754742873,
 "kappa_llm_vs_anchor_topics": 0.24017114717042057,
 "agreement_llm_vs_anchor_docs": 0.3730752922648252,
 "spearman_salience_vs_kalshi": 0.8524367141602663,
 "spearman_recent60_vs_kalshi": 0.8524367141602663,
 "kalshi_top_issue": "Economy and cost of living",
 "corr_news_vs_social_share": 0.8823529411764707
}

### kalshi_validation

| issue_tax                      | issue                       | ticker                        |   last |    volume |   salience_score |   salience_recent60 |
|:-------------------------------|:----------------------------|:------------------------------|-------:|----------:|-----------------:|--------------------:|
| Economy & cost of living       | Economy and cost of living  | KXEXITPOLLISSUE-26NOV06-ECON  |  0.920 |  7917.780 |            0.049 |               0.075 |
| Democracy & rule of law        | Democracy                   | KXEXITPOLLISSUE-26NOV06-DEMOC |  0.130 |  7285.060 |            0.334 |               0.398 |
| Foreign policy & Iran war      | Iran war and foreign policy | KXEXITPOLLISSUE-26NOV06-FPOL  |  0.090 | 13036.670 |            0.082 |               0.090 |
| Health care                    | Health care                 | KXEXITPOLLISSUE-26NOV06-HLTH  |  0.020 |  4216.920 |            0.045 |               0.028 |
| Immigration & ICE              | Immigration                 | KXEXITPOLLISSUE-26NOV06-IMM   |  0.010 |  4512.460 |            0.045 |               0.024 |
| Crime & public safety          | Crime                       | KXEXITPOLLISSUE-26NOV06-CRIM  |  0.010 |  5658.830 |            0.003 |               0.002 |
| Abortion & reproductive rights | Abortion                    | KXEXITPOLLISSUE-26NOV06-ABOR  |  0.010 |  5659.290 |            0.007 |               0.007 |

### efficiency

| series                     |   n |   AC(1) |   LB(5) p |   LB(10) p |   VR(2) |   VR(2) p |   VR(5) |   VR(5) p |   VR(10) p |   Runs p |   Bartels p |
|:---------------------------|----:|--------:|----------:|-----------:|--------:|----------:|--------:|----------:|-----------:|---------:|------------:|
| Polymarket D sweep         | 430 |  -0.004 |     0.027 |      0.103 |   0.998 |     0.976 |   0.847 |     0.299 |      0.349 |    0.504 |       0.671 |
| Kalshi D sweep             | 301 |   0.123 |     0.187 |      0.334 |   1.138 |     0.059 |   1.261 |     0.089 |      0.057 |    0.118 |       0.047 |
| Kalshi D House             | 446 |   0.048 |     0.115 |      0.129 |   1.051 |     0.473 |   1.155 |     0.275 |      0.037 |    0.306 |       0.437 |
| Kalshi D Senate            | 446 |   0.072 |     0.556 |      0.293 |   1.074 |     0.297 |   1.194 |     0.184 |      0.235 |    0.007 |       0.171 |
| Polymarket sweep, 1st half | 223 |  -0.088 |     0.002 |      0.017 |   0.917 |     0.424 |   0.673 |     0.102 |      0.119 |    0.232 |       0.157 |
| Polymarket sweep, 2nd half | 208 |   0.127 |     0.583 |      0.608 |   1.134 |     0.186 |   1.157 |     0.459 |      0.345 |    0.675 |       0.384 |

### market_quality

{
 "corr_daily_dlogit_poly_kalshi": 0.38799007973966143,
 "corr_weekly_dlogit_poly_kalshi": 0.6426934799545446,
 "corr_level_poly_kalshi": 0.9788549964137372,
 "mean_abs_gap_pp": 2.3742214532871975,
 "implied_from_marginals_corr": 0.9852375184258584
}

### stationarity

| series                         |   ADF p (level) |   KPSS p (level) |   ADF p (diff) |
|:-------------------------------|----------------:|-----------------:|---------------:|
| logit p (Polymarket sweep)     |           0.948 |            0.010 |          0.000 |
| Democracy & rule of law        |           0.001 |            0.100 |          0.000 |
| Economy & cost of living       |           0.002 |            0.100 |          0.000 |
| Epstein files & corruption     |           0.000 |            0.010 |          0.000 |
| Foreign policy & Iran war      |           0.000 |            0.010 |          0.000 |
| Government shutdown & budget   |           0.000 |            0.010 |          0.000 |
| Health care                    |           0.000 |            0.010 |          0.000 |
| Immigration & ICE              |           0.000 |            0.100 |          0.000 |
| Redistricting & gerrymandering |           0.008 |            0.059 |          0.000 |
| MSI                            |           0.018 |            0.010 |          0.000 |
| Overall tone                   |           0.008 |            0.045 |          0.000 |

### sent_market

| index                          |   level_corr |   daily_dcorr |   ci_lo |   ci_hi |   t_hac |   p_hac |   q_bh |   weekly_dcorr |   granger_p_sent_to_mkt |   q_bh_granger |   granger_p_mkt_to_sent |
|:-------------------------------|-------------:|--------------:|--------:|--------:|--------:|--------:|-------:|---------------:|------------------------:|---------------:|------------------------:|
| Democracy & rule of law        |        0.072 |         0.056 |  -0.069 |   0.181 |   0.755 |   0.450 |  0.582 |          0.128 |                   0.313 |          0.833 |                   0.969 |
| Economy & cost of living       |        0.022 |         0.012 |  -0.075 |   0.088 |   0.292 |   0.770 |  0.770 |         -0.189 |                   0.762 |          0.872 |                   0.553 |
| Epstein files & corruption     |        0.643 |         0.068 |  -0.031 |   0.154 |   1.293 |   0.196 |  0.392 |          0.041 |                   0.522 |          0.833 |                   0.523 |
| Foreign policy & Iran war      |        0.333 |         0.014 |  -0.066 |   0.099 |   0.340 |   0.734 |  0.770 |         -0.085 |                   0.785 |          0.872 |                   0.811 |
| Government shutdown & budget   |        0.368 |        -0.059 |  -0.149 |   0.047 |  -1.121 |   0.262 |  0.437 |          0.105 |                   0.563 |          0.833 |                   0.893 |
| Health care                    |        0.648 |         0.096 |   0.003 |   0.176 |   2.020 |   0.043 |  0.390 |          0.210 |                   0.914 |          0.914 |                   0.043 |
| Immigration & ICE              |       -0.087 |         0.044 |  -0.019 |   0.106 |   1.386 |   0.166 |  0.392 |         -0.043 |                   0.007 |          0.075 |                   0.138 |
| Redistricting & gerrymandering |        0.244 |         0.042 |  -0.071 |   0.145 |   0.729 |   0.466 |  0.582 |          0.205 |                   0.094 |          0.469 |                   0.435 |
| MSI                            |        0.459 |         0.100 |  -0.022 |   0.209 |   1.375 |   0.169 |  0.392 |          0.196 |                   0.583 |          0.833 |                   0.603 |
| Overall tone                   |        0.201 |         0.112 |   0.005 |   0.211 |   1.762 |   0.078 |  0.390 |          0.039 |                   0.460 |          0.833 |                   0.327 |

### by_contract

| Unnamed: 0                     |   Polymarket D sweep |   Kalshi D sweep |   Kalshi D House |   Kalshi D Senate |
|:-------------------------------|---------------------:|-----------------:|-----------------:|------------------:|
| Democracy & rule of law        |                0.056 |            0.038 |            0.102 |             0.020 |
| Economy & cost of living       |                0.012 |            0.054 |            0.039 |             0.019 |
| Epstein files & corruption     |                0.068 |            0.084 |            0.057 |             0.044 |
| Foreign policy & Iran war      |                0.014 |            0.040 |           -0.005 |             0.058 |
| Government shutdown & budget   |               -0.059 |           -0.058 |           -0.022 |            -0.044 |
| Health care                    |                0.096 |           -0.045 |            0.049 |            -0.046 |
| Immigration & ICE              |                0.044 |            0.032 |           -0.012 |             0.044 |
| Redistricting & gerrymandering |                0.042 |           -0.009 |            0.148 |            -0.054 |
| MSI                            |                0.100 |            0.064 |            0.141 |             0.027 |
| Overall tone                   |                0.112 |            0.071 |            0.128 |             0.029 |

### oos

| predictors                 |   oos_r2 |   clark_west_p |
|:---------------------------|---------:|---------------:|
| MSI                        |   0.0041 |         0.0954 |
| Overall tone               |   0.0051 |         0.0939 |
| Top-4 issues               |  -0.0240 |         0.5464 |
| All issues                 |  -0.0375 |         0.4596 |
| AR(3) own lags (benchmark) |   0.0025 |         0.2230 |

### robustness

| index                          |   baseline (RoBERTa, unsmoothed, Polymarket) |   VADER sentiment |   Kalshi sweep target |   news channel only |   social channel only |   excl. 10 largest market moves |   level corr RoBERTa vs VADER index |   half-life 1d |   half-life 7d |
|:-------------------------------|---------------------------------------------:|------------------:|----------------------:|--------------------:|----------------------:|--------------------------------:|------------------------------------:|---------------:|---------------:|
| Democracy & rule of law        |                                        0.056 |             0.036 |                 0.038 |               0.024 |                 0.071 |                           0.035 |                               0.505 |          0.066 |          0.055 |
| Economy & cost of living       |                                        0.012 |             0.017 |                 0.054 |               0.001 |                 0.026 |                           0.022 |                               0.495 |          0.024 |          0.024 |
| Epstein files & corruption     |                                        0.068 |             0.027 |                 0.084 |               0.028 |                 0.112 |                           0.030 |                               0.492 |          0.109 |          0.082 |
| Foreign policy & Iran war      |                                        0.014 |             0.068 |                 0.040 |               0.034 |                -0.007 |                          -0.010 |                               0.070 |          0.007 |          0.023 |
| Government shutdown & budget   |                                       -0.059 |            -0.126 |                -0.058 |              -0.040 |                 0.005 |                          -0.066 |                               0.285 |         -0.035 |         -0.008 |
| Health care                    |                                        0.096 |             0.105 |                -0.045 |               0.051 |                 0.133 |                           0.075 |                               0.437 |          0.123 |          0.101 |
| Immigration & ICE              |                                        0.044 |             0.053 |                 0.032 |               0.009 |                -0.013 |                           0.037 |                               0.507 |          0.009 |         -0.026 |
| Redistricting & gerrymandering |                                        0.042 |             0.027 |                -0.009 |               0.006 |                 0.133 |                           0.110 |                               0.598 |          0.075 |          0.090 |
| MSI                            |                                        0.100 |             0.072 |                 0.064 |             nan     |               nan     |                           0.077 |                               0.541 |        nan     |        nan     |
| Overall tone                   |                                        0.112 |             0.088 |                 0.071 |             nan     |               nan     |                           0.097 |                               0.610 |        nan     |        nan     |

### political_betas

| ticker   | group           |   sign |   beta_mkt |   g_total_full |   t_total_full |   t_total_is |   t_total_oos |   car_window |
|:---------|:----------------|-------:|-----------:|---------------:|---------------:|-------------:|--------------:|-------------:|
| FSLR     | Clean energy    |      1 |      0.988 |          0.016 |          0.286 |        1.113 |        -0.796 |       -0.174 |
| ENPH     | Clean energy    |      1 |      0.941 |          0.063 |          0.808 |        1.210 |         0.110 |       -0.309 |
| RUN      | Clean energy    |      1 |      1.173 |          0.024 |          0.365 |        0.308 |         0.282 |       -0.410 |
| NEE      | Clean energy    |      1 |      0.544 |         -0.044 |         -2.296 |       -1.648 |        -1.626 |       -0.071 |
| PLUG     | Clean energy    |      1 |      1.282 |         -0.015 |         -0.206 |       -0.015 |        -0.259 |       -0.251 |
| CNC      | Medicaid / ACA  |      1 |      0.396 |         -0.007 |         -0.136 |        2.693 |        -2.266 |        1.110 |
| MOH      | Medicaid / ACA  |      1 |      0.423 |         -0.012 |         -0.292 |        0.951 |        -1.371 |       -0.033 |
| OSCR     | Medicaid / ACA  |      1 |      0.906 |         -0.035 |         -0.501 |        0.901 |        -2.135 |        1.032 |
| HCA      | Hospitals       |      1 |      0.527 |          0.018 |          0.791 |        0.943 |         0.024 |        0.099 |
| THC      | Hospitals       |      1 |      0.920 |          0.002 |          0.074 |        0.752 |        -0.935 |        0.203 |
| GEO      | Detention       |     -1 |      1.203 |         -0.041 |         -0.929 |        0.340 |        -1.706 |       -0.043 |
| CXW      | Detention       |     -1 |      0.909 |         -0.074 |         -1.671 |       -1.632 |        -0.832 |        0.270 |
| DJT      | Political brand |     -1 |      1.374 |         -0.011 |         -0.214 |        0.555 |        -1.089 |       -0.669 |
| COIN     | Crypto          |     -1 |      1.868 |         -0.042 |         -0.922 |        0.035 |        -1.333 |       -0.715 |
| HOOD     | Crypto          |     -1 |      2.063 |         -0.054 |         -0.975 |       -0.451 |        -1.011 |       -0.348 |
| RGR      | Firearms        |     -1 |      0.507 |         -0.009 |         -0.185 |       -0.568 |         0.874 |        0.138 |
| SWBI     | Firearms        |     -1 |      0.689 |         -0.022 |         -0.349 |       -1.159 |         1.149 |        0.609 |
| XOM      | Fossil energy   |     -1 |      0.659 |          0.006 |          0.258 |        0.974 |        -0.562 |        0.379 |
| BTU      | Fossil energy   |     -1 |      0.934 |          0.026 |          0.566 |        0.010 |         0.980 |        0.313 |
| LMT      | Defense         |     -1 |      0.434 |         -0.007 |         -0.317 |       -0.690 |         0.301 |        0.021 |

### basket

| basket         | sample        |   corr_dL |   g_total |   t_total |   ann_return |   ann_vol |   sharpe |   max_dd |   t_mean |   n |
|:---------------|:--------------|----------:|----------:|----------:|-------------:|----------:|---------:|---------:|---------:|----:|
| LS             | full          |     0.051 |     0.024 |     0.672 |        0.163 |     0.294 |    0.662 |   -0.296 |    0.731 | 308 |
| LS             | in-sample     |     0.103 |     0.065 |     1.614 |        0.546 |     0.317 |    1.531 |   -0.120 |    1.197 | 154 |
| LS             | out-of-sample |    -0.032 |    -0.030 |    -0.730 |       -0.125 |     0.268 |   -0.364 |   -0.296 |   -0.285 | 154 |
| long           | full          |     0.095 |     0.001 |     0.040 |        0.175 |     0.258 |    0.753 |   -0.171 |    0.833 | 308 |
| long           | in-sample     |     0.165 |     0.046 |     1.759 |        0.378 |     0.303 |    1.209 |   -0.158 |    0.945 | 154 |
| long           | out-of-sample |    -0.043 |    -0.058 |    -1.781 |        0.002 |     0.204 |    0.112 |   -0.171 |    0.088 | 154 |
| short          | full          |     0.047 |    -0.023 |    -1.128 |       -0.021 |     0.206 |   -0.000 |   -0.202 |   -0.001 | 308 |
| short          | in-sample     |     0.080 |    -0.019 |    -0.625 |       -0.134 |     0.217 |   -0.556 |   -0.202 |   -0.434 | 154 |
| short          | out-of-sample |    -0.001 |    -0.028 |    -1.245 |        0.107 |     0.195 |    0.616 |   -0.158 |    0.481 | 154 |
| LS_data_driven | full          |     0.069 |     0.045 |     0.893 |        0.120 |     0.355 |    0.497 |   -0.295 |    0.550 | 308 |
| LS_data_driven | in-sample     |     0.129 |     0.141 |     3.268 |        0.185 |     0.374 |    0.641 |   -0.149 |    0.501 | 154 |
| LS_data_driven | out-of-sample |    -0.021 |    -0.082 |    -1.559 |        0.059 |     0.336 |    0.337 |   -0.295 |    0.264 | 154 |

### event

| date       |     dL |   dp_pp |   LS_AR |   signed_AR |   z_vs_nonevent |
|:-----------|-------:|--------:|--------:|------------:|----------------:|
| 2025-08-19 |  0.138 |   2.500 |   0.018 |       0.018 |           0.908 |
| 2025-11-05 |  0.435 |   9.000 |   0.013 |       0.013 |           0.623 |
| 2025-11-10 | -0.193 |  -4.000 |  -0.030 |       0.030 |          -1.698 |
| 2025-12-03 |  0.246 |   5.500 |  -0.031 |      -0.031 |          -1.711 |
| 2025-12-05 | -0.134 |  -3.000 |  -0.024 |       0.024 |          -1.380 |
| 2026-02-17 |  0.148 |   3.500 |   0.022 |       0.022 |           1.113 |
| 2026-05-18 |  0.121 |   3.000 |  -0.017 |      -0.017 |          -1.001 |
| 2026-09-15 |  0.161 |   4.000 |  -0.002 |      -0.002 |          -0.187 |
| 2026-09-22 |  0.130 |   3.000 |  -0.011 |      -0.011 |          -0.667 |
| 2026-09-28 | -0.127 |  -3.000 |  -0.010 |       0.010 |          -0.573 |

### sent_basket

| index                          |   daily_corr |   beta_dI |   t_dI |   beta_dI_ctrl_dL |   t_dI_ctrl_dL |   t_dL_ctrl |   weekly_corr |   level_corr_CAR |   granger_p_sent_to_basket |
|:-------------------------------|-------------:|----------:|-------:|------------------:|---------------:|------------:|--------------:|-----------------:|---------------------------:|
| Democracy & rule of law        |       -0.011 |    -0.000 | -0.188 |            -0.000 |         -0.127 |       0.998 |         0.138 |            0.285 |                      0.016 |
| Economy & cost of living       |       -0.072 |    -0.001 | -1.281 |            -0.001 |         -1.258 |       0.913 |        -0.110 |            0.080 |                      0.303 |
| Epstein files & corruption     |        0.030 |     0.001 |  0.529 |             0.001 |          0.541 |       1.019 |         0.039 |            0.305 |                      0.608 |
| Foreign policy & Iran war      |       -0.098 |    -0.002 | -1.712 |            -0.002 |         -1.678 |       0.914 |        -0.045 |            0.000 |                      0.115 |
| Government shutdown & budget   |       -0.059 |    -0.001 | -0.954 |            -0.001 |         -0.911 |       0.812 |        -0.133 |            0.193 |                      0.357 |
| Health care                    |       -0.091 |    -0.002 | -1.683 |            -0.002 |         -1.691 |       1.003 |        -0.135 |            0.194 |                      0.979 |
| Immigration & ICE              |        0.085 |     0.001 |  1.303 |             0.001 |          1.317 |       1.007 |        -0.090 |            0.147 |                      0.278 |
| Redistricting & gerrymandering |        0.085 |     0.002 |  1.556 |             0.002 |          1.487 |       0.881 |         0.073 |            0.298 |                      0.775 |
| MSI                            |       -0.006 |    -0.000 | -0.114 |            -0.000 |         -0.058 |       1.004 |         0.090 |            0.461 |                      0.096 |
| Overall tone                   |        0.030 |     0.001 |  0.547 |             0.001 |          0.517 |       0.993 |         0.162 |            0.261 |                      0.922 |

### extras

{
 "factor_model": {
  "n": 280,
  "end": "2026-08-31",
  "alpha_ann_ff6": 0.12761590190864253,
  "t_alpha_ff6": 0.43400566270556257,
  "r2_ff6": 0.033767550211408315,
  "loadings_ff6": {
   "Mkt-RF": 0.335,
   "SMB": -0.097,
   "HML": 0.194,
   "RMW": 0.151,
   "CMA": 0.436,
   "Mom": 0.013
  },
  "t_loadings_ff6": {
   "Mkt-RF": 1.92,
   "SMB": -0.41,
   "HML": 0.8,
   "RMW": 0.7,
   "CMA": 1.19,
   "Mom": 0.1
  },
  "gamma_dL_ff6": 0.016109623255670526,
  "t_gamma_dL_ff6": 0.8323358277579633,
  "r2_ff6_plus_dL": 0.035572198440207425
 },
 "timing": {
  "msi_timing_oos": {
   "ann_return": -0.07862599862034614,
   "ann_vol": 0.2678351860089967,
   "sharpe": -0.17225489209753914,
   "sortino": -0.23726517411809053,
   "max_dd": -0.1498710002797905,
   "calmar": -0.52462450022727,
   "t_mean": -0.134657843433497,
   "n": 154
  },
  "is_sign_msi": 1.0,
  "prob_momentum_oos": {
   "ann_return": -0.10237221600031365,
   "ann_vol": 0.18391522120447013,
   "sharpe": -0.49525700022386693,
   "sortino": -0.651369996374414,
   "max_dd": -0.1424700366583851,
   "calmar": -0.7185526051753741,
   "t_mean": -0.3871602065021502,
   "n": 154
  },
  "buy_and_hold_LS_oos": {
   "ann_return": -0.12456434757942958,
   "ann_vol": 0.2677802159409443,
   "sharpe": -0.363983581780253,
   "sortino": -0.5219608565560302,
   "max_dd": -0.2964765350703905,
   "calmar": -0.4201490939235885,
   "t_mean": -0.2845390547165131,
   "n": 154
  },
  "info_ratio_msi_vs_bh": 0.12698947964540777
 },
 "event_study_t": 0.8611759303529853,
 "event_study_p": 0.4114964760615192,
 "corr_dL_LS_ci": [
  -0.04852394688835227,
  0.1479829740136382
 ]
}

### summary

{
 "p_start": 0.205,
 "p_end": 0.635,
 "dlogit_total": 1.9090597808261238,
 "LS_gamma_se": 0.03556821581177369,
 "MDE_gamma_80pct_power": 0.09959100427296633,
 "MDE_cum_AR_over_window": 0.19012518078960267,
 "sign_agree": 13,
 "binom_p": 0.13158798217773438,
 "n_sig_5pct": 1,
 "n_weeks": 64,
 "weekly_corr_se": 0.125
}

### composition

{"thesis_long": ["FSLR", "ENPH", "RUN", "NEE", "PLUG", "CNC", "MOH", "OSCR", "HCA", "THC"], "thesis_short": ["GEO", "CXW", "DJT", "COIN", "HOOD", "RGR", "SWBI", "XOM", "BTU", "LMT"], "data_long": ["MOH", "FSLR", "OSCR", "ENPH", "CNC"], "data_short": ["SWBI", "CXW", "NEE", "RGR", "HOOD"], "sign_agreement_is": 0.65, "sign_agreement_full": 0.65}

### msi_weights

| issue                          |   omega |
|:-------------------------------|--------:|
| Democracy & rule of law        |   0.386 |
| Economy & cost of living       |   0.041 |
| Epstein files & corruption     |   0.188 |
| Foreign policy & Iran war      |   0.078 |
| Government shutdown & budget   |   0.035 |
| Health care                    |   0.082 |
| Immigration & ICE              |   0.074 |
| Redistricting & gerrymandering |   0.115 |