# Phase 5: GBM + ensemble vs the generative ladder

Walk-forward 2018-01-01 -> 2026-06-10, 180d cadence, same matches for all four models.

## Scoreboard (shared window)

| model | n | rps | log_loss | brier | ece |
|---|---|---|---|---|---|
| elo_baseline | 8107 | 0.1713 | 0.8753 | 0.5147 | 0.0096 |
| dixon_coles | 8107 | 0.1675 | 0.8606 | 0.5048 | 0.0106 |
| bayes_poisson | 8107 | 0.1695 | 0.8675 | 0.5094 | 0.0165 |
| gbm | 8107 | 0.1705 | 0.8733 | 0.5133 | 0.0126 |

## Paired vs Dixon-Coles (per-match ΔRPS)

| comparison | n | mean_dRPS | ci_lo | ci_hi | p | verdict |
|---|---|---|---|---|---|---|
| gbm - dixon_coles | 8107 | 0.0030 | 0.0014 | 0.0047 | 0.0003 | dixon_coles better (p=3.2e-04) |
| bayes_poisson - dixon_coles | 8107 | 0.0020 | 0.0013 | 0.0027 | 0.0000 | dixon_coles better (p=3.0e-09) |
| elo_baseline - dixon_coles | 8107 | 0.0038 | 0.0023 | 0.0053 | 0.0000 | dixon_coles better (p=4.5e-07) |

## Ensemble — leak-free time-split convex blend

Weights fit on matches before 2022-06-01 (n=3854), blend evaluated on matches after (n=4253). Optimal weights: elo_baseline 0.02, dixon_coles 0.66, bayes_poisson 0.00, gbm 0.32.

| model | rps |
|---|---|
| blend | 0.1672 |
| dixon_coles | 0.1683 |
| bayes_poisson | 0.1703 |
| gbm | 0.1712 |
| elo_baseline | 0.1726 |

| comparison | n | mean_dRPS | ci_lo | ci_hi | p | verdict |
|---|---|---|---|---|---|---|
| blend - elo_baseline | 4253 | -0.0054 | -0.0069 | -0.0040 | 0.0000 | blend better (p=2.1e-12) |
| blend - dixon_coles | 4253 | -0.0011 | -0.0019 | -0.0003 | 0.0048 | blend better (p=4.8e-03) |
| blend - bayes_poisson | 4253 | -0.0031 | -0.0040 | -0.0022 | 0.0000 | blend better (p=2.2e-11) |
| blend - gbm | 4253 | -0.0040 | -0.0055 | -0.0025 | 0.0000 | blend better (p=1.4e-07) |

**Blend beats the best single model on the held-out window: yes.**
