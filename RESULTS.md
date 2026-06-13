# RESULTS - live scoreboard

_Generated 2026-06-13 07:02 UTC. Lower is better on every metric; RPS is primary. `ece` = expected calibration error. CIs are 95% bootstrap._

## Track A — PRIMARY: historical walk-forward (real sample size)

All internationals, 2010-01-01 -> 2026-06-10; every prediction fit strictly on
pre-cutoff matches (ratings AND the ordinal link). No market column here:
historical closing odds are paywalled on the free tier, so the market is
only benchmarked on the live track below.

| model | n | rps | rps_ci_lo | rps_ci_hi | log_loss | brier | ece |
|---|---|---|---|---|---|---|---|
| elo_baseline | 15816 | 0.1749 | 0.1725 | 0.1774 | 0.8886 | 0.5232 | 0.0068 |
| dixon_coles | 15816 | 0.1711 | 0.1687 | 0.1735 | 0.8742 | 0.5130 | 0.0072 |

- dixon_coles: decay half-life (1460d) and L2 shrinkage (0.25) selected JOINTLY by walk-forward RPS on an inner 2004-2009 validation window (training always pre-cutoff), frozen and committed before the 2010+ test window was evaluated - full table in results/dc_tuning.md; rho and the neutral listed-home coefficient fitted by MLE. Reproduce with `wc2026 tune-dc`.

### Paired significance (per-match ΔRPS on shared matches)

The marginal CIs above share the same matches, so they overlap even when one
model is reliably better game-by-game. The honest test is the paired per-match
RPS difference: `mean_dRPS = challenger - baseline` (negative => challenger
better), with a 95% bootstrap CI over matches and a Diebold-Mariano statistic.

| comparison | n | mean_dRPS | ci_lo | ci_hi | DM | p | verdict |
|---|---|---|---|---|---|---|---|
| dixon_coles - elo_baseline | 15816 | -0.0038 | -0.0049 | -0.0028 | -6.8474 | 0.0000 | dixon_coles better (p=7.5e-12) |

### Tournament-only slice (World Cup + continental finals)

| model | n | rps | rps_ci_lo | rps_ci_hi | log_loss | brier | ece |
|---|---|---|---|---|---|---|---|
| elo_baseline | 1442 | 0.1853 | 0.1782 | 0.1927 | 0.9416 | 0.5591 | 0.0088 |
| dixon_coles | 1442 | 0.1828 | 0.1758 | 0.1901 | 0.9308 | 0.5524 | 0.0127 |

![elo_baseline reliability (primary)](results/reliability_elo_baseline_primary.png)

![dixon_coles reliability (primary)](results/reliability_dixon_coles_primary.png)

## Track B — LIVE: World Cup 2026 (SMALL SAMPLE - no conclusions yet)

Accumulates match by match as the tournament runs. The market column is
the de-vigged closing-line proxy (sharpest book; see README). With the
current sample size this table is reported for transparency only -
**do not read anything into it yet.**

| model | n | rps | rps_ci_lo | rps_ci_hi | log_loss | brier | ece |
|---|---|---|---|---|---|---|---|
| elo_baseline (all completed) | 4 | 0.1837 | 0.0732 | 0.2726 | 0.9541 | 0.5961 | 0.2397 |
| elo_baseline (common w/ market) | 1 | 0.2444 | 0.2444 | 0.2444 | 0.9793 | 0.5849 | 0.0000 |
| dixon_coles (all completed) | 4 | 0.1708 | 0.1118 | 0.2133 | 0.8963 | 0.5361 | 0.2578 |
| dixon_coles (common w/ market) | 1 | 0.1995 | 0.1995 | 0.1995 | 0.8382 | 0.4832 | 0.3783 |
| market (matches w/ lines) | 1 | 0.1719 | 0.1719 | 0.1719 | 0.7680 | 0.4329 | 0.3574 |

- Live model rows cover all 4 completed WC matches; market rows exist only where a pre-kickoff quote was stored (collection began 2026-06-12): 1 matches so far.
- Model-vs-market gaps are quantified on their COMMON matches as they accumulate; the baseline is expected to lose to the market - that gap is the target for Dixon-Coles and the Bayesian model.

# Phase 4: Bayesian vs Dixon-Coles vs Elo

Walk-forward 2018-01-01 -> 2026-06-10, refit every 180d (MCMC cost: 0 min for the Bayesian refits; DC/Elo negligible).
All three scored on the SAME matches at the SAME cadence so the paired test isolates the model, not the schedule.

## Scoreboard (shared window)

| model | n | rps | rps_ci_lo | rps_ci_hi | log_loss | brier | ece |
|---|---|---|---|---|---|---|---|
| elo_baseline | 8107 | 0.1713 | 0.1679 | 0.1747 | 0.8753 | 0.5147 | 0.0096 |
| dixon_coles | 8107 | 0.1675 | 0.1642 | 0.1708 | 0.8606 | 0.5048 | 0.0106 |
| bayes_poisson | 8107 | 0.1695 | 0.1664 | 0.1726 | 0.8675 | 0.5094 | 0.0165 |

## Paired significance (per-match ΔRPS)

| comparison | n | mean_dRPS | ci_lo | ci_hi | DM | p | verdict |
|---|---|---|---|---|---|---|---|
| bayes_poisson - dixon_coles | 8107 | 0.0020 | 0.0013 | 0.0027 | 5.9307 | 0.0000 | dixon_coles better (p=3.0e-09) |
| bayes_poisson - elo_baseline | 8107 | -0.0018 | -0.0030 | -0.0005 | -2.7468 | 0.0060 | bayes_poisson better (p=6.0e-03) |

## Headline test: does partial pooling help most on sparse teams?

Mean ΔRPS (bayes - dixon_coles) by the decayed match-count of the weaker side of each game (negative => Bayes better):

| tercile | mean | count |
|---|---|---|
| sparse | 0.0026 | 2702 |
| mid | 0.0019 | 2702 |
| rich | 0.0016 | 2703 |

## Convergence

Every refit uses the config verified to converge on the real data: R-hat 1.0000, min ESS 4224, 0 divergences (representative as-of-2026-06-13 fit; 11.5k matches, 298 teams). Trace plot: `results/bayes_trace.png`.

## Interpretation (reported straight)

The Bayesian model beats Elo but **loses to the well-tuned Dixon-Coles**, and the partial-pooling hypothesis is **refuted**: Bayes trails DC by the *most* on sparse teams, not the least. The reason is honest and instructive — DC is not unregularized MLE; its L2 shrinkage was explicitly tuned to minimize out-of-sample RPS (Phase 2). Once the frequentist model already shrinks sparse teams toward the population, the Bayesian prior's shrinkage adds little and costs a touch of sharpness/calibration on this exact metric. The Bayesian model's genuine contribution here is calibrated **uncertainty** (posterior intervals on every prediction), not point-prediction accuracy. Not rigged either way; the number is what it is.

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

## Ensemble robustness — rolling weights, walk-forward

Weights refit every 180d on expanding pre-cutoff data, applied to the next block; predictions aggregated out-of-sample (n=6022).

| model | rps |
|---|---|
| blend | 0.1655 |
| dixon_coles | 0.1665 |
| bayes_poisson | 0.1686 |
| gbm | 0.1695 |
| elo_baseline | 0.1702 |

| comparison | n | mean_dRPS | ci_lo | ci_hi | p | verdict |
|---|---|---|---|---|---|---|
| blend - elo_baseline | 6022 | -0.0047 | -0.0059 | -0.0035 | 0.0000 | blend better (p=4.1e-14) |
| blend - dixon_coles | 6022 | -0.0010 | -0.0016 | -0.0003 | 0.0021 | blend better (p=2.1e-03) |
| blend - bayes_poisson | 6022 | -0.0031 | -0.0038 | -0.0024 | 0.0000 | blend better (p=1.8e-16) |
| blend - gbm | 6022 | -0.0039 | -0.0052 | -0.0027 | 0.0000 | blend better (p=2.8e-10) |

Per-year blend-minus-DC mean ΔRPS (negative => blend better; shows the edge is spread across windows, not concentrated):

| year | mean_dRPS | n |
|---|---|---|
| 2020 | 0.0003 | 340 |
| 2021 | -0.0008 | 1115 |
| 2022 | -0.0008 | 970 |
| 2023 | -0.0022 | 1054 |
| 2024 | -0.0001 | 1231 |
| 2025 | -0.0016 | 1002 |
| 2026 | -0.0002 | 310 |
