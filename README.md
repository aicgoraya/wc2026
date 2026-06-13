# wc2026 — World Cup 2026 Probabilistic Forecasting Engine

Calibrated probability forecasts for the 2026 FIFA World Cup — match outcomes,
scoreline distributions, and Monte-Carlo tournament simulation — evaluated with
**proper scoring rules and paired significance tests** against the de-vigged
betting market, live as the tournament runs.

A ladder of models (Elo → Dixon–Coles → Bayesian hierarchical Poisson → GBM),
combined into an **ensemble that provably beats every single model**, behind a
leak-free walk-forward harness, a served dashboard, and a one-command pipeline.

> All betting-related analysis is a **simulated, paper-trading study** for
> measuring calibration and edge. No real wagering, no real-money integration.

## Headline results

**Walk-forward, out-of-sample, 2018–2026 (n = 8,107 international matches).** RPS
is primary (lower is better); the betting market can't be scored here because
historical closing odds are paywalled — it enters on the live track.

| model | RPS | what it is |
|---|---:|---|
| Elo + ordinal logistic | 0.1713 | the baseline floor |
| Bayesian hierarchical Poisson | 0.1695 | partial pooling, full posterior uncertainty |
| LightGBM | 0.1705 | discriminative, form/rest/momentum features |
| **Dixon–Coles** | **0.1675** | best single model |
| **Ensemble (DC 0.67 + GBM 0.33)** | **0.1655** | **beats every single model** |

The **ensemble is the result**: no single model beats Dixon–Coles, but a
leak-free convex blend of DC and the GBM does — by ΔRPS −0.0010 vs DC,
**p = 2×10⁻³**, walk-forward with weights refit every 180 days, and the edge is
spread across 6 of 7 years rather than concentrated in one. Two honest negative
results sit underneath it: the Bayesian model **loses** to the (already
RPS-regularized) DC, and the partial-pooling hypothesis was **refuted** — both
reported straight, both pinned with regression tests. See [`RESULTS.md`](RESULTS.md)
for the full paired board, calibration, and the live World-Cup scoreboard.

## What this demonstrates

**For quant** — probabilistic forecasting (distributions, never bare labels);
proper scoring (RPS / log-loss / Brier) with bootstrap CIs; **paired
significance** (per-match ΔRPS bootstrap + Diebold–Mariano) instead of
overlapping marginal CIs; market-efficiency benchmarking against the de-vigged
sharp line; **model combination** (the optimal convex blend) as the source of
edge; strict **leak-free walk-forward** discipline throughout, including
hyperparameter selection on inner windows and ensemble weights fit on earlier
splits than they're tested on.

**For SWE / infra** — a typed, dependency-injected modular package (`mypy
--strict` clean); **200+ tests** including property tests (Hypothesis),
golden tests (the exact 495-row Annex C bracket table), and analytic-gradient
checks; GitHub Actions CI; **cloud data capture** every 6 hours via a scheduled
Actions workflow committing to a data branch; an idempotent refresh pipeline;
and a served FastAPI dashboard.

## Architecture

```
src/wc2026/
  data/         canonical schema, versioned parquet store, fail-loud name resolver, source adapters
  features/     leak-free match features (single chronological pass)
  models/       elo · dixon_coles · bayes_poisson (PyMC) · gbm (LightGBM) · blend  (a Forecaster protocol)
  tournament/   exact 2026 bracket + Annex C table, group tiebreakers, Monte-Carlo simulator
  eval/         scoring · calibration · paired compare · de-vig market · walk-forward · ensemble
  pipeline/     collect · ingest · evaluate/report · tune · refresh
  dashboard/    FastAPI app + static page over the snapshot JSON
  cli.py        the `wc2026` command
```

Every model implements one `Forecaster` protocol, so a new model (or the blend)
drops into the same walk-forward harness, scoreboard, and — for the generative
ones — the tournament simulator, unchanged.

## Reproducibility

```bash
make setup                     # uv sync (Python 3.12, pinned via uv.lock)
uv sync --extra bayes --extra gbm --extra dashboard   # heavier model deps
cp .env.example .env           # add free API keys (football-data.org, the-odds-api.com)
make check                     # ruff + mypy --strict + pytest

uv run wc2026 refresh          # pull data, rebuild dataset, regenerate RESULTS.md + dashboard
uv run wc2026 simulate         # live win-cup table (Dixon–Coles, 50k sims)
uv run wc2026 model-compare     # four-model paired board + ensemble (uses cached MCMC)
uv run wc2026 dashboard         # serve the dashboard at http://127.0.0.1:8000
```

Everything is deterministic: seeded RNG, versioned data snapshots, pinned
dependencies. `tune-dc` and `model-compare` reproduce the frozen
hyperparameters and the paired board exactly.

## Limitations (read this)

- **The market is hard to beat.** Matching the closing line out-of-sample is
  already a strong result; the live blend-vs-market scoreboard starts empty and
  accumulates as matches complete *with stored pre-kickoff lines* — it shows
  `n so far` and draws no conclusions while small. That honesty is the point.
- **No player-level data** for internationals (no lineups/injuries the way club
  leagues have); team-strength latent models are the right abstraction, and the
  features deliberately don't assume lineup data.
- **Small-sample sport.** International football is noisy; the edges here are
  real but modest (~0.001 RPS), and the Bayesian/GBM models *lose* to a
  well-regularized Dixon–Coles — reported straight rather than spun.
- **Closing-line proxy.** Free-tier odds are snapshotted every 6h, so the
  "closing" line trails the true close by up to ~6h, applied uniformly.
- **Tiebreaker approximation.** The simulator implements the exact 2026 group
  criteria (head-to-head first); the unsimulatable fair-play / FIFA-ranking
  final steps fall back to a seeded lot, reached too rarely to move results.

## License

MIT — see [LICENSE](LICENSE).
