# wc2026 — World Cup 2026 Probabilistic Forecasting Engine

Calibrated probability forecasts for the 2026 FIFA World Cup — match outcomes, scoreline
distributions, and Monte-Carlo tournament simulation — evaluated with proper scoring rules
against the de-vigged betting market, live as the tournament runs.

> **Status: Phase 0** — scaffold + core contracts. The modeling ladder (Elo → Dixon–Coles →
> Bayesian hierarchical Poisson → tournament simulator → GBM) lands phase by phase; see
> `RESULTS.md` (generated from Phase 1 onward) for the live scoreboard.

All betting-related analysis in this project is a **simulated, paper-trading analytical
exercise** for measuring calibration and edge. No real wagering, no real-money integration.

## Market benchmark methodology (locked)

- **Closing-line proxy.** Historical closing odds are paywalled on the free tier, so odds are
  snapshotted every 6 hours (GitHub Actions → `data` branch) and the **last stored quote before
  each kickoff** serves as the closing line — at most ~6h stale, uniformly for every match and
  model compared. Coverage starts at the first snapshot (2026-06-12).
- **Benchmark line.** Each book is de-vigged separately; the benchmark is the **sharpest
  available book** (Pinnacle when present), falling back to the vig-free sharp-exchange
  consensus, then to the lowest-overround consensus — never a median over all ~40 books.
  See `eval/market.py` (`BenchmarkPolicy`).

## Setup

```bash
make setup          # uv sync (Python 3.12, pinned via uv.lock)
cp .env.example .env  # add your free API keys (football-data.org, the-odds-api.com)
make check          # ruff + mypy --strict + pytest
```

A full README (results table, methodology, reproducibility, limitations) ships in Phase 6.
