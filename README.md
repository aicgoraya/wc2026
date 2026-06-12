# wc2026 data branch

Versioned snapshots collected by the `collect` GitHub Actions workflow
(every 6 hours): `snapshots/` (parquet + manifests) and `raw/` (verbatim API
responses for replay). This branch is data-only — code lives on `main`.

The last snapshot before each kickoff serves as the closing-line proxy for
the market benchmark.
