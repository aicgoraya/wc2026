# World Cup 2026 forecaster

I wanted to see if I could predict soccer games better than the betting market, so I spent a
while this summer building this. It puts probabilities on every 2026 World Cup match (chance of
home win / draw / away win, plus full scorelines), simulates the whole tournament to get each
team's odds of winning the cup, and then grades itself against the bookies to see how close I
actually got.

Short version of what I found: I could not beat a good single model with a fancier single model,
but I *could* beat it by mixing two of them together. That surprised me and is the most
interesting part of the project.

> Heads up: the betting stuff is all paper-trading / just for measuring how good the predictions
> are. I'm not actually wagering money and there's no real-money anything in here.

## The results

Everything is scored "walk-forward" — I only ever let a model train on games that happened
*before* the match it's predicting, so it can't cheat by peeking at the answer. This is on
~8,000 real international games from 2018–2026. The number is RPS (lower = better; it's the
standard way to score ordered probability forecasts).

| model | RPS |
|---|---:|
| Elo (the simple baseline) | 0.1713 |
| Bayesian hierarchical Poisson | 0.1695 |
| LightGBM (gradient boosting) | 0.1705 |
| **Dixon–Coles** (best single model) | **0.1675** |
| **Blend of Dixon–Coles + LightGBM** | **0.1655** |

So the Bayesian model and the gradient-boosted model both *lost* to plain Dixon–Coles, which I
did not expect going in. But if you blend Dixon–Coles with the gradient-boosted model (about 2/3
to 1/3), the combo beats every single model on its own, and it holds up when I re-pick the blend
weights every six months and roll forward — not just on one lucky split. The gap is small
(~0.001 RPS) but it's real and it shows up in 6 of the last 7 years. Full breakdown, calibration
plots, and the live World-Cup scoreboard are in [RESULTS.md](RESULTS.md).

## Stuff I learned the hard way

- **Combining models is where the edge is.** Each model on its own was fine; the win came from
  mixing them. The gradient-boosted model adds info the others can't see (recent form, rest days,
  travel) even though it's worse by itself.
- **The Bayesian model losing was a good lesson.** The whole pitch for partial pooling is that it
  helps teams with little data — but my Dixon–Coles model was already regularized, so the fancy
  prior didn't add much. I left the result in honestly instead of tuning until it "won."
- **Leakage is sneaky.** Getting the walk-forward right (including how I pick hyperparameters and
  blend weights) took more care than the models themselves.
- **Reading the actual rules matters.** The 2026 group tiebreakers use head-to-head *first* now,
  which is new, and the 8-of-12 third-place bracket is a real 495-row lookup table I had to parse
  out of FIFA's regulations and golden-test.

## What's in here

```
src/wc2026/
  data/         pulling + cleaning game data and odds, name matching, a little versioned store
  features/     match features the boosting model uses (form, rest, momentum) — all leak-free
  models/       elo · dixon_coles · bayes_poisson (PyMC) · gbm (LightGBM) · the blend
  tournament/   the real 2026 bracket + Annex C table, group tiebreakers, the Monte-Carlo sim
  eval/         scoring, calibration, the paired significance tests, market de-vigging, ensemble
  pipeline/     pull data → update models → re-simulate → regenerate the results
  dashboard/    a little FastAPI page that shows it all
```

Every model speaks the same interface, so the blend and the tournament simulator don't care which
one you hand them.

## Running it

```bash
make setup                                   # installs everything (uses uv + Python 3.12)
cp .env.example .env                         # add two free API keys if you want live data
make check                                   # lint + types + tests

uv run wc2026 simulate                       # win-the-cup table from 50k simulations
uv run wc2026 model-compare                  # the model-vs-model scoreboard + the blend
uv run wc2026 refresh                        # pull latest, rebuild everything
uv run wc2026 dashboard                      # http://127.0.0.1:8000
```

It's all seeded and reproducible, and there are 200+ tests (including a few property-based ones
and the golden test for the bracket) plus CI on every push.

## Things I'd flag honestly

- **The market is genuinely hard to beat.** Matching the closing line is already a strong result;
  the live "can I beat the bookies" scoreboard starts at zero matches and only grows as games
  finish *and* I had odds stored beforehand. It shows how many matches it's actually scored so far
  and I don't draw any conclusions while that number is tiny. That honesty is kind of the point.
- International soccer is small-sample and noisy, so all the edges here are small.
- There's no player/injury/lineup data for national teams the way there is for clubs, so the
  models work at the team-strength level on purpose.
- The "closing" odds I compare against are snapshotted every 6 hours, so they can be a little
  stale, but the same way for every match.

## License

MIT — do whatever you want with it.
