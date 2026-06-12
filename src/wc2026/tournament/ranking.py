"""Exact 2026 group and best-thirds ranking rules.

Implemented from the official FIFA regulations (H2H-first within groups — new
for 2026 — then overall GD/goals, conduct, FIFA ranking). The conduct-score
criterion cannot be simulated; the simulator falls back to FIFA ranking and
then seeded lots, and that approximation is documented where it applies.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd


def rank_group(
    group: str,
    played: pd.DataFrame,
    fifa_ranks: Mapping[str, int],
    rng: np.random.Generator,
) -> list[str]:
    """Rank a group's four teams per the 2026 regulations; best first."""
    raise NotImplementedError("ships in Phase 3")


def rank_thirds(thirds: pd.DataFrame, fifa_ranks: Mapping[str, int]) -> list[str]:
    """Rank the twelve third-placed teams; the first eight advance."""
    raise NotImplementedError("ships in Phase 3")
