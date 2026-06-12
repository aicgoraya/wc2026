import datetime as dt

import pytest

from wc2026.data.merge import merge_canonical
from wc2026.data.schema import Match, MatchStatus, Stage, matches_to_frame


def match(
    match_id: str,
    date: dt.date,
    home: str,
    away: str,
    tournament: str = "friendly",
    goals: tuple[int, int] = (1, 0),
    stage: Stage | None = None,
) -> Match:
    return Match(
        match_id=match_id,
        date=date,
        home_id=home,
        away_id=away,
        home_goals=goals[0],
        away_goals=goals[1],
        neutral=True,
        tournament=tournament,
        stage=stage,
        status=MatchStatus.FINISHED,
    )


def test_wc_window_owned_by_live_feed() -> None:
    history = matches_to_frame(
        [
            # same real match, recorded under the LOCAL date by martj42
            match("mj_a", dt.date(2026, 6, 12), "united_states", "paraguay", "fifa_world_cup"),
            match("mj_b", dt.date(2022, 12, 18), "argentina", "france", "fifa_world_cup"),
            match("mj_c", dt.date(2026, 6, 12), "ghana", "togo", "friendly"),
        ]
    )
    wc = matches_to_frame(
        [
            match(
                "fd_1",
                dt.date(2026, 6, 13),
                "united_states",
                "paraguay",
                "fifa_world_cup",
                stage=Stage.GROUP,
            )
        ]
    )
    merged = merge_canonical(history, wc)
    ids = set(merged["match_id"])
    assert "mj_a" not in ids  # WC2026 window belongs to the live feed
    assert {"mj_b", "mj_c", "fd_1"} <= ids  # old WCs and non-WC matches kept


def test_duplicate_match_id_raises() -> None:
    history = matches_to_frame([match("dup", dt.date(2020, 1, 1), "brazil", "peru")])
    wc = matches_to_frame([match("dup", dt.date(2026, 6, 12), "brazil", "peru", "fifa_world_cup")])
    with pytest.raises(ValueError, match="duplicate match ids"):
        merge_canonical(history, wc)


def test_exact_duplicate_match_raises() -> None:
    history = matches_to_frame(
        [
            match("a", dt.date(2020, 1, 1), "brazil", "peru"),
            match("b", dt.date(2020, 1, 1), "peru", "brazil", goals=(0, 1)),
        ]
    )
    with pytest.raises(ValueError, match="duplicate matches"):
        merge_canonical(history, matches_to_frame([]).astype(history.dtypes.to_dict()))


def test_double_header_passes() -> None:
    history = matches_to_frame(
        [
            match("a", dt.date(1974, 2, 17), "tahiti", "new_caledonia", goals=(2, 1)),
            match("a_2", dt.date(1974, 2, 17), "tahiti", "new_caledonia", goals=(1, 2)),
        ]
    )
    empty_wc = matches_to_frame([]).astype(history.dtypes.to_dict())
    merged = merge_canonical(history, empty_wc)
    assert len(merged) == 2
