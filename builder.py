import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

# --- CONSTANTS ---

DEFAULT_SALARY_CAP = 50000
DEFAULT_ROSTER_SIZE = 8

# Ownership bucket thresholds (in percent) – for visual buckets
PUNT_THR = 10.0       # < 10% = punt / contrarian
CHALK_THR = 30.0      # 10–30 mid, 30–40 chalk
MEGA_CHALK_THR = 40.0 # > 40 mega-chalk


# --- TEMPLATE OBJECT --------------------------------------------------------


@dataclass
class LineupTemplate:
    contest_type: str
    field_size: int
    pct_to_first: float
    roster_size: int = DEFAULT_ROSTER_SIZE
    salary_cap: int = DEFAULT_SALARY_CAP
    min_games: int = 2


def build_template_from_params(
    contest_type: str,
    field_size: int,
    pct_to_first: float,
    roster_size: int = DEFAULT_ROSTER_SIZE,
    salary_cap: int = DEFAULT_SALARY_CAP,
    min_games: int = 2,
) -> LineupTemplate:
    """
    Simple wrapper used by app.py to construct a template object.

    The template itself is intentionally lightweight; almost all of the
    strategy / ownership logic now lives in app.py where we can expose it
    cleanly to the user.
    """
    return LineupTemplate(
        contest_type=contest_type,
        field_size=field_size,
        pct_to_first=pct_to_first,
        roster_size=roster_size,
        salary_cap=salary_cap,
        min_games=min_games,
    )


# --- OWNERSHIP BUCKETING ----------------------------------------------------


def ownership_bucket(own: float) -> str:
    """Map projected ownership into a coarse bucket (for display)."""
    if pd.isna(own):
        return "mid"
    if own >= MEGA_CHALK_THR:
        return "mega"
    if own >= CHALK_THR:
        return "chalk"
    if own >= PUNT_THR:
        return "mid"
    return "punt"


# --- LINEUP GENERATION ------------------------------------------------------


def _prepare_player_pool(
    slate_df: pd.DataFrame,
    locked_player_ids: List[str],
    excluded_player_ids: List[str],
) -> pd.DataFrame:
    """
    Return a clean player pool.

    Assumes slate_df already has columns:
        - player_id (str)
        - salary (int)
        - proj (float)
        - own_proj (float)  [optional, but handy]
    """
    df = slate_df.copy()

    # Ensure needed columns exist
    required_cols = ["player_id", "salary", "proj"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Slate is missing required column: {col}")

    # Drop excluded
    if excluded_player_ids:
        df = df[~df["player_id"].isin(excluded_player_ids)]

    # Make types safe
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce").fillna(0.0)

    # Remove totally unplayable rows
    df = df[df["salary"] > 0]
    df = df[df["proj"] > 0]

    # Ensure locked players are available
    missing_locks = set(locked_player_ids) - set(df["player_id"].tolist())
    if missing_locks:
        raise ValueError(f"Locked players not found in slate: {', '.join(missing_locks)}")

    return df


def _random_lineup(
    pool: pd.DataFrame,
    template: LineupTemplate,
    locked_ids: List[str],
    max_salary_leaving: int = 1000,
    rng: Optional[random.Random] = None,
) -> Optional[List[str]]:
    """
    Generate a single random, salary-legal lineup.

    This is a very light-weight stochastic search used to feed the
    downstream scoring / template logic in app.py.  Being slightly noisy
    here is fine – we will generate many candidates and keep the best.
    """
    if rng is None:
        rng = random

    roster_size = template.roster_size
    cap = template.salary_cap

    # Start lineup with locked players
    current_ids = list(dict.fromkeys(locked_ids))  # de-dupe, preserve order

    # Quickly bail if locks already break salary cap or roster size
    locks_df = pool[pool["player_id"].isin(current_ids)]
    lock_salary = int(locks_df["salary"].sum())
    if len(current_ids) > roster_size or lock_salary > cap:
        return None

    remaining_spots = roster_size - len(current_ids)
    remaining_cap = cap - lock_salary

    # Pool of candidates we can still use
    available = pool[~pool["player_id"].isin(current_ids)].copy()
    if available.empty and remaining_spots > 0:
        return None

    # Greedy-ish random fill: sort by value + noise, then walk down
    if "value" in available.columns:
        base_score = available["value"].values
    else:
        # proj per 1k
        base_score = available["proj"].values / np.maximum(available["salary"].values / 1000, 1)

    scores = []
    for v in base_score:
        scores.append(v + rng.normalvariate(0, 0.15))
    available = available.assign(_score=scores).sort_values("_score", ascending=False)

    for _, row in available.iterrows():
        if remaining_spots == 0:
            break
        if row["salary"] <= remaining_cap:
            current_ids.append(row["player_id"])
            remaining_spots -= 1
            remaining_cap -= int(row["salary"])

    if remaining_spots != 0:
        # Could not fill the roster within the cap
        return None

    # Enforce soft minimum salary usage (don't leave > max_salary_leaving)
    used_salary = cap - remaining_cap
    if cap - used_salary > max_salary_leaving:
        return None

    return current_ids


def generate_top_n_lineups(
    slate_df: pd.DataFrame,
    template: LineupTemplate,
    n_lineups: int = 10,
    bucket_slack: int = 2,  # kept for API compatibility; not used here
    locked_player_ids: Optional[List[str]] = None,
    excluded_player_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate a set of high-projection, salary-legal lineups.

    This is intentionally generic – it does **not** try to enforce
    contest-specific strategy by itself.  Instead, it:

        * Respects salary cap and roster size
        * Respects Lock / Exclude
        * Maximises projection subject to a simple random search

    Then app.py applies contest-specific scoring (chalk / mid /
    contrarian mix, total ownership, etc.) and keeps the lineups that
    best match the selected strategy.

    Returns a list of dicts:
        {
            'player_ids': [str, ...],
            'proj_score': float,
            'salary_used': int,
        }
    """
    locked_player_ids = locked_player_ids or []
    excluded_player_ids = excluded_player_ids or []

    pool = _prepare_player_pool(slate_df, locked_player_ids, excluded_player_ids)

    rng = random.Random(42)

    best: Dict[frozenset, Dict[str, Any]] = {}

    # Number of random attempts – scale with desired lineups and pool size
    attempts = max(2000, 400 * n_lineups)
    for _ in range(attempts):
        lineup_ids = _random_lineup(pool, template, locked_player_ids, rng=rng)
        if lineup_ids is None:
            continue

        key = frozenset(lineup_ids)
        if key in best:
            continue

        lineup_df = pool[pool["player_id"].isin(lineup_ids)]
        proj_score = float(lineup_df["proj"].sum())
        salary_used = int(lineup_df["salary"].sum())

        best[key] = {
            "player_ids": lineup_ids,
            "proj_score": proj_score,
            "salary_used": salary_used,
        }

    # Keep top n by projected score
    all_lineups = sorted(best.values(), key=lambda x: x["proj_score"], reverse=True)
    return all_lineups[:n_lineups]
