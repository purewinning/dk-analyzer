import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

# --- CONSTANTS ---

DEFAULT_SALARY_CAP = 50000
DEFAULT_ROSTER_SIZE = 8

# Ownership bucket thresholds (in percent) – for visual buckets
PUNT_THR = 10.0       # < 10% = punt / contrarian
CHALK_THR = 30.0      # 10–30 mid, 30–40 chalk
MEGA_CHALK_THR = 40.0 # > 40% mega-chalk


# ======================================================================================
# DATA STRUCTURES
# ======================================================================================

@dataclass
class LineupTemplate:
    """
    Minimal lineup template for controlling salary, roster size, etc.
    """
    roster_size: int = DEFAULT_ROSTER_SIZE
    salary_cap: int = DEFAULT_SALARY_CAP
    min_salary: int = 0
    min_projection: float = 0.0
    min_floor: float = 0.0
    max_punts: int = 3   # max players in punt bucket
    allow_same_team: bool = True


# ======================================================================================
# OWNERSHIP BUCKETS
# ======================================================================================

def ownership_bucket(own_proj: float) -> str:
    """
    Map projected ownership (percentage) into a descriptive bucket:
      - 'Punt' < 10%
      - 'Mid' 10–30%
      - 'Chalk' 30–40%
      - 'Mega Chalk' > 40%
    """
    if pd.isna(own_proj):
        return "Unknown"

    if own_proj < PUNT_THR:
        return "Punt"
    elif own_proj < CHALK_THR:
        return "Mid"
    elif own_proj < MEGA_CHALK_THR:
        return "Chalk"
    else:
        return "Mega Chalk"


# ======================================================================================
# GAME ENVIRONMENTS
# ======================================================================================

def build_game_environments(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Builds a simple "game environments" dictionary keyed by 'Team_vs_Opp'.
    Currently this just groups players by matchup and stores some summary stats.

    We assume the DF has columns:
      - Team
      - Opponent
      - proj  (projection)

    Returns: 
      {
        "LAL_vs_GSW": {
            "matchup": "LAL @ GSW",
            "Team": "LAL",
            "Opponent": "GSW",
            "total_proj": ...,
            "pace": ... (placeholder),
            ...
        },
        ...
      }
    """
    if "Team" not in df.columns or "Opponent" not in df.columns:
        return {}

    game_envs: Dict[str, Dict[str, Any]] = {}

    for (team, opp), grp in df.groupby(["Team", "Opponent"]):
        key = f"{team}_vs_{opp}"
        total_proj = grp["proj"].sum()
        # Placeholder "pace" and "total" – in real use you'd import vegas data, etc.
        game_envs[key] = {
            "matchup": f"{team} vs {opp}",
            "Team": team,
            "Opponent": opp,
            "total_proj": float(total_proj),
            "pace": None,
            "total": None,
            "spread": None,
        }

    return game_envs


# ======================================================================================
# TEAM STACKS (HEURISTIC)
# ======================================================================================

def build_team_stacks(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    Identify viable team stacks (primary + secondary players).
    Returns dict keyed by Team with list of stack combinations.
    """
    team_stacks = {}
    
    for team in df["Team"].unique():
        team_players = df[df["Team"] == team].copy()
        
        if len(team_players) < 2:
            continue
        
        # Identify stars (top 30% salary on team)
        salary_threshold = team_players["Salary"].quantile(0.7)
        stars = team_players[team_players["Salary"] >= salary_threshold]
        
        # Identify secondary pieces (others)
        secondary = team_players[team_players["Salary"] < salary_threshold]
        
        stacks = []
        for _, star_row in stars.iterrows():
            for _, sec_row in secondary.iterrows():
                stacks.append({
                    "players": [star_row["Name"], sec_row["Name"]],
                    "avg_salary": float((star_row["Salary"] + sec_row["Salary"]) / 2),
                    "avg_proj": float((star_row["proj"] + sec_row["proj"]) / 2),
                })
        
        if stacks:
            # Sort stacks by average projection
            stacks = sorted(stacks, key=lambda x: x["avg_proj"], reverse=True)
            team_stacks[team] = stacks
    
    return team_stacks


# ======================================================================================
# LINEUP CORRELATION (NBA-style)
# ======================================================================================

def calculate_lineup_correlation_score(
    lineup_df: pd.DataFrame,
    game_envs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> float:
    """
    Rough correlation metric: 
      - Reward lineups with multiple players from same game environments,
        because they might benefit from same high-scoring game.
    """
    if lineup_df is None or lineup_df.empty:
        return 0.0

    if game_envs is None:
        # Fallback: small reward for players from same Team or Opp
        teams = lineup_df["Team"].tolist()
        team_counts = Counter(teams)
        score = sum((cnt - 1) for cnt in team_counts.values() if cnt > 1)
        return float(score)

    # If we have explicit game_envs, we attempt to match by Team/Opponent
    # We'll create keys "Team_vs_Opp" or "Opp_vs_Team" and see how many share them
    game_keys = []
    for _, row in lineup_df.iterrows():
        team = row.get("Team")
        opp = row.get("Opponent")
        if not team or not opp:
            continue
        key1 = f"{team}_vs_{opp}"
        key2 = f"{opp}_vs_{team}"
        if key1 in game_envs:
            game_keys.append(key1)
        elif key2 in game_envs:
            game_keys.append(key2)

    if not game_keys:
        return 0.0

    # Reward multiple players from same game environment
    env_counts = Counter(game_keys)
    score = 0.0
    for env, cnt in env_counts.items():
        if cnt > 1:
            score += (cnt - 1)  # small reward for extra players in same environment

    return float(score)


# ======================================================================================
# MISC UTILS
# ======================================================================================

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
    directly to the UI.
    """
    # You can tune defaults based on contest type if desired
    min_salary = int(0.95 * salary_cap) if "GPP" in contest_type else int(0.98 * salary_cap)
    min_projection = 0.0   # now app-side
    min_floor = 0.0        # now app-side
    max_punts = 3          # also controlled by app
    allow_same_team = True # for NBA / casual default

    return LineupTemplate(
        roster_size=roster_size,
        salary_cap=salary_cap,
        min_salary=min_salary,
        min_projection=min_projection,
        min_floor=min_floor,
        max_punts=max_punts,
        allow_same_team=allow_same_team,
    )


# ======================================================================================
# (Optionally) if you want a quick local test
# ======================================================================================

if __name__ == "__main__":
    # Very small smoke test for ownership_bucket
    test_owns = [np.nan, 5, 15, 35, 50]
    for o in test_owns:
        print(o, "->", ownership_bucket(o))
