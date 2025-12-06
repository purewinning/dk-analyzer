# builder.py

import pandas as pd
from pulp import *
from typing import Dict, List, Tuple, Union

# --- CONFIGURATION CONSTANTS ---
# Ownership Thresholds (0-100 scale)
PUNT_THR = 10.0
CHALK_THR = 30.0
MEGA_CHALK_THR = 40.0

DEFAULT_SALARY_CAP = 50000
DEFAULT_ROSTER_SIZE = 8

# --- 1. CONTEST TEMPLATES ---

# Defines the positional requirements and ownership targets for different contest types.
DK_NBA_TEMPLATES = {
    # DraftKings NBA Roster Slots (8 total players)
    # The structure must match the Lineup Requirements image provided.
    "ROSTER_SLOTS": {
        "PG": 1, 
        "SG": 1,
        "SF": 1,
        "PF": 1,
        "C": 1,
        "G": 1,  # Must be PG or SG
        "F": 1,  # Must be SF or PF
        "Util": 1 # Must be PG, SG, SF, PF, or C
    },
    
    # Ownership Buckets: (Min Count, Max Count) for 8 players
    "OWNERSHIP_TARGETS": {
        # Cash/Single Entry (SE) - Prioritizes high projection/value, moderate ownership.
        "CASH": {
            "punt": (0, 3),        # 0-10% Ownership
            "mid": (2, 6),         # 10-30% Ownership
            "chalk": (2, 6),       # 30-40% Ownership
            "mega": (0, 2)         # >40% Ownership
        },
        # Single Entry GPP (SE) - Balances chalk with leverage.
        "SE": {
            "punt": (1, 4),
            "mid": (2, 5),
            "chalk": (1, 4),
            "mega": (0, 2)
        },
        # Large Field GPP - Maximizes leverage; requires more low-owned players.
        "LARGE_GPP": {
            "punt": (2, 5),
            "mid": (1, 4),
            "chalk": (0, 3),
            "mega": (0, 1)
        }
    }
}

# --- 2. TEMPLATE CLASS (Helper) ---

class LineupTemplate:
    """Class to hold and manage contest constraints."""
    
    def __init__(self, contest_type: str, roster_size: int, salary_cap: int, min_games: int):
        self.contest_type = contest_type
        self.contest_label = self._get_contest_label(contest_type)
        self.roster_size = roster_size
        self.salary_cap = salary_cap
        self.min_games = min_games
        self.pos_req = DK_NBA_TEMPLATES["ROSTER_SLOTS"]
        self.own_targets = DK_NBA_TEMPLATES["OWNERSHIP_TARGETS"][contest_type]

    def _get_contest_label(self, code: str) -> str:
        """Translates contest code to a friendly label."""
        if code == 'CASH': return "Cash Game"
        if code == 'SE': return "Single Entry GPP"
        if code == 'LARGE_GPP': return "Large Field GPP"
        return "Unknown Contest"

    def bucket_ranges(self, slack: int = 1) -> Dict[str, Tuple[int, int]]:
        """Returns ownership ranges adjusted by slack (e.g., for ownership constraints)."""
        ranges = {}
        for bucket, (min_val, max_val) in self.own_targets.items():
            ranges[bucket] = (max(0, min_val - slack), max_val + slack)
        return ranges

def build_template_from_params(
    contest_type: str, 
    field_size: int, 
    pct_to_first: float, 
    roster_size: int,
    salary_cap: int,
    min_games: int
) -> LineupTemplate:
    """Initializes and returns a LineupTemplate instance."""
    return LineupTemplate(
        contest_type=contest_type, 
        roster_size=roster_size,
        salary_cap=salary_cap,
        min_games=min_games
    )

# --- 3. CORE LOGIC ---

def ownership_bucket(own_proj: float) -> str:
    """Categorizes a player based on their ownership projection."""
    if own_proj < PUNT_THR:
        return 'punt'
    elif own_proj < CHALK_THR:
        return 'mid'
    elif own_proj < MEGA_CHALK_THR:
        return 'chalk'
    else:
        return 'mega'

def build_optimal_lineup(
    slate_df: pd.DataFrame, 
    template: LineupTemplate, 
    bucket_slack: int,
    locked_player_ids: List[str], 
    excluded_player_ids: List[str]
) -> Union[pd.DataFrame, None]:
    """
    Finds the optimal lineup using PuLP based on constraints.
    Returns the optimal lineup DataFrame or None if infeasible.
    """
    
    # Filter for playable players
    playable_df = slate_df[~slate_df['player_id'].isin(excluded_player_ids)].copy()
    
    # 1. Setup the Problem
    prob = LpProblem("DFS Lineup Optimization", LpMaximize)
    
    # Decision Variables: 1 if player is chosen, 0 otherwise
    player_vars = LpVariable.dicts("Player", playable_df['player_id'], 0, 1, LpBinary)
    
    # 2. Objective Function: Maximize total projected points
    prob +=
