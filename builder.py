# builder.py
# (Contents remain as previously corrected, focusing on clean syntax)

import pandas as pd
import numpy as np 
from pulp import *
from typing import Dict, List, Tuple, Union, Any

# --- CONFIGURATION CONSTANTS ---
PUNT_THR = 10.0
CHALK_THR = 30.0
MEGA_CHALK_THR = 40.0

DEFAULT_SALARY_CAP = 50000
DEFAULT_ROSTER_SIZE = 8
DEFAULT_STD_DEV_PCT = 0.20

# --- 1. CONTEST TEMPLATES ---

DK_NBA_TEMPLATES = {
    "ROSTER_SLOTS": {
        "PG": 1, 
        "SG": 1,
        "SF": 1,
        "PF": 1,
        "C": 1,
        "G": 1,  
        "F": 1, 
        "Util": 1
    },
    
    "OWNERSHIP_TARGETS": {
        "CASH": {
            "punt": (0, 3),
            "mid": (2, 6),
            "chalk": (2, 6),
            "mega": (0, 2)
        },
        "SE": {
            "punt": (1, 4),
            "mid": (2, 5),
            "chalk": (1, 4),
            "mega": (0, 2)
        },
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
) -> 'LineupTemplate':
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

def optimize_single_lineup(
    slate_df: pd.DataFrame, 
    template: 'LineupTemplate', 
    bucket_slack: int,
    locked_player_ids: List[str], 
    excluded_player_ids: List[str]
) -> Union[List[str], None]:
    """
    Finds the lineup that maximizes projection (or sampled projection) while 
    prioritizing template adherence. Returns a list of player_ids.
    """
    
    # Filter for playable players
    playable_df = slate_df[~slate_df['player_id'].isin(excluded_player_ids)].copy()
    
    # CRITICAL: Create mappings for safe, quick access (fixes potential multiline issues)
    proj_map = playable_df.set_index('player_id')['proj'].to_dict()
    salary_map = playable_df.set_index('player_id')['salary'].to_dict()
    
    # 1. Setup the Problem
    prob = LpProblem("DFS Lineup Optimization", LpMaximize)
    
    # Decision Variables
    player_vars = LpVariable.dicts("Player", playable_df['player_id'], 0, 1, LpBinary)
    
    # 2. Objective Function Components
    
    # A. Maximization: Total Projected Points (Use the safe map)
    total_projection = lpSum(proj_map[pid] * player_vars[pid] 
                             for pid in playable_df['player_id'])
    
    # B. Minimization: Penalty for violating Ownership Targets (Soft Constraint)
    own_ranges = template.bucket_ranges(slack=bucket_slack)
    penalty_sum = 0
    
    for bucket, (min_count, max_count) in own_ranges.items():
        bucket_players = playable_df[playable_df['bucket'] == bucket]['player_id'].tolist()
        
        if not bucket_players:
            continue
            
        count = lpSum(player_vars[pid] for pid in bucket_players)
        
        # Penalize if count is below minimum target
        under_var = LpVariable(f"Under_{bucket}", lowBound=0)
        prob += count + under_var >= min_count, f"{bucket} Min Soft"
        
        # Penalize if count is above maximum target
        over_var = LpVariable(f"Over_{bucket}", lowBound=0)
        prob += count - over_var <= max_count, f"{bucket} Max Soft"
        
        # Add penalty: Violating the target costs points (500 points)
        penalty_sum += 500 * under_var + 500 * over_var

    # 3. Final Objective Function: Maximize Projection - Penalties
    prob += total_projection - penalty_sum, "Net Score (Maximize Projection & Minimize Penalty)"
    
    
    # 4. Hard Constraints (Must always be met for a valid DK lineup)
    
    # A. Salary Cap Constraint (Use the safe map)
    prob += lpSum(salary_map[pid] * player_vars[pid] 
                  for pid in playable_df['player_id']) <= template.salary_cap, "Salary Cap"
                  
    # B. Roster Size Constraint
    prob += lpSum(player_vars[pid] for pid in playable_df['player_id']) == template.roster_size, "Roster Size"

    # C. Min Games Constraint
    game_ids = playable_df['GameID'].unique()
    game_vars = LpVariable.dicts("Game", game_ids, 0, 1, LpBinary)
    
    for gid in game_ids:
        game_players = playable_df[playable_df['GameID'] == gid]['player_id'].tolist()
        prob += lpSum(player_vars[pid] for pid in game_players) <= len(game_players) * game_vars[gid], f"Game Active {gid}"
