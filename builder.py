# builder.py

import pandas as pd
import numpy as np
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
    # Roster slots must match the image provided by the user.
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
    prob += lpSum(playable_df.loc[playable_df['player_id'] == pid, 'proj'].iloc[0] * player_vars[pid] 
                  for pid in playable_df['player_id']), "Total Projection"
    
    # 3. Constraints
    
    # A. Salary Cap Constraint: Total salary must be <= salary cap
    prob += lpSum(playable_df.loc[playable_df['player_id'] == pid, 'salary'].iloc[0] * player_vars[pid] 
                  for pid in playable_df['player_id']) <= template.salary_cap, "Salary Cap"
                  
    # B. Roster Size Constraint: Total number of players must equal roster size
    prob += lpSum(player_vars[pid] for pid in playable_df['player_id']) == template.roster_size, "Roster Size"

    # C. Min Games Constraint: Players must be from at least 2 different games
    game_ids = playable_df['GameID'].unique()
    game_vars = LpVariable.dicts("Game", game_ids, 0, 1, LpBinary)
    
    # Map players to their games
    for gid in game_ids:
        game_players = playable_df[playable_df['GameID'] == gid]['player_id'].tolist()
        # If any player from a game is chosen, the game variable is set to 1
        prob += lpSum(player_vars[pid] for pid in game_players) <= len(game_players) * game_vars[gid], f"Game Active {gid}"
    
    # Total active games must be at least the minimum required
    prob += lpSum(game_vars[gid] for gid in game_ids) >= template.min_games, "Minimum Games"

    # D. Positional Constraints (The key fix!)
    pos_map = template.pos_req
    
    # Mapping of player position to DraftKings slots
    
    # PG Slot (Must be PG)
    prob += lpSum(player_vars[pid] for pid in playable_df[playable_df['positions'].str.contains('PG')]['player_id']) >= pos_map['PG'], "PG Slot Min"
    
    # SG Slot (Must be SG)
    prob += lpSum(player_vars[pid] for pid in playable_df[playable_df['positions'].str.contains('SG')]['player_id']) >= pos_map['SG'], "SG Slot Min"

    # SF Slot (Must be SF)
    prob += lpSum(player_vars[pid] for pid in playable_df[playable_df['positions'].str.contains('SF')]['player_id']) >= pos_map['SF'], "SF Slot Min"
    
    # PF Slot (Must be PF)
    prob += lpSum(player_vars[pid] for pid in playable_df[playable_df['positions'].str.contains('PF')]['player_id']) >= pos_map['PF'], "PF Slot Min"
    
    # C Slot (Must be C)
    prob += lpSum(player_vars[pid] for pid in playable_df[playable_df['positions'].str.contains('C')]['player_id']) >= pos_map['C'], "C Slot Min"

    # G Slot (Total PG + SG must be at least the sum of PG, SG, and G slots)
    total_g_slots = pos_map['PG'] + pos_map['SG'] + pos_map['G']
    prob += lpSum(player_vars[pid] for pid in playable_df[playable_df['positions'].str.contains('PG|SG')]['player_id']) >= total_g_slots, "Total G Slots Min"

    # F Slot (Total SF + PF must be at least the sum of SF, PF, and F slots)
    total_f_slots = pos_map['SF'] + pos_map['PF'] + pos_map['F']
    prob += lpSum(player_vars[pid] for pid in playable_df[playable_df['positions'].str.contains('SF|PF')]['player_id']) >= total_f_slots, "Total F Slots Min"

    # Total Utility (Total players must be equal to the roster size)
    # This is covered by the Roster Size constraint, but we add a total constraint for verification
    prob += lpSum(player_vars[pid] for pid in playable_df['player_id']) == template.roster_size, "Total Players Check"
    

    # E. Lock/Exclude Constraints
    for pid in locked_player_ids:
        if pid in player_vars:
            prob += player_vars[pid] == 1, f"Lock Player {pid}"
        
    # F. Ownership Constraints (Based on Contest Strategy)
    own_ranges = template.bucket_ranges(slack=bucket_slack)
    
    for bucket, (min_count, max_count) in own_ranges.items():
        bucket_players = playable_df[playable_df['bucket'] == bucket]['player_id'].tolist()
        
        if bucket_players:
            # Min constraint
            prob += lpSum(player_vars[pid] for pid in bucket_players) >= min_count, f"{bucket} Min Count"
            # Max constraint
            prob += lpSum(player_vars[pid] for pid in bucket_players) <= max_count, f"{bucket} Max Count"
    
    # 4. Solve the Problem
    prob.solve()
    
    # 5. Process Results
    if prob.status == LpStatusOptimal:
        selected_players = [pid for pid in playable_df['player_id'] if player_vars[pid].varValue == 1]
        return slate_df[slate_df['player_id'].isin(selected_players)]
    else:
        # Check for Infeasible status
        if prob.status == LpStatusInfeasible:
            print("Solver Status: Infeasible. Constraints cannot be met.")
        else:
            print(f"Solver Status: {LpStatus[prob.status]}.")
        return None
