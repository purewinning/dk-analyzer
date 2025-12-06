# builder.py

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

    def bucket_ranges(self, slack: int = 0) -> Dict[str, Tuple[int, int]]:
        """Returns ownership ranges adjusted by slack."""
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

def find_single_optimal_lineup(
    slate_df: pd.DataFrame, 
    template: 'LineupTemplate', 
    bucket_slack: int,
    locked_player_ids: List[str], 
    excluded_player_ids: List[str],
    previous_lineup_vectors: List[Dict[str, float]] = None
) -> Union[Dict[str, Any], None]:
    """
    Finds the single lineup that maximizes projection while adhering to all 
    template and user constraints and excluding previous solutions.
    """
    
    # Filter for playable players
    playable_df = slate_df[~slate_df['player_id'].isin(excluded_player_ids)].copy()
    
    # CRITICAL: Create mappings for safe, quick access 
    proj_map = playable_df.set_index('player_id')['proj'].to_dict()
    salary_map = playable_df.set_index('player_id')['salary'].to_dict()
    all_player_ids = playable_df['player_id'].tolist()
    
    # 1. Setup the Problem
    prob = LpProblem("DFS Lineup Optimization", LpMaximize)
    
    # Decision Variables
    player_vars = LpVariable.dicts("Player", all_player_ids, 0, 1, LpBinary)
    
    # 2. Objective Function Components
    
    # A. Maximization: Total Projected Points 
    total_projection = lpSum(proj_map[pid] * player_vars[pid] 
                             for pid in all_player_ids)
    
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
    
    # A. Salary Cap Constraint 
    prob += lpSum(salary_map[pid] * player_vars[pid] 
                  for pid in all_player_ids) <= template.salary_cap, "Salary Cap"
                  
    # B. Roster Size Constraint
    prob += lpSum(player_vars[pid] for pid in all_player_ids) == template.roster_size, "Roster Size"

    # C. Min Games Constraint
    game_ids = playable_df['GameID'].unique()
    game_vars = LpVariable.dicts("Game", game_ids, 0, 1, LpBinary)
    
    for gid in game_ids:
        game_players = playable_df[playable_df['GameID'] == gid]['player_id'].tolist()
        prob += lpSum(player_vars[pid] for pid in game_players) <= len(game_players) * game_vars[gid], f"Game Active {gid}"
    
    prob += lpSum(game_vars[gid] for gid in game_ids) >= template.min_games, "Minimum Games"

    # D. Positional Constraints
    pos_map = template.pos_req
    
    pg_players = playable_df[playable_df['positions'].str.contains('PG')]['player_id'].tolist()
    sg_players = playable_df[playable_df['positions'].str.contains('SG')]['player_id'].tolist()
    sf_players = playable_df[playable_df['positions'].str.contains('SF')]['player_id'].tolist()
    pf_players = playable_df[playable_df['positions'].str.contains('PF')]['player_id'].tolist()
    c_players = playable_df[playable_df['positions'].str.contains('C')]['player_id'].tolist()

    prob += lpSum(player_vars[pid] for pid in pg_players) >= pos_map['PG'], "PG Min"
    prob += lpSum(player_vars[pid] for pid in sg_players) >= pos_map['SG'], "SG Min"
    prob += lpSum(player_vars[pid] for pid in sf_players) >= pos_map['SF'], "SF Min"
    prob += lpSum(player_vars[pid] for pid in pf_players) >= pos_map['PF'], "PF Min"
    prob += lpSum(player_vars[pid] for pid in c_players) >= pos_map['C'], "C Min"
    
    # Note: The DK position requirements are implicitly covered by the individual slot minimums, 
    # but the explicit G and F constraints ensure eligibility is met for utility positions.
    prob += lpSum(player_vars[pid] for pid in set(pg_players) | set(sg_players)) >= (pos_map['PG'] + pos_map['SG'] + pos_map['G']), "G Slot Fulfillment"
    prob += lpSum(player_vars[pid] for pid in set(sf_players) | set(pf_players)) >= (pos_map['SF'] + pos_map['PF'] + pos_map['F']), "F Slot Fulfillment"
    
    # E. Lock/Exclude Constraints
    for pid in locked_player_ids:
        if pid in player_vars:
            prob += player_vars[pid] == 1, f"Lock Player {pid}"
        
    # F. PREVIOUS SOLUTION EXCLUSION CONSTRAINT
    if previous_lineup_vectors:
        for i, prev_vector in enumerate(previous_lineup_vectors):
            # Sum of the players in the previous lineup must be LESS THAN the total roster size (8)
            # This forces at least one player to be different.
            prob += lpSum(player_vars[pid] for pid in all_player_ids if prev_vector.get(pid, 0) == 1) <= template.roster_size - 1, f"Exclude Lineup {i+1}"

    # 5. Solve the Problem
    prob.solve(PULP_CBC_CMD(msg=0)) 

    # 6. Process Results
    if prob.status == LpStatusOptimal:
        # Get the selected players for the current optimal lineup
        selected_players = [pid for pid in all_player_ids if player_vars[pid].varValue == 1]
        total_proj = sum(proj_map.get(pid, 0) for pid in selected_players)
        
        # Prepare the binary solution vector for the next iteration's exclusion constraint
        solution_vector = {pid: player_vars[pid].varValue for pid in all_player_ids}
        
        return {
            'player_ids': selected_players,
            'proj_score': total_proj,
            'solution_vector': solution_vector # Store the solution for exclusion
        }
    else:
        return None


def generate_top_n_lineups(
    slate_df: pd.DataFrame, 
    template: 'LineupTemplate', 
    n_lineups: int, 
    bucket_slack: int,
    locked_player_ids: List[str], 
    excluded_player_ids: List[str]
) -> List[Dict[str, Any]]:
    """Generates the top N optimal lineups sequentially."""
    
    all_lineups = []
    previous_solution_vectors = []
    
    for i in range(n_lineups):
        
        # Solve for the next best lineup, excluding all previous ones
        result = find_single_optimal_lineup(
            slate_df=slate_df,
            template=template,
            bucket_slack=bucket_slack,
            locked_player_ids=locked_player_ids,
            excluded_player_ids=excluded_player_ids,
            previous_lineup_vectors=previous_solution_vectors
        )
        
        if result is None:
            # No more feasible solutions found
            break
            
        # Store the lineup data (without the large solution vector)
        all_lineups.append({
            'player_ids': result['player_ids'],
            'proj_score': result['proj_score']
        })
        
        # Save the current optimal lineup's solution vector for the next exclusion constraint
        previous_solution_vectors.append(result['solution_vector'])
        
    return all_lineups
