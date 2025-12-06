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
DEFAULT_STD_DEV_PCT = 0.20 # 20% of projection for standard deviation in MCS

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
    
    # 1. Setup the Problem
    prob = LpProblem("DFS Lineup Optimization", LpMaximize)
    
    # Decision Variables
    player_vars = LpVariable.dicts("Player", playable_df['player_id'], 0, 1, LpBinary)
    
    # 2. Objective Function Components
    
    # A. Maximization: Total Projected Points (The 'proj' column here can be raw or sampled projection)
    total_projection = lpSum(playable_df.loc[playable_df['player_id'] == pid, 'proj'].iloc[0] * player_vars[pid] 
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
    
    # A. Salary Cap Constraint
    prob += lpSum(playable_df.loc[playable_df['player_id'] == pid, 'salary'].iloc[0] * player_vars[pid] 
                  for pid in playable_df['player_id']) <= template.salary_cap, "Salary Cap"
                  
    # B. Roster Size Constraint
    prob += lpSum(player_vars[pid] for pid in playable_df['player_id']) == template.roster_size, "Roster Size"

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
    
    prob += lpSum(player_vars[pid] for pid in set(pg_players) | set(sg_players)) >= (pos_map['PG'] + pos_map['SG'] + pos_map['G']), "G Slot Fulfillment"
    prob += lpSum(player_vars[pid] for pid in set(sf_players) | set(pf_players)) >= (pos_map['SF'] + pos_map['PF'] + pos_map['F']), "F Slot Fulfillment"
    
    # E. Lock/Exclude Constraints
    for pid in locked_player_ids:
        if pid in player_vars:
            prob += player_vars[pid] == 1, f"Lock Player {pid}"
        
    # 5. Solve the Problem
    prob.solve(PULP_CBC_CMD(msg=0)) # Suppress output

    # 6. Process Results
    if prob.status == LpStatusOptimal:
        selected_players = [pid for pid in playable_df['player_id'] if player_vars[pid].varValue == 1]
        
        # Return only the list of player IDs (for efficient MCS collection)
        return selected_players
    else:
        return None

def run_monte_carlo_simulations(
    slate_df: pd.DataFrame, 
    template: 'LineupTemplate', 
    num_iterations: int,
    max_exposures: Dict[str, float],
    bucket_slack: int,
    locked_player_ids: List[str], 
    excluded_player_ids: List[str],
    min_lineup_diversity: int = 4
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Runs Monte Carlo simulations, collects optimal lineups, and applies 
    Max Exposure and Lineup Diversity constraints.
    Returns: (list of final lineups (dicts), dict of player exposures)
    """
    
    raw_optimal_lineups = []
    sim_df = slate_df.copy()
    
    # --- CRITICAL FIX: ENSURE ALL NUMERICAL COLUMNS ARE FLOAT ---
    # Convert core projection and salary columns to float64 for NumPy compatibility
    try:
        # Use .values.astype to get a pure NumPy array first, then put back into DataFrame
        sim_df['proj'] = sim_df['proj'].values.astype(np.float64) 
        sim_df['salary'] = sim_df['salary'].values.astype(np.float64)
    except Exception as e:
        print(f"ERROR: Final float conversion failed in builder.py: {e}") 
        return [], {}
    
    # 1. PRE-CALCULATE STATISTICAL VARIANCE
    sim_df['std_dev'] = sim_df['proj'] * DEFAULT_STD_DEV_PCT
    
    # Ensure players with 0 salary/proj/std_dev are safe
    sim_df.loc[sim_df['std_dev'] <= 0, 'std_dev'] = 0.1
    sim_df.loc[sim_df['proj'] <= 0, 'proj'] = 0.1
    
    # 2. RUN SIMULATIONS
    for i in range(num_iterations):
        
        # Sample new projections N(mu, sigma)
        sim_df['sampled_proj'] = np.random.normal(
            loc=sim_df['proj'], 
            scale=sim_df['std_dev']
        ).clip(lower=0.1)
        
        # Create a temporary DF with sampled proj as the primary 'proj' column
        temp_df = sim_df.rename(columns={'sampled_proj': 'proj'})

        lineup_ids = optimize_single_lineup(
            slate_df=temp_df,
            template=template,
            bucket_slack=bucket_slack,
            locked_player_ids=locked_player_ids,
            excluded_player_ids=excluded_player_ids
        )
        
        if lineup_ids:
            # Append the lineup IDs and the total projected score for this sim
            lineup_proj = temp_df[temp_df['player_id'].isin(lineup_ids)]['proj'].sum()
            
            raw_optimal_lineups.append({
                'player_ids': lineup_ids,
                'proj_score': lineup_proj
            })

    # --- 3. POST-SIMULATION ANALYSIS & FILTERING ---
    
    if not raw_optimal_lineups:
        return [], {}

    # Sort lineups by projection (highest projected lineups will be preferred)
    raw_optimal_lineups.sort(key=lambda x: x['proj_score'], reverse=True)
    
    final_lineups = []
    
    # Tally counts for exposure tracking
    player_counts = {pid: 0 for pid in slate_df['player_id']}
    
    # --- Filtered Output Set Size (Max 100 lineups for display) ---
    max_output_lineups = min(len(raw_optimal_lineups), 100) 
    
    
    for lineup in raw_optimal_lineups:
        lineup_ids = lineup['player_ids']
        
        # A. Check Max Exposure Constraint
        violates_exposure = False
        for pid in lineup_ids:
            # Calculate current projected exposure with this lineup included
            current_total = len(final_lineups)
            current_count = player_counts.get(pid, 0)
            
            max_pct = max_exposures.get(pid, 1.0) # Default Max Exposure is 1.0 (100%)
            
            # Use a tiny buffer to avoid division by zero or floating point issues on the first iteration
            if (current_count + 1) / (current_total + 1e-9) > max_pct:
                violates_exposure = True
                break
        
        if violates_exposure:
            continue

        # B. Check Lineup Diversity Constraint (min_lineup_diversity = max shared players)
        is_diverse = True
        for existing_lineup in final_lineups:
            shared_count = len(set(lineup_ids) & set(existing_lineup['player_ids']))
            if shared_count > min_lineup_diversity:
                is_diverse = False
                break
        
        if is_diverse:
            final_lineups.append(lineup)
            
            # Update player counts for exposure tracking
            for pid in lineup_ids:
                player_counts[pid] = player_counts.get(pid, 0) + 1
            
            if len(final_lineups) >= max_output_lineups:
                break
    
    # Calculate final exposures based on the filtered set
    total_lineups_count = len(final_lineups)
    
    final_exposures = {
        pid: (count / total_lineups_count) * 100 if total_lineups_count > 0 else 0
        for pid, count in player_counts.items() 
    }
    
    return final_lineups, final_exposures
