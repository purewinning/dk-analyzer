# builder.py (Replace the existing function with this one)

# builder.py (Lines 1-5 should look like this)

import pandas as pd  # <-- CRITICAL MISSING IMPORT
from pulp import *
from typing import Dict, List, Tuple, Union

def build_optimal_lineup(
    slate_df: pd.DataFrame, 
    template: LineupTemplate, 
    bucket_slack: int,
    locked_player_ids: List[str], 
    excluded_player_ids: List[str]
) -> Union[pd.DataFrame, None]:
    """
    Finds the lineup that maximizes projection while prioritizing template adherence.
    Soft constraints are used for ownership targets to ensure a lineup is always found.
    """
    
    # Filter for playable players
    playable_df = slate_df[~slate_df['player_id'].isin(excluded_player_ids)].copy()
    
    # 1. Setup the Problem
    prob = LpProblem("DFS Lineup Optimization", LpMaximize)
    
    # Decision Variables
    player_vars = LpVariable.dicts("Player", playable_df['player_id'], 0, 1, LpBinary)
    
    # 2. Objective Function Components
    
    # A. Maximization: Total Projected Points
    total_projection = lpSum(playable_df.loc[playable_df['player_id'] == pid, 'proj'].iloc[0] * player_vars[pid] 
                             for pid in playable_df['player_id'])
    
    # B. Minimization: Penalty for violating Ownership Targets (Soft Constraint)
    # Define variables and constraints for soft ownership limits
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
        
        # Add penalty: Violating the target costs points (e.g., 50 points per violation)
        # The penalty value (e.g., 50) must be significantly larger than any single player's projection.
        # We use a large coefficient to ensure adherence is prioritized over marginal projection gains.
        penalty_sum += 500 * under_var + 500 * over_var

    # 3. Final Objective Function: Maximize Projection - Penalties
    prob += total_projection - penalty_sum, "Net Score (Maximize Projection & Minimize Penalty)"
    
    
    # 4. Hard Constraints (Must always be met for a valid DK lineup)
    
    # A. Salary Cap Constraint
    prob += lpSum(playable_df.loc[playable_df['player_id'] == pid, 'salary'].iloc[0] * player_vars[pid] 
                  for pid in playable_df['player_id']) <= template.salary_cap, "Salary Cap"
                  
    # B. Roster Size Constraint
    prob += lpSum(player_vars[pid] for pid in playable_df['player_id']) == template.roster_size, "Roster Size"

    # C. Min Games Constraint (Leaving this as a hard constraint for integrity)
    game_ids = playable_df['GameID'].unique()
    game_vars = LpVariable.dicts("Game", game_ids, 0, 1, LpBinary)
    
    for gid in game_ids:
        game_players = playable_df[playable_df['GameID'] == gid]['player_id'].tolist()
        prob += lpSum(player_vars[pid] for pid in game_players) <= len(game_players) * game_vars[gid], f"Game Active {gid}"
    
    prob += lpSum(game_vars[gid] for gid in game_ids) >= template.min_games, "Minimum Games"

    # D. Positional Constraints (The Verified Logic)
    pos_map = template.pos_req
    
    pg_players = playable_df[playable_df['positions'].str.contains('PG')]['player_id'].tolist()
    sg_players = playable_df[playable_df['positions'].str.contains('SG')]['player_id'].tolist()
    sf_players = playable_df[playable_df['positions'].str.contains('SF')]['player_id'].tolist()
    pf_players = playable_df[playable_df['positions'].str.contains('PF')]['player_id'].tolist()
    c_players = playable_df[playable_df['positions'].str.contains('C')]['player_id'].tolist()

    # Hard Slot Requirements
    prob += lpSum(player_vars[pid] for pid in pg_players) >= pos_map['PG'], "PG Min"
    prob += lpSum(player_vars[pid] for pid in sg_players) >= pos_map['SG'], "SG Min"
    prob += lpSum(player_vars[pid] for pid in sf_players) >= pos_map['SF'], "SF Min"
    prob += lpSum(player_vars[pid] for pid in pf_players) >= pos_map['PF'], "PF Min"
    prob += lpSum(player_vars[pid] for pid in c_players) >= pos_map['C'], "C Min"
    
    # Flexible Slots (G & F)
    prob += lpSum(player_vars[pid] for pid in set(pg_players) | set(sg_players)) >= (pos_map['PG'] + pos_map['SG'] + pos_map['G']), "G Slot Fulfillment"
    prob += lpSum(player_vars[pid] for pid in set(sf_players) | set(pf_players)) >= (pos_map['SF'] + pos_map['PF'] + pos_map['F']), "F Slot Fulfillment"
    
    # E. Lock/Exclude Constraints
    for pid in locked_player_ids:
        if pid in player_vars:
            prob += player_vars[pid] == 1, f"Lock Player {pid}"
        
    # 5. Solve the Problem
    prob.solve()
    
    # 6. Process Results
    if prob.status == LpStatusOptimal:
        selected_players = [pid for pid in playable_df['player_id'] if player_vars[pid].varValue == 1]
        
        # Calculate the penalty incurred to show the user how much the template was violated
        # Note: If penalty_sum is > 0, the lineup is VALID but non-IDEAL.
        final_penalty = value(penalty_sum)
        
        # Display a warning if a soft constraint was broken
        if final_penalty > 0.0:
            print(f"⚠️ Optimal lineup found, but incurred a template penalty of {final_penalty / 500:.0f} violation(s) to maximize projection.")
            
        return slate_df[slate_df['player_id'].isin(selected_players)]
    else:
        # If the problem is still infeasible, it means a HARD constraint failed (Salary, Roster Size, Positions, or Min Games).
        print("❌ CRITICAL FAILURE: Could not find a valid lineup. Check hard constraints (Salary Cap, Roster Size, Positional Minimums, Min Games).")
        return None
