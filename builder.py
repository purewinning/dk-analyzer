# builder.py (Partial - showing the core changes)
# ... (Imports and Configuration Constants remain the same) ...

# --- 3. CORE LOGIC ---

def ownership_bucket(own_proj: float) -> str:
# ... (Function remains the same) ...
    pass

def find_single_optimal_lineup(
    slate_df: pd.DataFrame, 
    template: 'LineupTemplate', 
    bucket_slack: int,
    locked_player_ids: List[str], 
    excluded_player_ids: List[str],
    previous_lineup_vars: List[LpVariable] = None # NEW ARGUMENT
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
    
    # 2. Objective Function (Maximization - Penalties)
    # ... (Objective function definition remains the same) ...

    # 3. Final Objective Function: Maximize Projection - Penalties
    prob += total_projection - penalty_sum, "Net Score (Maximize Projection & Minimize Penalty)"
    
    
    # 4. Hard Constraints (Must always be met for a valid DK lineup)
    # ... (All previous constraints remain the same: Salary Cap, Roster Size, Min Games, Positional, Lock/Exclude) ...

    # ** E. NEW: PREVIOUS SOLUTION EXCLUSION CONSTRAINT **
    # This constraint ensures the current solution is different from previous ones
    if previous_lineup_vars:
        for i, prev_lineup in enumerate(previous_lineup_vars):
            # Sum of the players in the previous lineup must be LESS THAN the total roster size (8)
            # This forces at least one player to be different.
            # Roster Size is 8. Sum(PlayerVars * Old_Solution_Value) <= 7
            prob += lpSum(player_vars[pid] for pid in all_player_ids if prev_lineup[pid] == 1) <= template.roster_size - 1, f"Exclude Lineup {i+1}"

    # 5. Solve the Problem
    prob.solve(PULP_CBC_CMD(msg=0)) 

    # 6. Process Results
    if prob.status == LpStatusOptimal:
        # Get the selected players for the current optimal lineup
        selected_players = [pid for pid in all_player_ids if player_vars[pid].varValue == 1]
        total_proj = sum(proj_map[pid] for pid in selected_players)
        
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
            previous_lineup_vars=previous_solution_vectors
        )
        
        if result is None:
            # No more feasible solutions found
            break
            
        all_lineups.append(result)
        
        # Save the current optimal lineup's solution vector for the next exclusion constraint
        previous_solution_vectors.append(result['solution_vector'])
        
    return all_lineups
