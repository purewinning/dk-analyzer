import pulp
import pandas as pd
from typing import List, Dict, Any

# --- CONFIGURATION CONSTANTS ---
SALARY_CAP = 50000
TOTAL_PLAYERS = 8
MIN_GAMES_REQUIRED = 2 # DraftKings rule: Lineups must include players from at least 2 different games.

# --- NBA CLASSIC ROSTER SLOTS ---
# Defines the minimum number of players required for each basic position group.
# G (Guard) covers PG/SG, F (Forward) covers SF/PF.
ROSTER_REQUIREMENTS = {
    'PG': 1,  # Point Guard
    'SG': 1,  # Shooting Guard
    'SF': 1,  # Small Forward
    'PF': 1,  # Power Forward
    'C': 1,   # Center
    'G': 1,   # Guard (PG or SG)
    'F': 1,   # Forward (SF or PF)
    'UTIL': 1 # Utility (Any position - covered by TOTAL_PLAYERS constraint)
}

# --- 1. DATA PREPARATION ---

def load_and_preprocess_data(file_path: str = 'draftkings_projections.csv') -> pd.DataFrame:
    """
    Loads raw player data and processes it for the optimizer.
    
    NOTE: A critical enhancement is adding a 'GameID' or 'Team' column 
    for the diversity constraint.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}. Using placeholder data.")
        # Placeholder Data (Must include Name, Position, Salary, Projection, Team)
        data = {
            'Name': [f'P{i}' for i in range(1, 15)],
            'Position': ['PG/SG', 'PG', 'SG', 'SF', 'PF/C', 'PF', 'C', 'PG/SF', 'SG/PF', 'C', 'PG', 'SF', 'PF', 'SG'],
            'Salary': [6000, 7000, 5000, 8000, 4500, 4000, 9000, 5500, 6500, 4200, 7500, 8500, 4800, 5200],
            'Projection': [35.5, 40.2, 30.1, 45.8, 25.0, 22.1, 50.3, 32.7, 38.0, 20.9, 42.0, 48.0, 28.0, 31.0],
            'Team': ['LAL', 'LAL', 'BOS', 'BOS', 'MIL', 'MIL', 'PHX', 'PHX', 'DEN', 'DEN', 'LAL', 'BOS', 'MIL', 'PHX'],
            'GameID': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 2, 3, 4] # For diversity constraint
        }
        df = pd.DataFrame(data)

    # Ensure positions are string-formatted for parsing
    df['Position'] = df['Position'].astype(str)
    
    return df

# --- 2. OPTIMIZER CORE LOGIC ---

def optimize_lineup(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Solves the DraftKings lineup optimization problem using PuLP.
    """
    # 2.1. Prepare Data for LP Model
    player_ids = df['Name'].tolist()
    salaries = df.set_index('Name')['Salary'].to_dict()
    projections = df.set_index('Name')['Projection'].to_dict()
    
    # Extract positional eligibility for quick lookup
    def get_eligible_players(pos_key: str) -> List[str]:
        """Filters player IDs by position key ('PG', 'G', 'F', etc.)"""
        
        # Mapping to check player's 'Position' column against
        if pos_key == 'G': # Guard: PG or SG
            eligible_pos = ['PG', 'SG']
        elif pos_key == 'F': # Forward: SF or PF
            eligible_pos = ['SF', 'PF']
        else: # Single position: PG, SG, SF, PF, C
            eligible_pos = [pos_key]
        
        return df[
            df['Position'].apply(lambda x: any(p in x.split('/') for p in eligible_pos))
        ]['Name'].tolist()

    # 2.2. Initialize the Problem
    prob = pulp.LpProblem("DK_NBA_Lineup_Optimizer", pulp.LpMaximize)

    # 2.3. Define Decision Variables (Binary: 1 if selected, 0 otherwise)
    player_vars = pulp.LpVariable.dicts("Select", player_ids, 0, 1, pulp.LpBinary)

    # 2.4. Objective Function: Maximize Total Projected Points
    prob += (
        pulp.lpSum([projections[i] * player_vars[i] for i in player_ids]),
        "Maximize_Total_Points"
    )

    # 2.5. General Constraints
    
    # A. Total Lineup Size (Exactly 8 players)
    prob += (
        pulp.lpSum([player_vars[i] for i in player_ids]) == TOTAL_PLAYERS,
        "Exactly_8_Players"
    )

    # B. Salary Cap (<= $50,000)
    prob += (
        pulp.lpSum([salaries[i] * player_vars[i] for i in player_ids]) <= SALARY_CAP,
        "Salary_Cap"
    )
    
    # 2.6. Positional Constraints
    
    # The minimum required players for *each* position group must be met.
    # The 'Exactly_8_Players' constraint ensures the total number of slots 
    # (PG+SG+SF+PF+C+G+F+UTIL = 8) is filled without over-selecting.
    
    for pos, required in ROSTER_REQUIREMENTS.items():
        if pos == 'UTIL':
            continue # Covered by the TOTAL_PLAYERS constraint
            
        eligible_players = get_eligible_players(pos)
        
        # The sum of selected players who are eligible for this position must be >= the required count
        # This is the most accurate and flexible way to handle DraftKings' fluid positional eligibility (e.g., 'PG/SG')
        prob += (
            pulp.lpSum([player_vars[i] for i in eligible_players]) >= required,
            f"Min_{required}_Players_Eligible_for_{
