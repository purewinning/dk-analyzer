# app.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List
# Import the core logic functions and classes
from builder import build_template_from_params, build_optimal_lineup, ownership_bucket 

# --- CONFIGURATION CONSTANTS (Keep these consistent with builder.py if changed) ---
SALARY_CAP = 50000
TOTAL_PLAYERS = 8
MIN_GAMES_REQUIRED = 2

# --- 1. DATA PREPARATION ---

def load_and_preprocess_data(file_path: str = 'draftkings_projections.csv') -> pd.DataFrame:
    """
    Loads raw player data and processes it for the optimizer.
    
    A crucial step is ensuring 'player_id', 'positions', 'salary', 'proj', 
    'own_proj', and 'GameID' columns exist and are correctly typed.
    """
    try:
        # Load your actual data here
        # df = pd.read_csv(file_path)
        
        # --- Placeholder Data Setup for demonstration purposes ---
        # Data must have columns matching those used in builder.py
        data = {
            'player_id': [f'P{i}' for i in range(1, 15)],
            'positions': ['PG/SG', 'PG', 'SG', 'SF', 'PF/C', 'PF', 'C', 'PG/SF', 'SG/PF', 'C', 'PG', 'SF', 'PF', 'SG'],
            'salary': [6000, 7000, 5000, 8000, 4500, 4000, 9000, 5500, 6500, 4200, 7500, 8500, 4800, 5200],
            'proj': [35.5, 40.2, 30.1, 45.8, 25.0, 22.1, 50.3, 32.7, 38.0, 20.9, 42.0, 48.0, 28.0, 31.0],
            'own_proj': [0.45, 0.35, 0.15, 0.28, 0.05, 0.08, 0.40, 0.12, 0.20, 0.09, 0.33, 0.18, 0.04, 0.16], # Ownership % (as float)
            'Team': ['LAL', 'LAL', 'BOS', 'BOS', 'MIL', 'MIL', 'PHX', 'PHX', 'DEN', 'DEN', 'LAL', 'BOS', 'MIL', 'PHX'],
            'GameID': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 2, 3, 4] # Game ID is essential for diversity constraint
        }
        df = pd.DataFrame(data)
        print("✅ Using placeholder data. Remember to load your CSV.")
        # --- End Placeholder Setup ---

    except Exception as e:
        print(f"Error loading data: {e}. Cannot proceed.")
        return pd.DataFrame()

    # Pre-calculate the ownership bucket for the leverage constraint
    df['bucket'] = df['own_proj'].apply(ownership_bucket)
    
    return df

# --- 2. EXECUTION AND DISPLAY ---

if __name__ == '__main__':
    # 1. Load Data
    slate_df = load_and_preprocess_data()
    if slate_df.empty:
        exit()
        
    print("✅ Data Loaded. Building Contest Template...")

    # 2. Define the Target Contest Structure (Example: Single Entry, Top Heavy GPP)
    template = build_template_from_params(
        contest_type="SE", 
        field_size=10000, 
        pct_to_first=30.0,
        roster_size=TOTAL_PLAYERS,
        salary_cap=SALARY_CAP,
        min_games=MIN_GAMES_REQUIRED
    )
    
    print(f"🎯 Using Template: **{template.contest_label}**")
    print(f"Required Ownership Breakdown: {template.bucket_ranges(slack=1)}")
    
    print("\n--- Running Optimizer ---")

    # 3. Run Optimization (Classic NBA)
    optimal_lineup_df = build_optimal_lineup(
        slate_df=slate_df,
        template=template,
        bucket_slack=1,
        # Example: Exclude a high-owned player if you don't trust their projection
        # avoid_player_ids=['P1'] 
    )

    print("\n" + "="*50)
    
    # 4. Process and Display Results
    if optimal_lineup_df is not None:
        
        # Calculate summary metrics - Lines 91-93 are critical here
        total_salary = optimal_lineup_df['salary'].sum()
        total_points = optimal_lineup_df['proj'].sum()  # <--- THIS IS THE CORRECTED LINE
        games_used = optimal_lineup_df['GameID'].nunique()
        
        print("### 🏆 Optimal DraftKings NBA Lineup Found ###")
        
        # Display the Lineup in a clean DataFrame format
        display_cols = ['player_id', 'positions', 'Team', 'GameID', 'salary', 'proj', 'own_proj', 'bucket']
        lineup_df_display = optimal_lineup_df[display_cols].sort_values(by='proj', ascending=False).reset_index(drop=True)
        
        print(lineup_df_display.to_markdown(index=False, floatfmt=".2f"))
        
        print("\n--- Lineup Summary ---")
        print(f"Total Projected Points: **{total_points:.2f}**")
        print(f"Total Salary Used: **${total_salary:,}**")
        print(f"Games Represented: **{games_used}** (Required: {MIN_GAMES_REQUIRED})")
        print(f"Lineup Structure: \n{lineup_df_display['bucket'].value_counts().to_string()}")

    else:
        print("❌ Could not find an optimal solution. Check constraints and player pool.")
