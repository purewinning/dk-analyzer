# app.py

import pandas as pd
import numpy as np
import streamlit as st # Crucial for Streamlit apps
from typing import Dict, Any, List
# Import the core logic functions and classes from builder.py
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
        # --- Placeholder Data Setup for demonstration purposes ---
        # Data must have columns matching those used in builder.py
        data = {
            'player_id': [f'P{i}' for i in range(1, 15)],
            'positions': ['PG/SG', 'PG', 'SG', 'SF', 'PF/C', 'PF', 'C', 'PG/SF', 'SG/PF', 'C', 'PG', 'SF', 'PF', 'SG'],
            'salary': [6000, 7000, 5000, 8000, 4500, 4000, 9000, 5500, 6500, 4200, 7500, 8500, 4800, 5200],
            'proj': [35.5, 40.2, 30.1, 45.8, 25.0, 22.1, 50.3, 32.7, 38.0, 20.9, 42.0, 48.0, 28.0, 31.0],
            'own_proj': [0.45, 0.35, 0.15, 0.28, 0.05, 0.08, 0.40, 0.12, 0.20, 0.09, 0.33, 0.18, 0.04, 0.16], 
            'Team': ['LAL', 'LAL', 'BOS', 'BOS', 'MIL', 'MIL', 'PHX', 'PHX', 'DEN', 'DEN', 'LAL', 'BOS', 'MIL', 'PHX'],
            'GameID': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 2, 3, 4]
        }
        df = pd.DataFrame(data)
        st.write("✅ Using placeholder data. Remember to load your CSV.")
        # --- End Placeholder Setup ---

    except Exception as e:
        st.error(f"Error loading data: {e}. Cannot proceed.")
        return pd.DataFrame()

    # Pre-calculate the ownership bucket for the leverage constraint
    df['bucket'] = df['own_proj'].apply(ownership_bucket)
    
    return df

# --- 2. EXECUTION AND DISPLAY ---

def run_app():
    # 0. Initialize Streamlit UI
    st.title("DraftKings NBA Lineup Optimizer 🏀")
    st.markdown("---")

    # 1. Load Data
    slate_df = load_and_preprocess_data()
    if slate_df.empty:
        return # Stop execution if data failed to load
        
    st.header("1. Contest Setup")

    # 2. Define the Target Contest Structure (Example: Single Entry, Top Heavy GPP)
    template = build_template_from_params(
        contest_type="SE", 
        field_size=10000, 
        pct_to_first=30.0,
        roster_size=TOTAL_PLAYERS,
        salary_cap=SALARY_CAP,
        min_games=MIN_GAMES_REQUIRED
    )
    
    st.info(f"🎯 Using Template: **{template.contest_label}** | Target Ownership Breakdown: {template.bucket_ranges(slack=1)}")
    
    st.header("2. Optimization Results")

    # 3. Run Optimization (Classic NBA)
    optimal_lineup_df = build_optimal_lineup(
        slate_df=slate_df,
        template=template,
        bucket_slack=1,
    )
    
    # 4. Process and Display Results
    if optimal_lineup_df is not None:
        
        # Calculate summary metrics
        total_salary = optimal_lineup_df['salary'].sum()
        total_points = optimal_lineup_df['proj'].sum()
        games_used = optimal_lineup_df['GameID'].nunique()
        
        st.subheader("🏆 Optimal Lineup Found")
        
        # Display the Lineup in a clean DataFrame format
        display_cols = ['player_id', 'positions', 'Team', 'GameID', 'salary', 'proj', 'own_proj', 'bucket']
        lineup_df_display = optimal_lineup_df[display_cols
