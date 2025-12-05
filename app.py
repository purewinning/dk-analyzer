# app.py

import pandas as pd
import numpy as np
import streamlit as st 
from typing import Dict, Any, List
from builder import (
    build_template_from_params, 
    build_optimal_lineup, 
    ownership_bucket,
    PUNT_THR, CHALK_THR, MEGA_CHALK_THR
) 

# --- CONFIGURATION CONSTANTS ---
SALARY_CAP = 50000
TOTAL_PLAYERS = 8
MIN_GAMES_REQUIRED = 2

# --- NEW: HEADER MAPPING ---
# Maps the column names from your uploaded file (KEY) to the internal names (VALUE)
HEADER_MAP = {
    'Player': 'Name',             # Used for display
    'Salary': 'salary',           # Used for salary constraint
    'Position': 'positions',      # Used for positional constraints
    'Projection': 'proj',         # Used for objective function (points)
    'Ownership %': 'own_proj',    # Used for ownership/leverage constraints
    # 'Team' and 'Opponent' will be used to create 'GameID'
}

# --- 1. DATA PREPARATION ---

def load_and_preprocess_data(uploaded_file=None) -> pd.DataFrame:
    """Loads uploaded CSV, renames headers, and processes data."""
    
    df = pd.DataFrame()
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Data loaded successfully from file.")

            # --- STEP 1: RENAME COLUMNS ---
            # Create a dictionary of present columns that need renaming
            rename_map = {
                old_name: new_name 
                for old_name, new_name in HEADER_MAP.items() 
                if old_name in df.columns
            }
            df.rename(columns=rename_map, inplace=True)
            
            # --- STEP 2: VALIDATE REQUIRED COLUMNS ---
            required_internal_cols = ['Name', 'salary', 'positions', 'proj', 'own_proj']
            if not all(col in df.columns for col in required_internal_cols):
                missing = [col for col in required_internal_cols if col not in df.columns]
                st.error(f"Missing one or more required columns after renaming. Missing: {missing}")
                return pd.DataFrame()

            # --- STEP 3: CREATE GAMEID (CRITICAL FIX) ---
            # If GameID is missing, use Team and Opponent to create a unique GameID string.
            if 'GameID' not in df.columns:
                if 'Team' in df.columns and 'Opponent' in df.columns:
                    # Create a consistent game identifier (e.g., 'LAL@BOS')
                    df['GameID'] = df.apply(
                        lambda row: '@'.join(sorted([row['Team'], row['Opponent']])), axis=1
                    )
                    st.info("ℹ️ Created **GameID** using Team and Opponent columns.")
                else:
                    st.error("Missing required column **GameID**. Cannot create game diversity constraint.")
                    return pd.DataFrame()
            
            # --- STEP 4: CLEANUP OWNERSHIP AND SALARY ---
            # Ensure ownership is between 0 and 1 (if uploaded as percent, divide by 100)
            if df['own_proj'].max() > 10: 
                 df['own_proj'] = df['own_proj'] / 100
                 st.info("ℹ️ Divided 'own_proj' by 100 (assuming % format).")

            df['player_id'] = df['Name'] # Use Name as the internal player_id for the optimizer
            df['salary'] = df['salary'].astype(int)
            df['proj'] = df['proj'].astype(float)
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return pd.DataFrame()
    else:
        # --- Placeholder Data Setup (Only used if no file is uploaded) ---
        # (Placeholder data creation remains as before for safety)
        data = {
            'player_id': [f'P{i}' for i in range(1, 15)],
            'Name': [f'Player {i}' for i in range(1, 15)],
            'positions': ['PG/SG', 'PG', 'SG', 'SF', 'PF/C', 'PF', 'C', 'PG/SF', 'SG/PF', 'C', 'PG', 'SF', 'PF', 'SG'],
            'salary': [6000, 7000, 5000, 8000, 4500, 4000, 9000, 5500, 6500, 4200, 7500, 8500, 4800, 5200],
            'proj': [35.5, 40.2, 30.1, 45.8, 25.0, 22.1, 50.3, 32.7, 38.0, 20.9, 42.0, 48.0, 28.0, 31.0],
            'own_proj': [0.45, 0.35, 0.15, 0.28, 0.05, 0.08, 0.40, 0.12, 0.20, 0.09, 0.33, 0.18, 0.04, 0.16], 
            'Team': ['LAL', 'LAL', 'BOS', 'BOS', 'MIL', 'MIL', 'PHX', 'PHX', 'DEN', 'DEN', 'LAL', 'BOS', 'MIL', 'PHX'],
            'GameID': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 2, 3, 4]
        }
        df = pd.DataFrame(data)
        st.info("ℹ️ Using placeholder data. Upload your CSV to analyze a real slate.")

    # Pre-calculate the ownership bucket for the leverage constraint
    df['bucket'] = df['own_proj'].apply(ownership_bucket)
    return df

# --- (The rest of app.py remains the same, including tab functions) ---
# ... (tab_lineup_builder, tab_contest_analyzer, and if __name__ == '__main__': block) ...
