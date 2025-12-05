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

# --- CONFIGURATION CONSTANTS (Keep these consistent) ---
SALARY_CAP = 50000
TOTAL_PLAYERS = 8
MIN_GAMES_REQUIRED = 2

# --- NEW: HEADER MAPPING (Only map Player for display and internal player_id) ---
# We are aligning internal names, so we only need to map 'Player' to 'Name'
HEADER_MAP = {
    'Player': 'Name',             
}

# --- 1. DATA PREPARATION ---

def load_and_preprocess_data(uploaded_file=None) -> pd.DataFrame:
    """Loads uploaded CSV, renames 'Player' to 'Name', and processes data."""
    
    df = pd.DataFrame()
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Data loaded successfully from file.")

            # --- STEP 1: RENAME 'Player' to 'Name' for consistency ---
            if 'Player' in df.columns:
                df.rename(columns={'Player': 'Name'}, inplace=True)
            
            # --- STEP 2: VALIDATE REQUIRED COLUMNS (Using your CSV names) ---
            # NOTE: The *internal* builder code still expects 'salary', 'proj', and 'own_proj' 
            # We must map your CSV names to these internal names.
            
            REQUIRED_CSV_TO_INTERNAL_MAP = {
                'Salary': 'salary',
                'Position': 'positions',
                'Projection': 'proj',
                'Ownership %': 'own_proj',
            }
            
            # Check for missing columns
            missing_csv_cols = [col for col in REQUIRED_CSV_TO_INTERNAL_MAP.keys() if col not in df.columns]
            if missing_csv_cols:
                st.error(f"Missing one or more required columns. Need: {list(REQUIRED_CSV_TO_INTERNAL_MAP.keys())}. Missing: {missing_csv_cols}")
                return pd.DataFrame()

            # Perform the internal renaming for the optimizer logic
            df.rename(columns=REQUIRED_CSV_TO_INTERNAL_MAP, inplace=True)
            
            # --- STEP 3: CREATE GAMEID (CRITICAL FIX) ---
            required_game_cols = ['Team', 'Opponent']
            if 'GameID' not in df.columns:
                if all(col in df.columns for col in required_game_cols):
                    df['GameID'] = df.apply(
                        lambda row: '@'.join(sorted([str(row['Team']), str(row['Opponent'])])), axis=1
                    )
                    st.info("ℹ️ Created **GameID** using Team and Opponent columns.")
                else:
                    st.error("Missing required column **GameID**. Cannot create game diversity constraint.")
                    return pd.DataFrame()
            
            # --- STEP 4: CLEANUP DATA TYPES ---
            df['player_id'] = df['Name'] # Use Name as the internal player_id
            
            # Ensure ownership is between 0 and 1
            if df['own_proj'].max() > 10: 
                 df['own_proj'] = df['own_proj'] / 100
                 st.info("ℹ️ Divided 'own_proj' by 100 (assuming % format).")

            df['salary'] = df['salary'].astype(int)
            df['proj'] = df['proj'].astype(float)
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return pd.DataFrame()
    else:
        # --- Placeholder Data Setup (Only used if no file is uploaded) ---
        data = {
            'player_id': [f'P{i}' for i in range(1, 15)],
            'Name': [f'Player {i}' for i in range(1, 15)],
            'positions': ['PG/SG', 'PG', 'SG', 'SF', 'PF/C', 'PF', 'C', 'PG/SF', 'SG/PF', 'C', 'PG', 'SF', 'PF', 'SG'],
            'salary': [6000, 7000, 5000, 8000, 4500, 4000, 9000, 5500, 6500, 4200, 7500, 8500, 4800, 5200],
            'proj': [35.5, 40.2, 30.1, 45.8, 25.0, 22.1, 50.3, 32.7, 38.0, 20.9, 42.0, 48.0, 28.0, 31.0],
            'own_proj': [0.45, 0.35, 0.15, 0.28, 0.05, 0.08, 0.40, 0.12, 0.20, 0.09, 0.33, 0.18, 0.04, 0.16], 
            'Team': ['LAL', 'LAL', 'BOS', 'BOS', 'MIL', 'MIL', 'PHX', 'PHX', 'DEN', 'DEN', 'LAL', 'BOS', 'MIL', 'PHX'],
            'Opponent': ['BOS', 'BOS', 'LAL', 'LAL', 'DEN', 'DEN', 'MIL', 'MIL', 'PHX', 'PHX', 'BOS', 'LAL', 'DEN', 'MIL'],
            'GameID': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 2, 3, 4]
        }
        df = pd.DataFrame(data)
        st.info("ℹ️ Using placeholder data. Upload your CSV to analyze a real slate.")

    # Pre-calculate the ownership bucket for the leverage constraint
    df['bucket'] = df['own_proj'].apply(ownership_bucket)
    return df

# --- (The rest of app.py remains the same: tab functions, main execution) ---
