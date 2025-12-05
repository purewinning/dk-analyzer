# app.py

import pandas as pd 
import numpy as np
import streamlit as st 
from typing import Dict, Any, List
from builder import (
    build_template_from_params, 
    build_optimal_lineup, 
    ownership_bucket,
    PUNT_THR, CHALK_THR, MEGA_CHALK_THR,
    DEFAULT_SALARY_CAP, DEFAULT_ROSTER_SIZE
) 

# --- CONFIGURATION CONSTANTS ---
MIN_GAMES_REQUIRED = 2

# --- HEADER MAPPING (For your CSV format) ---
REQUIRED_CSV_TO_INTERNAL_MAP = {
    'Salary': 'salary',
    'Position': 'positions',
    'Projection': 'proj',
    'Ownership %': 'own_proj',
}


# --- 1. DATA PREPARATION ---

def load_and_preprocess_data(uploaded_file=None) -> pd.DataFrame:
    """Loads uploaded CSV, renames headers, and processes data."""
    
    df = pd.DataFrame()
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Data loaded successfully from uploaded file.")

            # --- STEP 1: RENAME COLUMNS ---
            if 'Player' in df.columns:
                df.rename(columns={'Player': 'Name'}, inplace=True)
            
            # Perform internal renaming for the optimizer logic
            missing_csv_cols = [col for col in REQUIRED_CSV_TO_INTERNAL_MAP.keys() if col not in df.columns]
            if missing_csv_cols:
                st.error(f"Missing required columns. Please check your CSV headers. Missing: {missing_csv_cols}")
                return pd.DataFrame()
            
            df.rename(columns=REQUIRED_CSV_TO_INTERNAL_MAP, inplace=True)
            
            # --- STEP 2: CREATE GAMEID ---
            required_game_cols = ['Team', 'Opponent']
            if 'GameID' not in df.columns:
                if all(col in df.columns for col in required_game_cols):
                    df['GameID'] = df.apply(
                        lambda row: '@'.join(sorted([str(row['Team']), str(row['Opponent'])])), axis=1
                    )
                    st.info("ℹ️ Created **GameID** using Team and Opponent columns.")
                else:
                    st.error("Missing required columns: **Team** and **Opponent** (or **GameID**).")
                    return pd.DataFrame()
            
            # --- STEP 3: CLEANUP DATA TYPES ---
            df['player_id'] = df['Name'] 
            
            # CRITICAL FIX: Convert ownership projection to numeric, handling errors
            initial_len = len(df)
            df['own_proj'] = pd.to_numeric(df['own_proj'], errors='coerce')
            
            # Drop any rows where own_proj is now invalid (NaN)
            df.dropna(subset=['own_proj'], inplace=True)
            dropped_len = initial_len - len(df)
            
            if dropped_len > 0:
                 st.warning(f"⚠️ Dropped {dropped_len} player(s) due to invalid 'Ownership %' data.")

            # Ensure ownership is between 0 and 1
            if len(df) > 0 and df['own_proj'].max() > 10: 
                 df['own_proj'] = df['own_proj'] / 100
                 st.info("ℹ️ Divided 'own_proj' by 100 (assuming % format).")
                 
            # Final type conversions
            df['salary'] = df['salary'].astype(int)
            df['proj'] = df['proj'].astype(float)

            if len(df) == 0:
                 st.error("❌ Final player pool is empty after cleaning.")
                 return pd.DataFrame()
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return pd.DataFrame()
    else:
        # --- Placeholder Data Setup (Fallback) ---
        data = {
            'player_id': [f'P{i}' for i in range(1, 15)],
            'Name': [f'Player {i}' for i in range(1, 15)],
            'positions': ['PG/SG', 'PG', 'SG', 'SF', 'PF/C', 'PF', 'C', 'PG/SF', 'SG/PF', 'C', 'PG', 'SF', 'PF', 'SG'],
            'salary': [6000, 7000, 5000, 8000, 4500, 4000, 9000, 5500, 6500, 4200, 7500, 8500, 4800, 5200],
            'proj': [35.5, 40.2, 30.1, 45.8, 25.0, 22.1, 50.3, 32.7, 38.0, 20.9, 42.0, 48.
