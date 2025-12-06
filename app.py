# app.py - REVERTED TO WORKING STATE WITH PLACEHOLDERS AND ROBUST HEADER FIX

import pandas as pd 
import numpy as np
import streamlit as st 
from typing import Dict, Any, List, Tuple

# NOTE: The 'builder' module is assumed to be correct and unchanged.
from builder import (
    build_template_from_params, 
    generate_top_n_lineups, 
    ownership_bucket,
    PUNT_THR, CHALK_THR, MEGA_CHALK_THR,
    DEFAULT_SALARY_CAP, DEFAULT_ROSTER_SIZE
) 

# --- CONFIGURATION CONSTANTS ---
MIN_GAMES_REQUIRED = 2

# --- HEADER MAPPING (Simplified to exact Title Case headers from the image) ---
# NOTE: Keys MUST match the exact headers in your CSV file.
REQUIRED_CSV_TO_INTERNAL_MAP = {
    'Player': 'Name',
    'Salary': 'salary', 
    'Position': 'positions',
    'Team': 'Team',
    'Opponent': 'Opponent',
    'Projection': 'proj',
    'Ownership': 'own_proj',
    
    # Other supporting columns
    'Minutes': 'Minutes',
    'FPPM': 'FPPM',
    'Value': 'Value'
}
CORE_INTERNAL_COLS = ['salary', 'positions', 'proj', 'own_proj', 'Name', 'Team', 'Opponent']

# --- 1. DATA PREPARATION ---

def load_and_preprocess_data(uploaded_file=None) -> pd.DataFrame:
    """Loads CSV, standardizes ownership, and processes data."""
    
    df = pd.DataFrame()
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Data loaded successfully. Checking headers...")
            
            # --- CRITICAL FIX: STRIP WHITESPACE FROM ALL HEADERS ---
            df.columns = df.columns.str.strip()
            
            # --- VALIDATE AND MAP REQUIRED COLUMNS ---
            actual_map = {}
            
            # Build the actual mapping based on what's in the CSV and what's required
            for csv_name, internal_name in REQUIRED_CSV_TO_INTERNAL_MAP.items():
                if csv_name in df.columns:
                    actual_map[csv_name] = internal_name
            
            # Check for missing required columns
            essential_csv_names = ['Player', 'Salary', 'Position', 'Team', 'Opponent', 'Projection', 'Ownership']
            final_missing = [name for name in essential_csv_names if name not in actual_map]

            if final_missing:
                st.error(f"Missing essential headers: The following columns are missing or incorrectly named: {', '.join(final_missing)}.")
                st.error("Please ensure your CSV headers exactly match: Player, Salary, Position, Team, Opponent, Projection, Ownership.")
                return pd.DataFrame()

            # Rename columns using the filtered, actual map
            df.rename(columns=actual_map, inplace=True)
            
            # --- Ensure all internal required columns exist before proceeding ---
            if not all(col in df.columns for col in CORE_INTERNAL_COLS):
                st.error("Internal processing error: Required columns failed to map correctly.")
                return pd.DataFrame()

            
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            return pd.DataFrame()
    else:
        # --- Placeholder Data (REINSERTED) ---
        data = {
            'player_id': [f'P{i}' for i in range(1, 15)],
            'Name': [f'Player {i}' for i in range(1, 15)],
            'positions': ['PG/SG', 'PG', 'SG', 'SF', 'PF/C', 'PF', 'C', 'PG/SF', 'SG/PF', 'C', 'PG', 'SF', 'PF', 'SG'],
            'salary': [6000, 7000, 5000, 8000, 4500, 4000, 9000, 5500, 6500, 4200, 7500, 8500, 4800, 5200],
            'proj': [35.5, 40.2, 30.1, 45.8, 25.0, 22.1, 50.3, 32.7, 38.0, 20.9, 42.0, 48.0, 28.0, 31.0], 
            'own_proj': [45.0, 35.0, 15.0, 28.0, 5.0, 8.0, 40.0, 12.0, 20.0, 9.0, 33.0, 18.0, 4.0, 16.0], 
            'Team': ['LAL', 'LAL', 'BOS', 'BOS', 'MIL',
