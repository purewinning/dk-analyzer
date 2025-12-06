# app.py

import pandas as pd 
import numpy as np
import streamlit as st 
from typing import Dict, Any, List, Tuple
# Ensure this import block is perfectly copied
from builder import (
    build_template_from_params, 
    run_monte_carlo_simulations, 
    optimize_single_lineup, 
    ownership_bucket,
    PUNT_THR, CHALK_THR, MEGA_CHALK_THR,
    DEFAULT_SALARY_CAP, DEFAULT_ROSTER_SIZE
) 

# --- CONFIGURATION CONSTANTS ---
MIN_GAMES_REQUIRED = 2
DEFAULT_ITERATIONS = 500
DEFAULT_DIVERSITY = 4 # Max shared players in any two final lineups

# --- HEADER MAPPING ---
REQUIRED_CSV_TO_INTERNAL_MAP = {
    'Salary': 'salary',
    'Position': 'positions',
    'Projection': 'proj',
    'Ownership': 'own_proj',
}


# --- 1. DATA PREPARATION ---

def load_and_preprocess_data(uploaded_file=None) -> pd.DataFrame:
    """Loads CSV, standardizes ownership to 0-100 scale, and processes data."""
    
    df = pd.DataFrame()
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Data loaded successfully.")

            # Rename columns
            if 'Player' in df.columns:
                df.rename(columns={'Player': 'Name'}, inplace=True)
            
            missing_csv_cols = [col for col in REQUIRED_CSV_TO_INTERNAL_MAP.keys() if col not in df.columns]
            if missing_csv_cols:
                st.error(f"Missing headers: {missing_csv_cols}. Required: {list(REQUIRED_CSV_TO_INTERNAL_MAP.keys())}")
                return pd.DataFrame()
            
            df.rename(columns=REQUIRED_CSV_TO_INTERNAL_MAP, inplace=True)
            
            # Create GameID (Requires Team and Opponent columns)
            required_game_cols = ['Team', 'Opponent']
            if 'GameID' not in df.columns:
                if all(col in df.columns for col in required_game_cols):
                    df['GameID'] = df.apply(
                        lambda row: '@'.join(sorted([str(row['Team']), str(row['Opponent'])])), axis=1
                    )
                else:
                    st.error("Missing required columns: **Team** and **Opponent**.")
                    return pd.DataFrame()
            
            # --- CLEANUP & STANDARDIZE ---
            df['player_id'] = df['Name'] 
            
            # Clean Ownership (Handle commas, %, convert to numeric)
            df['own_proj'] = df['own_proj'].astype(str).str.replace('%', '', regex=False).str.replace(',', '.', regex=False)
            df['own_proj'] = pd.to_numeric(df['own_proj'], errors='coerce')
            df.dropna(subset=['own_proj'], inplace=True)

            # Standardize Ownership to 0-100 Scale (Whole Numbers)
            if df['own_proj'].max() <= 1.0:
                 df['own_proj'] = df['own_proj'] * 100
            
            df['own_proj'] = df['own_proj'].round(1)

            # CRITICAL SALARY CLEANING AND TYPE CONVERSION
            try:
                initial_len_salary = len(df)
                df['salary'] = df['salary'].astype(str).str.strip().str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                # Convert to nullable Int64 initially to handle NaNs/errors, then drop
                df['salary'] = pd.to_numeric(df['salary'], errors='coerce').astype('Int64') 
                df.dropna(subset=['salary'], inplace=True)
                dropped_len_salary = initial_len_salary - len(df)
                
                if dropped_len_salary > 0:
                    st.warning(f"⚠️ Dropped {dropped_len_salary} player(s) due to unfixable salary data.")

                # Final type conversions
                df['salary'] = df['salary'].astype(int) 
                df['proj'] = df['proj'].astype(float)
            except Exception as e:
                st.error(f"Failed final conversion (Salary/Projection): {e}")
                return pd.DataFrame()
            
            if len(df) == 0:
                 st.error("❌ Final player pool is empty after cleaning.")
                 return pd.DataFrame()
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return pd.DataFrame()
    else:
        # --- Placeholder Data (VERIFIED) ---
        data = {
            'player_id': [f'P{i}' for i in range(1, 15)],
            'Name': [f'Player {i}' for i in range(1, 15)],
            'positions': ['PG/SG', 'PG', 'SG', 'SF', 'PF/C', 'PF', 'C', 'PG/SF', 'SG/PF', 'C', 'PG', 'SF', 'PF', 'SG'],
            'salary': [6000, 7000, 5000, 8000, 4500, 4000, 9000, 5500, 6500, 4200, 7500, 8500, 4800, 5200],
            'proj': [35.5, 40.2, 30.1, 45.8, 25.0, 22.1, 50.3, 32.7, 38.0, 20.9, 42.0, 48.0, 28.0, 31.0], 
            'own_proj': [45.0, 35.0, 15.0, 28.0, 5.0, 8.0, 40.0, 12.0, 20.0, 9.0, 33.0, 18.0, 4.0, 16.0], 
            'Team': ['LAL', 'LAL', 'BOS', 'BOS', 'MIL', 'MIL', 'PHX', 'PHX', 'DEN', 'DEN', 'LAL', 'BOS', 'MIL', 'PHX'],
            'Opponent': ['BOS', 'BOS', 'LAL', 'LAL', 'MIL', 'MIL', 'PHX', 'PHX', 'DEN', 'DEN', 'BOS', 'LAL', 'DEN', 'MIL'],
            'GameID': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 2, 3, 4]
        }
        df = pd.DataFrame(data)
        st.warning("⚠️ Using placeholder data. Upload your CSV for real analysis.")

    # Assign Buckets 
    df['bucket'] = df['own_proj'].apply(ownership_bucket)
    
    # --- CALCULATE VALUE ---
    # Value = Projected Points / (Salary / 1000)
    df['value'] = np.where(df['salary'] > 0, (df['proj'] / (df['salary'] / 1000)).round(2), 0.0)

    # Initialize UI Control Columns
    if 'Lock' not in df.columns: df['Lock'] = False
    if 'Exclude' not in df.columns: df['Exclude'] = False
    if 'Max_Exposure' not in df.columns: df['Max_Exposure'] = 100 # Default to 100%
    
    return df

# --- 2. TAB FUNCTIONS ---

# Use session state to store simulation results
if 'sim_results' not in st.session_state:
    st.session_state['sim_results'] = {'lineups': [], 'exposures': {}}
if 'edited_df' not in st.session_state:
    st.session_state['edited_df'] = pd.DataFrame()


def tab_lineup_builder(slate_df, template):
    """Render the Interactive Lineup Builder and run MCS."""
    st.header("1. Player Pool & Constraints")
    
    # --- A. PLAYER POOL EDITOR ---
    st.markdown("Use the table to **Lock**, **Exclude**, or set **Max Exposure** for Monte Carlo.")
    
    # Define column config for the editor (CONDENSED)
    column_config = {
        "Name": st.column_config.TextColumn("Player Name", disabled=True), 
        "positions": st.column_config.TextColumn("Pos", disabled=True), 
        "salary": st.column_config.NumberColumn("Salary", format="$%d"), 
        "proj": st.column_config.NumberColumn("Proj Pts", format="%.1f"), 
        "value": st.column_config.NumberColumn("Value (X)", format="%.2f", disabled=True), 
        "own_proj": st.column_config.NumberColumn("Own %", format="%.1f"), 
        "Lock": st.column_config.CheckboxColumn("🔒 Lock", help="Force this player into the lineup"), 
        "Exclude": st.column_config.CheckboxColumn("❌ Exclude", help="Ban this player from the lineup"), 
        "Max_Exposure": st.column_config.NumberColumn("Max Exposure (%)", min_value=0, max_value
