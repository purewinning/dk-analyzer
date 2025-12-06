# app.py - FINAL FIX: ADDED MISSING COLON AFTER 'with col2'

import pandas as pd 
import numpy as np
import streamlit as st 
from typing import Dict, Any, List, Tuple
import io 

# NOTE: The 'builder' module is assumed to be correct and unchanged.
from builder import (
    build_template_from_params, 
    generate_top_n_lineups, 
    ownership_bucket,
    PUNT_THR, CHALK_THR, MEGA_CHALK_THR,
    DEFAULT_SALARY_CAP, DEFAULT_ROSTER_SIZE
) 

# --- STREAMLIT CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(layout="wide", page_title="🏀 DK Lineup Optimizer")
# ---------------------------------------------

# --- CONFIGURATION CONSTANTS ---
MIN_GAMES_REQUIRED = 2

# --- HEADER MAPPING ---
REQUIRED_CSV_TO_INTERNAL_MAP = {
    'Player': 'Name',
    'Salary': 'salary', 
    'Position': 'positions',
    'Team': 'Team',
    'Opponent': 'Opponent',
    'PROJECTED FP': 'proj',      
    'OWNERSHIP %': 'own_proj',   
    
    # Fallback for common one-word headers
    'Projection': 'proj',
    'Ownership': 'own_proj',
    
    # Other supporting columns
    'Minutes': 'Minutes',
    'FPPM': 'FPPM',
    'Value': 'Value'
}
CORE_INTERNAL_COLS = ['salary', 'positions', 'proj', 'own_proj', 'Name', 'Team', 'Opponent']

# --- 1. DATA PREPARATION ---

def load_and_preprocess_data(pasted_data: str = None) -> pd.DataFrame:
    """Loads CSV from a pasted string, standardizes ownership, and processes data."""
    
    if pasted_data is None or not pasted_data.strip():
        # Return an empty DataFrame with the expected columns for initialization
        df = pd.DataFrame(columns=CORE_INTERNAL_COLS + ['player_id', 'GameID', 'bucket', 'value', 'Lock', 'Exclude'])
        return df
        
    try:
        data_io = io.StringIO(pasted_data)
        df = pd.read_csv(data_io)
        st.success("✅ Data pasted successfully. Checking headers...")
        
        # --- CRITICAL FIX: STRIP WHITESPACE FROM ALL HEADERS ---
        df.columns = df.columns.str.strip()
        
        # --- VALIDATE AND MAP REQUIRED COLUMNS ---
        actual_map = {}
        required_internal = ['Name', 'salary', 'positions', 'Team', 'Opponent', 'proj', 'own_proj']
        
        for csv_name, internal_name in REQUIRED_CSV_TO_INTERNAL_MAP.items():
            if csv_name in df.columns:
                actual_map[csv_name] = internal_name
        
        mapped_internal_names = set(actual_map.values())
        final_missing_internal = [name for name in required_internal if name not in mapped_internal_names]

        if final_missing_internal:
            missing_csv_names = [k for k, v in REQUIRED_CSV_TO_INTERNAL_MAP.items() if v in final_missing_internal and k in ['Player', 'Salary', 'Position', 'Team', 'Opponent', 'PROJECTED FP', 'OWNERSHIP %']]
            
            st.error(f"Missing essential headers: The following columns are missing or incorrectly named: {', '.join(set(missing_csv_names))}.")
            st.error("Please ensure your pasted data's first row (headers) exactly matches: Player, Salary, Position, Team, Opponent, PROJECTED FP, OWNERSHIP %.")
            return pd.DataFrame()

        df.rename(columns=actual_map, inplace=True)
        
        if not all(col in df.columns for col in CORE_INTERNAL_COLS):
            st.error("Internal processing error: Required columns failed to map correctly.")
            return pd.DataFrame()

        
    except Exception as e:
        st.error(f"Error processing pasted data: {e}")
        return pd.DataFrame()


    # --- FINAL PROCESSING ---
    
    df['Team'] = df['Team'].astype(str)
    df['Opponent'] = df['Opponent'].astype(str)
    df['GameID'] = df.apply(
        lambda row: '@'.join(sorted([row['Team'], row['Opponent']])), axis=1
    )
    df['player_id'] = df['Name'] 
            
    df['own_proj'] = df['own_proj'].astype(str).str.replace('%', '', regex=False).str.replace(',', '.', regex=False)
    df['own_proj'] = pd.to_numeric(df['own_proj'], errors='coerce')
    df.dropna(subset=CORE_INTERNAL_COLS, inplace=True)

    if df['own_proj'].max() <= 1.0 and df['own_proj'].max() > 0:
            df['own_proj'] = df['own_proj'] * 100
    
    df['own_proj'] = df['own_proj'].round(1)

    try:
        df['salary'] = df['salary'].astype(str).str.strip().str.replace('$', '', regex=False).str.replace(',', '', regex=False)
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce').astype('Int64') 
        df.dropna(subset=['salary'], inplace=True)

        df['salary'] = df['salary'].astype(int) 
        df['proj'] = df['proj'].astype(float)
        
        df_columns = df.columns.tolist()
        
        if 'Minutes' in df_columns:
            df['Minutes'] = pd.to_numeric(df.get('Minutes', 0), errors='coerce').astype(float).round(2)
        if 'FPPM' in df_columns:
            df['FPPM'] = pd.to_numeric(df.get('FPPM', 0), errors='coerce').astype(float).round(2)
        if 'Value' in df_columns:
            df['Value'] = pd.to_numeric(df.get('Value', 0), errors='coerce').astype(float).round(2)
            
    except Exception as e:
        st.error(f"Post-load conversion failed: {e}")
        return pd.DataFrame()

    if len(df) == 0:
        st.error("❌ Final player pool is empty after cleaning.")
        return pd.DataFrame()

    df['bucket'] = df['own_proj'].apply(ownership_bucket)
    df['value'] = np.where(df['salary'] > 0, (df['proj'] / (df['salary'] / 1000)).round(2), 0.0)

    if 'Lock' not in df.columns: df['Lock'] = False
    if 'Exclude' not in df.columns: df['Exclude'] = False
    
    return df

# --- 2. TAB FUNCTIONS ---

if 'optimal_lineups_results' not in st.session_state:
    st.session_state['optimal_lineups_results'] = {'lineups': [], 'ran': False}
if 'edited_df' not in st.session_state:
    st.session_state['edited_df'] = pd.DataFrame(columns=CORE_INTERNAL_COLS + ['player_id', 'GameID', 'bucket', 'value', 'Lock', 'Exclude'])


# --- STYLING FUNCTION FOR LINEUP DETAIL ---
def color_bucket(s):
    """Applies color to the 'CATEGORY' column based on the value."""
    if s == 'mega':
        color = 'background-color: #9C3838; color: white'  
    elif s == 'chalk':
        color = 'background-color: #A37F34; color: white' 
    elif s == 'mid':
        color = 'background-color: #
