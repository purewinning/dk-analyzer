# app.py - FINAL COMPLETE CODE

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

# --- HEADER MAPPING (Comprehensive to handle different capitalization/symbols) ---
# The keys here are the *possible* header names in your CSV file.
# The values are the *internal* names used by the script.
REQUIRED_CSV_TO_INTERNAL_MAP = {
    # Salary
    'SALARY': 'salary', 
    'Salary': 'salary', 
    # Position
    'POSITION': 'positions', 
    'Position': 'positions',
    # Projection
    'PROJECTED FP': 'proj',   
    'Projection': 'proj',
    # Ownership
    'OWNERSHIP%': 'own_proj',  
    'Ownership': 'own_proj',
    # Player
    'PLAYER': 'Name',
    'Player': 'Name',
    # Team
    'TEAM': 'Team',
    'Team': 'Team',
    # Opponent
    'OPPONENT': 'Opponent',
    'Opponent': 'Opponent',
    
    # Other supporting columns (no conflict)
    'Minutes': 'Minutes',
    'FPPM': 'FPPM',
    'Value': 'Value'
}

# --- 1. DATA PREPARATION ---

def load_and_preprocess_data(uploaded_file=None) -> pd.DataFrame:
    """Loads CSV, standardizes ownership to 0-100 scale, and processes data."""
    
    df = pd.DataFrame()
    CORE_INTERNAL_COLS = ['salary', 'positions', 'proj', 'own_proj', 'Name', 'Team', 'Opponent']
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Data loaded successfully.")
            
            # --- VALIDATE AND MAP REQUIRED COLUMNS ---
            internal_to_possible_csv = {
                'salary': ['Salary', 'SALARY'],
                'positions': ['Position', 'POSITION'],
                'proj': ['Projection', 'PROJECTED FP'],
                'own_proj': ['Ownership', 'OWNERSHIP%'],
                'Name': ['Player', 'PLAYER'],
                'Team': ['Team', 'TEAM'],
                'Opponent': ['Opponent', 'OPPONENT']
            }
            
            actual_map = {}
            for internal_name, possible_names in internal_to_possible_csv.items():
                found = False
                for name in possible_names:
                    if name in df.columns:
                        actual_map[name] = internal_name
                        found = True
                        break
                
                # If a required internal column could not be mapped, show error
                if not found and internal_name in CORE_INTERNAL_COLS:
                    st.error(f"Missing essential headers. Could not find a match for '{internal_name}'.")
                    st.error("Please ensure your CSV contains one of the following columns for each field: Player/PLAYER, Salary/SALARY, Position/POSITION, Projection/PROJECTED FP, Ownership/OWNERSHIP%, Team/TEAM, Opponent/OPPONENT.")
                    return pd.DataFrame()

            # Rename columns using the filtered, actual map
            df.rename(columns=actual_map, inplace=True)
            
            # --- CREATE GameID ---
            df['Team'] = df['Team'].astype(str)
            df['Opponent'] = df['Opponent'].astype(str)
            df['GameID'] = df.apply(
                lambda row: '@'.join(sorted([row['Team'], row['Opponent']])), axis=1
            )
            
            # --- CLEANUP & STANDARDIZE ---
            df['player_id'] = df['Name'] 
            
            # Clean Ownership
            df['own_proj'] = df['own_proj'].astype(str).str.replace('%', '', regex=False).str.replace(',', '.', regex=False)
            df['own_proj'] = pd.to_numeric(df['own_proj'], errors='coerce')
            df.dropna(subset=CORE_INTERNAL_COLS, inplace=True)

            if df['own_proj'].max() <= 1.0 and df['own_proj'].max() > 0:
                 df['own_proj'] = df['own_proj'] * 100
            
            df['own_proj'] = df['own_proj'].round(1)

            try:
                # Cleaning salary
                df['salary'] = df['salary'].astype(str).str.strip().str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                df['salary'] = pd.to_numeric(df['salary'], errors='coerce').astype('Int64') 
                df.dropna(subset=['salary'], inplace=True)

                df['salary'] = df['salary'].astype(int) 
                df['proj'] = df['proj'].astype(float)
                
                # Handle other columns dynamically
                df_columns = df.columns.tolist()
                if 'Minutes' in df_columns:
                    df['Minutes'] = pd.to_numeric(df.get('Minutes', 0), errors='coerce').astype(float).round(2)
                if 'FPPM' in df_columns:
                    df['FPPM'] = pd.to_numeric(df.get('FPPM', 0), errors='coerce').astype(float).round(2)
                if 'Value' in df_columns:
                    df['Value'] = pd.to_numeric(df.get('Value', 0), errors='coerce').astype(float).round(2)
                    
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
            'Opponent': ['BOS', 'BOS', 'LAL', 'LAL', 'MIL', 'MIL', 'PHX', 'PHX
