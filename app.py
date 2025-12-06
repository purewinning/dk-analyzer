# app.py - FINAL HEADER MAPPING FIX

import pandas as pd 
import numpy as np
import streamlit as st 
from typing import Dict, Any, List, Tuple
from builder import (
    build_template_from_params, 
    generate_top_n_lineups, 
    ownership_bucket,
    PUNT_THR, CHALK_THR, MEGA_CHALK_THR,
    DEFAULT_SALARY_CAP, DEFAULT_ROSTER_SIZE
) 

# --- CONFIGURATION CONSTANTS ---
MIN_GAMES_REQUIRED = 2

# --- HEADER MAPPING (UPDATED FOR YOUR LIKELY CSV FORMAT) ---
# NOTE: The keys here MUST match the exact text strings in the header row of your CSV.
REQUIRED_CSV_TO_INTERNAL_MAP = {
    # Attempting to match based on the source image, which uses all caps and symbols:
    'SALARY': 'salary', # Often capitalized
    'Salary': 'salary', # Or standard capitalization
    'POSITION': 'positions', 
    'Position': 'positions',
    'PROJECTED FP': 'proj',   # Mismatch 1: 'Projection' is likely 'PROJECTED FP'
    'Projection': 'proj',
    'OWNERSHIP%': 'own_proj',  # Mismatch 2: 'Ownership' is likely 'OWNERSHIP%'
    'Ownership': 'own_proj',
    'PLAYER': 'Name',
    'Player': 'Name',
    'TEAM': 'Team',
    'Team': 'Team',
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
    CORE_INTERNAL_COLS = ['salary', 'positions', 'proj', 'own_proj', 'Name']
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Data loaded successfully.")
            
            # --- VALIDATE REQUIRED COLUMNS ---
            # We now check if ANY of the possible source names map to the required internal name
            internal_to_possible_csv = {
                'salary': ['Salary', 'SALARY'],
                'positions': ['Position', 'POSITION'],
                'proj': ['Projection', 'PROJECTED FP'],
                'own_proj': ['Ownership', 'OWNERSHIP%'],
                'Name': ['Player', 'PLAYER'],
                'Team': ['Team', 'TEAM'],
                'Opponent': ['Opponent', 'OPPONENT']
            }
            
            # Create a simplified map using only the names found in the CSV
            actual_map = {}
            for internal_name, possible_names in internal_to_possible_csv.items():
                found = False
                for name in possible_names:
                    if name in df.columns:
                        actual_map[name] = internal_name
                        found = True
                        break
                
                # If we couldn't find a matching column for a required internal name, throw the error
                if not found and internal_name in CORE_INTERNAL_COLS:
                    # Provide the exact list the script failed on previously, plus the new ones
                    st.error(f"Missing essential headers. Could not find a match for '{internal_name}'.")
                    st.error("Please ensure your CSV contains *one* of the following columns for each field: Player/PLAYER, Salary/SALARY, Position/POSITION, Projection/PROJECTED FP, Ownership/OWNERSHIP%, Team/TEAM, Opponent/OPPONENT.")
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
            
            # Handle ownership clean up based on internal name
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
                
                # Handle other columns
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
            'Opponent': ['BOS', 'BOS', 'LAL', 'LAL', 'MIL', 'MIL', 'PHX', 'PHX', 'DEN', 'DEN', 'BOS', 'LAL', 'DEN', 'MIL'],
            'GameID': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 2, 3, 4],
            'Minutes': [30.0, 34.0, 28.0, 35.0, 20.0, 18.0, 36.0, 25.0, 32.0, 15.0, 33.0, 37.0, 22.0, 29.0],
            'FPPM': [1.18, 1.18, 1.07, 1.31, 1.25, 1.23, 1.40, 1.31, 1.19, 1.40, 1.27, 1.30, 1.27, 1.07],
            'Value': [5.92, 5.74, 6.02, 5.73, 5.56, 5.53, 5.59, 5.95, 5.85, 4.98, 5.60, 5.65, 5.83, 5.96]
        }
        df = pd.DataFrame(data)
        st.warning("⚠️ Using placeholder data. Upload your CSV for real analysis.")

    df['bucket'] = df['own_proj'].apply(ownership_bucket)
    df['value'] = np.where(df['salary'] > 0, (df['proj'] / (df['salary'] / 1000)).round(2), 0.0)

    if 'Lock' not in df.columns: df['Lock'] = False
    if 'Exclude' not in df.columns: df['Exclude'] = False
    
    return df

# --- 2. TAB FUNCTIONS (REMAINING CODE) ---

# Global session state update for multiple lineups
if 'optimal_lineups_results' not in st.session_state:
    st.session_state['optimal_lineups_results'] = {'lineups': [], 'ran': False}
if 'edited_df' not in st.session_state:
    st.session_state['edited_df'] = pd.DataFrame()

# --- STYLING FUNCTION FOR LINEUP DETAIL ---
def color_bucket(s):
    """Applies color to the 'CATEGORY' column based on the value."""
    if s == 'mega':
        color = 'background-color: #9C3838; color: white'  # Dark Red/Brown
    elif s == 'chalk':
        color = 'background-color: #A37F34; color: white'  # Dark Yellow/Gold
    elif s == 'mid':
        color = 'background-color: #38761D; color: white'  # Dark Green
    elif s == 'punt':
        color = 'background-color: #3D85C6; color: white'  # Dark Blue
    else:
        color = ''
    return color
# ----------------------------------------


def display_multiple_lineups(slate_df, template, lineup_list):
    """Function to display the top N optimized lineups with improved UI."""
    
    if not lineup_list:
        st.error("❌ No valid lineups could be found that meet all constraints.")
        st.warning("Try loosening your constraints or reducing the number of lineups requested.")
        return
    
    # --- METRICS SECTION (UI Improvement) ---
    best_lineup_data = lineup_list[0]
    best_proj = best_lineup_data['proj_score']
    
    # Calculate stats for the best lineup
    best_lineup_players_df = slate_df[slate_df['player_id'].isin(best_lineup_data['player_ids'])]
    best_salary = best_lineup_players_df['salary'].sum()
    best_value = best_proj / (best_salary / 1000) if best_salary else 0
    
    st.subheader("🚀 Top Lineup Metrics (Lineup 1)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Total Projected Points", 
                  value=f"{best_proj:.2f}", 
                  delta="Optimal Lineup Score")
    with col2:
        st.metric(label="Salary Used", 
                  value=f"${best_salary:,}", 
                  delta=f"${template.salary_cap - best_salary:,} Remaining")
    with col3:
        st.metric(label="Projection Value (X)", 
                  value=f"{best_value:.2f}", 
                  delta="Points per $1,000")

    st.markdown("---") 

    # --- SUMMARY TABLE ---
    st.subheader("📋 Top Lineups Summary")
    
    summary_data = []
    
    for i, lineup_data in enumerate(lineup_list):
        lineup_players_df = slate_df[slate_df['player_id'].isin(lineup_data['player_ids'])]
        summary_data.append({
            'Lineup': i + 1,
            'Total Proj': lineup_data['proj_score'],
            'Salary Used': lineup_players_df['salary'].sum(),
            'Games Used': lineup_players_df['GameID'].nunique()
        })
        
    summary_df = pd.DataFrame(summary_data).set_index('Lineup')
    
    st.dataframe(
        summary_df.style.format({"Total Proj": "{:.2f}", "Salary Used": "${:,}"}), 
        use_container_width=True
    )
    
    st.subheader("🔎 Lineup Detail View")
    
    # User selection for detailed lineup
    lineup_options = [f"Lineup {i+1} (Proj: {lineup_list[i]['proj_score']:.2f})" for i in range(len(lineup_list))]
    lineup_selection = st.selectbox("Select Lineup for Detail View", options=lineup_options)
    
    lineup_index = lineup_options.index(lineup_selection)
    selected_lineup_data = lineup_list[lineup_index]
    selected_lineup_ids = selected_lineup_data['player_ids']

    # Rebuild the dataframe for display
    lineup_df = slate_df[slate_df['player_id'].isin(selected_lineup_ids)].copy()
    
    # 1. Assign Roster Position (HACK for display)
    ROSTER_ORDER = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    lineup_df = lineup_df.head(8).assign(roster_position=ROSTER_ORDER) 

    # 2. Sort the DataFrame by the custom position order
    position_type = pd.CategoricalDtype(ROSTER_ORDER, ordered=True)
    lineup_df['roster_position'] = lineup_df['roster_position'].astype(position_type)
    lineup_df.sort_values(by='roster_position', inplace=True)
    
    # 3. Define display columns
    display_cols = ['roster_position', 'Name', 'positions', 'Team', 'Opponent', 'salary', 'proj', 'value', 'own_proj', 'bucket', '
