# app.py - FINAL CLEANED CODE (No Placeholders)

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

# --- HEADER MAPPING (Simplified to exact Title Case headers) ---
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
    
    if uploaded_file is None:
        # Return an empty DataFrame with the expected columns for initialization
        df = pd.DataFrame(columns=CORE_INTERNAL_COLS + ['player_id', 'GameID', 'bucket', 'value', 'Lock', 'Exclude'])
        return df
        
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


    # --- FINAL PROCESSING ---
    
    # --- CREATE GameID and PlayerID ---
    df['Team'] = df['Team'].astype(str)
    df['Opponent'] = df['Opponent'].astype(str)
    df['GameID'] = df.apply(
        lambda row: '@'.join(sorted([row['Team'], row['Opponent']])), axis=1
    )
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
    # Initialize with an empty DataFrame that includes the expected columns
    st.session_state['edited_df'] = pd.DataFrame(columns=CORE_INTERNAL_COLS + ['player_id', 'GameID', 'bucket', 'value', 'Lock', 'Exclude'])


# --- STYLING FUNCTION FOR LINEUP DETAIL ---
def color_bucket(s):
    """Applies color to the 'CATEGORY' column based on the value."""
    if s == 'mega':
        color = 'background-color: #9C3838; color: white'  
    elif s == 'chalk':
        color = 'background-color: #A37F34; color: white' 
    elif s == 'mid':
        color = 'background-color: #38761D; color: white'  
    elif s == 'punt':
        color = 'background-color: #3D85C6; color: white'  
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
    display_cols = ['roster_position', 'Name', 'positions', 'Team', 'Opponent', 'salary', 'proj', 'value', 'own_proj', 'bucket', 'Minutes', 'FPPM'] 
    lineup_df_display = lineup_df[display_cols].reset_index(drop=True)
    
    # 4. Rename the column for display
    lineup_df_display.rename(columns={'roster_position': 'SLOT', 'positions': 'POS', 'own_proj': 'OWN%', 'Minutes': 'MIN', 'FPPM': 'FP/M', 'bucket': 'CATEGORY'}, inplace=True)
    
    # Display the detailed lineup with styling
    styled_lineup_df = lineup_df_display.style.applymap(
        color_bucket, subset=['CATEGORY']
    ).format({
        "salary": "${:,}", 
        "proj": "{:.1f}", 
        "value": "{:.2f}", 
        "OWN%": "{:.1f}%", 
        "MIN": "{:.1f}", 
        "FP/M": "{:.2f}"
    })
    
    st.dataframe(
        styled_lineup_df, 
        use_container_width=True,
        hide_index=True 
    )


def tab_lineup_builder(slate_df, template):
    """Render the Interactive Lineup Builder and run the multi-lineup Optimizer."""
    st.header(f"1. Player Pool & Constraints for **{template.contest_label}**")
    
    # --- A. PLAYER POOL EDITOR ---
    st.markdown("Use the table to **🔒 Lock** or **❌ Exclude** players. The **Category** column shows ownership risk.")
    
    column_config = {
        "Name": st.column_config.TextColumn("Player", disabled=True), 
        "bucket": st.column_config.TextColumn("Category", disabled=True, help="punt (<10%), mid (10-30%), chalk (30-40%), mega (>40%)", width="small"),
        "positions": st.column_config.TextColumn("Pos", disabled=True), 
        "Team": st.column_config.TextColumn("Team", disabled=True, width="small"), 
        "Opponent": st.column_config.TextColumn("Opp", disabled=True, width="small"),
        "salary": st.column_config.NumberColumn("Salary", format="$%d", width="small"), 
        "proj": st.column_config.NumberColumn("Proj Pts", format="%.1f", width="small"), 
        "value": st.column_config.NumberColumn("Value (X)", format="
