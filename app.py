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

            # CRITICAL SALARY CLEANING AND TYPE CONVERSION (Fix for $ , and whitespace)
            try:
                initial_len_salary = len(df)
                # 1. Convert to string and strip whitespace
                df['salary'] = df['salary'].astype(str).str.strip() 
                
                # 2. Remove $ and ,
                df['salary'] = df['salary'].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                
                # 3. Convert to integer (Use Int64 to allow NaN for failed conversions)
                df['salary'] = pd.to_numeric(df['salary'], errors='coerce').astype('Int64') 
                
                # Drop rows that failed salary conversion
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
            'Opponent': ['BOS', 'BOS', 'LAL', 'LAL', 'DEN', 'DEN', 'MIL', 'MIL', 'PHX', 'PHX', 'BOS', 'LAL', 'DEN', 'MIL'],
            'GameID': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 2, 3, 4]
        }
        df = pd.DataFrame(data)
        st.warning("⚠️ Using placeholder data. Upload your CSV for real analysis.")

    # Assign Buckets (Now using 0-100 scale logic)
    df['bucket'] = df['own_proj'].apply(ownership_bucket)
    
    # Initialize UI Control Columns
    if 'Lock' not in df.columns: df['Lock'] = False
    if 'Exclude' not in df.columns: df['Exclude'] = False
    
    return df

# --- 2. HELPER: RANDOMIZE PROJECTIONS ---
def apply_variance(df, variance_pct):
    """Applies random variance to projections."""
    df_varied = df.copy()
    rng = np.random.default_rng()
    noise = rng.uniform(1 - (variance_pct/100), 1 + (variance_pct/100), size=len(df))
    df_varied['proj'] = df_varied['proj'] * noise
    return df_varied

# --- 3. TAB FUNCTIONS ---

def tab_lineup_builder(slate_df, template):
    """Render the Interactive Lineup Builder."""
    st.header("1. Player Pool & Constraints")
    
    # --- A. PLAYER POOL EDITOR ---
    st.markdown("Use the table below to **Lock** (Force In), **Exclude** (Ban), or **Edit** projections.")
    
    # Define column config for the editor (CONDENSED)
    column_config = {"Name": st.column_config.TextColumn("Player Name", disabled=True), "positions": st.column_config.TextColumn("Pos", disabled=True), "salary": st.column_config.NumberColumn("Salary", format="$%d"), "proj": st.column_config.NumberColumn("Proj Pts", format="%.1f"), "own_proj": st.column_config.NumberColumn("Own %", format="%.1f"), "Lock": st.column_config.CheckboxColumn("🔒 Lock", help="Force this player into the lineup"), "Exclude": st.column_config.CheckboxColumn("❌ Exclude", help="Ban this player from the lineup"), "player_id": None, "GameID": None}
    
    # The Interactive Data Editor
    edited_df = st.data_editor(
        slate_df[['Lock', 'Exclude', 'Name', 'positions', 'salary', 'proj', 'own_proj', 'Team', 'Opponent', 'GameID', 'player_id']],
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        height=400,
        key="player_editor"
    )
    
    # Extract Locks and Excludes
    locked_players = edited_df[edited_df['Lock'] == True]['player_id'].tolist()
    excluded_players = edited_df[edited_df['Exclude'] == True]['player_id'].tolist()
    
    if locked_players or excluded_players:
        st.caption(f"🔒 **Locked:** {len(locked_players)} | ❌ **Excluded:** {len(excluded_players)}")

    st.markdown("---")
    
    # --- B. SIMULATION SETTINGS ---
    st.header("2. Generate Optimal Lineup")
    
    col_var, col_btn = st.columns([2, 1])
    with col_var:
        variance = st.slider("Randomize Projections (+/- %)", 0, 30, 0, help="Add randomness to 'simulate' a slate outcomes.")
    
    with col_btn:
        st.write("") # Spacer
        run_btn = st.button("🚀 Build Lineup", use_container_width=True)
    
    if run_btn:
        # Apply variance if requested
        final_df = edited_df.copy()
        if variance > 0:
            final_df = apply_variance(final_df, variance)
            st.toast(f"Applied +/- {variance}% variance to projections.")
            
        # Recalculate buckets based on potentially edited ownership
        final_df['bucket'] = final_df['own_proj'].apply(ownership_bucket)

        with st.spinner(f'Optimizing...'):
            optimal_lineup_df = build_optimal_lineup(
                slate_df=final_df,
                template=template,
                bucket_slack=1,
                locked_player_ids=locked_players,
                excluded_player_ids=excluded_players
            )
        
        if optimal_lineup_df is not None:
            total_salary = optimal_lineup_df['salary'].sum()
            total_points = optimal_lineup_df['proj'].sum()
            games_used = optimal_lineup_df['GameID'].nunique()
            
            st.success("### 🏆 Optimal Lineup Found")
            
            display_cols = ['Name', 'positions', 'Team', 'GameID', 'salary', 'proj', 'own_proj', 'bucket']
            lineup_df_display = optimal_lineup_df[display_cols].sort_values(by='proj', ascending=False).reset_index(drop=True)
            
            # Format dataframe for pretty display
            st.dataframe(
                lineup_df_display.style.format({"salary": "${:,}", "proj": "{:.1f}", "own_proj": "{:.1f}%"}),
                use_container_width=True
            )
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Projection", f"{total_points:.2f}")
            col2.metric("Salary Used", f"${total_salary:,}")
            col3.metric("Games", f"{games_used}")
            
        else:
            st.error("❌ No valid lineup found. This means your player pool (data) conflicts with the contest rules (constraints).")

def tab_contest_analyzer(slate_df, template):
    """Render the Contest Analyzer."""
    st.header("Contest Strategy Analyzer")
    st.info(f"Analysis based on: **{template.contest_label}**")

    st.subheader("Target Ownership Structure")
    ranges = template.bucket_ranges(slack=1) 
    
    range_data = {
        "Ownership Bucket": ["Punt (<10%)", "Mid (10-30%)", "Chalk (30-40%)", "Mega Chalk (>40%)"],
        "Target Player Count": [
            f"{ranges['punt'][0]}-{ranges['punt'][1]}",
            f"{ranges['mid'][0]}-{ranges['mid'][1]}",
            f"{ranges['chalk'][0]}-{ranges['chalk'][1]}",
            f"{ranges['mega'][0]}-{ranges['mega'][1]}"
        ]
    }
    st.table(pd.DataFrame(range_data))

    st.subheader("Your Player Pool Distribution")
    pool_counts = slate_df['bucket'].value_counts().reindex(list(ranges.keys()), fill_value=0)
    st.bar_chart(pool_counts)


# --- 4. MAIN ENTRY POINT ---

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="DK Lineup Builder")
    
    # Sidebar
    with st.sidebar:
        st.title("🏀 DK Builder")
        contest_type = st.selectbox("Contest Strategy", ['GPP (Single Entry)', 'GPP (Large Field)', 'CASH'])
        
        # Map selection
        c_map = {'GPP (Single Entry)': 'SE', 'GPP (Large Field)': 'LARGE_GPP', 'CASH': 'CASH'}
        contest_code = c_map[contest_type]
        
        st.divider()
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
    # Load Data
    slate_df = load_and_preprocess_data(uploaded_file)
    if slate_df.empty:
        st.stop()
        
    # Build Template
    template = build_template_from_params(
        contest_type=contest_code, 
        field_size=10000, 
        pct_to_first=30.0,
        roster_size=DEFAULT_ROSTER_SIZE,
        salary_cap=DEFAULT_SALARY_CAP,
        min_games=MIN_GAMES_REQUIRED
    )

    # Tabs
    t1, t2 = st.tabs(["🚀 Lineup Builder", "📊 Contest Analyzer"])
    
    with t1:
        tab_lineup_builder(slate_df, template)
    with t2:
        tab_contest_analyzer(slate_df, template)
