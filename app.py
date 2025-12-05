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
            'proj': [35.5, 40.2, 30.1, 45.8, 25.0, 22.1, 50.3, 32.7, 38.0, 20.9, 42.0, 48.0, 28.0, 31.0],
            'own_proj': [0.45, 0.35, 0.15, 0.28, 0.05, 0.08, 0.40, 0.12, 0.20, 0.09, 0.33, 0.18, 0.04, 0.16], 
            'Team': ['LAL', 'LAL', 'BOS', 'BOS', 'MIL', 'MIL', 'PHX', 'PHX', 'DEN', 'DEN', 'LAL', 'BOS', 'MIL', 'PHX'],
            'Opponent': ['BOS', 'BOS', 'LAL', 'LAL', 'DEN', 'DEN', 'MIL', 'MIL', 'PHX', 'PHX', 'BOS', 'LAL', 'DEN', 'MIL'],
            'GameID': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 2, 3, 4]
        }
        df = pd.DataFrame(data)
        st.warning("⚠️ Using placeholder data. Upload your CSV for real analysis.")

    # Pre-calculate the ownership bucket
    df['bucket'] = df['own_proj'].apply(ownership_bucket)
    return df

# --- 2. TAB FUNCTIONS ---

def tab_lineup_builder(slate_df, template):
    """Function to render the Lineup Builder tab."""
    st.header("Optimal Lineup Generation")
    
    st.info(f"🎯 Template: **{template.contest_label}** | Target Ownership Breakdown: {template.bucket_ranges(slack=1)}")
    
    if st.button("Generate Optimal Lineup"):
        
        with st.spinner(f'Calculating {template.contest_label} optimal lineup...'):
            optimal_lineup_df = build_optimal_lineup(
                slate_df=slate_df,
                template=template,
                bucket_slack=1,
            )
        
        if optimal_lineup_df is not None:
            total_salary = optimal_lineup_df['salary'].sum()
            total_points = optimal_lineup_df['proj'].sum()
            games_used = optimal_lineup_df['GameID'].nunique()
            
            st.subheader("🏆 Optimal Lineup Found")
            
            display_cols = ['Name', 'positions', 'Team', 'GameID', 'salary', 'proj', 'own_proj', 'bucket']
            lineup_df_display = optimal_lineup_df[display_cols].sort_values(by='proj', ascending=False).reset_index(drop=True)
            
            st.markdown(lineup_df_display.to_markdown(index=False, floatfmt=".2f"))
            
            st.markdown("---")
            st.subheader("Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Projection", f"{total_points:.2f} Pts")
            col2.metric("Salary Used", f"${total_salary:,}")
            col3.metric("Games Represented", f"{games_used} / {MIN_GAMES_REQUIRED} Min")
            
        else:
            st.error("❌ Could not find an optimal solution. Try adjusting constraints or player pool.")

def tab_contest_analyzer(slate_df, template):
    """Function to render the Contest Analyzer tab."""
    st.header("Contest and Ownership Analysis")
    st.markdown(f"This analyzer is targeting the **{template.contest_label}** structure.")
    st.markdown("---")

    st.subheader("Template Settings")
    st.json({
        "Contest Type": template.contest_label,
        "Roster Size": template.roster_size,
        "Salary Cap": f"${template.salary_cap:,}",
        "Min Games Required": template.min_games
    })
    
    st.subheader("Ownership Ranges (Leverage Constraint)")
    ranges = template.bucket_ranges(slack=1) 
    
    range_data = {
        "Bucket": list(ranges.keys()),
        "Ownership Threshold": [
            f"< {PUNT_THR*100:.0f}%", 
            f"{PUNT_THR*100:.0f}% - {CHALK_THR*100:.0f}%", 
            f"{CHALK_THR*100:.0f}% - {MEGA_CHALK_THR*100:.0f}%", 
            f"> {MEGA_CHALK_THR*100:.0f}%"
        ],
        "Target Count (Min-Max)": [f"{v[0]} - {v[1]}" for v in ranges.values()]
    }
    st.dataframe(pd.DataFrame(range_data), hide_index=True)

    st.subheader("Current Player Pool Ownership Distribution")
    pool_counts = slate_df['bucket'].value_counts().reindex(list(ranges.keys()), fill_value=0)
    st.dataframe(pool_counts.rename("Player Count in Pool"), use_container_width=True)


# --- 3. MAIN APPLICATION ENTRY POINT ---

if __name__ == '__main__':
    
    st.set_page_config(layout="wide")
    st.title("DraftKings NBA Optimizer & Analyzer 📊")
    st.markdown("---")
    
    # --- Sidebar Configuration (Contest Type & File Upload) ---
    with st.sidebar:
        st.header("1. Contest Setup")
        
        contest_type = st.radio(
            "Select Contest Type:",
            ('CASH', 'GPP (Single Entry)', 'GPP (Large Field)')
        )
        
        if contest_type == 'GPP (Single Entry)':
            contest_code = 'SE'
        elif contest_type == 'GPP (Large Field)':
            contest_code = 'LARGE_GPP'
        else: # CASH
            contest_code = 'CASH'
            
        st.markdown("---")
        
        st.header("2. Player Data")
        uploaded_file = st.file_uploader(
            "Upload Player Projections (CSV)", 
            type=['csv'],
            help="Required headers: Player, Salary, Position, Projection, Ownership %, Team, Opponent."
        )
        
    # 1. Load Data
    slate_df = load_and_preprocess_data(uploaded_file)
    if slate_df.empty:
        st.stop()
        
    # 2. Define the Target Contest Structure 
    template = build_template_from_params(
        contest_type=contest_code, 
        field_size=10000, 
        pct_to_first=30.0,
        roster_size=DEFAULT_ROSTER_SIZE,
        salary_cap=DEFAULT_SALARY_CAP,
        min_games=MIN_GAMES_REQUIRED
    )

    # 3. Create the Tabs
    tab1, tab2 = st.tabs(["🚀 Lineup Builder", "🔍 Contest Analyzer"])

    with tab1:
        tab_lineup_builder(slate_df, template)

    with tab2:
        tab_contest_analyzer(slate_df, template)
