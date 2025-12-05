# app.py

import pandas as pd
import numpy as np
import streamlit as st 
from typing import Dict, Any, List
# Import everything from builder to ensure access to constants
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

# --- HEADER MAPPING (Not used for placeholder, but kept for context) ---
HEADER_MAP = {'Player': 'Name'} 

# --- 1. DATA PREPARATION (Simplified to ONLY use safe placeholder data) ---

def load_and_preprocess_data(uploaded_file=None) -> pd.DataFrame:
    """Uses only safe, hardcoded placeholder data to guarantee a working DataFrame."""
    
    st.info("ℹ️ Running DIAGNOSTIC MODE: Using only internal placeholder data.")
    
    try:
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
        
        # Pre-calculate the ownership bucket
        df['bucket'] = df['own_proj'].apply(ownership_bucket)
        
        st.success("✅ Placeholder data loaded and pre-processed successfully.")
        return df
        
    except Exception as e:
        st.error(f"CRITICAL FAILURE: Placeholder data crashed the app. Error: {e}")
        return pd.DataFrame()


# --- 2. TAB FUNCTIONS (REMAIN UNCHANGED) ---

# ... (Include your tab_lineup_builder function here) ...
def tab_lineup_builder(slate_df, template):
    st.header("Optimal Lineup Generation")
    # ... (rest of the function logic) ...
    st.info(f"🎯 Using Template: **{template.contest_label}** | Target Ownership Breakdown: {template.bucket_ranges(slack=1)}")
    if st.button("Generate Optimal Lineup"):
        # ... (optimization logic) ...
        with st.spinner('Calculating optimal lineup...'):
            optimal_lineup_df = build_optimal_lineup(
                slate_df=slate_df,
                template=template,
                bucket_slack=1,
            )
        # ... (display results logic) ...
        if optimal_lineup_df is not None:
             # Calculate summary metrics
            total_salary = optimal_lineup_df['salary'].sum()
            total_points = optimal_lineup_df['proj'].sum()
            games_used = optimal_lineup_df['GameID'].nunique()
            st.subheader("🏆 Optimal Lineup Found")
            # Display the Lineup
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
            st.error("❌ Could not find an optimal solution. Check constraints and player pool.")

# ... (Include your tab_contest_analyzer function here) ...
def tab_contest_analyzer(slate_df, template):
    st.header("Contest and Ownership Analysis")
    # ... (rest of the function logic) ...
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
    
    # File uploader is hidden during this diagnostic test
    with st.sidebar:
        st.header("📥 Player Data")
        st.write("File upload is DISABLED for diagnostic test.")
        uploaded_file = None # Force no file to be loaded
        
    # 1. Load Data
    # This call must succeed if the app is working.
    slate_df = load_and_preprocess_data(uploaded_file)
    
    if slate_df.empty:
        st.error("Application is halted due to placeholder data failure.")
        st.stop()
    
    # 2. Define the Target Contest Structure 
    template = build_template_from_params(
        contest_type="SE", 
        field_size=10000, 
        pct_to_first=30.0,
        roster_size=TOTAL_PLAYERS,
        salary_cap=SALARY_CAP,
        min_games=MIN_GAMES_REQUIRED
    )

    # 3. Create the Tabs
    tab1, tab2 = st.tabs(["🚀 Lineup Builder", "🔍 Contest Analyzer"])

    with tab1:
        tab_lineup_builder(slate_df, template)

    with tab2:
        tab_contest_analyzer(slate_df, template)
