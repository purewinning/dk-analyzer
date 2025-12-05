# app.py

import pandas as pd
import numpy as np
import streamlit as st 
from typing import Dict, Any, List
# Import the core logic functions and classes, including the ownership thresholds
from builder import (
    build_template_from_params, 
    build_optimal_lineup, 
    ownership_bucket,
    PUNT_THR, CHALK_THR, MEGA_CHALK_THR
) 

# --- CONFIGURATION CONSTANTS ---
SALARY_CAP = 50000
TOTAL_PLAYERS = 8
MIN_GAMES_REQUIRED = 2

# --- HEADER MAPPING (Translates external CSV headers to internal headers) ---
HEADER_MAP = {
    'Player': 'Name',             
    'Salary': 'salary',           
    'Position': 'positions',      
    'Projection': 'proj',         
    'Ownership %': 'own_proj',    
    'Team': 'Team',               
    'Opponent': 'Opponent'        
}

# --- 1. DATA PREPARATION ---

def load_and_preprocess_data(uploaded_file=None) -> pd.DataFrame:
    """Loads uploaded CSV, renames headers, and processes data."""
    
    df = pd.DataFrame()
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Data loaded successfully from file.")

            # --- STEP 1: RENAME COLUMNS ---
            rename_map = {
                old_name: new_name 
                for old_name, new_name in HEADER_MAP.items() 
                if old_name in df.columns
            }
            df.rename(columns=rename_map, inplace=True)
            
            # --- STEP 2: VALIDATE REQUIRED COLUMNS (After Renaming) ---
            required_internal_cols = ['Name', 'salary', 'positions', 'proj', 'own_proj']
            required_game_cols = ['Team', 'Opponent']
            
            if not all(col in df.columns for col in required_internal_cols):
                missing = [col for col in required_internal_cols if col not in df.columns]
                st.error(f"Missing one or more required columns after renaming. Missing: {missing}")
                return pd.DataFrame()

            # --- STEP 3: CREATE GAMEID (CRITICAL FIX) ---
            if 'GameID' not in df.columns:
                if all(col in df.columns for col in required_game_cols):
                    # Create a consistent game identifier (e.g., 'BOS@LAL')
                    df['GameID'] = df.apply(
                        lambda row: '@'.join(sorted([str(row['Team']), str(row['Opponent'])])), axis=1
                    )
                    st.info("ℹ️ Created **GameID** using Team and Opponent columns.")
                else:
                    st.error("Missing required column **GameID**. Cannot create game diversity constraint.")
                    return pd.DataFrame()
            
            # --- STEP 4: CLEANUP DATA TYPES ---
            df['player_id'] = df['Name'] 
            
            # Ensure ownership is between 0 and 1
            if df['own_proj'].max() > 10: 
                 df['own_proj'] = df['own_proj'] / 100
                 st.info("ℹ️ Divided 'own_proj' by 100 (assuming % format).")

            df['salary'] = df['salary'].astype(int)
            df['proj'] = df['proj'].astype(float)
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return pd.DataFrame()
    else:
        # --- Placeholder Data Setup (Only used if no file is uploaded) ---
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
        st.info("ℹ️ Using placeholder data. Upload your CSV to analyze a real slate.")

    # Pre-calculate the ownership bucket for the leverage constraint
    df['bucket'] = df['own_proj'].apply(ownership_bucket)
    return df

# --- 2. TAB FUNCTIONS ---

def tab_lineup_builder(slate_df, template):
    """Function to render the Lineup Builder tab."""
    st.header("Optimal Lineup Generation")
    
    st.info(f"🎯 Using Template: **{template.contest_label}** | Target Ownership Breakdown: {template.bucket_ranges(slack=1)}")
    
    if st.button("Generate Optimal Lineup"):
        
        # 3. Run Optimization (Classic NBA)
        with st.spinner('Calculating optimal lineup...'):
            optimal_lineup_df = build_optimal_lineup(
                slate_df=slate_df,
                template=template,
                bucket_slack=1,
            )
        
        # 4. Process and Display Results
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

def tab_contest_analyzer(slate_df, template):
    """Function to render the Contest Analyzer tab."""
    st.header("Contest and Ownership Analysis")
    st.markdown(f"This analyzer is targeting the **{template.contest_label}** structure.")
    st.markdown("---")

    st.subheader("Template Settings")
    
    # Display the Template parameters
    st.json({
        "Contest Type": template.contest_label,
        "Roster Size": template.roster_size,
        "Salary Cap": f"${template.salary_cap:,}",
        "Min Games Required": template.min_games
    })
    
    st.subheader("Ownership Ranges (Leverage Constraint)")
    # This line (line 167 in your context) is now correctly closed:
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
    
    # Display the actual distribution of players in the pool
    pool_counts = slate_df['bucket'].value_counts().reindex(list(ranges.keys()), fill_value=0)
    st.dataframe(pool_counts.rename("Player Count in Pool"), use_container_width=True)


# --- 3. MAIN APPLICATION ENTRY POINT ---

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("DraftKings NBA Optimizer & Analyzer 📊")
    st.markdown("---")
    
    # File uploader in the sidebar
    with st.sidebar:
        st.header("📥 Player Data")
        uploaded_file = st
