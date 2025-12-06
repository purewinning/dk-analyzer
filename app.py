# app.py

import pandas as pd 
import numpy as np
import streamlit as st 
from typing import Dict, Any, List, Tuple
# Import the multi-lineup generation function
from builder import (
    build_template_from_params, 
    generate_top_n_lineups, 
    ownership_bucket,
    PUNT_THR, CHALK_THR, MEGA_CHALK_THR,
    DEFAULT_SALARY_CAP, DEFAULT_ROSTER_SIZE
) 

# --- CONFIGURATION CONSTANTS ---
MIN_GAMES_REQUIRED = 2

# --- HEADER MAPPING (UPDATED FOR YOUR CSV) ---
# Maps your input CSV headers to the internal names used by the script.
# **THIS SECTION HAS BEEN UPDATED**
REQUIRED_CSV_TO_INTERNAL_MAP = {
    'Salary': 'salary',
    'Position': 'positions',
    'Projection': 'proj',
    'Ownership': 'own_proj',
    'Player': 'Name',
    'Team': 'Team',
    'Opponent': 'Opponent',
    'Minutes': 'Minutes',
    'FPPM': 'FPPM',
    'Value': 'Value'
}


# --- 1. DATA PREPARATION ---

def load_and_preprocess_data(uploaded_file=None) -> pd.DataFrame:
    """Loads CSV, standardizes ownership to 0-100 scale, and processes data."""
    
    df = pd.DataFrame()
    
    # Required columns that must be in the final, internal dataframe
    CORE_INTERNAL_COLS = ['salary', 'positions', 'proj', 'own_proj', 'Name']
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Data loaded successfully.")
            
            # --- VALIDATE REQUIRED COLUMNS ---
            required_csv_cols = ['Salary', 'Position', 'Projection', 'Ownership', 'Player', 'Team', 'Opponent']
            missing_csv_cols = [col for col in required_csv_cols if col not in df.columns]
            
            if missing_csv_cols:
                st.error(f"Missing essential headers: **{', '.join(missing_csv_cols)}**. Please ensure your CSV contains: {', '.join(required_csv_cols)}.")
                return pd.DataFrame()
            
            # Rename columns using the mapping
            df.rename(columns=REQUIRED_CSV_TO_INTERNAL_MAP, inplace=True)
            
            # --- CREATE GameID ---
            df['Team'] = df['Team'].astype(str)
            df['Opponent'] = df['Opponent'].astype(str)
            df['GameID'] = df.apply(
                lambda row: '@'.join(sorted([row['Team'], row['Opponent']])), axis=1
            )
            
            # --- CLEANUP & STANDARDIZE ---
            # Set player_id from the 'Name' column
            df['player_id'] = df['Name'] 
            
            # Clean Ownership (Handle commas, %, convert to numeric)
            df['own_proj'] = df['own_proj'].astype(str).str.replace('%', '', regex=False).str.replace(',', '.', regex=False)
            df['own_proj'] = pd.to_numeric(df['own_proj'], errors='coerce')
            df.dropna(subset=CORE_INTERNAL_COLS, inplace=True)

            # Standardize Ownership to 0-100 Scale (Whole Numbers)
            if df['own_proj'].max() <= 1.0 and df['own_proj'].max() > 0:
                 df['own_proj'] = df['own_proj'] * 100
            
            df['own_proj'] = df['own_proj'].round(1)

            # Final type conversions
            try:
                # Cleaning salary
                df['salary'] = df['salary'].astype(str).str.strip().str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                df['salary'] = pd.to_numeric(df['salary'], errors='coerce').astype('Int64') 
                df.dropna(subset=['salary'], inplace=True)

                df['salary'] = df['salary'].astype(int) 
                df['proj'] = df['proj'].astype(float)
                # Ensure other numeric fields are float/int
                for col in ['Minutes', 'FPPM', 'Value']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float).round(2)

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

    # Assign Buckets 
    df['bucket'] = df['own_proj'].apply(ownership_bucket)
    
    # --- CALCULATE VALUE (Re-calculate if column was missing or for consistency) ---
    df['value'] = np.where(df['salary'] > 0, (df['proj'] / (df['salary'] / 1000)).round(2), 0.0)

    # Initialize UI Control Columns
    if 'Lock' not in df.columns: df['Lock'] = False
    if 'Exclude' not in df.columns: df['Exclude'] = False
    
    return df

# --- 2. TAB FUNCTIONS ---

# Global session state update for multiple lineups
if 'optimal_lineups_results' not in st.session_state:
    st.session_state['optimal_lineups_results'] = {'lineups': [], 'ran': False}
if 'edited_df' not in st.session_state:
    st.session_state['edited_df'] = pd.DataFrame()


def display_multiple_lineups(slate_df, lineup_list):
    """Function to display the top N optimized lineups."""
    
    if not lineup_list:
        st.error("❌ No valid lineups could be found that meet all constraints.")
        st.warning("Try loosening your constraints or reducing the number of lineups requested.")
        return
        
    
    st.subheader("Top Lineups Summary")
    
    summary_data = []
    
    for i, lineup_data in enumerate(lineup_list):
        summary_data.append({
            'Lineup': i + 1,
            'Total Proj': lineup_data['proj_score'],
            'Salary Used': slate_df[slate_df['player_id'].isin(lineup_data['player_ids'])]['salary'].sum(),
            'Games Used': slate_df[slate_df['player_id'].isin(lineup_data['player_ids'])]['GameID'].nunique()
        })
        
    summary_df = pd.DataFrame(summary_data).set_index('Lineup')
    
    st.dataframe(
        summary_df.style.format({"Total Proj": "{:.2f}", "Salary Used": "${:,}"}), 
        use_container_width=True
    )
    
    st.subheader("Lineup Details")
    
    # User selection for detailed lineup
    lineup_index = st.selectbox("Select Lineup # for Detail View", 
                                options=list(range(1, len(lineup_list) + 1)), 
                                format_func=lambda x: f"Lineup {x} (Proj: {lineup_list[x-1]['proj_score']:.2f})")
    
    # Get the selected lineup data
    selected_lineup_data = lineup_list[lineup_index - 1]
    selected_lineup_ids = selected_lineup_data['player_ids']

    # Rebuild the dataframe for display
    lineup_df = slate_df[slate_df['player_id'].isin(selected_lineup_ids)].copy()
    
    # 1. Assign Roster Position (HACK for display)
    ROSTER_ORDER = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    # Use .head(8) defensively, though the optimizer should ensure 8 players
    lineup_df = lineup_df.head(8).assign(roster_position=ROSTER_ORDER) 

    # 2. Sort the DataFrame by the custom position order
    position_type = pd.CategoricalDtype(ROSTER_ORDER, ordered=True)
    lineup_df['roster_position'] = lineup_df['roster_position'].astype(position_type)
    lineup_df.sort_values(by='roster_position', inplace=True)
    
    # 3. Define display columns
    # **UPDATED DISPLAY COLUMNS**
    display_cols = ['roster_position', 'Name', 'positions', 'Team', 'Opponent', 'salary', 'proj', 'value', 'own_proj', 'Minutes', 'FPPM', 'bucket'] 
    lineup_df_display = lineup_df[display_cols].reset_index(drop=True)
    
    # 4. Rename the column for display
    lineup_df_display.rename(columns={'roster_position': 'SLOT', 'positions': 'POS', 'own_proj': 'OWN%', 'Minutes': 'MIN', 'FPPM': 'FP/M'}, inplace=True)
    
    # Display the detailed lineup
    st.dataframe(
        lineup_df_display.style.format({"salary": "${:,}", "proj": "{:.1f}", "value": "{:.2f}", "OWN%": "{:.1f}%", "MIN": "{:.1f}", "FP/M": "{:.2f}"}), 
        use_container_width=True,
        hide_index=True 
    )


def tab_lineup_builder(slate_df, template):
    """Render the Interactive Lineup Builder and run the multi-lineup Optimizer."""
    st.header(f"1. Player Pool & Constraints for **{template.contest_label}**")
    
    # --- A. PLAYER POOL EDITOR ---
    st.markdown("Use the table to **Lock** or **Exclude** players for the optimal lineup.")
    
    # **UPDATED COLUMN CONFIGURATION**
    column_config = {
        "Name": st.column_config.TextColumn("Player", disabled=True), 
        "positions": st.column_config.TextColumn("Pos", disabled=True), 
        "Team": st.column_config.TextColumn("Team", disabled=True, width="small"), 
        "Opponent": st.column_config.TextColumn("Opp", disabled=True, width="small"),
        "salary": st.column_config.NumberColumn("Salary", format="$%d", width="small"), 
        "proj": st.column_config.NumberColumn("Proj Pts", format="%.1f", width="small"), 
        "value": st.column_config.NumberColumn("Value (X)", format="%.2f", disabled=True, width="small"), 
        "own_proj": st.column_config.NumberColumn("Own %", format="%.1f%%", width="small"),
        "Minutes": st.column_config.NumberColumn("Min", format="%.1f", width="small"),
        "FPPM": st.column_config.NumberColumn("FP/M", format="%.2f", width="small"),
        "Lock": st.column_config.CheckboxColumn("🔒 Lock", help="Force this player into the lineup", width="small"), 
        "Exclude": st.column_config.CheckboxColumn("❌ Exclude", help="Ban this player from the lineup", width="small"), 
        "player_id": None, "GameID": None, "bucket": None
    }
    
    # **UPDATED COLUMN ORDER**
    column_order = [
        'Lock', 'Exclude', 'Name', 'positions', 'Team', 'Opponent', 
        'salary', 'proj', 'value', 'own_proj', 'Minutes', 'FPPM'
    ]
    
    # The dataframe passed to data_editor must contain all keys in column_order
    df_for_editor = slate_df[column_order + ['player_id', 'GameID', 'bucket']].copy()

    edited_df = st.data_editor(
        df_for_editor, 
        column_config=column_config,
        column_order=column_order, 
        hide_index=True,
        use_container_width=True,
        height=400,
        key="player_editor_final"
    )
    st.session_state['edited_df'] = edited_df
    
    # Extract Constraints
    edited_df['player_id'] = edited_df['player_id'].astype(str)
    
    locked_player_ids = edited_df[edited_df['Lock'] == True]['player_id'].tolist()
    excluded_player_ids = edited_df[edited_df['Exclude'] == True]['player_id'].tolist()

    if locked_player_ids or excluded_player_ids:
        st.caption(f"🔒 **Locked:** {len(locked_player_ids)} | ❌ **Excluded:** {len(excluded_player_ids)}")

    st.markdown("---")
    
    # --- B. OPTIMIZATION CONTROLS ---
    st.header("2. Find Optimal Lineups")
    
    col_n, col_slack = st.columns(2)
    
    with col_n:
        n_lineups = st.slider("Number of Lineups to Generate (N)", 
                              min_value=1, max_value=20, value=10, step=1,
                              help="The optimizer will find the N highest projected, unique lineups that meet all constraints.")
    
    with col_slack:
        slack = st.slider("Ownership Target Slack (Flexibility)", 
                          min_value=0, max_value=4, value=1, step=1,
                          help="Higher slack allows the optimizer to deviate more from the template's target player counts for each ownership bucket to find a higher projected score.")
    
    
    run_btn = st.button(f"✨ Generate Top {n_lineups} Lineups", use_container_width=True)
    
    if run_btn:
        final_df = st.session_state['edited_df'].copy()
        
        # Recalculate buckets based on potentially edited ownership
        final_df['bucket'] = final_df['own_proj'].apply(ownership_bucket)
        
        # Check for Lock/Exclude conflicts
        conflict = set(locked_player_ids) & set(excluded_player_ids)
        if conflict:
            st.error(f"❌ CONFLICT: Player(s) {', '.join(conflict)} are both locked and excluded.")
            return

        with st.spinner(f'Calculating top {n_lineups} optimal lineups...'):
            top_lineups = generate_top_n_lineups(
                slate_df=final_df,
                template=template,
                n_lineups=n_lineups,
                bucket_slack=slack,
                locked_player_ids=locked_player_ids, 
                excluded_player_ids=excluded_player_ids, 
            )
        
        st.session_state['optimal_lineups_results'] = {
            'lineups': top_lineups, 
            'ran': True
        }
        
        st.success(f"✅ Optimization complete! Found {len(top_lineups)} unique lineups.")

    
    st.markdown("---")
    st.header(f"3. Top {n_lineups} Lineups")
    
    if st.session_state['optimal_lineups_results'].get('ran', False):
        display_multiple_lineups(slate_df, st.session_state['optimal_lineups_results']['lineups'])

    else:
        st.info("Select the number of lineups and click 'Generate Top N Lineups' above to run the multi-lineup builder.")


def tab_contest_analyzer(slate_df, template):
    """Render the Contest Analyzer."""
    st.header("Contest Strategy Analyzer")
    st.info(f"Analysis based on: **{template.contest_label}**")

    st.subheader("Target Ownership Structure")
    ranges = template.bucket_ranges(slack=0) # Display targets with no slack
    
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
    st.set_page_config(layout="wide", page_title="DK Lineup Optimizer")
    
    # Sidebar
    with st.sidebar:
        st.title("🏀 DK Lineup Optimizer")
        st.caption("Maximize Projection based on Template")
        contest_type = st.selectbox("Contest Strategy", ['GPP (Single Entry)', 'GPP (Large Field)', 'CASH'])
        
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
    t1, t2 = st.tabs(["✨ Optimal Lineup Builder", "📝 Contest Analyzer"])
    
    with t1:
        tab_lineup_builder(slate_df, template)
    with t2:
        tab_contest_analyzer(slate_df, template)
