# app.py

import pandas as pd 
import numpy as np
import streamlit as st 
from typing import Dict, Any, List, Tuple
# Import the new multi-lineup generation function
from builder import (
    build_template_from_params, 
    generate_top_n_lineups, 
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
        # Data loading and preprocessing logic (as previously verified)
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Data loaded successfully.")

            if 'Player' in df.columns:
                df.rename(columns={'Player': 'Name'}, inplace=True)
            
            missing_csv_cols = [col for col in REQUIRED_CSV_TO_INTERNAL_MAP.keys() if col not in df.columns]
            if missing_csv_cols:
                st.error(f"Missing headers: {missing_csv_cols}. Required: {list(REQUIRED_CSV_TO_INTERNAL_MAP.keys())}")
                return pd.DataFrame()
            
            df.rename(columns=REQUIRED_CSV_TO_INTERNAL_MAP, inplace=True)
            
            required_game_cols = ['Team', 'Opponent']
            if 'GameID' not in df.columns:
                if all(col in df.columns for col in required_game_cols):
                    df['GameID'] = df.apply(
                        lambda row: '@'.join(sorted([str(row['Team']), str(row['Opponent'])])), axis=1
                    )
                else:
                    st.error("Missing required columns: **Team** and **Opponent**.")
                    return pd.DataFrame()
            
            df['player_id'] = df['Name'] 
            
            df['own_proj'] = df['own_proj'].astype(str).str.replace('%', '', regex=False).str.replace(',', '.', regex=False)
            df['own_proj'] = pd.to_numeric(df['own_proj'], errors='coerce')
            df.dropna(subset=['own_proj'], inplace=True)

            if df['own_proj'].max() <= 1.0:
                 df['own_proj'] = df['own_proj'] * 100
            
            df['own_proj'] = df['own_proj'].round(1)

            try:
                initial_len_salary = len(df)
                df['salary'] = df['salary'].astype(str).str.strip().str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                df['salary'] = pd.to_numeric(df['salary'], errors='coerce').astype('Int64') 
                df.dropna(subset=['salary'], inplace=True)
                dropped_len_salary = initial_len_salary - len(df)
                
                if dropped_len_salary > 0:
                    st.warning(f"⚠️ Dropped {dropped_len_salary} player(s) due to unfixable salary data.")

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
    display_cols = ['roster_position', 'Name', 'positions', 'Team', 'GameID', 'salary', 'proj', 'value', 'own_proj', 'bucket'] 
    lineup_df_display = lineup_df[display_cols].reset_index(drop=True)
    
    # 4. Rename the column for display
    lineup_df_display.rename(columns={'roster_position': 'SLOT'}, inplace=True)
    
    # Display the detailed lineup
    st.dataframe(
        lineup_df_display.style.format({"salary": "${:,}", "proj": "{:.1f}", "value": "{:.2f}", "own_proj": "{:.1f}%"}), 
        use_container_width=True,
        hide_index=True 
    )


def tab_lineup_builder(slate_df, template):
    """Render the Interactive Lineup Builder and run the multi-lineup Optimizer."""
    st.header(f"1. Player Pool & Constraints for **{template.contest_label}**")
    
    # --- A. PLAYER POOL EDITOR ---
    st.markdown("Use the table to **Lock** or **Exclude** players for the optimal lineup.")
    
    column_config = {
        "Name": st.column_config.TextColumn("Player Name", disabled=True), 
        "positions": st.column_config.TextColumn("Pos", disabled=True), 
        "salary": st.column_config.NumberColumn("Salary", format="$%d"), 
        "proj": st.column_config.NumberColumn("Proj Pts", format="%.1f"), 
        "value": st.column_config.NumberColumn("Value (X)", format="%.2f", disabled=True), 
        "own_proj": st.column_config.NumberColumn("Own %", format="%.1f"), 
        "Lock": st.column_config.CheckboxColumn("🔒 Lock", help="Force this player into the lineup"), 
        "Exclude": st.column_config.CheckboxColumn("❌ Exclude", help="Ban this player from the lineup"), 
        "player_id": None, "GameID": None, "Team": None, "Opponent": None
    }
    
    column_order = [
        'Lock', 'Exclude', 'Name', 'positions', 'salary', 
        'proj', 'value', 'own_proj', 
        'Team', 'Opponent', 
    ]
    
    edited_df = st.data_editor(
        slate_df[column_order + ['player_id', 'GameID']], 
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
        
        # The download button is commented out as the export logic is complex
        # st.download_button(
        #     label="⬇️ Download All Lineups (CSV)",
        #     data="Example CSV Data",
        #     file_name="top_lineups.csv",
        #     mime="text/csv",
        #     disabled=True, 
        # )

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
