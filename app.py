# app.py

import pandas as pd 
import numpy as np
import streamlit as st 
from typing import Dict, Any, List, Tuple
# Ensure this import block is perfectly copied
from builder import (
    build_template_from_params, 
    run_monte_carlo_simulations, # <-- NEW IMPORT
    optimize_single_lineup, # <-- NEW IMPORT
    ownership_bucket,
    PUNT_THR, CHALK_THR, MEGA_CHALK_THR,
    DEFAULT_SALARY_CAP, DEFAULT_ROSTER_SIZE
) 

# --- CONFIGURATION CONSTANTS ---
MIN_GAMES_REQUIRED = 2
DEFAULT_ITERATIONS = 500
DEFAULT_DIVERSITY = 4 # Max shared players in any two final lineups

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
    if 'Max_Exposure' not in df.columns: df['Max_Exposure'] = 100 # Default to 100%
    
    return df

# --- 2. HELPER: RANDOMIZE PROJECTIONS (No longer needed, replaced by MCS) ---

# --- 3. TAB FUNCTIONS ---

# Use session state to store simulation results
if 'sim_results' not in st.session_state:
    st.session_state['sim_results'] = {'lineups': [], 'exposures': {}}
if 'edited_df' not in st.session_state:
    st.session_state['edited_df'] = pd.DataFrame()


def tab_lineup_builder(slate_df, template):
    """Render the Interactive Lineup Builder and run MCS."""
    st.header("1. Player Pool & Constraints")
    
    # --- A. PLAYER POOL EDITOR ---
    st.markdown("Use the table to **Lock**, **Exclude**, or set **Max Exposure** for Monte Carlo.")
    
    # Define column config for the editor (CONDENSED)
    column_config = {
        "Name": st.column_config.TextColumn("Player Name", disabled=True), 
        "positions": st.column_config.TextColumn("Pos", disabled=True), 
        "salary": st.column_config.NumberColumn("Salary", format="$%d"), 
        "proj": st.column_config.NumberColumn("Proj Pts", format="%.1f"), 
        "own_proj": st.column_config.NumberColumn("Own %", format="%.1f"), 
        "Lock": st.column_config.CheckboxColumn("🔒 Lock", help="Force this player into the lineup"), 
        "Exclude": st.column_config.CheckboxColumn("❌ Exclude", help="Ban this player from the lineup"), 
        "Max_Exposure": st.column_config.NumberColumn("Max Exposure (%)", min_value=0, max_value=100, default=100, format="%d%%", help="Max % of final lineups player can appear in."),
        "player_id": None, "GameID": None, "Team": None, "Opponent": None
    }
    
    # The Interactive Data Editor
    edited_df = st.data_editor(
        slate_df[['Lock', 'Exclude', 'Max_Exposure', 'Name', 'positions', 'salary', 'proj', 'own_proj', 'Team', 'Opponent', 'GameID', 'player_id']],
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        height=400,
        key="player_editor"
    )
    st.session_state['edited_df'] = edited_df
    
    # Extract Constraints
    locked_players = edited_df[edited_df['Lock'] == True]['player_id'].tolist()
    excluded_players = edited_df[edited_df['Exclude'] == True]['player_id'].tolist()
    
    # Create Max Exposure Dictionary (converted to 0.0 to 1.0 for the builder)
    max_exposures = edited_df.set_index('player_id')['Max_Exposure'].div(100).to_dict()
    
    if locked_players or excluded_players:
        st.caption(f"🔒 **Locked:** {len(locked_players)} | ❌ **Excluded:** {len(excluded_players)}")

    st.markdown("---")
    
    # --- B. SIMULATION SETTINGS ---
    st.header("2. Run Monte Carlo Simulation")
    
    col_iter, col_div, col_btn = st.columns([2, 2, 1])
    
    with col_iter:
        iterations = st.number_input("Simulation Iterations (Higher is better)", 
                                     min_value=100, max_value=5000, value=DEFAULT_ITERATIONS, step=100)
    
    with col_div:
        diversity = st.number_input("Lineup Diversity (Max Shared Players)", 
                                    min_value=1, max_value=7, value=DEFAULT_DIVERSITY, 
                                    help="Maximum number of shared players allowed between any two final lineups.")
    
    with col_btn:
        st.write("") # Spacer
        run_btn = st.button("🚀 Run Simulation", use_container_width=True)
    
    if run_btn:
        final_df = edited_df.copy()
        
        # Recalculate buckets based on potentially edited ownership
        final_df['bucket'] = final_df['own_proj'].apply(ownership_bucket)
        
        # Check for Lock/Exclude conflicts
        conflict = set(locked_players) & set(excluded_players)
        if conflict:
            st.error(f"❌ CONFLICT: Player(s) {', '.join(conflict)} are both locked and excluded.")
            return

        with st.spinner(f'Running {iterations} Monte Carlo simulations...'):
            final_lineups, final_exposures = run_monte_carlo_simulations(
                slate_df=final_df,
                template=template,
                num_iterations=iterations,
                max_exposures=max_exposures,
                bucket_slack=1,
                locked_player_ids=locked_players,
                excluded_player_ids=excluded_players,
                min_lineup_diversity=diversity
            )
        
        st.session_state['sim_results'] = {
            'lineups': final_lineups, 
            'exposures': final_exposures,
            'ran': True
        }
        
        if final_lineups:
            st.success(f"✅ Simulation complete! Found {len(final_lineups)} diverse, optimal lineups.")
        else:
            st.warning("⚠️ No valid lineups found. Check your hard constraints (Locks, Excludes, Salary).")
            st.session_state['sim_results']['ran'] = False

def tab_simulation_results(slate_df):
    """Render the results of the Monte Carlo Simulation."""
    if st.session_state['sim_results'].get('ran', False) and st.session_state['sim_results']['lineups']:
        st.header("3. Monte Carlo Simulation Results")
        
        final_lineups = st.session_state['sim_results']['lineups']
        final_exposures = st.session_state['sim_results']['exposures']
        edited_df = st.session_state['edited_df']
        
        st.subheader(f"Player Exposure ({len(final_lineups)} Lineups)")

        # --- A. EXPOSURE TABLE ---
        exposure_df = edited_df[['Name', 'positions', 'proj', 'own_proj', 'Max_Exposure', 'player_id']].copy()
        exposure_df['Exposure_Pct'] = exposure_df['player_id'].map(final_exposures).fillna(0).round(1)
        
        # Calculate Over-Exposed/Under-Exposed Status
        exposure_df['Status'] = np.where(
            (exposure_df['Exposure_Pct'] > exposure_df['Max_Exposure']) & (exposure_df['Max_Exposure'] < 100), 
            '🚨 Over-Exposed', 
            '✅ OK'
        )
        
        exposure_df.sort_values(by='Exposure_Pct', ascending=False, inplace=True)
        
        exposure_df_display = exposure_df[['Name', 'positions', 'proj', 'own_proj', 'Max_Exposure', 'Exposure_Pct', 'Status']]
        
        st.dataframe(
            exposure_df_display.style.format({
                "proj": "{:.1f}", 
                "own_proj": "{:.1f}%", 
                "Max_Exposure": "{:.0f}%", 
                "Exposure_Pct": "{:.1f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        st.subheader("Generated Lineups (Highest Projection First)")

        # --- B. LINEUP DISPLAY ---
        
        # User selection for lineup
        lineup_index = st.selectbox("Select Lineup #", 
                                    options=list(range(1, len(final_lineups) + 1)), 
                                    format_func=lambda x: f"Lineup {x} (Proj: {final_lineups[x-1]['proj_score']:.2f})")
        
        selected_lineup_data = final_lineups[lineup_index - 1]
        selected_lineup_ids = selected_lineup_data['player_ids']
        
        # Rebuild the dataframe for display
        lineup_df = slate_df[slate_df['player_id'].isin(selected_lineup_ids)].copy()
        
        # 1. Assign Roster Position
        ROSTER_ORDER = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
        
        # Match players to their required primary slot (this is still a HACK for display)
        # For a truly accurate display, the solver must return the assigned slot.
        # This implementation simply assigns the slots in order of the ROSTER_ORDER list.
        lineup_df = lineup_df.head(8).assign(roster_position=ROSTER_ORDER)

        # 2. Sort the DataFrame by the custom position order
        position_type = pd.CategoricalDtype(ROSTER_ORDER, ordered=True)
        lineup_df['roster_position'] = lineup_df['roster_position'].astype(position_type)
        lineup_df.sort_values(by='roster_position', inplace=True)
        
        # 3. Define display columns
        display_cols = ['roster_position', 'Name', 'positions', 'Team', 'GameID', 'salary', 'proj', 'own_proj', 'bucket']
        lineup_df_display = lineup_df[display_cols].reset_index(drop=True)
        
        # 4. Rename the column for display
        lineup_df_display.rename(columns={'roster_position': 'SLOT'}, inplace=True)
        
        # Display Metrics
        total_salary = lineup_df['salary'].sum()
        total_points = lineup_df['proj'].sum()
        games_used = lineup_df['GameID'].nunique()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Projection", f"{total_points:.2f}")
        col2.metric("Salary Used", f"${total_salary:,}")
        col3.metric("Games", f"{games_used}")
        
        # Display Lineup
        st.dataframe(
            lineup_df_display.style.format({"salary": "${:,}", "proj": "{:.1f}", "own_proj": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True 
        )

    else:
        st.info("Run the simulation on the 'Lineup Builder' tab first to see results here.")


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
        st.title("🏀 DK Lineup Sim")
        st.caption("Monte Carlo & Max Exposure")
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
    t1, t2, t3 = st.tabs(["🚀 Lineup Builder & Sim", "📊 Exposure & Results", "📝 Contest Analyzer"])
    
    with t1:
        tab_lineup_builder(slate_df, template)
    with t2:
        tab_simulation_results(slate_df)
    with t3:
        tab_contest_analyzer(slate_df, template)
