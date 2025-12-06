# app.py

import pandas as pd 
import numpy as np
import streamlit as st 
from typing import Dict, Any, List, Tuple
# Ensure this import block is perfectly copied
from builder import (
    build_template_from_params, 
    run_monte_carlo_simulations, 
    optimize_single_lineup, 
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

            # CRITICAL SALARY CLEANING AND TYPE CONVERSION
            try:
                initial_len_salary = len(df)
                df['salary'] = df['salary'].astype(str).str.strip().str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                # Convert to nullable Int64 initially to handle NaNs/errors, then drop
                df['salary'] = pd.to_numeric(df['salary'], errors='coerce').astype('Int64') 
                df.dropna(subset=['salary'], inplace=True)
                dropped_len_salary = initial_len_salary - len(df)
                
                if dropped_len_salary > 0:
                    st.warning(f"⚠️ Dropped {dropped_len_salary} player(s) due to unfixable salary data.")

                # Final type conversions
                df['salary'] = df['salary'].astype(int) 
                df['proj'] = df['proj'].astype(float)
            except Exception as e:
                # This block is now correctly indented
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
    # Value = Projected Points / (Salary / 1000)
    df['value'] = np.where(df['salary'] > 0, (df['proj'] / (df['salary'] / 1000)).round(2), 0.0)

    # Initialize UI Control Columns
    if 'Lock' not in df.columns: df['Lock'] = False
    if 'Exclude' not in df.columns: df['Exclude'] = False
    if 'Max_Exposure' not in df.columns: df['Max_Exposure'] = 100 # Default to 100%
    
    return df

# --- 2. TAB FUNCTIONS ---

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
        "value": st.column_config.NumberColumn("Value (X)", format="%.2f", disabled=True), 
        "own_proj": st.column_config.NumberColumn("Own %", format="%.1f"), 
        "Lock": st.column_config.CheckboxColumn("🔒 Lock", help="Force this player into the lineup"), 
        "Exclude": st.column_config.CheckboxColumn("❌ Exclude", help="Ban this player from the lineup"), 
        "Max_Exposure": st.column_config.NumberColumn("Max Exposure (%)", min_value=0, max_value=100, default=100, format="%d%%", help="Max % of final lineups player can appear in."),
        "player_id": None, "GameID": None, "Team": None, "Opponent": None
    }
    
    # Define the exact order of columns for the data editor
    column_order = [
        'Lock', 'Exclude', 'Name', 'positions', 'salary', 
        'proj', 'value', 'own_proj', 
        'Team', 'Opponent', 
        'Max_Exposure' # MOVED TO END
    ]
    
    # The Interactive Data Editor
    edited_df = st.data_editor(
        slate_df[column_order + ['player_id', 'GameID']], # Pass all required columns to the data editor
        column_config=column_config,
        column_order=column_order, # Apply the defined order
        hide_index=True,
        use_container_width=True,
        height=400,
        key="player_editor"
    )
    st.session_state['edited_df'] = edited_df
    
    # Extract Constraints
    # CRITICAL FIX: Ensure player_id is string and filter for clean lists
    edited_df['player_id'] = edited_df['player_id'].astype(str)
    
    locked_players_raw = edited_df[edited_df['Lock'] == True]['player_id'].tolist()
    excluded_players_raw = edited_df[edited_df['Exclude'] == True]['player_id'].tolist()

    # Clean the lists: filter out any empty strings or non-string types
    locked_player_ids = [pid for pid in locked_players_raw if isinstance(pid, str) and pid.strip()]
    excluded_player_ids = [pid for pid in excluded_players_raw if isinstance(pid, str) and pid.strip()]

    # Create Max Exposure Dictionary (converted to 0.0 to 1.0 for the builder)
    max_exposures = edited_df.set_index('player_id')['Max_Exposure'].div(100).to_dict()
    
    if locked_player_ids or excluded_player_ids:
        st.caption(f"🔒 **Locked:** {len(locked_player_ids)} | ❌ **Excluded:** {len(excluded_player_ids)}")

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
        
        # --- CRITICAL FIX: TYPE CASTING FOR NUMPY/PULP (Reinforced) ---
        try:
            final_df['proj'] = final_df['proj'].astype(np.float64)
            final_df['salary'] = final_df['salary'].astype(np.float64).astype(int)
        except Exception as e:
            st.error(f"Data type conversion failed before simulation. Check if Projections or Salaries contain non-numeric characters: {e}")
            return
        # --------------------------------------------------------
        
        # Recalculate buckets based on potentially edited ownership
        final_df['bucket'] = final_df['own_proj'].apply(ownership_bucket)
        
        # Check for Lock/Exclude conflicts (using the cleaned lists)
        conflict = set(locked_player_ids) & set(excluded_player_ids)
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
                locked_player_ids=locked_player_ids, # Pass the cleaned lists
                excluded_player_ids=excluded_player_ids, # Pass the cleaned lists
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
        exposure_df = edited_df[['Name', 'positions', 'proj', 'value', 'own_proj', 'Max_Exposure', 'player_id']].copy() 
        exposure_df['Exposure_Pct'] = exposure_df['player_id'].map(final_exposures).fillna(0).round(1)
        
        # Calculate Over-Exposed/Under-Exposed Status
        exposure_df['Status'] = np.where(
            (exposure_df['Exposure_Pct'] > exposure_df['Max_Exposure']) & (exposure_df['Max_Exposure'] < 100), 
            '🚨 Over-Exposed', 
            '✅ OK'
        )
        
        exposure_df.sort_values(by='Exposure_Pct', ascending=False, inplace=True)
        
        exposure_df_display = exposure_df[['Name', 'positions', 'proj', 'value', 'own_proj', 'Max_Exposure', 'Exposure_Pct', 'Status']] 
        
        st.dataframe(
            exposure_df_display.style.format({
                "proj": "{:.1f}", 
                "value": "{:.2f}", 
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
        
        # This is a HACK for display: Assign slots in order.
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
