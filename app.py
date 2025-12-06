import pandas as pd 
import numpy as np
import streamlit as st 
from typing import Dict, Any, List, Tuple
import io 

# NOTE: The 'builder' module is assumed to be correct and unchanged.
from builder import (
    build_template_from_params, 
    generate_top_n_lineups, 
    ownership_bucket,
    PUNT_THR, CHALK_THR, MEGA_CHALK_THR,
    DEFAULT_SALARY_CAP, DEFAULT_ROSTER_SIZE
) 

# --- STREAMLIT CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(layout="wide", page_title="🏀 DK Lineup Optimizer")
# ---------------------------------------------

# --- CONFIGURATION CONSTANTS ---
MIN_GAMES_REQUIRED = 2

# --- HEADER MAPPING ---
REQUIRED_CSV_TO_INTERNAL_MAP = {
    'Player': 'Name',
    'Salary': 'salary', 
    'Position': 'positions',
    'Team': 'Team',
    'Opponent': 'Opponent',
    'PROJECTED FP': 'proj',      
    'OWNERSHIP %': 'own_proj',   
    
    # Fallback for common one-word headers
    'Projection': 'proj',
    'Ownership': 'own_proj',
    
    # Other supporting columns
    'Minutes': 'Minutes',
    'FPPM': 'FPPM',
    'Value': 'Value'
}
CORE_INTERNAL_COLS = ['salary', 'positions', 'proj', 'own_proj', 'Name', 'Team', 'Opponent']

# --- 1. DATA PREPARATION ---

def load_and_preprocess_data(pasted_data: str = None) -> pd.DataFrame:
    """Loads CSV from a pasted string, standardizes ownership, and processes data."""
    
    # Initialization for empty/none data
    empty_df_cols = CORE_INTERNAL_COLS + ['player_id', 'GameID', 'bucket', 'value', 'Lock', 'Exclude']
    if pasted_data is None or not pasted_data.strip():
        df = pd.DataFrame(columns=empty_df_cols)
        return df
        
    try:
        data_io = io.StringIO(pasted_data)
        df = pd.read_csv(data_io)
        st.success("✅ Data pasted successfully. Checking headers...")
        
        # --- CRITICAL FIX: STRIP WHITESPACE FROM ALL HEADERS ---
        df.columns = df.columns.str.strip()
        
        # --- VALIDATE AND MAP REQUIRED COLUMNS ---
        actual_map = {}
        required_internal = ['Name', 'salary', 'positions', 'Team', 'Opponent', 'proj', 'own_proj']
        
        for csv_name, internal_name in REQUIRED_CSV_TO_INTERNAL_MAP.items():
            if csv_name in df.columns:
                actual_map[csv_name] = internal_name
        
        mapped_internal_names = set(actual_map.values())
        final_missing_internal = [name for name in required_internal if name not in mapped_internal_names]

        if final_missing_internal:
            missing_csv_names = [k for k, v in REQUIRED_CSV_TO_INTERNAL_MAP.items() if v in final_missing_internal and k in ['Player', 'Salary', 'Position', 'Team', 'Opponent', 'PROJECTED FP', 'OWNERSHIP %']]
            
            st.error(f"Missing essential headers: The following columns are missing or incorrectly named: {', '.join(set(missing_csv_names))}.")
            st.error("Please ensure your pasted data's first row (headers) exactly matches: Player, Salary, Position, Team, Opponent, PROJECTED FP, OWNERSHIP %.")
            return pd.DataFrame(columns=empty_df_cols)

        df.rename(columns=actual_map, inplace=True)
        
        if not all(col in df.columns for col in CORE_INTERNAL_COLS):
            st.error("Internal processing error: Required columns failed to map correctly.")
            return pd.DataFrame(columns=empty_df_cols)

        
    except Exception as e:
        st.error(f"Error processing pasted data: {e}")
        return pd.DataFrame(columns=empty_df_cols)


    # --- FINAL PROCESSING ---
    
    df['Team'] = df['Team'].astype(str)
    df['Opponent'] = df['Opponent'].astype(str)
    df['GameID'] = df.apply(
        lambda row: '@'.join(sorted([row['Team'], row['Opponent']])), axis=1
    )
    df['player_id'] = df['Name'] 
            
    df['own_proj'] = df['own_proj'].astype(str).str.replace('%', '', regex=False).str.replace(',', '.', regex=False)
    df['own_proj'] = pd.to_numeric(df['own_proj'], errors='coerce')
    df.dropna(subset=CORE_INTERNAL_COLS, inplace=True)

    if df['own_proj'].max() <= 1.0 and df['own_proj'].max() > 0:
            df['own_proj'] = df['own_proj'] * 100
    
    df['own_proj'] = df['own_proj'].round(1)

    try:
        df['salary'] = df['salary'].astype(str).str.strip().str.replace('$', '', regex=False).str.replace(',', '', regex=False)
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce').astype('Int64') 
        df.dropna(subset=['salary'], inplace=True)

        df['salary'] = df['salary'].astype(int) 
        df['proj'] = df['proj'].astype(float)
        
        df_columns = df.columns.tolist()
        
        if 'Minutes' in df_columns:
            df['Minutes'] = pd.to_numeric(df.get('Minutes', 0), errors='coerce').astype(float).round(2)
        if 'FPPM' in df_columns:
            df['FPPM'] = pd.to_numeric(df.get('FPPM', 0), errors='coerce').astype(float).round(2)
        if 'Value' in df_columns:
            df['Value'] = pd.to_numeric(df.get('Value', 0), errors='coerce').astype(float).round(2)
            
    except Exception as e:
        st.error(f"Post-load conversion failed: {e}")
        return pd.DataFrame(columns=empty_df_cols)

    if len(df) == 0:
        st.error("❌ Final player pool is empty after cleaning.")
        return pd.DataFrame(columns=empty_df_cols)

    df['bucket'] = df['own_proj'].apply(ownership_bucket)
    df['value'] = np.where(df['salary'] > 0, (df['proj'] / (df['salary'] / 1000)).round(2), 0.0)

    if 'Lock' not in df.columns: df['Lock'] = False
    if 'Exclude' not in df.columns: df['Exclude'] = False
    
    # Fill any missing columns (e.g., Minutes, FPPM) for stability, although they might not be in the original data
    for col in empty_df_cols:
        if col not in df.columns:
            df[col] = None 
            
    return df

# --- 2. TAB FUNCTIONS ---

if 'optimal_lineups_results' not in st.session_state:
    st.session_state['optimal_lineups_results'] = {'lineups': [], 'ran': False}
if 'edited_df' not in st.session_state:
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
        # --- INDENTATION FIX IS HERE ---
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
        "value": st.column_config.NumberColumn("Value (X)", format="%.2f", disabled=True, width="small"), 
        "own_proj": st.column_config.NumberColumn("Own %", format="%.1f%%", width="small"),
        "Minutes": st.column_config.NumberColumn("Min", format="%.1f", width="small"),
        "FPPM": st.column_config.NumberColumn("FP/M", format="%.2f", width="small"),
        "Lock": st.column_config.CheckboxColumn("🔒 Lock", help="Force this player into the lineup", width="small"), 
        "Exclude": st.column_config.CheckboxColumn("❌ Exclude", help="Ban this player from the lineup", width="small"), 
        "player_id": None, "GameID": None 
    }
    
    column_order = [
        'Lock', 'Exclude', 'Name', 'bucket', 'positions', 'Team', 'Opponent', 
        'salary', 'proj', 'value', 'own_proj', 'Minutes', 'FPPM'
    ]
    
    df_for_editor = slate_df.copy()
    
    if df_for_editor.empty:
        st.info("✍️ Paste your player data into the text area in the sidebar and click the button to load the pool.")
        
        blank_df = pd.DataFrame(columns=column_order)
        edited_df = st.data_editor(
            blank_df, 
            column_config=column_config,
            column_order=column_order, 
            hide_index=True,
            use_container_width=True,
            height=200, 
            key="player_editor_blank"
        )
        st.session_state['edited_df'] = blank_df
        st.markdown("---")
        st.header("2. Find Optimal Lineups")
        st.info("Optimization controls will be enabled once data is loaded.")
        st.markdown("---")
        st.header("3. Top 10 Lineups")
        st.info("Lineup results will appear here.")
        return 

    df_for_editor = df_for_editor[column_order + ['player_id', 'GameID']]
    
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
        
        final_df['bucket'] = final_df['own_proj'].apply(ownership_bucket)
        
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
        display_multiple_lineups(slate_df, template, st.session_state['optimal_lineups_results']['lineups'])

    else:
        st.info("Select the number of lineups and click 'Generate Top N Lineups' above to run the multi-lineup builder.")


def tab_contest_analyzer(slate_df, template):
    """Render the Contest Analyzer."""
    st.header("Contest Strategy Analyzer")
    
    if slate_df.empty:
        st.info("✍️ Paste your data into the sidebar text area to view the contest analyzer.")
        return

    st.info(f"Analysis based on: **{template.contest_label}**")

    st.subheader("Target Ownership Structure")
    ranges = template.bucket_ranges(slack=0) 
    
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
    
    # Sidebar
    with st.sidebar:
        st.title("🏀 DK Lineup Optimizer")
        st.caption("Maximize Projection based on Template")
        
        contest_type = st.selectbox("Contest Strategy", ['GPP (Single Entry)', 'GPP (Large Field)', 'CASH'])
        
        c_map = {'GPP (Single Entry)': 'SE', 'GPP (Large Field)': 'LARGE_GPP', 'CASH': 'CASH'}
        contest_code = c_map[contest_type]
        
        st.divider()
        st.subheader("Paste Player Pool Data (CSV Format)")
        
        pasted_csv_data = st.text_area(
            "Copy your player pool data (including headers) and paste it here.", 
            height=200,
            key="csv_paste_area",
            help="The data should be a table copied directly from a spreadsheet or text file."
        )
        
        load_button = st.button("Load Pasted Data", use_container_width=True)

    # Load Data: Use session state to store the processed DataFrame across runs
    if 'slate_df' not in st.session_state:
        st.session_state['slate_df'] = pd.DataFrame()
        
    if load_button and pasted_csv_data.strip():
        # Load the data and store the processed DataFrame in session state
        st.session_state['slate_df'] = load_and_preprocess_data(pasted_csv_data)
        # Clear the pasted text area to prevent accidental reload/rerun issues
        st.session_state["csv_paste_area"] = "" 
        
    # Use the DataFrame from session state for the rest of the app
    slate_df = st.session_state['slate_df'] 
        
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
