import pandas as pd 
import numpy as np
import streamlit as st 
from typing import Dict, Any, List
import io 

from builder import (
    build_template_from_params, 
    generate_top_n_lineups, 
    ownership_bucket,
    PUNT_THR, CHALK_THR, MEGA_CHALK_THR,
    DEFAULT_SALARY_CAP, DEFAULT_ROSTER_SIZE
) 

# --- STREAMLIT CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(layout="wide", page_title="🏀 DK Lineup Optimizer")

# --- CONFIGURATION CONSTANTS ---
MIN_GAMES_REQUIRED = 2

# --- STRATEGY PROFILES (CASH / SE / 3-MAX / 20-MAX / MILLY) ---

STRATEGY_PROFILES = {
    "CASH": {
        "name": "Cash Games (50/50, Double-Up)",
        "description": "Beat ~50% of the field. Embrace projection and floor.",
        "chalk_range": (4, 6),        # Players over 20% owned
        "mid_range": (2, 3),          # 10–20%
        "contrarian_range": (0, 1),   # <10%
        "total_own_range": (260, 340),
        "salary_leave_range": (0, 500),
        "core_locks": (4, 6),
        "player_pool_size": (10, 20),
        "priority": "Projection & floor over uniqueness"
    },
    "SE": {
        "name": "Single Entry GPP",
        "description": "Smaller field GPP. Stay mostly with the field, pick 1–2 smart pivots.",
        "chalk_range": (5, 6),
        "mid_range": (1, 2),
        "contrarian_range": (1, 1),
        "total_own_range": (120, 150),
        "salary_leave_range": (0, 300),
        "core_locks": (4, 5),
        "player_pool_size": (15, 20),
        "priority": "Prioritize projection, add 1–2 leverage spots"
    },
    "3MAX": {
        "name": "3-Max GPP",
        "description": "Three bullets. A little more leverage than SE, still projection-heavy.",
        "chalk_range": (3, 4),
        "mid_range": (1, 2),
        "contrarian_range": (1, 1),
        "total_own_range": (120, 150),
        "salary_leave_range": (0, 300),
        "core_locks": (4, 5),
        "player_pool_size": (25, 35),
        "priority": "Projection first, one solid contrarian angle"
    },
    "20MAX": {
        "name": "20-Max GPP",
        "description": "Mid-sized GPP. Need projection + leverage across a 20-lineup set.",
        "chalk_range": (2, 3),
        "mid_range": (1, 2),
        "contrarian_range": (1, 2),
        "total_own_range": (120, 160),
        "salary_leave_range": (0, 400),
        "core_locks": (3, 5),
        "player_pool_size": (35, 45),
        "priority": "Mix of projection and differentiated builds"
    },
    "MILLIMAX": {
        "name": "Large Field / Milly Maker",
        "description": "Massive fields. Need strong projection *and* serious leverage.",
        "chalk_range": (1, 2),
        "mid_range": (2, 3),
        "contrarian_range": (2, 4),
        "total_own_range": (140, 220),
        "salary_leave_range": (0, 800),
        "core_locks": (2, 4),
        "player_pool_size": (45, 60),
        "priority": "Ceiling and leverage; avoid fully chalky constructions"
    },
}

# Entry fee tiers and their typical field sizes
ENTRY_FEE_TIERS = {
    "$0.25": {"min_field": 100, "max_field": 2500, "skill_level": "Beginner"},
    "$1": {"min_field": 100, "max_field": 5000, "skill_level": "Recreational"},
    "$3": {"min_field": 500, "max_field": 10000, "skill_level": "Intermediate"},
    "$5": {"min_field": 500, "max_field": 25000, "skill_level": "Intermediate"},
    "$10": {"min_field": 1000, "max_field": 50000, "skill_level": "Advanced"},
    "$20": {"min_field": 2000, "max_field": 100000, "skill_level": "Advanced"},
    "$50+": {"min_field": 5000, "max_field": 200000, "skill_level": "Expert/Shark"}
}

# --- HEADER MAPPING ---
REQUIRED_CSV_TO_INTERNAL_MAP = {
    'Player': 'Name',
    'Salary': 'salary', 
    'Position': 'positions',
    'Team': 'Team',
    'Opponent': 'Opponent',
    'PROJECTED FP': 'proj',
    'Projection': 'proj',
    'Proj': 'proj',
    'OWNERSHIP %': 'own_proj',
    'Ownership': 'own_proj',
    'Own': 'own_proj',
    'Own%': 'own_proj',
    'Minutes': 'Minutes',
    'FPPM': 'FPPM',
    'Value': 'Value'
}
CORE_INTERNAL_COLS = ['salary', 'positions', 'proj', 'own_proj', 'Name', 'Team', 'Opponent']

# --- EDGE-FINDING ENGINE ----------------------------------------------------


def calculate_player_leverage(row):
    # Expected optimal % based on value
    expected_optimal_pct = (row['value'] / 5.0) * 100  # 5.0x value = 100% optimal
    expected_optimal_pct = min(expected_optimal_pct, 100)
    leverage = expected_optimal_pct - row['own_proj']
    return round(leverage, 1)


def calculate_ceiling_score(row):
    base_ceiling = row['proj'] * 1.35
    salary_factor = (10000 - row['salary']) / 10000
    ceiling_boost = base_ceiling * salary_factor * 0.2
    return round(base_ceiling + ceiling_boost, 1)


def assign_edge_category(row):
    if row['leverage_score'] > 15 and row['value'] > 4.5:
        return "🔥 Elite Leverage"
    elif row['leverage_score'] > 10:
        return "⭐ High Leverage"
    elif row['leverage_score'] > 5:
        return "✅ Good Leverage"
    elif row['leverage_score'] > -5:
        return "➖ Neutral"
    elif row['leverage_score'] > -15:
        return "⚠️ Slight Chalk"
    else:
        return "❌ Chalk Trap"


def calculate_gpp_score(row):
    ceiling_component = (row['ceiling'] / 100) * 0.4
    value_component = (row['value'] / 7) * 0.3
    leverage_normalized = (row['leverage_score'] + 20) / 40
    leverage_component = max(0, leverage_normalized) * 0.3
    gpp_score = (ceiling_component + value_component + leverage_component) * 100
    return round(gpp_score, 1)


# --- DATA PREPARATION -------------------------------------------------------


def load_and_preprocess_data(pasted_data: str = None) -> pd.DataFrame:
    empty_df_cols = CORE_INTERNAL_COLS + ['player_id', 'GameID', 'bucket', 'value', 'Lock', 'Exclude']
    if pasted_data is None or not pasted_data.strip():
        return pd.DataFrame(columns=empty_df_cols)
        
    try:
        data_io = io.StringIO(pasted_data)
        first_line = pasted_data.split('\n')[0]
        if '\t' in first_line:
            df = pd.read_csv(data_io, sep='\t')
        else:
            df = pd.read_csv(data_io)
        st.success("✅ Data pasted successfully. Checking headers...")
        df.columns = df.columns.str.strip()
        
        actual_map = {}
        required_internal = ['Name', 'salary', 'positions', 'Team', 'Opponent', 'proj', 'own_proj']
        for csv_name, internal_name in REQUIRED_CSV_TO_INTERNAL_MAP.items():
            if csv_name in df.columns:
                actual_map[csv_name] = internal_name
        
        mapped_internal_names = set(actual_map.values())
        final_missing_internal = [name for name in required_internal if name not in mapped_internal_names]
        if final_missing_internal:
            st.error("❌ Missing essential columns. Please ensure your data has these required headers:")
            st.error("**Required:** Player, Salary, Position, Team, Opponent")
            st.error("**Projection column:** One of: 'Projection', 'PROJECTED FP', or 'Proj'")
            st.error("**Ownership column:** One of: 'Ownership', 'OWNERSHIP %', 'Own', or 'Own%'")
            st.error(f"**Missing:** {', '.join(final_missing_internal)}")
            return pd.DataFrame(columns=empty_df_cols)

        df.rename(columns=actual_map, inplace=True)
        if not all(col in df.columns for col in CORE_INTERNAL_COLS):
            st.error("Internal processing error: Required columns failed to map correctly.")
            return pd.DataFrame(columns=empty_df_cols)
        
    except Exception as e:
        st.error(f"Error processing pasted data: {e}")
        return pd.DataFrame(columns=empty_df_cols)

    df['Team'] = df['Team'].astype(str)
    df['Opponent'] = df['Opponent'].astype(str)
    df['GameID'] = df.apply(lambda row: '@'.join(sorted([row['Team'], row['Opponent']])), axis=1)
    df['player_id'] = df['Name'] 
            
    df['own_proj'] = df['own_proj'].astype(str).str.replace('%', '', regex=False).str.replace(',', '.', regex=False)
    df['own_proj'] = pd.to_numeric(df['own_proj'], errors='coerce')
    if df['own_proj'].max() <= 1.0 and df['own_proj'].max() > 0:
        df['own_proj'] = df['own_proj'] * 100
    df['own_proj'] = df['own_proj'].round(1)
    df.dropna(subset=CORE_INTERNAL_COLS, inplace=True)

    try:
        df['salary'] = df['salary'].astype(str).str.strip()
        df['salary'] = df['salary'].str.replace('$', '', regex=False)
        df['salary'] = df['salary'].str.replace(',', '', regex=False)
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
    
    df['leverage_score'] = df.apply(calculate_player_leverage, axis=1)
    df['ceiling'] = df.apply(calculate_ceiling_score, axis=1)
    df['edge_category'] = df.apply(assign_edge_category, axis=1)
    df['gpp_score'] = df.apply(calculate_gpp_score, axis=1)

    if 'Lock' not in df.columns: 
        df['Lock'] = False
    if 'Exclude' not in df.columns: 
        df['Exclude'] = False
    
    for col in empty_df_cols:
        if col not in df.columns:
            df[col] = None 
    return df


# --- SESSION STATE INITIALIZATION -------------------------------------------

if 'optimal_lineups_results' not in st.session_state:
    st.session_state['optimal_lineups_results'] = {'lineups': [], 'ran': False}
if 'edited_df' not in st.session_state:
    st.session_state['edited_df'] = pd.DataFrame(
        columns=CORE_INTERNAL_COLS + ['player_id', 'GameID', 'bucket', 'value', 'Lock', 'Exclude']
    )


# --- STYLING HELPERS --------------------------------------------------------


def color_bucket(s):
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


def assign_lineup_positions(lineup_df):
    """Assign roster positions to players in a lineup for display only."""
    slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    assigned_players = set()
    slot_assignments = {}
    
    def can_play_slot(pos_string, slot):
        if pd.isna(pos_string):
            return False
        positions = [p.strip() for p in str(pos_string).split('/')]
        if slot == 'PG':
            return 'PG' in positions
        elif slot == 'SG':
            return 'SG' in positions
        elif slot == 'SF':
            return 'SF' in positions
        elif slot == 'PF':
            return 'PF' in positions
        elif slot == 'C':
            return 'C' in positions
        elif slot == 'G':
            return 'PG' in positions or 'SG' in positions
        elif slot == 'F':
            return 'SF' in positions or 'PF' in positions
        elif slot == 'UTIL':
            return True
        return False
    
    def get_position_flexibility(pos_string):
        if pd.isna(pos_string):
            return 0
        positions = [p.strip() for p in str(pos_string).split('/')]
        flex_count = len(positions)
        if 'PG' in positions or 'SG' in positions:
            flex_count += 1
        if 'SF' in positions or 'PF' in positions:
            flex_count += 1
        return flex_count
    
    specific_slots = ['C', 'PG', 'SG', 'SF', 'PF']
    flex_slots = ['G', 'F', 'UTIL']
    ordered_slots = specific_slots + flex_slots
    
    success = True
    for slot in ordered_slots:
        available = lineup_df[~lineup_df['player_id'].isin(assigned_players)].copy()
        if len(available) == 0:
            success = False
            break
        eligible = available[available['positions'].apply(lambda x: can_play_slot(x, slot))]
        if len(eligible) == 0:
            success = False
            break
        if slot in specific_slots:
            eligible['flexibility'] = eligible['positions'].apply(get_position_flexibility)
            eligible = eligible.sort_values('flexibility')
        chosen = eligible.iloc[0]
        slot_assignments[slot] = chosen['player_id']
        assigned_players.add(chosen['player_id'])
    
    if not success:
        # Fallback: just assign UTIL to everyone
        result_df = lineup_df.copy()
        result_df['roster_slot'] = 'UTIL'
        return result_df
    
    result_df = lineup_df.copy()
    result_df['roster_slot'] = result_df['player_id'].map({v: k for k, v in slot_assignments.items()})
    return result_df


# --- STRATEGY SCORING -------------------------------------------------------


def score_lineup_against_profile(lineup_players: pd.DataFrame, profile_key: str, salary_cap: int) -> Dict[str, Any]:
    profile = STRATEGY_PROFILES.get(profile_key, STRATEGY_PROFILES['SE'])
    
    # Ownership buckets for strategy purposes
    own = lineup_players['own_proj'].fillna(0)
    chalk_count = int((own >= 20).sum())
    mid_count = int(((own >= 10) & (own < 20)).sum())
    contrarian_count = int((own < 10).sum())
    total_own = float(own.sum())
    
    used_salary = int(lineup_players['salary'].sum())
    salary_left = max(salary_cap - used_salary, 0)
    
    def range_penalty(val, rng):
        lo, hi = rng
        if val < lo:
            return (lo - val) ** 2
        if val > hi:
            return (val - hi) ** 2
        return 0.0
    
    pen_chalk = range_penalty(chalk_count, profile['chalk_range'])
    pen_mid = range_penalty(mid_count, profile['mid_range'])
    pen_contra = range_penalty(contrarian_count, profile['contrarian_range'])
    pen_own = range_penalty(total_own, profile['total_own_range'])
    pen_sal = range_penalty(salary_left, profile['salary_leave_range'])
    
    total_penalty = (
        2.0 * pen_chalk
        + 1.5 * pen_mid
        + 2.0 * pen_contra
        + 1.0 * pen_own
        + 0.5 * pen_sal
    )
    
    return {
        "chalk_count": chalk_count,
        "mid_count": mid_count,
        "contrarian_count": contrarian_count,
        "total_own": total_own,
        "salary_used": used_salary,
        "salary_left": salary_left,
        "strategy_penalty": total_penalty,
    }


# --- DISPLAY MULTIPLE LINEUPS -----------------------------------------------


def display_multiple_lineups(slate_df, template, lineup_list, profile_key: str):
    if not lineup_list:
        st.error("❌ No valid lineups could be found that meet all constraints.")
        st.warning("Try loosening your constraints or reducing the number of lineups requested.")
        return
    
    has_actuals = st.session_state.get('has_actuals', False)
    
    best_lineup_data = lineup_list[0]
    best_proj = best_lineup_data['proj_score']
    
    best_lineup_players_df = slate_df[slate_df['player_id'].isin(best_lineup_data['player_ids'])]
    best_salary = best_lineup_players_df['salary'].sum()
    best_value = best_proj / (best_salary / 1000) if best_salary else 0
    
    if has_actuals and 'actual_pts' in best_lineup_players_df.columns:
        best_actual = best_lineup_players_df['actual_pts'].sum()
        has_lineup_actuals = best_lineup_players_df['actual_pts'].notna().all()
    else:
        has_lineup_actuals = False
    
    st.subheader("🚀 Top Lineup Metrics (Lineup 1)")
    if has_lineup_actuals:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Projected Points", value=f"{best_proj:.2f}")
        with col2:
            st.metric(label="Actual Points", value=f"{best_actual:.2f}", delta=f"{best_actual - best_proj:+.2f}")
        with col3:
            st.metric(label="Salary Used", value=f"${best_salary:,}")
        with col4:
            actual_value = best_actual / (best_salary / 1000) if best_salary else 0
            st.metric(label="Actual Value", value=f"{actual_value:.2f}x", delta=f"{actual_value - best_value:+.2f}x")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Projected Points", value=f"{best_proj:.2f}", delta="Optimal-ish")
        with col2:
            st.metric(label="Salary Used", value=f"${best_salary:,}", delta=f"${template.salary_cap - best_salary:,} Remaining")
        with col3:
            st.metric(label="Projection Value (X)", value=f"{best_value:.2f}", delta="Points per $1,000")
    
    st.markdown("---")
    
    st.subheader("📋 Lineup Summary vs Strategy")
    summary_rows = []
    for i, lineup_data in enumerate(lineup_list):
        lineup_players_df = slate_df[slate_df['player_id'].isin(lineup_data['player_ids'])]
        strat = score_lineup_against_profile(lineup_players_df, profile_key, template.salary_cap)
        
        row = {
            "Lineup": i + 1,
            "Proj": lineup_data['proj_score'],
            "Total Own%": strat['total_own'],
            "Chalk": strat['chalk_count'],
            "Mid": strat['mid_count'],
            "Contrarian": strat['contrarian_count'],
            "Salary": strat['salary_used'],
            "Salary Left": strat['salary_left'],
            "Strategy Penalty": strat['strategy_penalty'],
        }
        if has_actuals and 'actual_pts' in lineup_players_df.columns and lineup_players_df['actual_pts'].notna().all():
            actual_score = lineup_players_df['actual_pts'].sum()
            row['Actual'] = actual_score
            row['Diff'] = actual_score - lineup_data['proj_score']
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows).set_index('Lineup')
    
    if 'Actual' in summary_df.columns:
        summary_df = summary_df.sort_values(['Strategy Penalty', 'Actual'], ascending=[True, False])
    else:
        summary_df = summary_df.sort_values(['Strategy Penalty', 'Proj'], ascending=[True, False])
    
    st.dataframe(
        summary_df.style.format({
            "Proj": "{:.2f}",
            "Actual": "{:.2f}",
            "Diff": "{:+.2f}",
            "Total Own%": "{:.1f}",
            "Salary": "${:,}",
            "Salary Left": "${:,}",
            "Strategy Penalty": "{:.1f}",
        }),
        use_container_width=True
    )
    
    st.markdown("---")
    st.subheader("🔎 Lineup Detail View")
    
    lineup_options = [f"Lineup {i+1} (Proj: {lineup_list[i]['proj_score']:.2f})" for i in range(len(lineup_list))]
    lineup_selection = st.selectbox("Select Lineup", options=lineup_options)
    lineup_index = lineup_options.index(lineup_selection)
    selected_lineup_data = lineup_list[lineup_index]
    
    selected_lineup_ids = selected_lineup_data['player_ids']
    lineup_df = slate_df[slate_df['player_id'].isin(selected_lineup_ids)].copy()
    
    if len(lineup_df) != template.roster_size:
        st.error(f"❌ Invalid lineup size: {len(lineup_df)} players (expected {template.roster_size})")
        st.write("Players in lineup:", lineup_df['Name'].tolist())
        return
    
    lineup_df = assign_lineup_positions(lineup_df)
    
    ROSTER_ORDER = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    position_type = pd.CategoricalDtype(ROSTER_ORDER, ordered=True)
    lineup_df['roster_slot'] = lineup_df['roster_slot'].astype(position_type)
    lineup_df.sort_values(by='roster_slot', inplace=True)
    
    cols = ['roster_slot', 'Name', 'positions', 'Team', 'Opponent', 'salary', 'proj', 'value', 'own_proj', 'bucket']
    if 'actual_pts' in lineup_df.columns:
        cols.append('actual_pts')
    lineup_df_display = lineup_df[cols].reset_index(drop=True)
    
    rename_dict = {'roster_slot': 'SLOT', 'positions': 'POS', 'own_proj': 'Proj Own%', 'bucket': 'CATEGORY', 'proj': 'Proj Pts'}
    if 'actual_pts' in lineup_df_display.columns:
        rename_dict['actual_pts'] = 'Actual Pts'
    lineup_df_display.rename(columns=rename_dict, inplace=True)
    
    styled = lineup_df_display.style.applymap(color_bucket, subset=['CATEGORY']).format({
        "salary": "${:,}",
        "Proj Pts": "{:.1f}",
        "value": "{:.2f}",
        "Proj Own%": "{:.1f}%",
        "Actual Pts": "{:.1f}",
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)


# --- MAIN TAB: LINEUP BUILDER -----------------------------------------------


def tab_lineup_builder(slate_df, template, profile_key: str):
    tournament_type = st.session_state.get('tournament_type', 'Single Entry GPP')
    tournament_config = st.session_state.get('tournament_config', {})
    
    st.markdown(f"## 🎯 {tournament_type}")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.markdown(f"**Strategy Profile:** {STRATEGY_PROFILES[profile_key]['name']}")
        st.caption(f"Field Size: ~{tournament_config.get('field_size', 0):,} entries")
    with col2:
        st.markdown(f"**Priority:** {STRATEGY_PROFILES[profile_key]['priority']}")
        st.caption(STRATEGY_PROFILES[profile_key]['description'])
    with col3:
        st.metric("Entries", tournament_config.get('recommended_lineups', 1))
    
    st.markdown("---")
    st.header("1. Player Pool + Edge View")
    
    column_config = {
        "Name": st.column_config.TextColumn("Player", disabled=True, width="medium"), 
        "edge_category": st.column_config.TextColumn("Edge", disabled=True, width="medium"),
        "gpp_score": st.column_config.NumberColumn("GPP Score", disabled=True, format="%.1f", width="small"),
        "leverage_score": st.column_config.NumberColumn("Leverage", disabled=True, format="%+.1f", width="small"),
        "ceiling": st.column_config.NumberColumn("Ceiling", disabled=True, format="%.1f", width="small"),
        "positions": st.column_config.TextColumn("Pos", disabled=True, width="small"), 
        "salary": st.column_config.NumberColumn("Salary", format="$%d", width="small"), 
        "proj": st.column_config.NumberColumn("Proj", format="%.1f", width="small"), 
        "value": st.column_config.NumberColumn("Value", format="%.2f", disabled=True, width="small"), 
        "own_proj": st.column_config.NumberColumn("Own%", format="%.1f%%", width="small"),
        "Lock": st.column_config.CheckboxColumn("🔒", help="Lock into lineup", width="small"), 
        "Exclude": st.column_config.CheckboxColumn("❌", help="Exclude from lineups", width="small"), 
        "Team": None, 
        "Opponent": None, 
        "bucket": None, 
        "Minutes": None, 
        "FPPM": None,
        "player_id": None,
        "GameID": None
    }
    
    column_order = [
        'Lock', 'Exclude', 'Name', 'edge_category', 'gpp_score', 'leverage_score',
        'positions', 'salary', 'proj', 'ceiling', 'value', 'own_proj', 'player_id'
    ]
    
    df_for_editor = slate_df.copy()
    if df_for_editor.empty:
        st.info("✏️ Paste your player data into the text area in the sidebar and click the button to load the pool.")
        blank_df = pd.DataFrame(columns=column_order)
        st.data_editor(blank_df, column_config=column_config, column_order=column_order,
                       hide_index=True, use_container_width=True, height=200,
                       key="player_editor_blank")
        st.markdown("---")
        st.header("2. Contest Strategy")
        st.info("Strategy guidance will appear once data is loaded.")
        st.markdown("---")
        st.header("3. Build Lineups")
        st.info("Lineup builder will be enabled after data is loaded.")
        return 
    
    for col in column_order:
        if col not in df_for_editor.columns:
            if col in ('Lock', 'Exclude'):
                df_for_editor[col] = False
            else:
                df_for_editor[col] = None
    df_for_editor = df_for_editor[column_order]
    
    edited_df = st.data_editor(
        df_for_editor, 
        column_config=column_config, 
        column_order=column_order, 
        hide_index=True, 
        use_container_width=True, 
        height=420, 
        key="player_editor_final"
    )
    st.session_state['edited_df'] = edited_df
    edited_df['player_id'] = edited_df['player_id'].astype(str)
    
    locked_player_ids = edited_df[edited_df['Lock'] == True]['player_id'].tolist()
    excluded_player_ids = edited_df[edited_df['Exclude'] == True]['player_id'].tolist()
    if locked_player_ids or excluded_player_ids:
        st.caption(f"🔒 Locked: {len(locked_player_ids)} | ❌ Excluded: {len(excluded_player_ids)}")
    
    st.markdown("---")
    st.header("2. Contest Strategy Summary")
    
    prof = STRATEGY_PROFILES[profile_key]
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Lineup Construction Targets (per lineup)**")
        st.table(pd.DataFrame({
            "Bucket": ["Chalk (>20%)", "Mid (10–20%)", "Contrarian (<10%)", "Total Ownership", "Salary Left"],
            "Target Range": [
                f"{prof['chalk_range'][0]}–{prof['chalk_range'][1]} players",
                f"{prof['mid_range'][0]}–{prof['mid_range'][1]} players",
                f"{prof['contrarian_range'][0]}–{prof['contrarian_range'][1]} players",
                f"{prof['total_own_range'][0]}–{prof['total_own_range'][1]}%",
                f"$ {prof['salary_leave_range'][0]} – $ {prof['salary_leave_range'][1]}",
            ],
        }))
    with col_r:
        st.markdown("**Core & Player Pool Guidance (for multi-entry)**")
        st.table(pd.DataFrame({
            "Item": ["Core Players to Lock", "Recommended Player Pool Size"],
            "Guideline": [
                f"{prof['core_locks'][0]}–{prof['core_locks'][1]} players",
                f"{prof['player_pool_size'][0]}–{prof['player_pool_size'][1]} players"
            ]
        }))
    
    st.markdown("---")
    st.header("3. Generate Strategy-Aligned Lineups")
    
    n_lineups = st.slider("Number of lineups to build", min_value=1, max_value=40, value=10)
    auto_match = st.checkbox("Automatically match this contest's strategy profile", value=True)
    
    run_btn = st.button(f"🚀 Build {n_lineups} Lineups", type="primary", use_container_width=True)
    
    if run_btn:
        conflict = set(locked_player_ids) & set(excluded_player_ids)
        if conflict:
            st.error(f"❌ Conflict: {', '.join(conflict)} are both locked and excluded.")
            return
        
        final_df = st.session_state['edited_df'].copy()
        merge_cols = ['player_id', 'bucket', 'GameID', 'Team', 'Opponent',
                      'edge_category', 'leverage_score', 'ceiling',
                      'gpp_score', 'value']
        merge_cols = [c for c in merge_cols if c in slate_df.columns]
        drop_cols = [c for c in merge_cols if c in final_df.columns and c != 'player_id']
        if drop_cols:
            final_df = final_df.drop(columns=drop_cols)
        final_df = final_df.merge(slate_df[merge_cols], on='player_id', how='left')
        
        with st.spinner(f"Building {n_lineups} lineups..."):
            # Get a bigger candidate set, then score vs strategy
            raw_lineups = generate_top_n_lineups(
                slate_df=final_df,
                template=template,
                n_lineups=n_lineups * 5,
                bucket_slack=2,
                locked_player_ids=locked_player_ids,
                excluded_player_ids=excluded_player_ids,
            )
            
            if not raw_lineups:
                st.error("❌ Could not generate any valid lineups. Try loosening constraints or locks.")
                return
            
            scored = []
            for lu in raw_lineups:
                players_df = final_df[final_df['player_id'].isin(lu['player_ids'])]
                strat = score_lineup_against_profile(players_df, profile_key, template.salary_cap)
                lu2 = lu.copy()
                lu2.update(strat)
                scored.append(lu2)
            
            if auto_match:
                scored.sort(key=lambda x: (x['strategy_penalty'], -x['proj_score']))
            else:
                scored.sort(key=lambda x: -x['proj_score'])
            
            top_lineups = scored[:n_lineups]
            st.session_state['optimal_lineups_results'] = {'lineups': top_lineups, 'ran': True}
            st.success(f"✅ Built {len(top_lineups)} strategy-aligned lineups!")
    
    st.markdown("---")
    st.header("4. Your Lineups")
    
    if st.session_state['optimal_lineups_results'].get('ran', False):
        display_multiple_lineups(
            slate_df,
            template,
            st.session_state['optimal_lineups_results']['lineups'],
            profile_key
        )
    else:
        st.info("Build lineups above and they'll appear here.")


# --- OTHER TABS (stubs for now) ---------------------------------------------


def tab_strategy_lab(slate_df, template):
    st.header("🧪 Strategy Lab")
    st.info("Coming soon – deeper sim tools.")


def tab_results_analysis(slate_df, template):
    st.header("📈 Results Analysis")
    st.info("Coming soon – import CSV of results and see how your lineups did.")


def tab_contest_analyzer(slate_df, template):
    st.header("📊 Contest Analyzer")
    st.info("Coming soon – analyze contest fields and ownership distributions.")


# --- MAIN APP ---------------------------------------------------------------

if __name__ == '__main__':
    # Sidebar
    with st.sidebar:
        st.title("🏀 DK Lineup Optimizer")
        st.caption("Research-backed contest templates")
        
        st.subheader("🎯 Contest Details")
        
        tournament_type = st.selectbox(
            "Tournament Type",
            options=[
                "Cash Game (50/50, Double-Up)",
                "Single Entry GPP",
                "3-Max GPP", 
                "20-Max GPP",
                "Large Field GPP (Milly Maker)",
            ],
        )
        
        entry_fee = st.selectbox(
            "Entry Fee",
            options=list(ENTRY_FEE_TIERS.keys()),
            index=2,
            help="Higher entry fees = tougher competition"
        )
        fee_info = ENTRY_FEE_TIERS[entry_fee]
        
        field_size = st.slider(
            "Expected Field Size",
            min_value=fee_info["min_field"],
            max_value=fee_info["max_field"],
            value=min(10000, fee_info["max_field"]),
            step=100 if fee_info["max_field"] < 10000 else 1000,
            help="Larger fields = need more leverage"
        )
        st.caption(f"💪 Competition Level: **{fee_info['skill_level']}**")
        
        # Map UI tournament choice to internal code + strategy profile
        tournament_map = {
            "Cash Game (50/50, Double-Up)": ("CASH", "CASH"),
            "Single Entry GPP": ("SE", "SE"),
            "3-Max GPP": ("3MAX", "3MAX"),
            "20-Max GPP": ("20MAX", "20MAX"),
            "Large Field GPP (Milly Maker)": ("LARGE_GPP", "MILLIMAX"),
        }
        contest_code, profile_key = tournament_map[tournament_type]
        
        # Simple payout % / recommended lines
        payout_map = {
            "CASH": (0.45, 1),
            "SE": (0.20, 1),
            "3MAX": (0.20, 3),
            "20MAX": (0.15, 20),
            "LARGE_GPP": (0.20, 150),
        }
        top_payout_pct, rec_lineups = payout_map[contest_code]
        
        st.info(f"Top ~{top_payout_pct*100:.0f}% of field cashes. Recommended entries: {rec_lineups}.")
        
        st.divider()
        st.subheader("📊 Load Player Data")
        pasted_csv_data = st.text_area(
            "Copy your player pool data (including headers) and paste it here:",
            height=200,
            placeholder="Player\tSalary\tPosition\tTeam\tOpponent\tMinutes\tFPPM\tProjection\tValue\tOwnership",
        )
        load_data_btn = st.button("Load Pasted Data", use_container_width=True)
        if load_data_btn:
            if pasted_csv_data and pasted_csv_data.strip():
                with st.spinner("Processing your data..."):
                    loaded_df = load_and_preprocess_data(pasted_csv_data)
                    st.session_state['slate_df'] = loaded_df
                    if not loaded_df.empty:
                        st.success(f"✅ Loaded {len(loaded_df)} players successfully!")
                    else:
                        st.error("❌ Failed to load data. Check the format and try again.")
            else:
                st.warning("⚠️ Please paste some data first!")
        
        tournament_config = {
            "field_size": field_size,
            "entry_fee": entry_fee,
            "top_payout_pct": top_payout_pct,
            "recommended_lineups": rec_lineups,
        }
    
    st.session_state['tournament_type'] = tournament_type
    st.session_state['tournament_config'] = tournament_config
    st.session_state['contest_code'] = contest_code
    st.session_state['profile_key'] = profile_key
    
    if 'slate_df' not in st.session_state:
        st.session_state['slate_df'] = pd.DataFrame(
            columns=CORE_INTERNAL_COLS + ['player_id', 'GameID', 'bucket', 'value', 'Lock', 'Exclude']
        )
    slate_df = st.session_state['slate_df']
    
    template = build_template_from_params(
        contest_type=contest_code,
        field_size=field_size,
        pct_to_first=top_payout_pct,
        roster_size=DEFAULT_ROSTER_SIZE,
        salary_cap=DEFAULT_SALARY_CAP,
        min_games=MIN_GAMES_REQUIRED,
    )
    
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🏗️ Lineup Builder", "🧪 Strategy Lab", "📈 Results Analysis", "📊 Contest Analyzer"]
    )
    with tab1:
        tab_lineup_builder(slate_df, template, profile_key)
    with tab2:
        tab_strategy_lab(slate_df, template)
    with tab3:
        tab_results_analysis(slate_df, template)
    with tab4:
        tab_contest_analyzer(slate_df, template)
