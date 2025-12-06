import pandas as pd 
import numpy as np
import streamlit as st 
from typing import Dict, Any, List, Tuple
import io 
import random

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

# --- CONFIGURATION CONSTANTS ---
MIN_GAMES_REQUIRED = 2

# --- TOURNAMENT TEMPLATES ---
TOURNAMENT_OWNERSHIP_TEMPLATES = {
    "SE": {
        "name": "Single Entry GPP",
        "description": "High leverage, differentiated builds",
        "ownership_targets": {
            "punt": (1, 4),
            "mid": (2, 5),
            "chalk": (1, 4),
            "mega": (0, 2)
        }
    },
    "3MAX": {
        "name": "3-Max GPP",
        "description": "Balanced with variety across 3 entries",
        "ownership_targets": {
            "punt": (1, 4),
            "mid": (2, 5),
            "chalk": (2, 5),
            "mega": (0, 2)
        }
    },
    "20MAX": {
        "name": "20-Max GPP",
        "description": "Diverse builds with exposure control",
        "ownership_targets": {
            "punt": (0, 5),
            "mid": (1, 6),
            "chalk": (1, 5),
            "mega": (0, 3)
        }
    },
    "LARGE_GPP": {
        "name": "Large Field GPP (150-Max)",
        "description": "Maximum leverage, contrarian heavy",
        "ownership_targets": {
            "punt": (2, 5),
            "mid": (1, 4),
            "chalk": (0, 3),
            "mega": (0, 1)
        }
    },
    "CASH": {
        "name": "Cash Game",
        "description": "Safe, high floor, chalk plays",
        "ownership_targets": {
            "punt": (0, 3),
            "mid": (2, 6),
            "chalk": (2, 6),
            "mega": (0, 2)
        }
    },
    "SHOWDOWN": {
        "name": "Showdown Captain Mode",
        "description": "Single game, correlation focused",
        "ownership_targets": {
            "punt": (1, 3),
            "mid": (2, 4),
            "chalk": (1, 3),
            "mega": (0, 2)
        }
    }
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

# --- TOURNAMENT SIMULATOR CLASS ---
class TournamentSimulator:
    def __init__(self, slate_df: pd.DataFrame, field_size: int = 10000):
        self.slate_df = slate_df.copy()
        self.field_size = field_size
        self.slate_df['std_dev'] = self.slate_df['proj'] * 0.25
        
    def simulate_player_score(self, player_row, n_simulations=1):
        return np.random.normal(
            loc=player_row['proj'],
            scale=player_row['std_dev'],
            size=n_simulations
        ).clip(min=0)
    
    def simulate_lineup_score(self, player_ids: List[str], n_simulations=1000):
        lineup_players = self.slate_df[self.slate_df['player_id'].isin(player_ids)]
        simulated_scores = np.zeros(n_simulations)
        
        for _, player in lineup_players.iterrows():
            player_sims = self.simulate_player_score(player, n_simulations)
            simulated_scores += player_sims
            
        return simulated_scores
    
    def generate_field_ownership(self):
        field_usage = {}
        for _, player in self.slate_df.iterrows():
            n_lineups_with_player = int(self.field_size * (player['own_proj'] / 100))
            field_usage[player['player_id']] = n_lineups_with_player
        return field_usage
    
    def calculate_win_probability(self, lineup_player_ids: List[str], n_simulations: int = 1000) -> Dict[str, float]:
        your_scores = self.simulate_lineup_score(lineup_player_ids, n_simulations)
        field_usage = self.generate_field_ownership()
        
        total_field_exposure = sum(field_usage.get(pid, 0) for pid in lineup_player_ids)
        avg_exposure = total_field_exposure / (len(lineup_player_ids) * self.field_size)
        
        avg_field_projection = self.slate_df.nsmallest(8, 'salary')['proj'].sum()
        field_std = avg_field_projection * 0.15
        
        winning_scores = np.random.normal(loc=avg_field_projection * 1.3, scale=field_std, size=n_simulations)
        top10_scores = np.random.normal(loc=avg_field_projection * 1.15, scale=field_std, size=n_simulations)
        
        win_rate = np.mean(your_scores > winning_scores)
        top10_rate = np.mean(your_scores > top10_scores)
        
        return {
            'win_rate': win_rate * 100,
            'top10_rate': top10_rate * 100,
            'avg_score': np.mean(your_scores),
            'ceiling': np.percentile(your_scores, 90),
            'floor': np.percentile(your_scores, 10),
            'ownership_leverage': (1 - avg_exposure) * 100,
            'leverage_score': win_rate * (1 - avg_exposure) * 1000
        }
    
    def analyze_ownership_pattern(self, lineup_player_ids: List[str]) -> Dict[str, any]:
        lineup_players = self.slate_df[self.slate_df['player_id'].isin(lineup_player_ids)]
        bucket_counts = lineup_players['bucket'].value_counts().to_dict()
        avg_ownership = lineup_players['own_proj'].mean()
        
        leverage_score = 0
        for _, player in lineup_players.iterrows():
            player_leverage = (100 - player['own_proj']) / 100
            leverage_score += player_leverage
        leverage_score = leverage_score / len(lineup_players)
        
        return {
            'mega_chalk_count': bucket_counts.get('mega', 0),
            'chalk_count': bucket_counts.get('chalk', 0),
            'mid_count': bucket_counts.get('mid', 0),
            'punt_count': bucket_counts.get('punt', 0),
            'avg_ownership': avg_ownership,
            'leverage_score': leverage_score * 100,
            'strategy': self._classify_strategy(bucket_counts, avg_ownership)
        }
    
    def _classify_strategy(self, bucket_counts, avg_ownership):
        mega = bucket_counts.get('mega', 0)
        chalk = bucket_counts.get('chalk', 0)
        
        if mega >= 2 or chalk >= 4:
            return "Chalk Heavy"
        elif bucket_counts.get('punt', 0) >= 3:
            return "Contrarian Punt"
        elif avg_ownership < 20:
            return "Full Contrarian"
        else:
            return "Balanced"
    
    def compare_strategies(self, lineups_dict: Dict[str, List[str]], n_simulations: int = 1000) -> pd.DataFrame:
        results = []
        
        for name, player_ids in lineups_dict.items():
            win_prob = self.calculate_win_probability(player_ids, n_simulations)
            ownership = self.analyze_ownership_pattern(player_ids)
            
            results.append({
                'Lineup': name,
                'Strategy': ownership['strategy'],
                'Win Rate %': win_prob['win_rate'],
                'Top 10% Rate': win_prob['top10_rate'],
                'Avg Score': win_prob['avg_score'],
                'Ceiling (90th)': win_prob['ceiling'],
                'Floor (10th)': win_prob['floor'],
                'Avg Own %': ownership['avg_ownership'],
                'Leverage Score': win_prob['leverage_score'],
                'Mega Chalk': ownership['mega_chalk_count'],
                'Chalk': ownership['chalk_count'],
                'Mid': ownership['mid_count'],
                'Punt': ownership['punt_count']
            })
        
        return pd.DataFrame(results)

# --- DATA PREPARATION ---
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

    if 'Lock' not in df.columns: 
        df['Lock'] = False
    if 'Exclude' not in df.columns: 
        df['Exclude'] = False
    
    for col in empty_df_cols:
        if col not in df.columns:
            df[col] = None 
            
    return df

# --- SESSION STATE INITIALIZATION ---
if 'optimal_lineups_results' not in st.session_state:
    st.session_state['optimal_lineups_results'] = {'lineups': [], 'ran': False}
if 'edited_df' not in st.session_state:
    st.session_state['edited_df'] = pd.DataFrame(columns=CORE_INTERNAL_COLS + ['player_id', 'GameID', 'bucket', 'value', 'Lock', 'Exclude'])

# --- STYLING FUNCTION ---
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
    slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    assigned_players = set()
    slot_assignments = {}
    
    def can_play_slot(pos_string, slot):
        positions = [p.strip() for p in pos_string.split('/')]
        
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
    
    for slot in slots:
        available = lineup_df[~lineup_df['player_id'].isin(assigned_players)].copy()
        eligible = available[available['positions'].apply(lambda x: can_play_slot(x, slot))]
        
        if len(eligible) == 0:
            return None
            
        chosen = eligible.iloc[0]
        slot_assignments[slot] = chosen['player_id']
        assigned_players.add(chosen['player_id'])
    
    result_df = lineup_df.copy()
    result_df['roster_slot'] = result_df['player_id'].map({v: k for k, v in slot_assignments.items()})
    
    return result_df

# --- TAB FUNCTIONS ---

def display_multiple_lineups(slate_df, template, lineup_list):
    """Function to display the top N optimized lineups with improved UI."""
    
    if not lineup_list:
        st.error("❌ No valid lineups could be found that meet all constraints.")
        st.warning("Try loosening your constraints or reducing the number of lineups requested.")
        return
    
    best_lineup_data = lineup_list[0]
    best_proj = best_lineup_data['proj_score']
    
    best_lineup_players_df = slate_df[slate_df['player_id'].isin(best_lineup_data['player_ids'])]
    best_salary = best_lineup_players_df['salary'].sum()
    best_value = best_proj / (best_salary / 1000) if best_salary else 0
    
    st.subheader("🚀 Top Lineup Metrics (Lineup 1)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Total Projected Points", value=f"{best_proj:.2f}", delta="Optimal Lineup Score")
    with col2:
        st.metric(label="Salary Used", value=f"${best_salary:,}", delta=f"${template.salary_cap - best_salary:,} Remaining")
    with col3:
        st.metric(label="Projection Value (X)", value=f"{best_value:.2f}", delta="Points per $1,000")

    st.markdown("---") 

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
    
    st.dataframe(summary_df.style.format({"Total Proj": "{:.2f}", "Salary Used": "${:,}"}), use_container_width=True)
    
    st.subheader("🔎 Lineup Detail View")
    
    lineup_options = [f"Lineup {i+1} (Proj: {lineup_list[i]['proj_score']:.2f})" for i in range(len(lineup_list))]
    lineup_selection = st.selectbox("Select Lineup for Detail View", options=lineup_options)
    
    lineup_index = lineup_options.index(lineup_selection)
    selected_lineup_data = lineup_list[lineup_index]
    selected_lineup_ids = selected_lineup_data['player_ids']

    lineup_df = slate_df[slate_df['player_id'].isin(selected_lineup_ids)].copy()
    
    lineup_df = assign_lineup_positions(lineup_df)
    
    if lineup_df is None:
        st.error("❌ Could not assign valid roster positions.")
        return
    
    ROSTER_ORDER = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    position_type = pd.CategoricalDtype(ROSTER_ORDER, ordered=True)
    lineup_df['roster_slot'] = lineup_df['roster_slot'].astype(position_type)
    lineup_df.sort_values(by='roster_slot', inplace=True)
    
    display_cols = ['roster_slot', 'Name', 'positions', 'Team', 'Opponent', 'salary', 'proj', 'value', 'own_proj', 'bucket', 'Minutes', 'FPPM'] 
    lineup_df_display = lineup_df[display_cols].reset_index(drop=True)
    
    lineup_df_display.rename(columns={
        'roster_slot': 'SLOT', 
        'positions': 'POS', 
        'own_proj': 'OWN%', 
        'Minutes': 'MIN', 
        'FPPM': 'FP/M', 
        'bucket': 'CATEGORY'
    }, inplace=True)
    
    styled_lineup_df = lineup_df_display.style.applymap(color_bucket, subset=['CATEGORY']).format({
        "salary": "${:,}", 
        "proj": "{:.1f}", 
        "value": "{:.2f}", 
        "OWN%": "{:.1f}%", 
        "MIN": "{:.1f}", 
        "FP/M": "{:.2f}"
    })
    
    st.dataframe(styled_lineup_df, use_container_width=True, hide_index=True)


def tab_lineup_builder(slate_df, template):
    """Render the Interactive Lineup Builder and run the multi-lineup Optimizer."""
    
    tournament_type = st.session_state.get('tournament_type', 'Single Entry GPP')
    tournament_config = st.session_state.get('tournament_config', {})
    contest_code = st.session_state.get('contest_code', 'SE')
    
    st.markdown(f"## 🎯 {tournament_type}")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown(f"**Strategy:** {tournament_config.get('description', 'N/A')}")
        st.caption(f"Field Size: ~{tournament_config.get('field_size', 0):,} entries")
    
    with col2:
        payout_pct = tournament_config.get('top_payout_pct', 0.01) * 100
        st.markdown(f"**Payout Structure:** Top {payout_pct:.2f}% pays")
        
        if payout_pct >= 40:
            st.caption("🟢 High cash rate - prioritize floor/consistency")
        elif payout_pct >= 10:
            st.caption("🟡 Medium cash rate - balance floor and ceiling")
        else:
            st.caption("🔴 Low cash rate - prioritize ceiling/leverage")
    
    with col3:
        rec_lineups = tournament_config.get('recommended_lineups', 1)
        if rec_lineups == 1:
            st.metric("Entries", "1", delta="Single entry")
        else:
            st.metric("Max Entries", rec_lineups)
    
    if contest_code in TOURNAMENT_OWNERSHIP_TEMPLATES:
        template_info = TOURNAMENT_OWNERSHIP_TEMPLATES[contest_code]
        targets = template_info['ownership_targets']
        
        with st.expander("📋 Recommended Ownership Construction"):
            st.markdown(f"**{template_info['name']}** - {template_info['description']}")
            
            construction_df = pd.DataFrame({
                'Bucket': ['Punt (<10%)', 'Mid (10-30%)', 'Chalk (30-40%)', 'Mega (>40%)'],
                'Target Range': [
                    f"{targets['punt'][0]}-{targets['punt'][1]}",
                    f"{targets['mid'][0]}-{targets['mid'][1]}",
                    f"{targets['chalk'][0]}-{targets['chalk'][1]}",
                    f"{targets['mega'][0]}-{targets['mega'][1]}"
                ],
                'Strategy': [
                    '🔵 Low owned leverage plays',
                    '🟢 Solid mid-tier value',
                    '🟡 Popular but good plays',
                    '🔴 Ultra-chalk stars'
                ]
            })
            
            st.table(construction_df)
    
    st.markdown("---")
    
    st.header(f"1. Player Pool & Constraints")
    
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
        st.info("✏️ Paste your player data into the text area in the sidebar and click the button to load the pool.")
        
        blank_df = pd.DataFrame(columns=column_order)
        edited_df = st.data_editor(blank_df, column_config=column_config, column_order=column_order, hide_index=True, use_container_width=True, height=200, key="player_editor_blank")
        st.session_state['edited_df'] = blank_df
        st.markdown("---")
        st.header("2. Find Optimal Lineups")
        st.info("Optimization controls will be enabled once data is loaded.")
        st.markdown("---")
        st.header("3. Top 10 Lineups")
        st.info("Lineup results will appear here.")
        return 

    df_for_editor = df_for_editor[column_order + ['player_id', 'GameID']]
    
    edited_df = st.data_editor(df_for_editor, column_config=column_config, column_order=column_order, hide_index=True, use_container_width=True, height=400, key="player_editor_final")
    st.session_state['edited_df'] = edited_df
    
    edited_df['player_id'] = edited_df['player_id'].astype(str)
    
    locked_player_ids = edited_df[edited_df['Lock'] == True]['player_id'].tolist()
    excluded_player_ids = edited_df[edited_df['Exclude'] == True]['player_id'].tolist()

    if locked_player_ids or excluded_player_ids:
        st.caption(f"🔒 **Locked:** {len(locked_player_ids)} | ❌ **Excluded:** {len(excluded_player_ids)}")

    st.markdown("---")
    
    st.header("2. Find Optimal Lineups")
    
    recommended_n = tournament_config.get('recommended_lineups', 10)
    
    st.info(f"""
    **💡 Recommendation for {tournament_type}:**
    - Generate **{recommended_n}** lineup(s) for this tournament type
    - Field size: ~{tournament_config.get('field_size', 10000):,} entries  
    - {tournament_config.get('description', '')}
    """)
    
    col_n, col_slack = st.columns(2)
    
    with col_n:
        default_n = min(recommended_n, 20)
        n_lineups = st.slider("Number of Lineups to Generate (N)", min_value=1, max_value=150, value=default_n, step=1, help=f"Recommended: {recommended_n} for {tournament_type}")
    
    with col_slack:
        if contest_code == "CASH":
            default_slack = 0
        elif contest_code == "LARGE_GPP":
            default_slack = 2
        else:
            default_slack = 1
            
        slack = st.slider("Ownership Target Slack (Flexibility)", min_value=0, max_value=4, value=default_slack, step=1, help="Higher slack allows more deviation from ownership targets.")
    
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
        
        st.session_state['optimal_lineups_results'] = {'lineups': top_lineups, 'ran': True}
        st.success(f"✅ Optimization complete! Found {len(top_lineups)} unique lineups.")

    st.markdown("---")
    st.header(f"3. Top {n_lineups} Lineups")
    
    if st.session_state['optimal_lineups_results'].get('ran', False):
        display_multiple_lineups(slate_df, template, st.session_state['optimal_lineups_results']['lineups'])
    else:
        st.info("Select the number of lineups and click 'Generate Top N Lineups' above to run the multi-lineup builder.")


def tab_strategy_lab(slate_df, template):
    st.header("🧪 Strategy Lab - Tournament Simulation")
    st.info("Strategy Lab - Coming soon!")


def tab_results_analysis(slate_df, template):
    st.header("📈 Results Analysis - Actual Performance")
    st.info("Results Analysis - Coming soon!")


def tab_contest_analyzer(slate_df, template):
    st.header("📊 Contest Strategy Analyzer")
    st.info("Contest Analyzer - Coming soon!")
if __name__ == '__main__':
    
    # Sidebar
    with st.sidebar:
        st.title("🏀 DK Lineup Optimizer")
        st.caption("Maximize Projection based on Template")
        
        st.subheader("🎯 Tournament Selection")
        
        tournament_type = st.selectbox(
            "Select Tournament Type",
            options=[
                "Single Entry GPP",
                "3-Max GPP", 
                "20-Max GPP",
                "150-Max GPP (Large Field)",
                "Cash Game (50/50, Double Up)",
                "Showdown Captain Mode",
                "Custom"
            ],
            help="Different tournaments require different strategies"
        )
        
        tournament_map = {
            "Single Entry GPP": {"code": "SE", "description": "High leverage, unique builds. Go contrarian to win big.", "field_size": 10000, "top_payout_pct": 0.001, "recommended_lineups": 1},
            "3-Max GPP": {"code": "3MAX", "description": "Balanced variety. Mix chalk with contrarian.", "field_size": 15000, "top_payout_pct": 0.001, "recommended_lineups": 3},
            "20-Max GPP": {"code": "20MAX", "description": "High diversity required. Multiple construction styles.", "field_size": 50000, "top_payout_pct": 0.0005, "recommended_lineups": 20},
            "150-Max GPP (Large Field)": {"code": "LARGE_GPP", "description": "Maximum leverage and diversity needed.", "field_size": 150000, "top_payout_pct": 0.0001, "recommended_lineups": 150},
            "Cash Game (50/50, Double Up)": {"code": "CASH", "description": "Safety first. High floor, chalk plays, consistency.", "field_size": 1000, "top_payout_pct": 0.50, "recommended_lineups": 1},
            "Showdown Captain Mode": {"code": "SHOWDOWN", "description": "Single game. Correlations matter most.", "field_size": 5000, "top_payout_pct": 0.001, "recommended_lineups": 20},
            "Custom": {"code": "SE", "description": "Custom settings", "field_size": 10000, "top_payout_pct": 0.01, "recommended_lineups": 10}
        }
        
        tournament_config = tournament_map[tournament_type]
        
        st.info(f"**Strategy:** {tournament_config['description']}")
        st.caption(f"Avg Field Size: ~{tournament_config['field_size']:,}")
        st.caption(f"Top {tournament_config['top_payout_pct']*100}% pays out")
        
        with st.expander("⚙️ Advanced Tournament Settings"):
            ownership_strategy = "Balanced"
            
            if tournament_type == "Custom":
                custom_field_size = st.number_input("Field Size", min_value=100, max_value=500000, value=10000, step=1000)
                tournament_config['field_size'] = custom_field_size
                
                custom_payout_pct = st.slider("Top % That Pays", min_value=0.01, max_value=50.0, value=1.0, step=0.01, format="%.2f%%") / 100
                tournament_config['top_payout_pct'] = custom_payout_pct
            
            ownership_strategy = st.select_slider("Ownership Strategy", options=["Full Chalk", "Balanced", "Contrarian", "Max Leverage"], value="Balanced", help="Override the default strategy for this tournament type")
            
            min_salary = st.slider("Minimum Salary to Use", min_value=40000, max_value=50000, value=48000, step=500, help="Don't leave too much salary on the table")
        
        if ownership_strategy == "Full Chalk":
            contest_code = "CASH"
        elif ownership_strategy == "Contrarian":
            contest_code = "LARGE_GPP"
        elif ownership_strategy == "Max Leverage":
            contest_code = "LARGE_GPP"
        else:
            contest_code = tournament_config['code']
        
        st.divider()
        st.subheader("Paste Player Pool Data (CSV Format)")
        
        pasted_csv_data = st.text_area("Copy your player pool data (including headers) and paste it here:", height=200, placeholder="Player\tSalary\tPosition\tTeam\tOpponent\tMinutes\tFPPM\tProjection\tValue\tOwnership")
        
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
    
    # Store tournament config
    st.session_state['tournament_type'] = tournament_type
    st.session_state['tournament_config'] = tournament_config
    st.session_state['contest_code'] = contest_code
    st.session_state['ownership_strategy'] = ownership_strategy
    
    # Initialize slate_df
    if 'slate_df' not in st.session_state:
        st.session_state['slate_df'] = pd.DataFrame(columns=CORE_INTERNAL_COLS + ['player_id', 'GameID', 'bucket', 'value', 'Lock', 'Exclude'])
    
    slate_df = st.session_state['slate_df']
    
    # Build template
    template = build_template_from_params(
        contest_type=contest_code,
        field_size=tournament_config['field_size'],
        pct_to_first=tournament_config['top_payout_pct'],
        roster_size=DEFAULT_ROSTER_SIZE,
        salary_cap=DEFAULT_SALARY_CAP,
        min_games=MIN_GAMES_REQUIRED
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏗️ Lineup Builder", "🧪 Strategy Lab", "📈 Results Analysis", "📊 Contest Analyzer"])
    
    with tab1:
        tab_lineup_builder(slate_df, template)
    
    with tab2:
        tab_strategy_lab(slate_df, template)
    
    with tab3:
        tab_results_analysis(slate_df, template)
    
    with tab4:
        tab_contest_analyzer(slate_df, template)
