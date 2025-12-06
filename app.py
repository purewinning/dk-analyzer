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

# ... [Continue with all the tab functions - I'll provide them in the next message to stay within limits]

# --- MAIN ENTRY POINT ---
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
    
    # Placeholder for tabs - will add full tab functions in requirements
    st.info("App structure loaded successfully. Tab functions loading...")
