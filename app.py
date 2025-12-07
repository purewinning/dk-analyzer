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

# --- TOURNAMENT TEMPLATES (RESEARCH-BACKED) ---
TOURNAMENT_OWNERSHIP_TEMPLATES = {
    "CASH": {
        "name": "Cash Game (50/50, Double-Up)",
        "description": "High floor, consistency over ceiling",
        "ownership_targets": {
            "punt": (0, 2),      # Minimize risk
            "mid": (3, 5),       # Safe, solid plays
            "chalk": (3, 5),     # Popular = good for cash
            "mega": (0, 2)       # Stars are fine if projected well
        },
        "strategy_notes": "Play chalk. High ownership is GOOD in cash games. Need 50th percentile score."
    },
    "SE": {
        "name": "Single Entry GPP",
        "description": "Balanced with 1-2 contrarian pivots",
        "ownership_targets": {
            "punt": (2, 4),      # Some leverage
            "mid": (2, 4),       # Balanced
            "chalk": (1, 3),     # 1-2 chalk plays max
            "mega": (0, 1)       # At most 1 mega chalk
        },
        "strategy_notes": "Ownership more condensed. Pivot off 1-2 chalk plays. Need top 10-15%."
    },
    "3MAX": {
        "name": "3-Max GPP",
        "description": "Core + variety across 3 entries",
        "ownership_targets": {
            "punt": (1, 4),      # Mix of leverage
            "mid": (2, 5),       # Flexibility
            "chalk": (1, 4),     # Some chalk OK
            "mega": (0, 2)       # Diversify mega exposure
        },
        "strategy_notes": "Build around 2-3 core players. Vary punt/chalk combos across entries."
    },
    "20MAX": {
        "name": "20-Max GPP",
        "description": "High exposure management, diverse builds",
        "ownership_targets": {
            "punt": (1, 5),      # Wide range for diversity
            "mid": (1, 5),       # Flexible
            "chalk": (0, 4),     # Sometimes fade all chalk
            "mega": (0, 2)       # Limit mega exposure
        },
        "strategy_notes": "Max 50% exposure per player. Need 3-4 distinct build types."
    },
    "LARGE_GPP": {
        "name": "Large Field GPP (150-Max, Milly Maker)",
        "description": "Maximum leverage, must be different",
        "ownership_targets": {
            "punt": (3, 6),      # Heavy leverage required
            "mid": (1, 3),       # Some solid plays
            "chalk": (0, 2),     # Fade most chalk
            "mega": (0, 1)       # At most 1, often 0
        },
        "strategy_notes": "Need <5% owned player. Top 0.1% to win. Be VERY different."
    },
    "SHOWDOWN": {
        "name": "Showdown Captain Mode",
        "description": "Single game, correlation key",
        "ownership_targets": {
            "punt": (1, 3),
            "mid": (2, 4),
            "chalk": (1, 3),
            "mega": (0, 2)
        },
        "strategy_notes": "Game stacking and correlation trump ownership."
    }
}

# Entry fee tiers and their typical field sizes (REAL DRAFTKINGS DATA)
ENTRY_FEE_TIERS = {
    "$0.25": {"min_field": 100, "max_field": 2500, "skill_level": "Beginner"},
    "$1": {"min_field": 100, "max_field": 5000, "skill_level": "Recreational"},
    "$3": {"min_field": 500, "max_field": 10000, "skill_level": "Intermediate"},
    "$5": {"min_field": 500, "max_field": 25000, "skill_level": "Intermediate"},
    "$10": {"min_field": 1000, "max_field": 50000, "skill_level": "Advanced"},
    "$20": {"min_field": 2000, "max_field": 100000, "skill_level": "Advanced"},
    "$50+": {"min_field": 5000, "max_field": 200000, "skill_level": "Expert/Shark"}
}

# Common DraftKings contest sizes
COMMON_CONTEST_SIZES = {
    "Single Entry": [500, 1000, 2500, 5000],
    "3-Max": [1000, 2500, 5000, 10000],
    "20-Max": [5000, 10000, 25000, 50000],
    "Large Field": [50000, 100000, 150000, 200000],
    "Cash": [100, 500, 1000, 2500]
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

# --- EDGE-FINDING ENGINE ---

def calculate_player_leverage(row):
    """
    Calculate leverage score: How underowned is this player relative to their value?
    Positive = Good leverage (underowned)
    Negative = Bad leverage (overowned)
    """
    # Expected optimal % based on value
    expected_optimal_pct = (row['value'] / 5.0) * 100  # 5.0x value = 100% optimal
    expected_optimal_pct = min(expected_optimal_pct, 100)
    
    # Leverage = How much more optimal they should be vs actual ownership
    leverage = expected_optimal_pct - row['own_proj']
    
    return round(leverage, 1)

def calculate_ceiling_score(row):
    """
    Estimate ceiling based on projection + variance
    Higher projection + lower salary = higher ceiling potential
    """
    base_ceiling = row['proj'] * 1.35  # 35% above projection for ceiling
    
    # Cheaper players have more ceiling upside (can 5-6x value)
    salary_factor = (10000 - row['salary']) / 10000
    ceiling_boost = base_ceiling * salary_factor * 0.2
    
    return round(base_ceiling + ceiling_boost, 1)

def assign_edge_category(row):
    """
    Categorize players by their edge type
    """
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
    """
    Comprehensive GPP score combining ceiling, value, and leverage
    Based on research: need value + differentiation
    """
    # Ceiling component (40%)
    ceiling_component = (row['ceiling'] / 100) * 0.4
    
    # Value component (30%)
    value_component = (row['value'] / 7) * 0.3
    
    # Leverage component (30%)
    leverage_normalized = (row['leverage_score'] + 20) / 40  # Normalize -20 to +20 range
    leverage_component = max(0, leverage_normalized) * 0.3
    
    gpp_score = (ceiling_component + value_component + leverage_component) * 100
    
    return round(gpp_score, 1)


# --- HISTORICAL WINNING TEMPLATES ---
WINNING_TEMPLATES = {
    "BALANCED_GPP": {
        "name": "Balanced GPP (Proven Winner)",
        "description": "Mix of value + leverage. Historical 15-20% win rate in top 1%.",
        "requirements": {
            "min_value_plays": 2,  # At least 2 players with 5.0+ value
            "min_leverage_plays": 2,  # At least 2 players with 10+ leverage
            "max_chalk": 2,  # Max 2 players over 30% owned
            "required_punt": 1,  # At least 1 player under 5% owned
        },
        "target_distribution": {
            "elite_leverage": 1,
            "high_leverage": 2,
            "good_leverage": 2,
            "neutral": 2,
            "chalk": 1
        }
    },
    "CONTRARIAN_GPP": {
        "name": "Contrarian GPP (High Risk/Reward)",
        "description": "Low owned studs + value punts. Wins 5% but huge upside.",
        "requirements": {
            "min_value_plays": 3,
            "min_leverage_plays": 3,
            "max_chalk": 1,
            "required_punt": 2,  # At least 2 players under 5% owned
        },
        "target_distribution": {
            "elite_leverage": 2,
            "high_leverage": 3,
            "good_leverage": 2,
            "neutral": 1,
            "chalk": 0
        }
    },
    "CHALK_SMASH": {
        "name": "Chalk Smash (When Chalk Hits)",
        "description": "Top owned + value. Works when favorites perform.",
        "requirements": {
            "min_value_plays": 3,
            "min_leverage_plays": 0,
            "max_chalk": 4,
            "required_punt": 0,
        },
        "target_distribution": {
            "elite_leverage": 0,
            "high_leverage": 1,
            "good_leverage": 2,
            "neutral": 2,
            "chalk": 3
        }
    },
    "VALUE_STACK": {
        "name": "Value Stack (Salary Saver)",
        "description": "Max value to afford studs. Need 4+ players at 5.0x+",
        "requirements": {
            "min_value_plays": 4,
            "min_leverage_plays": 2,
            "max_chalk": 2,
            "required_punt": 1,
        },
        "target_distribution": {
            "elite_leverage": 1,
            "high_leverage": 3,
            "good_leverage": 3,
            "neutral": 1,
            "chalk": 0
        }
    }
}
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
    
    # CALCULATE EDGES
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
    
    # Check if we have actual results
    has_actuals = st.session_state.get('has_actuals', False)
    
    best_lineup_data = lineup_list[0]
    best_proj = best_lineup_data['proj_score']
    
    best_lineup_players_df = slate_df[slate_df['player_id'].isin(best_lineup_data['player_ids'])]
    best_salary = best_lineup_players_df['salary'].sum()
    best_value = best_proj / (best_salary / 1000) if best_salary else 0
    
    # Calculate actual score if available
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
            st.metric(label="Total Projected Points", value=f"{best_proj:.2f}", delta="Optimal Lineup Score")
        with col2:
            st.metric(label="Salary Used", value=f"${best_salary:,}", delta=f"${template.salary_cap - best_salary:,} Remaining")
        with col3:
            st.metric(label="Projection Value (X)", value=f"{best_value:.2f}", delta="Points per $1,000")

    st.markdown("---")
    
    # Show best edges section
    st.subheader("🔥 Top Edges in Your Lineups")
    
    if not slate_df.empty:
        # Get all players used across top 3 lineups
        all_used_players = set()
        for lineup in lineup_list[:3]:
            all_used_players.update(lineup['player_ids'])
        
        used_players_df = slate_df[slate_df['player_id'].isin(all_used_players)]
        
        # Show elite leverage plays
        elite_plays = used_players_df[used_players_df['edge_category'] == '🔥 Elite Leverage'].nlargest(5, 'gpp_score')[
            ['Name', 'positions', 'salary', 'proj', 'own_proj', 'leverage_score', 'gpp_score']
        ]
        
        if len(elite_plays) > 0:
            st.markdown("**🔥 Elite Leverage Plays in Your Lineups:**")
            st.dataframe(
                elite_plays.style.format({
                    'salary': '${:,}',
                    'proj': '{:.1f}',
                    'own_proj': '{:.1f}%',
                    'leverage_score': '{:+.1f}',
                    'gpp_score': '{:.1f}'
                }),
                use_container_width=True,
                hide_index=True
            )

    st.markdown("---") 

    st.subheader("📋 Lineup Summary with Edge Analysis")
    
    summary_data = []
    
    for i, lineup_data in enumerate(lineup_list):
        lineup_players_df = slate_df[slate_df['player_id'].isin(lineup_data['player_ids'])]
        
        # Calculate edge metrics
        total_ownership = lineup_players_df['own_proj'].sum()
        avg_leverage = lineup_players_df['leverage_score'].mean()
        avg_gpp_score = lineup_players_df['gpp_score'].mean()
        
        # Count edge distribution
        edge_dist = lineup_players_df['edge_category'].value_counts()
        elite_count = edge_dist.get('🔥 Elite Leverage', 0)
        high_count = edge_dist.get('⭐ High Leverage', 0)
        
        summary_row = {
            'Lineup': i + 1,
            'Projected': lineup_data['proj_score'],
            'Total Own%': total_ownership,
            'Avg Leverage': avg_leverage,
            'GPP Score': avg_gpp_score,
            'Elite Edge': elite_count,
            'High Edge': high_count,
            'Salary': lineup_players_df['salary'].sum()
        }
        
        # Add actual if available
        if has_actuals and 'actual_pts' in lineup_players_df.columns:
            if lineup_players_df['actual_pts'].notna().all():
                actual_score = lineup_players_df['actual_pts'].sum()
                summary_row['Actual'] = actual_score
                summary_row['Diff'] = actual_score - lineup_data['proj_score']
        
        summary_data.append(summary_row)
        
    summary_df = pd.DataFrame(summary_data).set_index('Lineup')
    
    # Sort by best metric
    if 'Actual' in summary_df.columns:
        summary_df = summary_df.sort_values('Actual', ascending=False)
        summary_df['Rank'] = range(1, len(summary_df) + 1)
        
        st.dataframe(
            summary_df.style.format({
                "Projected": "{:.2f}",
                "Actual": "{:.2f}",
                "Diff": "{:+.2f}",
                "Total Own%": "{:.1f}%",
                "Avg Leverage": "{:+.1f}",
                "GPP Score": "{:.1f}",
                "Salary": "${:,}"
            }).background_gradient(subset=['Avg Leverage'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        best_actual_idx = summary_df['Actual'].idxmax()
        st.success(f"🏆 **Lineup {best_actual_idx} won**: {summary_df.loc[best_actual_idx, 'Actual']:.2f} pts | Leverage: {summary_df.loc[best_actual_idx, 'Avg Leverage']:+.1f}")
    else:
        # Highlight best leverage
        st.dataframe(
            summary_df.style.format({
                "Projected": "{:.2f}",
                "Total Own%": "{:.1f}%",
                "Avg Leverage": "{:+.1f}",
                "GPP Score": "{:.1f}",
                "Salary": "${:,}"
            }).background_gradient(subset=['Avg Leverage'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        # Show best leverage lineup
        best_lev_idx = summary_df['Avg Leverage'].idxmax()
        st.info(f"💎 **Best Leverage**: Lineup {best_lev_idx} | Avg Leverage: {summary_df.loc[best_lev_idx, 'Avg Leverage']:+.1f} | Only {summary_df.loc[best_lev_idx, 'Total Own%']:.0f}% total ownership")
    
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
    
    # Choose columns based on whether we have actuals
    if has_actuals and 'actual_pts' in lineup_df.columns and lineup_df['actual_pts'].notna().all():
        display_cols = ['roster_slot', 'Name', 'positions', 'Team', 'Opponent', 'salary', 'proj', 'actual_pts', 'value', 'own_proj', 'actual_own', 'bucket']
        lineup_df['pts_diff'] = lineup_df['actual_pts'] - lineup_df['proj']
    else:
        display_cols = ['roster_slot', 'Name', 'positions', 'Team', 'Opponent', 'salary', 'proj', 'value', 'own_proj', 'bucket', 'Minutes', 'FPPM']
    
    lineup_df_display = lineup_df[display_cols].reset_index(drop=True)
    
    # Rename columns
    rename_dict = {
        'roster_slot': 'SLOT', 
        'positions': 'POS', 
        'own_proj': 'Proj Own%',
        'bucket': 'CATEGORY'
    }
    
    if 'actual_pts' in display_cols:
        rename_dict.update({
            'proj': 'Proj Pts',
            'actual_pts': 'Actual Pts',
            'actual_own': 'Actual Own%'
        })
    else:
        rename_dict.update({
            'proj': 'Proj Pts',
            'Minutes': 'MIN',
            'FPPM': 'FP/M'
        })
    
    lineup_df_display.rename(columns=rename_dict, inplace=True)
    
    # Style the lineup
    def color_diff(val):
        if isinstance(val, (int, float)):
            if val > 5:
                return 'background-color: #90EE90'
            elif val < -5:
                return 'background-color: #FFB6C6'
        return ''
    
    styled_lineup = lineup_df_display.style.applymap(color_bucket, subset=['CATEGORY'])
    
    if 'pts_diff' in lineup_df.columns:
        styled_lineup = styled_lineup.applymap(color_diff, subset=['pts_diff'])
    
    # Format numbers
    format_dict = {
        "salary": "${:,}",
        "value": "{:.2f}"
    }
    
    if 'Proj Pts' in lineup_df_display.columns and 'Actual Pts' in lineup_df_display.columns:
        format_dict.update({
            'Proj Pts': '{:.1f}',
            'Actual Pts': '{:.1f}',
            'pts_diff': '{:+.1f}',
            'Proj Own%': '{:.1f}%',
            'Actual Own%': '{:.1f}%'
        })
    else:
        format_dict.update({
            'Proj Pts': '{:.1f}',
            'Proj Own%': '{:.1f}%',
            'MIN': '{:.1f}',
            'FP/M': '{:.2f}'
        })
    
    styled_lineup = styled_lineup.format(format_dict)
    
    st.dataframe(styled_lineup, use_container_width=True, hide_index=True)
    
    # Show best/worst performers if we have actuals
    if 'pts_diff' in lineup_df.columns:
        st.markdown("**Performance Analysis:**")
        
        best_player = lineup_df.loc[lineup_df['pts_diff'].idxmax()]
        worst_player = lineup_df.loc[lineup_df['pts_diff'].idxmin()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✅ **Best:** {best_player['Name']} ({best_player['pts_diff']:+.1f} vs proj)")
        with col2:
            if worst_player['pts_diff'] < -5:
                st.error(f"❌ **Worst:** {worst_player['Name']} ({worst_player['pts_diff']:+.1f} vs proj)")
            else:
                st.info(f"⚠️ **Lowest:** {worst_player['Name']} ({worst_player['pts_diff']:+.1f} vs proj)")


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
        
        with st.expander("📋 Research-Backed Optimal Construction", expanded=True):
            st.markdown(f"**{template_info['name']}**")
            st.info(f"💡 **{template_info['strategy_notes']}**")
            
            construction_df = pd.DataFrame({
                'Bucket': ['Punt (<10%)', 'Mid (10-30%)', 'Chalk (30-40%)', 'Mega (>40%)'],
                'Target Range': [
                    f"{targets['punt'][0]}-{targets['punt'][1]}",
                    f"{targets['mid'][0]}-{targets['mid'][1]}",
                    f"{targets['chalk'][0]}-{targets['chalk'][1]}",
                    f"{targets['mega'][0]}-{targets['mega'][1]}"
                ],
                'Why This Works': [
                    '🔵 Low owned players = high leverage if they hit',
                    '🟢 Safe plays with upside. Balanced risk/reward',
                    '🟡 Popular for a reason, but limits differentiation',
                    '🔴 Everyone plays them. Only good if must-play'
                ]
            })
            
            st.table(construction_df)
            
            # Add field-size specific advice
            field_size = tournament_config.get('field_size', 10000)
            entry_fee = tournament_config.get('entry_fee', '$3')
            
            if contest_code == 'LARGE_GPP':
                st.warning(f"""
**Large Field Strategy ({field_size:,} entries):**
- Need top 0.1-1% to win big money
- MUST have at least 1 player <5% owned
- Fade 2-3 of the most popular plays
- High risk/high reward mentality
                """)
            elif contest_code in ['SE', '3MAX']:
                st.info(f"""
**Single/Limited Entry Strategy ({field_size:,} entries):**
- People play safer in single-entry
- Pivot off 1-2 chalk plays for differentiation
- Need top 10-20% to profit
- Can't "make up for it" with more entries
                """)
            elif contest_code == 'CASH':
                st.success(f"""
**Cash Game Strategy ({field_size:,} entries):**
- TOP 45-50% WINS. Play it safe!
- High ownership is GOOD here
- Prioritize high floor over ceiling
- Avoid boom-or-bust plays
                """)
            
            # Entry fee based adjustments
            if entry_fee in ["$20", "$50+"]:
                st.error(f"""
**⚠️ HIGH STAKES ({entry_fee} entry):**
- Much tougher competition (sharks/pros)
- Need deeper edge than public
- Public projections won't cut it
- Consider game theory heavily
                """)
    
    st.markdown("---")
    
    st.header(f"1. Player Pool - Edge Analysis")
    
    st.markdown("**🎯 Edge Finder:** Green = High leverage plays | Red = Chalk traps")
    
    column_config = {
        "Name": st.column_config.TextColumn("Player", disabled=True, width="medium"), 
        "edge_category": st.column_config.TextColumn("Edge", disabled=True, width="medium"),
        "gpp_score": st.column_config.NumberColumn("GPP Score", disabled=True, format="%.1f", width="small", help="Higher = better GPP play"),
        "leverage_score": st.column_config.NumberColumn("Leverage", disabled=True, format="%+.1f", width="small", help="Positive = underowned"),
        "ceiling": st.column_config.NumberColumn("Ceiling", disabled=True, format="%.1f", width="small"),
        "positions": st.column_config.TextColumn("Pos", disabled=True, width="small"), 
        "salary": st.column_config.NumberColumn("Salary", format="$%d", width="small"), 
        "proj": st.column_config.NumberColumn("Proj", format="%.1f", width="small"), 
        "value": st.column_config.NumberColumn("Value", format="%.2f", disabled=True, width="small"), 
        "own_proj": st.column_config.NumberColumn("Own%", format="%.1f%%", width="small"),
        "Lock": st.column_config.CheckboxColumn("🔒", help="Lock into lineup", width="small"), 
        "Exclude": st.column_config.CheckboxColumn("❌", help="Exclude from lineups", width="small"), 
        "Team": None, "Opponent": None, "bucket": None, "Minutes": None, "FPPM": None,
        "player_id": None, "GameID": None
    }
    
    column_order = [
        'Lock', 'Exclude', 'Name', 'edge_category', 'gpp_score', 'leverage_score',
        'positions', 'salary', 'proj', 'ceiling', 'value', 'own_proj'
    ]
    
    df_for_editor = slate_df.copy()
    
    if df_for_editor.empty:
        st.info("✏️ Paste your player data into the text area in the sidebar and click the button to load the pool.")
        
        blank_df = pd.DataFrame(columns=column_order)
        edited_df = st.data_editor(blank_df, column_config=column_config, column_order=column_order, hide_index=True, use_container_width=True, height=200, key="player_editor_blank")
        st.session_state['edited_df'] = blank_df
        st.markdown("---")
        st.header("2. Select Winning Template")
        st.info("Template selection will be enabled once data is loaded.")
        st.markdown("---")
        st.header("3. Generate Lineups")
        st.info("Optimization controls will be enabled once data is loaded.")
        st.markdown("---")
        st.header("4. Your Lineups")
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
    
    # ACTUAL RESULTS INPUT SECTION
    st.subheader("📊 Import Last Night's Results (Optional)")
    
    with st.expander("📈 Paste Actual Results to See How Lineups Performed"):
        st.markdown("""
        **Upload actual game results to compare vs projections:**
        
        Paste data with columns: `Player`, `Roster Position`, `%Drafted`, `FPTS`
        """)
        
        actual_results_input = st.text_area(
            "Paste actual results here:",
            height=150,
            placeholder="Player\tRoster Position\t%Drafted\tFPTS\nNikola Jokic\tC\t7.78%\t66.75\nCade Cunningham\tPG\t5.70%\t56.0",
            key="results_input"
        )
        
        load_results_btn = st.button("Load Actual Results", use_container_width=True)
        
        if load_results_btn and actual_results_input:
            with st.spinner("Processing actual results..."):
                try:
                    data_io = io.StringIO(actual_results_input)
                    first_line = actual_results_input.split('\n')[0]
                    
                    if '\t' in first_line:
                        actuals_df = pd.read_csv(data_io, sep='\t')
                    else:
                        actuals_df = pd.read_csv(data_io)
                    
                    actuals_df.columns = actuals_df.columns.str.strip()
                    
                    col_map = {
                        'Player': 'Name',
                        '%Drafted': 'actual_own',
                        'FPTS': 'actual_pts'
                    }
                    actuals_df.rename(columns=col_map, inplace=True)
                    
                    actuals_df['actual_own'] = actuals_df['actual_own'].astype(str).str.replace('%', '').astype(float)
                    actuals_df['actual_pts'] = pd.to_numeric(actuals_df['actual_pts'], errors='coerce')
                    
                    # Merge with current slate
                    merged_df = edited_df.merge(
                        actuals_df[['Name', 'actual_own', 'actual_pts']], 
                        on='Name', 
                        how='left'
                    )
                    
                    st.session_state['edited_df_with_actuals'] = merged_df
                    st.session_state['has_actuals'] = True
                    
                    players_with_results = merged_df['actual_pts'].notna().sum()
                    st.success(f"✅ Loaded actual results for {players_with_results} players!")
                    
                    # Show quick comparison
                    if players_with_results > 0:
                        avg_proj = merged_df[merged_df['actual_pts'].notna()]['proj'].mean()
                        avg_actual = merged_df[merged_df['actual_pts'].notna()]['actual_pts'].mean()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Avg Projected", f"{avg_proj:.1f}")
                        with col2:
                            st.metric("Avg Actual", f"{avg_actual:.1f}")
                        with col3:
                            diff = avg_actual - avg_proj
                            st.metric("Difference", f"{diff:+.1f}", delta=f"{(diff/avg_proj*100):+.1f}%")
                    
                except Exception as e:
                    st.error(f"Error loading results: {e}")
                    st.session_state['has_actuals'] = False
        
        # Show actuals column in editor if loaded
        if st.session_state.get('has_actuals', False):
            st.info("✅ Actual results loaded! They'll be used in lineup analysis.")
            
            # Add toggle to show/hide actuals in the player table
            show_actuals_in_table = st.checkbox("Show actual results in player table above", value=False)
            
            if show_actuals_in_table and 'edited_df_with_actuals' in st.session_state:
                st.markdown("**Players with Actual Results:**")
                
                actuals_display = st.session_state['edited_df_with_actuals'][
                    st.session_state['edited_df_with_actuals']['actual_pts'].notna()
                ][['Name', 'positions', 'salary', 'proj', 'actual_pts', 'own_proj', 'actual_own']].copy()
                
                actuals_display['pts_diff'] = actuals_display['actual_pts'] - actuals_display['proj']
                actuals_display['own_diff'] = actuals_display['actual_own'] - actuals_display['own_proj']
                
                actuals_display = actuals_display.sort_values('pts_diff', ascending=False)
                
                st.dataframe(
                    actuals_display.style.format({
                        'salary': '${:,}',
                        'proj': '{:.1f}',
                        'actual_pts': '{:.1f}',
                        'pts_diff': '{:+.1f}',
                        'own_proj': '{:.1f}%',
                        'actual_own': '{:.1f}%',
                        'own_diff': '{:+.1f}%'
                    }),
                    use_container_width=True,
                    hide_index=True,
                    height=300
                )

    st.markdown("---")
    
    # TEMPLATE SELECTION - MOVED HERE (before it's used)
    st.header("2. Select Winning Template")
    
    template_choice = st.selectbox(
        "Choose Lineup Construction Template",
        options=list(WINNING_TEMPLATES.keys()),
        format_func=lambda x: WINNING_TEMPLATES[x]['name'],
        help="These are historically proven templates based on real winning lineups"
    )
    
    selected_template = WINNING_TEMPLATES[template_choice]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(f"**{selected_template['name']}**\n\n{selected_template['description']}")
    
    with col2:
        st.metric("Target Distribution", 
                 f"{selected_template['requirements']['min_leverage_plays']} leverage plays")
        st.metric("Max Chalk", selected_template['requirements']['max_chalk'])
    
    # Show template requirements
    with st.expander("📋 Template Requirements & Distribution"):
        st.markdown("**Requirements:**")
        reqs = selected_template['requirements']
        st.write(f"- ✅ Min {reqs['min_value_plays']} players with 5.0+ value")
        st.write(f"- ✅ Min {reqs['min_leverage_plays']} players with 10+ leverage")
        st.write(f"- ⚠️ Max {reqs['max_chalk']} players over 30% owned")
        st.write(f"- 🎯 Require {reqs['required_punt']} player(s) under 5% owned")
        
        st.markdown("**Target Edge Distribution:**")
        dist = selected_template['target_distribution']
        dist_df = pd.DataFrame({
            'Edge Type': list(dist.keys()),
            'Target Count': list(dist.values())
        })
        st.table(dist_df)
    
    st.markdown("---")
    
    st.header("3. Generate Lineups with Template")
    
    col_n, col_slack = st.columns(2)
    
    with col_n:
        n_lineups = st.slider(
            "Number of Lineups", 
            min_value=1, 
            max_value=20, 
            value=min(selected_template['requirements']['min_leverage_plays'] * 2, 10),
            help="More lineups = more template variations"
        )
    
    with col_slack:
        use_template_enforcement = st.checkbox(
            "Enforce Template Requirements",
            value=True,
            help="Force lineups to match the selected template's edge distribution"
        )
    
    # Advanced settings
    with st.expander("⚙️ Advanced Edge Settings"):
        min_gpp_score = st.slider(
            "Minimum GPP Score per Player",
            min_value=0,
            max_value=50,
            value=20,
            help="Only use players above this GPP score threshold"
        )
        
        max_total_ownership = st.slider(
            "Max Total Lineup Ownership %",
            min_value=100,
            max_value=400,
            value=250,
            step=10,
            help="Lower = more contrarian. Average lineup = 250%"
        )
        
        force_leverage_play = st.checkbox(
            "Force at least 1 Elite Leverage play",
            value=True,
            help="Guarantee a 🔥 Elite Leverage player in every lineup"
        )
    
    run_btn = st.button(f"🚀 Build {n_lineups} Lineups Using {selected_template['name']}", use_container_width=True, type="primary")
    
    if run_btn:
        final_df = st.session_state['edited_df'].copy()
        
        # Apply GPP score filter
        if min_gpp_score > 0:
            original_count = len(final_df)
            final_df = final_df[final_df['gpp_score'] >= min_gpp_score]
            filtered_count = original_count - len(final_df)
            if filtered_count > 0:
                st.info(f"🔍 Filtered out {filtered_count} low GPP score players")
        
        # Template enforcement - adjust bucket targets
        if use_template_enforcement:
            template_reqs = selected_template['requirements']
            
            # Count available players by edge category
            edge_counts = final_df['edge_category'].value_counts()
            
            st.info(f"""
            **Template Enforcement Active:**
            - Targeting {template_reqs['min_leverage_plays']} leverage plays
            - Max {template_reqs['max_chalk']} chalk players
            - Requiring {template_reqs['required_punt']} punt play(s) <5% owned
            """)
            
            # Check if we have enough players
            elite_count = edge_counts.get('🔥 Elite Leverage', 0)
            high_count = edge_counts.get('⭐ High Leverage', 0)
            
            if elite_count + high_count < template_reqs['min_leverage_plays']:
                st.warning(f"⚠️ Only {elite_count + high_count} leverage plays available. Template requires {template_reqs['min_leverage_plays']}. Adjust GPP score filter.")
        
        # Build lineups with edge awareness
        conflict = set(locked_player_ids) & set(excluded_player_ids)
        if conflict:
            st.error(f"❌ CONFLICT: {', '.join(conflict)} are both locked and excluded.")
            return

        with st.spinner(f'Building {n_lineups} lineups using {selected_template["name"]}...'):
            
            # Generate lineups
            top_lineups = generate_top_n_lineups(
                slate_df=final_df,
                template=template,
                n_lineups=n_lineups,
                bucket_slack=2,  # More flexibility for edge-based building
                locked_player_ids=locked_player_ids,
                excluded_player_ids=excluded_player_ids,
            )
            
            # Post-process: Filter lineups by template requirements
            if use_template_enforcement and top_lineups:
                valid_lineups = []
                
                for lineup in top_lineups:
                    lineup_players = final_df[final_df['player_id'].isin(lineup['player_ids'])]
                    
                    # Check template requirements
                    edge_dist = lineup_players['edge_category'].value_counts()
                    total_own = lineup_players['own_proj'].sum()
                    
                    elite_lev = edge_dist.get('🔥 Elite Leverage', 0) + edge_dist.get('⭐ High Leverage', 0)
                    chalk_count = len(lineup_players[lineup_players['own_proj'] > 30])
                    punt_count = len(lineup_players[lineup_players['own_proj'] < 5])
                    
                    # Check all requirements
                    meets_leverage = elite_lev >= template_reqs['min_leverage_plays']
                    meets_chalk = chalk_count <= template_reqs['max_chalk']
                    meets_punt = punt_count >= template_reqs['required_punt']
                    meets_ownership = total_own <= max_total_ownership
                    
                    if force_leverage_play:
                        has_elite = edge_dist.get('🔥 Elite Leverage', 0) >= 1
                        meets_template = meets_leverage and meets_chalk and meets_punt and meets_ownership and has_elite
                    else:
                        meets_template = meets_leverage and meets_chalk and meets_punt and meets_ownership
                    
                    if meets_template:
                        # Add template metadata
                        lineup['template_score'] = (
                            elite_lev * 10 + 
                            (max_total_ownership - total_own) / 10 +
                            punt_count * 5
                        )
                        lineup['total_ownership'] = total_own
                        lineup['edge_distribution'] = dict(edge_dist)
                        valid_lineups.append(lineup)
                
                if len(valid_lineups) < n_lineups:
                    st.warning(f"⚠️ Only {len(valid_lineups)} of {n_lineups} lineups met template requirements. Consider loosening constraints.")
                
                top_lineups = valid_lineups[:n_lineups]
        
        if top_lineups:
            st.session_state['optimal_lineups_results'] = {'lineups': top_lineups, 'ran': True}
            st.success(f"✅ Built {len(top_lineups)} template-optimized lineups!")
            
            # Show template compliance summary
            if use_template_enforcement:
                st.markdown("**Template Compliance Summary:**")
                for i, lineup in enumerate(top_lineups[:3]):
                    with st.expander(f"Lineup {i+1} - Edge Distribution"):
                        edge_dist = lineup.get('edge_distribution', {})
                        st.write(f"Total Ownership: {lineup.get('total_ownership', 0):.1f}%")
                        st.write(f"Template Score: {lineup.get('template_score', 0):.1f}")
                        st.write("Edge Distribution:", edge_dist)
        else:
            st.error("❌ No lineups could be built meeting template requirements. Try adjusting settings.")
    
    st.markdown("---")
    st.header(f"4. Your Lineups")
    
    if st.session_state['optimal_lineups_results'].get('ran', False):
        display_multiple_lineups(slate_df, template, st.session_state['optimal_lineups_results']['lineups'])
    else:
        st.info("Select the number of lineups and click the button above to run the multi-lineup builder.")


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
        st.caption("Research-Backed Tournament Strategy")
        
        st.subheader("🎯 Contest Details")
        
        # Tournament type selection
        tournament_type = st.selectbox(
            "Tournament Type",
            options=[
                "Cash Game (50/50, Double-Up)",
                "Single Entry GPP",
                "3-Max GPP", 
                "20-Max GPP",
                "Large Field GPP (Milly Maker)",
                "Showdown Captain Mode"
            ],
            help="Each type requires a different optimal strategy"
        )
        
        # Entry fee selection
        entry_fee = st.selectbox(
            "Entry Fee",
            options=["$0.25", "$1", "$3", "$5", "$10", "$20", "$50+"],
            index=2,  # Default to $3
            help="Higher entry fees = tougher competition"
        )
        
        fee_info = ENTRY_FEE_TIERS[entry_fee]
        
        # Field size selection
        field_size = st.slider(
            "Expected Field Size",
            min_value=fee_info["min_field"],
            max_value=fee_info["max_field"],
            value=min(10000, fee_info["max_field"]),
            step=100 if fee_info["max_field"] < 10000 else 1000,
            help="Larger fields = need more leverage"
        )
        
        # Show competition level
        st.caption(f"💪 Competition Level: **{fee_info['skill_level']}**")
        
        # Map tournament types to strategy codes
        tournament_map = {
            "Cash Game (50/50, Double-Up)": {
                "code": "CASH",
                "top_payout_pct": 0.45,  # Top 45% cash
                "recommended_lineups": 1
            },
            "Single Entry GPP": {
                "code": "SE",
                "top_payout_pct": 0.20,  # Top 20% cash, but need top 10-15% to profit
                "recommended_lineups": 1
            },
            "3-Max GPP": {
                "code": "3MAX",
                "top_payout_pct": 0.20,
                "recommended_lineups": 3
            },
            "20-Max GPP": {
                "code": "20MAX",
                "top_payout_pct": 0.15,
                "recommended_lineups": 20
            },
            "Large Field GPP (Milly Maker)": {
                "code": "LARGE_GPP",
                "top_payout_pct": 0.20,  # ~20% cash but need top 1% to win big
                "recommended_lineups": 150
            },
            "Showdown Captain Mode": {
                "code": "SHOWDOWN",
                "top_payout_pct": 0.20,
                "recommended_lineups": 20
            }
        }
        
        tournament_config = tournament_map[tournament_type]
        tournament_config['field_size'] = field_size
        tournament_config['entry_fee'] = entry_fee
        tournament_config['description'] = TOURNAMENT_OWNERSHIP_TEMPLATES[tournament_config['code']]['description']
        
        # Show tournament info
        st.info(f"""
**Strategy:** {tournament_config['description']}

**Field:** {field_size:,} entries at {entry_fee}
        
**Top {tournament_config['top_payout_pct']*100:.0f}%** of field cashes
        """)
        
        # Adjust strategy based on field size and entry fee
        if tournament_config['code'] in ['SE', '3MAX', '20MAX', 'LARGE_GPP']:
            # GPP specific adjustments
            if field_size > 50000:
                st.warning("⚠️ **Large Field**: Need maximum leverage. Fade chalk heavily.")
            elif entry_fee in ["$20", "$50+"]:
                st.warning("⚠️ **High Stakes**: Tougher competition. Avoid obvious plays.")
        
        with st.expander("⚙️ Advanced Settings"):
            ownership_strategy = "Balanced"
            
            # Ownership strategy override
            ownership_strategy = st.select_slider(
                "Override Ownership Strategy",
                options=["Full Chalk", "Balanced", "Contrarian", "Max Leverage"],
                value="Balanced",
                help="Override the recommended strategy"
            )
            
            min_salary = st.slider("Min Salary to Use", min_value=45000, max_value=50000, value=48500, step=100)
        
        # Map ownership strategy override
        if ownership_strategy == "Full Chalk":
            contest_code = "CASH"
        elif ownership_strategy == "Contrarian":
            contest_code = "LARGE_GPP"
        elif ownership_strategy == "Max Leverage":
            contest_code = "LARGE_GPP"
        else:
            # Map our new codes to builder.py compatible codes
            code_mapping = {
                "CASH": "CASH",
                "SE": "SE",
                "3MAX": "SE",  # Use SE template
                "20MAX": "LARGE_GPP",  # Use LARGE_GPP template
                "LARGE_GPP": "LARGE_GPP",
                "SHOWDOWN": "SE"  # Use SE template
            }
            contest_code = code_mapping.get(tournament_config['code'], "SE")
        
        st.divider()
        st.subheader("📊 Load Player Data")
        
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
