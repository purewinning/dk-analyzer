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
# ---------------------------------------------

# --- CONFIGURATION CONSTANTS ---
MIN_GAMES_REQUIRED = 2

# --- TOURNAMENT SIMULATOR CLASS ---
class TournamentSimulator:
    """
    Simulates DFS tournaments to analyze lineup construction patterns and win probability.
    """
    
    def __init__(self, slate_df: pd.DataFrame, field_size: int = 10000):
        """
        Initialize simulator with player pool data.
        
        Args:
            slate_df: DataFrame with player data (proj, own_proj, salary, etc.)
            field_size: Number of entries in the tournament
        """
        self.slate_df = slate_df.copy()
        self.field_size = field_size
        
        # Calculate player variance (std dev) based on projection
        # Higher projection = higher variance (stars are more volatile)
        self.slate_df['std_dev'] = self.slate_df['proj'] * 0.25  # 25% std dev
        
    def simulate_player_score(self, player_row, n_simulations=1):
        """
        Simulate actual fantasy scores for a player using normal distribution.
        
        Args:
            player_row: Row from slate_df with player data
            n_simulations: Number of simulations to run
            
        Returns:
            Array of simulated scores
        """
        return np.random.normal(
            loc=player_row['proj'],
            scale=player_row['std_dev'],
            size=n_simulations
        ).clip(min=0)  # Can't score negative points
    
    def simulate_lineup_score(self, player_ids: List[str], n_simulations=1000):
        """
        Simulate a lineup's score distribution across many contests.
        
        Args:
            player_ids: List of player IDs in the lineup
            n_simulations: Number of simulations to run
            
        Returns:
            Array of simulated total scores for the lineup
        """
        lineup_players = self.slate_df[self.slate_df['player_id'].isin(player_ids)]
        
        # Simulate each player's scores
        simulated_scores = np.zeros(n_simulations)
        
        for _, player in lineup_players.iterrows():
            player_sims = self.simulate_player_score(player, n_simulations)
            simulated_scores += player_sims
            
        return simulated_scores
    
    def generate_field_ownership(self):
        """
        Generate a tournament field with ownership-based lineup distribution.
        Returns distribution of how many lineups used each player.
        """
        field_usage = {}
        
        for _, player in self.slate_df.iterrows():
            # Number of lineups using this player (based on ownership %)
            n_lineups_with_player = int(self.field_size * (player['own_proj'] / 100))
            field_usage[player['player_id']] = n_lineups_with_player
            
        return field_usage
    
    def calculate_win_probability(
        self, 
        lineup_player_ids: List[str],
        n_simulations: int = 1000
    ) -> Dict[str, float]:
        """
        Calculate probability of winning/placing based on simulations.
        
        Args:
            lineup_player_ids: List of player IDs in your lineup
            n_simulations: Number of tournament simulations to run
            
        Returns:
            Dictionary with win rates for different placements
        """
        # Get your lineup's simulated scores
        your_scores = self.simulate_lineup_score(lineup_player_ids, n_simulations)
        
        # Generate field ownership
        field_usage = self.generate_field_ownership()
        
        # Calculate ownership leverage (how unique is your lineup?)
        your_players = set(lineup_player_ids)
        total_field_exposure = sum(field_usage.get(pid, 0) for pid in lineup_player_ids)
        avg_exposure = total_field_exposure / (len(lineup_player_ids) * self.field_size)
        
        # Simulate winning scores (top 1% of field needs to beat)
        # Assuming field scores are normally distributed around average projection
        avg_field_projection = self.slate_df.nsmallest(8, 'salary')['proj'].sum()
        field_std = avg_field_projection * 0.15
        
        winning_scores = np.random.normal(
            loc=avg_field_projection * 1.3,  # Winners score 30% above average
            scale=field_std,
            size=n_simulations
        )
        
        top10_scores = np.random.normal(
            loc=avg_field_projection * 1.15,  # Top 10% scores 15% above average
            scale=field_std,
            size=n_simulations
        )
        
        # Calculate win probabilities
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
    
    def analyze_ownership_pattern(
        self,
        lineup_player_ids: List[str]
    ) -> Dict[str, any]:
        """
        Analyze the ownership construction pattern of a lineup.
        
        Returns breakdown by ownership bucket and strategic insights.
        """
        lineup_players = self.slate_df[self.slate_df['player_id'].isin(lineup_player_ids)]
        
        # Count players in each bucket
        bucket_counts = lineup_players['bucket'].value_counts().to_dict()
        
        # Calculate weighted ownership
        avg_ownership = lineup_players['own_proj'].mean()
        
        # Calculate leverage (uniqueness)
        leverage_score = 0
        for _, player in lineup_players.iterrows():
            # Lower ownership = higher leverage
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
        """Classify lineup strategy based on construction."""
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
    
    def compare_strategies(
        self,
        lineups_dict: Dict[str, List[str]],
        n_simulations: int = 1000
    ) -> pd.DataFrame:
        """
        Compare multiple lineup strategies side-by-side.
        
        Args:
            lineups_dict: Dict of {lineup_name: [player_ids]}
            n_simulations: Number of simulations per lineup
            
        Returns:
            DataFrame comparing all strategies
        """
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

# --- HEADER MAPPING ---
REQUIRED_CSV_TO_INTERNAL_MAP = {
    'Player': 'Name',
    'Salary': 'salary', 
    'Position': 'positions',
    'Team': 'Team',
    'Opponent': 'Opponent',
    
    # Primary projection headers
    'PROJECTED FP': 'proj',
    'Projection': 'proj',
    'Proj': 'proj',
    
    # Primary ownership headers
    'OWNERSHIP %': 'own_proj',
    'Ownership': 'own_proj',
    'Own': 'own_proj',
    'Own%': 'own_proj',
    
    # Other supporting columns
    'Minutes': 'Minutes',
    'FPPM': 'FPPM',
    'Value': 'Value'
}
CORE_INTERNAL_COLS = ['salary', 'positions', 'proj', 'own_proj', 'Name', 'Team', 'Opponent']

# --- 1. DATA PREPARATION ---

def load_and_preprocess_data(pasted_data: str = None) -> pd.DataFrame:
    """Loads CSV/TSV from a pasted string, standardizes ownership, and processes data."""
    
    # Initialization for empty/none data
    empty_df_cols = CORE_INTERNAL_COLS + ['player_id', 'GameID', 'bucket', 'value', 'Lock', 'Exclude']
    if pasted_data is None or not pasted_data.strip():
        df = pd.DataFrame(columns=empty_df_cols)
        return df
        
    try:
        data_io = io.StringIO(pasted_data)
        
        # Try to detect if it's tab-separated or comma-separated
        first_line = pasted_data.split('\n')[0]
        if '\t' in first_line:
            df = pd.read_csv(data_io, sep='\t')
        else:
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


    # --- FINAL PROCESSING ---
    
    df['Team'] = df['Team'].astype(str)
    df['Opponent'] = df['Opponent'].astype(str)
    df['GameID'] = df.apply(
        lambda row: '@'.join(sorted([row['Team'], row['Opponent']])), axis=1
    )
    df['player_id'] = df['Name'] 
            
    # Handle ownership - could be decimal (0.419) or percentage (41.9)
    df['own_proj'] = df['own_proj'].astype(str).str.replace('%', '', regex=False).str.replace(',', '.', regex=False)
    df['own_proj'] = pd.to_numeric(df['own_proj'], errors='coerce')
    
    # Convert decimal ownership (0.419) to percentage (41.9)
    if df['own_proj'].max() <= 1.0 and df['own_proj'].max() > 0:
        df['own_proj'] = df['own_proj'] * 100
    
    df['own_proj'] = df['own_proj'].round(1)
    
    # Drop rows with missing core data
    df.dropna(subset=CORE_INTERNAL_COLS, inplace=True)

    try:
        # Clean salary formatting
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
    
    # Fill any missing columns for stability
    for col in empty_df_cols:
        if col not in df.columns:
            df[col] = None 
            
    return df

# --- 2. SESSION STATE INITIALIZATION ---

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


def assign_lineup_positions(lineup_df):
    """
    Assigns each player to a legal DraftKings roster slot based on their position eligibility.
    Returns the dataframe with a 'roster_slot' column or None if no valid assignment exists.
    """
    # Define the slots we need to fill in order of specificity
    slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    
    # Track which players are assigned
    assigned_players = set()
    slot_assignments = {}
    
    def can_play_slot(pos_string, slot):
        """Check if a player with given positions can play a slot."""
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
            return True  # Anyone can play UTIL
        return False
    
    # Try to fill slots in order
    for slot in slots:
        # Find available players who can fill this slot
        available = lineup_df[~lineup_df['player_id'].isin(assigned_players)].copy()
        
        # Filter to those eligible for this slot
        eligible = available[available['positions'].apply(lambda x: can_play_slot(x, slot))]
        
        if len(eligible) == 0:
            return None  # Can't fill this slot
        
        # Assign the first eligible player (optimizer already chose optimal set)
        chosen = eligible.iloc[0]
        slot_assignments[slot] = chosen['player_id']
        assigned_players.add(chosen['player_id'])
    
    # Create a new dataframe with slot assignments
    result_df = lineup_df.copy()
    result_df['roster_slot'] = result_df['player_id'].map({v: k for k, v in slot_assignments.items()})
    
    return result_df


def display_multiple_lineups(slate_df, template, lineup_list):
    """Function to display the top N optimized lineups with improved UI."""
    
    if not lineup_list:
        st.error("❌ No valid lineups could be found that meet all constraints.")
        st.warning("Try loosening your constraints or reducing the number of lineups requested.")
        return
    
    # --- METRICS SECTION ---
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
    
    # IMPROVED: Assign positions based on actual eligibility
    lineup_df = assign_lineup_positions(lineup_df)
    
    if lineup_df is None:
        st.error("❌ Could not assign valid roster positions. This shouldn't happen if the optimizer is working correctly.")
        return
    
    # Sort by position order
    ROSTER_ORDER = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    position_type = pd.CategoricalDtype(ROSTER_ORDER, ordered=True)
    lineup_df['roster_slot'] = lineup_df['roster_slot'].astype(position_type)
    lineup_df.sort_values(by='roster_slot', inplace=True)
    
    # Define display columns
    display_cols = ['roster_slot', 'Name', 'positions', 'Team', 'Opponent', 'salary', 'proj', 'value', 'own_proj', 'bucket', 'Minutes', 'FPPM'] 
    lineup_df_display = lineup_df[display_cols].reset_index(drop=True)
    
    # Rename columns for display
    lineup_df_display.rename(columns={
        'roster_slot': 'SLOT', 
        'positions': 'POS', 
        'own_proj': 'OWN%', 
        'Minutes': 'MIN', 
        'FPPM': 'FP/M', 
        'bucket': 'CATEGORY'
    }, inplace=True)
    
    # Display with styling
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
        "player_id": None, 
        "GameID": None 
    }
    
    column_order = [
        'Lock', 'Exclude', 'Name', 'bucket', 'positions', 'Team', 'Opponent', 
        'salary', 'proj', 'value', 'own_proj', 'Minutes', 'FPPM'
    ]
    
    df_for_editor = slate_df.copy()
    
    if df_for_editor.empty:
        st.info("✏️ Paste your player data into the text area in the sidebar and click the button to load the pool.")
        
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


def tab_strategy_lab(slate_df, template):
    """Render the Strategy Lab - Simulation and Pattern Analysis."""
    st.header("🧪 Strategy Lab - Tournament Simulation")
    
    if slate_df.empty:
        st.info("✏️ Paste your data into the sidebar text area to access the Strategy Lab.")
        return
    
    st.markdown("""
    This lab simulates thousands of tournaments to discover **which lineup construction patterns actually win**.
    Compare chalk-heavy vs contrarian strategies using Monte Carlo simulation.
    """)
    
    # Check if we have lineups to analyze
    if not st.session_state['optimal_lineups_results'].get('ran', False):
        st.warning("⚠️ Generate lineups first in the 'Lineup Builder' tab, then return here for simulation analysis.")
        return
    
    lineups = st.session_state['optimal_lineups_results']['lineups']
    
    if not lineups:
        st.info("No lineups available to simulate. Generate some lineups first!")
        return
    
    # Simulation Controls
    st.subheader("Simulation Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        field_size = st.selectbox(
            "Tournament Field Size",
            options=[1000, 5000, 10000, 50000, 150000],
            index=2,
            help="Number of entries in the tournament"
        )
    
    with col2:
        n_simulations = st.slider(
            "Number of Simulations",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="More simulations = more accurate results (but slower)"
        )
    
    run_sim_btn = st.button("🎲 Run Tournament Simulation", use_container_width=True)
    
    if run_sim_btn or 'simulation_results' in st.session_state:
        
        if run_sim_btn:
            with st.spinner(f'Running {n_simulations} tournament simulations...'):
                simulator = TournamentSimulator(slate_df, field_size=field_size)
                
                # Analyze top 10 lineups
                lineups_dict = {
                    f"Lineup {i+1}": lineup['player_ids'] 
                    for i, lineup in enumerate(lineups[:10])
                }
                
                comparison_df = simulator.compare_strategies(lineups_dict, n_simulations=n_simulations)
                st.session_state['simulation_results'] = comparison_df
                st.session_state['simulator'] = simulator
        
        results_df = st.session_state.get('simulation_results')
        
        if results_df is not None and not results_df.empty:
            
            st.success("✅ Simulation Complete!")
            
            # Display Key Insights
            st.subheader("📊 Simulation Results")
            
            # Metrics for best lineup
            best_lineup = results_df.loc[results_df['Leverage Score'].idxmax()]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Best Leverage", 
                    f"{best_lineup['Lineup']}", 
                    f"{best_lineup['Leverage Score']:.1f}"
                )
            with col2:
                st.metric(
                    "Win Rate", 
                    f"{best_lineup['Win Rate %']:.2f}%",
                    delta=f"1 in {int(100/best_lineup['Win Rate %'])}"
                )
            with col3:
                st.metric(
                    "Strategy", 
                    best_lineup['Strategy']
                )
            with col4:
                st.metric(
                    "Avg Ownership",
                    f"{best_lineup['Avg Own %']:.1f}%"
                )
            
            st.markdown("---")
            
            # DETAILED LINEUP VIEWER
            st.subheader("🔍 Detailed Lineup Analysis")
            
            # Let user select which lineup to examine
            lineup_names = results_df['Lineup'].tolist()
            selected_lineup_name = st.selectbox(
                "Select lineup to view details:",
                options=lineup_names,
                help="Choose a lineup to see the full player breakdown"
            )
            
            # Get the selected lineup data
            selected_idx = lineup_names.index(selected_lineup_name)
            selected_lineup_player_ids = lineups[selected_idx]['player_ids']
            selected_lineup_stats = results_df[results_df['Lineup'] == selected_lineup_name].iloc[0]
            
            # Display lineup stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Strategy", selected_lineup_stats['Strategy'])
            with col2:
                st.metric("Leverage Score", f"{selected_lineup_stats['Leverage Score']:.1f}")
            with col3:
                st.metric("Win Rate", f"{selected_lineup_stats['Win Rate %']:.3f}%")
            with col4:
                st.metric("Avg Ownership", f"{selected_lineup_stats['Avg Own %']:.1f}%")
            
            # Get player details for this lineup
            lineup_detail_df = slate_df[slate_df['player_id'].isin(selected_lineup_player_ids)].copy()
            
            # Assign positions
            lineup_detail_df = assign_lineup_positions(lineup_detail_df)
            
            if lineup_detail_df is not None:
                # Sort by roster slot
                ROSTER_ORDER = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
                position_type = pd.CategoricalDtype(ROSTER_ORDER, ordered=True)
                lineup_detail_df['roster_slot'] = lineup_detail_df['roster_slot'].astype(position_type)
                lineup_detail_df = lineup_detail_df.sort_values('roster_slot')
                
                # Display columns
                display_cols = ['roster_slot', 'Name', 'positions', 'Team', 'Opponent', 
                               'salary', 'proj', 'value', 'own_proj', 'bucket']
                lineup_display = lineup_detail_df[display_cols].reset_index(drop=True)
                
                lineup_display.rename(columns={
                    'roster_slot': 'SLOT',
                    'positions': 'POS',
                    'own_proj': 'OWN%',
                    'bucket': 'CATEGORY'
                }, inplace=True)
                
                # Style the lineup
                styled_lineup = lineup_display.style.applymap(
                    color_bucket, subset=['CATEGORY']
                ).format({
                    'salary': '${:,}',
                    'proj': '{:.1f}',
                    'value': '{:.2f}',
                    'OWN%': '{:.1f}%'
                })
                
                st.dataframe(styled_lineup, use_container_width=True, hide_index=True)
                
                # Show lineup totals
                total_salary = lineup_detail_df['salary'].sum()
                total_proj = lineup_detail_df['proj'].sum()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Salary", f"${total_salary:,}")
                with col2:
                    st.metric("Total Projection", f"{total_proj:.1f}")
                with col3:
                    st.metric("Remaining Salary", f"${50000 - total_salary:,}")
            
            st.markdown("---")
            
            # Full Results Table
            st.subheader("Full Lineup Comparison")
            
            # Format and display
            display_df = results_df.copy()
            
            # Sort by leverage score for better display
            display_df = display_df.sort_values('Leverage Score', ascending=False)
            
            # Color code by strategy
            def color_strategy(val):
                if val == "Chalk Heavy":
                    return 'background-color: #9C3838; color: white'
                elif val == "Contrarian Punt":
                    return 'background-color: #3D85C6; color: white'
                elif val == "Full Contrarian":
                    return 'background-color: #38761D; color: white'
                else:
                    return 'background-color: #A37F34; color: white'
            
            # Highlight best leverage score
            def highlight_best_leverage(s):
                if s.name == 'Leverage Score':
                    max_val = s.max()
                    return ['background-color: #90EE90' if v == max_val else '' for v in s]
                return ['' for _ in s]
            
            styled_df = display_df.style.applymap(
                color_strategy, subset=['Strategy']
            ).apply(
                highlight_best_leverage, axis=0
            ).format({
                'Win Rate %': '{:.3f}%',
                'Top 10% Rate': '{:.2f}%',
                'Avg Score': '{:.1f}',
                'Ceiling (90th)': '{:.1f}',
                'Floor (10th)': '{:.1f}',
                'Avg Own %': '{:.1f}%',
                'Leverage Score': '{:.1f}'
            })
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Strategy Pattern Analysis
            st.subheader("📈 Ownership Construction Patterns")
            
            st.markdown("""
            **How to read this:**
            - **Mega Chalk** = Players with >40% ownership
            - **Chalk** = Players with 30-40% ownership  
            - **Mid** = Players with 10-30% ownership
            - **Punt** = Players with <10% ownership
            
            **Leverage Score** = Win Rate × Uniqueness (higher = better tournament EV)
            """)
            
            # Pattern breakdown
            pattern_cols = ['Lineup', 'Strategy', 'Mega Chalk', 'Chalk', 'Mid', 'Punt', 'Leverage Score']
            pattern_df = results_df[pattern_cols].copy()
            pattern_df = pattern_df.sort_values('Leverage Score', ascending=False)
            
            # Highlight top 3 leverage scores
            def highlight_top_leverage(s):
                if s.name == 'Leverage Score':
                    sorted_vals = s.sort_values(ascending=False)
                    top_3 = sorted_vals.head(3).values
                    colors = []
                    for v in s:
                        if v == top_3[0]:
                            colors.append('background-color: #90EE90; font-weight: bold')
                        elif len(top_3) > 1 and v == top_3[1]:
                            colors.append('background-color: #C1FFC1')
                        elif len(top_3) > 2 and v == top_3[2]:
                            colors.append('background-color: #E0FFE0')
                        else:
                            colors.append('')
                    return colors
                return ['' for _ in s]
            
            st.dataframe(
                pattern_df.style.apply(highlight_top_leverage, axis=0),
                use_container_width=True,
                hide_index=True
            )
            
            # Insights
            st.markdown("---")
            st.subheader("💡 Key Insights")
            
            # Calculate insights
            best_strategy = results_df.loc[results_df['Leverage Score'].idxmax(), 'Strategy']
            avg_leverage = results_df['Leverage Score'].mean()
            
            chalk_heavy = results_df[results_df['Strategy'] == 'Chalk Heavy']
            contrarian = results_df[results_df['Strategy'].str.contains('Contrarian')]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Optimal Strategy Found:**")
                st.write(f"- **{best_strategy}** has highest leverage")
                st.write(f"- Average lineup leverage: {avg_leverage:.1f}")
                
                if len(chalk_heavy) > 0:
                    st.write(f"- Chalk-heavy win rate: {chalk_heavy['Win Rate %'].mean():.3f}%")
                if len(contrarian) > 0:
                    st.write(f"- Contrarian win rate: {contrarian['Win Rate %'].mean():.3f}%")
            
            with col2:
                st.markdown("**📊 Pattern Analysis:**")
                avg_mega = results_df['Mega Chalk'].mean()
                avg_punt = results_df['Punt'].mean()
                
                st.write(f"- Average mega chalk players: {avg_mega:.1f}")
                st.write(f"- Average punt plays: {avg_punt:.1f}")
                st.write(f"- Winning construction: {int(best_lineup['Mega Chalk'])} mega, {int(best_lineup['Chalk'])} chalk, {int(best_lineup['Mid'])} mid, {int(best_lineup['Punt'])} punt")


def tab_contest_analyzer(slate_df, template):
    """Render the Contest Analyzer."""
    st.header("Contest Strategy Analyzer")
    
    if slate_df.empty:
        st.info("✏️ Paste your data into the sidebar text area to view the contest analyzer.")
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
            "Copy your player pool data (including headers) and paste it here:",
            height=200,
            placeholder="Player\tSalary\tPosition\tTeam\tOpponent\tMinutes\tFPPM\tProjection\tValue\tOwnership"
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
    
    # Initialize slate_df from session state
    if 'slate_df' not in st.session_state:
        st.session_state['slate_df'] = pd.DataFrame(columns=CORE_INTERNAL_COLS + ['player_id', 'GameID', 'bucket', 'value', 'Lock', 'Exclude'])
    
    slate_df = st.session_state['slate_df']
    
    # Build template
    template = build_template_from_params(
        contest_type=contest_code,
        field_size=1000,
        pct_to_first=1.0,
        roster_size=DEFAULT_ROSTER_SIZE,
        salary_cap=DEFAULT_SALARY_CAP,
        min_games=MIN_GAMES_REQUIRED
    )
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🏗️ Lineup Builder", "🧪 Strategy Lab", "📊 Contest Analyzer"])
    
    with tab1:
        tab_lineup_builder(slate_df, template)
    
    with tab2:
        tab_strategy_lab(slate_df, template)
    
    with tab3:
        tab_contest_analyzer(slate_df, template)
