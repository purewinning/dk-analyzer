"""
DFS Elite Tools - Pro Edition
Advanced player pool management + Smart lineup generation
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import io

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(page_title="DFS Elite Tools", layout="wide", initial_sidebar_state="collapsed")

# Initialize session state
if 'player_pool' not in st.session_state:
    st.session_state.player_pool = None
if 'generated_lineups' not in st.session_state:
    st.session_state.generated_lineups = []
if 'contest_data' not in st.session_state:
    st.session_state.contest_data = None
if 'player_actuals' not in st.session_state:
    st.session_state.player_actuals = {}

# ============================================================================
# SPORT CONFIGS
# ============================================================================

SPORT_CONFIGS = {
    "NBA": {
        "salary_cap": 50000,
        "roster_size": 8,
        "positions": {"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1, "G": 1, "F": 1, "UTIL": 1}
    },
    "NFL": {
        "salary_cap": 50000,
        "roster_size": 9,
        "positions": {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "DST": 1}
    }
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_csv(uploaded_file):
    """Load and normalize CSV with aggressive error handling."""
    try:
        # Try reading with different encodings
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1')
        
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        st.write("**Debug - Columns found:**", list(df.columns))
        st.write("**Debug - First row:**", df.head(1).to_dict('records'))
        
        # Map common column variations
        column_map = {
            'player': 'name',
            'name': 'name',
            'position': 'positions',
            'pos': 'positions',
            'positions': 'positions',
            'salary': 'salary',
            'sal': 'salary',
            'projection': 'proj',
            'proj': 'proj',
            'fpts': 'proj',
            'points': 'proj',
            'ownership': 'own',
            'ownership_%': 'own',
            'own%': 'own',
            'own': 'own',
            'team': 'team',
            'opp': 'opponent',
            'opponent': 'opponent',
            'value': 'value_raw',
            'minutes': 'minutes',
            'mins': 'minutes',
            'fppm': 'fppm',
            'optimal_%': 'optimal_pct'
        }
        
        # Handle leverage separately (it might exist and conflict)
        if 'leverage' in df.columns:
            df['leverage_source'] = df['leverage'].astype(str).str.replace('%', '')
            df['leverage_source'] = pd.to_numeric(df['leverage_source'], errors='coerce')
            df = df.drop(columns=['leverage'])  # Remove original to avoid conflicts
        
        # Rename columns
        for old_col, new_col in column_map.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Ensure required columns exist
        if 'name' not in df.columns:
            st.error("❌ Missing player name column. Found: " + str(list(df.columns)))
            return None
        
        if 'positions' not in df.columns:
            st.error("❌ Missing position column. Found: " + str(list(df.columns)))
            return None
        
        # Clean salary (remove $ and commas)
        if 'salary' in df.columns:
            df['salary'] = df['salary'].astype(str).str.replace('$', '').str.replace(',', '')
            df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
        else:
            st.error("❌ Missing salary column")
            return None
        
        # Clean projection
        if 'proj' in df.columns:
            df['proj'] = pd.to_numeric(df['proj'], errors='coerce')
        else:
            st.error("❌ Missing projection column")
            return None
        
        # Clean ownership
        if 'own' in df.columns:
            df['own'] = df['own'].astype(str).str.replace('%', '')
            df['own'] = pd.to_numeric(df['own'], errors='coerce')
            if df['own'].max() <= 1.0:
                df['own'] = df['own'] * 100
        else:
            df['own'] = 15.0
        
        # Clean leverage if present (from your data)
        if 'leverage_raw' in df.columns:
            df['leverage_raw'] = df['leverage_raw'].astype(str).str.replace('%', '')
            df['leverage_raw'] = pd.to_numeric(df['leverage_raw'], errors='coerce')
        
        # Clean optimal % if present
        if 'optimal_pct' in df.columns:
            df['optimal_pct'] = df['optimal_pct'].astype(str).str.replace('%', '')
            df['optimal_pct'] = pd.to_numeric(df['optimal_pct'], errors='coerce')
        
        # Clean value if it's a raw column
        if 'value_raw' in df.columns:
            df['value_raw'] = pd.to_numeric(df['value_raw'], errors='coerce')
        
        # Clean minutes
        if 'minutes' in df.columns:
            df['minutes'] = pd.to_numeric(df['minutes'], errors='coerce')
        
        # Clean FPPM
        if 'fppm' in df.columns:
            df['fppm'] = pd.to_numeric(df['fppm'], errors='coerce')
        
        # Drop rows with missing critical data
        before_drop = len(df)
        df = df.dropna(subset=['name', 'positions', 'salary', 'proj'])
        after_drop = len(df)
        
        if before_drop > after_drop:
            st.warning(f"⚠️ Dropped {before_drop - after_drop} rows with missing data")
        
        # Add player ID
        df['player_id'] = range(len(df))
        
        # Calculate metrics
        df = calculate_metrics(df)
        
        st.success(f"✅ Successfully loaded {len(df)} players")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading CSV: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def calculate_metrics(df):
    """Calculate all player metrics."""
    
    # Value
    df['value'] = df['proj'] / (df['salary'] / 1000)
    
    # Ceiling (40% upside)
    df['ceiling'] = df['proj'] * 1.4
    
    # Leverage - use source data if available, otherwise calculate
    if 'leverage_source' in df.columns:
        df['leverage'] = df['leverage_source']
    else:
        df['leverage'] = 100 - df['own']
    
    # GPP Score
    df['gpp_score'] = (
        df['ceiling'] * 0.30 +
        df['value'] * 8 * 0.25 +
        df['leverage'] * 0.20 +
        df['proj'] * 0.15 +
        (100 - df['own']) * 0.10
    )
    
    # Boom potential
    df['boom_potential'] = (df['ceiling'] - df['proj']) * (100 - df['own']) / 100
    
    # Categorize
    df = categorize_players(df)
    
    return df

def categorize_players(df):
    """Categorize players by tiers."""
    
    # Ownership tiers
    def ownership_tier(own):
        if own >= 40:
            return "❌ Chalk"
        elif own >= 25:
            return "⚠️ Popular"
        elif own >= 15:
            return "✅ Mid"
        elif own >= 8:
            return "💎 Contrarian"
        else:
            return "🎯 Super Low"
    
    df['own_tier'] = df['own'].apply(ownership_tier)
    
    # Value tiers
    def value_tier(value):
        if value >= 6.0:
            return "🔥 Elite"
        elif value >= 5.0:
            return "⭐ Great"
        elif value >= 4.5:
            return "✅ Good"
        elif value >= 4.0:
            return "⚠️ Fair"
        else:
            return "❌ Poor"
    
    df['value_tier'] = df['value'].apply(value_tier)
    
    # Edge category
    def edge_category(row):
        own = row['own']
        value = row['value']
        proj = row['proj']
        
        if own < 15 and value >= 5.5 and proj >= 30:
            return "🎯 SMASH"
        elif own < 20 and value >= 5.0 and proj >= 25:
            return "💎 LEVERAGE"
        elif own < 25 and proj >= 40:
            return "✨ PIVOT"
        elif own >= 40 and proj >= 45:
            return "⚠️ CHALK"
        elif own >= 30 and proj >= 35:
            return "📊 CORE"
        elif 15 <= own < 30 and proj >= 30:
            return "✅ BALANCED"
        elif value >= 6.0 and proj < 25:
            return "🎲 DART"
        else:
            return "📋 FILL"
    
    df['edge'] = df.apply(edge_category, axis=1)
    
    return df

# ============================================================================
# LINEUP GENERATION
# ============================================================================

def generate_lineups(df, sport, n_lineups, locks, excludes, strategy="balanced", weights=None):
    """Generate optimized lineups."""
    
    lineups = []
    rng = np.random.RandomState()
    config = SPORT_CONFIGS[sport]
    
    for i in range(n_lineups):
        lineup = build_single_lineup(df, sport, locks, excludes, strategy, weights, rng, config)
        if lineup:
            lineups.append(lineup)
    
    # Sort by GPP score
    lineups.sort(key=lambda x: x['gpp_score'], reverse=True)
    
    return lineups

def build_single_lineup(df, sport, locks, excludes, strategy, weights, rng, config):
    """Build one lineup."""
    
    available = df[~df["player_id"].isin(excludes)].copy()
    selected = list(locks)
    salary_used = df[df["player_id"].isin(selected)]["salary"].sum()
    
    # Use provided weights or default
    if weights:
        own_penalty = weights['own_penalty']
        value_weight = weights['value']
        ceiling_weight = weights['ceiling']
        floor_weight = weights.get('floor', 0.3)
    else:
        # Default balanced weights
        own_penalty = -0.3
        value_weight = 0.3
        ceiling_weight = 0.3
        floor_weight = 0.3
    
    # Strategy-based targets
    if strategy == "cash":
        target_contrarian = 0
        max_chalk = 4
    elif strategy == "extreme_contrarian":
        target_contrarian = 4
        max_chalk = 1
    elif strategy == "heavy_contrarian":
        target_contrarian = 3
        max_chalk = 1
    elif strategy in ["balanced_gpp", "balanced"]:
        target_contrarian = 2
        max_chalk = 2
    elif strategy == "boom_bust":
        target_contrarian = 2
        max_chalk = 2
    elif strategy == "high_floor_gpp":
        target_contrarian = 1
        max_chalk = 2
    elif strategy == "h2h":
        target_contrarian = 1
        max_chalk = 2
    else:  # aggressive
        target_contrarian = 1
        max_chalk = 3
    
    contrarian_count = 0
    chalk_count = 0
    
    # Build roster
    while len(selected) < config["roster_size"]:
        candidates = available[~available["player_id"].isin(selected)].copy()
        if candidates.empty:
            break
        
        remaining_salary = config["salary_cap"] - salary_used
        candidates = candidates[candidates["salary"] <= remaining_salary]
        if candidates.empty:
            break
        
        spots_remaining = config["roster_size"] - len(selected)
        
        # Prioritize based on current mix
        if contrarian_count < target_contrarian and spots_remaining > 3:
            candidates = candidates[candidates['own'] < 15]
            boost = 1.3
        elif chalk_count >= max_chalk:
            candidates = candidates[candidates['own'] < 30]
            boost = 1.1
        else:
            boost = 1.0
        
        if candidates.empty:
            candidates = available[~available["player_id"].isin(selected)].copy()
            candidates = candidates[candidates["salary"] <= remaining_salary]
            boost = 1.0
        
        # Score
        candidates['score'] = (
            candidates['ceiling'] * ceiling_weight +
            candidates['proj'] * floor_weight +  # Projection represents floor
            candidates['value'] * 10 * value_weight +
            candidates['own'] * own_penalty +
            candidates['gpp_score'] * 0.15 +
            rng.uniform(0, 15, size=len(candidates))
        ) * boost
        
        player = candidates.nlargest(1, 'score').iloc[0]
        selected.append(player["player_id"])
        salary_used += player["salary"]
        
        if player['own'] < 15:
            contrarian_count += 1
        elif player['own'] >= 30:
            chalk_count += 1
    
    if len(selected) < config["roster_size"]:
        return None
    
    lineup_df = df[df["player_id"].isin(selected)].copy()
    
    # Assign positions
    lineup_df = assign_positions(lineup_df, sport, config)
    
    if lineup_df is None:
        return None
    
    return {
        "players": lineup_df,
        "proj": lineup_df["proj"].sum(),
        "ceiling": lineup_df["ceiling"].sum(),
        "salary": lineup_df["salary"].sum(),
        "own": lineup_df["own"].sum(),
        "avg_own": lineup_df["own"].mean(),
        "leverage": lineup_df["leverage"].mean(),
        "gpp_score": lineup_df["gpp_score"].sum(),
        "contrarian": contrarian_count,
        "chalk": chalk_count,
    }

def assign_positions(lineup_df, sport, config):
    """Assign roster positions."""
    
    lineup_df["slot"] = None
    
    if sport == "NBA":
        slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
        
        for slot in slots:
            unassigned = lineup_df[lineup_df["slot"].isna()].copy()
            if unassigned.empty:
                break
            
            if slot == "G":
                eligible = unassigned[unassigned["positions"].str.contains("PG|SG", na=False)]
            elif slot == "F":
                eligible = unassigned[unassigned["positions"].str.contains("SF|PF", na=False)]
            elif slot == "UTIL":
                eligible = unassigned
            else:
                eligible = unassigned[unassigned["positions"].str.contains(slot, na=False)]
            
            if not eligible.empty:
                player = eligible.iloc[0]
                lineup_df.loc[lineup_df["player_id"] == player["player_id"], "slot"] = slot
    
    else:  # NFL
        slots = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
        
        for slot in slots:
            unassigned = lineup_df[lineup_df["slot"].isna()].copy()
            if unassigned.empty:
                break
            
            if slot == "FLEX":
                eligible = unassigned[unassigned["positions"].str.contains("RB|WR|TE", na=False)]
            elif slot == "DST":
                eligible = unassigned[unassigned["positions"].str.contains("DST|DEF|D", na=False)]
            else:
                eligible = unassigned[unassigned["positions"].str.contains(slot, na=False)]
            
            if not eligible.empty:
                player = eligible.iloc[0]
                lineup_df.loc[lineup_df["player_id"] == player["player_id"], "slot"] = slot
    
    if lineup_df["slot"].isna().any():
        return None
    
    return lineup_df

# ============================================================================
# MAIN APP
# ============================================================================

st.title("🏆 DFS Elite Tools")

tab1, tab2 = st.tabs(["🏗️ Lineup Builder", "📊 Contest Review"])

# ============================================================================
# TAB 1: BUILDER
# ============================================================================

with tab1:
    st.header("🏗️ Lineup Builder")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload DFS CSV",
        type=["csv"],
        key="builder_upload",
        help="Upload your DraftKings or FanDuel CSV export"
    )
    
    if uploaded_file:
        df = load_csv(uploaded_file)
        
        if df is not None and len(df) > 0:
            st.session_state.player_pool = df
            
            # Detect sport
            positions = set()
            for pos_str in df["positions"].dropna():
                if isinstance(pos_str, str):
                    positions.update(pos_str.split("/"))
            
            sport = "NFL" if any(p in positions for p in ["QB", "RB", "WR", "TE"]) else "NBA"
            config = SPORT_CONFIGS[sport]
            
            st.success(f"✅ Loaded {len(df)} players for **{sport}**")
            
            # Analytics dashboard
            st.markdown("---")
            st.subheader("📊 Player Pool Analytics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                smash = len(df[df['edge'] == "🎯 SMASH"])
                st.metric("🎯 Smash Spots", smash)
            
            with col2:
                leverage = len(df[df['edge'] == "💎 LEVERAGE"])
                st.metric("💎 Leverage", leverage)
            
            with col3:
                contrarian = len(df[df['own'] < 15])
                st.metric("Contrarian", contrarian)
            
            with col4:
                mid = len(df[(df['own'] >= 15) & (df['own'] < 30)])
                st.metric("Mid Own", mid)
            
            with col5:
                chalk = len(df[df['own'] >= 40])
                st.metric("Chalk", chalk)
            
            # Top recommendations
            with st.expander("💡 Smart Recommendations", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🎯 Must-Play (Smash Spots)**")
                    smash_plays = df[df['edge'] == "🎯 SMASH"].nlargest(5, 'gpp_score')
                    for _, p in smash_plays.iterrows():
                        st.markdown(f"- **{p['name']}** ({p['positions']}): ${p['salary']:,.0f} | {p['proj']:.1f} pts | {p['own']:.1f}% own")
                
                with col2:
                    st.markdown("**⚠️ Avoid (Too Chalky)**")
                    avoid = df[df['own'] >= 50].nlargest(5, 'own')
                    if len(avoid) > 0:
                        for _, p in avoid.iterrows():
                            st.markdown(f"- **{p['name']}**: {p['own']:.1f}% owned")
                    else:
                        st.info("No mega-chalk (50%+) detected")
            
            # Contest-Based Settings
            st.markdown("---")
            st.subheader("⚙️ Contest Settings")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                contest_type = st.selectbox(
                    "Contest Type",
                    ["GPP (Tournament)", "Cash Game (Double-Up)", "50/50", "Head-to-Head"],
                    help="Type of contest you're entering"
                )
            
            with col2:
                entry_fee = st.selectbox(
                    "Entry Fee",
                    ["$0.25", "$1", "$3", "$5", "$10", "$25", "$50", "$100+"],
                    index=2
                )
            
            with col3:
                contest_size = st.selectbox(
                    "Contest Size",
                    ["Small (3-20)", "Medium (20-100)", "Large (100-1000)", "Massive (1000+)"],
                    index=1
                )
            
            # Auto-generate strategy based on contest
            def get_contest_strategy(contest_type, entry_fee, contest_size):
                """Smart strategy based on contest parameters."""
                
                # Extract fee value
                fee_value = float(entry_fee.replace('$', '').replace('+', '').split('-')[0])
                
                # Determine strategy
                if contest_type == "Cash Game (Double-Up)" or contest_type == "50/50":
                    return {
                        "strategy": "cash",
                        "target_own": "35-45%",
                        "description": "Safe, High-Floor Build",
                        "contrarian": 0,
                        "mid": 6,
                        "chalk": 2,
                        "focus": "Maximize floor, minimize variance",
                        "weights": {
                            "own_penalty": -0.1,
                            "value": 0.4,
                            "ceiling": 0.2,
                            "floor": 0.4
                        }
                    }
                
                elif contest_type == "Head-to-Head":
                    return {
                        "strategy": "h2h",
                        "target_own": "30-40%",
                        "description": "Balanced High-Floor",
                        "contrarian": 1,
                        "mid": 5,
                        "chalk": 2,
                        "focus": "Beat one opponent, high consistency",
                        "weights": {
                            "own_penalty": -0.2,
                            "value": 0.35,
                            "ceiling": 0.25,
                            "floor": 0.4
                        }
                    }
                
                else:  # GPP
                    # Adjust based on size and stakes
                    if "Massive" in contest_size:
                        # Massive GPPs need extreme differentiation
                        return {
                            "strategy": "extreme_contrarian",
                            "target_own": "18-25%",
                            "description": "Ultra Contrarian Boom/Bust",
                            "contrarian": 4,
                            "mid": 3,
                            "chalk": 1,
                            "focus": "Maximum leverage, stand out from field",
                            "weights": {
                                "own_penalty": -0.6,
                                "value": 0.3,
                                "ceiling": 0.5,
                                "floor": 0.2
                            }
                        }
                    
                    elif "Large" in contest_size:
                        # Large GPPs need strong differentiation
                        return {
                            "strategy": "heavy_contrarian",
                            "target_own": "22-28%",
                            "description": "Heavy Contrarian with Upside",
                            "contrarian": 3,
                            "mid": 4,
                            "chalk": 1,
                            "focus": "Strong leverage + ceiling",
                            "weights": {
                                "own_penalty": -0.5,
                                "value": 0.3,
                                "ceiling": 0.45,
                                "floor": 0.25
                            }
                        }
                    
                    elif "Medium" in contest_size:
                        # Medium GPPs - balanced contrarian
                        return {
                            "strategy": "balanced_gpp",
                            "target_own": "25-32%",
                            "description": "Balanced GPP Mix",
                            "contrarian": 2,
                            "mid": 4,
                            "chalk": 2,
                            "focus": "Good leverage with stability",
                            "weights": {
                                "own_penalty": -0.35,
                                "value": 0.3,
                                "ceiling": 0.4,
                                "floor": 0.3
                            }
                        }
                    
                    else:  # Small
                        # Small GPPs - can play safer
                        if fee_value >= 25:
                            # High stakes small field
                            return {
                                "strategy": "high_floor_gpp",
                                "target_own": "28-35%",
                                "description": "High-Floor GPP Build",
                                "contrarian": 1,
                                "mid": 5,
                                "chalk": 2,
                                "focus": "Ceiling with safety",
                                "weights": {
                                    "own_penalty": -0.25,
                                    "value": 0.35,
                                    "ceiling": 0.35,
                                    "floor": 0.3
                                }
                            }
                        else:
                            # Low stakes small field - lottery ticket
                            return {
                                "strategy": "boom_bust",
                                "target_own": "22-30%",
                                "description": "Boom/Bust Upside",
                                "contrarian": 2,
                                "mid": 4,
                                "chalk": 2,
                                "focus": "Maximum ceiling potential",
                                "weights": {
                                    "own_penalty": -0.4,
                                    "value": 0.25,
                                    "ceiling": 0.5,
                                    "floor": 0.25
                                }
                            }
            
            strategy_config = get_contest_strategy(contest_type, entry_fee, contest_size)
            
            # Show recommendation
            st.info(f"""
            **📊 Recommended Strategy: {strategy_config['description']}**
            
            **Target Mix:**
            - Contrarian (<15%): {strategy_config['contrarian']} players
            - Mid Ownership (15-30%): {strategy_config['mid']} players
            - Popular/Chalk (30%+): {strategy_config['chalk']} players
            
            **Average Ownership:** {strategy_config['target_own']}
            
            **Focus:** {strategy_config['focus']}
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_lineups = st.number_input("# Lineups", 1, 150, 20, help="Number of lineups to generate")
            
            with col2:
                min_proj = st.number_input("Min Projection", 0.0, 100.0, 0.0, 0.5, help="Filter players below this projection")
            
            # Player pool with filters
            st.markdown("---")
            st.subheader("👥 Player Pool")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                own_filter = st.multiselect(
                    "Ownership Tiers",
                    ["🎯 Super Low", "💎 Contrarian", "✅ Mid", "⚠️ Popular", "❌ Chalk"],
                    default=["🎯 Super Low", "💎 Contrarian", "✅ Mid", "⚠️ Popular"]
                )
            
            with col2:
                if sport == "NBA":
                    pos_filter = st.multiselect("Positions", ["PG", "SG", "SF", "PF", "C"])
                else:
                    pos_filter = st.multiselect("Positions", ["QB", "RB", "WR", "TE", "DST"])
            
            with col3:
                edge_filter = st.multiselect(
                    "Edge Types",
                    ["🎯 SMASH", "💎 LEVERAGE", "✨ PIVOT", "✅ BALANCED", "🎲 DART"]
                )
            
            # Apply filters
            filtered = df.copy()
            
            if own_filter:
                filtered = filtered[filtered['own_tier'].isin(own_filter)]
            
            if pos_filter:
                filtered = filtered[filtered['positions'].apply(
                    lambda x: any(p in str(x) for p in pos_filter)
                )]
            
            if edge_filter:
                filtered = filtered[filtered['edge'].isin(edge_filter)]
            
            if min_proj > 0:
                filtered = filtered[filtered['proj'] >= min_proj]
            
            st.info(f"📊 Showing {len(filtered)} of {len(df)} players")
            
            # Add lock/exclude columns
            filtered["🔒"] = False
            filtered["❌"] = False
            
            # Display editable table
            edited = st.data_editor(
                filtered[[
                    "🔒", "❌", "name", "positions", "team", "salary", 
                    "proj", "own", "own_tier", "value", "edge", "gpp_score"
                ]].sort_values("gpp_score", ascending=False),
                column_config={
                    "🔒": st.column_config.CheckboxColumn("Lock"),
                    "❌": st.column_config.CheckboxColumn("Exclude"),
                    "name": "Player",
                    "positions": "Pos",
                    "team": "Team",
                    "salary": st.column_config.NumberColumn("Salary", format="$%d"),
                    "proj": st.column_config.NumberColumn("Proj", format="%.1f"),
                    "own": st.column_config.NumberColumn("Own%", format="%.1f%%"),
                    "own_tier": "Own Tier",
                    "value": st.column_config.NumberColumn("Value", format="%.2f"),
                    "edge": "Edge",
                    "gpp_score": st.column_config.NumberColumn("GPP", format="%.0f"),
                },
                hide_index=True,
                use_container_width=True,
                height=500
            )
            
            # Get locks and excludes
            locks = []
            excludes = []
            
            for idx, row in edited.iterrows():
                if row["🔒"]:
                    locks.append(df[df['name'] == row['name']].iloc[0]['player_id'])
                if row["❌"]:
                    excludes.append(df[df['name'] == row['name']].iloc[0]['player_id'])
            
            if locks:
                st.success(f"🔒 Locked: {len(locks)} players")
            if excludes:
                st.info(f"❌ Excluded: {len(excludes)} players")
            
            # Generate button
            st.markdown("---")
            if st.button("🚀 Generate Lineups", type="primary", use_container_width=True):
                with st.spinner(f"Generating {n_lineups} optimized lineups for {contest_type}..."):
                    lineups = generate_lineups(
                        df, sport, n_lineups, locks, excludes, 
                        strategy_config['strategy'], strategy_config['weights']
                    )
                    st.session_state.generated_lineups = lineups
                    st.success(f"✅ Generated {len(lineups)} {strategy_config['description']} lineups!")
                    st.rerun()
            
            # Display lineups
            if st.session_state.generated_lineups:
                st.markdown("---")
                st.subheader("📋 Generated Lineups")
                
                lineups = st.session_state.generated_lineups
                
                # Summary
                summary_data = []
                for i, lu in enumerate(lineups):
                    summary_data.append({
                        "#": i + 1,
                        "Proj": lu['proj'],
                        "Ceil": lu['ceiling'],
                        "Own": lu['avg_own'],
                        "Low": lu['contrarian'],
                        "Chalk": lu['chalk'],
                        "GPP": lu['gpp_score'],
                        "Sal": lu['salary']
                    })
                
                summary_df = pd.DataFrame(summary_data)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Proj", f"{summary_df['Proj'].mean():.1f}")
                with col2:
                    st.metric("Avg Ceil", f"{summary_df['Ceil'].mean():.1f}")
                with col3:
                    st.metric("Avg Own", f"{summary_df['Own'].mean():.1f}%")
                with col4:
                    st.metric("Avg Low Own", f"{summary_df['Low'].mean():.1f}")
                
                st.dataframe(
                    summary_df,
                    column_config={
                        "#": st.column_config.NumberColumn("#", width="small"),
                        "Proj": st.column_config.NumberColumn("Proj", format="%.0f"),
                        "Ceil": st.column_config.NumberColumn("Ceil", format="%.0f"),
                        "Own": st.column_config.NumberColumn("Own%", format="%.1f"),
                        "Low": "Contra",
                        "Chalk": "Chalk",
                        "GPP": st.column_config.NumberColumn("GPP", format="%.0f"),
                        "Sal": st.column_config.NumberColumn("Salary", format="$%d"),
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # View specific lineup
                st.markdown("---")
                lineup_choice = st.selectbox(
                    "View lineup details:",
                    [f"Lineup {i+1} - {lu['proj']:.0f} pts | {lu['avg_own']:.1f}% own" 
                     for i, lu in enumerate(lineups)]
                )
                
                idx = int(lineup_choice.split()[1]) - 1
                lineup = lineups[idx]
                
                # Add actuals column
                lineup_players = lineup['players'].copy()
                lineup_players['actual'] = lineup_players['name'].map(st.session_state.player_actuals).fillna(0)
                
                st.dataframe(
                    lineup_players[[
                        'slot', 'name', 'positions', 'salary', 'proj', 
                        'own', 'own_tier', 'edge', 'actual'
                    ]],
                    column_config={
                        'slot': 'Slot',
                        'name': 'Player',
                        'positions': 'Pos',
                        'salary': st.column_config.NumberColumn("Salary", format="$%d"),
                        'proj': st.column_config.NumberColumn("Proj", format="%.1f"),
                        'own': st.column_config.NumberColumn("Own%", format="%.1f"),
                        'own_tier': "Tier",
                        'edge': "Edge",
                        'actual': st.column_config.NumberColumn("✏️ Actual", format="%.1f"),
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Projection", f"{lineup['proj']:.1f}")
                with col2:
                    st.metric("Ceiling", f"{lineup['ceiling']:.1f}")
                with col3:
                    st.metric("Avg Own", f"{lineup['avg_own']:.1f}%")
                with col4:
                    total_actual = lineup_players['actual'].sum()
                    if total_actual > 0:
                        diff = total_actual - lineup['proj']
                        st.metric("Actual", f"{total_actual:.1f}", delta=f"{diff:+.1f}")
                    else:
                        st.metric("Actual", "Pending")
                
                # Export
                st.markdown("---")
                if st.button("💾 Export All Lineups"):
                    export_data = []
                    for i, lu in enumerate(lineups):
                        for _, p in lu['players'].iterrows():
                            export_data.append({
                                "Lineup": i + 1,
                                "Slot": p['slot'],
                                "Name": p['name'],
                                "ID": "",  # DraftKings will need player IDs
                                "Position": p['positions'],
                                "Salary": p['salary'],
                                "Projection": p['proj'],
                                "Ownership": p['own']
                            })
                    
                    export_df = pd.DataFrame(export_data)
                    csv = export_df.to_csv(index=False)
                    
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"{sport}_lineups_{n_lineups}.csv",
                        "text/csv",
                        use_container_width=True
                    )
    
    else:
        st.info("👆 Upload your CSV file to begin")
        st.markdown("""
        ### Required Columns:
        - **Player/Name** - Player name
        - **Position** - Player positions (PG, SG, etc.)
        - **Salary** - DK salary
        - **Projection/Proj/FPTS** - Projected points
        - **Ownership/Own%** (optional) - Projected ownership
        
        The app will auto-detect your CSV format and load accordingly.
        """)

# ============================================================================
# TAB 2: CONTEST REVIEW
# ============================================================================

with tab2:
    st.header("📊 Contest Review")
    
    contest_file = st.file_uploader(
        "Upload DraftKings Contest Export",
        type="csv",
        key="contest_upload"
    )
    
    if contest_file:
        df_contest = pd.read_csv(contest_file)
        
        # Parse player stats
        player_stats = df_contest[['Player', 'Roster Position', '%Drafted', 'FPTS']].copy()
        player_stats = player_stats.dropna(subset=['Player'])
        player_stats.columns = ['name', 'position', 'own', 'actual']
        player_stats['own'] = player_stats['own'].str.replace('%', '').astype(float)
        
        # Store actuals
        for _, row in player_stats.iterrows():
            st.session_state.player_actuals[row['name']] = row['actual']
        
        st.success(f"✅ Loaded contest with {len(df_contest):,} entries")
        st.info("💡 Actuals populated! Go to Builder tab to see your lineup performance")
        
        # Show top performers
        st.markdown("---")
        st.subheader("🌟 Top Performers")
        
        player_stats_display = player_stats.sort_values('actual', ascending=False).head(20)
        
        st.dataframe(
            player_stats_display,
            column_config={
                "name": "Player",
                "position": "Pos",
                "own": st.column_config.NumberColumn("Own%", format="%.1f%%"),
                "actual": st.column_config.NumberColumn("Points", format="%.1f"),
            },
            use_container_width=True,
            hide_index=True
        )
    
    else:
        st.info("👆 Upload DraftKings contest export to analyze results")
