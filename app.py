"""
DFS Elite Lineup Builder v4.0
Focus: High-scoring games + Smart stacking + Tournament strategy
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import io

# ============================================================================
# CONFIGURATION
# ============================================================================

class Sport(Enum):
    NBA = "NBA"
    NFL = "NFL"
    MLB = "MLB"
    NHL = "NHL"

@dataclass
class SportConfig:
    salary_cap: int
    roster_size: int
    positions: Dict[str, int]

SPORT_CONFIGS = {
    Sport.NBA: SportConfig(
        salary_cap=50000,
        roster_size=8,
        positions={"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1, "G": 1, "F": 1, "UTIL": 1}
    ),
    Sport.NFL: SportConfig(
        salary_cap=50000,
        roster_size=9,
        positions={"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "DST": 1}
    ),
    Sport.MLB: SportConfig(
        salary_cap=50000,
        roster_size=10,
        positions={"P": 2, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}
    ),
    Sport.NHL: SportConfig(
        salary_cap=50000,
        roster_size=9,
        positions={"C": 2, "W": 3, "D": 2, "G": 1, "UTIL": 1}
    ),
}

# ============================================================================
# DATA PROCESSING
# ============================================================================

def load_and_normalize_csv(uploaded_file) -> pd.DataFrame:
    """Load CSV and normalize to standard format."""
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    
    # Column mapping
    renames = {
        "player": "name",
        "position": "positions", 
        "pos": "positions",
        "sal": "salary",
        "opponent": "opp",
        "projection": "proj",
        "fpts": "proj",
        "points": "proj",
        "ownership": "own",
        "own%": "own",
        "ownership%": "own",
    }
    df = df.rename(columns=renames)
    
    # Required columns
    required = ["name", "positions", "salary", "proj"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return pd.DataFrame()
    
    # Clean salary
    df["salary"] = df["salary"].astype(str).str.replace(r"[\$,]", "", regex=True)
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)
    
    # Clean projection
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce").fillna(0).astype(float)
    
    # Clean ownership
    if "own" in df.columns:
        df["own"] = df["own"].astype(str).str.replace("%", "")
        df["own"] = pd.to_numeric(df["own"], errors="coerce").fillna(15)
        if df["own"].max() <= 1.0:
            df["own"] = df["own"] * 100
    else:
        df["own"] = 15.0
    
    # Team/opp
    if "team" not in df.columns:
        df["team"] = "UNK"
    if "opp" not in df.columns:
        df["opp"] = "UNK"
    
    # Game ID
    if "game_id" not in df.columns:
        df["game_id"] = df["team"] + "_" + df["opp"]
    
    df["player_id"] = df["name"] + "_" + df["positions"]
    
    return df

def detect_sport(df: pd.DataFrame) -> Sport:
    """Auto-detect sport from positions."""
    positions = set()
    for pos_str in df["positions"].dropna().unique():
        positions.update(str(pos_str).upper().split("/"))
    
    if {"QB", "RB", "WR"}.intersection(positions):
        return Sport.NFL
    elif {"PG", "SG", "SF"}.intersection(positions):
        return Sport.NBA
    elif {"1B", "2B", "SS"}.intersection(positions):
        return Sport.MLB
    elif {"LW", "RW", "D"}.intersection(positions):
        return Sport.NHL
    
    return Sport.NBA  # Default

def calculate_game_totals(df: pd.DataFrame, sport: Sport) -> pd.DataFrame:
    """Calculate game environment from projections."""
    df = df.copy()
    
    # Group by game and calculate totals
    game_stats = df.groupby("game_id").agg({
        "proj": "sum",
        "salary": "sum"
    }).reset_index()
    
    game_stats.columns = ["game_id", "game_total", "game_sal"]
    df = df.merge(game_stats, on="game_id", how="left")
    
    # Categorize game environment
    def categorize_game(total, sport):
        if sport == Sport.NBA:
            if total >= 240: return "🔥 Elite"
            if total >= 220: return "⭐ Great"
            if total >= 200: return "✅ Good"
            return "⚠️ Avoid"
        elif sport == Sport.NFL:
            if total >= 85: return "🔥 Elite"
            if total >= 70: return "⭐ Great"
            if total >= 60: return "✅ Good"
            return "⚠️ Avoid"
        else:
            return "✅ Good"
    
    df["game_env"] = df["game_total"].apply(lambda x: categorize_game(x, sport))
    
    return df

def calculate_metrics(df: pd.DataFrame, sport: Sport) -> pd.DataFrame:
    """Calculate all metrics."""
    df = df.copy()
    
    # Value
    df["value"] = np.where(df["salary"] > 0, df["proj"] / (df["salary"] / 1000), 0).round(2)
    
    # Leverage
    df["leverage"] = ((df["value"] / 5 * 100).clip(0, 100) - df["own"]).round(1)
    
    # Ceiling
    if sport == Sport.NFL:
        df["ceil_mult"] = df["positions"].apply(
            lambda x: 1.6 if "QB" in str(x) else 1.5 if "RB" in str(x) else 1.4
        )
    else:
        df["ceil_mult"] = 1.4
    
    df["ceiling"] = (df["proj"] * df["ceil_mult"]).round(1)
    
    # GPP Score with game environment
    df["gpp_score"] = (
        df["ceiling"] * 0.30 +
        df["value"] * 8 * 0.25 +
        df["leverage"] * 0.20 +
        df["game_total"] * 0.15 +  # Game environment!
        (100 - df["own"]) * 0.10
    ).round(1)
    
    return df

# ============================================================================
# LINEUP GENERATION
# ============================================================================

def find_game_stacks(df: pd.DataFrame, sport: Sport) -> List[Dict]:
    """Find best game stacks from high-scoring games."""
    stacks = []
    
    # Get elite/great games only
    if sport == Sport.NBA:
        elite_games = df[df["game_total"] >= 220].copy()
    elif sport == Sport.NFL:
        elite_games = df[df["game_total"] >= 70].copy()
    else:
        elite_games = df.copy()
    
    for game_id in elite_games["game_id"].unique():
        game_df = df[df["game_id"] == game_id].copy()
        
        # Get best players from this game
        game_df = game_df.nlargest(6, "ceiling")
        
        if len(game_df) >= 2:
            stacks.append({
                "game_id": game_id,
                "players": game_df["player_id"].tolist(),
                "game_total": game_df["game_total"].iloc[0],
                "score": game_df["ceiling"].sum() + game_df["leverage"].sum(),
                "teams": game_df["team"].unique().tolist()
            })
    
    return sorted(stacks, key=lambda x: x["score"], reverse=True)

def build_lineup(
    df: pd.DataFrame,
    sport: Sport,
    config: SportConfig,
    locks: List[str],
    excludes: List[str],
    stacks: List[Dict],
    correlation: float,
    rng: np.random.Generator
) -> Optional[Dict]:
    """Build single lineup focused on high-scoring games."""
    
    # Start with locks
    selected = list(locks)
    selected_df = df[df["player_id"].isin(selected)]
    
    used_sal = int(selected_df["salary"].sum()) if not selected_df.empty else 0
    remaining_sal = config.salary_cap - used_sal
    remaining_spots = config.roster_size - len(selected)
    
    if remaining_spots < 0 or used_sal > config.salary_cap:
        return None
    
    # Apply stack if high correlation
    if correlation >= 0.6 and stacks and remaining_spots >= 2 and not selected:
        # Pick best stack
        if stacks:
            stack = rng.choice(stacks[:5])  # Top 5 stacks
            
            for pid in stack["players"][:4]:  # Max 4 from stack
                if pid not in selected and pid not in excludes and remaining_spots > 0:
                    player = df[df["player_id"] == pid]
                    if not player.empty:
                        sal = int(player.iloc[0]["salary"])
                        if sal <= remaining_sal:
                            selected.append(pid)
                            remaining_sal -= sal
                            remaining_spots -= 1
    
    # Fill remaining - PRIORITIZE ELITE/GREAT GAMES
    available = df[
        (~df["player_id"].isin(selected)) &
        (~df["player_id"].isin(excludes))
    ].copy()
    
    # Separate by game environment
    elite_players = available[available["game_env"].isin(["🔥 Elite", "⭐ Great"])].copy()
    other_players = available[~available["game_env"].isin(["🔥 Elite", "⭐ Great"])].copy()
    
    # Score players
    elite_players["score"] = (
        elite_players["ceiling"] * 0.4 +
        elite_players["leverage"] * 1.2 +
        elite_players["game_total"] * 0.3 +
        rng.normal(0, 10, len(elite_players))
    )
    
    other_players["score"] = (
        other_players["ceiling"] * 0.4 +
        other_players["value"] * 10 +
        rng.normal(0, 10, len(other_players))
    )
    
    # Fill from elite games first
    elite_players = elite_players[elite_players["salary"] <= remaining_sal]
    elite_players = elite_players.sort_values("score", ascending=False)
    
    for _, player in elite_players.iterrows():
        if remaining_spots == 0:
            break
        if player["salary"] <= remaining_sal:
            selected.append(player["player_id"])
            remaining_sal -= int(player["salary"])
            remaining_spots -= 1
    
    # Fill remaining from any game if needed
    if remaining_spots > 0:
        other_players = other_players[other_players["salary"] <= remaining_sal]
        other_players = other_players.sort_values("score", ascending=False)
        
        for _, player in other_players.iterrows():
            if remaining_spots == 0:
                break
            if player["salary"] <= remaining_sal:
                selected.append(player["player_id"])
                remaining_sal -= int(player["salary"])
                remaining_spots -= 1
    
    if remaining_spots > 0:
        return None
    
    # Calculate lineup metrics
    lineup_df = df[df["player_id"].isin(selected)]
    
    return {
        "players": lineup_df,
        "proj": lineup_df["proj"].sum(),
        "salary": lineup_df["salary"].sum(),
        "own": lineup_df["own"].sum(),
        "ceiling": lineup_df["ceiling"].sum(),
        "leverage": lineup_df["leverage"].mean(),
        "game_env": lineup_df["game_total"].mean()
    }

def assign_positions_nba(lineup_df: pd.DataFrame) -> pd.DataFrame:
    """Assign NBA positions."""
    result = []
    available = lineup_df.copy()
    slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    
    for slot in slots:
        if slot == "G":
            eligible = available[available["positions"].str.contains("PG|SG", case=False, na=False)]
        elif slot == "F":
            eligible = available[available["positions"].str.contains("SF|PF", case=False, na=False)]
        elif slot == "UTIL":
            eligible = available
        else:
            eligible = available[available["positions"].str.contains(slot, case=False, na=False)]
        
        if not eligible.empty:
            player = eligible.iloc[0]
            result.append({**player.to_dict(), "slot": slot})
            available = available[available["player_id"] != player["player_id"]]
    
    return pd.DataFrame(result)

def assign_positions_nfl(lineup_df: pd.DataFrame) -> pd.DataFrame:
    """Assign NFL positions."""
    result = []
    available = lineup_df.copy()
    slots = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
    
    for slot in slots:
        if slot == "FLEX":
            eligible = available[available["positions"].str.contains("RB|WR|TE", case=False, na=False)]
        elif slot == "DST":
            eligible = available[available["positions"].str.contains("DST|DEF|D", case=False, na=False)]
        else:
            eligible = available[available["positions"].str.contains(slot, case=False, na=False)]
        
        if not eligible.empty:
            player = eligible.iloc[0]
            result.append({**player.to_dict(), "slot": slot})
            available = available[available["player_id"] != player["player_id"]]
    
    return pd.DataFrame(result)

def generate_lineups(
    df: pd.DataFrame,
    sport: Sport,
    config: SportConfig,
    n_lineups: int,
    locks: List[str],
    excludes: List[str],
    correlation: float
) -> List[Dict]:
    """Generate multiple lineups."""
    stacks = find_game_stacks(df, sport)
    lineups = []
    rng = np.random.default_rng()
    
    attempts = 0
    max_attempts = n_lineups * 50
    
    while len(lineups) < n_lineups and attempts < max_attempts:
        attempts += 1
        
        lineup = build_lineup(df, sport, config, locks, excludes, stacks, correlation, rng)
        
        if lineup:
            # Assign positions
            if sport == Sport.NBA:
                lineup["players"] = assign_positions_nba(lineup["players"])
            elif sport == Sport.NFL:
                lineup["players"] = assign_positions_nfl(lineup["players"])
            
            # Check if valid
            if len(lineup["players"]) == config.roster_size:
                lineups.append(lineup)
    
    # Sort by ceiling + leverage
    lineups = sorted(lineups, key=lambda x: x["ceiling"] * 0.5 + x["leverage"] * 3, reverse=True)
    
    return lineups

# ============================================================================
# UI
# ============================================================================

st.set_page_config(page_title="DFS Elite Builder v4.0", layout="wide")

st.title("🏆 DFS Elite Lineup Builder v4.0")
st.markdown("**Focus: High-Scoring Games + Smart Stacking**")

# File upload
uploaded_file = st.file_uploader("Upload DFS CSV", type="csv")

if uploaded_file:
    df = load_and_normalize_csv(uploaded_file)
    
    if df.empty:
        st.stop()
    
    sport = detect_sport(df)
    config = SPORT_CONFIGS[sport]
    
    st.success(f"✅ Detected: **{sport.value}** ({len(df)} players)")
    
    # Calculate game totals and metrics
    df = calculate_game_totals(df, sport)
    df = calculate_metrics(df, sport)
    
    # Game environment breakdown
    st.subheader("🎯 Game Environment Analysis")
    
    game_breakdown = df.groupby("game_env").agg({
        "player_id": "count",
        "game_total": "first"
    }).reset_index()
    
    cols = st.columns(4)
    for i, (_, row) in enumerate(game_breakdown.iterrows()):
        with cols[i % 4]:
            st.metric(
                row["game_env"],
                f"{row['player_id']} players",
                f"Avg: {row['game_total']:.0f}"
            )
    
    st.info("**Strategy:** Stack from 🔥 Elite and ⭐ Great games for maximum upside!")
    
    # Filters
    st.subheader("⚙️ Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        game_filter = st.multiselect(
            "Game Environment",
            ["🔥 Elite", "⭐ Great", "✅ Good", "⚠️ Avoid"],
            default=["🔥 Elite", "⭐ Great", "✅ Good"]
        )
    
    with col2:
        n_lineups = st.number_input("# Lineups", 1, 150, 20)
    
    with col3:
        correlation = st.slider("Correlation", 0.0, 1.0, 0.7, 0.1)
    
    # Filter players
    filtered = df[df["game_env"].isin(game_filter)].copy()
    filtered["lock"] = False
    filtered["exclude"] = False
    
    st.subheader(f"📊 Player Pool ({len(filtered)} players)")
    
    # Display (without player_id column, but keep it in filtered)
    display_df = filtered[[
        "lock", "exclude", "name", "positions", "team", "opp",
        "salary", "proj", "value", "own", "leverage", 
        "game_env", "ceiling", "gpp_score"
    ]].copy()
    
    edited = st.data_editor(
        display_df,
        column_config={
            "lock": st.column_config.CheckboxColumn("🔒"),
            "exclude": st.column_config.CheckboxColumn("❌"),
            "game_env": st.column_config.TextColumn("Game"),
            "salary": st.column_config.NumberColumn("Salary", format="$%d"),
            "proj": st.column_config.NumberColumn("Proj", format="%.1f"),
            "value": st.column_config.NumberColumn("Val", format="%.2f"),
            "own": st.column_config.NumberColumn("Own%", format="%.1f%%"),
            "leverage": st.column_config.NumberColumn("Lev", format="%+.1f"),
            "ceiling": st.column_config.NumberColumn("Ceil", format="%.1f"),
            "gpp_score": st.column_config.NumberColumn("GPP", format="%.1f"),
        },
        hide_index=True,
        use_container_width=True,
        height=400
    )
    
    # Get locks/excludes by matching back to original filtered dataframe
    locks = []
    excludes = []
    
    for idx, row in edited.iterrows():
        # Find matching player in filtered
        match = filtered[
            (filtered["name"] == row["name"]) & 
            (filtered["positions"] == row["positions"])
        ]
        
        if not match.empty:
            player_id = match.iloc[0]["player_id"]
            if row["lock"]:
                locks.append(player_id)
            if row["exclude"]:
                excludes.append(player_id)
    
    # Generate
    if st.button("🚀 Generate Lineups", type="primary", use_container_width=True):
        with st.spinner("Building elite lineups..."):
            lineups = generate_lineups(
                filtered, sport, config, n_lineups,
                locks, excludes, correlation
            )
        
        if not lineups:
            st.error("Could not generate lineups. Try adjusting filters.")
        else:
            st.session_state.lineups = lineups
            st.success(f"✅ Generated {len(lineups)} lineups!")
    
    # Display lineups
    if "lineups" in st.session_state and st.session_state.lineups:
        lineups = st.session_state.lineups
        
        st.subheader("📋 Generated Lineups")
        
        # Summary
        summary = pd.DataFrame([{
            "Lineup": i+1,
            "Ceiling": lu["ceiling"],
            "Proj": lu["proj"],
            "Own": lu["own"],
            "Lev": lu["leverage"],
            "Game": lu["game_env"],
            "Sal": lu["salary"]
        } for i, lu in enumerate(lineups)])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Ceiling", f"{summary['Ceiling'].mean():.1f}")
        with col2:
            st.metric("Avg Own", f"{summary['Own'].mean():.1f}%")
        with col3:
            st.metric("Avg Lev", f"{summary['Lev'].mean():+.1f}")
        with col4:
            st.metric("Avg Game", f"{summary['Game'].mean():.0f}")
        
        st.dataframe(summary, use_container_width=True, hide_index=True)
        
        # Detail
        st.markdown("---")
        choice = st.selectbox(
            "View lineup:",
            [f"Lineup {i+1} (Ceil {lu['ceiling']:.0f}, Own {lu['own']:.0f}%)" 
             for i, lu in enumerate(lineups)]
        )
        
        idx = int(choice.split()[1]) - 1
        lineup = lineups[idx]
        
        st.dataframe(
            lineup["players"][[
                "slot", "name", "positions", "team", 
                "salary", "proj", "own", "ceiling", "game_env"
            ]],
            use_container_width=True,
            hide_index=True
        )
        
        # Export
        if st.button("💾 Export All Lineups"):
            export_data = []
            for lu in lineups:
                row = {}
                for _, p in lu["players"].iterrows():
                    row[p["slot"]] = p["name"]
                export_data.append(row)
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                "Download CSV",
                csv,
                "lineups.csv",
                "text/csv"
            )
