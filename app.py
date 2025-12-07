"""
Elite Multi-Sport DFS Lineup Builder
Supports: NBA, NFL, MLB, NHL, PGA
Focus: Contrarian leverage, advanced stacking, tournament upside
"""

import io
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Elite DFS Builder",
    layout="wide",
    initial_sidebar_state="expanded"
)

class Sport(Enum):
    """Supported sports with their characteristics."""
    NBA = "NBA"
    NFL = "NFL"
    MLB = "MLB"
    NHL = "NHL"
    PGA = "PGA"
    UNKNOWN = "UNKNOWN"


@dataclass
class SportConfig:
    """Sport-specific configuration."""
    name: str
    roster_size: int
    salary_cap: int
    positions: Dict[str, int]
    flex_positions: List[str]
    primary_positions: List[str]
    
    
SPORT_CONFIGS = {
    Sport.NBA: SportConfig(
        name="NBA",
        roster_size=8,
        salary_cap=50000,
        positions={"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1, "G": 1, "F": 1, "UTIL": 1},
        flex_positions=["G", "F", "UTIL"],
        primary_positions=["PG", "SG", "SF", "PF", "C"]
    ),
    Sport.NFL: SportConfig(
        name="NFL",
        roster_size=9,
        salary_cap=50000,
        positions={"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "DST": 1},
        flex_positions=["FLEX"],
        primary_positions=["QB", "RB", "WR", "TE", "DST"]
    ),
    Sport.MLB: SportConfig(
        name="MLB",
        roster_size=10,
        salary_cap=50000,
        positions={"P": 2, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3},
        flex_positions=[],
        primary_positions=["P", "C", "1B", "2B", "3B", "SS", "OF"]
    ),
    Sport.NHL: SportConfig(
        name="NHL",
        roster_size=9,
        salary_cap=50000,
        positions={"C": 2, "W": 3, "D": 2, "G": 2, "UTIL": 1},
        flex_positions=["UTIL"],
        primary_positions=["C", "W", "D", "G"]
    ),
}

# -------------------------------------------------------------------
# SPORT DETECTION
# -------------------------------------------------------------------

def detect_sport_from_positions(df: pd.DataFrame) -> Sport:
    """Detect sport from position data."""
    if df.empty or "positions" not in df.columns:
        return Sport.UNKNOWN
    
    # Get all unique positions
    all_positions = set()
    for val in df["positions"].dropna():
        for pos in str(val).upper().replace(" ", "").split("/"):
            all_positions.add(pos.strip())
    
    # Sport signatures
    if {"QB", "RB", "WR", "TE"} & all_positions:
        return Sport.NFL
    if {"PG", "SG", "SF", "PF", "C"} & all_positions:
        return Sport.NBA
    if {"P", "1B", "2B", "3B", "SS", "OF"} & all_positions:
        return Sport.MLB
    if {"C", "W", "D", "G", "LW", "RW"} & all_positions:
        return Sport.NHL
    
    return Sport.UNKNOWN


# -------------------------------------------------------------------
# DATA NORMALIZATION
# -------------------------------------------------------------------

def normalize_csv_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize any DFS CSV into standard format."""
    df = df.copy()
    
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Standard column mapping
    column_map = {
        "player": "name",
        "name": "name",
        "salary": "salary",
        "sal": "salary",
        "position": "positions",
        "pos": "positions",
        "team": "team",
        "tm": "team",
        "opponent": "opp",
        "opp": "opp",
        "opp_team": "opp",
        "projection": "proj",
        "proj": "proj",
        "fpts": "proj",
        "points": "proj",
        "ownership": "own",
        "ownership%": "own",
        "own": "own",
        "own%": "own",
        "own_proj": "own",
    }
    
    df = df.rename(columns=column_map)
    
    # Required columns
    required = ["name", "salary", "positions", "proj"]
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        st.error(f"Found columns: {list(df.columns)}")
        return pd.DataFrame()
    
    # Clean salary
    df["salary"] = (
        df["salary"].astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
    )
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    
    # Clean ownership
    if "own" in df.columns:
        df["own"] = (
            df["own"].astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df["own"] = pd.to_numeric(df["own"], errors="coerce")
    else:
        df["own"] = 15.0  # Default
    
    # Clean numeric columns
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce")
    
    # Drop invalid rows
    df = df.dropna(subset=["salary", "proj"])
    df = df[df["salary"] > 0]
    df = df[df["proj"] > 0]
    
    # Create unique ID
    df["player_id"] = (
        df["name"].astype(str) + "_" +
        df["team"].fillna("").astype(str) + "_" +
        df["salary"].astype(str)
    )
    
    # Ensure team/opp exist
    if "team" not in df.columns:
        df["team"] = ""
    if "opp" not in df.columns:
        df["opp"] = ""
    
    # Create game ID
    df["game_id"] = df.apply(
        lambda row: "@".join(sorted([str(row["team"]), str(row["opp"])])) 
        if row["team"] and row["opp"] else "",
        axis=1
    )
    
    return df


def calculate_metrics(df: pd.DataFrame, sport: Sport) -> pd.DataFrame:
    """Calculate advanced metrics."""
    df = df.copy()
    
    # Value (points per $1K)
    df["value"] = np.where(
        df["salary"] > 0,
        (df["proj"] / (df["salary"] / 1000)).round(2),
        0.0
    )
    
    # Leverage (expected optimal% - actual own%)
    df["expected_optimal"] = (df["value"] / 5.0 * 100).clip(0, 100)
    df["leverage"] = (df["expected_optimal"] - df["own"]).round(1)
    
    # Ceiling (sport-specific multipliers)
    if sport == Sport.NFL:
        # QB/RB have higher variance
        df["ceiling_mult"] = df["positions"].apply(
            lambda x: 1.6 if "QB" in str(x) else 1.5 if "RB" in str(x) else 1.4
        )
    else:
        df["ceiling_mult"] = 1.4
    
    df["ceiling"] = (df["proj"] * df["ceiling_mult"]).round(1)
    
    # Floor (70% of projection)
    df["floor"] = (df["proj"] * 0.7).round(1)
    
    # Ownership tier
    def own_tier(own):
        if pd.isna(own):
            return "mid"
        if own >= 40:
            return "mega-chalk"
        if own >= 30:
            return "chalk"
        if own >= 15:
            return "mid"
        if own >= 8:
            return "contrarian"
        return "super-contrarian"
    
    df["own_tier"] = df["own"].apply(own_tier)
    
    # GPP score (weighted composite for tournament play)
    df["gpp_score"] = (
        df["ceiling"] * 0.4 +        # Ceiling matters most
        df["value"] * 10 * 0.3 +      # Value is important
        df["leverage"] * 0.2 +        # Leverage bonus
        (100 - df["own"]) * 0.1       # Ownership fade
    ).round(1)
    
    return df


# -------------------------------------------------------------------
# STACKING LOGIC (SPORT-SPECIFIC)
# -------------------------------------------------------------------

def build_nba_stacks(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Build NBA game and team stacks."""
    stacks = defaultdict(list)
    
    # Game stacks (multiple players from high-scoring games)
    for game_id, game_df in df.groupby("game_id"):
        if not game_id or len(game_df) < 2:
            continue
        
        total_proj = game_df["proj"].sum()
        avg_ceiling = game_df["ceiling"].mean()
        
        if total_proj < 200:  # Skip low-scoring games
            continue
        
        # Find stack combinations
        for i, p1 in game_df.iterrows():
            for j, p2 in game_df.iterrows():
                if i >= j:
                    continue
                
                # Prefer different teams (bring-back)
                same_team = p1["team"] == p2["team"]
                
                stack_score = (
                    (p1["ceiling"] + p2["ceiling"]) * 0.4 +
                    (p1["value"] + p2["value"]) * 5 * 0.3 +
                    (200 - p1["own"] - p2["own"]) * 0.2 +
                    (0 if same_team else 20)  # Bonus for bring-back
                )
                
                stacks[game_id].append({
                    "type": "game_stack",
                    "players": [p1["player_id"], p2["player_id"]],
                    "score": stack_score,
                    "game_id": game_id
                })
    
    # Sort each game's stacks
    for game_id in stacks:
        stacks[game_id] = sorted(stacks[game_id], key=lambda x: x["score"], reverse=True)[:10]
    
    return dict(stacks)


def build_nfl_stacks(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Build NFL QB stacks and bring-backs."""
    stacks = defaultdict(list)
    
    # QB + Pass catcher stacks
    qbs = df[df["positions"].str.contains("QB", case=False, na=False)]
    
    for _, qb in qbs.iterrows():
        team = qb["team"]
        
        # Find pass catchers on same team
        pass_catchers = df[
            (df["team"] == team) &
            (df["positions"].str.contains("WR|TE", case=False, na=False))
        ]
        
        for _, pc in pass_catchers.iterrows():
            # Find bring-back from opposing team
            opp_team = qb["opp"]
            bring_backs = df[
                (df["team"] == opp_team) &
                (df["positions"].str.contains("QB|WR|TE", case=False, na=False))
            ].nlargest(3, "ceiling")
            
            for _, bb in bring_backs.iterrows():
                stack_score = (
                    (qb["ceiling"] + pc["ceiling"] + bb["ceiling"]) * 0.4 +
                    (qb["value"] + pc["value"] + bb["value"]) * 5 * 0.3 +
                    (300 - qb["own"] - pc["own"] - bb["own"]) * 0.3
                )
                
                stacks[team].append({
                    "type": "qb_stack",
                    "players": [qb["player_id"], pc["player_id"], bb["player_id"]],
                    "score": stack_score,
                    "game_id": qb["game_id"]
                })
    
    # Sort
    for team in stacks:
        stacks[team] = sorted(stacks[team], key=lambda x: x["score"], reverse=True)[:10]
    
    return dict(stacks)


def build_stacks(df: pd.DataFrame, sport: Sport) -> Dict[str, List[Dict]]:
    """Build sport-specific stacks."""
    if sport == Sport.NBA:
        return build_nba_stacks(df)
    elif sport == Sport.NFL:
        return build_nfl_stacks(df)
    else:
        return {}


# -------------------------------------------------------------------
# LINEUP VALIDATION
# -------------------------------------------------------------------

def can_fill_position(pos_string: str, slot: str, sport: Sport) -> bool:
    """Check if player can fill a roster slot."""
    if pd.isna(pos_string):
        return False
    
    positions = [p.strip().upper() for p in str(pos_string).split("/")]
    
    if sport == Sport.NBA:
        if slot in ["PG", "SG", "SF", "PF", "C"]:
            return slot in positions
        if slot == "G":
            return "PG" in positions or "SG" in positions
        if slot == "F":
            return "SF" in positions or "PF" in positions
        return True  # UTIL
    
    elif sport == Sport.NFL:
        if slot in ["QB", "RB", "WR", "TE"]:
            return slot in positions
        if slot == "DST":
            return "DST" in positions or "DEF" in positions
        if slot == "FLEX":
            return any(p in positions for p in ["RB", "WR", "TE"])
        return False
    
    elif sport == Sport.MLB:
        if slot == "P":
            return "P" in positions or "SP" in positions or "RP" in positions
        return slot in positions
    
    elif sport == Sport.NHL:
        if slot == "W":
            return "W" in positions or "LW" in positions or "RW" in positions
        if slot == "UTIL":
            return True
        return slot in positions
    
    return False


def validate_position_requirements(lineup_df: pd.DataFrame, sport: Sport) -> bool:
    """Validate lineup meets position requirements."""
    config = SPORT_CONFIGS.get(sport)
    if not config:
        return False
    
    position_counts = defaultdict(int)
    
    for _, player in lineup_df.iterrows():
        pos_string = str(player.get("positions", "")).upper()
        positions = [p.strip() for p in pos_string.split("/")]
        
        # Count each position
        for pos in positions:
            position_counts[pos] += 1
    
    # Check requirements based on sport
    if sport == Sport.NBA:
        return all([
            position_counts.get("PG", 0) >= 1,
            position_counts.get("SG", 0) >= 1,
            position_counts.get("SF", 0) >= 1,
            position_counts.get("PF", 0) >= 1,
            position_counts.get("C", 0) >= 1,
        ])
    
    elif sport == Sport.NFL:
        return all([
            position_counts.get("QB", 0) >= 1,
            position_counts.get("RB", 0) >= 2,
            position_counts.get("WR", 0) >= 3,
            position_counts.get("TE", 0) >= 1,
            position_counts.get("DST", 0) + position_counts.get("DEF", 0) >= 1,
        ])
    
    return True


# -------------------------------------------------------------------
# LINEUP GENERATION (CONTRARIAN FOCUS)
# -------------------------------------------------------------------

def generate_contrarian_lineup(
    pool: pd.DataFrame,
    sport: Sport,
    config: SportConfig,
    locks: List[str],
    excludes: List[str],
    stacks: Dict,
    correlation_strength: float,
    rng: np.random.Generator
) -> Optional[Dict[str, Any]]:
    """
    Generate single contrarian lineup.
    
    Strategy:
    1. Start with locks
    2. Apply stacking if correlation > 0.5
    3. Fill with contrarian + value plays
    4. Avoid chalk traps
    """
    
    # Start with locks
    selected = list(locks)
    selected_df = pool[pool["player_id"].isin(selected)]
    
    used_salary = int(selected_df["salary"].sum()) if not selected_df.empty else 0
    remaining_cap = config.salary_cap - used_salary
    remaining_spots = config.roster_size - len(selected)
    
    if remaining_spots < 0 or used_salary > config.salary_cap:
        return None
    
    # Available pool
    available = pool[
        (~pool["player_id"].isin(selected)) &
        (~pool["player_id"].isin(excludes))
    ].copy()
    
    # Apply stacking if enabled
    if correlation_strength > 0.5 and stacks and remaining_spots >= 2:
        # Pick a random stack
        all_stacks = []
        for stack_list in stacks.values():
            all_stacks.extend(stack_list)
        
        if all_stacks:
            # Weight by score
            stack_scores = np.array([s["score"] for s in all_stacks])
            stack_scores = np.maximum(stack_scores, 1)  # Avoid zeros
            probs = stack_scores / stack_scores.sum()
            
            chosen_stack = rng.choice(all_stacks, p=probs)
            
            # Add stack players
            for pid in chosen_stack["players"]:
                if pid not in selected and remaining_spots > 0:
                    player = available[available["player_id"] == pid]
                    if not player.empty:
                        sal = int(player.iloc[0]["salary"])
                        if sal <= remaining_cap:
                            selected.append(pid)
                            remaining_cap -= sal
                            remaining_spots -= 1
    
    # Fill remaining with contrarian + value
    available = pool[
        (~pool["player_id"].isin(selected)) &
        (~pool["player_id"].isin(excludes))
    ].copy()
    
    if remaining_spots > 0 and not available.empty:
        # Score players: favor low ownership + high value
        available["selection_score"] = (
            available["gpp_score"] * 0.6 +
            (100 - available["own"]) * 0.4 +  # Contrarian bonus
            rng.normal(0, 10, len(available))  # Randomness
        )
        
        # Avoid mega-chalk
        available = available[available["own"] < 40]
        
        if available.empty:
            # If too restrictive, allow chalk but penalize
            available = pool[
                (~pool["player_id"].isin(selected)) &
                (~pool["player_id"].isin(excludes))
            ].copy()
            available["selection_score"] = available["gpp_score"]
        
        available = available.sort_values("selection_score", ascending=False)
        
        # Greedy fill
        for _, player in available.iterrows():
            if remaining_spots == 0:
                break
            if player["salary"] <= remaining_cap:
                selected.append(player["player_id"])
                remaining_cap -= int(player["salary"])
                remaining_spots -= 1
    
    if remaining_spots > 0:
        return None
    
    # Validate
    lineup_df = pool[pool["player_id"].isin(selected)]
    
    if not validate_position_requirements(lineup_df, sport):
        return None
    
    # Calculate metrics
    total_proj = float(lineup_df["proj"].sum())
    total_salary = int(lineup_df["salary"].sum())
    total_own = float(lineup_df["own"].sum())
    avg_leverage = float(lineup_df["leverage"].mean())
    total_ceiling = float(lineup_df["ceiling"].sum())
    
    return {
        "player_ids": selected,
        "proj": total_proj,
        "salary": total_salary,
        "own": total_own,
        "leverage": avg_leverage,
        "ceiling": total_ceiling,
        "score": total_proj + avg_leverage * 0.5 + (config.roster_size * 100 - total_own) * 0.3
    }


def generate_lineup_set(
    pool: pd.DataFrame,
    sport: Sport,
    n_lineups: int,
    locks: List[str],
    excludes: List[str],
    correlation_strength: float,
    max_attempts: int = 10000
) -> List[Dict[str, Any]]:
    """Generate multiple contrarian lineups."""
    
    config = SPORT_CONFIGS[sport]
    stacks = build_stacks(pool, sport)
    
    rng = np.random.default_rng(42)
    
    lineups = []
    seen = set()
    
    for _ in range(max_attempts):
        if len(lineups) >= n_lineups:
            break
        
        lineup = generate_contrarian_lineup(
            pool, sport, config, locks, excludes,
            stacks, correlation_strength, rng
        )
        
        if not lineup:
            continue
        
        key = tuple(sorted(lineup["player_ids"]))
        if key in seen:
            continue
        seen.add(key)
        
        lineups.append(lineup)
    
    # Sort by score
    lineups = sorted(lineups, key=lambda x: x["score"], reverse=True)
    return lineups


# -------------------------------------------------------------------
# POSITION ASSIGNMENT
# -------------------------------------------------------------------

def assign_positions(lineup_df: pd.DataFrame, sport: Sport) -> pd.DataFrame:
    """Assign optimal positions to lineup."""
    config = SPORT_CONFIGS[sport]
    
    assigned = set()
    assignments = {}
    
    # Get slot order (specific first, flex last)
    slots = list(config.primary_positions) + config.flex_positions
    
    for slot in slots:
        available = lineup_df[~lineup_df["player_id"].isin(assigned)]
        eligible = available[
            available["positions"].apply(lambda x: can_fill_position(x, slot, sport))
        ]
        
        if eligible.empty:
            continue
        
        # Pick least flexible player for specific positions
        if slot not in config.flex_positions:
            eligible = eligible.copy()
            eligible["flexibility"] = eligible["positions"].apply(
                lambda x: len(str(x).split("/"))
            )
            eligible = eligible.sort_values("flexibility")
        
        chosen = eligible.iloc[0]
        assignments[chosen["player_id"]] = slot
        assigned.add(chosen["player_id"])
    
    result = lineup_df.copy()
    result["slot"] = result["player_id"].map(assignments)
    return result


# -------------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------------

def init_session_state():
    """Initialize session state."""
    if "pool" not in st.session_state:
        st.session_state.pool = pd.DataFrame()
    if "sport" not in st.session_state:
        st.session_state.sport = Sport.UNKNOWN
    if "lineups" not in st.session_state:
        st.session_state.lineups = []


def main():
    init_session_state()
    
    st.title("🏆 Elite Multi-Sport DFS Lineup Builder")
    st.caption("Advanced contrarian lineup generation for NBA, NFL, MLB, NHL")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        uploaded_file = st.file_uploader(
            "Upload Player Pool CSV",
            type=["csv"],
            help="DraftKings format: Player, Salary, Position, Team, Opponent, Projection, Ownership"
        )
        
        if uploaded_file:
            with st.spinner("Loading data..."):
                raw_df = pd.read_csv(uploaded_file)
                pool = normalize_csv_data(raw_df)
                
                if not pool.empty:
                    sport = detect_sport_from_positions(pool)
                    
                    if sport == Sport.UNKNOWN:
                        st.error("❌ Could not detect sport from positions")
                        st.stop()
                    
                    pool = calculate_metrics(pool, sport)
                    
                    st.session_state.pool = pool
                    st.session_state.sport = sport
                    
                    config = SPORT_CONFIGS[sport]
                    sport_icon = {"NBA": "🏀", "NFL": "🏈", "MLB": "⚾", "NHL": "🏒"}.get(sport.value, "🎯")
                    
                    st.success(f"{sport_icon} Detected: **{sport.value}**")
                    st.metric("Players Loaded", len(pool))
        
        if st.session_state.sport != Sport.UNKNOWN:
            st.markdown("---")
            
            contest_type = st.selectbox(
                "Contest Type",
                ["GPP - Large Field", "GPP - Mid Field", "Single Entry", "Cash Game"],
                index=0
            )
            
            # Auto-set correlation based on contest
            default_corr = {
                "GPP - Large Field": 0.8,
                "GPP - Mid Field": 0.6,
                "Single Entry": 0.5,
                "Cash Game": 0.2
            }[contest_type]
            
            correlation = st.slider(
                "Correlation Strength",
                0.0, 1.0, default_corr, 0.05,
                help="Higher = more stacking and contrarian plays"
            )
            
            n_lineups = st.number_input(
                "Number of Lineups",
                min_value=1,
                max_value=150,
                value=20 if "Large Field" in contest_type else 1,
                step=1
            )
            
            st.markdown("---")
            st.caption("🎯 Strategy: Contrarian + Leverage")
            st.caption(f"📊 Ownership Target: <30%")
            st.caption(f"🔥 Focus: Ceiling > Floor")
    
    # Main content
    pool = st.session_state.pool
    sport = st.session_state.sport
    
    if pool.empty:
        st.info("👆 Upload a player pool CSV to begin")
        st.markdown("""
        ### Required Columns:
        - **Player** or Name
        - **Salary**
        - **Position** (PG/SG/SF/PF/C for NBA, QB/RB/WR/TE/DST for NFL)
        - **Team**
        - **Opponent**
        - **Projection** or Proj
        - **Ownership** or Own% (optional, will default to 15%)
        """)
        st.stop()
    
    # Display pool
    st.subheader(f"{SPORT_CONFIGS[sport].name} Player Pool")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        own_filters = st.multiselect(
            "Ownership Tier",
            ["super-contrarian", "contrarian", "mid", "chalk", "mega-chalk"],
            default=["super-contrarian", "contrarian", "mid"]
        )
    
    with col2:
        if pool["team"].notna().any():
            teams = sorted(pool["team"].dropna().unique())
            selected_teams = st.multiselect(
                "Teams",
                teams,
                default=teams
            )
        else:
            selected_teams = []
    
    with col3:
        min_value = st.slider(
            "Min Value",
            float(pool["value"].min()),
            float(pool["value"].max()),
            float(pool["value"].min()),
            0.1
        )
    
    # Filter pool
    filtered = pool[
        (pool["own_tier"].isin(own_filters)) &
        (pool["team"].isin(selected_teams) if selected_teams else True) &
        (pool["value"] >= min_value)
    ].copy()
    
    # Add lock/exclude columns
    if "lock" not in filtered.columns:
        filtered["lock"] = False
    if "exclude" not in filtered.columns:
        filtered["exclude"] = False
    
    # Display
    display_cols = [
        "lock", "exclude", "name", "positions", "team", "opp",
        "salary", "proj", "value", "own", "leverage", "ceiling", "own_tier", "gpp_score"
    ]
    
    edited = st.data_editor(
        filtered[display_cols],
        column_config={
            "lock": st.column_config.CheckboxColumn("🔒"),
            "exclude": st.column_config.CheckboxColumn("❌"),
            "name": st.column_config.TextColumn("Player", width="medium"),
            "positions": st.column_config.TextColumn("Pos", width="small"),
            "salary": st.column_config.NumberColumn("Salary", format="$%d"),
            "proj": st.column_config.NumberColumn("Proj", format="%.1f"),
            "value": st.column_config.NumberColumn("Value", format="%.2f"),
            "own": st.column_config.NumberColumn("Own%", format="%.1f%%"),
            "leverage": st.column_config.NumberColumn("Lev", format="%+.1f"),
            "ceiling": st.column_config.NumberColumn("Ceil", format="%.1f"),
            "own_tier": st.column_config.TextColumn("Tier", width="small"),
            "gpp_score": st.column_config.NumberColumn("GPP Score", format="%.1f"),
        },
        hide_index=True,
        use_container_width=True,
        height=400
    )
    
    # Get locks and excludes
    locks = edited[edited["lock"] == True]["player_id"].tolist() if "player_id" in edited.columns else []
    excludes = edited[edited["exclude"] == True]["player_id"].tolist() if "player_id" in edited.columns else []
    
    if locks or excludes:
        st.caption(f"🔒 Locked: {len(locks)}  •  ❌ Excluded: {len(excludes)}")
    
    st.markdown("---")
    
    # Generate button
    if st.button("🚀 Generate Lineups", type="primary", use_container_width=True):
        with st.spinner(f"Building {n_lineups} contrarian lineups..."):
            lineups = generate_lineup_set(
                pool=pool,
                sport=sport,
                n_lineups=int(n_lineups),
                locks=locks,
                excludes=excludes,
                correlation_strength=correlation
            )
        
        if not lineups:
            st.error("❌ Could not generate lineups. Try adjusting locks/excludes or correlation.")
        else:
            st.session_state.lineups = lineups
            st.success(f"✅ Generated {len(lineups)} lineups")
    
    # Display lineups
    if st.session_state.lineups:
        st.markdown("---")
        st.subheader("💎 Generated Lineups")
        
        lineups = st.session_state.lineups
        
        # Summary
        summary_data = []
        for i, lu in enumerate(lineups, 1):
            summary_data.append({
                "Lineup": i,
                "Proj": lu["proj"],
                "Ceil": lu["ceiling"],
                "Own%": lu["own"],
                "Lev": lu["leverage"],
                "Salary": lu["salary"]
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Projection", f"{summary_df['Proj'].mean():.1f}")
        with col2:
            st.metric("Avg Ownership", f"{summary_df['Own%'].mean():.1f}%")
        with col3:
            st.metric("Avg Leverage", f"{summary_df['Lev'].mean():.1f}")
        
        st.dataframe(
            summary_df.style.format({
                "Proj": "{:.1f}",
                "Ceil": "{:.1f}",
                "Own%": "{:.1f}",
                "Lev": "{:+.1f}",
                "Salary": "${:,}"
            }),
            use_container_width=True
        )
        
        # Detail view
        st.markdown("---")
        st.subheader("Lineup Detail")
        
        options = [f"Lineup {i} (Proj {lu['proj']:.1f}, Own {lu['own']:.1f}%)" 
                   for i, lu in enumerate(lineups, 1)]
        choice = st.selectbox("Select lineup:", options)
        idx = options.index(choice)
        
        chosen = lineups[idx]
        lineup_df = pool[pool["player_id"].isin(chosen["player_ids"])]
        lineup_df = assign_positions(lineup_df, sport)
        
        st.dataframe(
            lineup_df[[
                "slot", "name", "positions", "team", "opp",
                "salary", "proj", "own", "leverage", "ceiling", "own_tier"
            ]].style.format({
                "salary": "${:,}",
                "proj": "{:.1f}",
                "own": "{:.1f}%",
                "leverage": "{:+.1f}",
                "ceiling": "{:.1f}"
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Export
        st.markdown("---")
        if st.button("📥 Export All Lineups to CSV"):
            # Create export format
            export_rows = []
            for lu in lineups:
                lineup_df = pool[pool["player_id"].isin(lu["player_ids"])]
                lineup_df = assign_positions(lineup_df, sport)
                
                row = {}
                for _, player in lineup_df.iterrows():
                    slot = player["slot"]
                    row[f"{slot}"] = player["name"]
                    row[f"{slot}_ID"] = player["player_id"]
                
                row["Projection"] = lu["proj"]
                row["Salary"] = lu["salary"]
                row["Ownership"] = lu["own"]
                export_rows.append(row)
            
            export_df = pd.DataFrame(export_rows)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                "Download CSV",
                csv,
                "lineups_export.csv",
                "text/csv",
                key="download"
            )


if __name__ == "__main__":
    main()
