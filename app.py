"""
Advanced DFS Lineup Builder with Correlation & NFL Support
Includes: Game stacks, team stacks, bring-backs, and sport-specific logic
"""

import io
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import streamlit as st

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Advanced DFS Lineup Builder")

DEFAULT_SALARY_CAP = 50000
DEFAULT_ROSTER_SIZE = 8  # NBA default

# -------------------------------------------------------------------
# SPORT DETECTION & POSITION RULES
# -------------------------------------------------------------------

def detect_sport(df: pd.DataFrame) -> str:
    """Detect sport from position data."""
    if df.empty or "positions" not in df.columns:
        return "NBA"
    
    all_positions = set()
    for val in df["positions"]:
        if isinstance(val, str):
            for p in val.replace(" ", "").split("/"):
                all_positions.add(p.upper())
    
    nfl_positions = {"QB", "RB", "WR", "TE", "DST", "DEF", "FLEX"}
    nba_positions = {"PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"}
    
    if all_positions & nfl_positions:
        return "NFL"
    if all_positions & nba_positions:
        return "NBA"
    return "NBA"


def get_position_rules(sport: str) -> Dict[str, Any]:
    """Get position requirements and roster size for sport."""
    if sport == "NFL":
        return {
            "roster_size": 9,
            "positions": {
                "QB": 1,
                "RB": 2,
                "WR": 3,
                "TE": 1,
                "FLEX": 1,  # RB/WR/TE
                "DST": 1
            },
            "flex_positions": ["RB", "WR", "TE"],
            "salary_cap": 50000
        }
    else:  # NBA
        return {
            "roster_size": 8,
            "positions": {
                "PG": 1,
                "SG": 1,
                "SF": 1,
                "PF": 1,
                "C": 1,
                "G": 1,   # PG/SG
                "F": 1,   # SF/PF
                "UTIL": 1  # Any
            },
            "salary_cap": 50000
        }

# -------------------------------------------------------------------
# OWNERSHIP & METRICS
# -------------------------------------------------------------------

def ownership_bucket(own: float) -> str:
    """Map ownership to bucket."""
    if pd.isna(own):
        return "mid"
    if own >= 40:
        return "mega"
    if own >= 30:
        return "chalk"
    if own >= 10:
        return "mid"
    return "punt"


def calculate_leverage(row: pd.Series) -> float:
    """Calculate leverage score."""
    expected_optimal = (row["value"] / 5.0) * 100
    expected_optimal = min(expected_optimal, 100)
    leverage = expected_optimal - row["own_proj"]
    return round(leverage, 1)


def calculate_ceiling(row: pd.Series, sport: str) -> float:
    """Calculate ceiling projection."""
    if sport == "NFL":
        # NFL has more variance, especially for QBs
        if "QB" in str(row.get("positions", "")):
            mult = 1.5
        elif "RB" in str(row.get("positions", "")):
            mult = 1.4
        else:
            mult = 1.35
    else:
        mult = 1.35
    
    return round(row["proj"] * mult, 1)

# -------------------------------------------------------------------
# CSV NORMALIZATION
# -------------------------------------------------------------------

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize CSV into internal schema."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    
    rename_map = {
        "player": "Name",
        "name": "Name",
        "salary": "Salary",
        "position": "positions",
        "pos": "positions",
        "team": "Team",
        "tm": "Team",
        "opponent": "Opponent",
        "opp": "Opponent",
        "projection": "proj",
        "proj": "proj",
        "fpts": "proj",
        "points": "proj",
        "value": "Value",
        "ownership": "own_proj",
        "ownership%": "own_proj",
        "ownership %": "own_proj",
        "own": "own_proj",
        "own%": "own_proj",
    }
    df = df.rename(columns=rename_map)
    
    required = ["Name", "Team", "positions", "Salary", "proj"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        return pd.DataFrame()
    
    # Clean Salary
    if "Salary" in df.columns:
        df["Salary"] = (
            df["Salary"].astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .str.strip()
        )
    
    # Clean ownership
    if "own_proj" in df.columns:
        df["own_proj"] = (
            df["own_proj"].astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
    else:
        df["own_proj"] = 15.0
    
    # Create player_id
    df["player_id"] = (
        df["Name"].astype(str) + "_" + 
        df["Team"].astype(str) + "_" + 
        df["Salary"].astype(str)
    )
    
    # Numeric conversions
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce")
    df["own_proj"] = pd.to_numeric(df["own_proj"], errors="coerce").fillna(15.0)
    
    df = df.dropna(subset=["Salary", "proj"])
    
    # Create GameID
    if "Opponent" in df.columns:
        df["GameID"] = df.apply(
            lambda row: "@".join(sorted([str(row["Team"]), str(row["Opponent"])])),
            axis=1
        )
    else:
        df["GameID"] = ""
    
    return df


def add_metrics(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Add calculated metrics."""
    df["value"] = np.where(
        df["Salary"] > 0,
        (df["proj"] / (df["Salary"] / 1000)).round(2),
        0.0
    )
    
    df["own_bucket"] = df["own_proj"].apply(ownership_bucket)
    df["leverage"] = df.apply(calculate_leverage, axis=1)
    df["ceiling"] = df.apply(lambda row: calculate_ceiling(row, sport), axis=1)
    
    return df

# -------------------------------------------------------------------
# GAME ENVIRONMENTS
# -------------------------------------------------------------------

def build_game_environments(df: pd.DataFrame, sport: str) -> Dict[str, Dict[str, Any]]:
    """Build game environment dictionary."""
    if "Team" not in df.columns or "Opponent" not in df.columns:
        return {}
    
    game_envs = {}
    
    for game_id, grp in df.groupby("GameID"):
        if not game_id:
            continue
        
        teams = sorted(grp["Team"].unique())
        if len(teams) < 2:
            continue
        
        team1, team2 = teams[0], teams[1]
        
        team1_proj = grp[grp["Team"] == team1]["proj"].sum()
        team2_proj = grp[grp["Team"] == team2]["proj"].sum()
        total_proj = team1_proj + team2_proj
        
        # Quality score for game
        avg_ceiling = grp["ceiling"].mean()
        quality_score = (total_proj * 0.7) + (avg_ceiling * 0.3)
        
        game_envs[game_id] = {
            "game_id": game_id,
            "teams": [team1, team2],
            "total_proj": float(total_proj),
            "avg_ceiling": float(avg_ceiling),
            "quality_score": float(quality_score),
            "team1": team1,
            "team2": team2,
            "team1_proj": float(team1_proj),
            "team2_proj": float(team2_proj),
        }
    
    return game_envs

# -------------------------------------------------------------------
# NBA STACKING LOGIC
# -------------------------------------------------------------------

def build_nba_team_stacks(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Build NBA team stacks (star + value combos)."""
    team_stacks = {}
    
    for team in df["Team"].unique():
        team_players = df[df["Team"] == team].copy()
        
        if len(team_players) < 2:
            continue
        
        # Stars (top 30% salary)
        salary_threshold = team_players["Salary"].quantile(0.7)
        stars = team_players[team_players["Salary"] >= salary_threshold]
        secondary = team_players[team_players["Salary"] < salary_threshold]
        
        stacks = []
        for _, star in stars.iterrows():
            for _, sec in secondary.iterrows():
                combined_own = star["own_proj"] + sec["own_proj"]
                combined_ceiling = star["ceiling"] + sec["ceiling"]
                combined_proj = star["proj"] + sec["proj"]
                
                # Stack score
                stack_score = (
                    combined_ceiling * 0.4 +
                    combined_proj * 0.3 +
                    (100 - combined_own) * 0.3
                )
                
                stacks.append({
                    "team": team,
                    "primary": star["player_id"],
                    "secondary": sec["player_id"],
                    "primary_name": star["Name"],
                    "secondary_name": sec["Name"],
                    "combined_own": combined_own,
                    "combined_ceiling": combined_ceiling,
                    "stack_score": stack_score
                })
        
        if stacks:
            team_stacks[team] = sorted(stacks, key=lambda x: x["stack_score"], reverse=True)[:15]
    
    return team_stacks


def calculate_nba_correlation_score(lineup_df: pd.DataFrame, game_envs: Dict) -> float:
    """Calculate NBA correlation score."""
    if lineup_df.empty:
        return 0.0
    
    score = 0.0
    
    # Game stack bonus
    game_counts = Counter(lineup_df["GameID"].tolist())
    for game_id, count in game_counts.items():
        if count > 1 and game_id in game_envs:
            quality = game_envs[game_id]["quality_score"]
            score += (count - 1) * quality * 0.1
    
    # Team stack bonus
    team_counts = Counter(lineup_df["Team"].tolist())
    for team, count in team_counts.items():
        if count > 1:
            score += min(count - 1, 3) * 15
    
    return float(score)

# -------------------------------------------------------------------
# NFL STACKING LOGIC
# -------------------------------------------------------------------

def build_nfl_qb_stacks(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Build NFL QB + pass catcher stacks."""
    qb_stacks = {}
    
    for team in df["Team"].unique():
        team_players = df[df["Team"] == team].copy()
        
        qbs = team_players[team_players["positions"].str.contains("QB", case=False, na=False)]
        pass_catchers = team_players[
            team_players["positions"].str.contains("WR|TE", case=False, na=False)
        ]
        
        if qbs.empty or pass_catchers.empty:
            continue
        
        stacks = []
        for _, qb in qbs.iterrows():
            for _, pc in pass_catchers.iterrows():
                combined_own = qb["own_proj"] + pc["own_proj"]
                combined_ceiling = qb["ceiling"] + pc["ceiling"]
                combined_proj = qb["proj"] + pc["proj"]
                
                stack_score = (
                    combined_ceiling * 0.4 +
                    combined_proj * 0.3 +
                    (100 - combined_own) * 0.3
                )
                
                stacks.append({
                    "team": team,
                    "qb": qb["player_id"],
                    "pass_catcher": pc["player_id"],
                    "qb_name": qb["Name"],
                    "pc_name": pc["Name"],
                    "combined_own": combined_own,
                    "combined_ceiling": combined_ceiling,
                    "stack_score": stack_score
                })
        
        if stacks:
            qb_stacks[team] = sorted(stacks, key=lambda x: x["stack_score"], reverse=True)[:10]
    
    return qb_stacks


def validate_nfl_lineup(lineup_df: pd.DataFrame) -> bool:
    """Check for bad NFL correlations."""
    # QB vs opposing DST
    qbs = lineup_df[lineup_df["positions"].str.contains("QB", case=False, na=False)]
    dsts = lineup_df[lineup_df["positions"].str.contains("DST|DEF", case=False, na=False)]
    
    for _, qb in qbs.iterrows():
        qb_opp = qb.get("Opponent", "")
        for _, dst in dsts.iterrows():
            dst_team = dst.get("Team", "")
            if qb_opp == dst_team:
                return False  # QB facing his own DST
    
    # WR/TE vs opposing DST
    pass_catchers = lineup_df[
        lineup_df["positions"].str.contains("WR|TE", case=False, na=False)
    ]
    
    for _, pc in pass_catchers.iterrows():
        pc_opp = pc.get("Opponent", "")
        for _, dst in dsts.iterrows():
            dst_team = dst.get("Team", "")
            if pc_opp == dst_team:
                return False
    
    return True


def calculate_nfl_correlation_score(lineup_df: pd.DataFrame, game_envs: Dict) -> float:
    """Calculate NFL correlation score."""
    if lineup_df.empty:
        return 0.0
    
    score = 0.0
    
    # QB + pass catcher same team (primary stack)
    qbs = lineup_df[lineup_df["positions"].str.contains("QB", case=False, na=False)]
    pass_catchers = lineup_df[
        lineup_df["positions"].str.contains("WR|TE", case=False, na=False)
    ]
    
    for _, qb in qbs.iterrows():
        qb_team = qb["Team"]
        same_team_pc = pass_catchers[pass_catchers["Team"] == qb_team]
        
        if len(same_team_pc) >= 1:
            score += 30
        if len(same_team_pc) >= 2:
            score += 20
        if len(same_team_pc) >= 3:
            score += 15
    
    # RB + DST same team
    rbs = lineup_df[lineup_df["positions"].str.contains("RB", case=False, na=False)]
    dsts = lineup_df[lineup_df["positions"].str.contains("DST|DEF", case=False, na=False)]
    
    for _, rb in rbs.iterrows():
        rb_team = rb["Team"]
        same_team_dst = dsts[dsts["Team"] == rb_team]
        if len(same_team_dst) > 0:
            score += 25
    
    # Game stack bonus
    game_counts = Counter(lineup_df["GameID"].tolist())
    for game_id, count in game_counts.items():
        if count >= 3 and game_id in game_envs:
            quality = game_envs[game_id].get("quality_score", 0)
            score += (count - 2) * quality * 0.08
    
    # Bring-back bonus
    for game_id in lineup_df["GameID"].unique():
        game_players = lineup_df[lineup_df["GameID"] == game_id]
        teams_in_game = game_players["Team"].nunique()
        if teams_in_game == 2:
            score += 15
    
    return float(score)

# -------------------------------------------------------------------
# LINEUP GENERATION
# -------------------------------------------------------------------

def build_lineup_with_correlation(
    pool: pd.DataFrame,
    sport: str,
    roster_size: int,
    salary_cap: int,
    locked_ids: List[str],
    excluded_ids: List[str],
    correlation_strength: float,
    game_envs: Dict,
    team_stacks: Dict,
    rng
) -> Optional[Dict[str, Any]]:
    """Build single lineup with correlation logic."""
    
    # Filter pool
    available = pool[
        (~pool["player_id"].isin(excluded_ids))
    ].copy()
    
    # Start with locks
    current_ids = list(locked_ids)
    locks_df = available[available["player_id"].isin(current_ids)]
    lock_salary = int(locks_df["Salary"].sum()) if not locks_df.empty else 0
    
    if len(current_ids) > roster_size or lock_salary > salary_cap:
        return None
    
    remaining_spots = roster_size - len(current_ids)
    remaining_cap = salary_cap - lock_salary
    
    # Apply correlation if strength > 0.3
    if correlation_strength > 0.3 and team_stacks:
        # Try to add a stack
        if sport == "NBA":
            # Pick random team with stacks
            teams_with_stacks = list(team_stacks.keys())
            if teams_with_stacks:
                team = rng.choice(teams_with_stacks)
                stacks = team_stacks[team]
                if stacks:
                    stack = rng.choice(stacks[:5])  # Pick from top 5
                    
                    # Add stack players if not already in lineup
                    for pid in [stack["primary"], stack["secondary"]]:
                        if pid not in current_ids and remaining_spots > 0:
                            player = available[available["player_id"] == pid]
                            if not player.empty:
                                sal = int(player.iloc[0]["Salary"])
                                if sal <= remaining_cap:
                                    current_ids.append(pid)
                                    remaining_spots -= 1
                                    remaining_cap -= sal
        
        elif sport == "NFL":
            # Similar for NFL QB stacks
            teams_with_stacks = list(team_stacks.keys())
            if teams_with_stacks:
                team = rng.choice(teams_with_stacks)
                stacks = team_stacks[team]
                if stacks:
                    stack = rng.choice(stacks[:5])
                    
                    for pid in [stack["qb"], stack["pass_catcher"]]:
                        if pid not in current_ids and remaining_spots > 0:
                            player = available[available["player_id"] == pid]
                            if not player.empty:
                                sal = int(player.iloc[0]["Salary"])
                                if sal <= remaining_cap:
                                    current_ids.append(pid)
                                    remaining_spots -= 1
                                    remaining_cap -= sal
    
    # Fill remaining spots with value-based selection
    available_pool = available[~available["player_id"].isin(current_ids)].copy()
    
    if not available_pool.empty and remaining_spots > 0:
        scores = available_pool["value"].values + rng.normal(0, 0.2, len(available_pool))
        available_pool = available_pool.assign(_score=scores).sort_values("_score", ascending=False)
        
        for _, row in available_pool.iterrows():
            if remaining_spots == 0:
                break
            if row["Salary"] <= remaining_cap:
                current_ids.append(row["player_id"])
                remaining_spots -= 1
                remaining_cap -= int(row["Salary"])
    
    if remaining_spots != 0:
        return None
    
    # Validate lineup
    lineup_df = pool[pool["player_id"].isin(current_ids)]
    
    if sport == "NFL" and not validate_nfl_lineup(lineup_df):
        return None
    
    # Calculate scores
    proj_score = float(lineup_df["proj"].sum())
    salary_used = int(lineup_df["Salary"].sum())
    total_own = float(lineup_df["own_proj"].sum())
    
    if sport == "NBA":
        corr_score = calculate_nba_correlation_score(lineup_df, game_envs)
    else:
        corr_score = calculate_nfl_correlation_score(lineup_df, game_envs)
    
    # Composite score
    composite = (
        proj_score * (1.0 - correlation_strength * 0.3) +
        corr_score * 0.5 * correlation_strength
    )
    
    return {
        "player_ids": current_ids,
        "proj_score": proj_score,
        "salary_used": salary_used,
        "total_own": total_own,
        "correlation": corr_score,
        "composite": composite
    }


def generate_lineups_advanced(
    pool: pd.DataFrame,
    sport: str,
    n_lineups: int,
    roster_size: int,
    salary_cap: int,
    locked_ids: List[str],
    excluded_ids: List[str],
    correlation_strength: float,
    max_attempts: int = 5000
) -> List[Dict[str, Any]]:
    """Generate lineups with advanced correlation."""
    
    # Build environments and stacks
    game_envs = build_game_environments(pool, sport)
    
    if sport == "NBA":
        team_stacks = build_nba_team_stacks(pool)
    else:
        team_stacks = build_nfl_qb_stacks(pool)
    
    rng = np.random.default_rng(42)
    
    lineups = []
    seen_sets = set()
    
    for _ in range(max_attempts):
        if len(lineups) >= n_lineups:
            break
        
        lineup = build_lineup_with_correlation(
            pool=pool,
            sport=sport,
            roster_size=roster_size,
            salary_cap=salary_cap,
            locked_ids=locked_ids,
            excluded_ids=excluded_ids,
            correlation_strength=correlation_strength,
            game_envs=game_envs,
            team_stacks=team_stacks,
            rng=rng
        )
        
        if not lineup:
            continue
        
        key = tuple(sorted(lineup["player_ids"]))
        if key in seen_sets:
            continue
        seen_sets.add(key)
        
        lineups.append(lineup)
    
    # Sort by composite score
    lineups = sorted(lineups, key=lambda x: x["composite"], reverse=True)
    return lineups[:n_lineups]

# -------------------------------------------------------------------
# POSITION ASSIGNMENT
# -------------------------------------------------------------------

def assign_nba_positions(lineup_df: pd.DataFrame) -> pd.DataFrame:
    """Assign NBA positions."""
    assigned_players = set()
    slot_assignments = {}
    
    def can_play(pos_string, slot):
        if pd.isna(pos_string):
            return False
        positions = [p.strip() for p in str(pos_string).split("/")]
        if slot in ["PG", "SG", "SF", "PF", "C"]:
            return slot in positions
        if slot == "G":
            return "PG" in positions or "SG" in positions
        if slot == "F":
            return "SF" in positions or "PF" in positions
        return True  # UTIL
    
    slots = ["C", "PG", "SG", "SF", "PF", "G", "F", "UTIL"]
    
    for slot in slots:
        available = lineup_df[~lineup_df["player_id"].isin(assigned_players)]
        eligible = available[available["positions"].apply(lambda x: can_play(x, slot))]
        
        if eligible.empty:
            continue
        
        chosen = eligible.iloc[0]
        slot_assignments[slot] = chosen["player_id"]
        assigned_players.add(chosen["player_id"])
    
    result = lineup_df.copy()
    result["roster_slot"] = result["player_id"].map(
        {v: k for k, v in slot_assignments.items()}
    )
    return result


def assign_nfl_positions(lineup_df: pd.DataFrame) -> pd.DataFrame:
    """Assign NFL positions."""
    assigned_players = set()
    slot_assignments = {}
    
    def can_play(pos_string, slot):
        if pd.isna(pos_string):
            return False
        positions = [p.strip().upper() for p in str(pos_string).split("/")]
        if slot in ["QB", "RB", "WR", "TE", "DST"]:
            return slot in positions or (slot == "DST" and "DEF" in positions)
        if slot == "FLEX":
            return any(p in positions for p in ["RB", "WR", "TE"])
        return False
    
    slots = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
    
    for slot in slots:
        available = lineup_df[~lineup_df["player_id"].isin(assigned_players)]
        eligible = available[available["positions"].apply(lambda x: can_play(x, slot))]
        
        if eligible.empty:
            continue
        
        chosen = eligible.iloc[0]
        slot_assignments[chosen["player_id"]] = slot
        assigned_players.add(chosen["player_id"])
    
    result = lineup_df.copy()
    result["roster_slot"] = result["player_id"].map(slot_assignments)
    return result

# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------

def display_lineups(slate_df: pd.DataFrame, lineup_list: List[Dict[str, Any]], sport: str):
    """Display generated lineups."""
    if not lineup_list:
        st.error("❌ No lineups to display.")
        return
    
    best = lineup_list[0]
    
    st.subheader("Top Lineup Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Projected Points", f"{best['proj_score']:.2f}")
    with c2:
        st.metric("Salary Used", f"${best['salary_used']:,}")
    with c3:
        st.metric("Total Ownership", f"{best['total_own']:.1f}%")
    with c4:
        st.metric("Correlation Score", f"{best['correlation']:.1f}")
    
    st.markdown("---")
    st.subheader("Lineup Summary")
    
    rows = []
    for i, lu in enumerate(lineup_list, start=1):
        rows.append({
            "Lineup": i,
            "Proj": lu["proj_score"],
            "Corr": lu["correlation"],
            "Own%": lu["total_own"],
            "Salary": lu["salary_used"],
        })
    
    summary_df = pd.DataFrame(rows).set_index("Lineup")
    st.dataframe(
        summary_df.style.format({
            "Proj": "{:.2f}",
            "Corr": "{:.1f}",
            "Own%": "{:.1f}",
            "Salary": "${:,}"
        }),
        use_container_width=True,
    )
    
    st.markdown("---")
    st.subheader("Lineup Detail")
    
    options = [
        f"Lineup {i} (Proj {lu['proj_score']:.2f}, Corr {lu['correlation']:.1f})"
        for i, lu in enumerate(lineup_list, start=1)
    ]
    choice = st.selectbox("Choose lineup:", options)
    idx = options.index(choice)
    chosen = lineup_list[idx]
    chosen_df = slate_df[slate_df["player_id"].isin(chosen["player_ids"])].copy()
    
    if sport == "NBA":
        chosen_df = assign_nba_positions(chosen_df)
        roster_order = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    else:
        chosen_df = assign_nfl_positions(chosen_df)
        roster_order = ["QB", "RB", "WR", "TE", "FLEX", "DST"]
    
    cat_type = CategoricalDtype(roster_order, ordered=True)
    chosen_df["roster_slot"] = chosen_df["roster_slot"].astype(cat_type)
    chosen_df = chosen_df.sort_values("roster_slot")
    
    display_cols = [
        "roster_slot", "Name", "positions", "Team", "Opponent",
        "Salary", "proj", "value", "own_proj", "own_bucket"
    ]
    df_disp = chosen_df[display_cols].copy()
    df_disp.rename(columns={
        "roster_slot": "SLOT",
        "positions": "POS",
        "proj": "Proj",
        "own_proj": "Own%",
        "own_bucket": "Bucket",
    }, inplace=True)
    
    st.dataframe(
        df_disp.style.format({
            "Salary": "${:,}",
            "Proj": "{:.1f}",
            "value": "{:.2f}",
            "Own%": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True
    )


# -------------------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------------------

if "slate_df" not in st.session_state:
    st.session_state["slate_df"] = pd.DataFrame()
if "lineups" not in st.session_state:
    st.session_state["lineups"] = []
if "sport" not in st.session_state:
    st.session_state["sport"] = "NBA"

# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------

st.title("🏀🏈 Advanced DFS Lineup Builder")
st.caption("With Correlation Logic, Game Stacks & NFL Support")

# Sidebar
st.sidebar.title("Settings")

sport_icon = "🏀" if st.session_state.get("sport", "NBA") == "NBA" else "🏈"
st.sidebar.markdown(f"### {sport_icon} Detected Sport: **{st.session_state.get('sport', 'NBA')}**")

contest_type = st.sidebar.selectbox(
    "Contest Type",
    options=["Cash Game", "Single Entry", "3-Max", "20-Max", "150-Max"],
    index=1,
)

# Correlation strength
default_corr = {
    "Cash Game": 0.2,
    "Single Entry": 0.5,
    "3-Max": 0.6,
    "20-Max": 0.75,
    "150-Max": 0.85
}.get(contest_type, 0.5)

correlation_strength = st.sidebar.slider(
    "Correlation Strength",
    min_value=0.0,
    max_value=1.0,
    value=default_corr,
    step=0.05,
    help="0.0 = No correlation (value-based only)\n1.0 = Maximum stacking"
)

st.sidebar.caption(
    f"**Current:** {'Minimal' if correlation_strength < 0.3 else 'Moderate' if correlation_strength < 0.6 else 'High' if correlation_strength < 0.8 else 'Maximum'} correlation"
)

# Default n_lineups
default_n = {
    "Cash Game": 1,
    "Single Entry": 1,
    "3-Max": 3,
    "20-Max": 20,
    "150-Max": 40
}.get(contest_type, 10)

n_lineups = st.sidebar.slider(
    "Number of Lineups",
    min_value=1,
    max_value=50,
    value=default_n,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Load Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"],
    help="DraftKings format: Player, Salary, Position, Team, Opponent, Projection, Ownership"
)

if uploaded_file:
    with st.spinner("Loading data..."):
        raw_df = pd.read_csv(uploaded_file)
        slate_df = normalize_df(raw_df)
        
        if not slate_df.empty:
            sport = detect_sport(slate_df)
            slate_df = add_metrics(slate_df, sport)
            
            st.session_state["slate_df"] = slate_df
            st.session_state["sport"] = sport
            st.sidebar.success(f"✅ Loaded {len(slate_df)} players ({sport})")
        else:
            st.sidebar.error("❌ Failed to load data")

slate_df = st.session_state["slate_df"]
sport = st.session_state.get("sport", "NBA")

if slate_df.empty:
    st.info("👆 Upload a player pool CSV to begin")
    st.markdown("""
    ### Required CSV Columns:
    - **Player** or Name
    - **Salary**
    - **Position** (PG/SG/SF/PF/C for NBA, QB/RB/WR/TE/DST for NFL)
    - **Team**
    - **Opponent**
    - **Projection** or Proj
    - **Ownership** or Own%
    
    The app will auto-detect NBA vs NFL based on positions.
    """)
else:
    # Get position rules
    pos_rules = get_position_rules(sport)
    roster_size = pos_rules["roster_size"]
    salary_cap = pos_rules["salary_cap"]
    
    st.subheader(f"{sport_icon} Player Pool")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        teams = sorted(slate_df["Team"].unique())
        selected_teams = st.multiselect(
            "Filter by Team",
            options=teams,
            default=teams
        )
    
    with col2:
        buckets = ["mega", "chalk", "mid", "punt"]
        selected_buckets = st.multiselect(
            "Filter by Ownership Bucket",
            options=buckets,
            default=buckets
        )
    
    filtered_df = slate_df[
        (slate_df["Team"].isin(selected_teams)) &
        (slate_df["own_bucket"].isin(selected_buckets))
    ].copy()
    
    # Ensure columns
    if "Lock" not in filtered_df.columns:
        filtered_df["Lock"] = False
    if "Exclude" not in filtered_df.columns:
        filtered_df["Exclude"] = False
    if "player_id" not in filtered_df.columns:
        filtered_df["player_id"] = (
            filtered_df["Name"].astype(str) + "_" +
            filtered_df["Team"].astype(str) + "_" +
            filtered_df["Salary"].astype(str)
        )
    
    column_config = {
        "Lock": st.column_config.CheckboxColumn("🔒"),
        "Exclude": st.column_config.CheckboxColumn("❌"),
        "player_id": None,
        "Name": st.column_config.TextColumn("Player", disabled=True),
        "positions": st.column_config.TextColumn("Pos", disabled=True),
        "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
        "proj": st.column_config.NumberColumn("Proj", format="%.1f"),
        "value": st.column_config.NumberColumn("Value", format="%.2f"),
        "own_proj": st.column_config.NumberColumn("Own%", format="%.1f%%"),
        "leverage": st.column_config.NumberColumn("Lev", format="%+.1f"),
        "ceiling": st.column_config.NumberColumn("Ceil", format="%.1f"),
        "own_bucket": st.column_config.TextColumn("Bucket"),
    }
    
    column_order = [
        "Lock", "Exclude", "player_id", "Name", "positions", "Team", "Opponent",
        "Salary", "proj", "value", "own_proj", "leverage", "ceiling", "own_bucket"
    ]
    
    edited_df = st.data_editor(
        filtered_df[column_order],
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        height=400,
        key="player_editor"
    )
    
    # Update state safely
    if "player_id" in edited_df.columns:
        for idx, row in edited_df.iterrows():
            pid = row.get("player_id")
            if pid and pid in st.session_state["slate_df"]["player_id"].values:
                mask = st.session_state["slate_df"]["player_id"] == pid
                st.session_state["slate_df"].loc[mask, "Lock"] = row.get("Lock", False)
                st.session_state["slate_df"].loc[mask, "Exclude"] = row.get("Exclude", False)
    
    locked_ids = []
    excluded_ids = []
    
    if "Lock" in edited_df.columns and "player_id" in edited_df.columns:
        locked_ids = edited_df[edited_df["Lock"] == True]["player_id"].tolist()
    
    if "Exclude" in edited_df.columns and "player_id" in edited_df.columns:
        excluded_ids = edited_df[edited_df["Exclude"] == True]["player_id"].tolist()
    
    if locked_ids or excluded_ids:
        st.caption(f"🔒 Locked: {len(locked_ids)}  •  ❌ Excluded: {len(excluded_ids)}")
    
    st.markdown("---")
    
    if st.button("🚀 Generate Lineups", type="primary"):
        pool_for_build = st.session_state["slate_df"].copy()
        
        with st.spinner(f"Building {n_lineups} lineups with correlation..."):
            lineups = generate_lineups_advanced(
                pool=pool_for_build,
                sport=sport,
                n_lineups=n_lineups,
                roster_size=roster_size,
                salary_cap=salary_cap,
                locked_ids=locked_ids,
                excluded_ids=excluded_ids,
                correlation_strength=correlation_strength,
            )
        
        if not lineups:
            st.error(
                "❌ Could not generate valid lineups.\n\n"
                "Try:\n"
                "- Lowering correlation strength\n"
                "- Reducing locked players\n"
                "- Removing some excludes"
            )
        else:
            st.session_state["lineups"] = lineups
            st.success(f"✅ Generated {len(lineups)} lineups")
    
    if st.session_state["lineups"]:
        st.markdown("---")
        display_lineups(st.session_state["slate_df"], st.session_state["lineups"], sport)
