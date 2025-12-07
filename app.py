import io
from typing import Dict, Any, List
from collections import Counter

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import streamlit as st

# Import from builder
from builder import (
    ownership_bucket,
    build_game_environments,
    build_team_stacks,
    calculate_lineup_correlation_score,
)

# -------------------------------------------------------------------
# BASIC CONFIG / MULTI-SPORT RULES
# -------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="DFS Lineup Builder (NBA/NFL)")

# Sport-specific defaults
SPORT_CONFIGS = {
    "NBA": {
        "salary_cap": 50000,
        "roster_size": 8,
        "positions": ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"],
        "title": "NBA DFS Lineup Builder"
    },
    "NFL": {
        "salary_cap": 50000,
        "roster_size": 9,
        "positions": ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"],
        "title": "NFL DFS Lineup Builder"
    }
}

DEFAULT_SALARY_CAP = 50000
DEFAULT_ROSTER_SIZE = 8  # Will be overridden by sport detection

# -------------------------------------------------------------------
# CSV MAPPING / EDGE CALCS
# -------------------------------------------------------------------
REQUIRED_CSV_TO_INTERNAL_MAP = {
    "Player": "Name",
    "Salary": "salary",
    "Position": "positions",
    "Team": "Team",
    "Opponent": "Opponent",
    "PROJECTED FP": "proj",
    "Projection": "proj",
    "Proj": "proj",
    "OWNERSHIP %": "own_proj",
    "Ownership": "own_proj",
    "Own": "own_proj",
    "Own%": "own_proj",
    "Minutes": "Minutes",
    "FPPM": "FPPM",
    "Value": "Value",
}
CORE_INTERNAL_COLS = [
    "salary",
    "positions",
    "proj",
    "own_proj",
    "Name",
    "Team",
    "Opponent",
]


def detect_sport(df: pd.DataFrame) -> str:
    """
    Detect sport type from position data.
    Returns: "NBA", "NFL", or "UNKNOWN"
    """
    if df.empty or "positions" not in df.columns:
        return "NBA"  # Default
    
    positions_str = " ".join(df["positions"].astype(str).str.upper())
    
    # NFL positions
    nfl_positions = ["QB", "RB", "WR", "TE", "DST", "FLEX", "DEF"]
    nfl_count = sum(1 for pos in nfl_positions if pos in positions_str)
    
    # NBA positions  
    nba_positions = ["PG", "SG", "SF", "PF", "UTIL"]
    nba_count = sum(1 for pos in nba_positions if pos in positions_str)
    
    if nfl_count > nba_count:
        return "NFL"
    elif nba_count > 0:
        return "NBA"
    else:
        return "NBA"  # Default


def calculate_player_leverage(row):
    expected_optimal_pct = (row["value"] / 5.0) * 100
    expected_optimal_pct = min(expected_optimal_pct, 100)
    leverage = expected_optimal_pct - row["own_proj"]
    return round(leverage, 1)


def calculate_ceiling_score(row):
    base_ceiling = row["proj"] * 1.35
    salary_factor = (10000 - row["salary"]) / 10000
    ceiling_boost = base_ceiling * salary_factor * 0.2
    return round(base_ceiling + ceiling_boost, 1)


def assign_edge_category(row):
    """
    Assign edge category based on ownership bucket AND leverage.
    True edge = low ownership + positive leverage OR high ceiling + value.
    """
    own = row["own_proj"]
    leverage = row["leverage_score"]
    value = row["value"]
    
    # Ownership buckets (matches builder.py)
    PUNT_THR = 10.0
    CHALK_THR = 30.0
    MEGA_CHALK_THR = 40.0
    
    # Mega Chalk (40%+ owned) - rarely has edge
    if own >= MEGA_CHALK_THR:
        if leverage > 20 and value > 5.0:
            return "🔥 Mega Chalk Edge"  # Rare but possible
        elif leverage > 5:
            return "⚠️ Mega Chalk (OK)"
        else:
            return "❌ Mega Chalk Trap"
    
    # Chalk (30-40% owned)
    elif own >= CHALK_THR:
        if leverage > 15 and value > 4.8:
            return "🔥 Chalk w/ Edge"
        elif leverage > 5:
            return "⭐ Chalk (Playable)"
        else:
            return "⚠️ Chalk (Low Edge)"
    
    # Mid ownership (10-30%)
    elif own >= PUNT_THR:
        if leverage > 15 and value > 4.5:
            return "🔥 Elite Leverage"
        elif leverage > 10:
            return "⭐ High Leverage"
        elif leverage > 5:
            return "✅ Good Leverage"
        elif leverage > 0:
            return "➖ Mid (Neutral)"
        else:
            return "➖ Mid (Slight Chalk)"
    
    # Punt/Contrarian (< 10% owned) - inherent edge
    else:
        if leverage > 10 and value > 4.3:
            return "🔥 Elite Contrarian"
        elif leverage > 5:
            return "💎 Contrarian Edge"
        elif leverage > 0:
            return "💎 Contrarian Play"
        elif value > 4.0:
            return "✅ Contrarian Value"
        else:
            return "⚠️ Punt Risk"


def calculate_gpp_score(row):
    """
    Composite GPP score: value + leverage + ceiling.
    """
    ceiling_component = (row["ceiling"] / 100) * 0.4
    value_component = (row["value"] / 7) * 0.4
    leverage_normalized = (row["leverage_score"] + 20) / 40  # -20..+20 -> 0..1
    leverage_component = max(0, leverage_normalized) * 0.2
    gpp_score = (ceiling_component + value_component + leverage_component) * 100
    return round(gpp_score, 1)


def load_and_preprocess_data(pasted_data: str = None) -> pd.DataFrame:
    """
    Map headers, clean salary/own, compute buckets & edge fields.
    """
    empty_df_cols = CORE_INTERNAL_COLS + [
        "player_id",
        "GameID",
        "bucket",
        "value",
        "Lock",
        "Exclude",
    ]
    if pasted_data is None or not pasted_data.strip():
        return pd.DataFrame(columns=empty_df_cols)

    try:
        data_io = io.StringIO(pasted_data)
        first_line = pasted_data.split("\n")[0]
        if "\t" in first_line:
            df = pd.read_csv(data_io, sep="\t")
        else:
            df = pd.read_csv(data_io)

        df.columns = df.columns.str.strip()

        actual_map = {}
        required_internal = [
            "Name",
            "salary",
            "positions",
            "Team",
            "Opponent",
            "proj",
            "own_proj",
        ]
        for csv_name, internal_name in REQUIRED_CSV_TO_INTERNAL_MAP.items():
            if csv_name in df.columns:
                actual_map[csv_name] = internal_name

        mapped_internal_names = set(actual_map.values())
        final_missing_internal = [
            name for name in required_internal if name not in mapped_internal_names
        ]

        if final_missing_internal:
            st.error("❌ Missing required columns.")
            st.error("Required: Player, Salary, Position, Team, Opponent")
            st.error("Projection: one of [Projection, PROJECTED FP, Proj]")
            st.error("Ownership: one of [Ownership, OWNERSHIP %, Own, Own%]")
            st.error(f"Missing: {', '.join(final_missing_internal)}")
            return pd.DataFrame(columns=empty_df_cols)

        df.rename(columns=actual_map, inplace=True)
        if not all(col in df.columns for col in CORE_INTERNAL_COLS):
            st.error("Internal column mapping failed.")
            return pd.DataFrame(columns=empty_df_cols)

    except Exception as e:
        st.error(f"Error processing pasted data: {e}")
        return pd.DataFrame(columns=empty_df_cols)

    # Basic cleaning
    df["Team"] = df["Team"].astype(str)
    df["Opponent"] = df["Opponent"].astype(str)
    df["GameID"] = df.apply(
        lambda row: "@".join(sorted([row["Team"], row["Opponent"]])), axis=1
    )
    df["player_id"] = df["Name"]

    df["own_proj"] = (
        df["own_proj"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df["own_proj"] = pd.to_numeric(df["own_proj"], errors="coerce")
    if df["own_proj"].max() is not None and df["own_proj"].max() <= 1.0 and df[
        "own_proj"
    ].max() > 0:
        df["own_proj"] = df["own_proj"] * 100
    df["own_proj"] = df["own_proj"].round(1)
    df.dropna(subset=CORE_INTERNAL_COLS, inplace=True)

    try:
        df["salary"] = df["salary"].astype(str).str.strip()
        df["salary"] = df["salary"].str.replace("$", "", regex=False)
        df["salary"] = df["salary"].str.replace(",", "", regex=False)
        df["salary"] = pd.to_numeric(df["salary"], errors="coerce").astype("Int64")
        df.dropna(subset=["salary"], inplace=True)
        df["salary"] = df["salary"].astype(int)
        df["proj"] = df["proj"].astype(float)

        if "Minutes" in df.columns:
            df["Minutes"] = (
                pd.to_numeric(df.get("Minutes", 0), errors="coerce")
                .astype(float)
                .round(2)
            )
        if "FPPM" in df.columns:
            df["FPPM"] = (
                pd.to_numeric(df.get("FPPM", 0), errors="coerce")
                .astype(float)
                .round(2)
            )
        if "Value" in df.columns:
            df["Value"] = (
                pd.to_numeric(df.get("Value", 0), errors="coerce")
                .astype(float)
                .round(2)
            )
    except Exception as e:
        st.error(f"Conversion failed: {e}")
        return pd.DataFrame(columns=empty_df_cols)

    if len(df) == 0:
        st.error("❌ Final player pool is empty after cleaning.")
        return pd.DataFrame(columns=empty_df_cols)

    # ownership buckets & edges
    df["bucket"] = df["own_proj"].apply(ownership_bucket)
    df["value"] = np.where(
        df["salary"] > 0, (df["proj"] / (df["salary"] / 1000)).round(2), 0.0
    )

    df["leverage_score"] = df.apply(calculate_player_leverage, axis=1)
    df["ceiling"] = df.apply(calculate_ceiling_score, axis=1)
    df["edge_category"] = df.apply(assign_edge_category, axis=1)
    df["gpp_score"] = df.apply(calculate_gpp_score, axis=1)

    if "Lock" not in df.columns:
        df["Lock"] = False
    if "Exclude" not in df.columns:
        df["Exclude"] = False

    for col in empty_df_cols:
        if col not in df.columns:
            df[col] = None

    return df


# -------------------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------------------
if "slate_df" not in st.session_state:
    st.session_state["slate_df"] = pd.DataFrame(
        columns=CORE_INTERNAL_COLS
        + ["player_id", "GameID", "bucket", "value", "Lock", "Exclude"]
    )
if "edited_df" not in st.session_state:
    st.session_state["edited_df"] = st.session_state["slate_df"].copy()
if "optimal_lineups_results" not in st.session_state:
    st.session_state["optimal_lineups_results"] = {"lineups": [], "ran": False}
if "sport" not in st.session_state:
    st.session_state["sport"] = "NBA"


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def color_bucket(val):
    if val == "mega":
        return "background-color: #9C3838; color: white"
    if val == "chalk":
        return "background-color: #A37F34; color: white"
    if val == "mid":
        return "background-color: #38761D; color: white"
    if val == "punt":
        return "background-color: #3D85C6; color: white"
    return ""


def assign_lineup_positions(lineup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Greedy DK NBA slot assignment.
    """
    assigned_players = set()
    slot_assignments: Dict[str, str] = {}

    def can_play_slot(pos_string, slot):
        if pd.isna(pos_string):
            return False
        positions = [p.strip() for p in str(pos_string).split("/")]
        if slot == "PG":
            return "PG" in positions
        if slot == "SG":
            return "SG" in positions
        if slot == "SF":
            return "SF" in positions
        if slot == "PF":
            return "PF" in positions
        if slot == "C":
            return "C" in positions
        if slot == "G":
            return "PG" in positions or "SG" in positions
        if slot == "F":
            return "SF" in positions or "PF" in positions
        if slot == "UTIL":
            return True
        return False

    def flexibility(pos_string):
        if pd.isna(pos_string):
            return 0
        positions = [p.strip() for p in str(pos_string).split("/")]
        flex_count = len(positions)
        if "PG" in positions or "SG" in positions:
            flex_count += 1
        if "SF" in positions or "PF" in positions:
            flex_count += 1
        return flex_count

    specific_slots = ["C", "PG", "SG", "SF", "PF"]
    flex_slots = ["G", "F", "UTIL"]
    ordered_slots = specific_slots + flex_slots

    success = True
    for slot in ordered_slots:
        available = lineup_df[~lineup_df["player_id"].isin(assigned_players)].copy()
        if available.empty:
            success = False
            break
        eligible = available[
            available["positions"].apply(lambda x: can_play_slot(x, slot))
        ]
        if eligible.empty:
            success = False
            break
        if slot in specific_slots:
            eligible["flexibility"] = eligible["positions"].apply(flexibility)
            eligible = eligible.sort_values("flexibility")
        chosen = eligible.iloc[0]
        slot_assignments[slot] = chosen["player_id"]
        assigned_players.add(chosen["player_id"])

    result_df = lineup_df.copy()
    if not success:
        result_df["roster_slot"] = "UTIL"
        return result_df

    result_df["roster_slot"] = result_df["player_id"].map(
        {v: k for k, v in slot_assignments.items()}
    )
    return result_df


def display_lineups(slate_df: pd.DataFrame, lineup_list: List[Dict[str, Any]]):
    if not lineup_list:
        st.error("❌ No lineups to display.")
        return

    lineup_list = sorted(lineup_list, key=lambda x: x.get("composite_score", x["proj_score"]), reverse=True)
    best = lineup_list[0]
    best_players = slate_df[slate_df["player_id"].isin(best["player_ids"])]

    salary_used = int(best_players["salary"].sum())
    total_own = float(best_players["own_proj"].sum())
    corr_score = best.get("correlation_score", 0)

    st.subheader("Top Lineup Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Projected Points", f"{best['proj_score']:.2f}")
    with c2:
        st.metric("Salary Used", f"${salary_used:,}")
    with c3:
        st.metric("Total Ownership", f"{total_own:.1f}%")
    with c4:
        st.metric("Correlation Score", f"{corr_score:.1f}")

    st.markdown("---")
    st.subheader("Lineup Summary")

    rows = []
    for i, lu in enumerate(lineup_list, start=1):
        lp = slate_df[slate_df["player_id"].isin(lu["player_ids"])]
        rows.append(
            {
                "Lineup": i,
                "Proj": lu["proj_score"],
                "Corr": lu.get("correlation_score", 0),
                "Total Own%": lp["own_proj"].sum(),
                "Salary": lp["salary"].sum(),
                "Games": lu.get("num_games", "-"),
                "Teams": lu.get("num_teams", "-"),
            }
        )
    summary_df = pd.DataFrame(rows).set_index("Lineup")
    st.dataframe(
        summary_df.style.format(
            {
                "Proj": "{:.2f}",
                "Corr": "{:.1f}",
                "Total Own%": "{:.1f}",
                "Salary": "${:,}",
            }
        ),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Lineup Detail")

    options = [
        f"Lineup {i} (Proj {lu['proj_score']:.2f}, Corr {lu.get('correlation_score', 0):.1f})"
        for i, lu in enumerate(lineup_list, start=1)
    ]
    choice = st.selectbox("Choose lineup:", options)
    idx = options.index(choice)
    chosen = lineup_list[idx]
    chosen_df = slate_df[slate_df["player_id"].isin(chosen["player_ids"])].copy()

    # Show stack information
    team_counts = Counter(chosen_df["Team"])
    game_counts = Counter(chosen_df["GameID"])
    
    stacks_info = []
    for team, count in team_counts.items():
        if count >= 2:
            stacks_info.append(f"{team} ({count}x)")
    
    game_stacks_info = []
    for game, count in game_counts.items():
        if count >= 3:
            game_stacks_info.append(f"{game} ({count}x)")
    
    if stacks_info or game_stacks_info:
        st.info(
            f"**Stacks:** {', '.join(stacks_info) if stacks_info else 'None'}  "
            f"| **Game Stacks:** {', '.join(game_stacks_info) if game_stacks_info else 'None'}"
        )

    chosen_df = assign_lineup_positions(chosen_df)
    roster_order = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    cat_type = CategoricalDtype(roster_order, ordered=True)
    chosen_df["roster_slot"] = chosen_df["roster_slot"].astype(cat_type)
    chosen_df.sort_values("roster_slot", inplace=True)

    display_cols = [
        "roster_slot",
        "Name",
        "positions",
        "Team",
        "Opponent",
        "salary",
        "proj",
        "value",
        "own_proj",
        "bucket",
    ]
    df_disp = chosen_df[display_cols].copy()
    df_disp.rename(
        columns={
            "roster_slot": "SLOT",
            "positions": "POS",
            "proj": "Proj Pts",
            "own_proj": "Proj Own%",
            "bucket": "CATEGORY",
        },
        inplace=True,
    )
    styled = df_disp.style.applymap(color_bucket, subset=["CATEGORY"]).format(
        {
            "salary": "${:,}",
            "Proj Pts": "{:.1f}",
            "value": "{:.2f}",
            "Proj Own%": "{:.1f}%",
        }
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def get_builder_style(contest_label: str, field_size: int) -> str:
    """
    Decide "style" for scoring (cash vs SE vs 20-max vs milly).
    """
    if contest_label == "Cash Game (50/50, Double-Up)":
        return "cash"
    if contest_label == "Single Entry":
        return "single_entry"
    if contest_label == "3-Max":
        return "three_max"
    if contest_label == "20-Max":
        return "twenty_max"
    if contest_label == "150-Max (Milly Maker)":
        return "milly"
    return "single_entry"


def get_default_n_lineups(contest_label: str) -> int:
    if contest_label == "Cash Game (50/50, Double-Up)":
        return 1
    if contest_label == "Single Entry":
        return 1
    if contest_label == "3-Max":
        return 3
    if contest_label == "20-Max":
        return 20
    if contest_label == "150-Max (Milly Maker)":
        return 40
    return 10


def get_correlation_strength(contest_label: str) -> float:
    """
    How aggressively to build stacks based on contest type.
    0.0 = no correlation, 1.0 = maximum stacking.
    
    Updated with more conservative defaults for better lineup generation.
    """
    if contest_label == "Cash Game (50/50, Double-Up)":
        return 0.15  # Very light correlation for safety
    if contest_label == "Single Entry":
        return 0.4  # Moderate stacking
    if contest_label == "3-Max":
        return 0.5  # Balanced
    if contest_label == "20-Max":
        return 0.6  # Good correlation without being too strict
    if contest_label == "150-Max (Milly Maker)":
        return 0.7  # Strong stacking but not extreme
    return 0.4


def get_edge_weights(contest_style: str) -> Dict[str, float]:
    """
    Weights for projection / gpp_score / ownership by contest type.
    """
    if contest_style == "cash":
        return {"proj": 1.0, "gpp": 0.2, "own": 0.0}
    if contest_style == "single_entry":
        return {"proj": 0.9, "gpp": 0.5, "own": 0.05}
    if contest_style == "three_max":
        return {"proj": 0.8, "gpp": 0.7, "own": 0.08}
    if contest_style == "twenty_max":
        return {"proj": 0.7, "gpp": 0.9, "own": 0.12}
    if contest_style == "milly":
        return {"proj": 0.6, "gpp": 1.0, "own": 0.18}
    return {"proj": 0.9, "gpp": 0.5, "own": 0.05}


def build_simple_fallback_lineups(
    pool: pd.DataFrame,
    contest_style: str,
    n_lineups: int,
    salary_cap: int,
    roster_size: int,
    locked_ids: List[str],
    max_attempts: int = 8000,
) -> List[Dict[str, Any]]:
    """
    Simple lineup builder that works without Team/correlation data.
    Uses smart salary allocation and projection optimization.
    """
    # Validate basics
    locks_df = pool[pool["player_id"].isin(locked_ids)].copy()
    if len(locks_df) > roster_size:
        return []
    
    lock_salary = int(locks_df["salary"].sum()) if not locks_df.empty else 0
    if lock_salary > salary_cap:
        return []
    
    # Ensure basic columns
    for col in ["proj", "salary"]:
        if col not in pool.columns:
            return []
        pool[col] = pool[col].fillna(0)
    
    # Calculate value if not present
    if "value" not in pool.columns:
        pool["value"] = np.where(
            pool["salary"] > 0,
            (pool["proj"] / (pool["salary"] / 1000)).round(2),
            0.0
        )
    
    # Get weights
    weights = get_edge_weights(contest_style)
    
    # Base pool (non-locked)
    base_pool = pool[~pool["player_id"].isin(locked_ids)].copy()
    
    # Remove total garbage (bottom 5% projection)
    if not base_pool.empty:
        proj_floor = base_pool["proj"].quantile(0.05)
        base_pool = base_pool[base_pool["proj"] >= proj_floor]
    
    if len(base_pool) + len(locks_df) < roster_size:
        return []
    
    # Calculate target salary per remaining slot
    remaining_cap = salary_cap - lock_salary
    remaining_slots = roster_size - len(locks_df)
    target_salary_per_slot = remaining_cap / remaining_slots if remaining_slots > 0 else 0
    
    # Minimum salary usage (95% of cap for cash, 90% for GPP)
    min_salary = salary_cap * (0.95 if contest_style == "cash" else 0.90)
    max_salary_left = salary_cap - min_salary
    
    lineups: List[Dict[str, Any]] = []
    seen_sets = set()
    rng = np.random.default_rng(42)
    
    # Generate lineups with proper salary allocation
    attempts = 0
    while len(lineups) < n_lineups and attempts < max_attempts:
        attempts += 1
        
        # Start with locks
        lineup_ids = list(locks_df["player_id"])
        current_salary = lock_salary
        remaining = roster_size - len(lineup_ids)
        
        if remaining <= 0:
            break
        
        # Build lineup with smart salary allocation
        available = base_pool[~base_pool["player_id"].isin(lineup_ids)].copy()
        
        # For each remaining slot, pick from appropriate salary tier
        for slot_num in range(remaining):
            if available.empty:
                break
            
            slots_left_after = remaining - slot_num - 1
            cap_left = salary_cap - current_salary
            
            # Calculate salary range for this slot
            if slots_left_after > 0:
                # Leave enough for remaining slots (at least $3500 per slot)
                max_this_slot = cap_left - (slots_left_after * 3500)
                min_this_slot = max(3500, cap_left - (slots_left_after * 11000))
            else:
                # Last slot - use all remaining
                max_this_slot = cap_left
                min_this_slot = max(3500, cap_left - max_salary_left)
            
            # Filter to valid salary range
            candidates = available[
                (available["salary"] >= min_this_slot) &
                (available["salary"] <= max_this_slot)
            ].copy()
            
            if candidates.empty:
                # Relax constraints if needed
                candidates = available[available["salary"] <= cap_left].copy()
            
            if candidates.empty:
                break
            
            # Score candidates (projection-focused with value boost)
            if "own_proj" in candidates.columns:
                candidates["score"] = (
                    candidates["proj"] * 1.0 +
                    candidates["value"] * 2.0 -
                    candidates["own_proj"] * 0.05
                )
            else:
                candidates["score"] = (
                    candidates["proj"] * 1.0 +
                    candidates["value"] * 2.0
                )
            
            # Add randomness
            candidates["score"] = candidates["score"] * (0.85 + rng.random(len(candidates)) * 0.3)
            
            # Pick best scoring candidate
            best = candidates.nlargest(1, "score")
            if best.empty:
                break
            
            chosen = best.iloc[0]
            lineup_ids.append(chosen["player_id"])
            current_salary += int(chosen["salary"])
            
            # Remove chosen from available
            available = available[available["player_id"] != chosen["player_id"]]
        
        # Validate lineup
        if len(lineup_ids) != roster_size:
            continue
        
        # Check salary usage
        if current_salary > salary_cap:
            continue
        
        if salary_cap - current_salary > max_salary_left:
            continue  # Left too much money on table
        
        # Check uniqueness
        key = tuple(sorted(lineup_ids))
        if key in seen_sets:
            continue
        seen_sets.add(key)
        
        # Calculate lineup stats
        lineup_df = pool[pool["player_id"].isin(lineup_ids)]
        proj_score = float(lineup_df["proj"].sum())
        
        lineups.append({
            "player_ids": lineup_ids,
            "proj_score": proj_score,
            "salary_used": current_salary,
            "correlation_score": 0.0,
            "num_games": 0,
            "num_teams": 0,
        })
    
    # Sort by projection
    lineups = sorted(lineups, key=lambda x: x["proj_score"], reverse=True)
    return lineups[:n_lineups]


def build_enhanced_lineups(
    df: pd.DataFrame,
    contest_style: str,
    correlation_strength: float,
    n_lineups: int,
    salary_cap: int,
    roster_size: int,
    locked_ids: List[str],
    excluded_ids: List[str],
    sport: str = "NBA",
    max_tries_per_lineup: int = 3000,
) -> List[Dict[str, Any]]:
    """
    Enhanced lineup builder with correlation and stacking.
    
    Supports both NBA and NFL with sport-specific logic.
    """
    from builder import generate_top_n_lineups, LineupTemplate
    
    # Filter out excluded upfront
    pool = df[~df["player_id"].isin(excluded_ids)].copy()
    
    # Check if we have columns needed for correlation
    has_team_data = "Team" in pool.columns and "GameID" in pool.columns
    has_correlation_cols = all(col in pool.columns for col in ["ceiling", "value", "own_proj"])
    
    # If missing correlation data, warn and fall back to simpler building
    if not has_team_data or not has_correlation_cols:
        missing = []
        if not has_team_data:
            missing.append("Team/Opponent")
        if not has_correlation_cols:
            missing.append("edge metrics")
        
        st.warning(f"⚠️ Missing {', '.join(missing)} - using simplified lineup building without correlation")
        st.info("💡 For correlation/stacking, your CSV needs: Player, Salary, Position, Team, Opponent, Projection, Ownership")
        
        # Fall back to basic building
        return build_simple_fallback_lineups(
            pool=pool,
            contest_style=contest_style,
            n_lineups=n_lineups,
            salary_cap=salary_cap,
            roster_size=roster_size,
            locked_ids=locked_ids,
        )
    
    # Branch to sport-specific building
    if sport == "NFL":
        return build_nfl_lineups(
            pool=pool,
            contest_style=contest_style,
            correlation_strength=correlation_strength,
            n_lineups=n_lineups,
            salary_cap=salary_cap,
            roster_size=roster_size,
            locked_ids=locked_ids,
        )
    else:  # NBA
        return build_nba_lineups(
            pool=pool,
            contest_style=contest_style,
            correlation_strength=correlation_strength,
            n_lineups=n_lineups,
            salary_cap=salary_cap,
            roster_size=roster_size,
            locked_ids=locked_ids,
        )
    
def build_nba_lineups(
    pool: pd.DataFrame,
    contest_style: str,
    correlation_strength: float,
    n_lineups: int,
    salary_cap: int,
    roster_size: int,
    locked_ids: List[str],
) -> List[Dict[str, Any]]:
    """
    NBA-specific lineup builder using existing correlation logic.
    """
    from builder import generate_top_n_lineups, LineupTemplate
    
    # Validate locks
    locks_df = pool[pool["player_id"].isin(locked_ids)].copy()
    if len(locks_df) > roster_size:
        return []
    
    lock_salary = int(locks_df["salary"].sum()) if not locks_df.empty else 0
    if lock_salary > salary_cap:
        return []
    
    # Fill NaNs for all numeric columns needed
    for col in ["proj", "gpp_score", "own_proj", "ceiling", "value", "leverage_score"]:
        if col in pool.columns:
            pool[col] = pool[col].fillna(0)
    
    # Create template
    template = LineupTemplate(
        contest_type=contest_style,
        field_size=5000,
        pct_to_first=0.1,
        roster_size=roster_size,
        salary_cap=salary_cap,
        min_games=2,
    )
    
    # Generate lineups with NBA correlation (game/team stacks)
    lineups = generate_top_n_lineups(
        slate_df=pool,
        template=template,
        n_lineups=n_lineups,
        correlation_strength=correlation_strength,
        locked_player_ids=locked_ids,
        excluded_player_ids=[],  # Already filtered
    )
    
    return lineups


def build_nfl_lineups(
    pool: pd.DataFrame,
    contest_style: str,
    correlation_strength: float,
    n_lineups: int,
    salary_cap: int,
    roster_size: int,
    locked_ids: List[str],
    max_attempts: int = 8000,
) -> List[Dict[str, Any]]:
    """
    NFL-specific lineup builder with QB stacking logic.
    """
    import numpy as np
    from nfl_stacks import build_nfl_stacks, identify_nfl_bringback, validate_nfl_lineup, calculate_nfl_correlation_score
    from builder import build_game_environments
    
    # Validate locks
    locks_df = pool[pool["player_id"].isin(locked_ids)].copy()
    if len(locks_df) > roster_size:
        return []
    
    lock_salary = int(locks_df["salary"].sum()) if not locks_df.empty else 0
    if lock_salary > salary_cap:
        return []
    
    # Fill NaNs
    for col in ["proj", "salary", "own_proj", "ceiling", "value"]:
        if col in pool.columns:
            pool[col] = pool[col].fillna(0)
    
    # Build NFL stacks
    nfl_stacks = build_nfl_stacks(pool)
    qb_stacks = nfl_stacks.get("qb_stacks", {})
    rb_dst_stacks = nfl_stacks.get("rb_dst_stacks", {})
    
    # Build game environments
    game_envs = build_game_environments(pool)
    
    # Minimum salary usage
    min_salary = salary_cap * 0.90
    max_salary_left = salary_cap - min_salary
    
    lineups: List[Dict[str, Any]] = []
    seen_sets = set()
    rng = np.random.default_rng(42)
    
    attempts = 0
    while len(lineups) < n_lineups and attempts < max_attempts:
        attempts += 1
        
        # Start with locks
        lineup_ids = list(locks_df["player_id"])
        current_salary = lock_salary
        remaining = roster_size - len(lineup_ids)
        
        if remaining <= 0:
            break
        
        # Decide on stack type based on correlation_strength
        use_qb_stack = rng.random() < correlation_strength
        
        if use_qb_stack and remaining >= 2 and qb_stacks:
            # Try QB stack
            teams_with_stacks = list(qb_stacks.keys())
            if teams_with_stacks:
                team = rng.choice(teams_with_stacks)
                stacks = qb_stacks[team]
                if stacks:
                    stack = rng.choice(stacks[:5])  # Top 5 stacks
                    
                    stack_ids = [stack["qb"], stack["pass_catcher"]]
                    stack_df = pool[pool["player_id"].isin(stack_ids)]
                    stack_salary = int(stack_df["salary"].sum())
                    
                    if len(stack_df) == 2 and stack_salary <= (salary_cap - current_salary):
                        lineup_ids.extend(stack_ids)
                        current_salary += stack_salary
                        remaining -= 2
        
        # Fill remaining spots with smart salary allocation
        available = pool[~pool["player_id"].isin(lineup_ids)].copy()
        
        for slot_num in range(remaining):
            if available.empty:
                break
            
            slots_left_after = remaining - slot_num - 1
            cap_left = salary_cap - current_salary
            
            # Calculate salary range for this slot
            if slots_left_after > 0:
                max_this_slot = cap_left - (slots_left_after * 3000)
                min_this_slot = max(3000, cap_left - (slots_left_after * 10000))
            else:
                max_this_slot = cap_left
                min_this_slot = max(3000, cap_left - max_salary_left)
            
            # Filter to valid salary range
            candidates = available[
                (available["salary"] >= min_this_slot) &
                (available["salary"] <= max_this_slot)
            ].copy()
            
            if candidates.empty:
                candidates = available[available["salary"] <= cap_left].copy()
            
            if candidates.empty:
                break
            
            # Score candidates
            candidates["score"] = (
                candidates["proj"] * 1.0 +
                candidates["value"] * 2.0 -
                candidates["own_proj"] * 0.05
            )
            
            # Add randomness
            candidates["score"] = candidates["score"] * (0.85 + rng.random(len(candidates)) * 0.3)
            
            # Pick best
            best = candidates.nlargest(1, "score")
            if best.empty:
                break
            
            chosen = best.iloc[0]
            lineup_ids.append(chosen["player_id"])
            current_salary += int(chosen["salary"])
            
            available = available[available["player_id"] != chosen["player_id"]]
        
        # Validate lineup
        if len(lineup_ids) != roster_size:
            continue
        
        if current_salary > salary_cap:
            continue
        
        if salary_cap - current_salary > max_salary_left:
            continue
        
        # Check NFL anti-correlations
        lineup_df = pool[pool["player_id"].isin(lineup_ids)]
        if not validate_nfl_lineup(lineup_df):
            continue  # Has bad correlation (QB vs DST)
        
        # Check uniqueness
        key = tuple(sorted(lineup_ids))
        if key in seen_sets:
            continue
        seen_sets.add(key)
        
        # Calculate stats
        proj_score = float(lineup_df["proj"].sum())
        corr_score = calculate_nfl_correlation_score(lineup_df, game_envs)
        
        lineups.append({
            "player_ids": lineup_ids,
            "proj_score": proj_score,
            "salary_used": current_salary,
            "correlation_score": corr_score,
            "num_games": lineup_df["GameID"].nunique(),
            "num_teams": lineup_df["Team"].nunique(),
        })
    
    # Sort by projection + correlation
    for lu in lineups:
        norm_corr = lu["correlation_score"] / 100.0
        lu["composite_score"] = (
            lu["proj_score"] * 0.7 +
            norm_corr * 50 * correlation_strength
        )
    
    lineups = sorted(lineups, key=lambda x: x["composite_score"], reverse=True)
    return lineups[:n_lineups]


# Old code below this point - keeping for reference
# This is the original that called the builder template
# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------

# Sidebar – simple controls
st.sidebar.title("Contest Settings")

contest_type_label = st.sidebar.selectbox(
    "Contest Type",
    options=[
        "Cash Game (50/50, Double-Up)",
        "Single Entry",
        "3-Max",
        "20-Max",
        "150-Max (Milly Maker)",
    ],
    index=1,
)

field_size = st.sidebar.number_input(
    "Approx Field Size",
    min_value=100,
    max_value=200000,
    value=5000,
    step=100,
)

salary_cap = st.sidebar.number_input(
    "Salary Cap",
    min_value=30000,
    max_value=70000,
    value=DEFAULT_SALARY_CAP,
    step=5000,
)

roster_size = DEFAULT_ROSTER_SIZE

default_n = get_default_n_lineups(contest_type_label)
n_lineups = st.sidebar.slider(
    "Number of Lineups",
    min_value=1,
    max_value=40,
    value=default_n,
)

# NEW: Correlation strength control
default_corr = get_correlation_strength(contest_type_label)
correlation_strength = st.sidebar.slider(
    "Correlation/Stacking Strength",
    min_value=0.0,
    max_value=1.0,
    value=default_corr,
    step=0.05,
    help="Higher = more aggressive stacking for upside. Lower = more balanced/safe.",
)

contest_style = get_builder_style(contest_type_label, int(field_size))
st.sidebar.caption(
    f"Build style: **{contest_style}**\n\n"
    f"Correlation: **{int(correlation_strength * 100)}%** (stacking intensity)"
)

st.sidebar.markdown("---")
pasted_csv_data = st.sidebar.text_area(
    "Paste DK player pool (CSV/TSV with headers):",
    height=150,
    placeholder="Player\tSalary\tPosition\tTeam\tOpponent\tProjection\tOwnership\n...",
)
load_btn = st.sidebar.button("Load Player Data")

if load_btn:
    if pasted_csv_data and pasted_csv_data.strip():
        with st.spinner("Processing your data..."):
            loaded_df = load_and_preprocess_data(pasted_csv_data)
            st.session_state["slate_df"] = loaded_df
            st.session_state["edited_df"] = loaded_df.copy()
            
            # Detect sport
            detected_sport = detect_sport(loaded_df)
            st.session_state["sport"] = detected_sport
            
            if not loaded_df.empty:
                st.sidebar.success(f"✅ Loaded {len(loaded_df)} players ({detected_sport})")
                
                # Show game environment analysis
                if "GameID" in loaded_df.columns:
                    game_envs = build_game_environments(loaded_df)
                    high_upside = [g for g, info in game_envs.items() if info.get("is_high_upside")]
                    if high_upside:
                        st.sidebar.info(f"🎯 {len(high_upside)} high-upside games detected")
            else:
                st.sidebar.error("Failed to load data. Check format and try again.")
    else:
        st.sidebar.warning("Paste something first.")


# Main – builder + results
detected_sport = st.session_state.get("sport", "NBA")
sport_emoji = "🏀" if detected_sport == "NBA" else "🏈"
st.title(f"{sport_emoji} {detected_sport} DFS Lineup Builder (Enhanced Correlation)")

slate_df = st.session_state["slate_df"]

if slate_df.empty:
    st.info("Load a player pool in the sidebar to begin.")
else:
    st.subheader("Player Pool (Lock / Exclude)")

    column_config = {
        "Name": st.column_config.TextColumn("Player", disabled=True),
        "edge_category": st.column_config.TextColumn("Edge", disabled=True),
        "gpp_score": st.column_config.NumberColumn(
            "GPP Score", disabled=True, format="%.1f"
        ),
        "leverage_score": st.column_config.NumberColumn(
            "Leverage", disabled=True, format="%+.1f"
        ),
        "ceiling": st.column_config.NumberColumn(
            "Ceiling", disabled=True, format="%.1f"
        ),
        "positions": st.column_config.TextColumn("Pos", disabled=True),
        "salary": st.column_config.NumberColumn("Salary", format="$%d"),
        "proj": st.column_config.NumberColumn("Proj", format="%.1f"),
        "value": st.column_config.NumberColumn("Value", disabled=True, format="%.2f"),
        "own_proj": st.column_config.NumberColumn("Own%", format="%.1f%%"),
        "Lock": st.column_config.CheckboxColumn("🔒"),
        "Exclude": st.column_config.CheckboxColumn("❌"),
        "Team": None,
        "Opponent": None,
        "bucket": None,
        "Minutes": None,
        "FPPM": None,
        "player_id": None,
        "GameID": None,
    }
    column_order = [
        "Lock",
        "Exclude",
        "Name",
        "edge_category",
        "gpp_score",
        "leverage_score",
        "positions",
        "salary",
        "proj",
        "ceiling",
        "value",
        "own_proj",
        "player_id",
    ]

    df_for_editor = st.session_state["edited_df"].copy()
    for col in column_order:
        if col not in df_for_editor.columns:
            if col in ("Lock", "Exclude"):
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
        height=450,
        key="player_editor",
    )
    st.session_state["edited_df"] = edited_df
    edited_df["player_id"] = edited_df["player_id"].astype(str)

    locked_ids = edited_df[edited_df["Lock"] == True]["player_id"].tolist()
    excluded_ids = edited_df[edited_df["Exclude"] == True]["player_id"].tolist()
    if locked_ids or excluded_ids:
        st.caption(f"Locked: {len(locked_ids)}   •   Excluded: {len(excluded_ids)}")

    st.markdown("---")
    run_btn = st.button("Generate Lineups", type="primary")

    if run_btn:
        pool_for_check = edited_df[~edited_df["player_id"].isin(excluded_ids)].copy()

        if len(pool_for_check) < roster_size:
            st.error(
                f"❌ Not enough players to build a full lineup.\n\n"
                f"- Players available after excludes: {len(pool_for_check)}\n"
                f"- Required: {roster_size}\n\n"
                f"Remove some excludes or load a larger pool."
            )
        else:
            cheapest_sum = int(
                pool_for_check["salary"].nsmallest(roster_size).sum()
            )
            if cheapest_sum > salary_cap:
                st.error(
                    "❌ No valid lineup can exist with this salary cap.\n\n"
                    f"- Sum of {roster_size} cheapest players: ${cheapest_sum:,}\n"
                    f"- Salary cap: ${salary_cap:,}\n\n"
                    "Either your salary cap is set too low for this slate,\n"
                    "or your player pool is filtered to only expensive players.\n\n"
                    "Try increasing the cap in the sidebar or broadening the pool."
                )
            else:
                locks_df = pool_for_check[pool_for_check["player_id"].isin(locked_ids)]

                if len(locks_df) > roster_size:
                    st.error(
                        f"❌ Too many locked players.\n\n"
                        f"- Locked: {len(locks_df)}\n"
                        f"- Roster size: {roster_size}\n\n"
                        f"Unlock at least {len(locks_df) - roster_size} player(s)."
                    )
                else:
                    lock_salary = int(locks_df["salary"].sum()) if not locks_df.empty else 0
                    if lock_salary > salary_cap:
                        st.error(
                            f"❌ Salary cap exceeded by locks alone.\n\n"
                            f"- Locked salary: ${lock_salary:,}\n"
                            f"- Cap: ${salary_cap:,}\n\n"
                            f"Unlock a high-salary player or two."
                        )
                    else:
                        detected_sport = st.session_state.get("sport", "NBA")
                        sport_name = "NBA" if detected_sport == "NBA" else "NFL"
                        with st.spinner(f"Building {sport_name} lineups with stacking logic..."):
                            lineups = build_enhanced_lineups(
                                df=edited_df,
                                contest_style=contest_style,
                                correlation_strength=correlation_strength,
                                n_lineups=n_lineups,
                                salary_cap=salary_cap,
                                roster_size=roster_size,
                                locked_ids=locked_ids,
                                excluded_ids=excluded_ids,
                                sport=detected_sport,
                            )

                        if not lineups:
                            st.error(
                                "❌ Could not generate any valid lineups.\n\n"
                                "The correlation/stacking constraints may be too strict.\n\n"
                                "Try:\n"
                                "• Reducing correlation strength\n"
                                "• Reducing locked players\n"
                                "• Removing some excludes\n"
                                "• Increasing the salary cap"
                            )
                        else:
                            st.session_state["optimal_lineups_results"] = {
                                "lineups": lineups,
                                "ran": True,
                            }
                            st.success(f"✅ Built {len(lineups)} correlated lineups.")

    if st.session_state["optimal_lineups_results"].get("ran", False):
        st.markdown("---")
        display_lineups(
            slate_df,
            st.session_state["optimal_lineups_results"]["lineups"],
        )
