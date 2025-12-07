import io
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import streamlit as st

# -------------------------------------------------------------------
# BASIC CONFIG / NBA RULES
# -------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="NBA DFS Lineup Builder")

DEFAULT_SALARY_CAP = 50000
DEFAULT_ROSTER_SIZE = 8

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------

def ownership_bucket(own: float) -> str:
    """Map projected ownership into a coarse bucket."""
    if pd.isna(own):
        return "mid"
    if own >= 40:
        return "mega"
    if own >= 30:
        return "chalk"
    if own >= 10:
        return "mid"
    return "punt"


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DraftKings-style projections CSV into internal schema."""
    df = df.copy()
    
    # Normalize headers
    df.columns = df.columns.str.strip().str.lower()
    
    # Map DK headers
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
    
    # Required columns
    required = ["Name", "Team", "positions", "Salary", "proj"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        return pd.DataFrame()
    
    # Clean Salary
    if "Salary" in df.columns:
        df["Salary"] = (
            df["Salary"]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .str.strip()
        )
    
    # Clean ownership
    if "own_proj" in df.columns:
        df["own_proj"] = (
            df["own_proj"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
    else:
        df["own_proj"] = 15.0  # Default if missing
    
    # Create player_id
    if "player_id" not in df.columns:
        df["player_id"] = (
            df["Name"].astype(str)
            + "_" + df["Team"].astype(str)
            + "_" + df["Salary"].astype(str)
        )
    
    # Ensure numeric columns
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce")
    df["own_proj"] = pd.to_numeric(df["own_proj"], errors="coerce").fillna(15.0)
    
    # Drop rows missing essentials
    df = df.dropna(subset=["Salary", "proj"])
    
    # Calculate value
    df["value"] = np.where(
        df["Salary"] > 0, 
        (df["proj"] / (df["Salary"] / 1000)).round(2), 
        0.0
    )
    
    # Add ownership bucket
    df["own_bucket"] = df["own_proj"].apply(ownership_bucket)
    
    # Calculate edge metrics
    df["leverage"] = df.apply(
        lambda row: (row["value"] / 5.0 * 100) - row["own_proj"], 
        axis=1
    ).round(1)
    
    df["ceiling"] = (df["proj"] * 1.35).round(1)
    
    return df


def detect_sport(df: pd.DataFrame) -> str:
    """Detect sport type from position data."""
    if df.empty or "positions" not in df.columns:
        return "NBA"
    
    all_positions = set()
    for val in df["positions"]:
        if isinstance(val, str):
            for p in val.replace(" ", "").split("/"):
                all_positions.add(p.upper())
    
    nfl_positions = {"QB", "RB", "WR", "TE", "DST", "DEF"}
    nba_positions = {"PG", "SG", "SF", "PF", "C"}
    
    if all_positions & nfl_positions:
        return "NFL"
    if all_positions & nba_positions:
        return "NBA"
    return "NBA"  # Default


def assign_lineup_positions(lineup_df: pd.DataFrame) -> pd.DataFrame:
    """Greedy DK NBA slot assignment."""
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


# -------------------------------------------------------------------
# LINEUP GENERATION
# -------------------------------------------------------------------

def validate_lineup_positions(lineup_df: pd.DataFrame) -> bool:
    """Validate NBA lineup can fill all required positions."""
    required = {
        "PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1,
        "G": 1, "F": 1, "UTIL": 1
    }
    
    for slot, needed in required.items():
        eligible_count = 0
        for _, row in lineup_df.iterrows():
            pos_string = str(row.get("positions", ""))
            positions = [p.strip() for p in pos_string.split("/")]
            
            if slot in ["PG", "SG", "SF", "PF", "C"]:
                if slot in positions:
                    eligible_count += 1
            elif slot == "G":
                if "PG" in positions or "SG" in positions:
                    eligible_count += 1
            elif slot == "F":
                if "SF" in positions or "PF" in positions:
                    eligible_count += 1
            else:  # UTIL
                eligible_count += 1
        
        if eligible_count < needed:
            return False
    return True


def build_random_lineup(
    pool: pd.DataFrame,
    roster_size: int,
    salary_cap: int,
    locked_ids: List[str],
    excluded_ids: List[str],
    rng
) -> List[str]:
    """Generate one random valid lineup - ENSURE locks are included."""
    # Start with locks - they MUST be in the lineup
    current_ids = list(locked_ids)
    locks_df = pool[pool["player_id"].isin(current_ids)]
    lock_salary = int(locks_df["Salary"].sum()) if not locks_df.empty else 0
    
    if len(current_ids) > roster_size or lock_salary > salary_cap:
        return []
    
    remaining_spots = roster_size - len(current_ids)
    remaining_cap = salary_cap - lock_salary
    
    available = pool[
        (~pool["player_id"].isin(current_ids)) & 
        (~pool["player_id"].isin(excluded_ids))
    ].copy()
    
    if available.empty and remaining_spots > 0:
        return []
    
    # Weight by value with some randomness
    scores = available["value"].values + rng.normal(0, 0.2, len(available))
    available = available.assign(_score=scores).sort_values("_score", ascending=False)
    
    for _, row in available.iterrows():
        if remaining_spots == 0:
            break
        if row["Salary"] <= remaining_cap:
            current_ids.append(row["player_id"])
            remaining_spots -= 1
            remaining_cap -= int(row["Salary"])
    
    if remaining_spots != 0:
        return []
    
    return current_ids


def generate_lineups(
    pool: pd.DataFrame,
    n_lineups: int,
    roster_size: int,
    salary_cap: int,
    locked_ids: List[str],
    excluded_ids: List[str],
    max_attempts: int = 5000
) -> List[Dict[str, Any]]:
    """Generate multiple valid lineups with position validation."""
    rng = np.random.default_rng(42)
    
    lineups: List[Dict[str, Any]] = []
    seen_sets = set()
    
    for _ in range(max_attempts):
        if len(lineups) >= n_lineups:
            break
        
        lineup_ids = build_random_lineup(
            pool, roster_size, salary_cap, locked_ids, excluded_ids, rng
        )
        
        if not lineup_ids:
            continue
        
        key = tuple(sorted(lineup_ids))
        if key in seen_sets:
            continue
        
        # Validate lineup has valid positions
        lineup_df = pool[pool["player_id"].isin(lineup_ids)]
        if not validate_lineup_positions(lineup_df):
            continue
        
        seen_sets.add(key)
        
        proj_score = float(lineup_df["proj"].sum())
        salary_used = int(lineup_df["Salary"].sum())
        total_own = float(lineup_df["own_proj"].sum())
        
        lineups.append({
            "player_ids": lineup_ids,
            "proj_score": proj_score,
            "salary_used": salary_used,
            "total_own": total_own
        })
    
    # Sort by projection
    lineups = sorted(lineups, key=lambda x: x["proj_score"], reverse=True)
    return lineups[:n_lineups]


# -------------------------------------------------------------------
# UI COMPONENTS
# -------------------------------------------------------------------

def display_lineups(slate_df: pd.DataFrame, lineup_list: List[Dict[str, Any]]):
    """Display generated lineups."""
    if not lineup_list:
        st.error("❌ No lineups to display.")
        return
    
    lineup_list = sorted(lineup_list, key=lambda x: x["proj_score"], reverse=True)
    best = lineup_list[0]
    
    st.subheader("Top Lineup Snapshot")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Projected Points", f"{best['proj_score']:.2f}")
    with c2:
        st.metric("Salary Used", f"${best['salary_used']:,}")
    with c3:
        st.metric("Total Ownership", f"{best['total_own']:.1f}%")
    
    st.markdown("---")
    st.subheader("Lineup Summary")
    
    rows = []
    for i, lu in enumerate(lineup_list, start=1):
        rows.append({
            "Lineup": i,
            "Proj": lu["proj_score"],
            "Total Own%": lu["total_own"],
            "Salary": lu["salary_used"],
        })
    
    summary_df = pd.DataFrame(rows).set_index("Lineup")
    st.dataframe(
        summary_df.style.format({
            "Proj": "{:.2f}",
            "Total Own%": "{:.1f}",
            "Salary": "${:,}"
        }),
        use_container_width=True,
    )
    
    st.markdown("---")
    st.subheader("Lineup Detail")
    
    options = [
        f"Lineup {i} (Proj {lu['proj_score']:.2f})"
        for i, lu in enumerate(lineup_list, start=1)
    ]
    choice = st.selectbox("Choose lineup:", options)
    idx = options.index(choice)
    chosen = lineup_list[idx]
    chosen_df = slate_df[slate_df["player_id"].isin(chosen["player_ids"])].copy()
    
    chosen_df = assign_lineup_positions(chosen_df)
    roster_order = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    cat_type = CategoricalDtype(roster_order, ordered=True)
    chosen_df["roster_slot"] = chosen_df["roster_slot"].astype(cat_type)
    chosen_df.sort_values("roster_slot", inplace=True)
    
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


# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------

st.title("🏀 NBA DFS Lineup Builder")

# Sidebar
st.sidebar.title("Settings")

contest_type = st.sidebar.selectbox(
    "Contest Type",
    options=["Cash Game", "Single Entry", "3-Max", "20-Max", "150-Max"],
    index=1,
)

salary_cap = st.sidebar.number_input(
    "Salary Cap",
    min_value=30000,
    max_value=70000,
    value=DEFAULT_SALARY_CAP,
    step=1000,
)

roster_size = DEFAULT_ROSTER_SIZE

# Default n_lineups based on contest
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
    "Upload CSV (DraftKings format)",
    type=["csv"],
    help="Should have: Player, Salary, Position, Team, Opponent, Projection, Ownership"
)

if uploaded_file:
    with st.spinner("Loading data..."):
        raw_df = pd.read_csv(uploaded_file)
        slate_df = normalize_df(raw_df)
        
        if not slate_df.empty:
            st.session_state["slate_df"] = slate_df
            st.sidebar.success(f"✅ Loaded {len(slate_df)} players")
        else:
            st.sidebar.error("❌ Failed to load data")

slate_df = st.session_state["slate_df"]

if slate_df.empty:
    st.info("👆 Upload a player pool CSV to begin")
else:
    sport = detect_sport(slate_df)
    st.caption(f"Detected sport: **{sport}**")
    
    st.subheader("Player Pool")
    
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
    
    # Ensure required columns exist
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
        "player_id": None,  # Hide this column
        "Name": st.column_config.TextColumn("Player", disabled=True),
        "positions": st.column_config.TextColumn("Pos", disabled=True),
        "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
        "proj": st.column_config.NumberColumn("Proj", format="%.1f"),
        "value": st.column_config.NumberColumn("Value", format="%.2f"),
        "own_proj": st.column_config.NumberColumn("Own%", format="%.1f%%"),
        "leverage": st.column_config.NumberColumn("Leverage", format="%+.1f"),
        "ceiling": st.column_config.NumberColumn("Ceiling", format="%.1f"),
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
    
    # Update session state with edits - safer approach
    if "player_id" in edited_df.columns:
        for idx, row in edited_df.iterrows():
            pid = row.get("player_id")
            if pid and pid in st.session_state["slate_df"]["player_id"].values:
                mask = st.session_state["slate_df"]["player_id"] == pid
                st.session_state["slate_df"].loc[mask, "Lock"] = row.get("Lock", False)
                st.session_state["slate_df"].loc[mask, "Exclude"] = row.get("Exclude", False)
    
    # Get locked/excluded from edited df
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
        
        with st.spinner("Building lineups..."):
            lineups = generate_lineups(
                pool=pool_for_build,
                n_lineups=n_lineups,
                roster_size=roster_size,
                salary_cap=salary_cap,
                locked_ids=locked_ids,
                excluded_ids=excluded_ids,
            )
        
        if not lineups:
            st.error(
                "❌ Could not generate valid lineups.\n\n"
                "Try:\n"
                "- Reducing locked players\n"
                "- Removing some excludes\n"
                "- Increasing salary cap"
            )
        else:
            st.session_state["lineups"] = lineups
            st.success(f"✅ Generated {len(lineups)} lineups")
    
    if st.session_state["lineups"]:
        st.markdown("---")
        display_lineups(st.session_state["slate_df"], st.session_state["lineups"])
