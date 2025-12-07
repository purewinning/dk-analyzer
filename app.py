import io
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import streamlit as st

from builder import (
    build_template_from_params,
    generate_top_n_lineups,
    ownership_bucket,
    PUNT_THR,
    CHALK_THR,
    MEGA_CHALK_THR,
    DEFAULT_SALARY_CAP,
    DEFAULT_ROSTER_SIZE,
)

# --------------------------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="NBA DFS Strategy Engine",
)

# --------------------------------------------------------------------
# LIGHT CSS – keep it clean
# --------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #050814;
        color: #e5e7ff;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .dfs-card {
        background: #0b1020;
        border-radius: 10px;
        padding: 1.0rem 1.25rem;
        border: 1px solid #20263b;
        margin-bottom: 1rem;
    }
    .dfs-section-title {
        font-size: 0.85rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #5ce1ff;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .dfs-subtitle {
        color: #a2a7d4;
        font-size: 0.92rem;
    }
    .stButton>button {
        border-radius: 999px;
        background: #2563eb;
        color: white;
        border: none;
        font-weight: 600;
        height: 2.6rem;
    }
    [data-testid="stMetric"] {
        background: #050814;
        border-radius: 8px;
        padding: 0.45rem 0.7rem;
        border: 1px solid #1e2235;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------
# STRATEGY PROFILES (how we want lineups to look for each contest)
# --------------------------------------------------------------------
STRATEGY_PROFILES: Dict[str, Dict[str, Any]] = {
    "CASH": {
        "name": "Cash Games (50/50, Double-Up)",
        "description": "Beat ~50% of the field. Embrace projection and floor.",
        "chalk_range": (4, 6),        # >20% owned
        "mid_range": (1, 3),          # 10–20% owned
        "contrarian_range": (0, 1),   # <10% owned
        "total_own_range": (260, 340),
        "salary_leave_range": (0, 500),
        "core_locks": (4, 6),
        "player_pool_size": (10, 20),
        "priority": "Projection & floor over uniqueness.",
    },
    "SE": {
        "name": "Single Entry GPP",
        "description": "Smaller field GPP. Mostly with the field, plus 1–2 smart pivots.",
        "chalk_range": (5, 6),
        "mid_range": (1, 2),
        "contrarian_range": (1, 1),
        "total_own_range": (120, 150),
        "salary_leave_range": (0, 300),
        "core_locks": (4, 5),
        "player_pool_size": (15, 20),
        "priority": "Prioritize projection, add one leverage spot.",
    },
    "3MAX": {
        "name": "3-Max GPP",
        "description": "Three bullets. Slightly more leverage than SE, still projection-heavy.",
        "chalk_range": (3, 4),
        "mid_range": (1, 2),
        "contrarian_range": (1, 1),
        "total_own_range": (120, 150),
        "salary_leave_range": (0, 300),
        "core_locks": (4, 5),
        "player_pool_size": (25, 35),
        "priority": "Projection first, one clear contrarian angle.",
    },
    "20MAX": {
        "name": "20-Max GPP",
        "description": "Mid-sized GPP. Mix projection + leverage across 20 entries.",
        "chalk_range": (2, 3),
        "mid_range": (1, 2),
        "contrarian_range": (1, 2),
        "total_own_range": (120, 160),
        "salary_leave_range": (0, 400),
        "core_locks": (3, 5),
        "player_pool_size": (35, 45),
        "priority": "Stay different but not stupid; one contrarian each lineup.",
    },
    "MILLIMAX": {
        "name": "Large Field / Milly Maker",
        "description": "Massive fields. Need strong projection *and* serious leverage.",
        "chalk_range": (1, 2),
        "mid_range": (2, 3),
        "contrarian_range": (2, 4),
        "total_own_range": (140, 220),
        "salary_leave_range": (0, 800),
        "core_locks": (2, 4),
        "player_pool_size": (45, 60),
        "priority": "Ceiling and leverage over raw median projection.",
    },
}

ENTRY_FEE_TIERS = {
    "$0.25": {"min_field": 100, "max_field": 2500, "skill_level": "Beginner"},
    "$1": {"min_field": 100, "max_field": 5000, "skill_level": "Recreational"},
    "$3": {"min_field": 500, "max_field": 10000, "skill_level": "Intermediate"},
    "$5": {"min_field": 500, "max_field": 25000, "skill_level": "Intermediate"},
    "$10": {"min_field": 1000, "max_field": 50000, "skill_level": "Advanced"},
    "$20": {"min_field": 2000, "max_field": 100000, "skill_level": "Advanced"},
    "$50+": {"min_field": 5000, "max_field": 200000, "skill_level": "Expert/Shark"},
}

# --------------------------------------------------------------------
# CSV HEADER MAPPING / LOAD + EDGE CALCS
# --------------------------------------------------------------------
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
CORE_INTERNAL_COLS = ["salary", "positions", "proj", "own_proj", "Name", "Team", "Opponent"]


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
    if row["leverage_score"] > 15 and row["value"] > 4.5:
        return "🔥 Elite Leverage"
    elif row["leverage_score"] > 10:
        return "⭐ High Leverage"
    elif row["leverage_score"] > 5:
        return "✅ Good Leverage"
    elif row["leverage_score"] > -5:
        return "➖ Neutral"
    elif row["leverage_score"] > -15:
        return "⚠️ Slight Chalk"
    else:
        return "❌ Chalk Trap"


def calculate_gpp_score(row):
    ceiling_component = (row["ceiling"] / 100) * 0.4
    value_component = (row["value"] / 7) * 0.3
    leverage_normalized = (row["leverage_score"] + 20) / 40
    leverage_component = max(0, leverage_normalized) * 0.3
    gpp_score = (ceiling_component + value_component + leverage_component) * 100
    return round(gpp_score, 1)


def load_and_preprocess_data(pasted_data: str = None) -> pd.DataFrame:
    empty_df_cols = CORE_INTERNAL_COLS + [
        "player_id",
        "GameID",
        "bucket",
        "value",
        "Lock",
        "Exclude",
    ]
    if not pasted_data or not pasted_data.strip():
        return pd.DataFrame(columns=empty_df_cols)

    try:
        data_io = io.StringIO(pasted_data)
        first_line = pasted_data.split("\n")[0]
        if "\t" in first_line:
            df = pd.read_csv(data_io, sep="\t")
        else:
            df = pd.read_csv(data_io)

        st.success("✅ Data pasted successfully. Checking headers...")
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

        mapped_internal = set(actual_map.values())
        missing = [name for name in required_internal if name not in mapped_internal]
        if missing:
            st.error("❌ Missing essential columns.")
            st.error("Required: Player, Salary, Position, Team, Opponent")
            st.error("Projection: one of 'Projection', 'PROJECTED FP', or 'Proj'")
            st.error("Ownership: one of 'Ownership', 'OWNERSHIP %', 'Own', or 'Own%'")
            st.error(f"Missing: {', '.join(missing)}")
            return pd.DataFrame(columns=empty_df_cols)

        df.rename(columns=actual_map, inplace=True)
        if not all(col in df.columns for col in CORE_INTERNAL_COLS):
            st.error("Internal processing error: column mapping failed.")
            return pd.DataFrame(columns=empty_df_cols)

    except Exception as e:
        st.error(f"Error processing pasted data: {e}")
        return pd.DataFrame(columns=empty_df_cols)

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
    if df["own_proj"].max() <= 1.0 and df["own_proj"].max() > 0:
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

        cols = df.columns.tolist()
        if "Minutes" in cols:
            df["Minutes"] = (
                pd.to_numeric(df.get("Minutes", 0), errors="coerce")
                .astype(float)
                .round(2)
            )
        if "FPPM" in cols:
            df["FPPM"] = (
                pd.to_numeric(df.get("FPPM", 0), errors="coerce")
                .astype(float)
                .round(2)
            )
        if "Value" in cols:
            df["Value"] = (
                pd.to_numeric(df.get("Value", 0), errors="coerce")
                .astype(float)
                .round(2)
            )
    except Exception as e:
        st.error(f"Post-load conversion failed: {e}")
        return pd.DataFrame(columns=empty_df_cols)

    if len(df) == 0:
        st.error("❌ Final player pool is empty after cleaning.")
        return pd.DataFrame(columns=empty_df_cols)

    df["bucket"] = df["own_proj"].apply(ownership_bucket)
    df["value"] = np.where(
        df["salary"] > 0,
        (df["proj"] / (df["salary"] / 1000)).round(2),
        0.0,
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


# --------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------
if "slate_df" not in st.session_state:
    st.session_state["slate_df"] = pd.DataFrame(
        columns=CORE_INTERNAL_COLS
        + ["player_id", "GameID", "bucket", "value", "Lock", "Exclude"]
    )
if "optimal_lineups_results" not in st.session_state:
    st.session_state["optimal_lineups_results"] = {"lineups": [], "ran": False}
if "edited_df" not in st.session_state:
    st.session_state["edited_df"] = st.session_state["slate_df"].copy()

# --------------------------------------------------------------------
# VISUAL HELPERS
# --------------------------------------------------------------------
def color_bucket(s):
    if s == "mega":
        return "background-color: #9C3838; color: white"
    if s == "chalk":
        return "background-color: #A37F34; color: white"
    if s == "mid":
        return "background-color: #38761D; color: white"
    if s == "punt":
        return "background-color: #3D85C6; color: white"
    return ""


def assign_lineup_positions(lineup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Soft positional assignment for DK NBA lineups.
    If it can't find a perfect mapping, everyone defaults to UTIL
    instead of throwing errors.
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


def score_lineup_against_profile(
    lineup_players: pd.DataFrame, profile_key: str, salary_cap: int
) -> Dict[str, Any]:
    profile = STRATEGY_PROFILES.get(profile_key, STRATEGY_PROFILES["SE"])

    own = lineup_players["own_proj"].fillna(0)
    chalk_count = int((own >= 20).sum())
    mid_count = int(((own >= 10) & (own < 20)).sum())
    contrarian_count = int((own < 10).sum())
    total_own = float(own.sum())

    used_salary = int(lineup_players["salary"].sum())
    salary_left = max(salary_cap - used_salary, 0)

    def penalty(val, rng):
        lo, hi = rng
        if val < lo:
            return (lo - val) ** 2
        if val > hi:
            return (val - hi) ** 2
        return 0.0

    pen_chalk = penalty(chalk_count, profile["chalk_range"])
    pen_mid = penalty(mid_count, profile["mid_range"])
    pen_contra = penalty(contrarian_count, profile["contrarian_range"])
    pen_own = penalty(total_own, profile["total_own_range"])
    pen_sal = penalty(salary_left, profile["salary_leave_range"])

    total_penalty = (
        2.0 * pen_chalk
        + 1.5 * pen_mid
        + 2.0 * pen_contra
        + 1.0 * pen_own
        + 0.5 * pen_sal
    )

    return {
        "chalk_count": chalk_count,
        "mid_count": mid_count,
        "contrarian_count": contrarian_count,
        "total_own": total_own,
        "salary_used": used_salary,
        "salary_left": salary_left,
        "strategy_penalty": total_penalty,
    }


def display_lineup_results(
    slate_df: pd.DataFrame,
    template,
    lineup_list: List[Dict[str, Any]],
    profile_key: str,
):
    if not lineup_list:
        st.error("❌ No lineups to display.")
        return

    scored = []
    for lu in lineup_list:
        players_df = slate_df[slate_df["player_id"].isin(lu["player_ids"])]
        strat = score_lineup_against_profile(
            players_df, profile_key, template.salary_cap
        )
        d = lu.copy()
        d.update(strat)
        scored.append(d)

    scored.sort(key=lambda x: (x["strategy_penalty"], -x["proj_score"]))
    best = scored[0]

    top_players_df = slate_df[slate_df["player_id"].isin(best["player_ids"])]
    salary_used = int(top_players_df["salary"].sum())
    st.markdown("### 🔝 Top Lineup Snapshot")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Projected Points", f"{best['proj_score']:.2f}")
    with col2:
        st.metric("Salary Used", f"${salary_used:,}")
    with col3:
        st.metric("Total Ownership", f"{best['total_own']:.1f}%")
    with col4:
        st.metric("Strategy Fit (lower = better)", f"{best['strategy_penalty']:.1f}")

    st.markdown("---")
    st.markdown("### 📋 Lineup Summary")

    summary_rows = []
    for i, lu in enumerate(scored, start=1):
        row = {
            "Lineup": i,
            "Proj": lu["proj_score"],
            "Total Own%": lu["total_own"],
            "Chalk": lu["chalk_count"],
            "Mid": lu["mid_count"],
            "Contrarian": lu["contrarian_count"],
            "Salary Used": lu["salary_used"],
            "Salary Left": lu["salary_left"],
            "Strategy Penalty": lu["strategy_penalty"],
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("Lineup")
    st.dataframe(
        summary_df.style.format(
            {
                "Proj": "{:.2f}",
                "Total Own%": "{:.1f}",
                "Salary Used": "${:,}",
                "Salary Left": "${:,}",
                "Strategy Penalty": "{:.1f}",
            }
        ),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### 🔍 Detailed Lineup View")

    lineup_options = [
        f"Lineup {i} (Proj {lu['proj_score']:.2f})"
        for i, lu in enumerate(scored, start=1)
    ]
    selection = st.selectbox("Choose a lineup to inspect:", lineup_options)
    idx = lineup_options.index(selection)
    chosen = scored[idx]
    chosen_players = slate_df[slate_df["player_id"].isin(chosen["player_ids"])].copy()

    chosen_players = assign_lineup_positions(chosen_players)
    roster_order = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    cat_type = CategoricalDtype(roster_order, ordered=True)
    chosen_players["roster_slot"] = chosen_players["roster_slot"].astype(cat_type)
    chosen_players.sort_values("roster_slot", inplace=True)

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
    df_disp = chosen_players[display_cols].copy()
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
    styled = (
        df_disp.style.applymap(color_bucket, subset=["CATEGORY"])
        .format(
            {
                "salary": "${:,}",
                "Proj Pts": "{:.1f}",
                "value": "{:.2f}",
                "Proj Own%": "{:.1f}%",
            }
        )
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; padding-bottom: 1.2rem;">
      <div style="font-size: 2.0rem; font-weight: 800; color: #e5e7ff; margin-bottom:0.25rem;">
        NBA DFS <span style="color:#38bdf8;">Strategy Engine</span>
      </div>
      <div class="dfs-subtitle">
        Pick the contest. Paste the pool. I’ll build lineups that match a proven construction profile.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------
# CONTEST CONFIGURATION CARD
# --------------------------------------------------------------------
with st.container():
    st.markdown('<div class="dfs-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="dfs-section-title">Contest Configuration</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns([1.4, 1.2, 1.1, 1.1])

    with c1:
        tournament_type = st.selectbox(
            "Contest Type",
            options=[
                "Cash Game (50/50, Double-Up)",
                "Single Entry GPP",
                "3-Max GPP",
                "20-Max GPP",
                "Large Field GPP (Milly Maker)",
            ],
            index=1,
        )

    with c2:
        entry_fee_label = st.selectbox(
            "Entry Fee ($)",
            options=list(ENTRY_FEE_TIERS.keys()),
            index=2,
        )
        entry_fee_info = ENTRY_FEE_TIERS[entry_fee_label]

    with c3:
        field_band = st.selectbox(
            "Field Size Band",
            options=[
                "Small (<2K entries)",
                "Medium (2K–10K)",
                "Large (10K–50K)",
                "Massive (50K+)",
            ],
            index=1,
        )

    with c4:
        default_entries = max(
            entry_fee_info["min_field"],
            min(583, entry_fee_info["max_field"]),
        )
        total_entries = st.number_input(
            "Total Entries",
            min_value=entry_fee_info["min_field"],
            max_value=entry_fee_info["max_field"],
            value=default_entries,
            step=50,
        )

    c5, c6, c7 = st.columns([1.1, 1.1, 1.1])
    with c5:
        slate_games = st.number_input(
            "Slate Size (Games)", min_value=1, max_value=15, value=9
        )
    with c6:
        payout_structure = st.selectbox(
            "Payout Structure",
            ["Flat", "Balanced", "Top-Heavy"],
            index=1,
        )
    with c7:
        injury_vol = st.selectbox(
            "Injury Volatility",
            [
                "Low – most starters locked in",
                "Medium – some key players GTD",
                "High – late news chaos",
            ],
            index=1,
        )

    tournament_map = {
        "Cash Game (50/50, Double-Up)": ("CASH", "CASH", 0.45, 1),
        "Single Entry GPP": ("SE", "SE", 0.20, 1),
        "3-Max GPP": ("3MAX", "3MAX", 0.20, 3),
        "20-Max GPP": ("20MAX", "20MAX", 0.15, 20),
        "Large Field GPP (Milly Maker)": ("LARGE_GPP", "MILLIMAX", 0.20, 150),
    }
    contest_code, profile_key, top_payout_pct, recommended_lineups = tournament_map[
        tournament_type
    ]

    col_summary1, col_summary2, col_summary3 = st.columns(3)
    with col_summary1:
        st.metric("Field Size", f"{int(total_entries):,}")
    with col_summary2:
        st.metric("Payout % (rough)", f"{top_payout_pct*100:.0f}% cashes")
    with col_summary3:
        st.metric("Recommended Entries", f"{recommended_lineups}")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------
# PLAYER POOL CARD
# --------------------------------------------------------------------
with st.container():
    st.markdown('<div class="dfs-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="dfs-section-title">Player Pool</div>',
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns([1.4, 1])
    with c_left:
        pasted_csv_data = st.text_area(
            "Paste DK player pool (CSV/TSV with headers):",
            height=140,
            placeholder="Player\tSalary\tPosition\tTeam\tOpponent\tProjection\tOwnership\n...",
        )
        load_btn = st.button("Load Player Data", key="load_players")

    with c_right:
        st.markdown("**Format Tips**")
        st.markdown(
            "- Export from Labs / RG / Stokastic\n"
            "- Must include: Player, Salary, Position, Team, Opponent, Projection, Ownership\n"
            "- Ownership can be 0–1 or 0–100%"
        )

    if load_btn:
        if pasted_csv_data and pasted_csv_data.strip():
            with st.spinner("Processing your data..."):
                loaded_df = load_and_preprocess_data(pasted_csv_data)
                st.session_state["slate_df"] = loaded_df
                st.session_state["edited_df"] = loaded_df.copy()
                if not loaded_df.empty:
                    st.success(f"✅ Loaded {len(loaded_df)} players.")
                else:
                    st.error("❌ Failed to load data. Check format and try again.")
        else:
            st.warning("Paste something first 🤝")

    slate_df = st.session_state["slate_df"]

    if slate_df.empty:
        st.info("No players loaded yet.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("#### Editable Player Pool")

        col_conf = {
            "Name": st.column_config.TextColumn("Player", disabled=True, width="medium"),
            "edge_category": st.column_config.TextColumn("Edge", disabled=True, width="medium"),
            "gpp_score": st.column_config.NumberColumn("GPP Score", disabled=True, format="%.1f"),
            "leverage_score": st.column_config.NumberColumn("Leverage", disabled=True, format="%+.1f"),
            "ceiling": st.column_config.NumberColumn("Ceiling", disabled=True, format="%.1f"),
            "positions": st.column_config.TextColumn("Pos", disabled=True, width="small"),
            "salary": st.column_config.NumberColumn("Salary", format="$%d"),
            "proj": st.column_config.NumberColumn("Proj", format="%.1f"),
            "value": st.column_config.NumberColumn("Value", format="%.2f", disabled=True),
            "own_proj": st.column_config.NumberColumn("Own%", format="%.1f%%"),
            "Lock": st.column_config.CheckboxColumn("🔒", help="Force into all lineups"),
            "Exclude": st.column_config.CheckboxColumn("❌", help="Remove from all lineups"),
            "Team": None,
            "Opponent": None,
            "bucket": None,
            "Minutes": None,
            "FPPM": None,
            "player_id": None,
            "GameID": None,
        }
        col_order = [
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
        for col in col_order:
            if col not in df_for_editor.columns:
                if col in ("Lock", "Exclude"):
                    df_for_editor[col] = False
                else:
                    df_for_editor[col] = None
        df_for_editor = df_for_editor[col_order]

        edited_df = st.data_editor(
            df_for_editor,
            column_config=col_conf,
            column_order=col_order,
            hide_index=True,
            use_container_width=True,
            height=420,
            key="player_editor",
        )
        st.session_state["edited_df"] = edited_df
        edited_df["player_id"] = edited_df["player_id"].astype(str)

        locked_ids = edited_df[edited_df["Lock"] == True]["player_id"].tolist()
        excluded_ids = edited_df[edited_df["Exclude"] == True]["player_id"].tolist()
        if locked_ids or excluded_ids:
            st.caption(
                f"🔒 Locked: {len(locked_ids)}   •   ❌ Excluded: {len(excluded_ids)}"
            )

        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------
# STRATEGY SUMMARY CARD
# --------------------------------------------------------------------
profile = STRATEGY_PROFILES[profile_key]

with st.container():
    st.markdown('<div class="dfs-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="dfs-section-title">Your Strategy • {tournament_type} • {int(total_entries):,} entries • {entry_fee_label}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**{profile['name']}** – {profile['description']}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Lineup Construction")
        lc_df = pd.DataFrame(
            {
                "Metric": [
                    "Chalk Plays (>20%)",
                    "Mid-Owned (10–20%)",
                    "Contrarian (<10%)",
                    "Total Ownership Target",
                    "Salary Usage",
                    "Priority",
                ],
                "Target": [
                    f"{profile['chalk_range'][0]}–{profile['chalk_range'][1]} players",
                    f"{profile['mid_range'][0]}–{profile['mid_range'][1]} players",
                    f"{profile['contrarian_range'][0]}–{profile['contrarian_range'][1]} players",
                    f"{profile['total_own_range'][0]}–{profile['total_own_range'][1]}%",
                    f"Leave ${profile['salary_leave_range'][0]}–${profile['salary_leave_range'][1]}",
                    profile["priority"],
                ],
            }
        )
        st.table(lc_df)

    with c2:
        st.markdown("##### Core + Rotation")
        cr_df = pd.DataFrame(
            {
                "Item": [
                    "Core Players to Lock",
                    "Core Exposure Range",
                    "Value Locks",
                    "Player Pool Size",
                ],
                "Guideline": [
                    f"{profile['core_locks'][0]}–{profile['core_locks'][1]} core players",
                    "80–100% on top plays (for SE/3-Max); scale down for 20-Max/Milly",
                    "1–2 value plays under $5K",
                    f"{profile['player_pool_size'][0]}–{profile['player_pool_size'][1]} players",
                ],
            }
        )
        st.table(cr_df)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------
# LINEUP GENERATION CARD
# --------------------------------------------------------------------
with st.container():
    st.markdown('<div class="dfs-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="dfs-section-title">Generate Lineups</div>',
        unsafe_allow_html=True,
    )

    if slate_df.empty:
        st.info("Load a player pool above to enable the lineup builder.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        colA, colB = st.columns(2)
        with colA:
            n_lineups = st.slider(
                "Number of Lineups",
                min_value=1,
                max_value=40,
                value=10 if contest_code != "CASH" else 1,
            )
        with colB:
            auto_match = st.checkbox(
                "Match contest strategy profile (recommended)",
                value=True,
            )

        run_btn = st.button(
            "Generate Lineups", key="run_builder", use_container_width=True
        )

        if run_btn:
            conflict = set(locked_ids) & set(excluded_ids)
            if conflict:
                st.error(
                    f"❌ Conflict: {', '.join(sorted(conflict))} are both locked and excluded."
                )
            else:
                final_df = edited_df.copy()
                merge_cols = [
                    "player_id",
                    "bucket",
                    "GameID",
                    "Team",
                    "Opponent",
                    "edge_category",
                    "leverage_score",
                    "ceiling",
                    "gpp_score",
                    "value",
                ]
                merge_cols = [c for c in merge_cols if c in slate_df.columns]
                drop_cols = [
                    c
                    for c in merge_cols
                    if c in final_df.columns and c != "player_id"
                ]
                if drop_cols:
                    final_df = final_df.drop(columns=drop_cols)
                final_df = final_df.merge(
                    slate_df[merge_cols], on="player_id", how="left"
                )

                template = build_template_from_params(
                    contest_type=contest_code,
                    field_size=int(total_entries),
                    pct_to_first=top_payout_pct,
                    roster_size=DEFAULT_ROSTER_SIZE,
                    salary_cap=DEFAULT_SALARY_CAP,
                    min_games=slate_games,
                )

                with st.spinner("Building lineups..."):
                    raw_lineups = generate_top_n_lineups(
                        slate_df=final_df,
                        template=template,
                        n_lineups=n_lineups * 5,
                        bucket_slack=2,
                        locked_player_ids=locked_ids,
                        excluded_player_ids=excluded_ids,
                    )

                if not raw_lineups:
                    st.error(
                        "❌ Could not generate any valid lineups. "
                        "Try reducing locks/excludes or widening your player pool."
                    )
                else:
                    if auto_match:
                        scored = []
                        for lu in raw_lineups:
                            players_df = final_df[
                                final_df["player_id"].isin(lu["player_ids"])
                            ]
                            strat = score_lineup_against_profile(
                                players_df, profile_key, template.salary_cap
                            )
                            d = lu.copy()
                            d.update(strat)
                            scored.append(d)
                        scored.sort(
                            key=lambda x: (x["strategy_penalty"], -x["proj_score"])
                        )
                        top_lineups = scored[:n_lineups]
                    else:
                        raw_lineups.sort(
                            key=lambda x: x["proj_score"], reverse=True
                        )
                        top_lineups = raw_lineups[:n_lineups]

                    st.session_state["optimal_lineups_results"] = {
                        "lineups": top_lineups,
                        "ran": True,
                    }
                    st.success(f"✅ Built {len(top_lineups)} lineups.")

        if st.session_state["optimal_lineups_results"].get("ran", False):
            st.markdown("---")
            display_lineup_results(
                slate_df,
                build_template_from_params(
                    contest_type=contest_code,
                    field_size=int(total_entries),
                    pct_to_first=top_payout_pct,
                    roster_size=DEFAULT_ROSTER_SIZE,
                    salary_cap=DEFAULT_SALARY_CAP,
                    min_games=slate_games,
                ),
                st.session_state["optimal_lineups_results"]["lineups"],
                profile_key,
            )

        st.markdown("</div>", unsafe_allow_html=True)
