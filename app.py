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

# -------------------------------------------------------------------
# BASIC CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="NBA DFS Strategy Engine",
)

MIN_GAMES_REQUIRED = 2

# -------------------------------------------------------------------
# SIMPLE DARK THEME STYLING
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# STRATEGY PROFILES (just for display – does NOT hard-block lineups)
# -------------------------------------------------------------------
STRATEGY_PROFILES: Dict[str, Dict[str, Any]] = {
    "CASH": {
        "name": "Cash Games (50/50, Double-Up)",
        "description": "Beat ~50% of the field. Embrace projection and floor.",
        "chalk_range": "4–6 chalk plays (>20%)",
        "mid_range": "1–3 mid-owned (10–20%)",
        "contrarian_range": "0–1 contrarian (<10%)",
        "total_own": "260–340% total",
        "salary": "Leave $0–$500",
        "priority": "Projection & floor over uniqueness.",
    },
    "SE": {
        "name": "Single Entry GPP",
        "description": "Smaller field GPP. Mostly with the field, plus 1–2 smart pivots.",
        "chalk_range": "5–6 chalk plays (>20%)",
        "mid_range": "1–2 mid-owned (10–20%)",
        "contrarian_range": "1 contrarian (<10%)",
        "total_own": "120–150% total",
        "salary": "Leave $0–$300",
        "priority": "Projection first, one leverage spot.",
    },
    "3MAX": {
        "name": "3-Max GPP",
        "description": "Three bullets. Slightly more leverage than SE.",
        "chalk_range": "3–4 chalk plays (>20%)",
        "mid_range": "1–2 mid-owned (10–20%)",
        "contrarian_range": "1 contrarian (<10%)",
        "total_own": "120–150% total",
        "salary": "Leave $0–$300",
        "priority": "Projection first, one clear pivot.",
    },
    "20MAX": {
        "name": "20-Max GPP",
        "description": "Mid-sized GPP. Mix projection + leverage across 20 entries.",
        "chalk_range": "2–3 chalk plays (>20%)",
        "mid_range": "1–2 mid-owned (10–20%)",
        "contrarian_range": "1–2 contrarian (<10%)",
        "total_own": "120–160% total",
        "salary": "Leave $0–$400",
        "priority": "Stay different but not stupid; one contrarian each lineup.",
    },
    "MILLIMAX": {
        "name": "Large Field / Milly Maker",
        "description": "Huge fields. Need strong projection *and* serious leverage.",
        "chalk_range": "1–2 chalk plays (>20%)",
        "mid_range": "2–3 mid-owned (10–20%)",
        "contrarian_range": "2–4 contrarian (<10%)",
        "total_own": "140–220% total",
        "salary": "Leave $0–$800",
        "priority": "Ceiling and leverage over raw median.",
    },
}

ENTRY_FEE_TIERS = {
    "$0.25": {"min_field": 100, "max_field": 2500, "skill_level": "Beginner"},
    "$1": {"min_field": 100, "max_field": 5000, "skill_level": "Recreational"},
    "$3": {"min_field": 500, "max_field": 10000, "skill_level": "Intermediate"},
    "$5": {"min_field": 500, "max_field": 25000, "skill_level": "Intermediate"},
    "$10": {"min_field": 1000, "max_field": 50000, "skill_level": "Advanced"},
    "$20": {"min_field": 2000, "max_field": 100000, "skill_level": "Advanced"},
    "$50+": {"min_field": 5000, "max_field": 200000, "skill_level": "Expert / Shark"},
}

# -------------------------------------------------------------------
# CSV MAPPING / EDGE CALCS (your original logic)
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
    if pasted_data is None or not pasted_data.strip():
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

        mapped_internal_names = set(actual_map.values())
        final_missing_internal = [
            name for name in required_internal if name not in mapped_internal_names
        ]

        if final_missing_internal:
            st.error("❌ Missing essential columns.")
            st.error("Required: Player, Salary, Position, Team, Opponent")
            st.error("Projection: 'Projection' / 'PROJECTED FP' / 'Proj'")
            st.error("Ownership: 'Ownership' / 'OWNERSHIP %' / 'Own' / 'Own%'")
            st.error(f"Missing: {', '.join(final_missing_internal)}")
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

# -------------------------------------------------------------------
# SMALL HELPERS
# -------------------------------------------------------------------
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
    Greedy positional assignment for DK NBA.
    If it can't be perfect, everything just shows as UTIL instead of erroring.
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


def display_lineups(
    slate_df: pd.DataFrame,
    template,
    lineup_list: List[Dict[str, Any]],
):
    if not lineup_list:
        st.error("❌ No lineups to display.")
        return

    # sort by projection
    lineup_list = sorted(lineup_list, key=lambda x: x["proj_score"], reverse=True)
    best = lineup_list[0]
    best_players = slate_df[slate_df["player_id"].isin(best["player_ids"])]

    salary_used = int(best_players["salary"].sum())
    total_own = float(best_players["own_proj"].sum())

    st.markdown("### 🔝 Top Lineup Snapshot")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Projected Points", f"{best['proj_score']:.2f}")
    with c2:
        st.metric("Salary Used", f"${salary_used:,}", delta=f"${template.salary_cap - salary_used:,} left")
    with c3:
        st.metric("Total Ownership", f"{total_own:.1f}%")

    # summary table
    st.markdown("---")
    st.markdown("### 📋 Lineup Summary")
    rows = []
    for i, lu in enumerate(lineup_list, start=1):
        lp = slate_df[slate_df["player_id"].isin(lu["player_ids"])]
        rows.append(
            {
                "Lineup": i,
                "Proj": lu["proj_score"],
                "Total Own%": lp["own_proj"].sum(),
                "Salary": lp["salary"].sum(),
            }
        )
    summary_df = pd.DataFrame(rows).set_index("Lineup")
    st.dataframe(
        summary_df.style.format(
            {"Proj": "{:.2f}", "Total Own%": "{:.1f}", "Salary": "${:,}"}
        ),
        use_container_width=True,
    )

    # detail selector
    st.markdown("---")
    st.markdown("### 🔍 Detailed Lineup View")
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


# -------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; padding-bottom: 1.2rem;">
      <div style="font-size: 2.0rem; font-weight: 800; color: #e5e7ff; margin-bottom:0.25rem;">
        NBA DFS <span style="color:#38bdf8;">Strategy Engine</span>
      </div>
      <div class="dfs-subtitle">
        Pick the contest, paste your pool, lock a few plays. I’ll build lineups that fit the contest type.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# CONTEST CONFIG CARD
# -------------------------------------------------------------------
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
        fee_info = ENTRY_FEE_TIERS[entry_fee_label]

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
            fee_info["min_field"],
            min(583, fee_info["max_field"]),
        )
        total_entries = st.number_input(
            "Total Entries",
            min_value=fee_info["min_field"],
            max_value=fee_info["max_field"],
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
            "Payout Structure", ["Flat", "Balanced", "Top-Heavy"], index=1
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

    # map to builder + profile keys
    tournament_map = {
        "Cash Game (50/50, Double-Up)": ("CASH", "CASH"),
        "Single Entry GPP": ("SE", "SE"),
        "3-Max GPP": ("SE", "3MAX"),        # builder SE template, 3-max profile
        "20-Max GPP": ("LARGE_GPP", "20MAX"),
        "Large Field GPP (Milly Maker)": ("LARGE_GPP", "MILLIMAX"),
    }
    contest_code, profile_key = tournament_map[tournament_type]
    profile = STRATEGY_PROFILES[profile_key]

    colS1, colS2, colS3 = st.columns(3)
    with colS1:
        st.metric("Field Size", f"{int(total_entries):,}")
    with colS2:
        skill = fee_info["skill_level"]
        st.metric("Skill Level", skill)
    with colS3:
        st.metric("Profile", profile["name"])

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# PLAYER POOL CARD
# -------------------------------------------------------------------
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
            height=150,
            placeholder="Player\tSalary\tPosition\tTeam\tOpponent\tProjection\tOwnership\n...",
        )
        load_btn = st.button("Load Player Data")

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

        column_config = {
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
            "Lock": st.column_config.CheckboxColumn("🔒", help="Lock into all lineups"),
            "Exclude": st.column_config.CheckboxColumn("❌", help="Exclude from all lineups"),
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

# -------------------------------------------------------------------
# STRATEGY SUMMARY CARD (just display, no constraints)
# -------------------------------------------------------------------
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
        st.table(
            pd.DataFrame(
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
                        profile["chalk_range"],
                        profile["mid_range"],
                        profile["contrarian_range"],
                        profile["total_own"],
                        profile["salary"],
                        profile["priority"],
                    ],
                }
            )
        )
    with c2:
        st.markdown("##### Notes")
        st.markdown(
            "- Use this as **guidelines**, not hard rules.\n"
            "- The builder will focus on **projection** and basic constraints.\n"
            "- You handle game theory; this gives you a quick starting point."
        )

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# LINEUP GENERATION CARD
# -------------------------------------------------------------------
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
            default_n = 1 if contest_code == "CASH" else 10
            n_lineups = st.slider(
                "Number of Lineups",
                min_value=1,
                max_value=40,
                value=default_n,
            )
        with colB:
            bucket_slack = st.slider(
                "Ownership Bucket Flex (higher = looser)",
                min_value=0,
                max_value=6,
                value=3,
                help="This is passed straight into builder.generate_top_n_lineups.",
            )

        run_btn = st.button(
            "Generate Lineups", use_container_width=True, type="primary"
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
                    slate_df[merge_cols],
                    on="player_id",
                    how="left",
                )

                template = build_template_from_params(
                    contest_type=contest_code,
                    field_size=int(total_entries),
                    pct_to_first=0.20,  # simple default; your builder can handle it
                    roster_size=DEFAULT_ROSTER_SIZE,
                    salary_cap=DEFAULT_SALARY_CAP,
                    min_games=MIN_GAMES_REQUIRED,
                )

                with st.spinner("Building lineups..."):
                    top_lineups = generate_top_n_lineups(
                        slate_df=final_df,
                        template=template,
                        n_lineups=n_lineups,
                        bucket_slack=bucket_slack,
                        locked_player_ids=locked_ids,
                        excluded_player_ids=excluded_ids,
                    )

                if not top_lineups:
                    st.error(
                        "❌ Could not generate any valid lineups. "
                        "This is coming straight from `generate_top_n_lineups`.\n\n"
                        "Try:\n- Reducing locks\n- Removing some excludes\n- Expanding your player pool."
                    )
                else:
                    st.session_state["optimal_lineups_results"] = {
                        "lineups": top_lineups,
                        "ran": True,
                    }
                    st.success(f"✅ Built {len(top_lineups)} lineups.")

        if st.session_state["optimal_lineups_results"].get("ran", False):
            st.markdown("---")
            # rebuild template for display (cheap)
            template = build_template_from_params(
                contest_type=contest_code,
                field_size=int(total_entries),
                pct_to_first=0.20,
                roster_size=DEFAULT_ROSTER_SIZE,
                salary_cap=DEFAULT_SALARY_CAP,
                min_games=MIN_GAMES_REQUIRED,
            )
            display_lineups(
                slate_df,
                template,
                st.session_state["optimal_lineups_results"]["lineups"],
            )

        st.markdown("</div>", unsafe_allow_html=True)
