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
# LIGHT CSS – clean dark theme
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
# STRATEGY PROFILES (construction rules by contest type)
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
    d
