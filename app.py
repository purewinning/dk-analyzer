import io
from typing import Dict, Any, List

import numpy as np
import pandas as pd
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
# BASIC CSS TO MIMIC THE STYLED LAYOUT
# --------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #050814;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .dfs-card {
        background: #0c1220;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        border: 1px solid #20263b;
        box-shadow: 0 0 20px rgba(0,0,0,0.45);
    }
    .dfs-card h3, .dfs-card h4 {
        margin-top: 0;
    }
    .dfs-section-title {
        font-size: 0.85rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #5ce1ff;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .dfs-subtitle {
        color: #9ca3c7;
        font-size: 0.90rem;
    }
    .stButton>button {
        border-radius: 999px;
        background: linear-gradient(90deg, #16a9ff, #04e2b0);
        color: white;
        border: none;
        font-weight: 600;
        height: 3rem;
    }
    [data-testid="stMetric"] {
        background: #060a14;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        border: 1px solid #1e2235;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------
# STRATEGY PROFILES (CASH / SE / 3-MAX / 20-MAX / MILLY)
# --------------------------------------------------------------------
STRATEGY_PROFILES: Dict[str, Dict[str, Any]] = {
    "CASH": {
        "name": "Cash Games (50/50, Double-Up)",
        "description": "Beat ~50% of the field. Embrace projection and floor.",
        "chalk_range": (4, 6),        # >20%
        "mid_range": (1, 3),          # 10–20%
        "contrarian_range": (0, 1),   # <10%
        "total_own_range": (260, 340),
        "salary_leave_range": (0, 500),
        "core_locks": (4, 6),
        "player_pool_size": (10, 20),
        "priority": "Projection & floor over uniqueness",
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
        "priority": "Prioritize projection, add one leverage spot",
    },
    "3MAX": {
        "name": "3-Max GPP",
        "description": "Three bullets. A little more leverage than SE, still projection-heavy.",
        "chalk_range": (3, 4),
        "mid_range": (1, 2),
        "contrarian_range": (1, 1),
        "total_own_range": (120, 150),
        "salary_leave_range": (0, 300),
        "core_locks": (4, 5),
        "player_pool_size": (25, 35),
        "priority": "Projection first, one clear contrarian angle",
    },
    "20MAX": {
        "name": "20-Max GPP",
        "description": "Mid-sized GPP. Mix projection + leverage across 20 entries.",
        "chalk_range": (2, 3),
        "mid_range": (1, 2),
        "contraria
