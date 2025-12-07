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
    /* overall background */
    .main {
        background-color: #050814;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    /* Generic card */
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
    /* Section titles */
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
    /* Buttons */
