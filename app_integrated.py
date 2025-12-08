"""
DFS Elite Tools - Integrated
Builder + Reviewer in one app
"""

import streamlit as st

st.set_page_config(page_title="DFS Elite Tools", layout="wide")

st.title("🏆 DFS Elite Tools")

# Sidebar navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio(
    "Select Tool:",
    ["🏗️ Lineup Builder", "🔍 Lineup Reviewer"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Quick Guide

**Lineup Builder:**
- Upload CSV
- Generate lineups
- Focus on elite games

**Lineup Reviewer:**
- Upload winner
- See why it won
- Get replication steps
""")

# Show selected page
if page == "🏗️ Lineup Builder":
    st.info("🏗️ Open the builder: `streamlit run app_simple_elite.py`")
    st.markdown("""
    ### Lineup Builder Features
    - ✅ Multi-sport support (NBA, NFL, MLB, NHL)
    - ✅ Game environment analysis
    - ✅ Smart stacking from elite games
    - ✅ Projection + Ceiling tracking
    - ✅ Actual points entry
    - ✅ Performance metrics
    
    **File:** `app_simple_elite.py`
    """)

elif page == "🔍 Lineup Reviewer":
    st.info("🔍 Open the reviewer: `streamlit run lineup_reviewer.py`")
    st.markdown("""
    ### Lineup Reviewer Features  
    - ✅ Analyze winning lineups
    - ✅ Auto-generate insights
    - ✅ Replication blueprint
    - ✅ Leverage analysis
    - ✅ Bulk upload (quick paste)
    - ✅ Full CSV upload
    
    **File:** `lineup_reviewer.py`
    """)

st.markdown("---")
st.markdown("### 🚀 Quick Start")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Build Lineups:**
    ```bash
    streamlit run app_simple_elite.py
    ```
    
    1. Upload DFS CSV
    2. Filter to elite games
    3. Generate 20 lineups
    4. Enter actual scores
    5. Track performance
    """)

with col2:
    st.markdown("""
    **Review Winners:**
    ```bash
    streamlit run lineup_reviewer.py
    ```
    
    1. Paste player data
    2. Get instant analysis
    3. See key insights
    4. Get replication steps
    """)
