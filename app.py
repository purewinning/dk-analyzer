"""
DFS Elite Tools - Complete Suite
Tab 1: Lineup Builder (build lineups, track performance)
Tab 2: Contest Review (import contest, analyze winners, populate actuals)
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import io

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(page_title="DFS Elite Tools", layout="wide")

# Initialize session state
if 'generated_lineups' not in st.session_state:
    st.session_state.generated_lineups = []
if 'contest_data' not in st.session_state:
    st.session_state.contest_data = None
if 'player_actuals' not in st.session_state:
    st.session_state.player_actuals = {}

# ============================================================================
# CONFIGURATION
# ============================================================================

class Sport(Enum):
    NBA = "NBA"
    NFL = "NFL"

@dataclass
class SportConfig:
    salary_cap: int
    roster_size: int
    positions: Dict[str, int]

SPORT_CONFIGS = {
    Sport.NBA: SportConfig(
        salary_cap=50000,
        roster_size=8,
        positions={"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1, "G": 1, "F": 1, "UTIL": 1}
    ),
    Sport.NFL: SportConfig(
        salary_cap=50000,
        roster_size=9,
        positions={"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "DST": 1}
    ),
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_sport(df: pd.DataFrame) -> Sport:
    """Auto-detect sport from positions."""
    positions = set()
    if "positions" in df.columns:
        for pos_str in df["positions"].dropna():
            if isinstance(pos_str, str):
                positions.update(pos_str.split("/"))
    
    if any(p in positions for p in ["QB", "RB", "WR", "TE"]):
        return Sport.NFL
    return Sport.NBA

def load_and_normalize_csv(uploaded_file) -> pd.DataFrame:
    """Load and normalize player CSV."""
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    
    renames = {
        "player": "name",
        "position": "positions",
        "pos": "positions",
        "salary": "salary",
        "sal": "salary",
        "projection": "proj",
        "fpts": "proj",
        "ownership": "own",
        "own%": "own",
    }
    df = df.rename(columns=renames)
    
    required = ["name", "positions", "salary", "proj"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return None
    
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce")
    
    if "own" in df.columns:
        df["own"] = pd.to_numeric(df["own"], errors="coerce")
        if df["own"].max() <= 1.0:
            df["own"] = df["own"] * 100
    else:
        df["own"] = 15.0
    
    df = df.dropna(subset=["salary", "proj"])
    df["player_id"] = range(len(df))
    
    return df

def calculate_metrics(df: pd.DataFrame, sport: Sport) -> pd.DataFrame:
    """Calculate all player metrics."""
    df["value"] = df["proj"] / (df["salary"] / 1000)
    df["ceiling"] = df["proj"] * 1.4
    df["leverage"] = 100 - df["own"]
    
    df["gpp_score"] = (
        df["ceiling"] * 0.30 +
        df["value"] * 8 * 0.25 +
        df["leverage"] * 0.20 +
        (100 - df["own"]) * 0.10
    )
    
    return df

def generate_lineups(df, sport, config, n_lineups, locks, excludes, correlation, seed=None):
    """Generate tournament lineups."""
    rng = np.random.RandomState(seed)
    lineups = []
    
    for i in range(n_lineups):
        lineup = build_single_lineup(df, sport, config, locks, excludes, correlation, rng)
        if lineup:
            lineups.append(lineup)
    
    return lineups

def build_single_lineup(df, sport, config, locks, excludes, correlation, rng):
    """Build a single lineup."""
    available = df[~df["player_id"].isin(excludes)].copy()
    selected = list(locks)
    salary_used = df[df["player_id"].isin(selected)]["salary"].sum()
    
    # Phase 1: Fill positions
    while len(selected) < config.roster_size:
        candidates = available[~available["player_id"].isin(selected)]
        if candidates.empty:
            break
        
        remaining_salary = config.salary_cap - salary_used
        candidates = candidates[candidates["salary"] <= remaining_salary]
        if candidates.empty:
            break
        
        # Score with randomness
        candidates = candidates.copy()
        candidates["score"] = (
            candidates["ceiling"] * 0.4 +
            candidates["leverage"] * 0.3 +
            candidates["gpp_score"] * 0.3 +
            rng.uniform(0, 20, size=len(candidates))
        )
        
        player = candidates.nlargest(1, "score").iloc[0]
        selected.append(player["player_id"])
        salary_used += player["salary"]
    
    if len(selected) < config.roster_size:
        return None
    
    lineup_df = df[df["player_id"].isin(selected)].copy()
    
    # Assign positions
    if sport == Sport.NBA:
        lineup_df = assign_positions_nba(lineup_df)
    else:
        lineup_df = assign_positions_nfl(lineup_df)
    
    if lineup_df is None:
        return None
    
    return {
        "players": lineup_df,
        "proj": lineup_df["proj"].sum(),
        "ceiling": lineup_df["ceiling"].sum(),
        "salary": lineup_df["salary"].sum(),
        "own": lineup_df["own"].sum(),
        "leverage": lineup_df["leverage"].mean(),
    }

def assign_positions_nba(lineup_df):
    """Assign NBA positions."""
    slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    lineup_df["slot"] = None
    
    for slot in slots:
        unassigned = lineup_df[lineup_df["slot"].isna()].copy()
        if unassigned.empty:
            break
        
        if slot == "G":
            eligible = unassigned[unassigned["positions"].str.contains("PG|SG", na=False)]
        elif slot == "F":
            eligible = unassigned[unassigned["positions"].str.contains("SF|PF", na=False)]
        elif slot == "UTIL":
            eligible = unassigned
        else:
            eligible = unassigned[unassigned["positions"].str.contains(slot, na=False)]
        
        if not eligible.empty:
            player = eligible.iloc[0]
            lineup_df.loc[lineup_df["player_id"] == player["player_id"], "slot"] = slot
    
    if lineup_df["slot"].isna().any():
        return None
    
    return lineup_df

def assign_positions_nfl(lineup_df):
    """Assign NFL positions."""
    slots = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
    lineup_df["slot"] = None
    
    for slot in slots:
        unassigned = lineup_df[lineup_df["slot"].isna()].copy()
        if unassigned.empty:
            break
        
        if slot == "FLEX":
            eligible = unassigned[unassigned["positions"].str.contains("RB|WR|TE", na=False)]
        elif slot == "DST":
            eligible = unassigned[unassigned["positions"].str.contains("DST|DEF|D", na=False)]
        else:
            eligible = unassigned[unassigned["positions"].str.contains(slot, na=False)]
        
        if not eligible.empty:
            player = eligible.iloc[0]
            lineup_df.loc[lineup_df["player_id"] == player["player_id"], "slot"] = slot
    
    if lineup_df["slot"].isna().any():
        return None
    
    return lineup_df

# ============================================================================
# MAIN APP
# ============================================================================

st.title("🏆 DFS Elite Tools")

tab1, tab2 = st.tabs(["🏗️ Lineup Builder", "📊 Contest Review"])

# ============================================================================
# TAB 1: LINEUP BUILDER
# ============================================================================

with tab1:
    st.header("🏗️ Lineup Builder")
    
    uploaded_file = st.file_uploader("Upload DFS CSV", type="csv", key="builder_upload")
    
    if uploaded_file:
        df = load_and_normalize_csv(uploaded_file)
        
        if df is not None:
            sport = detect_sport(df)
            config = SPORT_CONFIGS[sport]
            
            st.success(f"✅ Loaded {len(df)} players for {sport.value}")
            
            df = calculate_metrics(df, sport)
            
            # Settings
            col1, col2 = st.columns(2)
            
            with col1:
                n_lineups = st.number_input("# Lineups", 1, 150, 20)
            
            with col2:
                correlation = st.slider("Stacking", 0.0, 1.0, 0.7, 0.1)
            
            # Player pool
            st.subheader("📊 Player Pool")
            
            df["lock"] = False
            df["exclude"] = False
            
            edited = st.data_editor(
                df[["lock", "exclude", "name", "positions", "team", "salary", "proj", "own", "value", "ceiling"]],
                column_config={
                    "lock": st.column_config.CheckboxColumn("🔒"),
                    "exclude": st.column_config.CheckboxColumn("❌"),
                    "salary": st.column_config.NumberColumn("Salary", format="$%d"),
                    "proj": st.column_config.NumberColumn("Proj", format="%.1f"),
                    "own": st.column_config.NumberColumn("Own%", format="%.1f%%"),
                    "value": st.column_config.NumberColumn("Val", format="%.2f"),
                    "ceiling": st.column_config.NumberColumn("Ceil", format="%.1f"),
                },
                hide_index=True,
                use_container_width=True,
                height=400
            )
            
            # Get locks/excludes
            locks = []
            excludes = []
            
            for idx, row in edited.iterrows():
                match = df[
                    (df["name"] == row["name"]) & 
                    (df["positions"] == row["positions"])
                ]
                
                if not match.empty:
                    player_id = match.iloc[0]["player_id"]
                    if row["lock"]:
                        locks.append(player_id)
                    if row["exclude"]:
                        excludes.append(player_id)
            
            # Generate
            if st.button("🚀 Generate Lineups", type="primary"):
                with st.spinner("Generating lineups..."):
                    lineups = generate_lineups(df, sport, config, n_lineups, locks, excludes, correlation)
                    st.session_state.generated_lineups = lineups
                    st.success(f"✅ Generated {len(lineups)} lineups")
            
            # Display lineups
            if st.session_state.generated_lineups:
                st.markdown("---")
                st.subheader("📋 Generated Lineups")
                
                # Summary
                lineups = st.session_state.generated_lineups
                
                summary = pd.DataFrame([{
                    "Lineup": i+1,
                    "Proj": lu["proj"],
                    "Ceiling": lu["ceiling"],
                    "Own": lu["own"],
                    "Lev": lu["leverage"],
                    "Sal": lu["salary"]
                } for i, lu in enumerate(lineups)])
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Avg Projection", f"{summary['Proj'].mean():.1f}")
                
                with col2:
                    st.metric("Avg Ceiling", f"{summary['Ceiling'].mean():.1f}")
                
                with col3:
                    st.metric("Avg Own", f"{summary['Own'].mean():.1f}%")
                
                with col4:
                    st.metric("Avg Leverage", f"{summary['Lev'].mean():+.1f}")
                
                st.dataframe(summary, use_container_width=True, hide_index=True)
                
                # Detail view
                st.markdown("---")
                choice = st.selectbox(
                    "View lineup:",
                    [f"Lineup {i+1} (Proj {lu['proj']:.0f} | Ceil {lu['ceiling']:.0f})" 
                     for i, lu in enumerate(lineups)]
                )
                
                idx = int(choice.split()[1]) - 1
                lineup = lineups[idx]
                
                # Check if we have actuals from contest
                lineup_players = lineup["players"].copy()
                lineup_players["actual"] = 0.0
                
                # Try to populate from session state
                for i, row in lineup_players.iterrows():
                    player_name = row["name"]
                    if player_name in st.session_state.player_actuals:
                        lineup_players.at[i, "actual"] = st.session_state.player_actuals[player_name]
                
                edited_lineup = st.data_editor(
                    lineup_players[["slot", "name", "positions", "salary", "proj", "ceiling", "own", "actual"]],
                    column_config={
                        "slot": "Slot",
                        "name": "Player",
                        "positions": "Pos",
                        "salary": st.column_config.NumberColumn("Salary", format="$%d"),
                        "proj": st.column_config.NumberColumn("Proj", format="%.1f"),
                        "ceiling": st.column_config.NumberColumn("Ceil", format="%.1f"),
                        "own": st.column_config.NumberColumn("Own%", format="%.1f%%"),
                        "actual": st.column_config.NumberColumn("✏️ Actual", format="%.1f"),
                    },
                    disabled=["slot", "name", "positions", "salary", "proj", "ceiling", "own"],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Metrics
                total_actual = edited_lineup["actual"].sum()
                
                st.markdown("### 📊 Lineup Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Projection", f"{lineup['proj']:.1f}")
                
                with col2:
                    st.metric("Ceiling", f"{lineup['ceiling']:.1f}")
                
                with col3:
                    if total_actual > 0:
                        diff = total_actual - lineup['proj']
                        st.metric("Actual", f"{total_actual:.1f}", delta=f"{diff:+.1f} vs proj")
                    else:
                        st.metric("Actual", "Enter scores →")
                
                # Export
                st.markdown("---")
                if st.button("💾 Export All Lineups"):
                    export_data = []
                    for i, lu in enumerate(lineups):
                        for _, p in lu["players"].iterrows():
                            export_data.append({
                                "Lineup": i + 1,
                                "Slot": p["slot"],
                                "Name": p["name"],
                                "Salary": p["salary"],
                                "Projection": p["proj"],
                                "Ownership": p["own"]
                            })
                    
                    export_df = pd.DataFrame(export_data)
                    csv = export_df.to_csv(index=False)
                    
                    st.download_button(
                        "Download CSV",
                        csv,
                        "lineups_export.csv",
                        "text/csv"
                    )

# ============================================================================
# TAB 2: CONTEST REVIEW
# ============================================================================

with tab2:
    st.header("📊 Contest Review & Analysis")
    
    st.markdown("""
    **Import actual contest results to:**
    1. See what won and why
    2. Analyze winning strategies
    3. Auto-populate actuals in your builder lineups
    """)
    
    contest_file = st.file_uploader(
        "Upload DraftKings Contest Export",
        type="csv",
        key="contest_upload",
        help="Download from DraftKings contest results"
    )
    
    if contest_file:
        df_contest = pd.read_csv(contest_file)
        st.session_state.contest_data = df_contest
        
        st.success(f"✅ Loaded contest with {len(df_contest):,} entries")
        
        # Parse player stats
        player_stats = df_contest[['Player', 'Roster Position', '%Drafted', 'FPTS']].copy()
        player_stats = player_stats.dropna(subset=['Player'])
        player_stats.columns = ['name', 'position', 'own', 'actual']
        player_stats['own'] = player_stats['own'].str.replace('%', '').astype(float)
        
        # Store actuals in session state
        for _, row in player_stats.iterrows():
            st.session_state.player_actuals[row['name']] = row['actual']
        
        # Contest overview
        st.markdown("---")
        st.subheader("📈 Contest Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Entries", f"{len(df_contest):,}")
        
        with col2:
            winning_score = df_contest['Points'].max()
            st.metric("Winning Score", f"{winning_score:.1f}")
        
        with col3:
            avg_score = df_contest[df_contest['Points'] > 0]['Points'].mean()
            st.metric("Avg Score", f"{avg_score:.1f}")
        
        with col4:
            st.metric("Unique Players", len(player_stats))
        
        st.info("💡 **Actuals populated!** Go to Builder tab → View lineup → Actual column now has contest results")
        
        # Player pool analysis
        st.markdown("---")
        st.subheader("🎯 Player Performance")
        
        player_stats['performance'] = player_stats['actual'].apply(lambda x:
            "🔥 Elite (40+)" if x >= 40 else
            "⭐ Great (30-40)" if x >= 30 else
            "✅ Good (20-30)" if x >= 20 else
            "⚠️ Okay (10-20)" if x >= 10 else
            "❌ Bust (<10)"
        )
        
        player_stats_display = player_stats.sort_values('actual', ascending=False)
        
        st.dataframe(
            player_stats_display,
            column_config={
                "name": "Player",
                "position": "Pos",
                "own": st.column_config.NumberColumn("Own%", format="%.1f%%"),
                "actual": st.column_config.NumberColumn("Points", format="%.1f"),
                "performance": "Performance",
            },
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Winning lineups analysis
        st.markdown("---")
        st.subheader("🏆 Winning Lineup Analysis")
        
        top_n = st.slider("Analyze top N lineups", 1, 20, 5)
        top_lineups = df_contest.head(top_n)
        
        def parse_lineup(lineup_str):
            if pd.isna(lineup_str) or lineup_str == '':
                return []
            
            players = []
            parts = lineup_str.split()
            
            i = 0
            while i < len(parts):
                if parts[i] in ['C', 'F', 'G', 'PF', 'PG', 'SF', 'SG', 'UTIL', 'QB', 'RB', 'WR', 'TE', 'FLEX', 'DST']:
                    pos = parts[i]
                    i += 1
                    name_parts = []
                    while i < len(parts) and parts[i] not in ['C', 'F', 'G', 'PF', 'PG', 'SF', 'SG', 'UTIL', 'QB', 'RB', 'WR', 'TE', 'FLEX', 'DST']:
                        name_parts.append(parts[i])
                        i += 1
                    if name_parts:
                        players.append({'position': pos, 'name': ' '.join(name_parts)})
                else:
                    i += 1
            
            return players
        
        for idx, row in top_lineups.iterrows():
            rank = row['Rank']
            points = row['Points']
            lineup_str = row['Lineup']
            
            with st.expander(f"🏆 Rank #{rank} - {points:.1f} points"):
                if pd.isna(lineup_str) or lineup_str == '':
                    st.warning("No lineup data")
                    continue
                
                lineup_players = parse_lineup(lineup_str)
                
                lineup_data = []
                for lp in lineup_players:
                    player_match = player_stats[player_stats['name'] == lp['name']]
                    if not player_match.empty:
                        player_info = player_match.iloc[0]
                        lineup_data.append({
                            'position': lp['position'],
                            'name': lp['name'],
                            'own': player_info['own'],
                            'actual': player_info['actual'],
                            'performance': player_info['performance']
                        })
                
                if lineup_data:
                    lineup_df = pd.DataFrame(lineup_data)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total", f"{lineup_df['actual'].sum():.1f}")
                    
                    with col2:
                        st.metric("Avg Own", f"{lineup_df['own'].mean():.1f}%")
                    
                    with col3:
                        elite = len(lineup_df[lineup_df['actual'] >= 40])
                        st.metric("Elite (40+)", elite)
                    
                    with col4:
                        contrarian = len(lineup_df[lineup_df['own'] < 15])
                        st.metric("Contrarian", contrarian)
                    
                    st.dataframe(
                        lineup_df,
                        column_config={
                            "position": "Slot",
                            "name": "Player",
                            "own": st.column_config.NumberColumn("Own%", format="%.1f%%"),
                            "actual": st.column_config.NumberColumn("Pts", format="%.1f"),
                            "performance": "Performance",
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Insights
                    insights = []
                    avg_own = lineup_df['own'].mean()
                    
                    if avg_own < 25:
                        insights.append(f"✅ Low ownership ({avg_own:.1f}%)")
                    if elite >= 3:
                        insights.append(f"✅ {elite} elite plays")
                    if contrarian >= 3:
                        insights.append(f"✅ {contrarian} contrarian plays")
                    
                    for insight in insights:
                        st.markdown(f"- {insight}")
        
        # Aggregate patterns
        st.markdown("---")
        st.subheader("📈 Winning Patterns")
        
        all_winners = []
        for idx, row in top_lineups.iterrows():
            lineup_str = row['Lineup']
            if not pd.isna(lineup_str) and lineup_str != '':
                lineup_players = parse_lineup(lineup_str)
                for lp in lineup_players:
                    player_match = player_stats[player_stats['name'] == lp['name']]
                    if not player_match.empty:
                        all_winners.append(player_match.iloc[0].to_dict())
        
        if all_winners:
            winners_df = pd.DataFrame(all_winners)
            
            player_counts = winners_df['name'].value_counts().head(10)
            usage_df = pd.DataFrame({
                'Player': player_counts.index,
                'Times Used': player_counts.values,
                'Usage %': (player_counts.values / top_n * 100).round(1)
            })
            
            for idx, row in usage_df.iterrows():
                player_data = player_stats[player_stats['name'] == row['Player']]
                if not player_data.empty:
                    usage_df.at[idx, 'Own%'] = player_data.iloc[0]['own']
                    usage_df.at[idx, 'Points'] = player_data.iloc[0]['actual']
            
            st.markdown("**Most Common in Winners:**")
            st.dataframe(
                usage_df,
                column_config={
                    "Player": "Player",
                    "Times Used": "Used",
                    "Usage %": st.column_config.NumberColumn("Usage", format="%.0f%%"),
                    "Own%": st.column_config.NumberColumn("Field Own", format="%.1f%%"),
                    "Points": st.column_config.NumberColumn("Points", format="%.1f"),
                },
                use_container_width=True,
                hide_index=True
            )
            
            avg_own_winners = winners_df['own'].mean()
            avg_pts_winners = winners_df['actual'].mean()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Avg Ownership in Winners", f"{avg_own_winners:.1f}%")
            
            with col2:
                st.metric("Avg Points per Player", f"{avg_pts_winners:.1f}")
            
            st.markdown("---")
            st.success(f"""
            **💡 Key Takeaways:**
            - Target {avg_own_winners:.1f}% average ownership
            - Aim for {avg_pts_winners:.1f}+ points per player
            - Core plays: {', '.join(usage_df.head(3)['Player'].tolist())}
            """)
    
    else:
        st.info("👆 Upload DraftKings contest export to begin")
        
        st.markdown("""
        ### 📋 How to Get Contest Export
        
        1. Go to your DraftKings contest
        2. Click "Contest Details"
        3. Scroll to "Download Results"
        4. Upload CSV here
        
        Once uploaded:
        - See winning strategies
        - Analyze top lineups
        - **Actuals auto-populate in Builder tab!**
        """)
