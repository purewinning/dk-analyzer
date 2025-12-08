"""
DFS Tools - Complete Suite
Contest Analyzer + Lineup Reviewer
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict

st.set_page_config(page_title="DFS Tools", layout="wide")

# Sidebar navigation
st.sidebar.title("🏆 DFS Tools")
page = st.sidebar.radio(
    "Choose Tool:",
    ["📊 Contest Analyzer", "🔍 Lineup Reviewer"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Contest Analyzer:**
Upload DraftKings contest export to see what won

**Lineup Reviewer:**  
Paste your lineup to analyze performance
""")

# ===========================================================================
# CONTEST ANALYZER
# ===========================================================================

if page == "📊 Contest Analyzer":
    st.title("📊 DraftKings Contest Analyzer")
    st.markdown("**Upload contest export to analyze winning strategies**")
    
    uploaded_file = st.file_uploader(
        "Upload DraftKings Contest CSV",
        type="csv",
        help="Export from DraftKings contest standings"
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Loaded contest with {len(df):,} entries")
        
        # Get player stats
        player_stats = df[['Player', 'Roster Position', '%Drafted', 'FPTS']].copy()
        player_stats = player_stats.dropna(subset=['Player'])
        player_stats.columns = ['name', 'position', 'own', 'actual']
        player_stats['own'] = player_stats['own'].str.replace('%', '').astype(float)
        
        # Overview
        st.markdown("---")
        st.header("📊 Contest Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Entries", f"{len(df):,}")
        
        with col2:
            winning_score = df['Points'].max()
            st.metric("Winning Score", f"{winning_score:.1f}")
        
        with col3:
            avg_score = df[df['Points'] > 0]['Points'].mean()
            st.metric("Avg Score", f"{avg_score:.1f}")
        
        with col4:
            st.metric("Unique Players", len(player_stats))
        
        # Player pool
        st.markdown("---")
        st.subheader("🎯 Player Pool Stats")
        
        player_stats['performance'] = player_stats['actual'].apply(lambda x:
            "🔥 Elite (40+)" if x >= 40 else
            "⭐ Great (30-40)" if x >= 30 else
            "✅ Good (20-30)" if x >= 20 else
            "⚠️ Below (10-20)" if x >= 10 else
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
        
        # Winning lineups
        st.markdown("---")
        st.header("🏆 Winning Lineup Analysis")
        
        top_n = st.slider("Analyze top N lineups", 1, 20, 5)
        top_lineups = df.head(top_n)
        
        def parse_lineup(lineup_str):
            if pd.isna(lineup_str) or lineup_str == '':
                return []
            
            players = []
            parts = lineup_str.split()
            
            i = 0
            while i < len(parts):
                if parts[i] in ['C', 'F', 'G', 'PF', 'PG', 'SF', 'SG', 'UTIL']:
                    pos = parts[i]
                    i += 1
                    name_parts = []
                    while i < len(parts) and parts[i] not in ['C', 'F', 'G', 'PF', 'PG', 'SF', 'SG', 'UTIL']:
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
            
            st.markdown(f"### Rank #{rank} - {points:.1f} points")
            
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
                    st.metric("Total Points", f"{lineup_df['actual'].sum():.1f}")
                
                with col2:
                    st.metric("Avg Own", f"{lineup_df['own'].mean():.1f}%")
                
                with col3:
                    elite_count = len(lineup_df[lineup_df['actual'] >= 40])
                    st.metric("Elite (40+)", elite_count)
                
                with col4:
                    contrarian = len(lineup_df[lineup_df['own'] < 15])
                    st.metric("Contrarian (<15%)", contrarian)
                
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
                
                st.markdown("---")
    
    else:
        st.info("👆 Upload a DraftKings contest export to begin")
        st.markdown("""
        ### 📋 How to Get Contest Export
        
        1. Go to your DraftKings contest
        2. Click "Contest Details"  
        3. Scroll down to "Download Results"
        4. Upload CSV here
        """)

# ===========================================================================
# LINEUP REVIEWER
# ===========================================================================

elif page == "🔍 Lineup Reviewer":
    st.title("🔍 Lineup Reviewer")
    st.markdown("**Analyze your lineup performance**")
    
    def calculate_player_performance(row: pd.Series) -> Dict:
        proj = row.get("proj", row.get("actual", 0) * 0.9)
        ceil = row.get("ceiling", proj * 1.4)
        actual = row.get("actual", 0)
        own = row.get("own", 15)
        
        vs_proj = actual - proj
        
        if actual >= ceil * 0.95:
            performance = "🔥 Smashed Ceiling"
        elif actual >= ceil * 0.8:
            performance = "⭐ Hit Ceiling"
        elif actual >= proj * 1.2:
            performance = "✅ Beat Projection"
        elif actual >= proj:
            performance = "✓ Met Projection"
        elif actual >= proj * 0.8:
            performance = "⚠️ Close Miss"
        else:
            performance = "❌ Bust"
        
        leverage_value = vs_proj * (100 - own) / 100
        
        if own < 10:
            ownership_tier = "🎯 Super Contrarian"
        elif own < 20:
            ownership_tier = "💎 Contrarian"
        elif own < 30:
            ownership_tier = "✅ Mid Ownership"
        elif own < 40:
            ownership_tier = "⚠️ Chalky"
        else:
            ownership_tier = "❌ Mega Chalk"
        
        return {
            "performance": performance,
            "vs_proj": vs_proj,
            "leverage_value": leverage_value,
            "ownership_tier": ownership_tier
        }
    
    st.subheader("📤 Upload Lineup")
    
    tab1, tab2 = st.tabs(["⚡ Quick Paste", "📄 Full CSV"])
    
    lineup_df = None
    
    with tab1:
        st.markdown("**Paste: Player | Position | Owned% | Points**")
        
        bulk_text = st.text_area(
            "Paste lineup (one player per line)",
            height=200,
            placeholder="""Luka Doncic    PG    51    73.25
Jalen Williams    SF    69    45.25
Kyle Filipowski    C    65    44.00""",
        )
        
        if bulk_text and st.button("📊 Parse & Analyze", type="primary"):
            lines = [line.strip() for line in bulk_text.split('\n') if line.strip()]
            parsed_data = []
            
            for line in lines:
                parts = None
                if '\t' in line:
                    parts = [p.strip() for p in line.split('\t') if p.strip()]
                elif ',' in line:
                    parts = [p.strip() for p in line.split(',') if p.strip()]
                else:
                    parts = [p.strip() for p in line.split() if p.strip()]
                
                if parts and len(parts) >= 4:
                    try:
                        name = ' '.join(parts[:-3])
                        position = parts[-3]
                        owned = parts[-2].replace('%', '')
                        points = parts[-1]
                        
                        parsed_data.append({
                            'name': name,
                            'positions': position,
                            'owned': float(owned),
                            'actual': float(points)
                        })
                    except:
                        continue
            
            if parsed_data:
                lineup_df = pd.DataFrame(parsed_data)
                lineup_df['salary'] = (lineup_df['actual'] * 150 + 3000).astype(int)
                lineup_df['proj'] = lineup_df['actual'] * 0.9
                lineup_df['ceiling'] = lineup_df['proj'] * 1.4
                lineup_df = lineup_df.rename(columns={'owned': 'own'})
                
                st.success(f"✅ Parsed {len(parsed_data)} players")
    
    with tab2:
        lineup_file = st.file_uploader("Upload CSV", type="csv", key="full_upload")
        
        if lineup_file:
            lineup_df = pd.read_csv(lineup_file)
            lineup_df.columns = lineup_df.columns.str.strip().str.lower()
            
            renames = {
                "player": "name",
                "position": "positions",
                "pos": "positions",
                "ownership": "own",
                "own%": "own",
            }
            lineup_df = lineup_df.rename(columns=renames)
    
    if lineup_df is not None and not lineup_df.empty:
        st.markdown("---")
        st.header("📊 Analysis")
        
        if 'positions' not in lineup_df.columns:
            lineup_df['positions'] = lineup_df.get('position', 'UNK')
        
        lineup_df['perf_metrics'] = lineup_df.apply(calculate_player_performance, axis=1)
        lineup_df['performance'] = lineup_df['perf_metrics'].apply(lambda x: x["performance"])
        lineup_df['vs_proj'] = lineup_df['perf_metrics'].apply(lambda x: x["vs_proj"])
        lineup_df['leverage_value'] = lineup_df['perf_metrics'].apply(lambda x: x["leverage_value"])
        lineup_df['ownership_tier'] = lineup_df['perf_metrics'].apply(lambda x: x["ownership_tier"])
        
        total_proj = lineup_df.get("proj", lineup_df['actual'] * 0.9).sum()
        total_ceil = lineup_df.get("ceiling", total_proj * 1.4).sum()
        total_actual = lineup_df["actual"].sum()
        total_own = lineup_df["own"].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Actual", f"{total_actual:.1f}")
        
        with col2:
            diff_proj = total_actual - total_proj
            st.metric("vs Projection", f"{diff_proj:+.1f}", 
                     delta=f"{diff_proj/total_proj*100:+.1f}%")
        
        with col3:
            diff_ceil = total_actual - total_ceil
            st.metric("vs Ceiling", f"{diff_ceil:+.1f}",
                     delta=f"{diff_ceil/total_ceil*100:+.1f}%")
        
        with col4:
            st.metric("Total Own", f"{total_own:.0f}%")
        
        st.subheader("👥 Player Performance")
        
        display_cols = ['name', 'positions', 'own', 'ownership_tier', 
                       'actual', 'vs_proj', 'performance', 'leverage_value']
        display_cols = [c for c in display_cols if c in lineup_df.columns]
        
        st.dataframe(
            lineup_df[display_cols],
            column_config={
                "name": "Player",
                "positions": "Pos",
                "own": st.column_config.NumberColumn("Own%", format="%.1f%%"),
                "ownership_tier": "Tier",
                "actual": st.column_config.NumberColumn("Actual", format="%.1f"),
                "vs_proj": st.column_config.NumberColumn("vs Proj", format="%+.1f"),
                "performance": "Performance",
                "leverage_value": st.column_config.NumberColumn("Leverage", format="%.1f"),
            },
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        st.subheader("💡 Key Insights")
        
        met_proj = len(lineup_df[lineup_df['actual'] >= lineup_df.get('proj', lineup_df['actual'] * 0.9)])
        hit_rate = met_proj / len(lineup_df) * 100
        
        ceiling_hits = len(lineup_df[lineup_df['actual'] >= lineup_df.get('ceiling', lineup_df.get('proj', lineup_df['actual'] * 0.9) * 1.4) * 0.8])
        
        low_owned = len(lineup_df[lineup_df['own'] < 15])
        avg_own = total_own / len(lineup_df)
        
        insights = []
        
        if avg_own < 25:
            insights.append(f"✅ Low ownership ({avg_own:.1f}%) - differentiated")
        
        if low_owned >= 3:
            insights.append(f"✅ {low_owned} contrarian plays (<15%)")
        
        if hit_rate >= 75:
            insights.append(f"✅ High hit rate ({hit_rate:.0f}%)")
        
        if ceiling_hits >= 3:
            insights.append(f"✅ {ceiling_hits} ceiling hits (80%+)")
        
        total_leverage = lineup_df['leverage_value'].sum()
        if total_leverage > 30:
            insights.append(f"✅ {total_leverage:.1f} leverage points gained")
        
        for insight in insights:
            st.markdown(f"- {insight}")
        
        if not insights:
            st.info("Standard performance - look for more contrarian plays next time")
        
        st.markdown("---")
        st.subheader("🎯 Replication Blueprint")
        
        st.markdown(f"""
        **Target for Next Time:**
        - Ownership: {avg_own:.1f}% average
        - Include {low_owned}+ players under 15%
        - Hit rate: {hit_rate:.0f}%+
        - Total leverage: {total_leverage:.1f}+ points
        """)
        
        top_leverage = lineup_df.nlargest(3, 'leverage_value')
        st.markdown("**Top Leverage Plays:**")
        for _, p in top_leverage.iterrows():
            st.markdown(f"- **{p['name']}**: {p['own']:.0f}% own, {p['vs_proj']:+.1f} vs proj = {p['leverage_value']:.1f} leverage")
    
    else:
        st.info("👆 Paste lineup data to begin")
        st.code("""Luka Doncic    PG    51    73.25
Jalen Williams    SF    69    45.25
Kyle Filipowski    C    65    44.00""")
