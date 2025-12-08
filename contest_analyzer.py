"""
DraftKings Contest Analyzer
Parse contest exports and analyze winning lineups
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List

st.set_page_config(page_title="DK Contest Analyzer", layout="wide")

st.title("🏆 DraftKings Contest Analyzer")
st.markdown("**Upload DraftKings contest export to analyze winners**")

# Upload file
uploaded_file = st.file_uploader(
    "Upload DraftKings Contest CSV",
    type="csv",
    help="Export from DraftKings contest standings"
)

if uploaded_file:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    st.success(f"✅ Loaded contest with {len(df)} entries")
    
    # Parse the data
    # Left side: Rank, EntryId, EntryName, Points, Lineup
    # Right side: Player, Position, %Drafted, FPTS
    
    # Get player stats from right side
    player_stats = df[['Player', 'Roster Position', '%Drafted', 'FPTS']].copy()
    player_stats = player_stats.dropna(subset=['Player'])
    player_stats.columns = ['name', 'position', 'own', 'actual']
    
    # Clean ownership (remove % sign)
    player_stats['own'] = player_stats['own'].str.replace('%', '').astype(float)
    
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
        total_players = len(player_stats)
        st.metric("Unique Players", total_players)
    
    # Show player pool
    st.markdown("---")
    st.subheader("🎯 Player Pool Stats")
    
    # Add performance tiers
    player_stats['performance'] = player_stats['actual'].apply(lambda x:
        "🔥 Elite (40+)" if x >= 40 else
        "⭐ Great (30-40)" if x >= 30 else
        "✅ Good (20-30)" if x >= 20 else
        "⚠️ Below (10-20)" if x >= 10 else
        "❌ Bust (<10)"
    )
    
    player_stats['ownership_tier'] = player_stats['own'].apply(lambda x:
        "❌ Mega Chalk (50%+)" if x >= 50 else
        "⚠️ Chalky (30-50%)" if x >= 30 else
        "✅ Mid (15-30%)" if x >= 15 else
        "💎 Contrarian (5-15%)" if x >= 5 else
        "🎯 Super Contrarian (<5%)"
    )
    
    # Sort by actual points
    player_stats_display = player_stats.sort_values('actual', ascending=False)
    
    st.dataframe(
        player_stats_display,
        column_config={
            "name": "Player",
            "position": "Position",
            "own": st.column_config.NumberColumn("Own%", format="%.2f%%"),
            "actual": st.column_config.NumberColumn("Points", format="%.2f"),
            "performance": "Performance",
            "ownership_tier": "Ownership Tier",
        },
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    # Analyze winning lineups
    st.markdown("---")
    st.header("🏆 Winning Lineup Analysis")
    
    # Select which place to analyze
    top_n = st.slider("Analyze top N lineups", 1, 20, 5)
    
    top_lineups = df.head(top_n)
    
    # Parse lineups
    def parse_lineup(lineup_str):
        """Parse lineup string into list of players."""
        if pd.isna(lineup_str) or lineup_str == '':
            return []
        
        players = []
        # Format: "C Player1 F Player2 G Player3 ..."
        parts = lineup_str.split()
        
        i = 0
        while i < len(parts):
            if parts[i] in ['C', 'F', 'G', 'PF', 'PG', 'SF', 'SG', 'UTIL']:
                # This is a position
                pos = parts[i]
                i += 1
                
                # Next parts are player name until we hit another position
                name_parts = []
                while i < len(parts) and parts[i] not in ['C', 'F', 'G', 'PF', 'PG', 'SF', 'SG', 'UTIL']:
                    name_parts.append(parts[i])
                    i += 1
                
                if name_parts:
                    player_name = ' '.join(name_parts)
                    players.append({'position': pos, 'name': player_name})
            else:
                i += 1
        
        return players
    
    # Analyze each top lineup
    for idx, row in top_lineups.iterrows():
        rank = row['Rank']
        points = row['Points']
        lineup_str = row['Lineup']
        
        st.markdown(f"### Rank #{rank} - {points:.1f} points")
        
        if pd.isna(lineup_str) or lineup_str == '':
            st.warning("No lineup data")
            continue
        
        lineup_players = parse_lineup(lineup_str)
        
        if not lineup_players:
            st.warning("Could not parse lineup")
            continue
        
        # Match with player stats
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
                    'performance': player_info['performance'],
                    'ownership_tier': player_info['ownership_tier']
                })
        
        if lineup_data:
            lineup_df = pd.DataFrame(lineup_data)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Points", f"{lineup_df['actual'].sum():.1f}")
            
            with col2:
                st.metric("Avg Ownership", f"{lineup_df['own'].mean():.1f}%")
            
            with col3:
                elite_count = len(lineup_df[lineup_df['actual'] >= 40])
                st.metric("Elite Plays (40+)", elite_count)
            
            with col4:
                contrarian_count = len(lineup_df[lineup_df['own'] < 15])
                st.metric("Contrarian (<15%)", contrarian_count)
            
            # Show lineup
            st.dataframe(
                lineup_df,
                column_config={
                    "position": "Slot",
                    "name": "Player",
                    "own": st.column_config.NumberColumn("Own%", format="%.2f%%"),
                    "actual": st.column_config.NumberColumn("Points", format="%.2f"),
                    "performance": "Performance",
                    "ownership_tier": "Tier",
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Key insights
            st.markdown("**🔑 Key Insights:**")
            
            insights = []
            
            avg_own = lineup_df['own'].mean()
            if avg_own < 25:
                insights.append(f"✅ Low ownership ({avg_own:.1f}%) - differentiated from field")
            
            if elite_count >= 3:
                insights.append(f"✅ {elite_count} elite performances (40+ points)")
            
            if contrarian_count >= 3:
                insights.append(f"✅ {contrarian_count} contrarian plays (<15% owned)")
            
            bust_count = len(lineup_df[lineup_df['actual'] < 10])
            if bust_count == 0:
                insights.append("✅ No busts - all players produced")
            
            for insight in insights:
                st.markdown(f"- {insight}")
            
            st.markdown("---")
    
    # Aggregate analysis
    st.header("📈 Aggregate Winner Analysis")
    
    st.markdown(f"**Analyzing top {top_n} lineups:**")
    
    # Collect all players from top lineups
    all_top_players = []
    for idx, row in top_lineups.iterrows():
        lineup_str = row['Lineup']
        if not pd.isna(lineup_str) and lineup_str != '':
            lineup_players = parse_lineup(lineup_str)
            for lp in lineup_players:
                player_match = player_stats[player_stats['name'] == lp['name']]
                if not player_match.empty:
                    all_top_players.append(player_match.iloc[0].to_dict())
    
    if all_top_players:
        top_players_df = pd.DataFrame(all_top_players)
        
        # Most common players in winning lineups
        st.subheader("🌟 Most Used Players in Winners")
        
        player_counts = top_players_df['name'].value_counts().head(10)
        usage_df = pd.DataFrame({
            'Player': player_counts.index,
            'Times Used': player_counts.values,
            'Usage %': (player_counts.values / top_n * 100).round(1)
        })
        
        # Add their ownership and points
        for idx, row in usage_df.iterrows():
            player_data = player_stats[player_stats['name'] == row['Player']]
            if not player_data.empty:
                usage_df.at[idx, 'Own%'] = player_data.iloc[0]['own']
                usage_df.at[idx, 'Points'] = player_data.iloc[0]['actual']
        
        st.dataframe(
            usage_df,
            column_config={
                "Player": "Player",
                "Times Used": st.column_config.NumberColumn("Used", format="%d"),
                "Usage %": st.column_config.NumberColumn("Usage%", format="%.1f%%"),
                "Own%": st.column_config.NumberColumn("Field Own%", format="%.1f%%"),
                "Points": st.column_config.NumberColumn("Points", format="%.1f"),
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Key patterns
        st.markdown("---")
        st.subheader("🎯 Winning Patterns")
        
        avg_own_winners = top_players_df['own'].mean()
        avg_points_winners = top_players_df['actual'].mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Avg Ownership in Winners", f"{avg_own_winners:.1f}%")
            st.metric("Avg Points per Player", f"{avg_points_winners:.1f}")
        
        with col2:
            contrarian_pct = (top_players_df['own'] < 15).sum() / len(top_players_df) * 100
            elite_pct = (top_players_df['actual'] >= 40).sum() / len(top_players_df) * 100
            
            st.metric("% Contrarian Plays", f"{contrarian_pct:.1f}%")
            st.metric("% Elite Performances", f"{elite_pct:.1f}%")
        
        st.markdown("---")
        st.markdown("### 💡 Replication Strategy")
        st.markdown(f"""
        Based on top {top_n} lineups:
        
        **1. Ownership Targets:**
        - Average: {avg_own_winners:.1f}% per player
        - Include {contrarian_pct:.0f}% contrarian plays (<15%)
        - Avoid mega-chalk (50%+)
        
        **2. Performance Targets:**
        - Average: {avg_points_winners:.1f} points per player
        - Target {elite_pct:.0f}% elite performances (40+)
        - Minimize busts (<10 points)
        
        **3. Core Plays to Consider:**
        """)
        
        # Top 5 most used
        top_5_used = usage_df.head(5)
        for _, p in top_5_used.iterrows():
            st.markdown(f"- **{p['Player']}**: {p['Usage %']:.0f}% usage in winners, {p['Own%']:.1f}% owned, {p['Points']:.1f} pts")
    
    # Export analysis
    st.markdown("---")
    if st.button("💾 Export Analysis"):
        report = f"""# DraftKings Contest Analysis

## Contest Overview
- Total Entries: {len(df):,}
- Winning Score: {winning_score:.1f}
- Average Score: {avg_score:.1f}

## Top {top_n} Lineups Analysis
- Average Ownership: {avg_own_winners:.1f}%
- Average Points per Player: {avg_points_winners:.1f}
- % Contrarian Plays: {contrarian_pct:.1f}%
- % Elite Performances: {elite_pct:.1f}%

## Most Used Players in Winners
{usage_df.to_markdown(index=False)}

## Replication Strategy
1. Target {avg_own_winners:.1f}% average ownership
2. Include {contrarian_pct:.0f}% contrarian plays
3. Aim for {avg_points_winners:.1f}+ points per player
4. Focus on these core plays: {', '.join(top_5_used['Player'].tolist())}

---
Generated by DK Contest Analyzer
"""
        
        st.download_button(
            "Download Report",
            report,
            "contest_analysis.md",
            "text/markdown"
        )

else:
    st.info("👆 Upload a DraftKings contest export CSV to begin")
    
    st.markdown("---")
    st.markdown("### 📋 How to Export from DraftKings")
    
    st.markdown("""
    1. Go to your contest on DraftKings
    2. Click "Contest Details"
    3. Scroll down to "Download Results"
    4. Upload the CSV here
    
    The CSV should have columns like:
    - Rank, EntryId, EntryName, Points, Lineup (left side)
    - Player, Roster Position, %Drafted, FPTS (right side)
    """)
    
    st.markdown("---")
    st.success("💡 **Pro Tip:** This tool shows you exactly what worked in the contest - use these patterns for your next build!")
