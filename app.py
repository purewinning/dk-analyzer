"""
DFS Elite Tools - Streamlit Cloud Version
Access Lineup Reviewer directly
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict

st.set_page_config(page_title="DFS Lineup Reviewer", layout="wide")

st.title("🔍 DFS Lineup Reviewer")
st.markdown("**Analyze winning lineups to understand WHY they won**")

# ========================================================================
# REVIEWER CODE
# ========================================================================

def calculate_player_performance(row: pd.Series) -> Dict:
    """Analyze individual player performance."""
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

st.subheader("📤 Upload Winning Lineup")
st.markdown("**Two ways to upload:**")

tab1, tab2 = st.tabs(["📄 Full CSV", "⚡ Quick Paste"])

lineup_df = None

with tab1:
    st.markdown("**Upload CSV with all details**")
    lineup_file = st.file_uploader(
        "Upload lineup CSV",
        type="csv",
        help="name, position, salary, proj, ceiling, own, actual",
        key="full_upload"
    )
    
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

with tab2:
    st.markdown("**Paste: Player | Position | Owned% | Points**")
    st.caption("📋 Copy directly from DraftKings contest results!")
    
    bulk_text = st.text_area(
        "Paste lineup (one player per line)",
        height=200,
        placeholder="""LeBron James    SF    22    58.5
Stephen Curry    PG    28    52.3
Kevin Durant    SF    25    71.2
Giannis Antetokounmpo    PF    18    65.1
Joel Embiid    C    24    48.2
Damian Lillard    PG    15    42.8
Jayson Tatum    SF    30    55.6
Bam Adebayo    C    12    38.9""",
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
        else:
            st.error("❌ Could not parse data. Make sure format is: Player | Position | Own% | Points")

if lineup_df is not None and not lineup_df.empty:
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    # Ensure columns
    if 'positions' not in lineup_df.columns:
        lineup_df['positions'] = lineup_df.get('position', 'UNK')
    
    # Calculate metrics
    lineup_df['perf_metrics'] = lineup_df.apply(calculate_player_performance, axis=1)
    lineup_df['performance'] = lineup_df['perf_metrics'].apply(lambda x: x["performance"])
    lineup_df['vs_proj'] = lineup_df['perf_metrics'].apply(lambda x: x["vs_proj"])
    lineup_df['leverage_value'] = lineup_df['perf_metrics'].apply(lambda x: x["leverage_value"])
    lineup_df['ownership_tier'] = lineup_df['perf_metrics'].apply(lambda x: x["ownership_tier"])
    
    # Top metrics
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
    
    # Player breakdown
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
    
    # Key insights
    st.markdown("---")
    st.subheader("💡 Key Insights: Why This Lineup Won")
    
    met_proj = len(lineup_df[lineup_df['actual'] >= lineup_df.get('proj', lineup_df['actual'] * 0.9)])
    hit_rate = met_proj / len(lineup_df) * 100
    
    ceiling_hits = len(lineup_df[lineup_df['actual'] >= lineup_df.get('ceiling', lineup_df.get('proj', lineup_df['actual'] * 0.9) * 1.4) * 0.8])
    
    low_owned = len(lineup_df[lineup_df['own'] < 15])
    avg_own = total_own / len(lineup_df)
    
    insights = []
    
    if avg_own < 25:
        insights.append(f"✅ **Low Ownership Edge**: {avg_own:.1f}% avg ownership - differentiated from field")
    
    if low_owned >= 3:
        insights.append(f"✅ **Contrarian Plays**: {low_owned} players under 15% ownership")
    
    if hit_rate >= 75:
        insights.append(f"✅ **High Hit Rate**: {hit_rate:.0f}% of players hit projection")
    
    if ceiling_hits >= 3:
        insights.append(f"✅ **Ceiling Games**: {ceiling_hits} players hit ceiling (80%+)")
    
    total_leverage = lineup_df['leverage_value'].sum()
    if total_leverage > 30:
        insights.append(f"✅ **Leverage Value**: {total_leverage:.1f} points gained vs field")
    
    for insight in insights:
        st.markdown(insight)
    
    if not insights:
        st.info("Standard performance. Look for more contrarian plays next time.")
    
    # Replication blueprint
    st.markdown("---")
    st.subheader("🎯 Replication Blueprint")
    
    st.markdown(f"""
    ### How to Replicate This Success
    
    **1. Ownership Strategy**
    - Target: {avg_own:.1f}% average ownership
    - Include: {low_owned}+ players under 15%
    - Avoid: Mega-chalk (40%+)
    
    **2. Performance Targets**
    - Hit Rate: {hit_rate:.0f}%+ (players meeting projection)
    - Total Leverage: {total_leverage:.1f}+ points
    
    **3. Top Leverage Plays**
    """)
    
    top_leverage = lineup_df.nlargest(3, 'leverage_value')
    for _, p in top_leverage.iterrows():
        st.markdown(f"- **{p['name']}** ({p['positions']}): {p['own']:.0f}% owned, {p['vs_proj']:+.1f} vs proj = **{p['leverage_value']:.1f} leverage points**")
    
    st.markdown("""
    **Action Steps for Next Slate:**
    - Find similar low-owned players in elite games
    - Target players with high ceiling potential  
    - Balance studs with contrarian value
    - Keep total ownership under 200%
    """)

else:
    st.info("👆 Upload lineup data or paste results to begin analysis")
    
    st.markdown("---")
    st.markdown("### 📋 Example Formats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Full CSV Format:**")
        st.code("""name,position,salary,proj,ceiling,own,actual
LeBron James,SF,9500,48.2,67.5,22,58.5
Stephen Curry,PG,9200,45.1,63.1,28,52.3""")
    
    with col2:
        st.markdown("**Quick Paste Format:**")
        st.code("""LeBron James    SF    22    58.5
Stephen Curry    PG    28    52.3
Kevin Durant    SF    25    71.2""")
    
    st.markdown("---")
    st.success("💡 **Pro Tip:** Copy lineup results directly from DraftKings → Paste in Quick Paste tab → Click Parse & Analyze!")
