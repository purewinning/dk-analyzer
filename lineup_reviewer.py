"""
DFS Lineup Reviewer v1.0
Analyze winning lineups to understand WHY they won and HOW to replicate success
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import io

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_player_performance(row: pd.Series) -> Dict:
    """Analyze individual player performance."""
    proj = row["proj"]
    ceil = row["ceiling"]
    actual = row["actual"]
    own = row["own"]
    
    # Performance metrics
    vs_proj = actual - proj
    vs_ceil = actual - ceil
    proj_pct = (actual / proj * 100) if proj > 0 else 0
    ceil_pct = (actual / ceil * 100) if ceil > 0 else 0
    
    # Classification
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
    
    # Leverage value
    leverage_value = vs_proj * (100 - own) / 100  # Points gained weighted by low ownership
    
    # Tournament value (considering ownership)
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
        "vs_ceil": vs_ceil,
        "proj_pct": proj_pct,
        "ceil_pct": ceil_pct,
        "leverage_value": leverage_value,
        "ownership_tier": ownership_tier
    }

def analyze_lineup_construction(df: pd.DataFrame) -> Dict:
    """Analyze why the lineup construction worked."""
    analysis = {}
    
    # Game environment analysis
    game_counts = df["game_env"].value_counts()
    analysis["game_strategy"] = {
        "breakdown": game_counts.to_dict(),
        "elite_count": game_counts.get("🔥 Elite", 0),
        "great_count": game_counts.get("⭐ Great", 0),
        "avg_game_total": df["game_total"].mean()
    }
    
    # Stacking analysis
    team_counts = df["team"].value_counts()
    max_stack = team_counts.max()
    stacked_team = team_counts.idxmax()
    
    analysis["stacking"] = {
        "max_stack_size": max_stack,
        "stacked_team": stacked_team,
        "team_distribution": team_counts.to_dict()
    }
    
    # Ownership analysis
    avg_own = df["own"].mean()
    low_own_count = len(df[df["own"] < 15])
    chalk_count = len(df[df["own"] >= 30])
    
    analysis["ownership"] = {
        "avg_ownership": avg_own,
        "low_owned_plays": low_own_count,
        "chalk_plays": chalk_count,
        "total_ownership": df["own"].sum()
    }
    
    # Salary analysis
    total_sal = df["salary"].sum()
    analysis["salary"] = {
        "total_used": total_sal,
        "remaining": 50000 - total_sal,
        "avg_salary": df["salary"].mean(),
        "studs": len(df[df["salary"] >= 8500]),
        "value": len(df[df["salary"] <= 5000])
    }
    
    # Performance analysis
    df["perf"] = df.apply(calculate_player_performance, axis=1)
    
    smashes = len(df[df["actual"] >= df["ceiling"] * 0.95])
    hits = len(df[df["actual"] >= df["ceiling"] * 0.8])
    met_proj = len(df[df["actual"] >= df["proj"]])
    busts = len(df[df["actual"] < df["proj"] * 0.8])
    
    analysis["performance"] = {
        "ceiling_smashes": smashes,
        "ceiling_hits": hits,
        "met_projection": met_proj,
        "busts": busts,
        "hit_rate": met_proj / len(df) * 100,
        "ceiling_rate": hits / len(df) * 100
    }
    
    return analysis

def generate_key_insights(df: pd.DataFrame, analysis: Dict) -> List[str]:
    """Generate key insights about why the lineup won."""
    insights = []
    
    # Game environment insight
    elite_pct = analysis["game_strategy"]["elite_count"] / len(df) * 100
    if elite_pct >= 50:
        insights.append(f"✅ **Elite Game Focus**: {elite_pct:.0f}% of players from 🔥 Elite games (240+ total) - high-scoring environments drove success")
    
    # Stacking insight
    if analysis["stacking"]["max_stack_size"] >= 3:
        team = analysis["stacking"]["stacked_team"]
        insights.append(f"✅ **Correlation Play**: {analysis['stacking']['max_stack_size']} players from {team} - correlated outcomes amplified upside")
    
    # Ownership insight
    if analysis["ownership"]["avg_ownership"] < 180:
        insights.append(f"✅ **Low Ownership Edge**: {analysis['ownership']['avg_ownership']:.0f}% total ownership - differentiated from field")
    
    if analysis["ownership"]["low_owned_plays"] >= 3:
        insights.append(f"✅ **Contrarian Plays**: {analysis['ownership']['low_owned_plays']} players under 15% ownership - leverage spots hit")
    
    # Performance insight
    if analysis["performance"]["ceiling_smashes"] >= 2:
        insights.append(f"✅ **Ceiling Games**: {analysis['performance']['ceiling_smashes']} players smashed ceiling (95%+) - captured elite outcomes")
    
    if analysis["performance"]["busts"] <= 1:
        insights.append(f"✅ **Limited Busts**: Only {analysis['performance']['busts']} player(s) busted - stable floor")
    
    # Leverage insight
    df["leverage_value"] = (df["actual"] - df["proj"]) * (100 - df["own"]) / 100
    total_leverage_value = df["leverage_value"].sum()
    if total_leverage_value > 50:
        insights.append(f"✅ **Leverage Value**: {total_leverage_value:.1f} points gained vs field through low-owned outperformance")
    
    # Salary efficiency
    if analysis["salary"]["remaining"] <= 500:
        insights.append(f"✅ **Salary Optimization**: Used ${analysis['salary']['total_used']:,} (${analysis['salary']['remaining']} left) - maximized roster value")
    
    return insights

def generate_replication_blueprint(df: pd.DataFrame, analysis: Dict) -> List[str]:
    """Generate actionable steps to replicate this success."""
    blueprint = []
    
    blueprint.append("## 🎯 Replication Blueprint")
    blueprint.append("")
    
    # 1. Game Selection
    elite_pct = analysis["game_strategy"]["elite_count"] / len(df) * 100
    if elite_pct >= 50:
        blueprint.append("### 1. Game Environment Selection")
        blueprint.append(f"- **Target**: {elite_pct:.0f}%+ of players from 🔥 Elite games (240+ NBA, 85+ NFL)")
        blueprint.append(f"- **Avg Game Total**: {analysis['game_strategy']['avg_game_total']:.0f}")
        blueprint.append("- **Action**: Filter to only elite/great games before building")
        blueprint.append("")
    
    # 2. Stacking
    if analysis["stacking"]["max_stack_size"] >= 3:
        blueprint.append("### 2. Stacking Strategy")
        blueprint.append(f"- **Target**: {analysis['stacking']['max_stack_size']} player stacks from high-scoring games")
        blueprint.append(f"- **Example**: {analysis['stacking']['stacked_team']} stack worked")
        blueprint.append("- **Action**: Set correlation to 0.7-0.8 for heavy stacking")
        blueprint.append("")
    
    # 3. Ownership
    if analysis["ownership"]["avg_ownership"] < 200:
        blueprint.append("### 3. Ownership Targeting")
        blueprint.append(f"- **Target**: {analysis['ownership']['avg_ownership']:.0f}% total ownership (under 200%)")
        blueprint.append(f"- **Contrarian Plays**: {analysis['ownership']['low_owned_plays']}+ players under 15%")
        blueprint.append("- **Action**: Filter to elite-edge + strong-edge, avoid mega-chalk")
        blueprint.append("")
    
    # 4. Salary
    blueprint.append("### 4. Salary Construction")
    blueprint.append(f"- **Studs**: {analysis['salary']['studs']} players at $8,500+")
    blueprint.append(f"- **Value**: {analysis['salary']['value']} players under $5,000")
    blueprint.append(f"- **Remaining**: ${analysis['salary']['remaining']} (use most of cap)")
    blueprint.append("- **Action**: Balance stars with value plays")
    blueprint.append("")
    
    # 5. Key positions
    ceiling_players = df[df["actual"] >= df["ceiling"] * 0.8].sort_values("actual", ascending=False)
    if len(ceiling_players) > 0:
        blueprint.append("### 5. Key Position Priorities")
        blueprint.append("**Ceiling hits (replicate these):**")
        for _, p in ceiling_players.head(3).iterrows():
            blueprint.append(f"- {p['positions']} in 🔥 elite game, <20% owned, high ceiling projection")
        blueprint.append("")
    
    return "\n".join(blueprint)

def find_similar_opportunities(df: pd.DataFrame, current_slate_df: pd.DataFrame = None) -> str:
    """Find similar opportunities in current slate (if provided)."""
    if current_slate_df is None or current_slate_df.empty:
        return "**Upload current slate CSV to find similar opportunities**"
    
    # Criteria from winning lineup
    avg_own = df["own"].mean()
    avg_game = df["game_total"].mean()
    
    # Find similar players in current slate
    similar = current_slate_df[
        (current_slate_df["own"] < avg_own + 10) &
        (current_slate_df["game_total"] >= avg_game - 20)
    ].copy()
    
    if similar.empty:
        return "**No similar opportunities found in current slate**"
    
    similar = similar.nlargest(10, "gpp_score")
    
    output = "### 🎯 Similar Opportunities in Current Slate\n\n"
    output += "Based on winning lineup criteria:\n\n"
    
    output += "| Player | Pos | Own% | Game | Ceiling | Value |\n"
    output += "|--------|-----|------|------|---------|-------|\n"
    
    for _, p in similar.iterrows():
        output += f"| {p['name']} | {p['positions']} | {p['own']:.0f}% | {p['game_env']} | {p['ceiling']:.1f} | {p['value']:.2f} |\n"
    
    return output

# ============================================================================
# UI
# ============================================================================

st.set_page_config(page_title="DFS Lineup Reviewer", layout="wide")

st.title("🔍 DFS Lineup Reviewer")
st.markdown("**Analyze winning lineups to understand WHY they won and HOW to replicate success**")

st.markdown("---")

# Upload winning lineup
st.subheader("📤 Upload Winning Lineup")

st.markdown("**Two ways to upload:**")

tab1, tab2 = st.tabs(["📄 Full CSV (Detailed)", "⚡ Quick Upload (Bulk)"])

with tab1:
    st.markdown("**Upload full lineup CSV with all details**")
    lineup_file = st.file_uploader(
        "Upload lineup CSV",
        type="csv",
        help="Must include: name, position, salary, proj, ceiling, own, actual",
        key="full_upload"
    )
    
    with st.expander("📋 See required format"):
        example_df = pd.DataFrame({
            "name": ["LeBron James", "Stephen Curry", "Kevin Durant"],
            "position": ["SF", "PG", "SF"],
            "salary": [9500, 9200, 9800],
            "proj": [48.2, 45.1, 50.3],
            "ceiling": [67.5, 63.1, 70.4],
            "own": [22, 28, 25],
            "actual": [58.5, 52.3, 71.2],
        })
        st.dataframe(example_df, use_container_width=True)

with tab2:
    st.markdown("**Quick upload: Just paste Player, Position, Owned, Points**")
    st.info("💡 Copy from DraftKings contest results or any simple format")
    
    bulk_text = st.text_area(
        "Paste lineup data (one player per line)",
        height=200,
        placeholder="""LeBron James    SF    22%    58.5
Stephen Curry    PG    28%    52.3
Kevin Durant    SF    25%    71.2
Giannis Antetokounmpo    PF    18%    65.1
Joel Embiid    C    24%    48.2
Damian Lillard    PG    15%    42.8
Jayson Tatum    SF    30%    55.6
Bam Adebayo    C    12%    38.9""",
        help="Tab or space separated: Player | Position | Owned% | Points"
    )
    
    use_bulk = st.checkbox("Use bulk upload data", value=False)
    
    if use_bulk and bulk_text:
        # Parse bulk text
        lines = [line.strip() for line in bulk_text.split('\n') if line.strip()]
        
        parsed_data = []
        for line in lines:
            # Try multiple separators (tab, multiple spaces, comma)
            parts = None
            if '\t' in line:
                parts = [p.strip() for p in line.split('\t') if p.strip()]
            elif ',' in line:
                parts = [p.strip() for p in line.split(',') if p.strip()]
            else:
                # Multiple spaces
                parts = [p.strip() for p in line.split() if p.strip()]
            
            if parts and len(parts) >= 4:
                # Last two should be numbers (owned, points)
                try:
                    name = ' '.join(parts[:-3])  # Everything except last 3
                    position = parts[-3]  # Position
                    owned = parts[-2].replace('%', '')  # Ownership
                    points = parts[-1]  # Points
                    
                    parsed_data.append({
                        'name': name,
                        'position': position,
                        'owned': float(owned),
                        'points': float(points)
                    })
                except:
                    continue
        
        if parsed_data:
            lineup_file = pd.DataFrame(parsed_data)
            
            st.success(f"✅ Parsed {len(parsed_data)} players")
            st.dataframe(lineup_file, use_container_width=True)
            
            # Estimate missing data
            st.info("📊 Estimating missing data (salary, proj, ceiling)...")
            
            # Estimate salary based on points (rough heuristic)
            lineup_file['salary'] = (lineup_file['points'] * 150 + 3000).astype(int)
            
            # Estimate projection as 90% of actual (they beat projection)
            lineup_file['proj'] = lineup_file['points'] * 0.9
            
            # Estimate ceiling as 1.4x projection
            lineup_file['ceiling'] = lineup_file['proj'] * 1.4
            
            # Rename for consistency
            lineup_file = lineup_file.rename(columns={'owned': 'own', 'points': 'actual'})
            
            # Add game environment placeholders
            lineup_file['team'] = 'UNK'
            lineup_file['game_env'] = '✅ Good'
            lineup_file['game_total'] = 220
            
            st.warning("⚠️ Note: Salary, projection, and ceiling are estimated. Analysis will focus on ownership and actual performance.")
        else:
            st.error("❌ Could not parse data. Make sure format is: Player | Position | Owned% | Points")
            lineup_file = None
    else:
        lineup_file = None

col1_spacer, col2_spacer = st.columns(2)

with col2_spacer:
    if not use_bulk:
        current_slate = st.file_uploader(
            "Upload current slate CSV (optional)",
            type="csv",
            help="To find similar opportunities in upcoming slate",
            key="slate_upload"
        )
    else:
        current_slate = None

if lineup_file is not None:
    # Check if it's already a DataFrame (from bulk upload) or needs to be loaded
    if isinstance(lineup_file, pd.DataFrame):
        lineup_df = lineup_file.copy()
    else:
        # Load lineup from file
        lineup_df = pd.read_csv(lineup_file)
        lineup_df.columns = lineup_df.columns.str.strip().str.lower()
        
        # Normalize columns
        renames = {
            "player": "name",
            "position": "positions",
            "pos": "positions",
            "ownership": "own",
            "own%": "own",
            "projection": "proj",
            "fpts": "proj",
        }
        lineup_df = lineup_df.rename(columns=renames)
    
    # Ensure positions column exists
    if "positions" not in lineup_df.columns and "position" in lineup_df.columns:
        lineup_df["positions"] = lineup_df["position"]
    
    # Required columns
    required = ["name", "positions", "salary", "proj", "ceiling", "own", "actual"]
    missing = [c for c in required if c not in lineup_df.columns]
    
    if missing:
        st.error(f"❌ Missing columns: {missing}")
        st.info("CSV must have: name, position, salary, proj, ceiling, own, actual")
        st.stop()
    
    # Clean data
    lineup_df["salary"] = pd.to_numeric(lineup_df["salary"], errors="coerce")
    lineup_df["proj"] = pd.to_numeric(lineup_df["proj"], errors="coerce")
    lineup_df["ceiling"] = pd.to_numeric(lineup_df["ceiling"], errors="coerce")
    lineup_df["own"] = pd.to_numeric(lineup_df["own"], errors="coerce")
    lineup_df["actual"] = pd.to_numeric(lineup_df["actual"], errors="coerce")
    
    # Add game info if not present
    if "game_env" not in lineup_df.columns:
        lineup_df["game_env"] = "✅ Good"
    if "game_total" not in lineup_df.columns:
        lineup_df["game_total"] = 220
    if "team" not in lineup_df.columns:
        lineup_df["team"] = "UNK"
    
    # Calculate performance metrics for each player
    lineup_df["perf_metrics"] = lineup_df.apply(calculate_player_performance, axis=1)
    lineup_df["performance"] = lineup_df["perf_metrics"].apply(lambda x: x["performance"])
    lineup_df["vs_proj"] = lineup_df["perf_metrics"].apply(lambda x: x["vs_proj"])
    lineup_df["vs_ceil"] = lineup_df["perf_metrics"].apply(lambda x: x["vs_ceil"])
    lineup_df["leverage_value"] = lineup_df["perf_metrics"].apply(lambda x: x["leverage_value"])
    lineup_df["ownership_tier"] = lineup_df["perf_metrics"].apply(lambda x: x["ownership_tier"])
    
    # Analyze construction
    analysis = analyze_lineup_construction(lineup_df)
    insights = generate_key_insights(lineup_df, analysis)
    
    # ========================================================================
    # DISPLAY ANALYSIS
    # ========================================================================
    
    st.markdown("---")
    st.header("📊 Lineup Performance Overview")
    
    # Top metrics
    total_proj = lineup_df["proj"].sum()
    total_ceil = lineup_df["ceiling"].sum()
    total_actual = lineup_df["actual"].sum()
    total_own = lineup_df["own"].sum()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Actual", f"{total_actual:.1f}")
    
    with col2:
        diff_proj = total_actual - total_proj
        st.metric("vs Projection", f"{diff_proj:+.1f}", delta=f"{diff_proj/total_proj*100:+.1f}%")
    
    with col3:
        diff_ceil = total_actual - total_ceil
        st.metric("vs Ceiling", f"{diff_ceil:+.1f}", delta=f"{diff_ceil/total_ceil*100:+.1f}%")
    
    with col4:
        st.metric("Total Ownership", f"{total_own:.0f}%")
    
    with col5:
        st.metric("Hit Rate", f"{analysis['performance']['hit_rate']:.0f}%")
    
    # Player breakdown
    st.markdown("---")
    st.subheader("👥 Player Performance Breakdown")
    
    display_df = lineup_df[[
        "name", "positions", "salary", "own", "ownership_tier",
        "proj", "ceiling", "actual", "vs_proj", "performance", "leverage_value"
    ]].copy()
    
    st.dataframe(
        display_df,
        column_config={
            "name": st.column_config.TextColumn("Player", width="medium"),
            "positions": st.column_config.TextColumn("Pos", width="small"),
            "salary": st.column_config.NumberColumn("Salary", format="$%d"),
            "own": st.column_config.NumberColumn("Own%", format="%.1f%%"),
            "ownership_tier": st.column_config.TextColumn("Ownership Tier", width="medium"),
            "proj": st.column_config.NumberColumn("Proj", format="%.1f"),
            "ceiling": st.column_config.NumberColumn("Ceil", format="%.1f"),
            "actual": st.column_config.NumberColumn("Actual", format="%.1f"),
            "vs_proj": st.column_config.NumberColumn("vs Proj", format="%+.1f"),
            "performance": st.column_config.TextColumn("Performance", width="medium"),
            "leverage_value": st.column_config.NumberColumn("Leverage Value", format="%.1f", 
                help="Points gained vs field (weighted by ownership)")
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Key insights
    st.markdown("---")
    st.header("💡 Key Insights: Why This Lineup Won")
    
    for insight in insights:
        st.markdown(insight)
    
    # Construction analysis
    st.markdown("---")
    st.header("🏗️ Construction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎮 Game Environment")
        game_breakdown = analysis["game_strategy"]["breakdown"]
        for env, count in game_breakdown.items():
            st.metric(env, f"{count} players ({count/len(lineup_df)*100:.0f}%)")
        st.metric("Avg Game Total", f"{analysis['game_strategy']['avg_game_total']:.0f}")
    
    with col2:
        st.markdown("### 📊 Ownership Distribution")
        st.metric("Avg Ownership", f"{analysis['ownership']['avg_ownership']:.1f}%")
        st.metric("Low-Owned (<15%)", f"{analysis['ownership']['low_owned_plays']} players")
        st.metric("Chalk (30%+)", f"{analysis['ownership']['chalk_plays']} players")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤝 Stacking")
        st.metric("Max Stack Size", f"{analysis['stacking']['max_stack_size']} players")
        st.metric("Stacked Team", analysis['stacking']['stacked_team'])
        
        with st.expander("Team Distribution"):
            team_dist = analysis['stacking']['team_distribution']
            for team, count in sorted(team_dist.items(), key=lambda x: x[1], reverse=True):
                st.write(f"**{team}**: {count} players")
    
    with col2:
        st.markdown("### 💰 Salary Construction")
        st.metric("Total Used", f"${analysis['salary']['total_used']:,}")
        st.metric("Remaining", f"${analysis['salary']['remaining']}")
        st.metric("Studs ($8,500+)", f"{analysis['salary']['studs']} players")
        st.metric("Value (<$5,000)", f"{analysis['salary']['value']} players")
    
    # Performance breakdown
    st.markdown("---")
    st.header("🎯 Performance Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔥 Ceiling Smashes", f"{analysis['performance']['ceiling_smashes']} players",
                 help="Scored 95%+ of ceiling")
    
    with col2:
        st.metric("⭐ Ceiling Hits", f"{analysis['performance']['ceiling_hits']} players",
                 help="Scored 80%+ of ceiling")
    
    with col3:
        st.metric("✅ Met Projection", f"{analysis['performance']['met_projection']} players",
                 help="Scored >= projection")
    
    with col4:
        st.metric("❌ Busts", f"{analysis['performance']['busts']} players",
                 help="Scored <80% of projection")
    
    # Leverage analysis
    st.markdown("---")
    st.header("📈 Leverage Analysis")
    
    leverage_df = lineup_df.nlargest(5, "leverage_value")[[
        "name", "own", "proj", "actual", "vs_proj", "leverage_value"
    ]]
    
    st.markdown("**Top 5 Leverage Plays:**")
    st.dataframe(
        leverage_df,
        column_config={
            "name": "Player",
            "own": st.column_config.NumberColumn("Own%", format="%.1f%%"),
            "proj": st.column_config.NumberColumn("Proj", format="%.1f"),
            "actual": st.column_config.NumberColumn("Actual", format="%.1f"),
            "vs_proj": st.column_config.NumberColumn("vs Proj", format="%+.1f"),
            "leverage_value": st.column_config.NumberColumn("Leverage Value", format="%.1f")
        },
        use_container_width=True,
        hide_index=True
    )
    
    total_leverage = lineup_df["leverage_value"].sum()
    st.info(f"**Total Leverage Value**: {total_leverage:.1f} points - This is how many points you gained vs the field through low-owned outperformance")
    
    # Replication blueprint
    st.markdown("---")
    st.header("🎯 Replication Blueprint")
    
    blueprint = generate_replication_blueprint(lineup_df, analysis)
    st.markdown(blueprint)
    
    # Find similar opportunities
    if current_slate:
        st.markdown("---")
        st.header("🔍 Similar Opportunities in Current Slate")
        
        # Load current slate
        current_df = pd.read_csv(current_slate)
        current_df.columns = current_df.columns.str.strip().str.lower()
        
        renames = {
            "player": "name",
            "position": "positions",
            "pos": "positions",
            "ownership": "own",
            "own%": "own",
            "projection": "proj",
        }
        current_df = current_df.rename(columns=renames)
        
        # Calculate metrics if needed
        if "ceiling" not in current_df.columns and "proj" in current_df.columns:
            current_df["ceiling"] = current_df["proj"] * 1.4
        
        if "gpp_score" not in current_df.columns:
            current_df["gpp_score"] = (
                current_df.get("ceiling", 0) * 0.4 +
                current_df.get("value", 0) * 10 * 0.3
            )
        
        if "game_env" not in current_df.columns:
            current_df["game_env"] = "✅ Good"
        
        if "game_total" not in current_df.columns:
            current_df["game_total"] = 220
        
        similar_output = find_similar_opportunities(lineup_df, current_df)
        st.markdown(similar_output)
    
    # Export analysis
    st.markdown("---")
    st.subheader("💾 Export Analysis")
    
    if st.button("📄 Generate Full Report"):
        report = f"""
# DFS Lineup Review Report

## Summary
- **Total Actual**: {total_actual:.1f}
- **vs Projection**: {total_actual - total_proj:+.1f} ({(total_actual - total_proj)/total_proj*100:+.1f}%)
- **vs Ceiling**: {total_actual - total_ceil:+.1f} ({(total_actual - total_ceil)/total_ceil*100:+.1f}%)
- **Total Ownership**: {total_own:.0f}%
- **Hit Rate**: {analysis['performance']['hit_rate']:.0f}%

## Key Insights
{chr(10).join(['- ' + i.replace('**', '').replace('✅ ', '') for i in insights])}

{blueprint}

## Player Performance
{display_df.to_markdown(index=False)}

---
Generated by DFS Lineup Reviewer v1.0
"""
        
        st.download_button(
            "Download Report (Markdown)",
            report,
            "lineup_review.md",
            "text/markdown"
        )

else:
    st.info("👆 Upload a lineup to begin analysis")
    
    st.markdown("---")
    st.subheader("📋 Upload Formats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Full CSV Format")
        st.markdown("**For detailed analysis with all metrics**")
        
        example_df = pd.DataFrame({
            "name": ["LeBron James", "Stephen Curry", "Kevin Durant"],
            "position": ["SF", "PG", "SF"],
            "salary": [9500, 9200, 9800],
            "proj": [48.2, 45.1, 50.3],
            "ceiling": [67.5, 63.1, 70.4],
            "own": [22, 28, 25],
            "actual": [58.5, 52.3, 71.2],
            "team": ["LAL", "GSW", "PHX"],
            "game_env": ["🔥 Elite", "🔥 Elite", "⭐ Great"]
        })
        
        st.dataframe(example_df, use_container_width=True)
        
        st.markdown("""
        **Required columns:**
        - `name`: Player name
        - `position`: Position (PG, SG, SF, etc.)
        - `salary`: DraftKings/FanDuel salary
        - `proj`: Projected points
        - `ceiling`: Ceiling projection
        - `own`: Ownership percentage
        - `actual`: Actual points scored ⚡
        
        **Optional columns:**
        - `team`: Team abbreviation
        - `game_env`: Game environment
        - `game_total`: Game total projection
        """)
    
    with col2:
        st.markdown("### Quick Bulk Format")
        st.markdown("**Just copy/paste from contest results**")
        
        st.code("""LeBron James    SF    22    58.5
Stephen Curry    PG    28    52.3
Kevin Durant    SF    25    71.2
Giannis Antetokounmpo    PF    18    65.1
Joel Embiid    C    24    48.2
Damian Lillard    PG    15    42.8
Jayson Tatum    SF    30    55.6
Bam Adebayo    C    12    38.9""", language="text")
        
        st.markdown("""
        **Format:**
        ```
        Player Name | Position | Owned% | Points
        ```
        
        **Separators:** Tab, comma, or spaces
        
        **Examples:**
        - `LeBron James    SF    22    58.5`
        - `LeBron James,SF,22,58.5`
        - `LeBron James\tSF\t22\t58.5`
        
        **The tool will estimate:**
        - Salary (based on points)
        - Projection (90% of actual)
        - Ceiling (1.4x projection)
        
        **Focus:** Ownership & performance analysis
        """)
    
    st.markdown("---")
    st.markdown("### 💡 Pro Tip")
    st.info("""
    **DraftKings Contest Results:**
    1. Go to your contest
    2. Click on your lineup
    3. Copy player name, position, owned, points
    4. Paste into bulk upload
    5. Instant analysis!
    
    **Full CSV gives more detailed analysis, but bulk upload is faster for quick reviews.**
    """)
