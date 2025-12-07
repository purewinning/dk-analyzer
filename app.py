import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, List
import io

from builder import (
    build_template_from_params,
    generate_top_n_lineups,
    ownership_bucket,
    PUNT_THR, CHALK_THR, MEGA_CHALK_THR,
    DEFAULT_SALARY_CAP, DEFAULT_ROSTER_SIZE
)

# --- STREAMLIT CONFIGURATION ---
st.set_page_config(layout="wide", page_title="🏀 DK Lineup Optimizer")

# --- CONFIGURATION CONSTANTS ---
MIN_GAMES_REQUIRED = 2

# --- TOURNAMENT OWNERSHIP TEMPLATES (DISPLAY / GUIDANCE) ---
TOURNAMENT_OWNERSHIP_TEMPLATES = {
    "CASH": {
        "name": "Cash Game (50/50, Double-Up)",
        "description": "High floor, consistency over ceiling",
        "ownership_targets": {
            "punt": (0, 2),
            "mid": (3, 5),
            "chalk": (3, 5),
            "mega": (0, 2)
        },
        "strategy_notes": "Play chalk. High ownership is GOOD in cash games. Need 50th percentile score."
    },
    "SE": {
        "name": "Single Entry GPP",
        "description": "Balanced with 1-2 contrarian pivots",
        "ownership_targets": {
            "punt": (2, 4),
            "mid": (2, 4),
            "chalk": (1, 3),
            "mega": (0, 1)
        },
        "strategy_notes": "Ownership more condensed. Pivot off 1-2 chalk plays. Need top 10-15%."
    },
    "3MAX": {
        "name": "3-Max GPP",
        "description": "Core + variety across 3 entries",
        "ownership_targets": {
            "punt": (1, 4),
            "mid": (2, 5),
            "chalk": (1, 4),
            "mega": (0, 2)
        },
        "strategy_notes": "Build around 2-3 core players. Vary punt/chalk combos across entries."
    },
    "20MAX": {
        "name": "20-Max GPP",
        "description": "High exposure management, diverse builds",
        "ownership_targets": {
            "punt": (1, 5),
            "mid": (1, 5),
            "chalk": (0, 4),
            "mega": (0, 2)
        },
        "strategy_notes": "Max 50% exposure per player. Need 3-4 distinct build types."
    },
    "LARGE_GPP": {
        "name": "Large Field GPP (Milly Maker)",
        "description": "Maximum leverage, must be different",
        "ownership_targets": {
            "punt": (3, 6),
            "mid": (1, 3),
            "chalk": (0, 2),
            "mega": (0, 1)
        },
        "strategy_notes": "Need <5% owned player. Top 0.1% to win. Be VERY different."
    },
    "SHOWDOWN": {
        "name": "Showdown Captain Mode",
        "description": "Single game, correlation key",
        "ownership_targets": {
            "punt": (1, 3),
            "mid": (2, 4),
            "chalk": (1, 3),
            "mega": (0, 2)
        },
        "strategy_notes": "Game stacking and correlation trump ownership."
    }
}

# --- ENTRY FEE TIERS (DISPLAY ONLY) ---
ENTRY_FEE_TIERS = {
    "$0.25": {"min_field": 100, "max_field": 2500, "skill_level": "Beginner"},
    "$1": {"min_field": 100, "max_field": 5000, "skill_level": "Recreational"},
    "$3": {"min_field": 500, "max_field": 10000, "skill_level": "Intermediate"},
    "$5": {"min_field": 500, "max_field": 25000, "skill_level": "Intermediate"},
    "$10": {"min_field": 1000, "max_field": 50000, "skill_level": "Advanced"},
    "$20": {"min_field": 2000, "max_field": 100000, "skill_level": "Advanced"},
    "$50+": {"min_field": 5000, "max_field": 200000, "skill_level": "Expert/Shark"}
}

# --- WINNING TEMPLATES (WE KEEP THESE!) ---
WINNING_TEMPLATES = {
    "BALANCED_GPP": {
        "name": "Balanced GPP (Proven Winner)",
        "description": "Mix of value + leverage. Historical 15-20% win rate in top 1%.",
        "requirements": {
            "min_value_plays": 2,
            "min_leverage_plays": 2,
            "max_chalk": 2,
            "required_punt": 1,
        }
    },
    "CONTRARIAN_GPP": {
        "name": "Contrarian GPP (High Risk/Reward)",
        "description": "Low owned studs + value punts. Wins 5% but huge upside.",
        "requirements": {
            "min_value_plays": 3,
            "min_leverage_plays": 3,
            "max_chalk": 1,
            "required_punt": 2,
        }
    },
    "CHALK_SMASH": {
        "name": "Chalk Smash (When Chalk Hits)",
        "description": "Top owned + value. Works when favorites perform.",
        "requirements": {
            "min_value_plays": 3,
            "min_leverage_plays": 0,
            "max_chalk": 4,
            "required_punt": 0,
        }
    },
    "VALUE_STACK": {
        "name": "Value Stack (Salary Saver)",
        "description": "Max value to afford studs. Need 4+ players at 5.0x+",
        "requirements": {
            "min_value_plays": 4,
            "min_leverage_plays": 2,
            "max_chalk": 2,
            "required_punt": 1,
        }
    }
}

# --- HEADER MAPPING ---
REQUIRED_CSV_TO_INTERNAL_MAP = {
    'Player': 'Name',
    'Salary': 'salary',
    'Position': 'positions',
    'Team': 'Team',
    'Opponent': 'Opponent',
    'PROJECTED FP': 'proj',
    'Projection': 'proj',
    'Proj': 'proj',
    'OWNERSHIP %': 'own_proj',
    'Ownership': 'own_proj',
    'Own': 'own_proj',
    'Own%': 'own_proj',
    'Minutes': 'Minutes',
    'FPPM': 'FPPM',
    'Value': 'Value'
}
CORE_INTERNAL_COLS = ['salary', 'positions', 'proj', 'own_proj', 'Name', 'Team', 'Opponent']

# --- EDGE ENGINE ---

def calculate_player_leverage(row):
    expected_optimal_pct = (row['value'] / 5.0) * 100
    expected_optimal_pct = min(expected_optimal_pct, 100)
    leverage = expected_optimal_pct - row['own_proj']
    return round(leverage, 1)

def calculate_ceiling_score(row):
    base_ceiling = row['proj'] * 1.35
    salary_factor = (10000 - row['salary']) / 10000
    ceiling_boost = base_ceiling * salary_factor * 0.2
    return round(base_ceiling + ceiling_boost, 1)

def assign_edge_category(row):
    if row['leverage_score'] > 15 and row['value'] > 4.5:
        return "🔥 Elite Leverage"
    elif row['leverage_score'] > 10:
        return "⭐ High Leverage"
    elif row['leverage_score'] > 5:
        return "✅ Good Leverage"
    elif row['leverage_score'] > -5:
        return "➖ Neutral"
    elif row['leverage_score'] > -15:
        return "⚠️ Slight Chalk"
    else:
        return "❌ Chalk Trap"

def calculate_gpp_score(row):
    ceiling_component = (row['ceiling'] / 100) * 0.4
    value_component = (row['value'] / 7) * 0.3
    leverage_normalized = (row['leverage_score'] + 20) / 40
    leverage_component = max(0, leverage_normalized) * 0.3
    gpp_score = (ceiling_component + value_component + leverage_component) * 100
    return round(gpp_score, 1)

# --- SESSION STATE INIT ---
if 'optimal_lineups_results' not in st.session_state:
    st.session_state['optimal_lineups_results'] = {'lineups': [], 'ran': False}
if 'edited_df' not in st.session_state:
    st.session_state['edited_df'] = pd.DataFrame(
        columns=CORE_INTERNAL_COLS + ['player_id', 'GameID', 'bucket', 'value', 'Lock', 'Exclude']
    )

# --- DATA PREP ---

def load_and_preprocess_data(pasted_data: str = None) -> pd.DataFrame:
    empty_df_cols = CORE_INTERNAL_COLS + ['player_id', 'GameID', 'bucket', 'value', 'Lock', 'Exclude']
    if pasted_data is None or not pasted_data.strip():
        return pd.DataFrame(columns=empty_df_cols)

    try:
        data_io = io.StringIO(pasted_data)
        first_line = pasted_data.split('\n')[0]
        if '\t' in first_line:
            df = pd.read_csv(data_io, sep='\t')
        else:
            df = pd.read_csv(data_io)

        st.success("✅ Data pasted successfully. Checking headers...")
        df.columns = df.columns.str.strip()

        actual_map = {}
        required_internal = ['Name', 'salary', 'positions', 'Team', 'Opponent', 'proj', 'own_proj']

        for csv_name, internal_name in REQUIRED_CSV_TO_INTERNAL_MAP.items():
            if csv_name in df.columns:
                actual_map[csv_name] = internal_name

        mapped_internal_names = set(actual_map.values())
        final_missing_internal = [name for name in required_internal if name not in mapped_internal_names]

        if final_missing_internal:
            st.error("❌ Missing essential columns. Please ensure your data has these required headers:")
            st.error("**Required:** Player, Salary, Position, Team, Opponent")
            st.error("**Projection column:** One of: 'Projection', 'PROJECTED FP', or 'Proj'")
            st.error("**Ownership column:** One of: 'Ownership', 'OWNERSHIP %', 'Own', or 'Own%'")
            st.error(f"**Missing:** {', '.join(final_missing_internal)}")
            return pd.DataFrame(columns=empty_df_cols)

        df.rename(columns=actual_map, inplace=True)

        if not all(col in df.columns for col in CORE_INTERNAL_COLS):
            st.error("Internal processing error: Required columns failed to map correctly.")
            return pd.DataFrame(columns=empty_df_cols)

    except Exception as e:
        st.error(f"Error processing pasted data: {e}")
        return pd.DataFrame(columns=empty_df_cols)

    df['Team'] = df['Team'].astype(str)
    df['Opponent'] = df['Opponent'].astype(str)
    df['GameID'] = df.apply(
        lambda row: '@'.join(sorted([row['Team'], row['Opponent']])),
        axis=1
    )
    df['player_id'] = df['Name']

    df['own_proj'] = df['own_proj'].astype(str).str.replace('%', '', regex=False).str.replace(',', '.', regex=False)
    df['own_proj'] = pd.to_numeric(df['own_proj'], errors='coerce')

    if df['own_proj'].max() <= 1.0 and df['own_proj'].max() > 0:
        df['own_proj'] = df['own_proj'] * 100

    df['own_proj'] = df['own_proj'].round(1)
    df.dropna(subset=CORE_INTERNAL_COLS, inplace=True)

    try:
        df['salary'] = df['salary'].astype(str).str.strip()
        df['salary'] = df['salary'].str.replace('$', '', regex=False)
        df['salary'] = df['salary'].str.replace(',', '', regex=False)
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce').astype('Int64')
        df.dropna(subset=['salary'], inplace=True)

        df['salary'] = df['salary'].astype(int)
        df['proj'] = df['proj'].astype(float)

        df_columns = df.columns.tolist()

        if 'Minutes' in df_columns:
            df['Minutes'] = pd.to_numeric(df.get('Minutes', 0), errors='coerce').astype(float).round(2)
        if 'FPPM' in df_columns:
            df['FPPM'] = pd.to_numeric(df.get('FPPM', 0), errors='coerce').astype(float).round(2)
        if 'Value' in df_columns:
            df['Value'] = pd.to_numeric(df.get('Value', 0), errors='coerce').astype(float).round(2)

    except Exception as e:
        st.error(f"Post-load conversion failed: {e}")
        return pd.DataFrame(columns=empty_df_cols)

    if len(df) == 0:
        st.error("❌ Final player pool is empty after cleaning.")
        return pd.DataFrame(columns=empty_df_cols)

    df['bucket'] = df['own_proj'].apply(ownership_bucket)
    df['value'] = np.where(df['salary'] > 0, (df['proj'] / (df['salary'] / 1000)).round(2), 0.0)

    df['leverage_score'] = df.apply(calculate_player_leverage, axis=1)
    df['ceiling'] = df.apply(calculate_ceiling_score, axis=1)
    df['edge_category'] = df.apply(assign_edge_category, axis=1)
    df['gpp_score'] = df.apply(calculate_gpp_score, axis=1)

    if 'Lock' not in df.columns:
        df['Lock'] = False
    if 'Exclude' not in df.columns:
        df['Exclude'] = False

    for col in empty_df_cols:
        if col not in df.columns:
            df[col] = None

    return df

# --- STYLING HELPERS ---

def color_bucket(s):
    if s == 'mega':
        return 'background-color: #9C3838; color: white'
    elif s == 'chalk':
        return 'background-color: #A37F34; color: white'
    elif s == 'mid':
        return 'background-color: #38761D; color: white'
    elif s == 'punt':
        return 'background-color: #3D85C6; color: white'
    return ''

def assign_lineup_positions(lineup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust DK NBA classic slot assignment:
    PG, SG, SF, PF, C, G, F, UTIL.
    Backtracking search; if it fails, we still show the lineup with all UTIL.
    """
    slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    players = lineup_df.copy().reset_index(drop=True)

    def can_play_slot(pos_string: str, slot: str) -> bool:
        if pd.isna(pos_string):
            return False
        positions = [p.strip() for p in str(pos_string).split('/')]
        if slot == 'PG':
            return 'PG' in positions
        if slot == 'SG':
            return 'SG' in positions
        if slot == 'SF':
            return 'SF' in positions
        if slot == 'PF':
            return 'PF' in positions
        if slot == 'C':
            return 'C' in positions
        if slot == 'G':
            return 'PG' in positions or 'SG' in positions
        if slot == 'F':
            return 'SF' in positions or 'PF' in positions
        if slot == 'UTIL':
            return True
        return False

    eligibility: List[List[str]] = []
    for _, row in players.iterrows():
        pos = row.get('positions', '')
        eligible_slots = [slot for slot in slots if can_play_slot(pos, slot)]
        eligibility.append(eligible_slots)

    assignment: Dict[str, int] = {}
    used_players: set = set()

    def backtrack(slot_idx: int) -> bool:
        if slot_idx == len(slots):
            return True
        slot = slots[slot_idx]
        for i in range(len(players)):
            if i in used_players:
                continue
            if slot not in eligibility[i]:
                continue
            assignment[slot] = i
            used_players.add(i)
            if backtrack(slot_idx + 1):
                return True
            used_players.remove(i)
            del assignment[slot]
        return False

    success = backtrack(0)
    result_df = players.copy()

    if success:
        idx_to_slot = {player_idx: slot for slot, player_idx in assignment.items()}
        result_df['roster_slot'] = result_df.index.map(lambda i: idx_to_slot.get(i, 'UTIL'))
    else:
        result_df['roster_slot'] = 'UTIL'

    return result_df

# --- DISPLAY MULTIPLE LINEUPS ---

def display_multiple_lineups(slate_df, template, lineup_list):
    if not lineup_list:
        st.error("❌ No valid lineups could be found that meet all constraints.")
        st.warning("Try unlocking more players or reducing the number of lineups.")
        return

    has_actuals = st.session_state.get('has_actuals', False)

    best_lineup_data = lineup_list[0]
    best_proj = best_lineup_data['proj_score']

    best_lineup_players_df = slate_df[slate_df['player_id'].isin(best_lineup_data['player_ids'])]
    best_salary = best_lineup_players_df['salary'].sum()
    best_value = best_proj / (best_salary / 1000) if best_salary else 0

    if has_actuals and 'actual_pts' in best_lineup_players_df.columns:
        best_actual = best_lineup_players_df['actual_pts'].sum()
        has_lineup_actuals = best_lineup_players_df['actual_pts'].notna().all()
    else:
        has_lineup_actuals = False

    st.subheader("🚀 Top Lineup Metrics (Lineup 1)")

    if has_lineup_actuals:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Projected Points", f"{best_proj:.2f}")
        with col2:
            st.metric("Actual Points", f"{best_actual:.2f}", delta=f"{best_actual - best_proj:+.2f}")
        with col3:
            st.metric("Salary Used", f"${best_salary:,}")
        with col4:
            actual_value = best_actual / (best_salary / 1000) if best_salary else 0
            st.metric("Actual Value", f"{actual_value:.2f}x", delta=f"{actual_value - best_value:+.2f}x")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Projected Points", f"{best_proj:.2f}", delta="Optimal Lineup Score")
        with col2:
            st.metric("Salary Used", f"${best_salary:,}", delta=f"${template.salary_cap - best_salary:,} Remaining")
        with col3:
            st.metric("Projection Value (X)", f"{best_value:.2f}", delta="Points per $1,000")

    st.markdown("---")

    st.subheader("📋 Lineup Summary")

    summary_data = []
    for i, lineup_data in enumerate(lineup_list):
        lineup_players_df = slate_df[slate_df['player_id'].isin(lineup_data['player_ids'])]
        total_ownership = lineup_players_df['own_proj'].sum()
        avg_leverage = lineup_players_df['leverage_score'].mean()
        avg_gpp_score = lineup_players_df['gpp_score'].mean()

        summary_row = {
            'Lineup': i + 1,
            'Projected': lineup_data['proj_score'],
            'Total Own%': total_ownership,
            'Avg Leverage': avg_leverage,
            'GPP Score': avg_gpp_score,
            'Salary': lineup_players_df['salary'].sum()
        }

        if has_actuals and 'actual_pts' in lineup_players_df.columns and lineup_players_df['actual_pts'].notna().all():
            actual_score = lineup_players_df['actual_pts'].sum()
            summary_row['Actual'] = actual_score
            summary_row['Diff'] = actual_score - lineup_data['proj_score']

        summary_data.append(summary_row)

    summary_df = pd.DataFrame(summary_data).set_index('Lineup')

    if 'Actual' in summary_df.columns:
        summary_df = summary_df.sort_values('Actual', ascending=False)
        st.dataframe(
            summary_df.style.format({
                "Projected": "{:.2f}",
                "Actual": "{:.2f}",
                "Diff": "{:+.2f}",
                "Total Own%": "{:.1f}%",
                "Avg Leverage": "{:+.1f}",
                "GPP Score": "{:.1f}",
                "Salary": "${:,}"
            }),
            use_container_width=True
        )
    else:
        st.dataframe(
            summary_df.style.format({
                "Projected": "{:.2f}",
                "Total Own%": "{:.1f}%",
                "Avg Leverage": "{:+.1f}",
                "GPP Score": "{:.1f}",
                "Salary": "${:,}"
            }),
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("🔎 Lineup Detail View")

    lineup_options = [f"Lineup {i+1} (Proj: {lineup_list[i]['proj_score']:.2f})" for i in range(len(lineup_list))]
    lineup_selection = st.selectbox("Select Lineup", options=lineup_options)
    lineup_index = lineup_options.index(lineup_selection)
    selected_lineup_data = lineup_list[lineup_index]
    selected_lineup_ids = selected_lineup_data['player_ids']

    lineup_df = slate_df[slate_df['player_id'].isin(selected_lineup_ids)].copy()

    if len(lineup_df) != 8:
        st.error(f"❌ Invalid lineup size: {len(lineup_df)} players (expected 8)")
        st.write("Players in lineup:", lineup_df['Name'].tolist())
        return

    position_counts = {}
    for _, player in lineup_df.iterrows():
        positions = str(player['positions']).split('/')
        for pos in positions:
            pos = pos.strip()
            position_counts[pos] = position_counts.get(pos, 0) + 1
    st.caption(f"Position breakdown: {position_counts}")

    lineup_df = assign_lineup_positions(lineup_df)

    ROSTER_ORDER = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    position_type = pd.CategoricalDtype(ROSTER_ORDER, ordered=True)
    lineup_df['roster_slot'] = lineup_df['roster_slot'].astype(position_type)
    lineup_df.sort_values(by='roster_slot', inplace=True)

    has_actuals = st.session_state.get('has_actuals', False)

    if has_actuals and 'actual_pts' in lineup_df.columns and lineup_df['actual_pts'].notna().all():
        display_cols = [
            'roster_slot', 'Name', 'positions', 'Team', 'Opponent', 'salary',
            'proj', 'actual_pts', 'value', 'own_proj', 'actual_own', 'bucket'
        ]
        lineup_df['pts_diff'] = lineup_df['actual_pts'] - lineup_df['proj']
    else:
        display_cols = [
            'roster_slot', 'Name', 'positions', 'Team', 'Opponent',
            'salary', 'proj', 'value', 'own_proj', 'bucket', 'Minutes', 'FPPM'
        ]

    lineup_df_display = lineup_df[display_cols].reset_index(drop=True)

    rename_dict = {
        'roster_slot': 'SLOT',
        'positions': 'POS',
        'own_proj': 'Proj Own%',
        'bucket': 'CATEGORY'
    }

    if has_actuals and 'actual_pts' in display_cols:
        rename_dict.update({
            'proj': 'Proj Pts',
            'actual_pts': 'Actual Pts',
            'actual_own': 'Actual Own%'
        })
    else:
        rename_dict.update({
            'proj': 'Proj Pts',
            'Minutes': 'MIN',
            'FPPM': 'FP/M'
        })

    lineup_df_display.rename(columns=rename_dict, inplace=True)

    def color_diff(val):
        if isinstance(val, (int, float)):
            if val > 5:
                return 'background-color: #90EE90'
            elif val < -5:
                return 'background-color: #FFB6C6'
        return ''

    styled_lineup = lineup_df_display.style.applymap(color_bucket, subset=['CATEGORY'])

    if 'pts_diff' in lineup_df.columns:
        styled_lineup = styled_lineup.applymap(color_diff, subset=['pts_diff'])

    format_dict = {
        "salary": "${:,}",
        "value": "{:.2f}"
    }

    if 'Proj Pts' in lineup_df_display.columns and 'Actual Pts' in lineup_df_display.columns:
        format_dict.update({
            'Proj Pts': '{:.1f}',
            'Actual Pts': '{:.1f}',
            'pts_diff': '{:+.1f}',
            'Proj Own%': '{:.1f}%',
            'Actual Own%': '{:.1f}%'
        })
    else:
        format_dict.update({
            'Proj Pts': '{:.1f}',
            'Proj Own%': '{:.1f}%',
            'MIN': '{:.1f}',
            'FP/M': '{:.2f}'
        })

    styled_lineup = styled_lineup.format(format_dict)

    st.dataframe(styled_lineup, use_container_width=True, hide_index=True)

# --- MAIN TAB: LINEUP BUILDER ---

def tab_lineup_builder(slate_df, template):
    tournament_type = st.session_state.get('tournament_type', 'Single Entry GPP')
    tournament_config = st.session_state.get('tournament_config', {})
    contest_code = st.session_state.get('contest_code', 'SE')

    st.markdown(f"## 🎯 {tournament_type}")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.markdown(f"**Strategy:** {tournament_config.get('description', 'N/A')}")
        st.caption(f"Field Size: ~{tournament_config.get('field_size', 0):,} entries")
    with col2:
        payout_pct = tournament_config.get('top_payout_pct', 0.01) * 100
        st.markdown(f"**Payout Structure:** Top {payout_pct:.2f}% pays")
    with col3:
        rec_lineups = tournament_config.get('recommended_lineups', 1)
        if rec_lineups == 1:
            st.metric("Entries", "1", delta="Single entry")
        else:
            st.metric("Max Entries", rec_lineups)

    if contest_code in TOURNAMENT_OWNERSHIP_TEMPLATES:
        info = TOURNAMENT_OWNERSHIP_TEMPLATES[contest_code]
        targets = info['ownership_targets']
        with st.expander("📋 Ownership Guide (Not Strict, Just Guidance)", expanded=True):
            st.markdown(f"**{info['name']}**")
            st.info(f"💡 {info['strategy_notes']}")
            construction_df = pd.DataFrame({
                'Bucket': ['Punt (<10%)', 'Mid (10-30%)', 'Chalk (30-40%)', 'Mega (>40%)'],
                'Target Range': [
                    f"{targets['punt'][0]}-{targets['punt'][1]}",
                    f"{targets['mid'][0]}-{targets['mid'][1]}",
                    f"{targets['chalk'][0]}-{targets['chalk'][1]}",
                    f"{targets['mega'][0]}-{targets['mega'][1]}"
                ]
            })
            st.table(construction_df)

    st.markdown("---")
    st.header("1️⃣ Player Pool & Edge View")

    column_config = {
        "Name": st.column_config.TextColumn("Player", disabled=True, width="medium"),
        "edge_category": st.column_config.TextColumn("Edge", disabled=True, width="medium"),
        "gpp_score": st.column_config.NumberColumn("GPP Score", disabled=True, format="%.1f", width="small"),
        "leverage_score": st.column_config.NumberColumn("Leverage", disabled=True, format="%+.1f", width="small"),
        "ceiling": st.column_config.NumberColumn("Ceiling", disabled=True, format="%.1f", width="small"),
        "positions": st.column_config.TextColumn("Pos", disabled=True, width="small"),
        "salary": st.column_config.NumberColumn("Salary", format="$%d", width="small"),
        "proj": st.column_config.NumberColumn("Proj", format="%.1f", width="small"),
        "value": st.column_config.NumberColumn("Value", format="%.2f", disabled=True, width="small"),
        "own_proj": st.column_config.NumberColumn("Own%", format="%.1f%%", width="small"),
        "Lock": st.column_config.CheckboxColumn("🔒", help="Lock into lineup", width="small"),
        "Exclude": st.column_config.CheckboxColumn("❌", help="Exclude from lineups", width="small"),
        "Team": None,
        "Opponent": None,
        "bucket": None,
        "Minutes": None,
        "FPPM": None,
        "player_id": None,
        "GameID": None
    }

    column_order = [
        'Lock', 'Exclude', 'Name', 'edge_category', 'gpp_score', 'leverage_score',
        'positions', 'salary', 'proj', 'ceiling', 'value', 'own_proj', 'player_id'
    ]

    df_for_editor = slate_df.copy()

    if df_for_editor.empty:
        st.info("✏️ Paste your player data in the sidebar and click **Load Pasted Data**.")
        blank_df = pd.DataFrame(columns=column_order)
        st.data_editor(
            blank_df,
            column_config=column_config,
            column_order=column_order,
            hide_index=True,
            use_container_width=True,
            height=200,
            key="player_editor_blank"
        )
        st.markdown("---")
        st.header("2️⃣ Winning Template")
        st.info("Templates will unlock after data is loaded.")
        st.markdown("---")
        st.header("3️⃣ Build Lineups")
        st.info("Lineup builder will unlock after data is loaded.")
        return

    for col in column_order:
        if col not in df_for_editor.columns:
            if col in ['Lock', 'Exclude']:
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
        height=400,
        key="player_editor_final"
    )
    st.session_state['edited_df'] = edited_df

    edited_df['player_id'] = edited_df['player_id'].astype(str)
    locked_player_ids = edited_df[edited_df['Lock'] == True]['player_id'].tolist()
    excluded_player_ids = edited_df[edited_df['Exclude'] == True]['player_id'].tolist()

    if locked_player_ids or excluded_player_ids:
        st.caption(f"🔒 Locked: {len(locked_player_ids)} | ❌ Excluded: {len(excluded_player_ids)}")

    st.markdown("---")
    st.header("2️⃣ Winning Template")

    template_choice = st.selectbox(
        "Choose Winning Template Shape",
        options=list(WINNING_TEMPLATES.keys()),
        format_func=lambda x: WINNING_TEMPLATES[x]['name']
    )
    selected_template = WINNING_TEMPLATES[template_choice]
    reqs = selected_template['requirements']

    st.info(f"**{selected_template['name']}**\n\n{selected_template['description']}")

    st.write("**Guideline (not hard rules):**")
    st.write(f"- Min {reqs['min_value_plays']} players with **5.0+ value**")
    st.write(f"- Min {reqs['min_leverage_plays']} players with **10+ leverage**")
    st.write(f"- Max {reqs['max_chalk']} players over **30% ownership**")
    st.write(f"- At least {reqs['required_punt']} player(s) under **5% ownership**")

    st.markdown("---")
    st.header("3️⃣ Build Lineups")

    col_n, col_opts = st.columns(2)
    with col_n:
        n_lineups = st.slider("Number of Lineups", 1, 20, 10)
    with col_opts:
        use_template_enforcement = st.checkbox(
            "Nudge lineups toward this winning template",
            value=True,
            help="Soft enforcement: will prefer lineups matching the template, but never return zero."
        )

    with st.expander("⚙️ Optional Filters", expanded=False):
        min_gpp_score = st.slider(
            "Minimum GPP Score per Player (0 = off)",
            min_value=0,
            max_value=60,
            value=20
        )
        max_total_ownership = st.slider(
            "Max Total Lineup Ownership % (400 = off)",
            min_value=100,
            max_value=400,
            value=300,
            step=10
        )
        force_elite = st.checkbox(
            "Try to include at least 1 🔥 Elite Leverage player",
            value=True
        )

    run_btn = st.button(
        f"🚀 Build {n_lineups} Lineups",
        use_container_width=True,
        type="primary"
    )

    if run_btn:
        final_df = st.session_state['edited_df'].copy()
        cols_to_merge = [
            'player_id', 'bucket', 'GameID', 'Team', 'Opponent',
            'edge_category', 'leverage_score', 'ceiling', 'gpp_score', 'value'
        ]
        cols_to_merge = [c for c in cols_to_merge if c in slate_df.columns]
        cols_to_drop = [c for c in cols_to_merge if c in final_df.columns and c != 'player_id']
        if cols_to_drop:
            final_df = final_df.drop(columns=cols_to_drop)

        final_df = final_df.merge(
            slate_df[cols_to_merge],
            on='player_id',
            how='left'
        )

        if min_gpp_score > 0:
            original_count = len(final_df)
            final_df = final_df[final_df['gpp_score'] >= min_gpp_score]
            removed = original_count - len(final_df)
            if removed > 0:
                st.info(f"🔍 Filtered out {removed} low GPP-score players.")

        conflict = set(locked_player_ids) & set(excluded_player_ids)
        if conflict:
            st.error(f"❌ Conflict: {', '.join(conflict)} are both locked and excluded.")
            return

        with st.spinner(f"Building {n_lineups} lineups..."):
            top_lineups = generate_top_n_lineups(
                slate_df=final_df,
                template=template,
                n_lineups=n_lineups,
                bucket_slack=2,
                locked_player_ids=locked_player_ids,
                excluded_player_ids=excluded_player_ids,
            )

            if not top_lineups:
                st.error("❌ No lineups returned from optimizer. Try loosening constraints.")
                return

            if use_template_enforcement:
                valid_lineups = []
                fallback_scores = []

                for lineup in top_lineups:
                    ldf = final_df[final_df['player_id'].isin(lineup['player_ids'])]

                    value_plays = (ldf['value'] >= 5.0).sum()
                    leverage_plays = (ldf['leverage_score'] >= 10).sum()
                    chalk_count = (ldf['own_proj'] > 30).sum()
                    punt_count = (ldf['own_proj'] < 5).sum()
                    total_own = ldf['own_proj'].sum()
                    has_elite = (ldf['edge_category'] == '🔥 Elite Leverage').any()

                    meets_leverage = leverage_plays >= reqs['min_leverage_plays']
                    meets_value = value_plays >= reqs['min_value_plays']
                    meets_chalk = chalk_count <= reqs['max_chalk']
                    meets_punt = punt_count >= reqs['required_punt']
                    meets_own_cap = total_own <= max_total_ownership or max_total_ownership >= 400

                    template_hit = (
                        meets_leverage and
                        meets_value and
                        meets_chalk and
                        meets_punt and
                        meets_own_cap and
                        (not force_elite or has_elite)
                    )

                    # Score how close we are to the template for fallback sorting
                    score = (
                        leverage_plays * 8 +
                        value_plays * 5 +
                        punt_count * 4 -
                        max(0, chalk_count - reqs['max_chalk']) * 6
                    )
                    fallback_scores.append((lineup, score))

                    if template_hit:
                        valid_lineups.append(lineup)

                if valid_lineups:
                    if len(valid_lineups) < n_lineups:
                        st.warning(
                            f"⚠️ {len(valid_lineups)} of {n_lineups} lineups match the template. "
                            f"Showing those that do."
                        )
                    top_lineups = valid_lineups[:n_lineups]
                else:
                    st.warning(
                        "⚠️ None of the lineups fully match the template. "
                        "Showing the best available lineups ranked by how close they are."
                    )
                    fallback_scores.sort(key=lambda x: x[1], reverse=True)
                    top_lineups = [x[0] for x in fallback_scores[:n_lineups]]

        st.session_state['optimal_lineups_results'] = {'lineups': top_lineups, 'ran': True}
        st.success(f"✅ Built {len(top_lineups)} lineups.")

    st.markdown("---")
    st.header("4️⃣ Your Lineups")

    if st.session_state['optimal_lineups_results'].get('ran', False):
        display_multiple_lineups(
            slate_df,
            template,
            st.session_state['optimal_lineups_results']['lineups']
        )
    else:
        st.info("Build lineups above to see them here.")

# --- OTHER TABS (STUBS) ---

def tab_strategy_lab(slate_df, template):
    st.header("🧪 Strategy Lab")
    st.info("Coming soon – this tab will simulate different lineup styles vs a field.")

def tab_results_analysis(slate_df, template):
    st.header("📈 Results Analysis")
    st.info("Coming soon – this tab will compare projections vs actuals over time.")

def tab_contest_analyzer(slate_df, template):
    st.header("📊 Contest Analyzer")
    st.info("Coming soon – this tab will analyze payout structures and contest types.")

# --- MAIN APP ---

if __name__ == '__main__':
    with st.sidebar:
        st.title("🏀 DK Lineup Optimizer")
        st.caption("We’re not from here, so we see things different — but we’re here.")

        st.subheader("🎯 Contest Details")
        tournament_type = st.selectbox(
            "Tournament Type",
            options=[
                "Cash Game (50/50, Double-Up)",
                "Single Entry GPP",
                "3-Max GPP",
                "20-Max GPP",
                "Large Field GPP (Milly Maker)",
                "Showdown Captain Mode"
            ]
        )

        entry_fee = st.selectbox(
            "Entry Fee",
            options=["$0.25", "$1", "$3", "$5", "$10", "$20", "$50+"],
            index=2
        )

        fee_info = ENTRY_FEE_TIERS[entry_fee]
        field_size = st.slider(
            "Expected Field Size",
            min_value=fee_info["min_field"],
            max_value=fee_info["max_field"],
            value=min(10000, fee_info["max_field"]),
            step=100 if fee_info["max_field"] < 10000 else 1000
        )

        st.caption(f"💪 Competition Level: **{fee_info['skill_level']}**")

        tournament_map = {
            "Cash Game (50/50, Double-Up)": {
                "code": "CASH",
                "top_payout_pct": 0.45,
                "recommended_lineups": 1
            },
            "Single Entry GPP": {
                "code": "SE",
                "top_payout_pct": 0.20,
                "recommended_lineups": 1
            },
            "3-Max GPP": {
                "code": "3MAX",
                "top_payout_pct": 0.20,
                "recommended_lineups": 3
            },
            "20-Max GPP": {
                "code": "20MAX",
                "top_payout_pct": 0.15,
                "recommended_lineups": 20
            },
            "Large Field GPP (Milly Maker)": {
                "code": "LARGE_GPP",
                "top_payout_pct": 0.20,
                "recommended_lineups": 150
            },
            "Showdown Captain Mode": {
                "code": "SHOWDOWN",
                "top_payout_pct": 0.20,
                "recommended_lineups": 20
            }
        }

        tournament_config = tournament_map[tournament_type]
        tournament_config['field_size'] = field_size
        tournament_config['entry_fee'] = entry_fee
        tournament_config['description'] = TOURNAMENT_OWNERSHIP_TEMPLATES[tournament_config['code']]['description']

        st.info(
            f"**Strategy:** {tournament_config['description']}\n\n"
            f"**Field:** {field_size:,} entries at {entry_fee}\n\n"
            f"**Top {tournament_config['top_payout_pct']*100:.0f}%** of field cashes"
        )

        with st.expander("⚙️ Ownership Flavor Override", expanded=False):
            ownership_strategy = st.select_slider(
                "Ownership Style",
                options=["Full Chalk", "Balanced", "Contrarian", "Max Leverage"],
                value="Balanced"
            )

        if ownership_strategy == "Full Chalk":
            contest_code = "CASH"
        elif ownership_strategy in ["Contrarian", "Max Leverage"]:
            contest_code = "LARGE_GPP"
        else:
            code_mapping = {
                "CASH": "CASH",
                "SE": "SE",
                "3MAX": "SE",
                "20MAX": "LARGE_GPP",
                "LARGE_GPP": "LARGE_GPP",
                "SHOWDOWN": "SE"
            }
            contest_code = code_mapping.get(tournament_config['code'], "SE")

        st.divider()
        st.subheader("📊 Load Player Data")

        pasted_csv_data = st.text_area(
            "Paste player pool (with headers):",
            height=200,
            placeholder="Player\tSalary\tPosition\tTeam\tOpponent\tMinutes\tFPPM\tProjection\tValue\tOwnership"
        )

        load_data_btn = st.button("Load Pasted Data", use_container_width=True)

        if load_data_btn:
            if pasted_csv_data and pasted_csv_data.strip():
                with st.spinner("Processing your data..."):
                    loaded_df = load_and_preprocess_data(pasted_csv_data)
                    st.session_state['slate_df'] = loaded_df
                    if not loaded_df.empty:
                        st.success(f"✅ Loaded {len(loaded_df)} players successfully!")
                    else:
                        st.error("❌ Failed to load data. Check format and try again.")
            else:
                st.warning("⚠️ Please paste some data first!")

    st.session_state['tournament_type'] = tournament_type
    st.session_state['tournament_config'] = tournament_config
    st.session_state['contest_code'] = contest_code

    if 'slate_df' not in st.session_state:
        st.session_state['slate_df'] = pd.DataFrame(
            columns=CORE_INTERNAL_COLS + ['player_id', 'GameID', 'bucket', 'value', 'Lock', 'Exclude']
        )

    slate_df = st.session_state['slate_df']

    template = build_template_from_params(
        contest_type=contest_code,
        field_size=tournament_config['field_size'],
        pct_to_first=tournament_config['top_payout_pct'],
        roster_size=DEFAULT_ROSTER_SIZE,
        salary_cap=DEFAULT_SALARY_CAP,
        min_games=MIN_GAMES_REQUIRED
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏗️ Lineup Builder",
        "🧪 Strategy Lab",
        "📈 Results Analysis",
        "📊 Contest Analyzer"
    ])

    with tab1:
        tab_lineup_builder(slate_df, template)
    with tab2:
        tab_strategy_lab(slate_df, template)
    with tab3:
        tab_results_analysis(slate_df, template)
    with tab4:
        tab_contest_analyzer(slate_df, template)
