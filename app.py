import io
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import streamlit as st

from builder import ownership_bucket  # ONLY this; no DEFAULT_* surprises

# -------------------------------------------------------------------
# BASIC CONFIG / NBA RULES
# -------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="NBA DFS Lineup Builder")

SALARY_CAP = 50000   # DraftKings NBA classic
ROSTER_SIZE = 8      # PG, SG, SF, PF, C, G, F, UTIL

# -------------------------------------------------------------------
# CSV MAPPING / EDGE CALCS
# -------------------------------------------------------------------
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
CORE_INTERNAL_COLS = [
    "salary",
    "positions",
    "proj",
    "own_proj",
    "Name",
    "Team",
    "Opponent",
]


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
    """
    Map headers, clean salary/own, compute buckets & edge fields.
    """
    empty_df_cols = CORE_INTERNAL_COLS + [
        "player_id",
        "GameID",
        "bucket",
        "value",
        "Lock",
        "Exclude",
    ]
    if pasted_data is None or not pasted_data.strip():
        return pd.DataFrame(columns=empty_df_cols)

    try:
        data_io = io.StringIO(pasted_data)
        first_line = pasted_data.split("\n")[0]
        if "\t" in first_line:
            df = pd.read_csv(data_io, sep="\t")
        else:
            df = pd.read_csv(data_io)

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

        mapped_internal_names = set(actual_map.values())
        final_missing_internal = [
            name for name in required_internal if name not in mapped_internal_names
        ]

        if final_missing_internal:
            st.error("❌ Missing required columns.")
            st.error("Required: Player, Salary, Position, Team, Opponent")
            st.error("Projection: one of [Projection, PROJECTED FP, Proj]")
            st.error("Ownership: one of [Ownership, OWNERSHIP %, Own, Own%]")
            st.error(f"Missing: {', '.join(final_missing_internal)}")
            return pd.DataFrame(columns=empty_df_cols)

        df.rename(columns=actual_map, inplace=True)
        if not all(col in df.columns for col in CORE_INTERNAL_COLS):
            st.error("Internal column mapping failed.")
            return pd.DataFrame(columns=empty_df_cols)

    except Exception as e:
        st.error(f"Error processing pasted data: {e}")
        return pd.DataFrame(columns=empty_df_cols)

    # Basic cleaning
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
    if df["own_proj"].max() is not None and df["own_proj"].max() <= 1.0 and df[
        "own_proj"
    ].max() > 0:
        df["own_proj"] = df["own_proj"] * 100
    df["own_proj"] = df["own_proj"].round(1)
    df.dropna(subset=CORE_INTERNAL_COLS, inplace=True)

    try:
        df["salary"] = df["salary"].astype(str).str.strip()
        df["salary"] = df["salary"].str.replace("$", "", regex=False)
        df["salary"] = df["salary"].str.replace(",", "", regex=False)
        df["salary"] = pd.to_numeric(df["salary"], errors="coerce").astype("Int64")
        df.dropna(subset=["salary"], inplace=True)
        df["salary"] = df["salary"].astype(int)
        df["proj"] = df["proj"].astype(float)

        if "Minutes" in df.columns:
            df["Minutes"] = (
                pd.to_numeric(df.get("Minutes", 0), errors="coerce")
                .astype(float)
                .round(2)
            )
        if "FPPM" in df.columns:
            df["FPPM"] = (
                pd.to_numeric(df.get("FPPM", 0), errors="coerce")
                .astype(float)
                .round(2)
            )
        if "Value" in df.columns:
            df["Value"] = (
                pd.to_numeric(df.get("Value", 0), errors="coerce")
                .astype(float)
                .round(2)
            )
    except Exception as e:
        st.error(f"Conversion failed: {e}")
        return pd.DataFrame(columns=empty_df_cols)

    if len(df) == 0:
        st.error("❌ Final player pool is empty after cleaning.")
        return pd.DataFrame(columns=empty_df_cols)

    # ownership buckets & edges
    df["bucket"] = df["own_proj"].apply(ownership_bucket)
    df["value"] = np.where(
        df["salary"] > 0, (df["proj"] / (df["salary"] / 1000)).round(2), 0.0
    )

    df["leverage_score"] = df.apply(calculate_player_leverage, axis=1)
    df["ceiling"] = df.apply(calculate_ceiling_score, axis=1)
    df["edge_category"] = df.apply(assign_edge_category, axis=1)
    df["gpp_score"] = df.apply(calculate_gpp_score, axis=1)

    if "Lock" not in df.columns:
        df["Lock"] = False
    if "Exclude" not in df.columns:
        df["Exclude"] = False

    for col in empty_df_cols:
        if col not in df.columns:
            df[col] = None

    return df


# -------------------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------------------
if "slate_df" not in st.session_state:
    st.session_state["slate_df"] = pd.DataFrame(
        columns=CORE_INTERNAL_COLS
        + ["player_id", "GameID", "bucket", "value", "Lock", "Exclude"]
    )
if "edited_df" not in st.session_state:
    st.session_state["edited_df"] = st.session_state["slate_df"].copy()
if "optimal_lineups_results" not in st.session_state:
    st.session_state["optimal_lineups_results"] = {"lineups": [], "ran": False}


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def color_bucket(val):
    if val == "mega":
        return "background-color: #9C3838; color: white"
    if val == "chalk":
        return "background-color: #A37F34; color: white"
    if val == "mid":
        return "background-color: #38761D; color: white"
    if val == "punt":
        return "background-color: #3D85C6; color: white"
    return ""


def assign_lineup_positions(lineup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Greedy DK NBA slot assignment.
    If it can't be perfect, everyone becomes UTIL instead of erroring out.
    """
    assigned_players = set()
    slot_assignments: Dict[str, str] = {}

    def can_play_slot(pos_string, slot):
        if pd.isna(pos_string):
            return False
        positions = [p.strip() for p in str(pos_string).split("/")]
        if slot == "PG":
            return "PG" in positions
        if slot == "SG":
            return "SG" in positions
        if slot == "SF":
            return "SF" in positions
        if slot == "PF":
            return "PF" in positions
        if slot == "C":
            return "C" in positions
        if slot == "G":
            return "PG" in positions or "SG" in positions
        if slot == "F":
            return "SF" in positions or "PF" in positions
        if slot == "UTIL":
            return True
        return False

    def flexibility(pos_string):
        if pd.isna(pos_string):
            return 0
        positions = [p.strip() for p in str(pos_string).split("/")]
        flex_count = len(positions)
        if "PG" in positions or "SG" in positions:
            flex_count += 1
        if "SF" in positions or "PF" in positions:
            flex_count += 1
        return flex_count

    specific_slots = ["C", "PG", "SG", "SF", "PF"]
    flex_slots = ["G", "F", "UTIL"]
    ordered_slots = specific_slots + flex_slots

    success = True
    for slot in ordered_slots:
        available = lineup_df[~lineup_df["player_id"].isin(assigned_players)].copy()
        if available.empty:
            success = False
            break
        eligible = available[
            available["positions"].apply(lambda x: can_play_slot(x, slot))
        ]
        if eligible.empty:
            success = False
            break
        if slot in specific_slots:
            eligible["flexibility"] = eligible["positions"].apply(flexibility)
            eligible = eligible.sort_values("flexibility")
        chosen = eligible.iloc[0]
        slot_assignments[slot] = chosen["player_id"]
        assigned_players.add(chosen["player_id"])

    result_df = lineup_df.copy()
    if not success:
        result_df["roster_slot"] = "UTIL"
        return result_df

    result_df["roster_slot"] = result_df["player_id"].map(
        {v: k for k, v in slot_assignments.items()}
    )
    return result_df


def display_lineups(slate_df: pd.DataFrame, lineup_list: List[Dict[str, Any]]):
    if not lineup_list:
        st.error("❌ No lineups to display.")
        return

    lineup_list = sorted(lineup_list, key=lambda x: x["proj_score"], reverse=True)
    best = lineup_list[0]
    best_players = slate_df[slate_df["player_id"].isin(best["player_ids"])]

    salary_used = int(best_players["salary"].sum())
    total_own = float(best_players["own_proj"].sum())

    st.subheader("Top Lineup Snapshot")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Projected Points", f"{best['proj_score']:.2f}")
    with c2:
        st.metric("Salary Used", f"${salary_used:,}")
    with c3:
        st.metric("Total Ownership", f"{total_own:.1f}%")

    st.markdown("---")
    st.subheader("Lineup Summary")

    rows = []
    for i, lu in enumerate(lineup_list, start=1):
        lp = slate_df[slate_df["player_id"].isin(lu["player_ids"])]
        rows.append(
            {
                "Lineup": i,
                "Proj": lu["proj_score"],
                "Total Own%": lp["own_proj"].sum(),
                "Salary": lp["salary"].sum(),
            }
        )
    summary_df = pd.DataFrame(rows).set_index("Lineup")
    st.dataframe(
        summary_df.style.format(
            {"Proj": "{:.2f}", "Total Own%": "{:.1f}", "Salary": "${:,}"}
        ),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Lineup Detail")

    options = [
        f"Lineup {i} (Proj {lu['proj_score']:.2f})"
        for i, lu in enumerate(lineup_list, start=1)
    ]
    choice = st.selectbox("Choose lineup:", options)
    idx = options.index(choice)
    chosen = lineup_list[idx]
    chosen_df = slate_df[slate_df["player_id"].isin(chosen["player_ids"])].copy()

    chosen_df = assign_lineup_positions(chosen_df)
    roster_order = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    cat_type = CategoricalDtype(roster_order, ordered=True)
    chosen_df["roster_slot"] = chosen_df["roster_slot"].astype(cat_type)
    chosen_df.sort_values("roster_slot", inplace=True)

    display_cols = [
        "roster_slot",
        "Name",
        "positions",
        "Team",
        "Opponent",
        "salary",
        "proj",
        "value",
        "own_proj",
        "bucket",
    ]
    df_disp = chosen_df[display_cols].copy()
    df_disp.rename(
        columns={
            "roster_slot": "SLOT",
            "positions": "POS",
            "proj": "Proj Pts",
            "own_proj": "Proj Own%",
            "bucket": "CATEGORY",
        },
        inplace=True,
    )
    styled = df_disp.style.applymap(color_bucket, subset=["CATEGORY"]).format(
        {
            "salary": "${:,}",
            "Proj Pts": "{:.1f}",
            "value": "{:.2f}",
            "Proj Own%": "{:.1f}%",
        }
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def get_builder_style(contest_label: str, field_size: int) -> str:
    """
    Decide "style" for scoring (cash vs SE vs 20-max vs milly).
    """
    if contest_label == "Cash Game (50/50, Double-Up)":
        return "cash"
    if contest_label == "Single Entry":
        return "single_entry"
    if contest_label == "3-Max":
        return "three_max"
    if contest_label == "20-Max":
        return "twenty_max"
    if contest_label == "150-Max (Milly Maker)":
        return "milly"
    return "single_entry"


def get_default_n_lineups(contest_label: str) -> int:
    if contest_label == "Cash Game (50/50, Double-Up)":
        return 1
    if contest_label == "Single Entry":
        return 1
    if contest_label == "3-Max":
        return 3
    if contest_label == "20-Max":
        return 20
    if contest_label == "150-Max (Milly Maker)":
        return 40
    return 10


def get_edge_weights(contest_style: str) -> Dict[str, float]:
    """
    Weights for projection / gpp_score / leverage / ownership by contest type.
    Higher own penalty + leverage weight = more contrarian.
    """
    if contest_style == "cash":
        return {"proj": 1.0, "gpp": 0.1, "lev": 0.0, "own": 0.0}
    if contest_style == "single_entry":
        return {"proj": 0.8, "gpp": 0.6, "lev": 0.3, "own": 0.05}
    if contest_style == "three_max":
        return {"proj": 0.7, "gpp": 0.8, "lev": 0.4, "own": 0.10}
    if contest_style == "twenty_max":
        return {"proj": 0.6, "gpp": 1.0, "lev": 0.6, "own": 0.15}
    if contest_style == "milly":
        return {"proj": 0.5, "gpp": 1.2, "lev": 0.8, "own": 0.25}
    return {"proj": 0.8, "gpp": 0.6, "lev": 0.3, "own": 0.05}


def build_simple_lineups(
    df: pd.DataFrame,
    contest_style: str,
    n_lineups: int,
    salary_cap: int,
    roster_size: int,
    locked_ids: List[str],
    excluded_ids: List[str],
) -> List[Dict[str, Any]]:
    """
    Lineup builder with edge logic:
    - Uses proj + gpp_score + leverage - ownership penalty
    - Weights depend on contest_style (cash vs SE vs 20-max vs milly)
    - Always respects locks/excludes, salary cap, roster size
    """

    # Filter out excluded upfront
    pool = df[~df["player_id"].isin(excluded_ids)].copy()

    # Make sure locks are in pool
    locks_df = pool[pool["player_id"].isin(locked_ids)].copy()

    if len(locks_df) > roster_size:
        return []  # impossible

    lock_salary = locks_df["salary"].sum()
    if lock_salary > salary_cap:
        return []  # impossible

    # Fill NaNs just in case
    for col in ["proj", "gpp_score", "leverage_score", "own_proj"]:
        if col in pool.columns:
            pool[col] = pool[col].fillna(0)

    weights = get_edge_weights(contest_style)

    pool["edge_score_base"] = (
        weights["proj"] * pool["proj"]
        + weights["gpp"] * (pool["gpp_score"] / 3.0)
        + weights["lev"] * pool["leverage_score"]
        - weights["own"] * pool["own_proj"]
    )

    lineups = []

    for i in range(n_lineups):
        used_ids = set(locked_ids)
        current_salary = lock_salary
        lineup_players = list(locks_df["player_id"])

        # Tiny noise per lineup so we don't get the same thing every time
        noise = np.random.normal(0, 0.5, size=len(pool))
        pool["edge_score"] = pool["edge_score_base"] + noise

        # Sort by edge_score
        candidates = pool.sort_values("edge_score", ascending=False)

        for _, row in candidates.iterrows():
            pid = row["player_id"]
            if pid in used_ids:
                continue
            if len(lineup_players) >= roster_size:
                break
            if current_salary + row["salary"] > salary_cap:
                continue
            lineup_players.append(pid)
            used_ids.add(pid)
            current_salary += row["salary"]

        # If we couldn't fill, try a second pass by cheapest salary
        if len(lineup_players) < roster_size:
            remaining = pool[~pool["player_id"].isin(used_ids)].sort_values("salary")
            for _, row in remaining.iterrows():
                if len(lineup_players) >= roster_size:
                    break
                pid = row["player_id"]
                if current_salary + row["salary"] > salary_cap:
                    continue
                lineup_players.append(pid)
                used_ids.add(pid)
                current_salary += row["salary"]

        if len(lineup_players) != roster_size:
            # If still not full, skip this lineup
            continue

        lineup_proj = float(
            pool[pool["player_id"].isin(lineup_players)]["proj"].sum()
        )
        lineups.append(
            {
                "player_ids": lineup_players,
                "proj_score": lineup_proj,
            }
        )

    # Deduplicate lineups by player set
    unique_lineups = []
    seen = set()
    for lu in lineups:
        key = tuple(sorted(lu["player_ids"]))
        if key in seen:
            continue
        seen.add(key)
        unique_lineups.append(lu)

    return unique_lineups


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------

# Sidebar – simple controls
st.sidebar.title("Contest Settings")

contest_type_label = st.sidebar.selectbox(
    "Contest Type",
    options=[
        "Cash Game (50/50, Double-Up)",
        "Single Entry",
        "3-Max",
        "20-Max",
        "150-Max (Milly Maker)",
    ],
    index=1,
)

field_size = st.sidebar.number_input(
    "Approx Field Size",
    min_value=100,
    max_value=200000,
    value=5000,
    step=100,
)

default_n = get_default_n_lineups(contest_type_label)
n_lineups = st.sidebar.slider(
    "Number of Lineups",
    min_value=1,
    max_value=40,
    value=default_n,
)

contest_style = get_builder_style(contest_type_label, int(field_size))
st.sidebar.caption(f"Build style: **{contest_style}** (proj + upside + leverage – ownership)")

st.sidebar.markdown("---")
pasted_csv_data = st.sidebar.text_area(
    "Paste DK player pool (CSV/TSV with headers):",
    height=150,
    placeholder="Player\tSalary\tPosition\tTeam\tOpponent\tProjection\tOwnership\n...",
)
load_btn = st.sidebar.button("Load Player Data")

if load_btn:
    if pasted_csv_data and pasted_csv_data.strip():
        with st.spinner("Processing your data..."):
            loaded_df = load_and_preprocess_data(pasted_csv_data)
            st.session_state["slate_df"] = loaded_df
            st.session_state["edited_df"] = loaded_df.copy()
            if not loaded_df.empty:
                st.sidebar.success(f"Loaded {len(loaded_df)} players.")
            else:
                st.sidebar.error("Failed to load data. Check format and try again.")
    else:
        st.sidebar.warning("Paste something first.")


# Main – builder + results
st.title("NBA DFS Lineup Builder (Edge-Aware)")

slate_df = st.session_state["slate_df"]

if slate_df.empty:
    st.info("Load a player pool in the sidebar to begin.")
else:
    st.subheader("Player Pool (Lock / Exclude)")

    column_config = {
        "Name": st.column_config.TextColumn("Player", disabled=True),
        "edge_category": st.column_config.TextColumn("Edge", disabled=True),
        "gpp_score": st.column_config.NumberColumn(
            "GPP Score", disabled=True, format="%.1f"
        ),
        "leverage_score": st.column_config.NumberColumn(
            "Leverage", disabled=True, format="%+.1f"
        ),
        "ceiling": st.column_config.NumberColumn(
            "Ceiling", disabled=True, format="%.1f"
        ),
        "positions": st.column_config.TextColumn("Pos", disabled=True),
        "salary": st.column_config.NumberColumn("Salary", format="$%d"),
        "proj": st.column_config.NumberColumn("Proj", format="%.1f"),
        "value": st.column_config.NumberColumn(
            "Value", disabled=True, format="%.2f"
        ),
        "own_proj": st.column_config.NumberColumn("Own%", format="%.1f%%"),
        "Lock": st.column_config.CheckboxColumn("🔒"),
        "Exclude": st.column_config.CheckboxColumn("❌"),
        "Team": None,
        "Opponent": None,
        "bucket": None,
        "Minutes": None,
        "FPPM": None,
        "player_id": None,
        "GameID": None,
    }
    column_order = [
        "Lock",
        "Exclude",
        "Name",
        "edge_category",
        "gpp_score",
        "leverage_score",
        "positions",
        "salary",
        "proj",
        "ceiling",
        "value",
        "own_proj",
        "player_id",
    ]

    df_for_editor = st.session_state["edited_df"].copy()
    for col in column_order:
        if col not in df_for_editor.columns:
            if col in ("Lock", "Exclude"):
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
        height=450,
        key="player_editor",
    )
    st.session_state["edited_df"] = edited_df
    edited_df["player_id"] = edited_df["player_id"].astype(str)

    locked_ids = edited_df[edited_df["Lock"] == True]["player_id"].tolist()
    excluded_ids = edited_df[edited_df["Exclude"] == True]["player_id"].tolist()
    if locked_ids or excluded_ids:
        st.caption(
            f"Locked: {len(locked_ids)}   •   Excluded: {len(excluded_ids)}"
        )

    st.markdown("---")
    run_btn = st.button("Generate Lineups", type="primary")

    if run_btn:
        # Pre-flight sanity checks to avoid mysterious "no valid lineups"
        pool_for_check = edited_df[~edited_df["player_id"].isin(excluded_ids)]

        if len(pool_for_check) < ROSTER_SIZE:
            st.error(
                f"❌ Not enough players to build a full lineup.\n\n"
                f"- Players available after excludes: {len(pool_for_check)}\n"
                f"- Required for NBA DK: {ROSTER_SIZE}\n\n"
                f"Remove some excludes or load a larger pool."
            )
        else:
            locks_df = pool_for_check[pool_for_check["player_id"].isin(locked_ids)]

            if len(locks_df) > ROSTER_SIZE:
                st.error(
                    f"❌ Too many locked players.\n\n"
                    f"- Locked: {len(locks_df)}\n"
                    f"- Roster size: {ROSTER_SIZE}\n\n"
                    f"Unlock at least {len(locks_df) - ROSTER_SIZE} player(s)."
                )
            else:
                lock_salary = int(locks_df["salary"].sum()) if not locks_df.empty else 0
                if lock_salary > SALARY_CAP:
                    st.error(
                        f"❌ Salary cap exceeded by locks alone.\n\n"
                        f"- Locked salary: ${lock_salary:,}\n"
                        f"- Cap: ${SALARY_CAP:,}\n\n"
                        f"Unlock a high-salary player or two."
                    )
                else:
                    with st.spinner("Building lineups with upside + leverage logic..."):
                        lineups = build_simple_lineups(
                            df=edited_df,
                            contest_style=contest_style,
                            n_lineups=n_lineups,
                            salary_cap=SALARY_CAP,
                            roster_size=ROSTER_SIZE,
                            locked_ids=locked_ids,
                            excluded_ids=excluded_ids,
                        )

                    if not lineups:
                        st.error(
                            "❌ Could not generate any valid lineups.\n\n"
                            "The builder tried many combinations but could not satisfy:\n"
                            f"- Roster size = {ROSTER_SIZE}\n"
                            f"- Salary cap = ${SALARY_CAP:,}\n"
                            "- Your current locks / excludes\n\n"
                            "Try:\n"
                            "• Reducing the number of locked players\n"
                            "• Removing some excludes\n"
                            "• Making sure you haven't filtered the player pool too tightly"
                        )
                    else:
                        st.session_state["optimal_lineups_results"] = {
                            "lineups": lineups,
                            "ran": True,
                        }
                        st.success(f"✅ Built {len(lineups)} lineups.")

    if st.session_state["optimal_lineups_results"].get("ran", False):
        st.markdown("---")
        display_lineups(
            slate_df,
            st.session_state["optimal_lineups_results"]["lineups"],
        )
