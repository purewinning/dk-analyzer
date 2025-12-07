import io
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import streamlit as st

# Import from builder (must live next to app.py)
from builder import (
    ownership_bucket,
    build_game_environments,
    build_team_stacks,
    calculate_lineup_correlation_score,
)

# NFL-specific logic
from nfl_stacks import (
    build_nfl_stacks,
    identify_nfl_bringback,
    validate_nfl_lineup,
    calculate_nfl_correlation_score,
)

# --------------------------------------------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------------------------------------------

st.set_page_config(
    page_title="DFS Lineup Explorer (Enhanced)",
    layout="wide"
)

# --------------------------------------------------------------------------------------
# NORMALIZATION HELPERS
# --------------------------------------------------------------------------------------

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DraftKings-style projections CSV into the internal schema.

    Works with headers like:
      Player, Salary, Position, Team, Opponent, Projection, Value, Ownership %
    (case- and space-insensitive).
    """

    df = df.copy()
    # Normalize headers
    df.columns = df.columns.str.strip().str.lower()

    # Map DK headers into our internal names
    rename_map = {
        "player": "Name",
        "name": "Name",
        "salary": "Salary",
        "position": "positions",
        "pos": "positions",
        "positions": "positions",
        "team": "Team",
        "tm": "Team",
        "opponent": "Opponent",
        "opp": "Opponent",
        "opp_team": "Opponent",
        "projection": "proj",
        "proj": "proj",
        "fpts": "proj",
        "fpts.": "proj",
        "points": "proj",
        "value": "Value",
        "ownership": "own_proj",
        "ownership%": "own_proj",
        "ownership %": "own_proj",   # your CSV: "Ownership %"
        "own": "own_proj",
        "own%": "own_proj",
        "exposure": "own_proj",
        "proj own": "own_proj",
    }
    df = df.rename(columns=rename_map)

    # Required logical columns (player_id will be created)
    required = ["Name", "Team", "positions", "Salary", "proj"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"CSV is missing required columns after normalization: {missing}")
        return pd.DataFrame()

    # Clean Salary strings like "$7,700"
    if "Salary" in df.columns:
        df["Salary"] = (
            df["Salary"]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .str.strip()
        )

    # Clean ownership strings like "13.9%"
    if "own_proj" in df.columns:
        df["own_proj"] = (
            df["own_proj"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )

    # Auto-create player_id if missing
    if "player_id" not in df.columns:
        df["player_id"] = (
            df["Name"].astype(str)
            + "_" + df["Team"].astype(str)
            + "_" + df["Salary"].astype(str)
        )

    # Ensure numeric columns
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce")
    df["own_proj"] = pd.to_numeric(df.get("own_proj", np.nan), errors="coerce")

    # Drop rows missing essential numerics
    df = df.dropna(subset=["Salary", "proj"])

    # Ensure positions is string
    df["positions"] = df["positions"].astype(str)

    return df


def add_ownership_bucket_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'own_bucket' column based on projected ownership."""
    if "own_proj" not in df.columns:
        df["own_bucket"] = "Unknown"
        return df

    df["own_bucket"] = df["own_proj"].apply(ownership_bucket)
    df["own_bucket"] = df["own_bucket"].astype("category")
    return df


def clean_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Create 'primary_pos' from multi-position strings like 'WR/RB'."""
    if "positions" not in df.columns:
        return df
    df["primary_pos"] = df["positions"].apply(lambda x: str(x).split("/")[0])
    return df


def detect_sport(df: pd.DataFrame) -> str:
    """Detect sport type from position data."""
    if df.empty or "positions" not in df.columns:
        return "UNKNOWN"

    all_positions = set()
    for val in df["positions"]:
        if isinstance(val, str):
            for p in val.replace(" ", "").split("/"):
                all_positions.add(p.upper())

    nfl_positions = {"QB", "RB", "WR", "TE", "DST", "DEF"}
    nba_positions = {"PG", "SG", "SF", "PF", "C"}

    if all_positions & nfl_positions:
        return "NFL"
    if all_positions & nba_positions:
        return "NBA"
    return "UNKNOWN"


def build_position_dtypes(sport: str):
    """Provide an ordered position dtype for nicer sorting."""
    if sport == "NBA":
        pos_order = ["PG", "SG", "SF", "PF", "C"]
    elif sport == "NFL":
        pos_order = ["QB", "RB", "WR", "TE", "DST"]
    else:
        pos_order = []
    return CategoricalDtype(categories=pos_order, ordered=True)


# --------------------------------------------------------------------------------------
# EDGE METRIC + MONTE CARLO LINEUP GENERATION
# --------------------------------------------------------------------------------------

def compute_player_edge(
    proj: float,
    own_proj: float,
    ceiling_mult: float = 1.35,
    leverage_weight: float = 1.0,
) -> float:
    """
    Simple "edge" metric combining:
      - ceiling-ish projection
      - leverage (lower ownership is slightly rewarded)
    """
    if np.isnan(own_proj):
        own_proj = 5.0

    ceiling = proj * ceiling_mult
    leverage_score = (30.0 - min(own_proj, 30.0)) / 30.0  # 0..1

    return ceiling + leverage_weight * 3.0 * leverage_score


def derive_edge_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 'edge_score' column."""
    if "proj" not in df.columns:
        df["edge_score"] = 0.0
        return df

    if "own_proj" not in df.columns:
        df["own_proj"] = np.nan

    df["edge_score"] = df.apply(
        lambda row: compute_player_edge(row["proj"], row["own_proj"]), axis=1
    )
    return df


def random_lineup_sample(
    pool: pd.DataFrame,
    roster_size: int,
    salary_cap: int,
    min_salary: int,
    locked_ids: List[str],
    max_attempts: int = 1000,
    require_unique_teams: bool = False,
) -> List[int]:
    """
    Simple random lineup generator respecting:
      - roster_size, salary_cap, min_salary
      - locked players
      - optional unique-teams constraint
    """
    if pool.empty:
        return []

    indices = pool.index.tolist()
    locked_indices = pool[pool["player_id"].isin(locked_ids)].index.tolist()

    for _ in range(max_attempts):
        remaining_slots = roster_size - len(locked_indices)
        if remaining_slots < 0:
            return []

        chosen = list(locked_indices)
        available = [ix for ix in indices if ix not in chosen]
        if len(available) < remaining_slots:
            continue

        sampled = np.random.choice(available, size=remaining_slots, replace=False)
        chosen.extend(sampled)

        lineup_df = pool.loc[chosen]
        total_salary = lineup_df["Salary"].sum()
        if total_salary > salary_cap or total_salary < min_salary:
            continue

        if require_unique_teams:
            if lineup_df["Team"].nunique() < roster_size:
                continue

        return chosen

    return []


def run_monte_carlo_lineups(
    pool: pd.DataFrame,
    sport: str,
    n_lineups: int,
    roster_size: int,
    salary_cap: int,
    min_salary: int,
    locked_ids: List[str],
    min_projection: float,
    min_floor: float,
    max_punts: int,
    allow_same_team: bool,
    game_envs=None,
    team_stacks=None,
    max_attempts: int = 2000,
) -> pd.DataFrame:
    """
    Monte Carlo lineup generator:
      1) random valid lineups
      2) enforce projection / floor / punt limits
      3) rank using edge + correlation
    """
    pool = derive_edge_column(pool)
    pool = add_ownership_bucket_column(pool)

    results = []
    essential_cols = [
        "player_id", "Name", "Team", "Opponent",
        "positions", "Salary", "proj", "own_proj", "edge_score",
    ]

    is_nfl = sport == "NFL"

    for _ in range(n_lineups * 10):  # oversample attempts
        if len(results) >= n_lineups:
            break

        chosen = random_lineup_sample(
            pool=pool,
            roster_size=roster_size,
            salary_cap=salary_cap,
            min_salary=min_salary,
            locked_ids=locked_ids,
            require_unique_teams=not allow_same_team if is_nfl else False,
        )
        if not chosen:
            continue

        lineup_df = pool.loc[chosen].copy()
        total_proj = lineup_df["proj"].sum()
        if total_proj < min_projection:
            continue

        total_floor = total_proj * 0.7
        if total_floor < min_floor:
            continue

        if "own_bucket" in lineup_df.columns:
            num_punts = (lineup_df["own_bucket"] == "Punt").sum()
            if num_punts > max_punts:
                continue

        # Correlation
        if sport == "NBA":
            corr_score = calculate_lineup_correlation_score(lineup_df, game_envs)
        elif sport == "NFL":
            corr_score = calculate_nfl_correlation_score(lineup_df, game_envs or {})
        else:
            corr_score = 0.0

        record = {
            "total_proj": total_proj,
            "total_salary": lineup_df["Salary"].sum(),
            "edge_sum": lineup_df["edge_score"].sum(),
            "correlation": corr_score,
        }

        for col in essential_cols:
            for i in range(roster_size):
                if i < len(lineup_df):
                    record[f"{col}_{i+1}"] = lineup_df.iloc[i][col]
                else:
                    record[f"{col}_{i+1}"] = None

        results.append(record)

    if not results:
        return pd.DataFrame()

    df_lineups = pd.DataFrame(results)
    df_lineups["rank_score"] = (
        df_lineups["total_proj"]
        + 0.05 * df_lineups["edge_sum"]
        + 0.5 * df_lineups["correlation"]
    )
    df_lineups = df_lineups.sort_values("rank_score", ascending=False)
    df_lineups.reset_index(drop=True, inplace=True)
    return df_lineups.head(n_lineups)


# --------------------------------------------------------------------------------------
# STREAMLIT UI HELPERS
# --------------------------------------------------------------------------------------

def sidebar_controls() -> Dict[str, Any]:
    """Sidebar sliders / inputs."""
    st.sidebar.header("Lineup Builder Controls")

    contest_type = st.sidebar.selectbox(
        "Contest Type",
        ["GPP - Large Field", "GPP - Mid Field", "Single Entry", "Cash"],
        index=0,
    )

    field_size = st.sidebar.number_input(
        "Field Size (approx)",
        min_value=10,
        max_value=200000,
        value=20000,
        step=100,
    )

    payout_top = st.sidebar.slider(
        "% of field paid",
        min_value=10,
        max_value=30,
        value=20,
        step=1,
    )

    pct_to_first = st.sidebar.slider(
        "% of prize pool to 1st",
        min_value=10,
        max_value=30,
        value=20,
        step=1,
    )

    roster_size = st.sidebar.number_input(
        "Roster Size",
        min_value=5,
        max_value=10,
        value=8,
    )

    salary_cap = st.sidebar.number_input(
        "Salary Cap",
        min_value=30000,
        max_value=70000,
        value=50000,
        step=500,
    )

    min_salary = st.sidebar.number_input(
        "Minimum Total Salary for Lineup",
        min_value=0,
        max_value=70000,
        value=48000,
        step=500,
    )

    n_lineups = st.sidebar.number_input(
        "Number of Lineups to Generate",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
    )

    # start loose so we definitely get something
    min_projection = st.sidebar.slider(
        "Min total projection",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=5.0,
    )

    min_floor = st.sidebar.slider(
        "Approx floor (proj * 0.7)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=5.0,
    )

    max_punts = st.sidebar.slider(
        "Max punts per lineup (ownership bucket)",
        min_value=0,
        max_value=8,
        value=8,
        step=1,
    )

    allow_same_team = st.sidebar.checkbox(
        "Allow many players from same team?",
        value=True,
        help="For NFL GPPs you might want heavy stacking; for NBA you may prefer spreading out.",
    )

    return dict(
        contest_type=contest_type,
        field_size=field_size,
        payout_top=payout_top,
        pct_to_first=pct_to_first,
        roster_size=roster_size,
        salary_cap=salary_cap,
        min_salary=min_salary,
        n_lineups=n_lineups,
        min_projection=min_projection,
        min_floor=min_floor,
        max_punts=max_punts,
        allow_same_team=allow_same_team,
    )


def show_player_pool(pool: pd.DataFrame, sport: str):
    """Display normalized player pool with filters."""
    st.subheader("Player Pool (normalized)")

    if pool.empty:
        st.info("No players available after normalization.")
        return

    pos_dtype = build_position_dtypes(sport)

    # FIX: can't test Index directly in if; use len(...)
    if len(pos_dtype.categories) > 0:
        pool["primary_pos"] = pool["positions"].apply(lambda x: str(x).split("/")[0])
        pool["primary_pos"] = pool["primary_pos"].astype(pos_dtype)
    else:
        pool["primary_pos"] = pool["positions"]

    unique_positions = sorted(pool["primary_pos"].dropna().unique(), key=str)
    selected_positions = st.multiselect(
        "Filter by primary position:",
        options=unique_positions,
        default=unique_positions,
    )
    filtered = pool[pool["primary_pos"].isin(selected_positions)]

    teams = sorted(filtered["Team"].dropna().unique())
    selected_teams = st.multiselect(
        "Filter by Team:",
        options=teams,
        default=teams,
    )
    filtered = filtered[filtered["Team"].isin(selected_teams)]

    if "own_bucket" in filtered.columns:
        buckets = (
            list(filtered["own_bucket"].cat.categories)
            if hasattr(filtered["own_bucket"], "cat")
            else sorted(filtered["own_bucket"].dropna().unique())
        )
    else:
        buckets = []

    if buckets:
        selected_buckets = st.multiselect(
            "Filter by ownership bucket:",
            options=buckets,
            default=buckets,
        )
        filtered = filtered[filtered["own_bucket"].isin(selected_buckets)]

    st.write("Summary:")
    col1, col2, col3 = st.columns(3)
    col1.metric("Player Count", len(filtered))
    col2.metric("Avg Projection", round(filtered["proj"].mean(), 2))
    col3.metric("Avg Salary", round(filtered["Salary"].mean(), 2))

    filtered = filtered.sort_values("proj", ascending=False)

    st.dataframe(
        filtered[
            [
                "player_id",
                "Name",
                "Team",
                "Opponent",
                "positions",
                "Salary",
                "proj",
                "own_proj",
                "own_bucket",
            ]
        ].reset_index(drop=True),
        use_container_width=True,
    )


def choose_locks(pool: pd.DataFrame) -> List[str]:
    """Let user lock players by name; returns list of player_ids."""
    if pool.empty:
        return []

    st.subheader("Locks / Exposures")

    all_players = pool["Name"].tolist()
    locked_names = st.multiselect(
        "Lock any players? (by name)",
        options=all_players,
        default=[],
    )

    locked_ids = pool[pool["Name"].isin(locked_names)]["player_id"].tolist()
    return locked_ids


def show_game_environments(game_envs: Dict[str, Any], sport: str):
    st.subheader("Game Environments")

    if not game_envs:
        st.info("No game environment data available.")
        return

    env_list = [env for _, env in game_envs.items()]
    df_env = pd.DataFrame(env_list)

    cols_order = [c for c in ["matchup", "Team", "Opponent", "pace", "total", "spread"] if c in df_env.columns]
    other_cols = [c for c in df_env.columns if c not in cols_order]
    df_env = df_env[cols_order + other_cols]

    st.dataframe(df_env, use_container_width=True)


def show_team_stacks(team_stacks: Dict[str, List[Dict[str, Any]]], sport: str):
    st.subheader("Team Stacks (Heuristic)")

    if not team_stacks:
        st.info("No team stacks identified.")
        return

    rows = []
    for team, stacks in team_stacks.items():
        for s in stacks:
            rows.append(
                {
                    "Team": team,
                    "Players": ", ".join(s["players"]),
                    "Avg Salary": s["avg_salary"],
                    "Avg Projection": s["avg_proj"],
                }
            )

    df_stacks = pd.DataFrame(rows)
    df_stacks = df_stacks.sort_values("Avg Projection", ascending=False)
    st.dataframe(df_stacks, use_container_width=True)


def flatten_lineups_for_display(df_lineups: pd.DataFrame, roster_size: int) -> pd.DataFrame:
    """Flatten lineup records into a readable table."""
    if df_lineups.empty:
        return df_lineups

    display_rows = []

    for _, row in df_lineups.iterrows():
        lineup_players = []
        for i in range(1, roster_size + 1):
            pid = row.get(f"player_id_{i}")
            name = row.get(f"Name_{i}")
            team = row.get(f"Team_{i}")
            pos = row.get(f"positions_{i}")
            salary = row.get(f"Salary_{i}")
            proj = row.get(f"proj_{i}")
            own = row.get(f"own_proj_{i}")
            if pd.isna(pid) or pid is None:
                continue
            lineup_players.append(
                {
                    "Slot": i,
                    "player_id": pid,
                    "Name": name,
                    "Team": team,
                    "positions": pos,
                    "Salary": salary,
                    "proj": proj,
                    "own_proj": own,
                }
            )

        display_rows.append(
            {
                "Lineup #": len(display_rows) + 1,
                "Players": ", ".join(
                    [f"{p['Name']} ({p['Team']} - {p['positions']})" for p in lineup_players]
                ),
                "Total Salary": row["total_salary"],
                "Total Projection": row["total_proj"],
                "Edge Sum": row["edge_sum"],
                "Correlation": row["correlation"],
            }
        )

    return pd.DataFrame(display_rows)


def download_lineups_csv(df_lineups: pd.DataFrame) -> bytes:
    csv_buf = io.StringIO()
    df_lineups.to_csv(csv_buf, index=False)
    return csv_buf.getvalue().encode("utf-8")


# --------------------------------------------------------------------------------------
# NFL-SPECIFIC WRAPPER
# --------------------------------------------------------------------------------------

def run_nfl_lineup_builder(
    pool: pd.DataFrame,
    config: Dict[str, Any],
    locked_ids: List[str],
) -> pd.DataFrame:
    sport = "NFL"
    roster_size = config["roster_size"]
    salary_cap = config["salary_cap"]
    min_salary = config["min_salary"]
    n_lineups = config["n_lineups"]
    min_projection = config["min_projection"]
    min_floor = config["min_floor"]
    max_punts = config["max_punts"]
    allow_same_team = config["allow_same_team"]

    if "Team" in pool.columns and "Opponent" in pool.columns:
        game_envs = build_game_environments(pool)
    else:
        game_envs = {}

    _ = build_nfl_stacks(pool)  # currently not hard-enforcing stacks

    df_lineups = run_monte_carlo_lineups(
        pool=pool,
        sport=sport,
        n_lineups=n_lineups,
        roster_size=roster_size,
        salary_cap=salary_cap,
        min_salary=min_salary,
        locked_ids=locked_ids,
        min_projection=min_projection,
        min_floor=min_floor,
        max_punts=max_punts,
        allow_same_team=allow_same_team,
        game_envs=game_envs,
        team_stacks=None,
    )

    if df_lineups.empty:
        return df_lineups

    valid_indices = []
    for idx, row in df_lineups.iterrows():
        lineup_players = []
        for i in range(1, roster_size + 1):
            pid = row.get(f"player_id_{i}")
            name = row.get(f"Name_{i}")
            team = row.get(f"Team_{i}")
            positions = row.get(f"positions_{i}")
            salary = row.get(f"Salary_{i}")
            proj = row.get(f"proj_{i}")
            own_proj = row.get(f"own_proj_{i}")
            if pd.isna(pid) or pid is None:
                continue
            lineup_players.append(
                {
                    "player_id": pid,
                    "Name": name,
                    "Team": team,
                    "positions": positions,
                    "Salary": salary,
                    "proj": proj,
                    "own_proj": own_proj,
                }
            )

        lineup_df = pd.DataFrame(lineup_players)

        if validate_nfl_lineup(lineup_df):
            valid_indices.append(idx)

    df_lineups = df_lineups.loc[valid_indices].reset_index(drop=True)
    return df_lineups.head(n_lineups)


# --------------------------------------------------------------------------------------
# MAIN APP
# --------------------------------------------------------------------------------------

def main():
    st.title("DFS Lineup Explorer (Enhanced Monte Carlo + Edge + Correlation)")

    st.write(
        """
        Upload a DraftKings-style CSV, tweak the controls, and generate **edge-aware** lineups
        that consider **projection, ownership, and correlation**.
        """
    )

    config = sidebar_controls()

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is None:
        st.info("Please upload a CSV file to begin.")
        return

    # Raw preview so we can see what came in
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("Raw CSV preview")
    st.write(f"Rows: {raw_df.shape[0]}, Columns: {raw_df.shape[1]}")
    st.dataframe(raw_df.head(), use_container_width=True)

    # Normalize into internal format
    df = normalize_df(raw_df)
    if df.empty:
        st.stop()

    df = clean_positions(df)
    df = add_ownership_bucket_column(df)

    sport = detect_sport(df)
    st.write(f"**Detected sport:** `{sport}`")

    if "Team" in df.columns and "Opponent" in df.columns:
        game_envs = build_game_environments(df)
    else:
        game_envs = {}

    if sport == "NBA":
        team_stacks = build_team_stacks(df)
    else:
        team_stacks = {}

    show_player_pool(df, sport)

    if game_envs:
        show_game_environments(game_envs, sport)

    if sport == "NBA" and team_stacks:
        show_team_stacks(team_stacks, sport)

    locked_ids = choose_locks(df)

    st.subheader("Generate Lineups")

    if st.button("Run Builder"):
        with st.spinner("Building lineups... (Monte Carlo + edge + correlation)"):
            if sport == "NFL":
                df_lineups = run_nfl_lineup_builder(
                    pool=df,
                    config=config,
                    locked_ids=locked_ids,
                )
            else:
                df_lineups = run_monte_carlo_lineups(
                    pool=df,
                    sport=sport,
                    n_lineups=config["n_lineups"],
                    roster_size=config["roster_size"],
                    salary_cap=config["salary_cap"],
                    min_salary=config["min_salary"],
                    locked_ids=locked_ids,
                    min_projection=config["min_projection"],
                    min_floor=config["min_floor"],
                    max_punts=config["max_punts"],
                    allow_same_team=config["allow_same_team"],
                    game_envs=game_envs,
                    team_stacks=team_stacks,
                )

        if df_lineups.empty:
            st.error(
                "❌ Could not generate any valid lineups. "
                "Try loosening constraints or expanding your player pool."
            )
        else:
            st.success(f"✅ Generated {len(df_lineups)} lineups.")

            df_display = flatten_lineups_for_display(df_lineups, config["roster_size"])
            st.dataframe(df_display, use_container_width=True)

            with st.expander("Show raw lineup data"):
                st.dataframe(df_lineups, use_container_width=True)

            csv_bytes = download_lineups_csv(df_lineups)
            st.download_button(
                "Download Lineups CSV",
                data=csv_bytes,
                file_name="lineups_enhanced.csv",
                mime="text/csv",
            )


# Run Streamlit app
main()
