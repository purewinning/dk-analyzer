import io
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import streamlit as st

from builder import (
    ownership_bucket,
    build_game_environments,
    build_team_stacks,
    calculate_lineup_correlation_score,
)
from nfl_stacks import (
    build_nfl_stacks,
    identify_nfl_bringback,
    validate_nfl_lineup,
    calculate_nfl_correlation_score,
)

st.set_page_config(
    page_title="DFS Lineup Explorer (Enhanced)",
    layout="wide"
)

st.write("✅ App booted to top-level")  # <-- add this


# Column mapping constants
PLAYER_ID_COLS = ["Id", "ID", "Player ID", "PlayerID"]
NAME_COLS = ["Name", "Player", "Player Name"]
TEAM_COLS = ["Team", "Tm"]
OPPONENT_COLS = ["Opponent", "Opp", "Opp Team"]
POSITION_COLS = ["Position", "Pos", "Positions"]
SALARY_COLS = ["Salary", "Sal", "Cost"]
PROJECTION_COLS = ["Projection", "Proj", "Fpts", "FPTS", "Points"]
OWNERSHIP_COLS = ["Ownership", "Own", "Exposure", "Proj Own", "Own%"]


def normalize_column(df: pd.DataFrame, possible_names: List[str], new_name: str) -> pd.DataFrame:
    """
    Rename any matching columns from possible_names to new_name if found.
    """
    for col in possible_names:
        if col in df.columns:
            df = df.rename(columns={col: new_name})
            break
    return df


def load_and_normalize_csv(file: io.BytesIO) -> pd.DataFrame:
    """
    Load CSV, normalize columns and perform basic cleanup.
    """
    df = pd.read_csv(file)

    # Normalize columns
    df = normalize_column(df, PLAYER_ID_COLS, "player_id")
    df = normalize_column(df, NAME_COLS, "Name")
    df = normalize_column(df, TEAM_COLS, "Team")
    df = normalize_column(df, OPPONENT_COLS, "Opponent")
    df = normalize_column(df, POSITION_COLS, "positions")
    df = normalize_column(df, SALARY_COLS, "Salary")
    df = normalize_column(df, PROJECTION_COLS, "proj")
    df = normalize_column(df, OWNERSHIP_COLS, "own_proj")

    # Basic checks
    required = ["player_id", "Name", "Team", "positions", "Salary", "proj"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        return pd.DataFrame()

    # Fill or handle optional columns
    if "own_proj" not in df.columns:
        df["own_proj"] = np.nan

    # Ensure types
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce")
    df["own_proj"] = pd.to_numeric(df["own_proj"], errors="coerce")

    df = df.dropna(subset=["Salary", "proj"])  # must have these
    df["positions"] = df["positions"].astype(str)

    return df


def add_ownership_bucket_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an 'own_bucket' column to df using the ownership_bucket function from builder.
    """
    if "own_proj" not in df.columns:
        df["own_bucket"] = "Unknown"
        return df

    df["own_bucket"] = df["own_proj"].apply(ownership_bucket)
    df["own_bucket"] = df["own_bucket"].astype("category")
    return df


def clean_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    For NBA data, we often have "PG/SG" style multi-position strings;
    we keep the raw 'positions' but create a primary_pos for convenience.
    """
    # Keep full positions
    if "positions" not in df.columns:
        return df

    df["primary_pos"] = df["positions"].apply(lambda x: str(x).split("/")[0])
    return df


def detect_sport(df: pd.DataFrame) -> str:
    """
    Detect sport type from position data.
    Returns: "NBA", "NFL", or "UNKNOWN"
    """
    if df.empty or "positions" not in df.columns:
        return "UNKNOWN"

    all_positions = set()
    for val in df["positions"]:
        if isinstance(val, str):
            for p in val.replace(" ", "").split("/"):
                all_positions.add(p.upper())

    # Quick heuristics
    nfl_positions = {"QB", "RB", "WR", "TE", "DST", "DEF"}
    nba_positions = {"PG", "SG", "SF", "PF", "C"}

    if all_positions & nfl_positions:
        return "NFL"
    if all_positions & nba_positions:
        return "NBA"

    return "UNKNOWN"


def build_position_dtypes(sport: str):
    """
    Provide an ordered position dtype for NBA or NFL (for nicer sorting).
    """
    if sport == "NBA":
        pos_order = ["PG", "SG", "SF", "PF", "C"]
    elif sport == "NFL":
        pos_order = ["QB", "RB", "WR", "TE", "DST"]
    else:
        pos_order = []

    return CategoricalDtype(categories=pos_order, ordered=True)


# ======================================================================================
# LINEUP BUILDING/MONTE CARLO HELPERS (SPORT-AGNOSTIC BASE + SPORT-SPECIFIC LOGIC)
# ======================================================================================

def compute_player_edge(
    proj: float,
    own_proj: float,
    ceiling_mult: float = 1.35,
    leverage_weight: float = 1.0,
) -> float:
    """
    Extremely simple "edge" metric combining:
      - Weighted projection (ceiling-ish)
      - Leverage (how under-owned relative to some reference)
    """
    # If ownership is missing, treat as small contrarian
    if np.isnan(own_proj):
        own_proj = 5.0

    # Basic "ceiling" style projection
    ceiling = proj * ceiling_mult

    # "Leverage" – if own is small, we slightly reward this
    # Example: 5% vs 25% might be more interesting
    # We'll transform own_proj such that smaller is better
    leverage_score = (30.0 - min(own_proj, 30.0)) / 30.0  # 0..1

    # Weighted combination
    return ceiling + leverage_weight * 3.0 * leverage_score


def derive_edge_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'edge_score' column used for sorting / tie-breaking in lineup construction.
    """
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
    Very simple random lineup generator that respects:
      - roster_size
      - salary_cap
      - min_salary
      - locked players (by player_id)
      - optional "unique teams" flag

    Returns a list of row indices from 'pool' that form a lineup, or [] if failed.
    """
    if pool.empty:
        return []

    indices = pool.index.tolist()
    locked_indices = pool[pool["player_id"].isin(locked_ids)].index.tolist()

    for _ in range(max_attempts):
        remaining_slots = roster_size - len(locked_indices)
        if remaining_slots < 0:
            return []

        # Start with locked players
        chosen = list(locked_indices)

        # Randomly sample from rest
        available = [ix for ix in indices if ix not in chosen]
        if len(available) < remaining_slots:
            continue

        sampled = np.random.choice(available, size=remaining_slots, replace=False)
        chosen.extend(sampled)

        lineup_df = pool.loc[chosen]
        total_salary = lineup_df["Salary"].sum()
        if total_salary > salary_cap or total_salary < min_salary:
            continue

        # Optional team uniqueness constraint – mainly interesting for NFL
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
    A simple Monte Carlo lineup "generator" that:
      1) randomly samples valid lineups
      2) keeps those with reasonable total projection and not too many punts
      3) is edge-aware (through "edge_score" + an approximate correlation score)
    """

    # Pre-calc edge_score
    pool = derive_edge_column(pool)
    pool = add_ownership_bucket_column(pool)

    # We'll store lineups in a list of dicts
    results = []

    # Build positional columns once
    essential_cols = ["player_id", "Name", "Team", "Opponent", "positions", "Salary", "proj", "own_proj", "edge_score"]

    is_nfl = sport == "NFL"

    for _ in range(n_lineups * 10):  # oversample attempts to get enough valid lineups
        if len(results) >= n_lineups:
            break

        # For NFL, we might require some stacking logic
        if is_nfl and game_envs is not None:
            # We will just use a random game stack approach – real logic lives in nfl_stacks
            pass

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

        # Basic constraints
        total_proj = lineup_df["proj"].sum()
        if total_proj < min_projection:
            continue

        # Rough "floor" – we could use proj * 0.7 or something; for now we just reuse
        total_floor = total_proj * 0.7
        if total_floor < min_floor:
            continue

        # Count punts
        if "own_bucket" in lineup_df.columns:
            num_punts = (lineup_df["own_bucket"] == "Punt").sum()
            if num_punts > max_punts:
                continue

        # Basic correlation, sport-specific
        if sport == "NBA":
            corr_score = calculate_lineup_correlation_score(lineup_df, game_envs)
        elif sport == "NFL":
            corr_score = calculate_nfl_correlation_score(lineup_df, game_envs)
        else:
            corr_score = 0.0

        # We'll store a lineup record
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

    # Sort by combination of projection + correlation + edge
    df_lineups["rank_score"] = (
        df_lineups["total_proj"]
        + 0.05 * df_lineups["edge_sum"]
        + 0.5 * df_lineups["correlation"]
    )

    df_lineups = df_lineups.sort_values("rank_score", ascending=False)
    df_lineups.reset_index(drop=True, inplace=True)
    return df_lineups.head(n_lineups)


# ======================================================================================
# STREAMLIT UI HELPERS
# ======================================================================================

def sidebar_controls() -> Dict[str, Any]:
    """
    Renders sidebar controls and returns config.
    """
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

    min_projection = st.sidebar.slider(
        "Min total projection",
        min_value=0.0,
        max_value=500.0,
        value=200.0,
        step=5.0,
    )

    min_floor = st.sidebar.slider(
        "Approx floor (proj * 0.7)",
        min_value=0.0,
        max_value=500.0,
        value=140.0,
        step=5.0,
    )

    max_punts = st.sidebar.slider(
        "Max punts per lineup (ownership bucket)",
        min_value=0,
        max_value=5,
        value=2,
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
    """
    Display player pool with filters and summary.
    """
    st.subheader("Player Pool")

    if pool.empty:
        st.info("Upload a CSV to show the player pool.")
        return

    pos_dtype = build_position_dtypes(sport)
    if pos_dtype.categories:
        # For filtering / sorting by primary position
        pool["primary_pos"] = pool["positions"].apply(lambda x: str(x).split("/")[0])
        pool["primary_pos"] = pool["primary_pos"].astype(pos_dtype)
    else:
        pool["primary_pos"] = pool["positions"]

    # Position filter
    unique_positions = sorted(pool["primary_pos"].dropna().unique(), key=str)
    selected_positions = st.multiselect(
        "Filter by primary position:",
        options=unique_positions,
        default=unique_positions,
    )

    filtered = pool[pool["primary_pos"].isin(selected_positions)]

    # Team filter
    teams = sorted(filtered["Team"].dropna().unique())
    selected_teams = st.multiselect(
        "Filter by Team:",
        options=teams,
        default=teams,
    )
    filtered = filtered[filtered["Team"].isin(selected_teams)]

    # Ownership bucket filter if available
    if "own_bucket" in filtered.columns:
        buckets = list(filtered["own_bucket"].cat.categories) if hasattr(filtered["own_bucket"], "cat") else sorted(
            filtered["own_bucket"].dropna().unique()
        )
        selected_buckets = st.multiselect(
            "Filter by ownership bucket:",
            options=buckets,
            default=buckets,
        )
        filtered = filtered[filtered["own_bucket"].isin(selected_buckets)]

    # Show summary stats
    st.write("Summary:")
    col1, col2, col3 = st.columns(3)
    col1.metric("Player Count", len(filtered))
    col2.metric("Avg Projection", round(filtered["proj"].mean(), 2))
    col3.metric("Avg Salary", round(filtered["Salary"].mean(), 2))

    # Sort table by projection
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
    """
    Let the user lock players by name or player_id.
    Returns a list of locked player IDs.
    """
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
    """
    Display game environments or matchups.
    """
    st.subheader("Game Environments")

    if not game_envs:
        st.info("No game environment data available (check that 'Team' and 'Opponent' columns exist).")
        return

    env_list = []
    for key, env in game_envs.items():
        env_list.append(env)

    df_env = pd.DataFrame(env_list)

    # Attempt to reorder columns
    cols_order = [c for c in ["matchup", "Team", "Opponent", "pace", "total", "spread"] if c in df_env.columns]
    other_cols = [c for c in df_env.columns if c not in cols_order]
    df_env = df_env[cols_order + other_cols]

    st.dataframe(df_env, use_container_width=True)


def show_team_stacks(team_stacks: Dict[str, List[Dict[str, Any]]], sport: str):
    """
    Show possible team stacks for correlation.
    """
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
    """
    Flatten the lineup DataFrame into a more human-readable table.
    """
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
                "Players": ", ".join([f"{p['Name']} ({p['Team']} - {p['positions']})" for p in lineup_players]),
                "Total Salary": row["total_salary"],
                "Total Projection": row["total_proj"],
                "Edge Sum": row["edge_sum"],
                "Correlation": row["correlation"],
            }
        )

    df_display = pd.DataFrame(display_rows)
    return df_display


def download_lineups_csv(df_lineups: pd.DataFrame) -> bytes:
    """
    Convert a lineup DataFrame to CSV bytes for download.
    """
    csv_buf = io.StringIO()
    df_lineups.to_csv(csv_buf, index=False)
    return csv_buf.getvalue().encode("utf-8")


# ======================================================================================
# NFL-SPECIFIC LINEUP GENERATION
# ======================================================================================

def run_nfl_lineup_builder(
    pool: pd.DataFrame,
    config: Dict[str, Any],
    locked_ids: List[str],
) -> pd.DataFrame:
    """
    NFL-specific lineup builder using stacking + correlation rules
    from nfl_stacks.py plus our basic Monte Carlo / constraints.
    """
    sport = "NFL"
    roster_size = config["roster_size"]
    salary_cap = config["salary_cap"]
    min_salary = config["min_salary"]
    n_lineups = config["n_lineups"]
    min_projection = config["min_projection"]
    min_floor = config["min_floor"]
    max_punts = config["max_punts"]
    allow_same_team = config["allow_same_team"]

    # Build game environments
    if "Opponent" in pool.columns:
        game_envs = build_game_environments(pool)
    else:
        game_envs = {}

    # Build NFL stacks
    nfl_stacks = build_nfl_stacks(pool)
    qb_stacks = nfl_stacks.get("qb_stacks", {})
    rb_dst_stacks = nfl_stacks.get("rb_dst_stacks", {})

    # For now we pass game_envs to the generic MC builder; NFL correlation handled in calculate_nfl_correlation_score
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

    # Validate lineups using nfl_stacks logic (qb stacks / bringbacks, etc.)
    if df_lineups.empty:
        return df_lineups

    valid_indices = []
    for idx, row in df_lineups.iterrows():
        # Rebuild lineup players from row
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

        if validate_nfl_lineup(lineup_df, qb_stacks, rb_dst_stacks):
            valid_indices.append(idx)

    df_lineups = df_lineups.loc[valid_indices].reset_index(drop=True)
    return df_lineups.head(n_lineups)


# ======================================================================================
# MAIN APP
# ======================================================================================

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

    df = load_and_normalize_csv(uploaded_file)
    if df.empty:
        st.stop()

    df = clean_positions(df)
    df = add_ownership_bucket_column(df)

    # Detect sport
    sport = detect_sport(df)
    st.write(f"**Detected sport:** `{sport}`")

    # Build game environments and team stacks for correlation
    if "Team" in df.columns and "Opponent" in df.columns:
        game_envs = build_game_environments(df)
    else:
        game_envs = {}

    if sport == "NBA":
        team_stacks = build_team_stacks(df)
    else:
        team
