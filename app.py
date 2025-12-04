import streamlit as st
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

# Import from builder.py (must be in same repo)
from builder import (
    build_template_from_params,
    build_optimal_lineup,
    build_optimal_lineup_showdown,
)

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="DK Lineup Teacher & Builder", layout="wide")

# DraftKings NBA-style slots – tweak for other sports if needed
POS_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

# Ownership buckets (must match builder.py)
MEGA_CHALK_THR = 0.40  # >= 40%
CHALK_THR = 0.30       # 30–39%
PUNT_THR = 0.10        # < 10%


# ---------------------------------------------------------
# STRUCTURE RULES (for classifying user styles)
# ---------------------------------------------------------
@dataclass
class StructureRule:
    contest_type: str
    roster_size: int
    mega_range: Tuple[int, int]
    chalk_range: Tuple[int, int]
    mid_range: Tuple[int, int]
    punt_range: Tuple[int, int]


def get_structure_rule(contest_type: str, roster_size: int = 8, field_size: int | None = None) -> StructureRule:
    """Return a StructureRule for a given contest type."""
    ct = contest_type.upper()

    if ct == "CASH":
        return StructureRule(
            contest_type="CASH",
            roster_size=roster_size,
            mega_range=(3, 5),
            chalk_range=(2, 4),
            mid_range=(0, 2),
            punt_range=(0, 1),
        )

    if ct == "SE":
        if field_size is not None and field_size >= 5000:
            return StructureRule(
                contest_type="SE_BIG",
                roster_size=roster_size,
                mega_range=(1, 2),
                chalk_range=(2, 3),
                mid_range=(2, 3),
                punt_range=(1, 2),
            )
        else:
            return StructureRule(
                contest_type="SE_SMALL",
                roster_size=roster_size,
                mega_range=(2, 3),
                chalk_range=(2, 3),
                mid_range=(2, 3),
                punt_range=(0, 1),
            )

    if ct == "3MAX":
        return StructureRule(
            contest_type="3MAX",
            roster_size=roster_size,
            mega_range=(1, 2),
            chalk_range=(2, 3),
            mid_range=(2, 3),
            punt_range=(1, 2),
        )

    if ct == "MME":
        return StructureRule(
            contest_type="MME",
            roster_size=roster_size,
            mega_range=(1, 2),
            chalk_range=(2, 3),
            mid_range=(2, 4),
            punt_range=(1, 3),
        )

    # Fallback generic GPP
    return StructureRule(
        contest_type="GENERIC_GPP",
        roster_size=roster_size,
        mega_range=(1, 3),
        chalk_range=(2, 3),
        mid_range=(2, 4),
        punt_range=(1, 2),
    )


def classify_user_style(
    avg_mega: float,
    avg_chalk: float,
    avg_mid: float,
    avg_punt: float,
    roster_size: int = 8,
) -> Tuple[str, float]:
    """Find which contest style a user's lineup structure is closest to."""
    styles = [
        get_structure_rule("CASH", roster_size),
        get_structure_rule("SE", roster_size, field_size=1000),
        get_structure_rule("SE", roster_size, field_size=10000),
        get_structure_rule("3MAX", roster_size),
        get_structure_rule("MME", roster_size),
    ]

    best_name = "UNKNOWN"
    best_dist = float("inf")

    for rule in styles:
        mega_mid = (rule.mega_range[0] + rule.mega_range[1]) / 2
        chalk_mid = (rule.chalk_range[0] + rule.chalk_range[1]) / 2
        mid_mid = (rule.mid_range[0] + rule.mid_range[1]) / 2
        punt_mid = (rule.punt_range[0] + rule.punt_range[1]) / 2

        dist = (
            (avg_mega - mega_mid) ** 2
            + (avg_chalk - chalk_mid) ** 2
            + (avg_mid - mid_mid) ** 2
            + (avg_punt - punt_mid) ** 2
        ) ** 0.5

        if dist < best_dist:
            best_dist = dist
            best_name = rule.contest_type

    return best_name, best_dist


# ---------------------------------------------------------
# HELPERS FOR CONTEST CSV
# ---------------------------------------------------------
def extract_username(entry_name: str) -> str:
    """DK EntryName 'youdacao (5/150)' -> 'youdacao'."""
    if pd.isna(entry_name):
        return ""
    s = str(entry_name)
    if " (" in s:
        return s.split(" (", 1)[0]
    return s


def parse_lineup_string(lineup: str):
    """
    Parse DK 'Lineup' string into list of {pos_slot, player_name}.
    Example: 'PG Luka Doncic SG ...'
    """
    if pd.isna(lineup):
        return []

    tokens = str(lineup).split()
    players = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in POS_SLOTS:
            pos = tok
            i += 1
            name_tokens = []
            while i < len(tokens) and tokens[i] not in POS_SLOTS:
                name_tokens.append(tokens[i])
                i += 1
            player_name = " ".join(name_tokens).strip()
            if player_name:
                players.append({"pos_slot": pos, "player_name": player_name})
        else:
            i += 1
    return players


def build_long_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Turn raw DK standings CSV into one-row-per-lineup-per-player.
    """
    df = df_raw.copy()

    rename_map: dict[str, str] = {}
    if "EntryId" in df.columns:
        rename_map["EntryId"] = "entry_id"
    if "Entry ID" in df.columns:
        rename_map["Entry ID"] = "entry_id"
    if "EntryName" in df.columns:
        rename_map["EntryName"] = "entry_name"
    if "Entry Name" in df.columns:
        rename_map["Entry Name"] = "entry_name"
    if "Rank" in df.columns:
        rename_map["Rank"] = "rank"
    if "Points" in df.columns:
        rename_map["Points"] = "points"
    if "Total Salary" in df.columns:
        rename_map["Total Salary"] = "total_salary"
    if "Lineup" in df.columns:
        rename_map["Lineup"] = "lineup"

    df = df.rename(columns=rename_map)

    required = ["entry_id", "entry_name", "rank", "points", "lineup"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Contest CSV is missing required columns: {missing}")
        st.stop()

    df["username"] = df["entry_name"].apply(extract_username)

    records = []
    for _, row in df.iterrows():
        entry_id = row["entry_id"]
        username = row["username"]
        rank = row["rank"]
        points = row["points"]
        total_salary = row.get("total_salary", None)
        lineup_str = row["lineup"]

        players = parse_lineup_string(lineup_str)
        for p in players:
            records.append(
                {
                    "entry_id": entry_id,
                    "username": username,
                    "rank": rank,
                    "points": points,
                    "total_salary": total_salary,
                    "pos_slot": p["pos_slot"],
                    "player_name": p["player_name"],
                }
            )

    long_df = pd.DataFrame.from_records(records)
    long_df = long_df.dropna(subset=["player_name"])
    return long_df


def add_field_ownership(long_df: pd.DataFrame) -> pd.DataFrame:
    """Compute field ownership for each player in this contest."""
    total_entries = long_df["entry_id"].nunique()
    own = (
        long_df.groupby("player_name")["entry_id"]
        .nunique()
        .reset_index(name="lineups_with_player")
    )
    own["field_own"] = own["lineups_with_player"] / total_entries
    merged = long_df.merge(own[["player_name", "field_own"]], on="player_name", how="left")
    return merged


def build_lineup_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build lineup-level features:
    - avg_own, sum_own, max_own, min_own
    - counts of mega-chalk, chalk, mid, and punt plays
    """

    def count_mega(s: pd.Series) -> int:
        return (s >= MEGA_CHALK_THR).sum()

    def count_chalk(s: pd.Series) -> int:
        return ((s >= CHALK_THR) & (s < MEGA_CHALK_THR)).sum()

    def count_mid(s: pd.Series) -> int:
        return ((s >= PUNT_THR) & (s < CHALK_THR)).sum()

    def count_punt(s: pd.Series) -> int:
        return (s < PUNT_THR).sum()

    grouped = (
        long_df.groupby(["entry_id", "username"])
        .agg(
            rank=("rank", "first"),
            points=("points", "first"),
            total_salary=("total_salary", "first"),
            avg_own=("field_own", "mean"),
            sum_own=("field_own", "sum"),
            max_own=("field_own", "max"),
            min_own=("field_own", "min"),
            n_players=("player_name", "nunique"),
            n_mega_chalk=("field_own", count_mega),
            n_chalk=("field_own", count_chalk),
            n_mid=("field_own", count_mid),
            n_punt=("field_own", count_punt),
        )
        .reset_index()
        .sort_values("rank")
    )
    return grouped


def get_lineup_detail(long_df: pd.DataFrame, entry_id) -> pd.DataFrame:
    """Show a single lineup's players, positions, and field ownership."""
    detail = long_df[long_df["entry_id"] == entry_id].copy()
    detail = detail.sort_values("pos_slot")
    detail["field_own"] = (detail["field_own"] * 100.0).map("{:.1f}%".format)
    return detail[["pos_slot", "player_name", "field_own"]]


def build_user_matrix(long_df: pd.DataFrame, username: str) -> pd.DataFrame:
    """
    For a given user:
    - each row = one lineup
    - columns = positions
    - cells = player names
    """
    user_long = long_df[long_df["username"] == username].copy()
    if user_long.empty:
        return pd.DataFrame()

    mat = (
        user_long.pivot_table(
            index=["entry_id", "rank", "points"],
            columns="pos_slot",
            values="player_name",
            aggfunc="first",
        )
        .reset_index()
        .sort_values("rank")
    )
    mat.columns.name = None
    return mat


# ---------------------------------------------------------
# HELPER FOR SLATE / PROJECTIONS CSV
# ---------------------------------------------------------
def load_slate_players_from_upload(uploaded_file) -> pd.DataFrame:
    """
    Load current slate player pool from uploaded CSV.

    Required (flexible names):
    - player_id
    - name / Name
    - salary / Salary
    - proj / Projection / Proj
    - own_proj / Own
    """
    df = pd.read_csv(uploaded_file)

    rename_map: dict[str, str] = {}
    if "Name" in df.columns:
        rename_map["Name"] = "name"
    if "Salary" in df.columns:
        rename_map["Salary"] = "salary"
    if "Projection" in df.columns:
        rename_map["Projection"] = "proj"
    if "Proj" in df.columns:
        rename_map["Proj"] = "proj"
    if "Own" in df.columns:
        rename_map["Own"] = "own_proj"
    if "Pos" in df.columns:
        rename_map["Pos"] = "positions"
    if "Positions" in df.columns:
        rename_map["Positions"] = "positions"
    if "Team" in df.columns:
        rename_map["Team"] = "team"

    df = df.rename(columns=rename_map)

    required = ["player_id", "name", "salary", "proj", "own_proj"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Slate / projections CSV is missing required columns: {missing}")
        st.stop()

    def bucket_from_own(own: float) -> str:
        if own >= MEGA_CHALK_THR:
            return "mega"
        if own >= CHALK_THR:
            return "chalk"
        if own >= PUNT_THR:
            return "mid"
        return "punt"

    df["bucket"] = df["own_proj"].apply(bucket_from_own)
    return df


# ---------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------
st.title("DraftKings Lineup Teacher & Structure-Based Builder")

st.markdown(
    """
Upload:
1. A **DraftKings results CSV** (standings export) to study what won.  
2. A **slate + projections CSV** (player pool) to build sample lineups
   that match the right ownership structure for your contest type.

Supports:
- **Classic**: 8-player DK lineups  
- **Showdown**: 1 Captain (1.5× points/salary) + 5 FLEX under $50k
"""
)

uploaded_file = st.file_uploader(
    "Step 1 – Upload a DraftKings contest CSV",
    type=["csv"],
    key="contest_csv",
)

if uploaded_file is None:
    long_df = None
    lineup_summary = None
    st.info("Upload a DK contest CSV to unlock the contest analysis tabs.")
else:
    df_raw = pd.read_csv(uploaded_file)
    long_df = build_long_df(df_raw)
    long_df = add_field_ownership(long_df)
    lineup_summary = build_lineup_features(long_df)
    if lineup_summary.empty:
        st.error("No lineups parsed. Check that the CSV has a DK-style 'Lineup' column.")
        long_df = None
        lineup_summary = None

# Tabs: contest analysis (needs CSV), slate/builder can work with just slate CSV
tab_overview, tab_slate, tab_patterns, tab_players, tab_users, tab_teach = st.tabs(
    [
        "1️⃣ Contest Overview",
        "2️⃣ Slate & Builder",
        "3️⃣ Winners vs Everyone",
        "4️⃣ Players",
        "5️⃣ User Explorer",
        "6️⃣ Plain-English Summary",
    ]
)

# -------------------------------------------------
# TAB 1 – Contest Overview
# -------------------------------------------------
with tab_overview:
    st.subheader("Contest Overview")

    if lineup_summary is None:
        st.info("Upload a DK contest CSV above to see the contest overview.")
    else:
        st.sidebar.header("Contest Settings")

        total_entries = lineup_summary.shape[0]
        top_pct = st.sidebar.slider(
            "What % of lineups should count as 'winners'?",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
        )

        top_cut_rank = max(1, int(total_entries * (top_pct / 100.0)))
        lineup_summary["is_top"] = lineup_summary["rank"] <= top_cut_rank

        top_entry_ids = set(
            lineup_summary[lineup_summary["is_top"]]["entry_id"].tolist()
        )
        long_df["is_top"] = long_df["entry_id"].isin(top_entry_ids)

        st.sidebar.markdown(f"**Top cutoff rank:** {top_cut_rank} / {total_entries}")
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Ownership tiers (fixed):**")
        st.sidebar.markdown(
            f"""
- Mega chalk: **≥ {int(MEGA_CHALK_THR * 100)}%**  
- Chalk: **{int(CHALK_THR * 100)}–{int(MEGA_CHALK_THR * 100) - 1}%**  
- Mid-owned: **{int(PUNT_THR * 100)}–{int(CHALK_THR * 100) - 1}%**  
- Punt / low-owned: **< {int(PUNT_THR * 100)}%**
"""
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Typical build patterns")
        st.sidebar.markdown(
            """
**150-max GPP (big fields):**
- Often **1–3 mega chalk** pieces  
- **2–4 chalk / mid** plays  
- **1–3 punts** for leverage  

**Single-entry / 3-max:**
- Often **1–2 mega chalk**  
- Fewer wild punts (0–1)  
- More balanced overall
"""
        )

        # Save to session for other tabs
        st.session_state["top_pct"] = top_pct
        st.session_state["top_cut_rank"] = top_cut_rank
        st.session_state["lineup_summary"] = lineup_summary
        st.session_state["long_df"] = long_df

        top_df = lineup_summary[lineup_summary["is_top"]].copy()
        rest_df = lineup_summary[~lineup_summary["is_top"]].copy()
        st.session_state["top_df"] = top_df
        st.session_state["rest_df"] = rest_df

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total entries", total_entries)
        col2.metric("Winners (top group)", top_df.shape[0])
        col3.metric(
            "Median points (all)",
            f"{lineup_summary['points'].median():.2f}",
        )
        col4.metric(
            "Median avg ownership (all)",
            f"{lineup_summary['avg_own'].median():.2%}",
        )

        simple_cols = ["rank", "entry_id", "username", "points", "avg_own", "is_top"]
        renamed = lineup_summary[simple_cols].rename(
            columns={
                "rank": "Rank",
                "entry_id": "Entry ID",
                "username": "User",
                "points": "Points",
                "avg_own": "Avg player ownership",
                "is_top": f"Winner? (Top {top_pct}%)",
            }
        )
        renamed["Avg player ownership"] = renamed["Avg player ownership"].map(
            lambda x: f"{x:.1%}"
        )
        st.dataframe(renamed, use_container_width=True)

        with st.expander("Show advanced lineup stats"):
            st.dataframe(lineup_summary, use_container_width=True)

        st.subheader("Visual: Rank vs Avg Ownership")
        st.caption(
            "Each dot is a lineup. Left = better finish. Up/down = more/less chalky on average."
        )
        st.scatter_chart(lineup_summary, x="rank", y="avg_own")


# -------------------------------------------------
# TAB 2 – Slate & Lineup Builder
# -------------------------------------------------
with tab_slate:
    st.subheader("Current Slate & Structure-Based Lineup Builder")

    st.markdown(
        """
Upload a **separate CSV** with the player pool and projections for tonight’s slate.

Expected columns (names can vary, we normalize):
- `player_id`
- `Name` or `name`
- `Salary` or `salary`
- `Projection` / `Proj` / `proj`
- `Own` or `own_proj` (fraction: 0.25 = 25%)

For **showdown**, just upload a single row per player (base version).  
The app will create CPT (1.5× salary, 1.5× points) and FLEX versions internally.
"""
    )

    slate_file = st.file_uploader(
        "Upload slate + projections CSV (player pool)",
        type=["csv"],
        key="slate_upload",
    )

    if slate_file is None:
        st.info("Upload a slate / projections CSV to see the player pool and build a lineup.")
    else:
        slate_df = load_slate_players_from_upload(slate_file)

        slate_mode = st.radio(
            "Slate format",
            ["Classic (8-player DK)", "Showdown (1 CPT + 5 FLEX)"],
            index=0,
        )

        st.markdown("### Player pool (base data)")
        player_view = slate_df.copy()
        player_view["own_proj"] = player_view["own_proj"].map(lambda x: f"{x:.1%}")
        player_view = player_view.rename(
            columns={
                "name": "Player",
                "salary": "Salary",
                "proj": "Proj",
                "own_proj": "Own %",
                "bucket": "Bucket",
            }
        )
        st.dataframe(player_view, use_container_width=True)

        st.markdown("### Build an example lineup based on contest parameters")

        col_a, col_b = st.columns(2)
        with col_a:
            contest_type = st.selectbox(
                "Contest type",
                ["CASH", "SE", "3MAX", "MME"],
                index=1,
            )
        with col_b:
            field_size = st.number_input(
                "Contest size (entries)",
                min_value=10,
                max_value=500000,
                value=2000,
                step=10,
            )

        pct_to_first = st.slider(
            "% of prize pool to first place",
            min_value=5,
            max_value=40,
            value=25,
            step=1,
        )

        bucket_slack = st.slider(
            "Structure flexibility (bucket slack)",
            min_value=0,
            max_value=2,
            value=1,
            step=1,
        )

        if st.button("Build sample lineup"):
            if "Showdown" in slate_mode:
                template = build_template_from_params(
                    contest_type=contest_type,
                    field_size=field_size,
                    pct_to_first=pct_to_first,
                    roster_size=6,      # 1 CPT + 5 FLEX
                    salary_cap=50000,
                )
                st.write(
                    f"Using structure profile: **{template.contest_label}** (Showdown, 6 players)"
                )
                st.write(
                    "Target bucket counts (approx): "
                    f"mega={template.target_mega:.1f}, "
                    f"chalk={template.target_chalk:.1f}, "
                    f"mid={template.target_mid:.1f}, "
                    f"punt={template.target_punt:.1f}"
                )

                lineup = build_optimal_lineup_showdown(
                    slate_df,
                    template=template,
                    bucket_slack=bucket_slack,
                )
            else:
                template = build_template_from_params(
                    contest_type=contest_type,
                    field_size=field_size,
                    pct_to_first=pct_to_first,
                    roster_size=8,
                    salary_cap=50000,
                )
                st.write(
                    f"Using structure profile: **{template.contest_label}** (Classic, 8 players)"
                )
                st.write(
                    "Target bucket counts (approx): "
                    f"mega={template.target_mega:.1f}, "
                    f"chalk={template.target_chalk:.1f}, "
                    f"mid={template.target_mid:.1f}, "
                    f"punt={template.target_punt:.1f}"
                )

                lineup = build_optimal_lineup(
                    slate_df,
                    template=template,
                    bucket_slack=bucket_slack,
                )

            if lineup is None or lineup.empty:
                st.error(
                    "Could not find a valid lineup with these constraints. "
                    "Try increasing bucket slack or adjusting contest parameters."
                )
            else:
                show = lineup[["name", "salary", "proj", "own_proj"]].copy()
                if "role" in lineup.columns:
                    show["role"] = lineup["role"]
                show["own_proj"] = show["own_proj"].map(lambda x: f"{x:.1%}")

                cols = (
                    ["name", "role", "salary", "proj", "own_proj"]
                    if "role" in show.columns
                    else ["name", "salary", "proj", "own_proj"]
                )
                show = show.rename(
                    columns={
                        "name": "Player",
                        "salary": "Salary",
                        "proj": "Proj",
                        "own_proj": "Own %",
                    }
                )[cols]

                st.subheader("Sample lineup")
                st.dataframe(show, use_container_width=True)

                total_salary = int(lineup["salary"].sum())
                total_proj = float(lineup["proj"].sum())
                st.write(f"**Total salary:** {total_salary} / 50000")
                st.write(f"**Total projection:** {total_proj:.2f} fpts")
                if "role" in lineup.columns:
                    st.caption("Showdown: exactly 1 CPT + 5 FLEX enforced.")


# -------------------------------------------------
# TAB 3 – Winners vs Everyone
# -------------------------------------------------
with tab_patterns:
    st.subheader("Winners vs Everyone Else")

    lineup_summary = st.session_state.get("lineup_summary")
    top_df = st.session_state.get("top_df")
    rest_df = st.session_state.get("rest_df")
    top_pct = st.session_state.get("top_pct")

    if lineup_summary is None or top_df is None or rest_df is None:
        st.info("Go to 'Contest Overview' and upload a contest CSV first.")
    else:
        def summarize(group: pd.DataFrame, label: str) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "Group": [label],
                    "Mean Points": [group["points"].mean()],
                    "Median Points": [group["points"].median()],
                    "Avg Player Ownership": [group["avg_own"].mean()],
                    "Total Ownership Sum": [group["sum_own"].mean()],
                    "Max Single-Player Ownership": [group["max_own"].mean()],
                    "Mega Chalk (>=40%) per lineup": [group["n_mega_chalk"].mean()],
                    "Chalk (30–39%) per lineup": [group["n_chalk"].mean()],
                    "Mid (10–29%) per lineup": [group["n_mid"].mean()],
                    "Punts (<10%) per lineup": [group["n_punt"].mean()],
                }
            )

        top_stats = summarize(top_df, f"Winners (Top {top_pct}%)")
        rest_stats = summarize(rest_df, "Everyone else")
        table = pd.concat([top_stats, rest_stats], ignore_index=True)

        table["Avg Player Ownership"] = table["Avg Player Ownership"].map(
            lambda x: f"{x:.1%}"
        )
        table["Max Single-Player Ownership"] = table["Max Single-Player Ownership"].map(
            lambda x: f"{x:.1%}"
        )
        table["Mean Points"] = table["Mean Points"].map(lambda x: f"{x:.2f}")
        table["Median Points"] = table["Median Points"].map(lambda x: f"{x:.2f}")
        table["Total Ownership Sum"] = table["Total Ownership Sum"].map(
            lambda x: f"{x:.2f}"
        )

        st.dataframe(table, use_container_width=True)


# -------------------------------------------------
# TAB 4 – Players
# -------------------------------------------------
with tab_players:
    st.subheader("Players – Winners vs Everyone Else")

    long_df = st.session_state.get("long_df")
    top_df = st.session_state.get("top_df")
    rest_df = st.session_state.get("rest_df")

    if long_df is None or top_df is None or rest_df is None:
        st.info("Go to 'Contest Overview' and upload a contest CSV first.")
    else:
        n_top = top_df["entry_id"].nunique()
        n_rest = rest_df["entry_id"].nunique()

        g = (
            long_df.groupby(["player_name", "is_top"])["entry_id"]
            .nunique()
            .unstack(fill_value=0)
            .rename(columns={False: "lineups_rest", True: "lineups_top"})
            .reset_index()
        )

        g["top_own"] = g["lineups_top"] / max(1, n_top)
        g["rest_own"] = g["lineups_rest"] / max(1, n_rest)

        field_own_ref = (
            long_df.groupby("player_name")["field_own"].first().reset_index()
        )
        player_stats = g.merge(field_own_ref, on="player_name", how="left")
        player_stats["top_minus_rest"] = (
            player_stats["top_own"] - player_stats["rest_own"]
        )

        display_df = player_stats.copy()
        for col in ["field_own", "top_own", "rest_own", "top_minus_rest"]:
            display_df[col] = display_df[col].map(lambda x: f"{x:.1%}")

        st.dataframe(
            display_df.sort_values("top_minus_rest", ascending=False)[
                [
                    "player_name",
                    "field_own",
                    "top_own",
                    "rest_own",
                    "top_minus_rest",
                    "lineups_top",
                    "lineups_rest",
                ]
            ].head(40),
            use_container_width=True,
        )


# -------------------------------------------------
# TAB 5 – User Explorer
# -------------------------------------------------
with tab_users:
    st.subheader("User Explorer – See How a User Built Their Lineups")

    lineup_summary = st.session_state.get("lineup_summary")
    long_df = st.session_state.get("long_df")
    top_pct = st.session_state.get("top_pct")

    if lineup_summary is None or long_df is None:
        st.info("Go to 'Contest Overview' and upload a contest CSV first.")
    else:
        username_list = sorted(lineup_summary["username"].unique())
        selected_user = st.selectbox("Choose a user", username_list)

        user_group = (
            lineup_summary[lineup_summary["username"] == selected_user]
            .copy()
            .sort_values("rank")
        )

        simple_user = user_group[
            ["rank", "entry_id", "points", "avg_own", "is_top"]
        ].rename(
            columns={
                "rank": "Rank",
                "entry_id": "Entry ID",
                "points": "Points",
                "avg_own": "Avg player ownership",
                "is_top": f"Winner? (Top {top_pct}%)",
            }
        )
        simple_user["Avg player ownership"] = simple_user["Avg player ownership"].map(
            lambda x: f"{x:.1%}"
        )

        st.markdown(f"**{selected_user}** – {len(user_group)} lineups")
        st.dataframe(simple_user, use_container_width=True)

        st.subheader("User ownership pattern summary")

        user_stats = user_group.agg(
            avg_points=("points", "mean"),
            avg_avg_own=("avg_own", "mean"),
            avg_mega=("n_mega_chalk", "mean"),
            avg_chalk=("n_chalk", "mean"),
            avg_mid=("n_mid", "mean"),
            avg_punt=("n_punt", "mean"),
        )

        st.markdown(
            f"""
On average, **{selected_user}** built lineups like this:

- Avg points per lineup: **{user_stats['avg_points']:.2f}**  
- Avg player ownership: **{user_stats['avg_avg_own']:.1%}**

Per lineup (on average):

- Mega chalk (≥40%): **{user_stats['avg_mega']:.2f}** players  
- Chalk (30–39%): **{user_stats['avg_chalk']:.2f}** players  
- Mid (10–29%): **{user_stats['avg_mid']:.2f}** players  
- Punts (<10%): **{user_stats['avg_punt']:.2f}** players  
"""
        )

        best_style, dist = classify_user_style(
            avg_mega=user_stats["avg_mega"],
            avg_chalk=user_stats["avg_chalk"],
            avg_mid=user_stats["avg_mid"],
            avg_punt=user_stats["avg_punt"],
            roster_size=8,
        )

        st.subheader("Lineup style profile")
        st.markdown(
            f"""
This user's builds most closely resemble:

> **{best_style}**-style lineups (distance score `{dist:.2f}`)

Rough interpretations:
- **CASH** → very chalky, almost no punts  
- **SE_SMALL** → chalky shell with maybe 0–1 punts  
- **SE_BIG** → more leverage, 1–2 punts  
- **3MAX** → in between SE and MME  
- **MME** → more punts & leverage per lineup
"""
        )

        with st.expander("Show full advanced stats for this user"):
            st.dataframe(user_group, use_container_width=True)

        st.subheader("User lineups – matrix (players in columns)")
        user_matrix = build_user_matrix(long_df, selected_user)
        if not user_matrix.empty:
            st.dataframe(user_matrix, use_container_width=True)
        else:
            st.info("No parsed lineups for this user.")

        st.subheader("Single lineup detail (ownership by player)")
        selected_entry = st.selectbox("Pick a lineup (Entry ID)", user_group["entry_id"])
        detail = get_lineup_detail(long_df, selected_entry)
        st.table(detail)


# -------------------------------------------------
# TAB 6 – Plain-English Summary
# -------------------------------------------------
with tab_teach:
    st.subheader("Plain-English Contest Takeaways")

    lineup_summary = st.session_state.get("lineup_summary")
    top_df = st.session_state.get("top_df")
    rest_df = st.session_state.get("rest_df")
    top_pct = st.session_state.get("top_pct")

    if lineup_summary is None or top_df is None or rest_df is None:
        st.info("Go to 'Contest Overview' and upload a contest CSV first.")
    else:
        top_mean_avg_own = top_df["avg_own"].mean()
        rest_mean_avg_own = rest_df["avg_own"].mean()

        top_avg_mega = top_df["n_mega_chalk"].mean()
        rest_avg_mega = rest_df["n_mega_chalk"].mean()

        top_avg_punt = top_df["n_punt"].mean()
        rest_avg_punt = rest_df["n_punt"].mean()

        st.markdown(
            f"""
### 1. How chalky are winning lineups?

- Avg player ownership in winners (top {top_pct}%): **{top_mean_avg_own:.1%}**  
- Avg player ownership in everyone else: **{rest_mean_avg_own:.1%}**
"""
        )

        st.markdown(
            f"""
### 2. How many chalk vs punts do winners use?

Per lineup:

**Winners (top {top_pct}%):**
- Mega chalk (≥40%): **{top_avg_mega:.2f}**  
- Punts (<10%): **{top_avg_punt:.2f}**

**Everyone else:**
- Mega chalk: **{rest_avg_mega:.2f}**  
- Punts: **{rest_avg_punt:.2f}**
"""
        )

        st.markdown(
            """
### 3. How to use this with the builder

For a similar contest:

1. Decide how many mega chalk pieces you want.
2. Decide how many low-owned punts you're comfortable with.
3. Use the **Slate & Builder** tab to:
   - Choose contest type (CASH / SE / 3MAX / MME),
   - Pick Classic or Showdown,
   - Let the app build a sample lineup that matches that structure.
"""
        )
