# app.py

import io
from dataclasses import dataclass
from typing import Tuple, Dict

import pandas as pd
import streamlit as st

from builder import (
    build_template_from_params,
    build_optimal_lineup,
    build_optimal_lineup_showdown,
)

# ---------------------------------------------------------
# GLOBAL CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="DK Lineup Builder & Contest Analyzer", layout="wide")

# DraftKings classic slots (for parsing contest "Lineup" strings)
POS_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

# Ownership thresholds (must match builder.py)
MEGA_CHALK_THR = 0.40  # >= 40%
CHALK_THR = 0.30       # 30–39%
PUNT_THR = 0.10        # < 10%


# ---------------------------------------------------------
# LINEUP BUILDER – LOAD SLATE / PROJECTIONS
# ---------------------------------------------------------
def load_slate_from_any_source(
    uploaded_file,
    pasted_text: str,
) -> pd.DataFrame | None:
    """
    Accept either:
      - uploaded CSV file, OR
      - pasted CSV text

    and return a normalized slate DataFrame with columns:
      player_id, name, salary, proj, own_proj, bucket

    Specifically supports files like your NBA_DK_Main_Projections.csv with:
      Player, Salary, Position, Team, Opponent, Minutes, FPPM,
      Projection, Value, Ownership %, Optimal %, Leverage
    """
    # ---- 1. Load from file or pasted text ----
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        pasted_text = pasted_text.strip()
        if not pasted_text:
            return None
        buffer = io.StringIO(pasted_text)
        df = pd.read_csv(buffer)

    # ---- 2. Normalize column names ----
    rename_map: Dict[str, str] = {}

    # Name / Player
    if "name" in df.columns:
        rename_map["name"] = "name"
    if "Name" in df.columns:
        rename_map["Name"] = "name"
    if "Player" in df.columns:
        rename_map["Player"] = "name"

    # Salary
    if "salary" in df.columns:
        rename_map["salary"] = "salary"
    if "Salary" in df.columns:
        rename_map["Salary"] = "salary"

    # Projection
    if "proj" in df.columns:
        rename_map["proj"] = "proj"
    if "Projection" in df.columns:
        rename_map["Projection"] = "proj"
    if "Proj" in df.columns:
        rename_map["Proj"] = "proj"

    # Ownership
    if "own_proj" in df.columns:
        rename_map["own_proj"] = "own_proj"
    if "Own" in df.columns:
        rename_map["Own"] = "own_proj"
    if "Ownership %" in df.columns:
        rename_map["Ownership %"] = "own_proj"

    # Positions (optional)
    if "Pos" in df.columns:
        rename_map["Pos"] = "positions"
    if "Position" in df.columns:
        rename_map["Position"] = "positions"
    if "Positions" in df.columns:
        rename_map["Positions"] = "positions"

    df = df.rename(columns=rename_map)

    # If no explicit player_id, create one
    if "player_id" not in df.columns:
        # Use an existing ID column if present, else fallback to index
        if "ID" in df.columns:
            df["player_id"] = df["ID"].astype(str)
        elif "id" in df.columns:
            df["player_id"] = df["id"].astype(str)
        else:
            df["player_id"] = df.index.astype(str)

    # ---- 3. Make sure required columns exist ----
    required = ["player_id", "name", "salary", "proj", "own_proj"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(
            "Slate / projections data is missing required columns: "
            f"{missing}\n\nExpected at least: "
            "`player_id`, `name`, `salary`, `proj` / `Projection` / `Proj`, "
            "`own_proj` / `Own` / `Ownership %`."
        )
        return None

    # ---- 4. Clean & convert numeric fields ----

    # Salary: "$12,900" -> 12900 (int)
    df["salary"] = (
        df["salary"]
        .astype(str)
        .str.replace(r"[$,]", "", regex=True)
    )
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")

    # Projection: ensure float
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce")

    # Ownership: "41.9%" -> 0.419
    df["own_proj"] = (
        df["own_proj"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    df["own_proj"] = pd.to_numeric(df["own_proj"], errors="coerce") / 100.0

    # Drop rows that are totally broken
    df = df.dropna(subset=["salary", "proj"])
    # If ownership missing, assume 0
    df["own_proj"] = df["own_proj"].fillna(0.0)

    # ---- 5. Bucket each player by projected ownership ----
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
# CONTEST ANALYZER HELPERS
# ---------------------------------------------------------
@dataclass
class StructureRule:
    contest_type: str
    roster_size: int
    mega_range: Tuple[int, int]
    chalk_range: Tuple[int, int]
    mid_range: Tuple[int, int]
    punt_range: Tuple[int, int]


def get_structure_rule(contest_type: str, roster_size: int = 8) -> StructureRule:
    ct = contest_type.upper()

    if ct == "CASH":
        return StructureRule("CASH", roster_size, (3, 5), (2, 4), (0, 2), (0, 1))
    if ct == "SE_SMALL":
        return StructureRule("SE_SMALL", roster_size, (2, 3), (2, 3), (2, 3), (0, 1))
    if ct == "SE_BIG":
        return StructureRule("SE_BIG", roster_size, (1, 2), (2, 3), (2, 3), (1, 2))
    if ct == "3MAX":
        return StructureRule("3MAX", roster_size, (1, 2), (2, 3), (2, 3), (1, 2))
    if ct == "MME":
        return StructureRule("MME", roster_size, (1, 2), (2, 3), (2, 4), (1, 3))

    return StructureRule("GENERIC_GPP", roster_size, (1, 3), (2, 3), (2, 4), (1, 2))


def classify_user_style(avg_mega, avg_chalk, avg_mid, avg_punt, roster_size=8):
    styles = [
        get_structure_rule("CASH", roster_size),
        get_structure_rule("SE_SMALL", roster_size),
        get_structure_rule("SE_BIG", roster_size),
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


def extract_username(entry_name: str) -> str:
    # "youdacao (5/150)" -> "youdacao"
    if pd.isna(entry_name):
        return ""
    s = str(entry_name)
    if " (" in s:
        return s.split(" (", 1)[0]
    return s


def parse_lineup_string(lineup: str):
    """
    Parse DK Lineup string:
      "PG Luka Doncic SG Shai Gilgeous-Alexander ..." ->
      list of {pos_slot, player_name}
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
            name = " ".join(name_tokens).strip()
            if name:
                players.append({"pos_slot": pos, "player_name": name})
        else:
            i += 1
    return players


def build_long_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Take DK standings export and create a long-form DF with:
      one row per lineup per player.
    """
    df = df_raw.copy()

    rename = {}
    if "EntryId" in df.columns:
        rename["EntryId"] = "entry_id"
    if "Entry ID" in df.columns:
        rename["Entry ID"] = "entry_id"
    if "EntryName" in df.columns:
        rename["EntryName"] = "entry_name"
    if "Entry Name" in df.columns:
        rename["Entry Name"] = "entry_name"
    if "Rank" in df.columns:
        rename["Rank"] = "rank"
    if "Points" in df.columns:
        rename["Points"] = "points"
    if "Total Salary" in df.columns:
        rename["Total Salary"] = "total_salary"
    if "Lineup" in df.columns:
        rename["Lineup"] = "lineup"

    df = df.rename(columns=rename)

    required = ["entry_id", "entry_name", "rank", "points", "lineup"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Contest CSV is missing required columns: {missing}")
        return pd.DataFrame()

    df["username"] = df["entry_name"].apply(extract_username)

    records = []
    for _, row in df.iterrows():
        entry_id = row["entry_id"]
        username = row["username"]
        rank = row["rank"]
        points = row["points"]
        total_salary = row.get("total_salary", None)
        lineup = row["lineup"]

        players = parse_lineup_string(lineup)
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
    total_entries = long_df["entry_id"].nunique()
    own = (
        long_df.groupby("player_name")["entry_id"]
        .nunique()
        .reset_index(name="lineups_with_player")
    )
    own["field_own"] = own["lineups_with_player"] / total_entries
    return long_df.merge(own[["player_name", "field_own"]], on="player_name", how="left")


def build_lineup_features(long_df: pd.DataFrame) -> pd.DataFrame:
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
    detail = long_df[long_df["entry_id"] == entry_id].copy()
    detail = detail.sort_values("pos_slot")
    detail["field_own"] = (detail["field_own"] * 100).map("{:.1f}%".format)
    return detail[["pos_slot", "player_name", "field_own"]]


def build_user_matrix(long_df: pd.DataFrame, username: str) -> pd.DataFrame:
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
# PAGE LAYOUT – TWO MAIN TABS
# ---------------------------------------------------------
st.title("DraftKings Tools – Lineup Builder & Contest Analyzer")

tab_builder, tab_analyzer = st.tabs(["🔥 Lineup Builder", "📊 Contest Analyzer"])

# ---------------------------------------------------------
# TAB 1 – LINEUP BUILDER
# ---------------------------------------------------------
with tab_builder:
    st.subheader("Lineup Builder – Feed Projections, Get Structures")

    st.markdown(
        """
Upload or paste a **player list with projections** for tonight's slate.

Your data can look like your NBA projections CSV with:

- `Player` (we treat as name)  
- `Salary` (e.g. `$12,900`)  
- `Projection` (fantasy points)  
- `Ownership %` (e.g. `41.9%`)

We'll:

- Normalize it into `player_id`, `name`, `salary`, `proj`, `own_proj`
- Bucket players as **mega chalk / chalk / mid / punt**
- Build lineups that match **contest type**, **field size**, and **% to first**
- Support **Classic (8 players)** and **Showdown (1 CPT + 5 FLEX under $50k)**
"""
    )

    col_file, col_paste = st.columns(2)

    with col_file:
        slate_file = st.file_uploader(
            "Upload projections CSV",
            type=["csv"],
            key="slate_upload",
        )

    with col_paste:
        pasted_csv = st.text_area(
            "Or paste CSV data here (header + rows)",
            height=200,
            key="slate_paste",
        )

    slate_df = load_slate_from_any_source(slate_file, pasted_csv)

    if slate_df is None:
        st.info("Upload or paste a projections CSV to start building lineups.")
    else:
        st.markdown("### Player pool (detected)")
        show = slate_df.copy()
        show["own_proj"] = show["own_proj"].map(lambda x: f"{x:.1%}")
        show = show.rename(
            columns={
                "name": "Player",
                "salary": "Salary",
                "proj": "Proj",
                "own_proj": "Own %",
                "bucket": "Bucket",
            }
        )
        st.dataframe(show, use_container_width=True)

        st.markdown("### Contest settings")

        col_a, col_b = st.columns(2)
        with col_a:
            contest_type = st.selectbox(
                "Contest type",
                ["CASH", "SE", "3MAX", "MME"],
                index=1,
            )
        with col_b:
            field_size = st.number_input(
                "Contest size (number of entries)",
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

        slate_mode = st.radio(
            "Slate format",
            ["Classic (8-player DK)", "Showdown (1 CPT + 5 FLEX)"],
            index=0,
        )

        bucket_slack = st.slider(
            "Ownership bucket slack (flexibility)",
            min_value=0,
            max_value=2,
            value=1,
            step=1,
            help="Higher means more wiggle room on how many mega/chalk/mid/punts you get.",
        )

        if st.button("Build banger lineup"):
            if "Showdown" in slate_mode:
                template = build_template_from_params(
                    contest_type=contest_type,
                    field_size=field_size,
                    pct_to_first=pct_to_first,
                    roster_size=6,      # 1 CPT + 5 FLEX
                    salary_cap=50000,
                )
                lineup = build_optimal_lineup_showdown(
                    slate_df,
                    template=template,
                    bucket_slack=bucket_slack,
                )
                st.write(f"Using structure profile: **{template.contest_label}** (Showdown)")
            else:
                template = build_template_from_params(
                    contest_type=contest_type,
                    field_size=field_size,
                    pct_to_first=pct_to_first,
                    roster_size=8,
                    salary_cap=50000,
                )
                lineup = build_optimal_lineup(
                    slate_df,
                    template=template,
                    bucket_slack=bucket_slack,
                )
                st.write(f"Using structure profile: **{template.contest_label}** (Classic)")

            if lineup is None or lineup.empty:
                st.error(
                    "No valid lineup found with these constraints. "
                    "Try increasing bucket slack or adjusting contest parameters."
                )
            else:
                out = lineup[["name", "salary", "proj", "own_proj"]].copy()
                if "role" in lineup.columns:
                    out["role"] = lineup["role"]
                out["own_proj"] = out["own_proj"].map(lambda x: f"{x:.1%}")

                if "role" in out.columns:
                    cols = ["name", "role", "salary", "proj", "own_proj"]
                else:
                    cols = ["name", "salary", "proj", "own_proj"]

                out = out.rename(
                    columns={
                        "name": "Player",
                        "salary": "Salary",
                        "proj": "Proj",
                        "own_proj": "Own %",
                        "role": "Role",
                    }
                )[cols]

                st.subheader("Sample lineup")
                st.dataframe(out, use_container_width=True)

                total_salary = int(lineup["salary"].sum())
                total_proj = float(lineup["proj"].sum())
                st.write(f"**Total salary:** {total_salary} / 50000")
                st.write(f"**Total projection:** {total_proj:.2f} fpts")


# ---------------------------------------------------------
# TAB 2 – CONTEST ANALYZER
# ---------------------------------------------------------
with tab_analyzer:
    st.subheader("Contest Analyzer – Learn from What Won")

    st.markdown(
        """
Upload a **DraftKings tournament results CSV** (standings export) to:

- Compute field ownership for every player  
- Compare winners vs the rest of the field  
- Explore any username’s lineup patterns (chalk vs punts)  
"""
    )

    contest_file = st.file_uploader(
        "Upload a DraftKings contest CSV",
        type=["csv"],
        key="contest_upload",
    )

    if contest_file is None:
        st.info("Upload a DK contest CSV to see contest analysis.")
    else:
        df_raw = pd.read_csv(contest_file)
        long_df = build_long_df(df_raw)
        if long_df.empty:
            st.error("Could not parse any lineups. Check that the CSV has a DK-style 'Lineup' column.")
        else:
            long_df = add_field_ownership(long_df)
            lineup_summary = build_lineup_features(long_df)

            top_pct = st.slider(
                "What % of lineups should count as 'winners'?",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key="analyzer_top_pct",
            )
            total_entries = lineup_summary.shape[0]
            top_cut_rank = max(1, int(total_entries * (top_pct / 100.0)))
            lineup_summary["is_top"] = lineup_summary["rank"] <= top_cut_rank
            long_df["is_top"] = long_df["entry_id"].isin(
                lineup_summary[lineup_summary["is_top"]]["entry_id"]
            )

            st.markdown(f"Top group = ranks **1–{top_cut_rank}** (top **{top_pct}%** of lineups)")

            # Quick overview
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total entries", total_entries)
            col2.metric("Winners (top group)", int(lineup_summary["is_top"].sum()))
            col3.metric("Median points (all)", f"{lineup_summary['points'].median():.2f}")
            col4.metric("Median avg ownership (all)", f"{lineup_summary['avg_own'].median():.2%}")

            st.markdown("### Winners vs Everyone Else (Ownership & Chalk Mix)")
            top_df = lineup_summary[lineup_summary["is_top"]]
            rest_df = lineup_summary[~lineup_summary["is_top"]]

            def summarize(group: pd.DataFrame, label: str) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "Group": [label],
                        "Mean Points": [group["points"].mean()],
                        "Median Points": [group["points"].median()],
                        "Avg Player Ownership": [group["avg_own"].mean()],
                        "Mega Chalk (>=40%) / lineup": [group["n_mega_chalk"].mean()],
                        "Chalk (30–39%) / lineup": [group["n_chalk"].mean()],
                        "Mid (10–29%) / lineup": [group["n_mid"].mean()],
                        "Punts (<10%) / lineup": [group["n_punt"].mean()],
                    }
                )

            top_stats = summarize(top_df, f"Winners (Top {top_pct}%)")
            rest_stats = summarize(rest_df, "Everyone else")
            table = pd.concat([top_stats, rest_stats], ignore_index=True)

            table["Mean Points"] = table["Mean Points"].map(lambda x: f"{x:.2f}")
            table["Median Points"] = table["Median Points"].map(lambda x: f"{x:.2f}")
            table["Avg Player Ownership"] = table["Avg Player Ownership"].map(
                lambda x: f"{x:.1%}"
            )

            st.dataframe(table, use_container_width=True)

            st.markdown("### User Explorer – See How a Username Builds Lineups")
            usernames = sorted(lineup_summary["username"].unique())
            selected_user = st.selectbox("Choose a user", usernames)

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

            stats = user_group.agg(
                avg_points=("points", "mean"),
                avg_avg_own=("avg_own", "mean"),
                avg_mega=("n_mega_chalk", "mean"),
                avg_chalk=("n_chalk", "mean"),
                avg_mid=("n_mid", "mean"),
                avg_punt=("n_punt", "mean"),
            )

            st.markdown(
                f"""
**{selected_user} – average lineup profile**

- Avg points: **{stats['avg_points']:.2f}**  
- Avg player ownership: **{stats['avg_avg_own']:.1%}**

Per lineup (average # players):

- Mega chalk (≥40%): **{stats['avg_mega']:.2f}**  
- Chalk (30–39%): **{stats['avg_chalk']:.2f}**  
- Mid (10–29%): **{stats['avg_mid']:.2f}**  
- Punts (<10%): **{stats['avg_punt']:.2f}**
"""
            )

            style_name, dist = classify_user_style(
                stats["avg_mega"],
                stats["avg_chalk"],
                stats["avg_mid"],
                stats["avg_punt"],
            )
            st.markdown(
                f"""
Likely style: **{style_name}** (distance score {dist:.2f})  

Very rough read:
- CASH → super chalky, almost no punts  
- SE_SMALL → chalky shell, maybe 0–1 punts  
- SE_BIG → more leverage, 1–2 punts  
- 3MAX → between SE and MME  
- MME → more punts & leverage per lineup
"""
            )

            st.markdown("#### User lineups – players in columns")
            user_matrix = build_user_matrix(long_df, selected_user)
            if not user_matrix.empty:
                st.dataframe(user_matrix, use_container_width=True)
            else:
                st.info("No parsed lineups for this user.")

            st.markdown("#### Single lineup detail (ownership by player)")
            selected_entry = st.selectbox("Pick a lineup (Entry ID)", user_group["entry_id"])
            detail = get_lineup_detail(long_df, selected_entry)
            st.table(detail)
