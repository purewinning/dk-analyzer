# app.py

import streamlit as st
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

from builder import (
    build_template_from_params,
    build_optimal_lineup,
    build_optimal_lineup_showdown,
)

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="DK Analyzer & Lineup Builder", layout="wide")

POS_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

# Ownership thresholds (must match builder.py)
MEGA_CHALK_THR = 0.40
CHALK_THR = 0.30
PUNT_THR = 0.10


# ---------------------------------------------------------
# USER STYLE CLASSIFICATION (for User Explorer)
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


# ---------------------------------------------------------
# CONTEST CSV HELPERS
# ---------------------------------------------------------
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
        st.stop()

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
# SLATE / PROJECTIONS CSV (for lineup builder)
# ---------------------------------------------------------
def load_slate_players_from_upload(uploaded_file) -> pd.DataFrame:
    """
    Expect columns (names flexible, we normalize):
      - player_id
      - name / Name
      - salary / Salary
      - proj / Projection / Proj
      - own_proj / Own  (fraction: 0.25 = 25%)
    """
    df = pd.read_csv(uploaded_file)

    rename_map: Dict[str, str] = {}
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

    df = df.rename(columns=rename_map)

    required = ["player_id", "name", "salary", "proj", "own_proj"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Slate / projections CSV is missing required columns: {missing}")
        st.stop()

    # Bucket by projected ownership
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
st.title("DraftKings Contest Analyzer + Structure-Based Lineup Builder")

st.markdown(
    """
**Step 1:** Upload a **DraftKings results CSV** (standings export).  
**Step 2:** (Optional) Upload a **slate + projections CSV** to build sample lineups.

What you get:

- Contest overview + winners vs everyone else  
- Per-player field ownership and “winner vs field” usage  
- User explorer: see how a username structures 150 MME lineups  
- Lineup builder (Classic + Showdown) that applies those structure ideas
"""
)

uploaded = st.file_uploader("Upload a DraftKings contest CSV", type=["csv"], key="contest_csv")

if uploaded is None:
    st.stop()

df_raw = pd.read_csv(uploaded)
long_df = build_long_df(df_raw)
long_df = add_field_ownership(long_df)
lineup_summary = build_lineup_features(long_df)

tab_overview, tab_builder, tab_patterns, tab_players, tab_users, tab_teach = st.tabs(
    [
        "1️⃣ Contest Overview",
        "2️⃣ Slate & Lineup Builder",
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

    total_entries = lineup_summary.shape[0]
    top_pct = st.slider(
        "What % of lineups should count as 'winners'?",
        min_value=1, max_value=20, value=5, step=1,
    )
    top_cut_rank = max(1, int(total_entries * (top_pct / 100.0)))

    lineup_summary["is_top"] = lineup_summary["rank"] <= top_cut_rank
    top_entry_ids = set(lineup_summary[lineup_summary["is_top"]]["entry_id"])
    long_df["is_top"] = long_df["entry_id"].isin(top_entry_ids)

    st.markdown(f"Top group = ranks **1–{top_cut_rank}** (top **{top_pct}%** of lineups)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total entries", total_entries)
    col2.metric("Winners (top group)", int(lineup_summary["is_top"].sum()))
    col3.metric("Median points (all)", f"{lineup_summary['points'].median():.2f}")
    col4.metric("Median avg ownership (all)", f"{lineup_summary['avg_own'].median():.2%}")

    simple = lineup_summary[
        ["rank", "entry_id", "username", "points", "avg_own", "is_top"]
    ].rename(
        columns={
            "rank": "Rank",
            "entry_id": "Entry ID",
            "username": "User",
            "points": "Points",
            "avg_own": "Avg player ownership",
            "is_top": f"Winner? (Top {top_pct}%)",
        }
    )
    simple["Avg player ownership"] = simple["Avg player ownership"].map(
        lambda x: f"{x:.1%}"
    )
    st.dataframe(simple, use_container_width=True)

    st.subheader("Rank vs Avg Ownership")
    st.caption("Each dot is a lineup. Left = better finish. Higher = more chalky on average.")
    st.scatter_chart(lineup_summary, x="rank", y="avg_own")

    # Save for other tabs
    st.session_state["top_pct"] = top_pct
    st.session_state["lineup_summary"] = lineup_summary
    st.session_state["long_df"] = long_df


# -------------------------------------------------
# TAB 2 – Slate & Lineup Builder
# -------------------------------------------------
with tab_builder:
    st.subheader("Slate & Structure-Based Lineup Builder")

    st.markdown(
        """
Upload a **separate CSV** with the player pool + projections for tonight's slate.

Expected columns (names flexible, we normalize):

- `player_id`
- `Name` or `name`
- `Salary` or `salary`
- `Projection` / `Proj` / `proj`
- `Own` or `own_proj` (fraction: 0.25 = 25%)

Then:

- Pick contest type (CASH / SE / 3MAX / MME)  
- Enter contest size and % to first  
- Choose **Classic** (8 players) or **Showdown** (1 CPT + 5 FLEX)  
- Hit “Build sample lineup”
"""
    )

    slate_file = st.file_uploader(
        "Upload slate + projections CSV (player pool)",
        type=["csv"],
        key="slate_upload",
    )

    if slate_file is None:
        st.info("Upload a slate / projections CSV to use the builder.")
    else:
        slate_df = load_slate_players_from_upload(slate_file)

        slate_mode = st.radio(
            "Slate format",
            ["Classic (8-player DK)", "Showdown (1 CPT + 5 FLEX)"],
            index=0,
        )

        st.markdown("### Player pool")
        view = slate_df.copy()
        view["own_proj"] = view["own_proj"].map(lambda x: f"{x:.1%}")
        view = view.rename(
            columns={
                "name": "Player",
                "salary": "Salary",
                "proj": "Proj",
                "own_proj": "Own %",
                "bucket": "Bucket",
            }
        )
        st.dataframe(view, use_container_width=True)

        st.markdown("### Contest parameters")

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
            "Ownership bucket slack (flexibility)",
            min_value=0,
            max_value=2,
            value=1,
            step=1,
            help="Higher = more flexible counts of mega/chalk/mid/punts.",
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
                show = lineup[["name", "salary", "proj", "own_proj"]].copy()
                if "role" in lineup.columns:
                    show["role"] = lineup["role"]
                show["own_proj"] = show["own_proj"].map(lambda x: f"{x:.1%}")

                if "role" in show.columns:
                    cols = ["name", "role", "salary", "proj", "own_proj"]
                else:
                    cols = ["name", "salary", "proj", "own_proj"]

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


# -------------------------------------------------
# TAB 3 – Winners vs Everyone
# -------------------------------------------------
with tab_patterns:
    st.subheader("Winners vs Everyone Else")

    top_pct = st.session_state.get("top_pct", 5)
    top_df = lineup_summary[lineup_summary["is_top"]].copy()
    rest_df = lineup_summary[~lineup_summary["is_top"]].copy()

    def summarize(group: pd.DataFrame, label: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Group": [label],
                "Mean Points": [group["points"].mean()],
                "Median Points": [group["points"].median()],
                "Avg Player Ownership": [group["avg_own"].mean()],
                "Total Ownership Sum": [group["sum_own"].mean()],
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
    table["Total Ownership Sum"] = table["Total Ownership Sum"].map(
        lambda x: f"{x:.2f}"
    )

    st.dataframe(table, use_container_width=True)


# -------------------------------------------------
# TAB 4 – Players
# -------------------------------------------------
with tab_players:
    st.subheader("Players – Winners vs Everyone Else")

    top_df = lineup_summary[lineup_summary["is_top"]].copy()
    rest_df = lineup_summary[~lineup_summary["is_top"]].copy()

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
    player_stats["top_minus_rest"] = player_stats["top_own"] - player_stats["rest_own"]

    display = player_stats.copy()
    for col in ["field_own", "top_own", "rest_own", "top_minus_rest"]:
        display[col] = display[col].map(lambda x: f"{x:.1%}")

    st.dataframe(
        display.sort_values("top_minus_rest", ascending=False)[
            ["player_name", "field_own", "top_own", "rest_own",
             "top_minus_rest", "lineups_top", "lineups_rest"]
        ].head(40),
        use_container_width=True,
    )


# -------------------------------------------------
# TAB 5 – User Explorer
# -------------------------------------------------
with tab_users:
    st.subheader("User Explorer – See How a User Built Their Lineups")

    usernames = sorted(lineup_summary["username"].unique())
    selected_user = st.selectbox("Choose a user", usernames)

    user_group = (
        lineup_summary[lineup_summary["username"] == selected_user]
        .copy()
        .sort_values("rank")
    )

    top_pct = st.session_state.get("top_pct", 5)

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
On average, **{selected_user}** built lineups like this:

- Avg points per lineup: **{stats['avg_points']:.2f}**  
- Avg player ownership: **{stats['avg_avg_own']:.1%}**

Per lineup (average players per lineup):

- Mega chalk (≥40%): **{stats['avg_mega']:.2f}**  
- Chalk (30–39%): **{stats['avg_chalk']:.2f}**  
- Mid (10–29%): **{stats['avg_mid']:.2f}**  
- Punts (<10%): **{stats['avg_punt']:.2f}**  
"""
    )

    style_name, dist = classify_user_style(
        stats["avg_mega"], stats["avg_chalk"], stats["avg_mid"], stats["avg_punt"]
    )
    st.subheader("Lineup style profile")
    st.markdown(
        f"""
This user's builds most closely resemble:

> **{style_name}**-style lineups (distance score {dist:.2f})

Very rough idea:
- CASH → super chalky, almost no punts  
- SE_SMALL → chalky shell, maybe 0–1 punts  
- SE_BIG → more leverage, 1–2 punts  
- 3MAX → between SE and MME  
- MME → more punts & leverage per lineup  
"""
    )

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

    top_pct = st.session_state.get("top_pct", 5)
    top_df = lineup_summary[lineup_summary["is_top"]]
    rest_df = lineup_summary[~lineup_summary["is_top"]]

    top_mean_avg_own = top_df["avg_own"].mean()
    rest_mean_avg_own = rest_df["avg_own"].mean()

    top_avg_mega = top_df["n_mega_chalk"].mean()
    rest_avg_mega = rest_df["n_mega_chalk"].mean()

    top_avg_punt = top_df["n_punt"].mean()
    rest_avg_punt = rest_df["n_punt"].mean()

    st.markdown(
        f"""
### 1. How chalky are winning lineups?

- Avg player ownership (winners, top {top_pct}%): **{top_mean_avg_own:.1%}**  
- Avg player ownership (everyone else): **{rest_mean_avg_own:.1%}**
"""
    )

    st.markdown(
        f"""
### 2. How many chalk vs punts do winners use?

**Winners (top {top_pct}%):**
- Mega chalk (≥40%): **{top_avg_mega:.2f}** per lineup  
- Punts (<10%): **{top_avg_punt:.2f}** per lineup  

**Everyone else:**
- Mega chalk: **{rest_avg_mega:.2f}**  
- Punts: **{rest_avg_punt:.2f}**
"""
    )

    st.markdown(
        """
### 3. How to use this with the builder

For similar contests:

- Decide how many mega-chalk and punt plays you want per lineup  
- Choose contest type + size + % to first in the **Slate & Lineup Builder** tab  
- Use the builder to generate sample lineups that **match that structure**, then swap in the actual players/sim angles you like
"""
    )
