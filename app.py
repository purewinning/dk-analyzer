import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="DK MME Lineup Analyzer (Simple View)", layout="wide")

# DraftKings NBA-style slots – tweak for other sports if needed
POS_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

# Ownership buckets (keep these simple & fixed)
MEGA_CHALK_THR = 0.40  # >= 40%
CHALK_THR = 0.30       # 30–39%
PUNT_THR = 0.10        # < 10%


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def extract_username(entry_name: str) -> str:
    """
    DK EntryName looks like: 'youdacao (5/150)' -> we just want 'youdacao'.
    """
    if pd.isna(entry_name):
        return ""
    s = str(entry_name)
    if " (" in s:
        return s.split(" (", 1)[0]
    return s


def parse_lineup_string(lineup: str):
    """
    Parse DK 'Lineup' string like:
    'C Goga Bitadze F Anthony Davis ... UTIL Stephen Curry'
    into list of {pos_slot, player_name}.
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

    # Normalize column names so we don't have to remember exact DK headings
    rename_map = {}
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
        st.error(f"CSV is missing required columns: {missing}")
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
            records.append({
                "entry_id": entry_id,
                "username": username,
                "rank": rank,
                "points": points,
                "total_salary": total_salary,
                "pos_slot": p["pos_slot"],
                "player_name": p["player_name"]
            })

    long_df = pd.DataFrame.from_records(records)
    long_df = long_df.dropna(subset=["player_name"])
    return long_df


def add_field_ownership(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute field ownership for each player in this contest.
    """
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

    def count_mega_chalk(s):
        return (s >= MEGA_CHALK_THR).sum()

    def count_chalk(s):
        return ((s >= CHALK_THR) & (s < MEGA_CHALK_THR)).sum()

    def count_mid(s):
        return ((s >= PUNT_THR) & (s < CHALK_THR)).sum()

    def count_punt(s):
        return (s < PUNT_THR).sum()

    grouped = (
        long_df
        .groupby(["entry_id", "username"])
        .agg(
            rank=("rank", "first"),
            points=("points", "first"),
            total_salary=("total_salary", "first"),
            avg_own=("field_own", "mean"),
            sum_own=("field_own", "sum"),
            max_own=("field_own", "max"),
            min_own=("field_own", "min"),
            n_players=("player_name", "nunique"),
            n_mega_chalk=("field_own", count_mega_chalk),
            n_chalk=("field_own", count_chalk),
            n_mid=("field_own", count_mid),
            n_punt=("field_own", count_punt),
        )
        .reset_index()
        .sort_values("rank")
    )

    return grouped


def get_lineup_detail(long_df: pd.DataFrame, entry_id):
    """
    Show a single lineup's players, positions, and field ownership.
    """
    detail = long_df[long_df["entry_id"] == entry_id].copy()
    detail = detail.sort_values("pos_slot")
    detail["field_own"] = (detail["field_own"] * 100).map("{:.1f}%".format)
    return detail[["pos_slot", "player_name", "field_own"]]


def build_user_matrix(long_df: pd.DataFrame, username: str) -> pd.DataFrame:
    """
    For a given user:
    - each row = one lineup
    - columns = positions (PG, SG, etc.)
    - cells = player names
    plus rank & points.
    """
    user_long = long_df[long_df["username"] == username].copy()
    if user_long.empty:
        return pd.DataFrame()

    user_matrix = (
        user_long
        .pivot_table(
            index=["entry_id", "rank", "points"],
            columns="pos_slot",
            values="player_name",
            aggfunc="first"
        )
        .reset_index()
        .sort_values("rank")
    )
    user_matrix.columns.name = None
    return user_matrix


# ---------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------
st.title("DraftKings Lineup Teacher")

st.markdown(
    """
**Goal:** Help you see *what kinds of lineups actually win* in this contest.

**Simple steps:**
1. Upload a DraftKings results CSV (standings export).  
2. Pick what counts as **“top lineups / winners”**.  
3. Look at how those winners are built vs everyone else.  
4. Spy on sharp users and copy the patterns (structure, not players).
"""
)

uploaded_file = st.file_uploader("Step 1 – Upload a DraftKings contest CSV", type=["csv"])

if not uploaded_file:
    st.info("Upload a DK contest CSV to get started.")
    st.stop()

# ----- Build data -----
df_raw = pd.read_csv(uploaded_file)
long_df = build_long_df(df_raw)
long_df = add_field_ownership(long_df)
lineup_summary = build_lineup_features(long_df)

if lineup_summary.empty:
    st.error("No lineups parsed. Double-check that the CSV has a DK-style 'Lineup' column.")
    st.stop()

# ----- Sidebar: simple settings -----
st.sidebar.header("Settings")

total_entries = lineup_summary.shape[0]
top_pct = st.sidebar.slider(
    "What % of lineups should count as 'winners'?",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
    help="Example: 5% in a 10,000 entry contest = top 500 lineups."
)

top_cut_rank = max(1, int(total_entries * (top_pct / 100.0)))
lineup_summary["is_top"] = lineup_summary["rank"] <= top_cut_rank

top_entry_ids = set(lineup_summary[lineup_summary["is_top"]]["entry_id"])
long_df["is_top"] = long_df["entry_id"].isin(top_entry_ids)

st.sidebar.markdown(f"**Top cutoff rank:** {top_cut_rank} / {total_entries}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Ownership tiers (fixed):**")
st.sidebar.markdown(
    f"""
- Mega chalk: **≥ {int(MEGA_CHALK_THR*100)}%**  
- Chalk: **{int(CHALK_THR*100)}–{int(MEGA_CHALK_THR*100)-1}%**  
- Mid-owned: **{int(PUNT_THR*100)}–{int(CHALK_THR*100)-1}%**  
- Punt / low-owned: **< {int(PUNT_THR*100)}%**
"""
)

# Precompute top vs rest for use in multiple tabs
top_df = lineup_summary[lineup_summary["is_top"]]
rest_df = lineup_summary[~lineup_summary["is_top"]]

# ----- Top-level metrics -----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total entries", total_entries)
col2.metric("Winners (top group)", top_df.shape[0])
col3.metric("Median points (all)", f"{lineup_summary['points'].median():.2f}")
col4.metric("Median avg ownership (all)", f"{lineup_summary['avg_own'].median():.2%}")

# ----- Tabs -----
tab_overview, tab_patterns, tab_players, tab_users, tab_teach = st.tabs(
    ["1️⃣ Contest Overview", "2️⃣ Winners vs Everyone", "3️⃣ Players", "4️⃣ User Explorer", "5️⃣ Plain-English Summary"]
)

# -------------------------------------------------
# TAB 1 – Contest Overview
# -------------------------------------------------
with tab_overview:
    st.subheader("Big picture of this contest")

    st.markdown(
        """
**How to use this:**

- Focus on the left side (rank / points / username).  
- Look at which lineups count as **winners** (True in the “Winner?” column).  
- Ignore most of the numbers at first – they’re there if you want to dig deeper.
"""
    )

    simple_cols = ["rank", "entry_id", "username", "points", "avg_own", "is_top"]
    renamed = lineup_summary[simple_cols].rename(columns={
        "rank": "Rank",
        "entry_id": "Entry ID",
        "username": "User",
        "points": "Points",
        "avg_own": "Avg player ownership",
        "is_top": f"Winner? (Top {top_pct}%)"
    })

    # Format avg ownership nicely
    renamed["Avg player ownership"] = renamed["Avg player ownership"].map(lambda x: f"{x:.1%}")

    st.dataframe(renamed, use_container_width=True)

    with st.expander("Show advanced columns (for nerds)"):
        st.dataframe(lineup_summary, use_container_width=True)

    st.subheader("Visual: Rank vs Avg Ownership")
    st.caption("Each dot is a lineup. Left = higher finish. Up/down = more/less chalky on average.")
    st.scatter_chart(
        lineup_summary,
        x="rank",
        y="avg_own",
    )

# -------------------------------------------------
# TAB 2 – Winners vs Everyone
# -------------------------------------------------
with tab_patterns:
    st.subheader("How are winners built differently?")

    if top_df.empty or rest_df.empty:
        st.warning("Top or rest group is empty. Try adjusting the top % slider.")
    else:
        def summarize(group, label):
            return pd.DataFrame({
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
            })

        top_stats = summarize(top_df, f"Winners (Top {top_pct}%)")
        rest_stats = summarize(rest_df, "Everyone else")
        pattern_table = pd.concat([top_stats, rest_stats], ignore_index=True)

        # Format percentages nicely
        pattern_table["Avg Player Ownership"] = pattern_table["Avg Player Ownership"].map(lambda x: f"{x:.1%}")
        pattern_table["Max Single-Player Ownership"] = pattern_table["Max Single-Player Ownership"].map(lambda x: f"{x:.1%}")
        pattern_table["Mean Points"] = pattern_table["Mean Points"].map(lambda x: f"{x:.2f}")
        pattern_table["Median Points"] = pattern_table["Median Points"].map(lambda x: f"{x:.2f}")
        pattern_table["Total Ownership Sum"] = pattern_table["Total Ownership Sum"].map(lambda x: f"{x:.2f}")

        st.dataframe(pattern_table, use_container_width=True)

        st.markdown(
            f"""
**Read this as:**

- First row = **average lineup** from the winners (top {top_pct}%).  
- Second row = **average lineup** from everyone else.  

Focus on:
- **Mega Chalk per lineup** → Do winners play more or fewer mega-chalk pieces?  
- **Punts per lineup** → Do winners use more low-owned darts?
"""
        )

# -------------------------------------------------
# TAB 3 – Players
# -------------------------------------------------
with tab_players:
    st.subheader("Which players showed up in winners vs everyone else?")

    if top_df.empty or rest_df.empty:
        st.warning("Top or rest group is empty. Try adjusting the top % slider.")
    else:
        n_top = top_df["entry_id"].nunique()
        n_rest = rest_df["entry_id"].nunique()

        g = (
            long_df
            .groupby(["player_name", "is_top"])["entry_id"]
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

        # Format percentages for display
        display_df = player_stats.copy()
        for col in ["field_own", "top_own", "rest_own", "top_minus_rest"]:
            display_df[col] = display_df[col].map(lambda x: f"{x:.1%}")

        st.markdown("**Players the winners used more than the rest of the field:**")
        st.caption("Sorted by how much more they appeared in winning lineups vs everyone else.")
        st.dataframe(
            display_df.sort_values("top_minus_rest", ascending=False)[
                ["player_name", "field_own", "top_own", "rest_own", "top_minus_rest",
                 "lineups_top", "lineups_rest"]
            ].head(40),
            use_container_width=True
        )

# -------------------------------------------------
# TAB 4 – User Explorer
# -------------------------------------------------
with tab_users:
    st.subheader("Spy on how a user built their lineups")

    st.markdown(
        """
**How to use this:**

1. Pick a username.  
2. See all their lineups and how chalky they are.  
3. Look at the “matrix view” to see what their 150 actually *looks* like.  
4. Drill into a single lineup to see ownership by player.
"""
    )

    username_list = sorted(lineup_summary["username"].unique())
    selected_user = st.selectbox("Choose a user", username_list)

    user_group = lineup_summary[lineup_summary["username"] == selected_user].copy()
    user_group = user_group.sort_values("rank")

    simple_user = user_group[["rank", "entry_id", "points", "avg_own", "is_top"]].rename(columns={
        "rank": "Rank",
        "entry_id": "Entry ID",
        "points": "Points",
        "avg_own": "Avg player ownership",
        "is_top": f"Winner? (Top {top_pct}%)"
    })
    simple_user["Avg player ownership"] = simple_user["Avg player ownership"].map(lambda x: f"{x:.1%}")

    st.markdown(f"**{selected_user}** – {len(user_group)} lineups")
    st.dataframe(simple_user, use_container_width=True)

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
# TAB 5 – Plain-English Summary
# -------------------------------------------------
with tab_teach:
    st.subheader("What this contest is teaching you")

    if top_df.empty or rest_df.empty:
        st.write("Adjust the 'top %' slider so there are some winners and some non-winners.")
    else:
        top_mean_avg_own = top_df["avg_own"].mean()
        rest_mean_avg_own = rest_df["avg_own"].mean()

        top_avg_mega = top_df["n_mega_chalk"].mean()
        rest_avg_mega = rest_df["n_mega_chalk"].mean()

        top_avg_punt = top_df["n_punt"].mean()
        rest_avg_punt = rest_df["n_punt"].mean()

        st.markdown(
            f"""
### 1. How chalky are winning lineups here?

- Average player ownership in **winners** (top {top_pct}%): **{top_mean_avg_own:.1%}**  
- Average player ownership in **everyone else**: **{rest_mean_avg_own:.1%}**

If winners are **less owned** on average → they leaned more into being different.  
If winners are **more owned** → the slate may have been very chalk-heavy, and you needed the right chalk combo.
"""
        )

        st.markdown(
            f"""
### 2. How many chalk vs punts do winners use?

Per lineup, on average:

**Winners (top {top_pct}%):**
- Mega chalk pieces (≥40% owned): **{top_avg_mega:.2f}**  
- Punts (<10% owned): **{top_avg_punt:.2f}**

**Everyone else:**
- Mega chalk pieces: **{rest_avg_mega:.2f}**  
- Punts: **{rest_avg_punt:.2f}**

Rough rule of thumb from this contest:

> Play around **{top_avg_mega:.1f} mega-chalk pieces** per lineup  
> and look to get about **{top_avg_punt:.1f} low-owned punts** that can separate you.
"""
        )

        st.markdown(
            """
### 3. How to turn this into a simple blueprint

For future slates in this same contest type:

1. Use projections or your favorite content to find **the good chalk**.  
2. Aim for:
   - A similar number of mega-chalk pieces as winning lineups here.  
   - A similar number of punts as winning lineups here.  

3. Use the **User Explorer** tab to copy the *structure* of sharp players:
   - How many studs they play  
   - How much salary they use  
   - How many spots they rotate vs lock

Run this on a few slates and you’ll start to see the same patterns repeat.  
That’s your personal cheat sheet for how to build lineups that can win in this DraftKings contest.
"""
        )
