import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# App config
# ---------------------------------------------------------
st.set_page_config(page_title="DK MME Lineup Analyzer", layout="wide")

POS_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]


# ---------------------------------------------------------
# Helpers: parsing + features
# ---------------------------------------------------------
def extract_username(entry_name: str) -> str:
    """DraftKings EntryName looks like: 'youdacao (5/150)' -> we just want 'youdacao'."""
    if pd.isna(entry_name):
        return ""
    s = str(entry_name)
    if " (" in s:
        return s.split(" (", 1)[0]
    return s


def parse_lineup_string(lineup: str):
    """
    Parse DK 'Lineup' string like:
    'C Goga Bitadze F Anthony Davis G Ryan Nembhard ... UTIL Kel'el Ware'
    into a list of {pos_slot, player_name}.
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
    Turn the raw DK standings CSV into one-row-per-lineup-per-player.
    """
    df = df_raw.copy()

    # Normalize column names
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
    Compute field ownership for each player in a single contest.
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
    Build lineup-level features useful for teaching patterns:
    - avg ownership
    - sum ownership
    - max/min ownership
    - number of mega-chalk, chalk, mid, and punt plays
    """
    # Ownership thresholds (tune these if you want)
    mega_chalk_thr = 0.40   # >= 40% owned
    chalk_thr = 0.30        # 30-39%
    punt_thr = 0.10         # < 10% owned

    def count_mega_chalk(s):
        return (s >= mega_chalk_thr).sum()

    def count_chalk(s):
        return ((s >= chalk_thr) & (s < mega_chalk_thr)).sum()

    def count_mid(s):
        return ((s >= punt_thr) & (s < chalk_thr)).sum()

    def count_punt(s):
        return (s < punt_thr).sum()

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
    Return the players (with ownership) for a single lineup.
    """
    detail = long_df[long_df["entry_id"] == entry_id].copy()
    detail = detail.sort_values("pos_slot")
    detail["field_own"] = (detail["field_own"] * 100).map("{:.1f}%".format)
    return detail[["pos_slot", "player_name", "field_own"]]


def build_user_matrix(long_df: pd.DataFrame, username: str) -> pd.DataFrame:
    """
    For a given user, build a matrix:
    - rows: lineups
    - columns: positions (PG/SG/…)
    - values: player names
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
# UI
# ---------------------------------------------------------
st.title("DraftKings MME Lineup Analyzer & Teaching Tool")

st.markdown(
    """
Upload a **DraftKings tournament CSV** (standings export), and this app will:

- Compute field ownership for every player  
- Label the **top X% of lineups as "winners"**  
- Compare **winning lineups vs the field** (ownership, chalk usage, punts)  
- Let you **inspect any user’s full 150 set** and lineups in matrix form  
- Show **player ownership in winners vs the rest**  
- Summarize **what types of lineups are winning in this contest**
"""
)

uploaded_file = st.file_uploader("Upload DK contest CSV", type=["csv"])

if not uploaded_file:
    st.info("Upload a DK Contest CSV to get started.")
    st.stop()

# Read & transform
df_raw = pd.read_csv(uploaded_file)
long_df = build_long_df(df_raw)
long_df = add_field_ownership(long_df)
lineup_summary = build_lineup_features(long_df)

if lineup_summary.empty:
    st.error("No lineups parsed. Check that the CSV has a 'Lineup' column in the DK format.")
    st.stop()

# ---------------- Sidebar controls ----------------
st.sidebar.header("Settings")

total_entries = lineup_summary.shape[0]

top_pct = st.sidebar.slider(
    "Top X% considered 'winning lineups'",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

top_cut_rank = max(1, int(total_entries * (top_pct / 100.0)))

lineup_summary["is_top"] = lineup_summary["rank"] <= top_cut_rank

st.sidebar.write(f"Top cutoff rank: **{top_cut_rank}** out of {total_entries} entries")

# Also flag in long_df
top_entry_ids = set(lineup_summary[lineup_summary["is_top"]]["entry_id"])
long_df["is_top"] = long_df["entry_id"].isin(top_entry_ids)

# ---------------- Overall metrics ----------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total entries", total_entries)
col2.metric("Top group size", lineup_summary["is_top"].sum())
col3.metric("Median points (all)", f"{lineup_summary['points'].median():.2f}")
col4.metric("Median avg ownership (all)", f"{lineup_summary['avg_own'].median():.2%}")

# ---------------- Tabs ----------------
tab_overview, tab_patterns, tab_players, tab_users, tab_teach = st.tabs(
    ["Contest Overview", "Winning vs Field Patterns", "Player Ownership", "User Explorer", "Education Summary"]
)

# -------------------------------------------------
# TAB 1: Contest Overview
# -------------------------------------------------
with tab_overview:
    st.subheader("Lineup summary table")

    st.dataframe(
        lineup_summary[
            ["rank", "entry_id", "username", "points", "total_salary",
             "avg_own", "sum_own", "max_own", "n_mega_chalk", "n_chalk", "n_mid", "n_punt", "is_top"]
        ].reset_index(drop=True)
    )

    st.subheader("Rank vs Average Ownership")
    st.caption("Each point is a lineup. You can visually see if winning lineups are more or less owned than the field.")

    st.scatter_chart(
        lineup_summary,
        x="rank",
        y="avg_own",
    )

# -------------------------------------------------
# TAB 2: Winning vs Field Patterns
# -------------------------------------------------
with tab_patterns:
    st.subheader("Winning vs Field – Summary stats")

    top_df = lineup_summary[lineup_summary["is_top"]]
    rest_df = lineup_summary[~lineup_summary["is_top"]]

    if top_df.empty or rest_df.empty:
        st.warning("Top or rest group is empty. Try adjusting the 'Top X%' slider.")
    else:
        cols_for_stats = [
            "points", "avg_own", "sum_own", "max_own",
            "n_mega_chalk", "n_chalk", "n_mid", "n_punt"
        ]

        def summarize(group, label):
            return pd.DataFrame({
                "group": label,
                "mean_points": [group["points"].mean()],
                "median_points": [group["points"].median()],
                "mean_avg_own": [group["avg_own"].mean()],
                "mean_sum_own": [group["sum_own"].mean()],
                "mean_max_own": [group["max_own"].mean()],
                "avg_mega_chalk": [group["n_mega_chalk"].mean()],
                "avg_chalk": [group["n_chalk"].mean()],
                "avg_mid": [group["n_mid"].mean()],
                "avg_punt": [group["n_punt"].mean()],
            })

        top_stats = summarize(top_df, "Top")
        rest_stats = summarize(rest_df, "Rest of field")
        pattern_table = pd.concat([top_stats, rest_stats], ignore_index=True)

        # Format some columns nicely
        pattern_table["mean_avg_own"] = pattern_table["mean_avg_own"].map(lambda x: f"{x:.2%}")
        pattern_table["mean_sum_own"] = pattern_table["mean_sum_own"].map(lambda x: f"{x:.2f}")
        pattern_table["mean_max_own"] = pattern_table["mean_max_own"].map(lambda x: f"{x:.2%}")
        pattern_table["mean_points"] = pattern_table["mean_points"].map(lambda x: f"{x:.2f}")
        pattern_table["median_points"] = pattern_table["median_points"].map(lambda x: f"{x:.2f}")

        st.dataframe(pattern_table)

        st.markdown(
            """
**How to read this:**

- **avg_mega_chalk / avg_chalk / avg_mid / avg_punt** → average number of players of each ownership tier *per lineup*  
- **mean_avg_own** → how owned the average player is in that lineup  
- **mean_sum_own** → total ownership sum across the lineup (higher = chalkier overall)
"""
        )

# -------------------------------------------------
# TAB 3: Player Ownership
# -------------------------------------------------
with tab_players:
    st.subheader("Player ownership – Top vs Field")

    # total lineups in each group
    n_top = top_df["entry_id"].nunique()
    n_rest = rest_df["entry_id"].nunique()

    # counts for each player in top vs rest
    g = (
        long_df
        .groupby(["player_name", "is_top"])["entry_id"]
        .nunique()
        .unstack(fill_value=0)
        .rename(columns={False: "lineups_rest", True: "lineups_top"})
        .reset_index()
    )

    if n_top > 0:
        g["top_own"] = g["lineups_top"] / n_top
    else:
        g["top_own"] = 0.0

    if n_rest > 0:
        g["rest_own"] = g["lineups_rest"] / n_rest
    else:
        g["rest_own"] = 0.0

    # overall field ownership (from earlier, same for all)
    field_own_ref = (
        long_df.groupby("player_name")["field_own"].first().reset_index()
    )

    player_stats = g.merge(field_own_ref, on="player_name", how="left")

    # Show players sorted by difference in top vs rest ownership
    player_stats["top_minus_rest"] = player_stats["top_own"] - player_stats["rest_own"]

    st.markdown("**Players most over-owned in winners vs rest (positive = appeared more in top lineups)**")
    st.dataframe(
        player_stats.sort_values("top_minus_rest", ascending=False)[
            ["player_name", "field_own", "top_own", "rest_own", "top_minus_rest",
             "lineups_top", "lineups_rest"]
        ].head(40)
    )

    st.markdown("All ownership values are fractions (0.25 = 25%).")

# -------------------------------------------------
# TAB 4: User Explorer
# -------------------------------------------------
with tab_users:
    st.subheader("Explore specific users and their builds")

    username_list = sorted(lineup_summary["username"].unique())
    selected_user = st.selectbox("Select user", username_list)

    user_group = lineup_summary[lineup_summary["username"] == selected_user].copy()
    user_group = user_group.sort_values("rank")

    st.markdown(f"**{selected_user}** – {len(user_group)} lineups")
    st.dataframe(
        user_group[
            ["rank", "entry_id", "points", "total_salary",
             "avg_own", "sum_own", "max_own",
             "n_mega_chalk", "n_chalk", "n_mid", "n_punt", "is_top"]
        ]
    )

    st.subheader("User lineups – players by position (matrix view)")

    user_matrix = build_user_matrix(long_df, selected_user)
    if not user_matrix.empty:
        st.dataframe(user_matrix)
    else:
        st.info("No player matrix available for this user (no parsed lineups).")

    st.subheader("Single lineup detail (ownership by player)")

    selected_entry = st.selectbox("Select Entry ID", user_group["entry_id"])
    detail = get_lineup_detail(long_df, selected_entry)
    st.table(detail)

# -------------------------------------------------
# TAB 5: Education Summary
# -------------------------------------------------
with tab_teach:
    st.subheader("What type of lineups are winning in THIS contest?")

    if top_df.empty or rest_df.empty:
        st.write("Adjust the top X% slider to create a non-empty top & rest group.")
    else:
        # Pull a few key numbers for narrative
        top_mean_avg_own = top_df["avg_own"].mean()
        rest_mean_avg_own = rest_df["avg_own"].mean()

        top_mean_sum_own = top_df["sum_own"].mean()
        rest_mean_sum_own = rest_df["sum_own"].mean()

        top_avg_mega = top_df["n_mega_chalk"].mean()
        rest_avg_mega = rest_df["n_mega_chalk"].mean()

        top_avg_punt = top_df["n_punt"].mean()
        rest_avg_punt = rest_df["n_punt"].mean()

        st.markdown(
            f"""
### 1. How chalky are winning lineups?

- Average player ownership in **top {top_pct}%** lineups: **{top_mean_avg_own:.1%}**  
- Average player ownership in **rest of field**: **{rest_mean_avg_own:.1%}**

If top lineups are **lower** here, it means they’re using more sneaky / contrarian pieces.  
If they’re **higher**, the field may just be bad and the best players simply jammed the right chalk.
"""
        )

        st.markdown(
            f"""
### 2. How many chalk vs punt plays do winners use?

Per lineup, on average:

- **Top {top_pct}% lineups**  
  - Mega-chalk pieces (≥40% owned): **{top_avg_mega:.2f}**  
  - Low-owned punts (<10% owned): **{top_avg_punt:.2f}**

- **Rest of field**  
  - Mega-chalk pieces: **{rest_avg_mega:.2f}**  
  - Low-owned punts: **{rest_avg_punt:.2f}**

If winners play **similar chalk** but **more smart punts**, it suggests:
> “Eat some chalk, but make sure you have a few low-owned leverage pieces in every lineup.”
"""
        )

        st.markdown(
            """
### 3. How to use this as a new player

1. **Scroll the Lineup summary table (Overview tab)**  
   - Look only at the top 1–5% lineups.  
   - Notice how many chalk pieces they play vs low-owned guys.

2. **Check the Player Ownership tab**  
   - See which players show up **way more** in winning lineups than in the field.  
   - Those are the types of leverage / core plays that separated winners here.

3. **Use the User Explorer tab**  
   - Pick a sharp username near the top.  
   - Look at their **matrix view**:  
     - Which positions are stable cores?  
     - Which spots rotate?  
     - Are they using the same star with different cheap pivots?

4. **Build rules for your own 150**  
   - e.g., “Each lineup should have:  
     - 1–2 mega-chalk pieces,  
     - 3–5 mid-owned guys,  
     - 1–2 low-owned punts.”  
   - Then enforce that in your optimizer/build process.
"""
        )

        st.success("Use this app on multiple slates and see if the same patterns repeat. That’s your personal playbook of what wins in these DraftKings contests.")
