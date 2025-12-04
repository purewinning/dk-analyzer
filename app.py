import streamlit as st
import pandas as pd

st.set_page_config(page_title="DK MME Lineup Analyzer", layout="wide")

POS_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

def extract_username(entry_name: str) -> str:
    if pd.isna(entry_name):
        return ""
    s = str(entry_name)
    if " (" in s:
        return s.split(" (", 1)[0]
    return s

def parse_lineup_string(lineup: str):
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
    df = df_raw.copy()

    rename_map = {
        "EntryId": "entry_id",
        "EntryName": "entry_name",
        "Rank": "rank",
        "Points": "points",
        "Lineup": "lineup"
    }

    df = df.rename(columns=rename_map)
    df["username"] = df["entry_name"].apply(extract_username)

    records = []
    for _, row in df.iterrows():
        entry_id = row["entry_id"]
        username = row["username"]
        rank = row["rank"]
        points = row["points"]
        lineup_str = row["lineup"]

        players = parse_lineup_string(lineup_str)
        for p in players:
            records.append({
                "entry_id": entry_id,
                "username": username,
                "rank": rank,
                "points": points,
                "pos_slot": p["pos_slot"],
                "player_name": p["player_name"]
            })

    long_df = pd.DataFrame.from_records(records)
    return long_df.dropna(subset=["player_name"])

def add_field_ownership(long_df: pd.DataFrame) -> pd.DataFrame:
    total_entries = long_df["entry_id"].nunique()
    own = (
        long_df.groupby("player_name")["entry_id"]
        .nunique()
        .reset_index(name="lineups_with_player")
    )
    own["field_own"] = own["lineups_with_player"] / total_entries
    return long_df.merge(own[["player_name", "field_own"]], on="player_name", how="left")

def build_lineup_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        long_df.groupby(["entry_id", "username"])
        .agg(
            rank=("rank", "first"),
            points=("points", "first"),
            avg_own=("field_own", "mean"),
            sum_own=("field_own", "sum"),
            max_own=("field_own", "max")
        )
        .reset_index()
        .sort_values("rank")
    )
    return grouped

def get_lineup_detail(long_df: pd.DataFrame, entry_id):
    detail = long_df[long_df["entry_id"] == entry_id].sort_values("pos_slot")
    detail["field_own"] = (detail["field_own"] * 100).map("{:.1f}%".format)
    return detail[["pos_slot", "player_name", "field_own"]]

st.title("DraftKings MME Lineup Analyzer")

uploaded_file = st.file_uploader("Upload DK contest CSV", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    long_df = build_long_df(df_raw)
    long_df = add_field_ownership(long_df)
    lineup_summary = build_lineup_summary(long_df)

    st.subheader("Lineup Summary")
    st.dataframe(lineup_summary)

    username_list = sorted(lineup_summary["username"].unique())
    selected_user = st.selectbox("Select User", username_list)

    user_group = lineup_summary[lineup_summary["username"] == selected_user]
    st.subheader("User Lineups")
    st.dataframe(user_group)

    selected_entry = st.selectbox("Select Entry ID", user_group["entry_id"])
    detail = get_lineup_detail(long_df, selected_entry)

    st.subheader("Lineup Detail")
    st.table(detail)

else:
    st.info("Upload a DK Contest CSV to begin.")
